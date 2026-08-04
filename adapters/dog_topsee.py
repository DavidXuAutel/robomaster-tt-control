"""拓普视平台侧的 NavBackend / PerceptionBackend / GasBackend 实现。

设计依据：`docs/design/2026-08-03-dog-integration-plan.md` §3、§5.1、§5.4

三条核心纪律：
1. **到点判定三态**。平台状态字符串全部无枚举（G2），对不上白名单一律记
   `UNKNOWN`，超容忍次数就通过 `poll_fault()` 如实上报，绝不静默返回 False
   把「接口语义不匹配」拖成「狗走得慢」。
2. **绝不在 tick 里发 HTTP**。周期状态走 PollCache 后台线程，`is_arrived()`
   只读缓存；`cancel()` 走短超时同步调用（方案 §5.3）。
3. **不伪造数据**。气体标定平台侧没有数据源（F13），缺台账就如实失败；
   感知结果里不许出现位姿/点云等 v1 禁止字段。
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from adapters.topsee_client import (
    PollCache,
    TopseeApiError,
    TopseeClient,
    TopseeError,
    TopseeUnreachable,
)
from mission_brain.events import FORBIDDEN_KEYS

logger = logging.getLogger(__name__)


class NavStatus(str, Enum):
    """到点三态。UNKNOWN 表示「平台说了话但我们不认识」，与 EN_ROUTE 必须区分。"""

    ARRIVED = "arrived"
    EN_ROUTE = "en_route"
    UNKNOWN = "unknown"


# poll_fault() 的取值（方案 §5.1）
FAULT_STATUS_UNRECOGNIZED = "nav_status_unrecognized"
FAULT_TASK_NOT_TRACKED = "nav_task_not_tracked"
FAULT_NAV_TIMEOUT = "nav_timeout"
FAULT_PLATFORM_UNREACHABLE = "platform_unreachable"
FAULT_GOAL_UNRESOLVED = "goal_not_in_active_map"
FAULT_SEND_REJECTED = "nav_send_rejected"

# F17：路线被挡且未设检修区时平台的告警文案
_NAV_TIMEOUT_HINTS = ("导航超时", "未找到有效路径", "no valid path")


def _dig(doc: Any, *names: str) -> Any:
    """从 mapping 里按多个候选键名取第一个非空值。

    平台响应字段命名在不同接口间不一致（且 G16 怀疑有导出标注错误），
    解析必须对键名做防御。
    """
    if not isinstance(doc, Mapping):
        return None
    for n in names:
        if n in doc and doc[n] not in (None, ""):
            return doc[n]
    return None


class TopseeNav:
    """NavBackend 实现：拓扑派单 + 三态到点判定。

    到点证据（互相独立，任一成立即 ARRIVED）：
      A. `currentState` 命中 `arrived_states` 白名单 —— 取值须由 E2 实测填入
      B. 位姿距目标点位 < `arrive_radius_m` —— 需注入 `pose_source` 与
         `goal_pose_resolver`，在平台状态字符串不可用时仍能工作
      C. `task_cleared_means_arrived=True` 时，任务从「有」变「无」视为到达
         —— 语义有歧义（可能是被取消），默认关闭

    `arrived_states` 与 `enroute_states` 默认都是空的：**这是有意的**。
    E2 实测枚举表落地前，平台字符串一律算 UNKNOWN，方案要求的就是这个行为。
    """

    def __init__(
        self,
        client: TopseeClient,
        *,
        robot_id: str,
        arbiter: Optional[Any] = None,
        goal_resolver: Optional[Callable[[str], Optional[str]]] = None,
        goal_pose_resolver: Optional[Callable[[str], Optional[Tuple[float, float]]]] = None,
        pose_source: Optional[Callable[[], Optional[Tuple[float, float, float]]]] = None,
        arrived_states: Sequence[str] = (),
        enroute_states: Sequence[str] = (),
        arrive_radius_m: float = 0.6,
        poll_interval_s: float = 1.0,
        stale_s: float = 6.0,
        unknown_tolerance: int = 10,
        task_cleared_means_arrived: bool = False,
        autostart_poller: bool = True,
    ) -> None:
        if poll_interval_s < 0.5:
            raise ValueError("poll_interval_s 不得低于 0.5s（平台只能轮询，别打爆网关）")
        self.client = client
        self.robot_id = str(robot_id)
        self.arbiter = arbiter
        self._goal_resolver = goal_resolver
        self._goal_pose_resolver = goal_pose_resolver
        self._pose_source = pose_source
        self.arrived_states = {s for s in arrived_states}
        self.enroute_states = {s for s in enroute_states}
        self.arrive_radius_m = float(arrive_radius_m)
        self.stale_s = float(stale_s)
        self.unknown_tolerance = int(unknown_tolerance)
        self.task_cleared_means_arrived = bool(task_cleared_means_arrived)
        self._autostart = bool(autostart_poller)

        self.cache: PollCache[Any] = PollCache(
            self._fetch_task, interval_s=poll_interval_s, name=f"task-{robot_id}"
        )
        self._goal_label: Optional[str] = None
        self._points_id: Optional[str] = None
        self._task_seen = False
        self._unknown_count = 0
        self._fault: Optional[str] = None
        self._last_status = NavStatus.UNKNOWN
        self.navigate_calls = 0
        self.cancel_calls = 0

    # ---------- 内部 ----------

    def _fetch_task(self) -> Any:
        return self.client.get_current_task(self.robot_id)

    def _resolve(self, dog_goal_id: str) -> Optional[str]:
        if self._goal_resolver is None:
            return dog_goal_id
        return self._goal_resolver(dog_goal_id)

    def _set_fault(self, reason: str) -> None:
        if self._fault is None:
            logger.warning("TopseeNav 故障: %s (goal=%s)", reason, self._goal_label)
        self._fault = reason

    # ---------- NavBackend ----------

    def goto_goal(self, dog_goal_id: str) -> bool:
        """派单去某个点位。被拒绝时返回 False（不抛），并留下 fault 供上报。"""
        self._fault = None
        self._unknown_count = 0
        self._task_seen = False
        self._last_status = NavStatus.UNKNOWN
        self._goal_label = dog_goal_id

        if self.arbiter is not None and not self.arbiter.allow_topsee_cmd():
            self._set_fault(FAULT_SEND_REJECTED)
            logger.warning("Arbiter 未授权慢通道，拒绝 sendNavigate")
            return False

        points_id = self._resolve(dog_goal_id)
        if not points_id:
            self._points_id = None
            self._set_fault(FAULT_GOAL_UNRESOLVED)
            return False
        self._points_id = str(points_id)

        try:
            self.client.send_navigate(self.robot_id, self._points_id)
        except TopseeError as exc:
            self._set_fault(
                FAULT_PLATFORM_UNREACHABLE
                if isinstance(exc, TopseeUnreachable)
                else FAULT_SEND_REJECTED
            )
            logger.warning("sendNavigate 失败: %s", exc)
            return False

        self.navigate_calls += 1
        if self._autostart:
            self.cache.start()
        return True

    def poll_status(self, now: Optional[float] = None) -> NavStatus:
        """算一次三态。`is_arrived()` 的实现体，单独暴露便于测试与诊断。"""
        task, age = self.cache.get(now=now)

        if age > self.stale_s:
            err = self.cache.last_error
            if isinstance(err, TopseeUnreachable):
                self._set_fault(FAULT_PLATFORM_UNREACHABLE)
            self._bump_unknown()
            self._last_status = NavStatus.UNKNOWN
            return self._last_status

        # F17：平台把「未找到有效路径」写进任务/告警文案
        blob = str(task)
        if any(h in blob for h in _NAV_TIMEOUT_HINTS):
            self._set_fault(FAULT_NAV_TIMEOUT)
            self._last_status = NavStatus.UNKNOWN
            return self._last_status

        has_task = isinstance(task, Mapping) and bool(task)
        if has_task:
            self._task_seen = True
        elif self._task_seen and self.task_cleared_means_arrived:
            self._last_status = NavStatus.ARRIVED
            return self._last_status

        # 证据 B：距离判据。独立于平台字符串，E2 未落地时这是唯一可靠证据。
        if self._distance_arrived():
            self._last_status = NavStatus.ARRIVED
            return self._last_status

        if not has_task:
            # E1 的失败特征：派单后平台压根查不到任务
            self._bump_unknown(FAULT_TASK_NOT_TRACKED)
            self._last_status = NavStatus.UNKNOWN
            return self._last_status

        # 证据 A：状态白名单
        state = _dig(task, "currentState", "totalState", "status")
        if state is not None:
            s = str(state)
            if s in self.arrived_states:
                self._last_status = NavStatus.ARRIVED
                return self._last_status
            if s in self.enroute_states:
                self._unknown_count = 0
                self._last_status = NavStatus.EN_ROUTE
                return self._last_status

        self._bump_unknown()
        self._last_status = NavStatus.UNKNOWN
        return self._last_status

    def _bump_unknown(self, reason: str = FAULT_STATUS_UNRECOGNIZED) -> None:
        self._unknown_count += 1
        if self._unknown_count >= self.unknown_tolerance:
            self._set_fault(reason)

    def _distance_arrived(self) -> bool:
        if self._pose_source is None or self._goal_pose_resolver is None:
            return False
        if not self._goal_label:
            return False
        goal = self._goal_pose_resolver(self._goal_label)
        pose = self._pose_source()
        if goal is None or pose is None:
            return False
        dx = float(pose[0]) - float(goal[0])
        dy = float(pose[1]) - float(goal[1])
        return (dx * dx + dy * dy) ** 0.5 <= self.arrive_radius_m

    def is_arrived(self) -> bool:
        return self.poll_status() is NavStatus.ARRIVED

    def poll_fault(self) -> Optional[str]:
        """可选扩展钩子（方案 §5.1）：DogSdkAdapter 用 getattr 探测。

        返回后清零，避免同一个故障被反复上报成多条事件。
        """
        f = self._fault
        self._fault = None
        return f

    def cancel(self) -> None:
        """停止平台任务。短超时 + 吞异常：abort 路径不许被网络拖住。"""
        self.cancel_calls += 1
        try:
            self.client.stop_task(self.robot_id)
        except TopseeError as exc:
            logger.warning("stopTask 失败（本地仍按已取消处理）: %s", exc)
        self._goal_label = None
        self._points_id = None
        self._task_seen = False
        self._last_status = NavStatus.UNKNOWN

    def close(self) -> None:
        self.cache.stop()

    # ---------- 诊断 ----------

    @property
    def last_status(self) -> NavStatus:
        return self._last_status

    @property
    def unknown_count(self) -> int:
        return self._unknown_count

    @property
    def points_id(self) -> Optional[str]:
        return self._points_id


class TopseePerception:
    """PerceptionBackend 实现。

    mode:
      - `local_vision`（默认，推荐）：用注入的 detector 在本地视频流上找目标。
        平台的 `distinguishType`（表计/红外/声纹/边界）是电力巡检业务算法，
        与我们的 target_label 语义无关，方案已明确规避。
      - `alarm_uri`：轮询告警列表，拿平台图片 URL 当 evidence。只作旁路证据，
        因为手动巡检段自动巡检任务已停跑，平台算法未必还在产出结果。
    """

    def __init__(
        self,
        client: TopseeClient,
        *,
        robot_id: str,
        mode: str = "local_vision",
        detector: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        alarm_poll_interval_s: float = 2.0,
        alarm_stale_s: float = 15.0,
        min_confidence: float = 0.5,
        autostart_poller: bool = True,
    ) -> None:
        if mode not in ("local_vision", "alarm_uri"):
            raise ValueError("TopseePerception.mode 必须是 local_vision 或 alarm_uri")
        if mode == "local_vision" and detector is None:
            raise ValueError("mode=local_vision 必须注入 detector（禁止隐式回退到平台算法）")
        self.client = client
        self.robot_id = str(robot_id)
        self.mode = mode
        self._detector = detector
        self.alarm_stale_s = float(alarm_stale_s)
        self.min_confidence = float(min_confidence)
        self.calls = 0
        self.cache: Optional[PollCache[Any]] = None
        if mode == "alarm_uri":
            self.cache = PollCache(
                self._fetch_alarms,
                interval_s=alarm_poll_interval_s,
                name=f"alarm-{robot_id}",
            )
            if autostart_poller:
                self.cache.start()

    def _fetch_alarms(self) -> Any:
        return self.client.get_alarm_list(robotId=self.robot_id)

    def search_target(self, target_label: str) -> Optional[Dict[str, Any]]:
        self.calls += 1
        hit = (
            self._detector(target_label)
            if self.mode == "local_vision"
            else self._from_alarms(target_label)
        )
        if hit is None:
            return None
        return self._sanitize(hit)

    def _from_alarms(self, target_label: str) -> Optional[Dict[str, Any]]:
        assert self.cache is not None
        doc, age = self.cache.get()
        if age > self.alarm_stale_s or doc is None:
            return None
        rows = doc if isinstance(doc, list) else _dig(doc, "records", "list", "rows")
        if not isinstance(rows, list):
            return None
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if target_label not in str(row):
                continue
            uri = _dig(row, "rstImage", "srcImage", "imageUrl", "picUrl")
            if not uri:
                continue
            return {"confidence": self.min_confidence, "evidence_uri": str(uri)}
        return None

    @staticmethod
    def _sanitize(hit: Mapping[str, Any]) -> Dict[str, Any]:
        """挡住 v1 禁止字段：感知结果只许回 confidence + evidence_uri。

        这一层是为了防止将来有人图方便把位姿/点云塞进感知返回值，
        那会在 validate_event 里炸掉整条 mission。
        """
        bad = FORBIDDEN_KEYS.intersection(hit.keys())
        if bad:
            raise ValueError(f"感知结果含 v1 禁止字段: {sorted(bad)}")
        return {
            "confidence": float(hit["confidence"]),
            "evidence_uri": str(hit["evidence_uri"]),
        }

    def close(self) -> None:
        if self.cache is not None:
            self.cache.stop()


class TopseeGas:
    """GasBackend 实现。

    平台的两个硬缺口（F13 / G9），这里都如实暴露而不是糊过去：
      - **没有标定记录**：`calibration_at()` 读人工台账（`GasCalibrationLedger`）。
        无台账时返回 0.0，让 DogSdkAdapter 的门禁判 stale；同时通过
        `calibration_reason()` 区分「无数据源」与「台账过期」。
      - **没有立即采样**：`sample()` 语义是按时间窗查 `gas/getGasHistory` 并聚合，
        不是触发一次采样。GAS_COMPLETED 里会带 `data_source` 说明这一点。
    """

    def __init__(
        self,
        client: TopseeClient,
        *,
        robot_id: str,
        ledger: Optional[Any] = None,
        map_id: Optional[str] = None,
        connect_window_s: float = 300.0,
        connected_probe: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.client = client
        self.robot_id = str(robot_id)
        self.ledger = ledger
        self.map_id = map_id
        self.connect_window_s = float(connect_window_s)
        self._probe = connected_probe
        self.sample_calls = 0

    def is_connected(self) -> bool:
        """传感器是否在线。

        平台没有「气体传感器在线」的直接字段，这里用「最近
        `connect_window_s` 内有历史数据」作代理判据，任何异常一律判 False
        （宁可漏采，不可假报）。需要更准的判据时注入 `connected_probe`。
        """
        if self._probe is not None:
            return bool(self._probe())
        try:
            rows = self._history(self.connect_window_s)
        except TopseeError as exc:
            logger.warning("气体在线判定失败，保守判离线: %s", exc)
            return False
        return bool(rows)

    def calibration_at(self) -> float:
        """标定时间（Unix 秒）。无台账时返回 0.0 → 门禁判 stale。"""
        if self.ledger is None:
            return 0.0
        return float(self.ledger.calibration_at(self.robot_id))

    def calibration_reason(self) -> Optional[str]:
        """可选扩展钩子：把「无数据源」与「台账过期」区分开（方案 §5.4）。"""
        if self.ledger is None:
            return "calibration_source_unavailable"
        return self.ledger.reason(self.robot_id)

    def sample(self, window_s: float) -> List[Dict[str, Any]]:
        """按 [now-window_s, now] 查气体历史并聚合成 readings。

        返回空列表是合法结果（窗口内平台没数据），由 DogSdkAdapter 转成
        GAS_FAILED(no_gas_data)，不要在这里编造读数。
        """
        self.sample_calls += 1
        try:
            rows = self._history(window_s)
        except TopseeError as exc:
            logger.warning("气体历史查询失败: %s", exc)
            return []
        readings: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            channel = _dig(row, "type", "gasType", "channel", "name")
            value = _dig(row, "value", "avg", "average", "val")
            if channel is None or value is None:
                continue
            readings.append(
                {
                    "channel": str(channel),
                    "value": float(value),
                    "unit": str(_dig(row, "unit") or ""),
                    # 平台无告警态枚举，不猜；由下游按阈值判断
                    "alarm_state": str(_dig(row, "alarmState", "alarm", "state") or "unknown"),
                }
            )
        return readings

    def _history(self, window_s: float) -> List[Any]:
        now = time.time()
        doc = self.client.get_gas_history(
            robotId=self.robot_id,
            mapId=self.map_id,
            startTime=_fmt_time(now - float(window_s)),
            endTime=_fmt_time(now),
        )
        if isinstance(doc, list):
            return doc
        rows = _dig(doc, "records", "list", "rows", "data")
        return rows if isinstance(rows, list) else []


def _fmt_time(ts: float) -> str:
    """平台时间字段格式未在 OpenAPI 声明，先按最常见的 `YYYY-MM-DD HH:MM:SS`。

    E6 抓包确认后若不符，只改这一处。
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
