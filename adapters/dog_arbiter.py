"""机器狗控制权仲裁：任务级慢通道与 WAM 级快通道分时互斥。

设计依据：`docs/design/2026-08-03-dog-integration-plan.md` §4.2–§4.5

为什么必须有这一层：手册锁死「一台机器人同一时刻仅一个控制人」（F19），
且本体控制必须在「手动巡检」模式下（F20）。两个通道同时下发速度会直接摔狗。

**关键设计取舍**：初版方案画的 `HANDOVER_TO_WAM` 自动交接被替换为
`WAITING_HUMAN_MODE_SWITCH`。理由是已核验 OpenAPI 275 个接口里**没有**
手动/自动巡检切换接口（F14），自动交接现在做不到。抓包补齐（E3）之前，
Arbiter 必须停在等待人工态，绝不许假装切换成功后直接下发 Move。

置信度门禁同理：全库 0 处 confidence 字段（F12/G4），所以 `ack_confidence()`
是显式的人工确认入口，带审计留痕，而不是伪造一个读数。
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from adapters.dog_unitree import UnitreeError, UnitreeSportClient
from adapters.topsee_client import TopseeBusyError, TopseeClient, TopseeError

logger = logging.getLogger(__name__)


class ArbiterState(str, Enum):
    IDLE = "IDLE"
    PREFLIGHT = "PREFLIGHT"
    MISSION_NAV = "MISSION_NAV"
    WAITING_HUMAN_MODE_SWITCH = "WAITING_HUMAN_MODE_SWITCH"
    WAM_ACTIVE = "WAM_ACTIVE"
    HANDOVER_TO_MISSION = "HANDOVER_TO_MISSION"
    SAFE_HOLD = "SAFE_HOLD"
    FAULT = "FAULT"


class LeaseOwner(str, Enum):
    NONE = "NONE"
    MISSION = "MISSION"
    WAM = "WAM"


class ArbiterRejected(RuntimeError):
    """非法转移或前置条件不满足。"""


# preflight 失败原因
PREFLIGHT_CONFIDENCE_UNAVAILABLE = "confidence_unavailable"
PREFLIGHT_CONFIDENCE_LOW = "confidence_below_gate"
PREFLIGHT_BATTERY_LOW = "battery_low"
PREFLIGHT_BATTERY_UNKNOWN = "battery_unknown"
PREFLIGHT_CONTROLLER_BUSY = "controller_busy"
PREFLIGHT_PLATFORM_ERROR = "platform_error"

# 只有 IDLE / SAFE_HOLD / FAULT 允许「无人持有」，其余状态必须恰有一个所有者
_NO_OWNER_STATES = frozenset(
    {ArbiterState.IDLE, ArbiterState.SAFE_HOLD, ArbiterState.FAULT}
)


class DogControlArbiter:
    """单命令所有者 + 租约 + preflight 门禁 + 命令看门狗。

    线程模型：与 MissionBrain 同一个单线程 tick 循环。内部所有平台调用都是
    短超时同步调用（stopTask / updateControllerUser），周期状态由调用方的
    PollCache 提供，不在这里起线程。
    """

    # 平台急停≈断电（F18），永远不在常规停止路径里
    normal_stop_methods = ("unitree_stop_move", "unitree_damp", "topsee_stop_task")

    def __init__(
        self,
        topsee: Optional[TopseeClient],
        unitree: Optional[UnitreeSportClient],
        *,
        robot_id: str,
        min_confidence: float = 0.9,
        min_battery_pct: float = 25.0,
        lease_ttl_s: float = 60.0,
        cmd_watchdog_s: float = 0.3,
        confidence_ack_ttl_s: float = 300.0,
        confidence_provider: Optional[Callable[[], Optional[float]]] = None,
        battery_provider: Optional[Callable[[], Optional[float]]] = None,
        controller_state: Optional[str] = None,
        controller_force: Optional[str] = None,
        audit_dir: Union[str, Path] = "logs/audit",
    ) -> None:
        """
        min_battery_pct: 手册电量分层（F29）是「格」，25% 约等于「低于两格」。
        confidence_ack_ttl_s: 人工确认的置信度过多久失效。长期趴下会丢定位
            （手册 §5.6），所以确认不能一劳永逸。
        controller_state / controller_force: `updateControllerUser` 的取值待
            抓包确认（G7）；未配置时抢占步骤会被跳过并记警告，不会假装成功。
        audit_dir: ack_confidence 审计 jsonl 目录（D0 Q3）。
        """
        self.topsee = topsee
        self.unitree = unitree
        self.robot_id = str(robot_id)
        self.min_confidence = float(min_confidence)
        self.min_battery_pct = float(min_battery_pct)
        self.lease_ttl_s = float(lease_ttl_s)
        self.cmd_watchdog_s = float(cmd_watchdog_s)
        self.confidence_ack_ttl_s = float(confidence_ack_ttl_s)
        self._confidence_provider = confidence_provider
        self._battery_provider = battery_provider
        self._controller_state = controller_state
        self._controller_force = controller_force
        self.audit_dir = Path(audit_dir)

        self._state = ArbiterState.IDLE
        self._owner = LeaseOwner.NONE
        self._lease_token: Optional[str] = None
        self._lease_expires_at: float = 0.0
        self._mission_id: Optional[str] = None
        self._human_mode_ack = False
        self._confidence_ack: Optional[float] = None
        self._confidence_ack_at: float = 0.0
        self._confidence_ack_by: str = ""
        self._confidence_ack_expiry: float = 0.0
        self._preflight_ok = False
        self._last_reject: Optional[str] = None

        # 供 episode meta 的 single_owner_ok 校验用（方案 §6.7）
        self.lease_log: List[Dict[str, Any]] = []
        self.illegal_transitions = 0
        self.watchdog_trips = 0
        self.confidence_audit_path: Optional[Path] = None

    # ---------- 只读视图 ----------

    @property
    def state(self) -> ArbiterState:
        return self._state

    @property
    def owner(self) -> LeaseOwner:
        return self._owner

    @property
    def lease_token(self) -> Optional[str]:
        return self._lease_token

    @property
    def mission_id(self) -> Optional[str]:
        return self._mission_id

    @property
    def human_mode_ack(self) -> bool:
        return self._human_mode_ack

    @property
    def preflight_ok(self) -> bool:
        return self._preflight_ok

    @property
    def last_reject(self) -> Optional[str]:
        return self._last_reject

    @property
    def topsee_cmd_enabled(self) -> bool:
        """慢通道是否被授权。与 unitree_cmd_enabled 互斥（不变量 I1）。"""
        return self._state in (ArbiterState.PREFLIGHT, ArbiterState.MISSION_NAV)

    @property
    def unitree_cmd_enabled(self) -> bool:
        """快通道是否被授权。仅 WAM_ACTIVE。"""
        return self._state is ArbiterState.WAM_ACTIVE

    @property
    def no_owner(self) -> bool:
        return self._owner is LeaseOwner.NONE

    @property
    def has_single_owner(self) -> bool:
        return not (self.topsee_cmd_enabled and self.unitree_cmd_enabled)

    def confidence_ack(self, now: Optional[float] = None) -> float:
        """当前有效的置信度确认值。缺失或过期返回 -1.0（必然挡住门禁）。"""
        if self._confidence_ack is None:
            return -1.0
        t = float(now if now is not None else time.time())
        if (t - self._confidence_ack_at) > self.confidence_ack_ttl_s:
            return -1.0
        return float(self._confidence_ack)

    def allow_topsee_cmd(self) -> bool:
        return self.topsee_cmd_enabled

    # ---------- 转移 ----------

    def _goto(self, new: ArbiterState, *, why: str) -> None:
        old = self._state
        self._state = new
        logger.info("Arbiter %s → %s (%s)", old.value, new.value, why)

    def _reject(self, reason: str) -> None:
        self._last_reject = reason
        self.illegal_transitions += 1
        raise ArbiterRejected(reason)

    def _log_lease(self, owner: LeaseOwner, action: str, now: float) -> None:
        self.lease_log.append(
            {"at": now, "owner": owner.value, "action": action, "state": self._state.value}
        )

    def ack_confidence(self, value: float, *, by: str) -> None:
        """人工录入定位置信度（G4：平台无接口）。

        必须带确认人，因为这是绕过程序化门禁的唯一入口，要能审计。
        D0 Q3：校验 [0,1] 有限数，并持久化 who/value/expiry 到 audit jsonl。
        """
        if not by:
            raise ValueError("ack_confidence 必须提供确认人（by）")
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("ack_confidence 值必须是 [0,1] 有限数") from exc
        if not math.isfinite(v) or v < 0.0 or v > 1.0:
            raise ValueError("ack_confidence 值必须是 [0,1] 有限数")
        now = time.time()
        expiry = now + self.confidence_ack_ttl_s
        # 先落盘再改内存：审计失败不得留下可用的未审计确认
        self._persist_confidence_audit(v, by=str(by), expiry=expiry)
        self._confidence_ack = v
        self._confidence_ack_at = now
        self._confidence_ack_by = str(by)
        self._confidence_ack_expiry = expiry
        logger.info("置信度人工确认: %.3f by=%s expiry=%.0f", v, by, expiry)

    def _persist_confidence_audit(self, value: float, *, by: str, expiry: float) -> None:
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        path = self.audit_dir / (
            f"confidence_{time.strftime('%Y%m%d', time.localtime())}.jsonl"
        )
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "who": by,
            "value": value,
            "expiry": expiry,
            "robot_id": self.robot_id,
            "ttl_s": self.confidence_ack_ttl_s,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.confidence_audit_path = path

    def acquire_for_mission(self, mission_id: str, *, now: Optional[float] = None) -> None:
        """IDLE → PREFLIGHT → MISSION_NAV 的准备段。

        失败时状态回到 IDLE（可重试）或进 FAULT（须人工），并抛 ArbiterRejected。
        """
        t = float(now if now is not None else time.time())
        if self._state not in (ArbiterState.IDLE, ArbiterState.MISSION_NAV):
            self._reject(f"acquire_for_mission 非法起始状态: {self._state.value}")
        self._mission_id = str(mission_id)
        self._goto(ArbiterState.PREFLIGHT, why=f"mission={mission_id}")
        self._preflight_ok = False
        try:
            self._run_preflight(t)
        except ArbiterRejected:
            self._goto(ArbiterState.IDLE, why="preflight 失败")
            raise
        self._preflight_ok = True
        self._owner = LeaseOwner.MISSION
        self._log_lease(LeaseOwner.MISSION, "acquire", t)
        self._goto(ArbiterState.MISSION_NAV, why="preflight 通过")

    def _run_preflight(self, t: float) -> None:
        # 1) 置信度（≥0.9，F22）。有 provider 用 provider，否则用人工确认值。
        conf = None
        if self._confidence_provider is not None:
            try:
                conf = self._confidence_provider()
            except Exception:  # noqa: BLE001 — provider 失败不等于通过
                logger.exception("confidence_provider 失败")
                conf = None
        if conf is None:
            conf = self.confidence_ack(now=t)
            if conf < 0:
                self._reject(PREFLIGHT_CONFIDENCE_UNAVAILABLE)
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            self._reject(PREFLIGHT_CONFIDENCE_UNAVAILABLE)
            return  # pragma: no cover — _reject 恒抛
        if not math.isfinite(conf_f):
            self._reject(PREFLIGHT_CONFIDENCE_UNAVAILABLE)
        if conf_f < self.min_confidence:
            self._reject(PREFLIGHT_CONFIDENCE_LOW)

        # 2) 电量分层（F29）—— D0 Q4 fail-closed：缺失 / None / NaN 一律拒绝
        if self._battery_provider is None:
            self._reject(PREFLIGHT_BATTERY_UNKNOWN)
        try:
            bat = self._battery_provider()
        except Exception:  # noqa: BLE001
            logger.exception("battery_provider 失败")
            bat = None
        try:
            bat_f = float(bat) if bat is not None else float("nan")
        except (TypeError, ValueError):
            bat_f = float("nan")
        if not math.isfinite(bat_f):
            self._reject(PREFLIGHT_BATTERY_UNKNOWN)
        if bat_f < self.min_battery_pct:
            self._reject(PREFLIGHT_BATTERY_LOW)

        if self.topsee is None:
            return

        # 3) 已有任务先停（避免双调度源）
        try:
            cur = self.topsee.get_current_task(self.robot_id)
            if isinstance(cur, Mapping) and cur:
                logger.info("preflight 发现在执行任务，先 stopTask")
                self.topsee.stop_task(self.robot_id)
        except TopseeError as exc:
            logger.warning("preflight 查询/停止既有任务失败: %s", exc)
            self._reject(PREFLIGHT_PLATFORM_ERROR)

        # 4) 抢控制权。取值未确认（G7）时跳过并记警告，不假装成功。
        if self._controller_state is None and self._controller_force is None:
            logger.warning(
                "updateControllerUser 的 state/force 取值未配置（G7 待抓包），"
                "跳过控制权抢占；真机联调前必须补上"
            )
            return
        try:
            self.topsee.update_controller_user(
                self.robot_id, state=self._controller_state, force=self._controller_force
            )
        except TopseeBusyError:
            self._reject(PREFLIGHT_CONTROLLER_BUSY)
        except TopseeError as exc:
            logger.warning("抢控制权失败: %s", exc)
            self._reject(PREFLIGHT_PLATFORM_ERROR)

    def request_wam(self, *, now: Optional[float] = None) -> ArbiterState:
        """MISSION_NAV → WAITING_HUMAN_MODE_SWITCH。

        这里就做两件事：停掉平台任务、把状态挂到等待人工。**不会**自动切模式，
        因为那个接口不存在（F14）。返回落地状态供调用方判断。
        """
        t = float(now if now is not None else time.time())
        if self._state not in (ArbiterState.MISSION_NAV, ArbiterState.PREFLIGHT):
            self._reject(f"request_wam 非法起始状态: {self._state.value}")
        if self.topsee is not None:
            try:
                self.topsee.stop_task(self.robot_id)
            except TopseeError as exc:
                logger.warning("request_wam 停任务失败，转 SAFE_HOLD: %s", exc)
                self.safe_hold("stop_task_failed")
                return self._state
        self._human_mode_ack = False
        self._goto(ArbiterState.WAITING_HUMAN_MODE_SWITCH, why="等待人工切手动巡检")
        self._log_lease(self._owner, "await_human_mode", t)
        return self._state

    def ack_human_mode_switch(self, *, by: str, now: Optional[float] = None) -> str:
        """人工已在 Web/APP 切到「手动巡检」后调用，签发 WAM 租约。

        仍要过 DDS 权限探测（E4/G11）；探测失败转 SAFE_HOLD 并抛异常，
        绝不带着不确定的控制权进 WAM_ACTIVE。
        """
        if not by:
            raise ValueError("ack_human_mode_switch 必须提供确认人（by）")
        t = float(now if now is not None else time.time())
        if self._state is not ArbiterState.WAITING_HUMAN_MODE_SWITCH:
            self._reject(f"ack_human_mode_switch 非法起始状态: {self._state.value}")
        if self.unitree is None:
            self.safe_hold("unitree_absent")
            self._reject("未配置 unitree 通道，无法进入 WAM")
        self._human_mode_ack = True
        logger.info("人工确认已切手动巡检 by=%s", by)
        if not self.unitree.probe_dds_authority():
            self._human_mode_ack = False
            self.safe_hold("dds_authority_probe_failed")
            self._reject("DDS 权限探测失败（G11），拒绝进入 WAM_ACTIVE")

        token = str(uuid.uuid4())
        self._lease_token = token
        self._lease_expires_at = t + self.lease_ttl_s
        self._owner = LeaseOwner.WAM
        self.unitree.lease_token = token
        self._goto(ArbiterState.WAM_ACTIVE, why=f"租约签发 by={by}")
        self._log_lease(LeaseOwner.WAM, "grant", t)
        return token

    def renew_lease(self, token: str, *, now: Optional[float] = None) -> None:
        t = float(now if now is not None else time.time())
        if self._state is not ArbiterState.WAM_ACTIVE or token != self._lease_token:
            self._reject("renew_lease 的 token 无效或状态不允许")
        self._lease_expires_at = t + self.lease_ttl_s

    def move(self, vx: float, vy: float, vyaw: float, *, token: str) -> None:
        """WAM 的唯一速度出口。状态不对或 token 不对一律拒绝。"""
        if self._state is not ArbiterState.WAM_ACTIVE:
            self._reject(f"move 在 {self._state.value} 下不允许")
        if token != self._lease_token:
            self._reject("move 的 token 与租约不符")
        assert self.unitree is not None
        self.unitree.move(vx, vy, vyaw, lease_token=token)

    def begin_handover_to_mission(self, *, now: Optional[float] = None) -> None:
        """WAM 段正常收尾：停速度 → 吊销租约 → 释放控制权 → IDLE。"""
        t = float(now if now is not None else time.time())
        if self._state not in (
            ArbiterState.WAM_ACTIVE,
            ArbiterState.WAITING_HUMAN_MODE_SWITCH,
        ):
            self._reject(f"begin_handover_to_mission 非法起始状态: {self._state.value}")
        self._goto(ArbiterState.HANDOVER_TO_MISSION, why="WAM 段结束")
        self._revoke_lease(t, action="handover")
        self._release_controller_best_effort()
        self._human_mode_ack = False
        self._preflight_ok = False
        self._owner = LeaseOwner.NONE
        self._goto(ArbiterState.IDLE, why="已释放")

    def force_release(self, reason: str, *, now: Optional[float] = None) -> None:
        """任务级 abort 的执行体。由 DogSdkAdapter.abort / Supervisor 调用。

        **绝不抛异常**：abort 路径不允许因为网络或状态问题失败。
        """
        t = float(now if now is not None else time.time())
        logger.warning("Arbiter force_release: %s", reason)
        self._stop_motion_best_effort()
        self._revoke_lease(t, action=f"force_release:{reason}")
        if self.topsee is not None:
            try:
                self.topsee.stop_task(self.robot_id)
            except TopseeError as exc:
                logger.warning("force_release 的 stopTask 失败（已忽略）: %s", exc)
        self._release_controller_best_effort()
        self._human_mode_ack = False
        self._preflight_ok = False
        self._owner = LeaseOwner.NONE
        self._goto(ArbiterState.IDLE, why=f"force_release({reason})")

    def safe_hold(self, reason: str, *, now: Optional[float] = None) -> None:
        """零速度保持，等人工或超时。也绝不抛异常。"""
        t = float(now if now is not None else time.time())
        logger.warning("Arbiter safe_hold: %s", reason)
        self._stop_motion_best_effort()
        self._revoke_lease(t, action=f"safe_hold:{reason}")
        self._owner = LeaseOwner.NONE
        self._preflight_ok = False
        self._goto(ArbiterState.SAFE_HOLD, why=reason)

    def fault(self, reason: str, *, now: Optional[float] = None) -> None:
        """进入需人工介入的终态。摔倒、DDS 被占、定位丢失都走这里。

        手册 §5.7 要求摔倒必须人工物理介入（切断动力→扶正→重启自检），
        所以这个状态**不允许**程序自动恢复。
        """
        t = float(now if now is not None else time.time())
        logger.error("Arbiter FAULT: %s（须人工 reset_after_fault）", reason)
        self._stop_motion_best_effort()
        self._revoke_lease(t, action=f"fault:{reason}")
        self._owner = LeaseOwner.NONE
        self._preflight_ok = False
        self._human_mode_ack = False
        self._goto(ArbiterState.FAULT, why=reason)

    def reset_after_fault(self, *, by: str) -> None:
        if not by:
            raise ValueError("reset_after_fault 必须提供操作人（by）")
        if self._state is not ArbiterState.FAULT:
            self._reject(f"reset_after_fault 只能从 FAULT 调用，当前 {self._state.value}")
        logger.info("人工复位 FAULT by=%s", by)
        self._goto(ArbiterState.IDLE, why=f"人工复位 by={by}")

    def resume_from_hold(self, *, by: str) -> None:
        if not by:
            raise ValueError("resume_from_hold 必须提供操作人（by）")
        if self._state is not ArbiterState.SAFE_HOLD:
            self._reject(f"resume_from_hold 只能从 SAFE_HOLD 调用，当前 {self._state.value}")
        self._goto(ArbiterState.IDLE, why=f"人工恢复 by={by}")

    # ---------- 周期 ----------

    def tick(self, now: Optional[float] = None) -> None:
        """看门狗 + 租约 TTL。必须被 MissionBrain 的 tick 循环带着跑。"""
        t = float(now if now is not None else time.time())
        if self._state is not ArbiterState.WAM_ACTIVE:
            return
        if t >= self._lease_expires_at:
            logger.warning("WAM 租约到期，收杆")
            self.begin_handover_to_mission(now=t)
            return
        u = self.unitree
        if u is not None and u.last_cmd_at > 0.0:
            idle = time.monotonic() - u.last_cmd_at
            if idle > self.cmd_watchdog_s and u.last_cmd != (0.0, 0.0, 0.0):
                self.watchdog_trips += 1
                logger.warning("命令看门狗超时 %.3fs，强制零速度", idle)
                self.safe_hold("cmd_watchdog_timeout", now=t)

    # ---------- 内部收尾 ----------

    def _revoke_lease(self, t: float, *, action: str) -> None:
        if self._lease_token is not None:
            self._log_lease(self._owner, action, t)
        self._lease_token = None
        self._lease_expires_at = 0.0
        if self.unitree is not None:
            self.unitree.lease_token = None

    def _stop_motion_best_effort(self) -> None:
        if self.unitree is None:
            return
        try:
            self.unitree.stop_move()
        except UnitreeError as exc:
            logger.warning("StopMove 失败，尝试 Damp: %s", exc)
            try:
                self.unitree.damp()
            except UnitreeError:
                logger.error("Damp 也失败——需人工用遥控器 L2+B 阻尼")

    def _release_controller_best_effort(self) -> None:
        if self.topsee is None or self._controller_state is None:
            return
        try:
            self.topsee.update_controller_user(self.robot_id, state=self._controller_state)
        except TopseeError as exc:
            logger.warning("释放控制权失败（已忽略，等平台侧超时）: %s", exc)
