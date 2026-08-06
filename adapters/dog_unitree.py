"""宇树 B2 运动通道（DDS）。

**红线（方案 §6.6）**：本模块产出的位姿、速度、本体感受数据只准写入 WAM
落盘通道，**禁止**经过 `mission_brain.events`。事件契约 v1 的 FORBIDDEN_KEYS
会在运行时拒绝 `pose_xyz` / `global_pose`，把这类数据塞进事件会炸掉整条
mission；而且 odom 未与平台地图对齐时冒充全局坐标，会让无人机 Scout 与狗的
goal 错配——这是最危险的失败模式。

**安全缺口（方案 §6.4 / F30）**：遥控器档位的软限位（低速档爬坡 <45°、
台阶 <25 cm、越障 <40 cm）在 DDS `Move()` 直控路径上**不生效**。所以速度
安全盒必须由我们自己实现，见 `SpeedLimits`。

传输层是可注入的：`DdsTransport` 走真机 unitree_sdk2py，`LoopbackTransport`
供无硬件时的单测与干跑使用。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

# 宇树 sport 模式 API ID（方案附录 B）
API_DAMP = 1001
API_STOP_MOVE = 1003
API_STAND_UP = 1004
API_STAND_DOWN = 1005
API_MOVE = 1008
API_SWITCH_GAIT = 1011
API_BODY_HEIGHT = 1013
API_SPEED_LEVEL = 1015
API_MOVE_TO_POS = 1036

TOPIC_SPORT_STATE = "rt/sportmodestate"
TOPIC_LOW_STATE = "rt/lowstate"

# D0 / v3.1 §3-D0：>500ms 无样本 → dds_stale
DDS_STALE_S = 0.5


def _as_float_list(value: Any, *, n: int) -> List[float]:
    if value is None:
        return [0.0] * n
    if isinstance(value, (list, tuple)):
        out = [float(x) for x in value[:n]]
    else:
        try:
            out = [float(x) for x in list(value)[:n]]
        except TypeError:
            out = [float(value)]
    while len(out) < n:
        out.append(0.0)
    return out


def sport_state_to_dict(msg: Any, *, t_mono: Optional[float] = None) -> Dict[str, Any]:
    """把 SportModeState IDL / Mapping 归一成 read() 契约字典。"""
    if isinstance(msg, Mapping):
        pos = _as_float_list(msg.get("position"), n=3)
        vel = _as_float_list(msg.get("velocity"), n=3)
        yaw_speed = float(msg.get("yaw_speed", 0.0))
        imu = msg.get("imu_state") or {}
        rpy = _as_float_list(
            imu.get("rpy") if isinstance(imu, Mapping) else getattr(imu, "rpy", None),
            n=3,
        )
    else:
        pos = _as_float_list(getattr(msg, "position", None), n=3)
        vel = _as_float_list(getattr(msg, "velocity", None), n=3)
        yaw_speed = float(getattr(msg, "yaw_speed", 0.0) or 0.0)
        imu = getattr(msg, "imu_state", None)
        rpy = _as_float_list(getattr(imu, "rpy", None) if imu is not None else None, n=3)
    return {
        "position": pos,
        "velocity": vel,
        "yaw_speed": yaw_speed,
        "imu_state": {"rpy": rpy},
        "t_mono": float(t_mono if t_mono is not None else time.monotonic()),
    }


def low_state_to_dict(msg: Any, *, t_mono: Optional[float] = None) -> Dict[str, Any]:
    """把 LowState IDL / Mapping 归一成 read() 契约字典。"""
    if isinstance(msg, Mapping):
        bms = msg.get("bms_state") or {}
        soc = bms.get("soc") if isinstance(bms, Mapping) else getattr(bms, "soc", None)
        motor = msg.get("motor_state")
    else:
        bms = getattr(msg, "bms_state", None)
        soc = getattr(bms, "soc", None) if bms is not None else None
        motor = getattr(msg, "motor_state", None)
    motor_out: List[Dict[str, float]] = []
    if motor is not None:
        for m in list(motor)[:12]:
            if isinstance(m, Mapping):
                motor_out.append({"q": float(m.get("q", 0.0))})
            else:
                motor_out.append({"q": float(getattr(m, "q", 0.0) or 0.0)})
    while len(motor_out) < 12:
        motor_out.append({"q": 0.0})
    return {
        "bms_state": {"soc": float(soc) if soc is not None else 0.0},
        "motor_state": motor_out,
        "t_mono": float(t_mono if t_mono is not None else time.monotonic()),
    }


class UnitreeError(Exception):
    """运动通道错误基类。"""


class UnitreeNotConnected(UnitreeError):
    """DDS 未连接或依赖缺失。"""


class UnitreeAuthorityError(UnitreeError):
    """无有效租约，或命令下发被边缘侧独占（G11 未验证前必须当真会发生）。"""


class UnitreeLimitError(UnitreeError):
    """命令超出安全盒。"""


@dataclass(frozen=True)
class SportPose:
    """`/sportmodestate` 的一帧。坐标系是 odom(sport)，**不是**平台地图系。"""

    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    vyaw: float
    t_mono: float


@dataclass(frozen=True)
class SpeedLimits:
    """速度安全盒。默认值远低于 B2 整机极限（5 m/s），采集起步用。

    方案 §6.4：训练采集建议 |v| ≤ 1.0 m/s。想放宽必须是显式决定，
    不能靠默认值悄悄放开。
    """

    max_vx: float = 1.0
    max_vy: float = 0.6
    max_vyaw: float = 1.0

    def clamp(self, vx: float, vy: float, vyaw: float) -> Tuple[float, float, float]:
        return (
            max(-self.max_vx, min(self.max_vx, float(vx))),
            max(-self.max_vy, min(self.max_vy, float(vy))),
            max(-self.max_vyaw, min(self.max_vyaw, float(vyaw))),
        )

    def exceeds(self, vx: float, vy: float, vyaw: float) -> bool:
        return (
            abs(float(vx)) > self.max_vx
            or abs(float(vy)) > self.max_vy
            or abs(float(vyaw)) > self.max_vyaw
        )


class SportTransport(Protocol):
    """运动通道传输层。真机是 DDS，测试是 loopback。"""

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def call(self, api_id: int, payload: Mapping[str, Any]) -> Any: ...

    def read(self, topic: str) -> Optional[Mapping[str, Any]]: ...

    def sample_age_s(self, topic: str) -> Optional[float]: ...


class LoopbackTransport:
    """无硬件的传输层：记录命令、按命令积分出一个可信的假位姿。

    存在意义不只是测试——它让 Arbiter 的全部状态转移和不变量在没有狗的
    情况下也能回归，符合项目「不依赖真机就能跑软件测试」的要求。
    """

    def __init__(
        self,
        *,
        authority: bool = True,
        state_available: bool = True,
        raise_on_call: bool = False,
    ) -> None:
        self.authority = bool(authority)
        self.state_available = bool(state_available)
        self.raise_on_call = bool(raise_on_call)
        self.connected = False
        self.calls: list[Tuple[int, Dict[str, Any]]] = []
        self._pose = SportPose(0.0, 0.0, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._last_cmd_at = 0.0

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def call(self, api_id: int, payload: Mapping[str, Any]) -> Any:
        if not self.connected:
            raise UnitreeNotConnected("loopback 未 connect")
        if self.raise_on_call:
            raise UnitreeError("loopback 故意失败")
        self.calls.append((int(api_id), dict(payload)))
        if not self.authority:
            # 模拟「命令发出去了但本体无响应」——G11 未验证时的典型失败模式
            return None
        if api_id == API_MOVE:
            self._integrate(
                float(payload.get("vx", 0.0)),
                float(payload.get("vy", 0.0)),
                float(payload.get("vyaw", 0.0)),
            )
        elif api_id in (API_STOP_MOVE, API_DAMP):
            self._integrate(0.0, 0.0, 0.0)
        return {"ok": True}

    def _integrate(self, vx: float, vy: float, vyaw: float) -> None:
        now = time.monotonic()
        dt = 0.0 if self._last_cmd_at == 0.0 else max(0.0, min(0.5, now - self._last_cmd_at))
        self._last_cmd_at = now
        yaw = self._pose.yaw + vyaw * dt
        self._pose = replace(
            self._pose,
            x=self._pose.x + vx * dt,
            y=self._pose.y + vy * dt,
            yaw=yaw,
            vx=vx,
            vy=vy,
            vyaw=vyaw,
            t_mono=now,
        )

    def read(self, topic: str) -> Optional[Mapping[str, Any]]:
        if not self.connected or not self.state_available:
            return None
        now = time.monotonic()
        if topic == TOPIC_SPORT_STATE:
            p = self._pose
            return {
                "position": [p.x, p.y, p.z],
                "velocity": [p.vx, p.vy, 0.0],
                "yaw_speed": p.vyaw,
                "imu_state": {"rpy": [0.0, 0.0, p.yaw]},
                "t_mono": now if p.t_mono == 0.0 else float(p.t_mono),
            }
        if topic == TOPIC_LOW_STATE:
            return {
                "bms_state": {"soc": 88},
                "motor_state": [{"q": 0.0}] * 12,
                "t_mono": now,
            }
        return None

    def sample_age_s(self, topic: str) -> Optional[float]:
        msg = self.read(topic)
        if msg is None:
            return None
        return max(0.0, time.monotonic() - float(msg["t_mono"]))

    # 测试辅助
    def teleport(self, x: float, y: float, yaw: float = 0.0) -> None:
        self._pose = replace(
            self._pose, x=float(x), y=float(y), yaw=float(yaw), t_mono=time.monotonic()
        )

    def calls_of(self, api_id: int) -> list[Dict[str, Any]]:
        return [p for a, p in self.calls if a == api_id]


class DdsTransport:
    """真机 DDS 传输层。依赖 `unitree_sdk2py`，导入失败时如实报错。

    D0（v3.1）：订阅 `/sportmodestate` 与 `/lowstate`；`read()` 返回带单调钟
    `t_mono` 的最新样本；`sample_age_s > 0.5` → 上层判 `dds_stale`。

    这里刻意不做「装不上就退化成 loopback」的兜底：那会让真机跑在假数据上。
    """

    def __init__(
        self,
        *,
        interface: str,
        domain_id: int = 0,
        subscriber_factory: Optional[Callable[[str, Any], Any]] = None,
    ) -> None:
        self.interface = interface
        self.domain_id = int(domain_id)
        self._subscriber_factory = subscriber_factory
        self._sport: Any = None
        self._subs: Dict[str, Any] = {}
        self._latest: Dict[str, Dict[str, Any]] = {}

    def connect(self) -> None:
        try:
            from unitree_sdk2py.core.channel import (  # type: ignore[import-not-found]
                ChannelFactoryInitialize,
                ChannelSubscriber,
            )
            from unitree_sdk2py.go2.sport.sport_client import (  # type: ignore[import-not-found]
                SportClient,
            )
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # type: ignore[import-not-found]
                LowState_,
                SportModeState_,
            )
        except ImportError as exc:  # pragma: no cover — 无 SDK 环境
            raise UnitreeNotConnected(
                "缺少 unitree_sdk2py。真机联调前先在狗侧网络的机器上安装官方 SDK，"
                "不要退化成 loopback 假数据。"
            ) from exc
        ChannelFactoryInitialize(self.domain_id, self.interface)
        self._sport = SportClient()
        self._sport.Init()
        factory = self._subscriber_factory or (
            lambda topic, msg_type: ChannelSubscriber(topic, msg_type)
        )
        self._start_subscriber(TOPIC_SPORT_STATE, SportModeState_, sport_state_to_dict, factory)
        self._start_subscriber(TOPIC_LOW_STATE, LowState_, low_state_to_dict, factory)
        logger.info(
            "DDS 已连接 interface=%s domain=%s topics=%s",
            self.interface,
            self.domain_id,
            sorted(self._subs),
        )

    def _start_subscriber(
        self,
        topic: str,
        msg_type: Any,
        converter: Callable[..., Dict[str, Any]],
        factory: Callable[[str, Any], Any],
    ) -> None:
        sub = factory(topic, msg_type)

        def _handler(msg: Any) -> None:
            self._ingest(topic, converter(msg))

        # unitree_sdk2py: Init(handler, queue_len)；测试桩可无 Init
        init = getattr(sub, "Init", None)
        if callable(init):
            init(_handler, 10)
        self._subs[topic] = sub

    def _ingest(self, topic: str, sample: Mapping[str, Any]) -> None:
        """回调与单测共用：写入带 t_mono 的最新样本。"""
        data = dict(sample)
        data.setdefault("t_mono", time.monotonic())
        self._latest[topic] = data

    def close(self) -> None:
        self._sport = None
        self._subs.clear()
        self._latest.clear()

    def call(self, api_id: int, payload: Mapping[str, Any]) -> Any:  # pragma: no cover
        if self._sport is None:
            raise UnitreeNotConnected("DdsTransport 未 connect")
        sport = self._sport
        if api_id == API_MOVE:
            return sport.Move(payload["vx"], payload["vy"], payload["vyaw"])
        if api_id == API_STOP_MOVE:
            return sport.StopMove()
        if api_id == API_DAMP:
            return sport.Damp()
        if api_id == API_STAND_UP:
            return sport.StandUp()
        if api_id == API_STAND_DOWN:
            return sport.StandDown()
        if api_id == API_SWITCH_GAIT:
            return sport.SwitchGait(int(payload["gait"]))
        if api_id == API_BODY_HEIGHT:
            return sport.BodyHeight(float(payload["height"]))
        if api_id == API_SPEED_LEVEL:
            return sport.SpeedLevel(int(payload["level"]))
        if api_id == API_MOVE_TO_POS:
            return sport.MoveToPos(payload["x"], payload["y"], payload["yaw"])
        raise UnitreeError(f"未支持的 api_id={api_id}")

    def read(self, topic: str) -> Optional[Mapping[str, Any]]:
        sample = self._latest.get(topic)
        return None if sample is None else dict(sample)

    def sample_age_s(self, topic: str) -> Optional[float]:
        sample = self._latest.get(topic)
        if sample is None:
            return None
        return max(0.0, time.monotonic() - float(sample["t_mono"]))


class UnitreeSportClient:
    """运动通道门面：位姿订阅 + 速度下发 + 安全停。

    所有 `move()` 必须带 Arbiter 签发的租约 token；这是「两个通道绝不同时
    下发速度」这条不变量在执行层的最后一道闩。
    """

    def __init__(
        self,
        transport: SportTransport,
        *,
        limits: Optional[SpeedLimits] = None,
        clamp_instead_of_raise: bool = True,
    ) -> None:
        self.transport = transport
        self.limits = limits or SpeedLimits()
        self.clamp_instead_of_raise = bool(clamp_instead_of_raise)
        self.lease_token: Optional[str] = None
        self.last_move_token: Optional[str] = None
        self.last_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_cmd_at: float = 0.0
        self.move_calls = 0
        self.clamp_events = 0
        self._connected = False

    # ---------- 生命周期 ----------

    def connect(self) -> None:
        self.transport.connect()
        self._connected = True

    def close(self) -> None:
        try:
            self.stop_move()
        except UnitreeError:
            logger.warning("关闭前 StopMove 失败")
        self.transport.close()
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise UnitreeNotConnected("UnitreeSportClient 未 connect")

    # ---------- 状态 ----------

    def get_sport_state(self) -> Optional[SportPose]:
        """读一帧 `/sportmodestate`。无样本返回 None。`t_mono` 取样本时间戳。"""
        self._require_connected()
        msg = self.transport.read(TOPIC_SPORT_STATE)
        if not isinstance(msg, Mapping):
            return None
        pos = msg.get("position") or [0.0, 0.0, 0.0]
        vel = msg.get("velocity") or [0.0, 0.0, 0.0]
        rpy = ((msg.get("imu_state") or {}).get("rpy")) or [0.0, 0.0, 0.0]
        t_mono = float(msg["t_mono"]) if msg.get("t_mono") is not None else time.monotonic()
        return SportPose(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]) if len(pos) > 2 else 0.0,
            yaw=float(rpy[2]) if len(rpy) > 2 else 0.0,
            vx=float(vel[0]),
            vy=float(vel[1]),
            vyaw=float(msg.get("yaw_speed", 0.0)),
            t_mono=t_mono,
        )

    def get_low_state(self) -> Optional[Mapping[str, Any]]:
        """读 `/lowstate`（≈500 Hz）。调用方须自限频率，勿逐 tick 拉。"""
        self._require_connected()
        return self.transport.read(TOPIC_LOW_STATE)

    def sample_age_s(self, topic: str = TOPIC_SPORT_STATE) -> Optional[float]:
        """最新样本年龄（秒）。无样本返回 None。"""
        self._require_connected()
        age_fn = getattr(self.transport, "sample_age_s", None)
        if callable(age_fn):
            return age_fn(topic)
        msg = self.transport.read(topic)
        if not isinstance(msg, Mapping) or msg.get("t_mono") is None:
            return None
        return max(0.0, time.monotonic() - float(msg["t_mono"]))

    def is_dds_stale(
        self, topic: str = TOPIC_SPORT_STATE, *, max_age_s: float = DDS_STALE_S
    ) -> bool:
        """>max_age_s 无新鲜样本 → True（含从未收到）。"""
        age = self.sample_age_s(topic)
        if age is None:
            return True
        return age > float(max_age_s)

    def pose_xy_yaw(self) -> Optional[Tuple[float, float, float]]:
        """便捷读数，供 TopseeNav 的距离到点判据用（odom 系）。"""
        p = self.get_sport_state()
        return None if p is None else (p.x, p.y, p.yaw)

    # ---------- 运动 ----------

    def move(self, vx: float, vy: float, vyaw: float, *, lease_token: str) -> None:
        """连续速度（API 1008）。token 不匹配直接拒绝。"""
        self._require_connected()
        if not lease_token or lease_token != self.lease_token:
            raise UnitreeAuthorityError("租约 token 不匹配，拒绝下发速度")
        if self.limits.exceeds(vx, vy, vyaw):
            if not self.clamp_instead_of_raise:
                raise UnitreeLimitError(
                    f"速度 ({vx:.2f},{vy:.2f},{vyaw:.2f}) 超出安全盒 {self.limits}"
                )
            self.clamp_events += 1
            logger.warning(
                "速度超安全盒已裁剪: (%.2f,%.2f,%.2f) → limits=%s", vx, vy, vyaw, self.limits
            )
        cvx, cvy, cvyaw = self.limits.clamp(vx, vy, vyaw)
        self.transport.call(API_MOVE, {"vx": cvx, "vy": cvy, "vyaw": cvyaw})
        self.move_calls += 1
        self.last_move_token = lease_token
        self.last_cmd = (cvx, cvy, cvyaw)
        self.last_cmd_at = time.monotonic()

    def stop_move(self) -> None:
        """StopMove（1003）：常规安全停。不需要租约——安全动作永远允许。"""
        self._require_connected()
        self.transport.call(API_STOP_MOVE, {})
        self.last_cmd = (0.0, 0.0, 0.0)

    def damp(self) -> None:
        """Damp（1001）：软急停。**不是**平台急停（那个≈断电，F18）。"""
        self._require_connected()
        self.transport.call(API_DAMP, {})
        self.last_cmd = (0.0, 0.0, 0.0)

    def stand_up(self) -> None:
        self._require_connected()
        self.transport.call(API_STAND_UP, {})

    def stand_down(self) -> None:
        """StandDown（1005）。采集间隙慎用：长期趴下会丢定位（手册 §5.6）。"""
        self._require_connected()
        self.transport.call(API_STAND_DOWN, {})

    def switch_gait(self, gait: int) -> None:
        self._require_connected()
        self.transport.call(API_SWITCH_GAIT, {"gait": int(gait)})

    def body_height(self, height_m: float) -> None:
        self._require_connected()
        self.transport.call(API_BODY_HEIGHT, {"height": float(height_m)})

    def speed_level(self, level: int) -> None:
        self._require_connected()
        self.transport.call(API_SPEED_LEVEL, {"level": int(level)})

    def move_to_pos(self, x: float, y: float, yaw: float) -> None:
        """MoveToPos（1036）：中层位置控制。WAM 默认不用，留作桥接实验。"""
        self._require_connected()
        self.transport.call(API_MOVE_TO_POS, {"x": float(x), "y": float(y), "yaw": float(yaw)})

    # ---------- 权限探测 ----------

    def probe_dds_authority(self) -> bool:
        """E4/G11 的程序化握手：发 StopMove 后确认能读到状态。

        返回 False 时**禁止**进入 WAM_ACTIVE。这里刻意只用零速度命令探测，
        不发试探性运动脉冲——那必须在有安全围栏的场地里由人工触发。
        """
        try:
            self._require_connected()
            self.transport.call(API_STOP_MOVE, {})
        except UnitreeError as exc:
            logger.warning("DDS 权限探测失败: %s", exc)
            return False
        return self.get_sport_state() is not None
