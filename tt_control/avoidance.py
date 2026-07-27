"""半自动视觉避障控制律：深度 → 分区启发式 → RcAxes。

与感知后端解耦：只吃一张「近度图」(nearness map)，输出杆量。
约定：nearness ∈ 约 [0,1]，**值越大表示越近/越挡路**（由 DepthAnythingBackend
按帧分位数归一化后给出，见 depth_backend.py）。这样阈值语义单一，便于单测。

新增 OrbitController：以障碍物为中心环绕飞行（POI Orbit）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tt_control.control import RcAxes


@dataclass
class AvoidParams:
    cruise_speed: int = 25      # 半自动前进杆量（低于手动默认 40）
    strafe_speed: int = 40      # 绕障横移杆量（roll，正=右移）：真正让开障碍宽度的主力
    yaw_speed: int = 35         # 航向恢复用的转向杆量（绕障不再靠 yaw）
    turn_pitch: int = 10        # 横移绕行时附带的小前进量（边平移边缓进）
    approach_pitch: int = 16    # 接近区前进量
    stop_thresh: float = 0.70   # 中区 nearness 超过 → 过近，纯横移不前进
    estop_thresh: float = 0.82  # 任一区 nearness 超过 → 遇障急停兜底(直接悬停,不前冲)
    clear_thresh: float = 0.45  # 中区 nearness 超过此值即进入「接近区」提前绕行
    side_margin: float = 0.08   # 左右 nearness 差超过此值才判为一侧更开阔
    band_top: float = 0.30      # 取深度图中部水平带 [top, bottom]（比例）
    band_bottom: float = 0.80


@dataclass
class AvoidDecision:
    axes: RcAxes
    state: str  # STOP | CRUISE | TURN_L | TURN_R | BLOCKED
    zones: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))  # 左/中/右 nearness

    def as_hud(self) -> str:
        l, m, r = self.zones
        return f"{self.state} L{l:.2f} M{m:.2f} R{r:.2f} rc{self.axes.as_tuple()}"


class AvoidanceController:
    """无状态控制律：给定一帧近度图，返回一步 RcAxes 决策。

    高度锁定（throttle=0，靠下视 VPS），不做横移（roll=0），
    绕障用 yaw + 小 pitch。三区都近则悬停，本版不做原地扫描。
    """

    def __init__(self, params: AvoidParams | None = None) -> None:
        self.p = params or AvoidParams()
        self._commit = 0  # 绕行方向滞回：-1 左(yaw-) / +1 右(yaw+) / 0 未锁定

    def reset(self) -> None:
        self._commit = 0

    def zone_nearness(self, nearness: np.ndarray) -> tuple[float, float, float]:
        """取中部水平带，按左/中/右三等分，返回各区中位近度。

        注：细障碍物检测不靠改分位数（75th 会误检墙角），
        而是靠降低 clear_thresh 让中位数也能捕捉到柱子信号。
        （2026-07-24 真机：75th→误检墙角；50th+低阈值→椅子柱子都可检测）
        """
        if nearness.ndim != 2:
            raise ValueError("nearness 必须是 2D 数组")
        h, w = nearness.shape
        y0 = int(h * self.p.band_top)
        y1 = max(y0 + 1, int(h * self.p.band_bottom))
        band = nearness[y0:y1, :]
        third = max(1, w // 3)
        left = float(np.median(band[:, :third]))
        mid = float(np.median(band[:, third : 2 * third]))
        right = float(np.median(band[:, 2 * third :]))
        return left, mid, right

    def decide(self, nearness: np.ndarray) -> AvoidDecision:
        p = self.p
        left, mid, right = self.zone_nearness(nearness)
        zones = (left, mid, right)
        # 危险度取全视场最大：障碍常只占某一区，只看中区会「斜插进侧向障碍」
        danger = max(left, mid, right)

        # 遇障急停兜底:正前方(中区)非常近 = 要正撞 → 直接悬停,不往里冲(优先级最高)
        # (侧向很近应转开而非急停,故只看 mid)
        if mid > p.estop_thresh:
            return AvoidDecision(RcAxes(), "BLOCKED", zones)

        # 整个前方视场都通畅 → 释放锁定、直行巡航
        if danger <= p.clear_thresh:
            self._commit = 0
            return AvoidDecision(RcAxes(pitch=p.cruise_speed), "CRUISE", zones)

        # 视场内有障碍：首次进入锁定「横移到更开阔一侧」的方向(对称居中则默认右)，
        # 之后保持不翻转，直到整个前方通畅。用 roll 横移而非 yaw 转向——
        # 转向只是扭头让障碍滑出视场中央却没真正让开，横移才能让机身避过障碍宽度。
        # （2026-07-24 真机验证：yaw 绕障几何上无法绕过椅子，改用 roll 横移）
        if self._commit == 0:
            self._commit = 1 if left >= right else -1
        roll = self._commit * p.strafe_speed
        state = "STRAFE_R" if self._commit > 0 else "STRAFE_L"

        # 正前方过近且两侧也近 → 被围住，悬停
        if mid > p.stop_thresh and min(left, right) > p.stop_thresh - p.side_margin:
            return AvoidDecision(RcAxes(), "BLOCKED", zones)

        # 前进量随「正前方」近度线性递减：中区远(刚触发避障)则大角度前冲+横移，
        # 形成自然弧线绕过；中区近(贴近障碍)则纯横移不前冲。
        # （2026-07-24 真机：之前用 turn_pitch 做基数，横移时几乎不前移，形成 L 形而非弧线）
        frac = (mid - p.clear_thresh) / max(1e-6, p.stop_thresh - p.clear_thresh)
        pitch = int(round(p.approach_pitch * (1.0 - min(1.0, max(0.0, frac)))))
        pitch = max(p.turn_pitch, pitch)  # 兜底：至少保持微小前移
        return AvoidDecision(RcAxes(pitch=pitch, roll=roll), state, zones)


# ── POI 环绕飞行（2026-07-24 新增）──────────────────────────


@dataclass
class OrbitParams:
    """环绕飞行参数。"""
    # 环绕方向：-1 逆时针 / +1 顺时针
    direction: int = 1

    # 目标 nearness：中区近度保持在此值附近
    # （2026-07-27: 0.55≈1m → 按 1/d 比例调到 0.69≈0.8m）
    target_nearness: float = 0.69

    # 距离控制死区：nearness 在此范围内不做前后调整
    distance_deadband: float = 0.05

    # 环绕横移杆量（只在椅子居中时才横移，椅子偏了先停住让yaw拉回中央）
    # （2026-07-25: 降到10，视觉伺服优先——慢慢横移，偏了就停）
    orbit_roll: int = 10

    # 椅子居中偏航增益：chair_pos ∈ [-1,1] → yaw 杆量
    # at pos=0.10 → yaw=20, at pos=0.25 → yaw=50(max)
    # （2026-07-25: 重新设计，直接乘增益不再乘max_yaw，避免增益语义混淆）
    yaw_centering_gain: float = 200.0

    # 横移门槛：|chair_pos| < 此值才允许横移，否则停横移全力yaw居中
    centering_deadband: float = 0.12

    # 距离控制增益：(target - mid) → pitch 杆量比例
    # （2026-07-25: 1.2→2.5，距离修正更强，靠近目标更快）
    pitch_distance_gain: float = 2.5

    # 杆量限制
    max_yaw: int = 50     # 椅子偏时全力拉回
    max_pitch: int = 35   # 更多前进余地
    min_yaw: int = 0      # 不做最小钳位，小误差不转
    min_pitch: int = 5

    # 接近检测：中区或侧区超过此值认为太近（危险）
    # 环绕横移时侧向才是主碰撞面，故左右区同样刹停（2026-07-26）
    danger_thresh: float = 0.78

    # 失锁：中区低于此值认为椅子丢掉了
    lost_thresh: float = 0.10


@dataclass
class OrbitDecision:
    axes: RcAxes
    state: str  # ORBIT | LOST | DANGER
    zones: tuple[float, float, float]
    yaw_correction: int = 0
    pitch_correction: int = 0
    chair_pos: Optional[float] = None
    orbit_phase: str = ""

    def as_hud(self) -> str:
        l, m, r = self.zones
        pos_str = f"pos{self.chair_pos:+.2f}" if self.chair_pos is not None else "pos?"
        return (
            f"{self.state} {pos_str} {self.orbit_phase} "
            f"L{l:.2f} M{m:.2f} R{r:.2f} "
            f"yaw{self.yaw_correction:+d} pit{self.pitch_correction:+d} "
            f"rc{self.axes.as_tuple()}"
        )


class OrbitController:
    """POI 环绕控制律：保持椅子在画面中央，以固定距离绕圈。

    控制策略：
    1. 椅子偏左 → yaw 左（-），偏右 → yaw 右（+），把它拉回中央
    2. 持续向环绕方向横移（roll），形成圆形轨迹
    3. 太近（中区高）→ pitch 后退；太远（中区低）→ pitch 前进
    """

    def __init__(self, params: OrbitParams | None = None) -> None:
        self.p = params or OrbitParams()

    def reset(self) -> None:
        pass

    def decide(self, nearness: np.ndarray) -> OrbitDecision:
        """每帧调用，返回环绕杆量。视觉伺服优先策略：

        1. 椅子居中 (|pos| < deadband) → 慢速横移 + 微调yaw
        2. 椅子偏了 → 停横移，全力yaw拉回中央
        3. 椅子丢了 / 危险 → 停一切，悬停等待
        """
        left, mid, right = self._zone_nearness(nearness)

        # ── 1) 椅子水平位置（重心法）───────────────────
        chair_pos = self._chair_horizontal(nearness)  # [-1,1]，0=正中

        # ── 2) yaw 居中（视觉伺服核心）─────────────────
        # chair_pos > 0 = 椅子偏右 → yaw 正(右转)追上
        if chair_pos is not None:
            yaw_raw = chair_pos * self.p.yaw_centering_gain
            yaw = int(round(max(-self.p.max_yaw, min(self.p.max_yaw, yaw_raw))))
        else:
            yaw = 0

        # ── 3) 环绕横移（只居中时才走）─────────────────
        # 椅子偏出 deadband 就停横移，让 yaw 先把椅子拉回来
        if chair_pos is not None and abs(chair_pos) < self.p.centering_deadband:
            roll = self.p.direction * self.p.orbit_roll
            orbit_phase = "strafe"
        elif chair_pos is not None:
            roll = 0
            orbit_phase = "center"  # 正在居中，暂停横移
        else:
            roll = 0
            orbit_phase = "search"  # 找不到椅子

        # ── 4) 距离保持（按椅子列近度，勿用中区——椅子偏了时 mid 是背景）──
        if chair_pos is None:
            pitch = 0
            target_near = mid
        else:
            target_near = self._nearness_at_pos(nearness, chair_pos)
            dist_err = self.p.target_nearness - target_near  # 正=太远需前进
            if abs(dist_err) <= self.p.distance_deadband:
                pitch = 0
            else:
                pitch_raw = int(round(dist_err * self.p.pitch_distance_gain * 20))
                pitch = max(-self.p.max_pitch, min(self.p.max_pitch, pitch_raw))
                if abs(pitch) < self.p.min_pitch and abs(pitch_raw) >= 1:
                    pitch = self.p.min_pitch if pitch_raw > 0 else -self.p.min_pitch

        # ── 5) 状态判定 ──────────────────────────────
        # 中区或侧区过近都 DANGER（横移时侧向是主碰撞面）
        state = "ORBIT"
        if max(left, mid, right) > self.p.danger_thresh:
            state = "DANGER"
        elif chair_pos is None:
            # 无可靠目标即 LOST（不要求 mid < lost_thresh，避免墙面 mid 中等时永不失锁）
            state = "LOST"

        # LOST / DANGER：强制零杆（宽限期内也不得前进）
        if state in ("LOST", "DANGER"):
            pitch = roll = yaw = 0

        return OrbitDecision(
            axes=RcAxes(pitch=pitch, roll=roll, yaw=yaw),
            state=state,
            zones=(left, mid, right),
            yaw_correction=yaw,
            pitch_correction=pitch,
            chair_pos=chair_pos,
            orbit_phase=orbit_phase,
        )

    @staticmethod
    def _chair_horizontal(nearness: np.ndarray) -> Optional[float]:
        """重心法计算椅子在画面中的水平位置。

        返回归一化值 [-1, 1]：0=正中，-1=最左，1=最右。
        返回 None 表示视野中找不到椅子。
        """
        h, w = nearness.shape
        y0 = int(h * 0.30)
        y1 = max(y0 + 1, int(h * 0.80))
        band = nearness[y0:y1, :]
        # 每列的中位近度
        col_nearness = np.median(band, axis=0)
        # 过滤背景噪声：近度低于 0.10 的列不参与
        weights = np.maximum(col_nearness - 0.10, 0.0)
        total = float(weights.sum())
        if total < 1e-6:
            return None
        # 拒绝均匀场（墙/地板当假椅子）：需有明显峰值列
        peak = float(weights.max())
        mean = total / float(weights.size)
        if peak < 0.12 or peak < mean * 2.0:
            return None
        center = float(np.average(np.arange(w, dtype=np.float64), weights=weights))
        return (center / max(1, w - 1)) * 2.0 - 1.0

    @staticmethod
    def _nearness_at_pos(nearness: np.ndarray, chair_pos: float) -> float:
        """取椅子水平位置附近列的中位近度，供距离保持使用。"""
        h, w = nearness.shape
        y0 = int(h * 0.30)
        y1 = max(y0 + 1, int(h * 0.80))
        col = int(round((chair_pos + 1.0) * 0.5 * (w - 1)))
        col = max(0, min(w - 1, col))
        half = max(1, w // 20)
        c0 = max(0, col - half)
        c1 = min(w, col + half + 1)
        return float(np.median(nearness[y0:y1, c0:c1]))

    @staticmethod
    def _zone_nearness(nearness: np.ndarray) -> tuple[float, float, float]:
        """与 AvoidanceController 一致的左/中/右分区。"""
        if nearness.ndim != 2:
            raise ValueError("nearness 必须是 2D 数组")
        h, w = nearness.shape
        y0 = int(h * 0.30)
        y1 = max(y0 + 1, int(h * 0.80))
        band = nearness[y0:y1, :]
        third = max(1, w // 3)
        left = float(np.median(band[:, :third]))
        mid = float(np.median(band[:, third : 2 * third]))
        right = float(np.median(band[:, 2 * third :]))
        return left, mid, right
