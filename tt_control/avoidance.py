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

    # 目标 nearness（场景相对量，非米制）：椅子列近度保持在此值附近
    target_nearness: float = 0.55

    # 距离控制死区：nearness 在此范围内不做前后调整
    distance_deadband: float = 0.05

    # 环绕横移杆量（随椅子偏离比例 taper，见 centering_deadband）
    orbit_roll: int = 10

    # 椅子居中偏航增益：chair_pos ∈ [-1,1] → yaw 杆量
    # （2026-07-27 真机：200 过猛，偏 0.25 就打满 → 来回摇头；降到 80）
    yaw_centering_gain: float = 80.0

    # roll taper 半宽：|chair_pos| 从 0 到此值，roll 从满杆线性降到 0。
    # （2026-07-27 真机：0.12 太窄 → roll 在 0↔满之间开关振荡；加宽到 0.25）
    centering_deadband: float = 0.25

    # yaw 前馈：横移环绕时椅子必然向横移反方向漂移，前馈一个与 roll
    # 成比例的反向 yaw，不必等椅子偏出去 P 修正才转头。
    # ff_yaw = -roll * yaw_ff_ratio（roll 含方向符号）
    yaw_ff_ratio: float = 0.6

    # 距离控制增益：(target - chair_near) → pitch 杆量比例
    # （2026-07-27 真机：2.5 太弱，tn 只到 0.5~0.6 从未达到目标 0.69 → 半径偏大）
    pitch_distance_gain: float = 4.0

    # 杆量限制（max_yaw=50 对慢速环绕过大，是摇头主因之一）
    max_yaw: int = 25
    max_pitch: int = 35
    min_yaw: int = 0
    min_pitch: int = 5

    # ── 抗摇头（2026-07-27）────────────────────────────
    # 椅子位置/距离 EMA 平滑：新帧权重（1.0=不平滑）。深度约 5Hz、
    # 控制约 24Hz，0.4 的时间常数只有 ~80ms，滤不掉深度帧级跳变 → 0.25。
    pos_smooth_alpha: float = 0.25
    # yaw 死区：|chair_pos| 低于此值不打 yaw P 修正（前馈仍生效）
    yaw_deadband: float = 0.05

    # 接近检测：中区或侧区超过此值认为太近（危险；亦为相对量）
    danger_thresh: float = 0.78

    # 历史字段（当前以检测失败判 LOST；保留以免旧配置炸）
    lost_thresh: float = 0.10

    # ── 目标获取 / 跟踪滞回（2026-07-27）──────────────────
    # 获取：全图峰均比（真机椅背场景约 1.9，故 1.6）；跟踪：ROI + 更松
    acquire_peak_ratio: float = 1.6
    track_peak_ratio: float = 1.35
    min_peak: float = 0.12
    acquire_frames: int = 1
    track_roi_half: float = 0.35   # 归一化坐标半宽
    # （2026-07-27 真机：0.45 放过了 ±0.3 级检测跳变 → yaw 甩头 + tn 尖刺
    #  → orbit_danger 误 abort；合法帧间移动约 0.1 级，收紧到 0.30）
    max_pos_jump: float = 0.30


@dataclass
class OrbitDecision:
    axes: RcAxes
    state: str  # ORBIT | LOST | DANGER | ACQUIRE
    zones: tuple[float, float, float]
    yaw_correction: int = 0
    pitch_correction: int = 0
    chair_pos: Optional[float] = None
    orbit_phase: str = ""
    target_near: float = 0.0
    reject_reason: str = ""

    def as_hud(self) -> str:
        l, m, r = self.zones
        pos_str = f"pos{self.chair_pos:+.2f}" if self.chair_pos is not None else "pos?"
        rej = f" {self.reject_reason}" if self.reject_reason else ""
        return (
            f"{self.state} {pos_str} {self.orbit_phase} "
            f"L{l:.2f} M{m:.2f} R{r:.2f} tn{self.target_near:.2f} "
            f"yaw{self.yaw_correction:+d} pit{self.pitch_correction:+d} "
            f"rc{self.axes.as_tuple()}{rej}"
        )


class OrbitController:
    """POI 环绕控制律：保持目标在画面中央，以相对近度绕圈。

    控制策略：
    1. 严格获取 → 锁定后 ROI 跟踪（避免宽目标被峰均比误杀）
    2. 目标居中时慢速横移；偏了先 yaw 拉回
    3. 用椅子列近度做前后距离保持（非米制）
    """

    def __init__(self, params: OrbitParams | None = None) -> None:
        self.p = params or OrbitParams()
        self._tracked = False
        self._last_pos: Optional[float] = None
        self._acquire_count = 0
        self._pos_filt: Optional[float] = None  # EMA 平滑后的椅子位置
        self._tn_filt: Optional[float] = None   # EMA 平滑后的椅子列近度

    def reset(self) -> None:
        self._tracked = False
        self._last_pos = None
        self._acquire_count = 0
        self._pos_filt = None
        self._tn_filt = None

    def decide(self, nearness: np.ndarray) -> OrbitDecision:
        """每帧调用，返回环绕杆量。"""
        left, mid, right = self._zone_nearness(nearness)

        chair_pos, reject_reason = self._lock_target(nearness)

        # 位置 EMA 平滑：检测重心逐帧抖动（±0.05 级），直接进 P 控制
        # 会被增益放大成来回打杆（2026-07-27 真机：靠近椅子后摇头）。
        if chair_pos is not None:
            if self._pos_filt is None:
                self._pos_filt = chair_pos
            else:
                a = self.p.pos_smooth_alpha
                self._pos_filt = a * chair_pos + (1.0 - a) * self._pos_filt
            chair_pos = self._pos_filt

        # 比例混合：椅子越正，横移越多；椅子越偏，横移越少、yaw 占比越大。
        # 避免二元开关造成的「横移推开→yaw拉回→再横移推开」振荡。
        # （2026-07-27 真机：binary deadband → 椅子左右摇摆出视野 → LOST）
        if chair_pos is not None:
            abs_pos = abs(chair_pos)
            if abs_pos < self.p.centering_deadband:
                # 线性 taper：中心 full roll，taper 边缘 roll=0
                roll_frac = 1.0 - (abs_pos / self.p.centering_deadband)
                roll = int(round(self.p.direction * self.p.orbit_roll * roll_frac))
                orbit_phase = "orbit"
            else:
                roll = 0
                orbit_phase = "center"
        else:
            roll = 0
            orbit_phase = "search"

        # yaw = 环绕前馈 + 居中 P 修正。
        # 前馈：横移必然让椅子向反方向漂移，与 roll 成比例地持续轻转，
        # 不必等椅子偏出去 P 才拉回（P 死区内前馈仍生效）。
        if chair_pos is not None and abs(chair_pos) >= self.p.yaw_deadband:
            yaw_corr = int(round(chair_pos * self.p.yaw_centering_gain))
        else:
            yaw_corr = 0
        ff_yaw = -int(round(roll * self.p.yaw_ff_ratio))
        yaw = int(max(-self.p.max_yaw, min(self.p.max_yaw, ff_yaw + yaw_corr)))

        if chair_pos is None:
            pitch = 0
            target_near = mid
        else:
            # 距离也做 EMA：tn 单帧尖刺（0.5→0.89）曾驱动 pitch ±10 前后猛冲
            raw_tn = self._nearness_at_pos(nearness, chair_pos)
            if self._tn_filt is None:
                self._tn_filt = raw_tn
            else:
                a = self.p.pos_smooth_alpha
                self._tn_filt = a * raw_tn + (1.0 - a) * self._tn_filt
            target_near = self._tn_filt
            dist_err = self.p.target_nearness - target_near
            if abs(dist_err) <= self.p.distance_deadband:
                pitch = 0
            else:
                pitch_raw = int(round(dist_err * self.p.pitch_distance_gain * 20))
                pitch = max(-self.p.max_pitch, min(self.p.max_pitch, pitch_raw))
                if abs(pitch) < self.p.min_pitch and abs(pitch_raw) >= 1:
                    pitch = self.p.min_pitch if pitch_raw > 0 else -self.p.min_pitch
            # 椅子明显偏离中央时距离读数不可靠，禁止向前冲（后退保命仍允许）
            if orbit_phase == "center" and pitch > 0:
                pitch = 0

        state = "ORBIT"
        if max(left, mid, right) > self.p.danger_thresh:
            state = "DANGER"
        elif chair_pos is None:
            # 获取中 ≠ 失锁：避免 acquire 帧启动 LOST episode 超时
            state = "ACQUIRE" if reject_reason.startswith("acquiring") else "LOST"

        if state in ("LOST", "DANGER", "ACQUIRE"):
            pitch = roll = yaw = 0

        return OrbitDecision(
            axes=RcAxes(pitch=pitch, roll=roll, yaw=yaw),
            state=state,
            zones=(left, mid, right),
            yaw_correction=yaw_corr,
            pitch_correction=pitch,
            chair_pos=chair_pos,
            orbit_phase=orbit_phase,
            target_near=float(target_near),
            reject_reason=reject_reason,
        )

    def _lock_target(self, nearness: np.ndarray) -> tuple[Optional[float], str]:
        """获取/跟踪滞回：返回 (chair_pos, reject_reason)。"""
        raw_pos, reason, peak, mean = self._detect_chair(nearness)
        if raw_pos is None:
            self._acquire_count = 0
            return None, reason

        if self._tracked:
            self._acquire_count = 0
            self._last_pos = raw_pos
            return raw_pos, ""

        self._acquire_count += 1
        if self._acquire_count >= self.p.acquire_frames:
            self._tracked = True
            self._last_pos = raw_pos
            return raw_pos, ""
        return None, f"acquiring{self._acquire_count}/{self.p.acquire_frames}"

    def _detect_chair(
        self, nearness: np.ndarray
    ) -> tuple[Optional[float], str, float, float]:
        """检测水平目标重心。跟踪时限制在上一位置 ROI。"""
        h, w = nearness.shape
        y0 = int(h * 0.30)
        y1 = max(y0 + 1, int(h * 0.80))
        col_nearness = np.median(nearness[y0:y1, :], axis=0)
        weights = np.maximum(col_nearness - 0.10, 0.0)

        if self._tracked and self._last_pos is not None:
            c0, c1 = self._roi_cols(w, self._last_pos, self.p.track_roi_half)
            roi_w = weights.copy()
            if c0 > 0:
                roi_w[:c0] = 0.0
            if c1 < w:
                roi_w[c1:] = 0.0
            weights = roi_w
            ratio_need = self.p.track_peak_ratio
        else:
            ratio_need = self.p.acquire_peak_ratio

        total = float(weights.sum())
        if total < 1e-6:
            return None, ("roi_empty" if self._tracked else "no_weight"), 0.0, 0.0

        peak = float(weights.max())
        # mean 用全宽（含零）更稳：跟踪 ROI 外已置零
        mean_all = total / float(weights.size)
        if peak < self.p.min_peak:
            return None, "peak_low", peak, mean_all
        if peak < mean_all * ratio_need:
            return None, f"flat:{peak:.2f}/{mean_all:.2f}", peak, mean_all

        center_idx = float(np.average(np.arange(w, dtype=np.float64), weights=weights))
        pos = (center_idx / max(1, w - 1)) * 2.0 - 1.0

        if self._tracked and self._last_pos is not None:
            if abs(pos - self._last_pos) > self.p.max_pos_jump:
                return None, "jump", peak, mean_all
        return pos, "", peak, mean_all

    @staticmethod
    def _roi_cols(width: int, pos: float, half: float) -> tuple[int, int]:
        center = (pos + 1.0) * 0.5 * (width - 1)
        span = half * 0.5 * (width - 1)  # half in [-1,1] space → column span
        c0 = max(0, int(np.floor(center - span)))
        c1 = min(width, int(np.ceil(center + span)) + 1)
        return c0, c1

    @staticmethod
    def _chair_horizontal(nearness: np.ndarray) -> Optional[float]:
        """兼容旧测试：无状态严格全图获取（等同未跟踪时的检测）。"""
        c = OrbitController()
        pos, _, _, _ = c._detect_chair(nearness)
        return pos

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
