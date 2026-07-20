"""半自动视觉避障控制律：深度 → 分区启发式 → RcAxes。

与感知后端解耦：只吃一张「近度图」(nearness map)，输出杆量。
约定：nearness ∈ 约 [0,1]，**值越大表示越近/越挡路**（由 DepthAnythingBackend
按帧分位数归一化后给出，见 depth_backend.py）。这样阈值语义单一，便于单测。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tt_control.control import RcAxes


@dataclass
class AvoidParams:
    cruise_speed: int = 25      # 半自动前进杆量（低于手动默认 40）
    yaw_speed: int = 35         # 转向杆量（离线验证：20 权限太弱，来不及绕开）
    turn_pitch: int = 10        # 微调转向时附带的小前进量
    approach_pitch: int = 16    # 接近区(边转边前进)的前进量，保证绕弧推进而非原地转
    stop_thresh: float = 0.70   # 中区 nearness 超过 → 过近，零前进原地转/悬停
    clear_thresh: float = 0.45  # 中区 nearness 超过此值即进入「接近区」提前转向
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
        """取中部水平带，按左/中/右三等分，返回各区中位近度。"""
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

        # 整个前方视场都通畅 → 释放锁定、直行巡航
        if danger <= p.clear_thresh:
            self._commit = 0
            return AvoidDecision(RcAxes(pitch=p.cruise_speed), "CRUISE", zones)

        # 视场内有障碍：首次进入锁定「远离更挡一侧」的转向(对称居中则默认右)，
        # 之后保持不翻转，直到整个前方通畅
        if self._commit == 0:
            self._commit = 1 if left >= right else -1
        yaw = self._commit * p.yaw_speed
        state = "TURN_L" if self._commit < 0 else "TURN_R"

        # 正前方过近且两侧也近 → 被围住，悬停
        if mid > p.stop_thresh and min(left, right) > p.stop_thresh - p.side_margin:
            return AvoidDecision(RcAxes(), "BLOCKED", zones)

        # 前进量随「正前方」近度线性递减：中区远则全速绕弧，中区近则刹到近零原地转
        frac = (mid - p.clear_thresh) / max(1e-6, p.stop_thresh - p.clear_thresh)
        pitch = int(round(p.approach_pitch * (1.0 - min(1.0, max(0.0, frac)))))
        return AvoidDecision(RcAxes(pitch=pitch, yaw=yaw), state, zones)
