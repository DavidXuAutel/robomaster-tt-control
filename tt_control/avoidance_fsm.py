"""避障状态机：在 AvoidanceController 之上管理绕飞全流程。

PREFLIGHT → HOVER → APPROACH → [AVOID_TURN → PASS_OBSTACLE
    → RECOVER_HEADING → POST_CLEAR → 回到 APPROACH] 或 ORBIT

全局安全：视频丢失/深度过期/低电量/遥测异常/超时/人工接管 → ABORT
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from tt_control.avoidance import AvoidanceController, AvoidDecision, AvoidParams, OrbitController, OrbitDecision, OrbitParams
from tt_control.control import RcAxes


class AvoidState(Enum):
    PREFLIGHT = "PREFLIGHT"        # 未起飞
    HOVER = "HOVER"                # 起飞后悬停，等待 ARM
    APPROACH = "APPROACH"          # 前进接近
    AVOID_TURN = "AVOID_TURN"      # 检测到障碍，偏航绕行
    PASS_OBSTACLE = "PASS_OBSTACLE"  # 绕过障碍，小前进量确认
    RECOVER_HEADING = "RECOVER_HEADING"  # 恢复初始航向
    HOVER_COMPLETE = "HOVER_COMPLETE"  # 绕飞完成，悬停等待
    POST_CLEAR = "POST_CLEAR"        # 越障后额外前进一段
    ORBIT = "ORBIT"                  # POI 环绕：以障碍物为中心绕圈飞


@dataclass
class FsmParams:
    """状态机可调参数。"""
    # 各状态超时（秒），超时触发 ABORT
    max_approach_s: float = 10.0
    max_turn_s: float = 8.0
    max_pass_s: float = 5.0
    max_recover_s: float = 5.0
    max_auto_engaged_s: float = 30.0

    # 各状态最低保持时间（秒），避免中区一清空就过早切走（横移不够远）
    min_turn_s: float = 5.0       # 横移绕行至少持续 N 秒，确保身体让开障碍宽度
    min_pass_s: float = 2.0       # 越障确认至少持续 N 秒，再补一段横移

    # 越障后额外前进（0 禁用）
    post_clear_s: float = 0.0       # 前进时长（秒）
    post_clear_pitch: int = 20      # 前进杆量

    # 越障确认：中区近度低于此阈值且连续 N 帧 → 判为越障成功
    pass_clear_thresh: float = 0.35
    pass_consecutive_frames: int = 5

    # 航向恢复：偏航误差小于此值 → 完成
    heading_tolerance_deg: float = 15.0
    heading_yaw_gain: float = 0.5   # yaw 误差 → 杆量的比例系数

    # 死胡同检测：连续 N 帧 danger 上升 → 判定为死胡同
    dead_end_frames: int = 10

    # 环绕模式（POI Orbit）：开启后 APPROACH 检测到障碍 → 切入 ORBIT 而非 AVOID_TURN
    orbit_mode: bool = False
    orbit_enter_nearness: float = 0.30  # 中区超过此值 → 进入 ORBIT 阶段
    orbit_lost_timeout_s: float = 5.0   # 失锁 episode 超过 N 秒 → 悬停（闪断不复位）
    orbit_relock_frames: int = 5        # 失锁后需连续 N 帧有效锁定才清 episode

    # 全局安全
    min_battery_pct: int = 30
    max_height_cm: int = 500
    depth_stale_s: float = 2.0


@dataclass
class FsmDecision:
    """状态机每帧输出。"""
    axes: RcAxes
    state: AvoidState
    sub_state: str = ""        # 底层控制器决策 (CRUISE/TURN_L/...)
    abort_reason: str = ""     # 非空 → 应立即悬停并解除 AUTO
    heading_error_deg: float = 0.0
    state_elapsed_s: float = 0.0

    def as_hud(self) -> str:
        parts = [self.state.value]
        if self.sub_state:
            parts.append(self.sub_state)
        if self.abort_reason:
            parts.append(f"ABORT:{self.abort_reason}")
        return " | ".join(parts)


class AvoidanceFSM:
    """避障状态机。

    用法（在控制循环中每帧调用）:
        result = fsm.step(nearness, telemetry, auto_on)
        if result.abort_reason:
            # 悬停 + 解除 AUTO
        else:
            tello.rc(**result.axes.as_tuple())
    """

    def __init__(
        self,
        controller: Optional[AvoidanceController] = None,
        params: Optional[FsmParams] = None,
        orbit_ctrl: Optional[OrbitController] = None,
    ) -> None:
        self._ctrl = controller or AvoidanceController()
        self._orbit = orbit_ctrl or OrbitController()
        self.p = params or FsmParams()

        self._state = AvoidState.PREFLIGHT
        self._state_entered: float = 0.0
        # None = 未挂载；不可用 0.0 作哨兵（仿真/单测 now=0 会每帧误触发「首次挂载」清掉 orbit）
        self._auto_engaged: Optional[float] = None
        self._initial_yaw: Optional[float] = None
        self._pass_clear_count: int = 0
        self._danger_history: list[float] = []
        self._last_depth_ts: float = 0.0
        self._abort_reason: str = ""
        self._orbit_lost_since: Optional[float] = None  # ORBIT 失锁 episode 起点；None=未失锁
        self._orbit_relock_count: int = 0  # 失锁后连续有效帧计数
        self._orbit_active: bool = False   # 环绕滞回：一旦进入即保持，避免 mid 波动导致进出振荡

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def state(self) -> AvoidState:
        return self._state

    @property
    def abort_reason(self) -> str:
        return self._abort_reason

    @property
    def initial_yaw(self) -> Optional[float]:
        return self._initial_yaw

    # ── 每帧步进 ──────────────────────────────────────────────

    def step(
        self,
        nearness: Optional["np.ndarray"],
        telemetry: dict[str, str],
        auto_on: bool,
        now: Optional[float] = None,
    ) -> FsmDecision:
        """每帧调用：输入近度图 + 遥测 + AUTO 开关，返回决策。"""
        if now is None:
            now = time.time()

        # 0) 全局安全检查
        abort = self._check_safety(telemetry, now)
        if abort:
            self._abort_reason = abort
            self._state = AvoidState.HOVER
            return FsmDecision(
                axes=RcAxes(),
                state=self._state,
                abort_reason=abort,
                state_elapsed_s=now - self._state_entered,
            )

        # 1) AUTO 未开启 → 悬停，复位
        if not auto_on:
            self._state = AvoidState.HOVER
            self._auto_engaged = None
            self._initial_yaw = None
            self._pass_clear_count = 0
            self._orbit_active = False
            self._orbit_lost_since = None
            self._orbit_relock_count = 0
            self._orbit.reset()
            return FsmDecision(
                axes=RcAxes(),
                state=self._state,
                state_elapsed_s=now - self._state_entered,
            )

        # 2) AUTO 首次挂载 → 记录初始 yaw，进入 APPROACH
        if self._auto_engaged is None:
            self._auto_engaged = now
            self._initial_yaw = self._read_yaw(telemetry)
            self._state = AvoidState.APPROACH
            self._state_entered = now
            self._pass_clear_count = 0
            self._danger_history.clear()
            self._abort_reason = ""
            self._orbit_active = False
            self._orbit_lost_since = None
            self._orbit_relock_count = 0
            self._orbit.reset()

        # 3) 检查 AUTO 全局超时
        if now - self._auto_engaged > self.p.max_auto_engaged_s:
            self._abort_reason = f"auto engaged >{self.p.max_auto_engaged_s:.0f}s"
            self._state = AvoidState.HOVER
            return FsmDecision(
                axes=RcAxes(),
                state=self._state,
                abort_reason=self._abort_reason,
                state_elapsed_s=now - self._state_entered,
            )

        # 4) 检查深度过期
        if nearness is not None:
            self._last_depth_ts = now
        if self._last_depth_ts <= 0:
            # 从未收到深度：从 AUTO 挂载起给宽限期（等远程推理首帧）
            if now - self._auto_engaged > self.p.depth_stale_s:
                self._abort_reason = f"no depth after {self.p.depth_stale_s:.0f}s"
                self._state = AvoidState.HOVER
                return FsmDecision(
                    axes=RcAxes(), state=self._state,
                    abort_reason=self._abort_reason,
                    state_elapsed_s=now - self._state_entered,
                )
        elif now - self._last_depth_ts > self.p.depth_stale_s:
            self._abort_reason = f"depth stale >{self.p.depth_stale_s:.1f}s"
            self._state = AvoidState.HOVER
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                abort_reason=self._abort_reason,
                state_elapsed_s=now - self._state_entered,
            )

        # 5) 状态机
        elapsed = now - self._state_entered

        if self._state == AvoidState.APPROACH:
            return self._step_approach(nearness, now, elapsed)
        elif self._state == AvoidState.AVOID_TURN:
            return self._step_avoid_turn(nearness, now, elapsed)
        elif self._state == AvoidState.PASS_OBSTACLE:
            return self._step_pass_obstacle(nearness, now, elapsed, telemetry)
        elif self._state == AvoidState.RECOVER_HEADING:
            return self._step_recover_heading(now, elapsed, telemetry)
        elif self._state == AvoidState.POST_CLEAR:
            return self._step_post_clear(nearness, now, elapsed)
        elif self._state == AvoidState.ORBIT:
            return self._step_orbit(nearness, now, elapsed)
        else:
            # HOVER / HOVER_COMPLETE / PREFLIGHT
            self._ctrl.reset()
            return FsmDecision(
                axes=RcAxes(),
                state=self._state,
                state_elapsed_s=elapsed,
            )

    # ── 各状态实现 ────────────────────────────────────────────

    def _step_approach(
        self, nearness: "np.ndarray", now: float, elapsed: float
    ) -> FsmDecision:
        # 深度未就绪 → 悬停等待首帧
        if nearness is None:
            return FsmDecision(
                axes=RcAxes(), state=self._state, sub_state="wait_depth",
                state_elapsed_s=elapsed,
            )
        # 状态超时（环绕模式下跳过：ORBIT 可持续数分钟，由自身 LOST/DANGER 保护）
        if not (self.p.orbit_mode and self._orbit_active) and elapsed > self.p.max_approach_s:
            self._abort_reason = f"approach >{self.p.max_approach_s:.0f}s"
            self._state = AvoidState.HOVER
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                abort_reason=self._abort_reason, state_elapsed_s=elapsed,
            )

        dec = self._ctrl.decide(nearness)
        zones = self._ctrl.zone_nearness(nearness)
        mid = zones[1]

        # 死胡同检测（环绕模式下跳过：靠近椅子是正常行为，近度必然上升）
        if not self.p.orbit_mode:
            danger = max(zones)
            self._danger_history.append(danger)
            if len(self._danger_history) > self.p.dead_end_frames:
                self._danger_history.pop(0)
            if (
                len(self._danger_history) >= self.p.dead_end_frames
                and all(
                    self._danger_history[i] < self._danger_history[i + 1]
                    for i in range(len(self._danger_history) - 1)
                )
            ):
                self._abort_reason = "dead end (danger rising)"
                self._state = AvoidState.HOVER
                self._ctrl.reset()
                return FsmDecision(
                    axes=RcAxes(), state=self._state,
                    abort_reason=self._abort_reason, state_elapsed_s=elapsed,
                )

        # ── 环绕模式（POI Orbit）────────────────────────
        # 一旦激活环绕，优先于 CRUISE/AVOID 检查，防止 mid 小波动
        # 导致每帧在「环绕后退→前进→环绕」间振荡（2026-07-25 真机定位）。
        if self.p.orbit_mode and self._orbit_active:
            return self._run_orbit(nearness, now, elapsed)

        # 前方通畅 → 保持前进
        if dec.state == "CRUISE":
            return FsmDecision(
                axes=dec.axes, state=self._state, sub_state=dec.state,
                state_elapsed_s=elapsed,
            )

        # 检测到障碍 → 绕行 or 环绕
        if self.p.orbit_mode:
            if mid > self.p.orbit_enter_nearness:
                self._orbit_active = True
                return self._run_orbit(nearness, now, elapsed)
            # 还不够近 → 继续直飞靠近；侧障按 orbit danger 门限刹停
            # （勿用 estop 0.82，否则与 orbit 0.78 之间存在侧刮带）
            if dec.state == "BLOCKED" or max(zones) > self._orbit.p.danger_thresh:
                return FsmDecision(
                    axes=RcAxes(),
                    state=self._state,
                    sub_state=f"approach_hold danger={max(zones):.2f}",
                    state_elapsed_s=elapsed,
                )
            return FsmDecision(
                axes=RcAxes(pitch=self._ctrl.p.cruise_speed),
                state=self._state,
                sub_state=f"approach_orbit M{mid:.2f}",
                state_elapsed_s=elapsed,
            )
        self._state = AvoidState.AVOID_TURN
        self._state_entered = now
        self._pass_clear_count = 0
        return FsmDecision(
            axes=dec.axes, state=self._state, sub_state=dec.state,
            state_elapsed_s=0.0,
        )

    def _run_orbit(
        self, nearness: "np.ndarray", now: float, elapsed: float
    ) -> FsmDecision:
        """环绕控制（独立方法，由 _step_approach 在 _orbit_active=True 时调用）。

        一旦进入环绕就持续运行，不因 mid 小波动退出。
        退出条件：DANGER（太近）或 LOST episode 超时（闪断不复位计时）。
        """
        orbit_dec = self._orbit.decide(nearness)
        return self._finish_orbit_decision(orbit_dec, now, elapsed)

    def _finish_orbit_decision(
        self, orbit_dec: OrbitDecision, now: float, elapsed: float
    ) -> FsmDecision:
        """共享：DANGER 立即 abort；LOST episode 超时；重获需连续锁定。"""
        if orbit_dec.state == "DANGER":
            self._orbit_active = False
            self._orbit_lost_since = None
            self._orbit_relock_count = 0
            self._abort_reason = "orbit_danger"
            self._state = AvoidState.HOVER
            self._ctrl.reset()
            self._orbit.reset()
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                abort_reason=self._abort_reason,
                sub_state="orbit_danger", state_elapsed_s=0.0,
            )

        if orbit_dec.state == "LOST":
            if self._orbit_lost_since is None:
                self._orbit_lost_since = now
            self._orbit_relock_count = 0
            if now - self._orbit_lost_since > self.p.orbit_lost_timeout_s:
                self._orbit_active = False
                self._orbit_lost_since = None
                self._orbit_relock_count = 0
                self._abort_reason = "orbit_lost"
                self._state = AvoidState.HOVER
                self._orbit.reset()
                return FsmDecision(
                    axes=RcAxes(), state=self._state,
                    abort_reason=self._abort_reason,
                    sub_state="orbit_lost", state_elapsed_s=0.0,
                )
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                sub_state=orbit_dec.as_hud(), state_elapsed_s=elapsed,
            )

        # 初始获取中：零杆，但不启动/不计 LOST episode
        if orbit_dec.state == "ACQUIRE":
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                sub_state=orbit_dec.as_hud(), state_elapsed_s=elapsed,
            )

        # 有效锁定：若处在失锁 episode，需连续 N 帧才清计时；期间零杆
        if self._orbit_lost_since is not None:
            self._orbit_relock_count += 1
            if self._orbit_relock_count >= self.p.orbit_relock_frames:
                self._orbit_lost_since = None
                self._orbit_relock_count = 0
            else:
                if now - self._orbit_lost_since > self.p.orbit_lost_timeout_s:
                    self._orbit_active = False
                    self._orbit_lost_since = None
                    self._orbit_relock_count = 0
                    self._abort_reason = "orbit_lost"
                    self._state = AvoidState.HOVER
                    self._orbit.reset()
                    return FsmDecision(
                        axes=RcAxes(), state=self._state,
                        abort_reason=self._abort_reason,
                        sub_state="orbit_lost", state_elapsed_s=0.0,
                    )
                return FsmDecision(
                    axes=RcAxes(), state=self._state,
                    sub_state=f"reacq {self._orbit_relock_count}/{self.p.orbit_relock_frames} | {orbit_dec.as_hud()}",
                    state_elapsed_s=elapsed,
                )

        return FsmDecision(
            axes=orbit_dec.axes, state=self._state,
            sub_state=orbit_dec.as_hud(), state_elapsed_s=elapsed,
        )

    def _step_avoid_turn(
        self, nearness: "np.ndarray", now: float, elapsed: float
    ) -> FsmDecision:
        if nearness is None:
            return FsmDecision(
                axes=RcAxes(), state=self._state, sub_state="wait_depth",
                state_elapsed_s=elapsed,
            )
        if elapsed > self.p.max_turn_s:
            self._abort_reason = f"turn >{self.p.max_turn_s:.0f}s"
            self._state = AvoidState.HOVER
            self._ctrl.reset()
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                abort_reason=self._abort_reason, state_elapsed_s=elapsed,
            )

        dec = self._ctrl.decide(nearness)
        zones = self._ctrl.zone_nearness(nearness)
        mid = zones[1]

        # AVOID_TURN 期间强制保持横移方向：即使障碍暂时从视野消失
        # （如墙角被横移避开→commit 释放→切回直飞），也不允许切 CRUISE。
        # 进入绕行就锁定方向，直到 min_turn_s 到期、确认真正绕过。
        # （2026-07-24 真机：墙角横移 3s 消失→commit 释放→直飞→撞上真柱子）
        if dec.state == "CRUISE" and self._ctrl._commit != 0:
            roll = self._ctrl._commit * self._ctrl.p.strafe_speed
            state = "STRAFE_R" if self._ctrl._commit > 0 else "STRAFE_L"
            dec = AvoidDecision(
                axes=RcAxes(pitch=self._ctrl.p.turn_pitch, roll=roll),
                state=state, zones=zones,
            )

        # 前方通畅 → 转入越障确认（需满足最低横移时间）
        if mid < self.p.pass_clear_thresh:
            self._pass_clear_count += 1
        else:
            self._pass_clear_count = 0

        # 横移绕行至少保持 min_turn_s，不能中区一空就过早切走
        # （2026-07-24 真机：横移 3 秒只让开 ~1m，椅子宽度未清空就切直飞）
        if elapsed > self.p.min_turn_s and self._pass_clear_count >= self.p.pass_consecutive_frames:
            self._state = AvoidState.PASS_OBSTACLE
            self._state_entered = now
            return FsmDecision(
                axes=dec.axes, state=self._state, sub_state=dec.state,
                state_elapsed_s=0.0,
            )

        return FsmDecision(
            axes=dec.axes, state=self._state, sub_state=dec.state,
            state_elapsed_s=elapsed,
        )

    def _step_pass_obstacle(
        self, nearness: "np.ndarray", now: float, elapsed: float,
        telemetry: dict[str, str],
    ) -> FsmDecision:
        if nearness is None:
            return FsmDecision(
                axes=RcAxes(), state=self._state, sub_state="wait_depth",
                state_elapsed_s=elapsed,
            )
        if elapsed > self.p.max_pass_s:
            self._abort_reason = f"pass >{self.p.max_pass_s:.0f}s"
            self._state = AvoidState.HOVER
            self._ctrl.reset()
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                abort_reason=self._abort_reason, state_elapsed_s=elapsed,
            )

        dec = self._ctrl.decide(nearness)
        zones = self._ctrl.zone_nearness(nearness)
        mid = zones[1]

        # 侧方也通畅 → 障碍已过，进入航向恢复
        if mid < self.p.pass_clear_thresh:
            self._pass_clear_count += 1
        else:
            self._pass_clear_count = 0
            # 又检测到障碍，回到绕行
            if mid > self.p.pass_clear_thresh + 0.1:
                self._state = AvoidState.AVOID_TURN
                self._state_entered = now
                return FsmDecision(
                    axes=dec.axes, state=self._state, sub_state=dec.state,
                    state_elapsed_s=0.0,
                )

        if elapsed > self.p.min_pass_s and self._pass_clear_count >= self.p.pass_consecutive_frames:
            self._state = AvoidState.RECOVER_HEADING
            self._state_entered = now
            self._ctrl.reset()
            return FsmDecision(
                axes=RcAxes(), state=self._state, sub_state="",
                state_elapsed_s=0.0,
            )

        # 保持小前进量
        return FsmDecision(
            axes=dec.axes, state=self._state, sub_state=dec.state,
            state_elapsed_s=elapsed,
        )

    def _step_recover_heading(
        self, now: float, elapsed: float, telemetry: dict[str, str],
    ) -> FsmDecision:
        if elapsed > self.p.max_recover_s:
            # 超时也视为完成（航向恢复不是关键安全项）
            if self.p.post_clear_s > 0:
                self._state = AvoidState.POST_CLEAR
            else:
                self._state = AvoidState.APPROACH
            self._state_entered = now
            self._ctrl.reset()
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                state_elapsed_s=0.0,
            )

        current_yaw = self._read_yaw(telemetry)
        if current_yaw is None or self._initial_yaw is None:
            # 无法获取 yaw → 跳过航向恢复
            if self.p.post_clear_s > 0:
                self._state = AvoidState.POST_CLEAR
            else:
                self._state = AvoidState.APPROACH
            self._state_entered = now
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                state_elapsed_s=0.0,
            )

        # 计算偏航误差 [-180, 180]
        error = (self._initial_yaw - current_yaw + 180) % 360 - 180
        if abs(error) < self.p.heading_tolerance_deg:
            if self.p.post_clear_s > 0:
                self._state = AvoidState.POST_CLEAR
            else:
                self._state = AvoidState.APPROACH
            self._state_entered = now
            self._ctrl.reset()
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                heading_error_deg=error, state_elapsed_s=0.0,
            )

        # yaw 杆量：误差 * 增益，限幅
        yaw = max(-self._ctrl.p.yaw_speed,
                  min(self._ctrl.p.yaw_speed,
                      int(round(error * self.p.heading_yaw_gain))))
        # 确保最小杆量
        if abs(yaw) < 10:
            yaw = 10 if error > 0 else -10

        return FsmDecision(
            axes=RcAxes(yaw=yaw),
            state=self._state,
            sub_state=f"yaw_err={error:.0f}",
            heading_error_deg=error,
            state_elapsed_s=elapsed,
        )

    def _step_post_clear(
        self, nearness: "np.ndarray", now: float, elapsed: float
    ) -> FsmDecision:
        """越障后额外前进，可选安全扫描。"""
        # 超时 → 回到巡航，继续往前飞（不再悬停等待）
        # （2026-07-24 真机：绕过后停住不符合预期，应继续巡航）
        if elapsed > self.p.post_clear_s:
            self._state = AvoidState.APPROACH
            self._state_entered = now
            self._ctrl.reset()
            return FsmDecision(
                axes=RcAxes(), state=self._state,
                sub_state="done", state_elapsed_s=0.0,
            )

        # 前方又检测到障碍 → 回到巡航，让 APPROACH 重新触发绕行
        if nearness is not None:
            zones = self._ctrl.zone_nearness(nearness)
            if zones[1] > self.p.pass_clear_thresh + 0.15:
                self._state = AvoidState.APPROACH
                self._state_entered = now
                self._ctrl.reset()
                return FsmDecision(
                    axes=RcAxes(), state=self._state,
                    sub_state="blocked", state_elapsed_s=0.0,
                )

        # 匀速前进
        return FsmDecision(
            axes=RcAxes(pitch=self.p.post_clear_pitch),
            state=self._state,
            sub_state=f"fwd {elapsed:.1f}s",
            state_elapsed_s=elapsed,
        )

    def _step_orbit(
        self, nearness: "np.ndarray", now: float, elapsed: float
    ) -> FsmDecision:
        """POI 环绕（备用路径；真路径走 APPROACH+_run_orbit）。"""
        if nearness is None:
            return FsmDecision(
                axes=RcAxes(), state=self._state, sub_state="wait_depth",
                state_elapsed_s=elapsed,
            )
        return self._finish_orbit_decision(self._orbit.decide(nearness), now, elapsed)

    # ── 辅助 ──────────────────────────────────────────────────

    def _check_safety(self, telemetry: dict[str, str], now: float) -> str:
        """全局安全检查，返回非空 = ABORT 原因。"""
        # 低电量
        try:
            bat = int(telemetry.get("bat", "100"))
            if bat < self.p.min_battery_pct:
                return f"low battery {bat}%"
        except (ValueError, TypeError):
            pass

        # 高度异常
        try:
            h = int(telemetry.get("h", "0"))
            if h < 0 or h > self.p.max_height_cm:
                return f"height anomaly {h}cm"
        except (ValueError, TypeError):
            pass

        return ""

    @staticmethod
    def _read_yaw(telemetry: dict[str, str]) -> Optional[float]:
        try:
            return float(telemetry.get("yaw", ""))
        except (ValueError, TypeError):
            return None

    def reset(self) -> None:
        """人工接管时复位。"""
        self._state = AvoidState.PREFLIGHT
        self._auto_engaged = None
        self._initial_yaw = None
        self._pass_clear_count = 0
        self._danger_history.clear()
        self._abort_reason = ""
        self._ctrl.reset()
        self._orbit.reset()
        self._orbit_active = False
        self._orbit_lost_since = None
        self._orbit_relock_count = 0
        self._last_depth_ts = 0.0
