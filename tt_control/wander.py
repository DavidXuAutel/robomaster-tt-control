"""随机漫游探索（Wander）控制内核。

纯函数式策略：decide(nearness, telemetry, now) → WanderDecision。
不碰 socket / 线程 / sleep；时间与 RNG 均可注入，便于单测复现。

规格：docs/design/2026-07-27-wander-explore-design.md
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from tt_control.avoidance import AvoidanceController
from tt_control.control import RcAxes

# 状态名（写入 frames.csv wander_state / HUD）
WANDER_CRUISE = "WANDER_CRUISE"
WANDER_TURN = "WANDER_TURN"
WANDER_VERIFY = "WANDER_VERIFY"
WANDER_PANO = "WANDER_PANO"
DANGER_HOLD = "DANGER_HOLD"
WANDER_RETREAT = "WANDER_RETREAT"


@dataclass
class WanderParams:
    """漫游参数；全部阈值只允许出现在 configs/default.json 的 wander 节。"""

    seed: int = 0  # 0 = 推迟到首次 decide(now) 用 now 定种
    cruise_pitch_min: int = 12
    cruise_pitch_max: int = 25
    segment_s_min: float = 3.0
    segment_s_max: float = 8.0
    turn_thresh: float = 0.58
    turn_confirm_frames: int = 2
    clear_thresh: float = 0.40
    verify_frames: int = 3
    verify_timeout_s: float = 3.0
    turn_min_deg: float = 50.0
    turn_max_deg: float = 130.0
    free_turn_prob: float = 0.5
    free_turn_min_deg: float = 20.0
    free_turn_max_deg: float = 60.0
    turn_open_bias: float = 0.8
    yaw_speed: int = 40
    turn_arrive_tol_deg: float = 8.0
    # 无 yaw 遥测时：估角速度 (deg/s) ≈ |yaw_cmd| * yaw_dead_reckon_dps_per_unit
    yaw_dead_reckon_dps_per_unit: float = 1.0
    danger_thresh: float = 0.78
    danger_hold_s: float = 0.4
    retreat_pitch: int = 15
    retreat_s: float = 0.7
    corner_window_s: float = 20.0
    corner_max_turns: int = 4
    h_min_cm: float = 80.0
    h_max_cm: float = 200.0
    alt_change_prob: float = 0.3
    alt_throttle: int = 25
    alt_segment_s_min: float = 1.0
    alt_segment_s_max: float = 2.0
    pano_yaw_speed: int = 30
    pano_complete_deg: float = 340.0
    pano_min_s: float = 2.0
    pano_timeout_s: float = 20.0
    pano_step_deadband_deg: float = 0.5
    h_missing_s: float = 1.0  # 连续读不到 h 超过此时长 → 闩锁；连续读到满此时长 → 解除（主人 2026-07-28 裁定可恢复）
    h_missing_frames: int = 5  # deprecated: 不再参与逻辑，保留兼容旧配置


@dataclass
class WanderDecision:
    """每帧输出。"""

    axes: RcAxes
    state: str
    zones: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
    event: str = ""
    abort_reason: str = ""
    turn_end_ts: float = 0.0
    yaw_mode: str = ""  # "" | "yaw_dead_reckon"
    sub_state: str = ""

    def as_hud(self) -> str:
        parts = [self.state]
        if self.sub_state:
            parts.append(self.sub_state)
        if self.event:
            parts.append(self.event)
        if self.abort_reason:
            parts.append(f"ABORT:{self.abort_reason}")
        return " | ".join(parts)


def _yaw_delta(target: float, current: float) -> float:
    """最短角差 ∈ (-180, 180]。"""
    return (target - current + 180.0) % 360.0 - 180.0



class WanderPolicy:
    """随机漫游策略内核。"""

    def __init__(
        self,
        params: Optional[WanderParams] = None,
        seed: Optional[int] = None,
        *,
        zones_ctrl: Optional[AvoidanceController] = None,
    ) -> None:
        self.p = params or WanderParams()
        raw_seed = self.p.seed if seed is None else seed
        self._seed_pending = int(raw_seed) == 0
        self.seed = 0 if self._seed_pending else int(raw_seed)
        self._rng = random.Random(self.seed)
        self._zones_ctrl = zones_ctrl or AvoidanceController()

        self.turns_total = 0
        self.panos_total = 0
        self.retreats_total = 0
        self.corner_aborts = 0

        self._state = WANDER_CRUISE
        self._state_entered: float = 0.0
        self._started = False

        # 深度帧计数（按 depth_ts 变化）
        self._last_depth_ts: Optional[float] = None
        self._turn_confirm = 0
        self._verify_clear = 0
        self._turn_end_ts: float = 0.0

        # CRUISE 段参数
        self._seg_pitch = 0
        self._seg_dur = 0.0
        self._seg_started: float = 0.0
        self._alt_throttle = 0
        self._alt_until: float = 0.0

        # TURN
        self._turn_dir = 1  # +1=右(yaw+) / -1=左(yaw-)
        self._turn_deg = 0.0
        self._turn_reason = "obstacle"
        self._turn_start_yaw: Optional[float] = None
        self._turn_start_t: float = 0.0
        self._turn_yaw_elapsed: float = 0.0
        self._turn_last_tick: Optional[float] = None
        self._yaw_dead_reckon = False
        self._pending_event = ""

        # PANO
        self._pano_start_yaw: Optional[float] = None
        self._pano_start_t: Optional[float] = None
        self._pano_seek_started: Optional[float] = None
        self._pano_samples: list[tuple[float, float]] = []
        self._pano_target_yaw: Optional[float] = None
        self._pano_phase = "scan"  # scan | seek
        self._pano_last_yaw: Optional[float] = None
        self._pano_travel_deg: float = 0.0
        self._pano_done_at: Optional[float] = None

        # DANGER / RETREAT
        self._danger_since: Optional[float] = None
        self._danger_hold_depth_s: float = 0.0  # 仅新深度帧累计的危险持续时间
        self._retreat_started: float = 0.0
        self._retreat_end_ts: float = 0.0  # retreat 结束墙钟；abort 需更新深度
        self._resume_after_danger = WANDER_CRUISE

        # 角落窗口：遇障转向时间戳
        self._obstacle_turn_times: list[float] = []

        # 高度
        self._h_missing_since: Optional[float] = None
        self._h_recover_since: Optional[float] = None
        self._h_latch_zero = False
        self._tel: dict[str, str] = {}

    # ── 公开 API ──────────────────────────────────────────────

    def reset(self) -> None:
        """复位控制状态；保留 seed / RNG / 累计计数（AUTO 暂停续飞用）。"""
        self._reset_control_state()

    def begin_episode(self, seed: Optional[int] = None) -> int:
        """新 episode：重设 RNG + 清零计数器，返回生效 seed（写入 meta）。

        seed=0 时推迟到首次 decide(now) 用注入的 now 定种（不调用 time.time）。
        """
        raw = self.p.seed if seed is None else seed
        self._seed_pending = int(raw) == 0
        self.seed = 0 if self._seed_pending else int(raw)
        self._rng = random.Random(self.seed)
        self.turns_total = 0
        self.panos_total = 0
        self.retreats_total = 0
        self.corner_aborts = 0
        self._reset_control_state()
        return self.seed

    def _ensure_seed(self, now: float) -> None:
        if not self._seed_pending:
            return
        self.seed = int(now * 1000.0) & 0x7FFFFFFF
        self._rng = random.Random(self.seed)
        self._seed_pending = False

    def _reset_control_state(self) -> None:
        self._state = WANDER_CRUISE
        self._state_entered = 0.0
        self._started = False
        self._last_depth_ts = None
        self._turn_confirm = 0
        self._verify_clear = 0
        self._turn_end_ts = 0.0
        self._seg_pitch = 0
        self._seg_dur = 0.0
        self._seg_started = 0.0
        self._alt_throttle = 0
        self._alt_until = 0.0
        self._turn_dir = 1
        self._turn_deg = 0.0
        self._turn_reason = "obstacle"
        self._turn_start_yaw = None
        self._turn_start_t = 0.0
        self._turn_yaw_elapsed = 0.0
        self._turn_last_tick = None
        self._yaw_dead_reckon = False
        self._pending_event = ""
        self._pano_start_yaw = None
        self._pano_start_t = None
        self._pano_seek_started = None
        self._pano_samples = []
        self._pano_target_yaw = None
        self._pano_phase = "scan"
        self._pano_last_yaw = None
        self._pano_travel_deg = 0.0
        self._pano_done_at = None
        self._danger_since = None
        self._danger_hold_depth_s = 0.0
        self._retreat_started = 0.0
        self._retreat_end_ts = 0.0
        self._resume_after_danger = WANDER_CRUISE
        self._obstacle_turn_times.clear()
        self._h_missing_since = None
        self._h_recover_since = None
        self._h_latch_zero = False
        self._tel = {}

    def params_dict(self) -> dict[str, Any]:
        return asdict(self.p)

    def stats_dict(self) -> dict[str, Any]:
        return {
            "wander_seed": self.seed,
            "wander_params": self.params_dict(),
            "turns_total": self.turns_total,
            "panos_total": self.panos_total,
            "retreats_total": self.retreats_total,
            "corner_aborts": self.corner_aborts,
        }

    def decide(
        self,
        nearness: Optional[np.ndarray],
        telemetry: dict[str, str],
        now: float,
        *,
        depth_ts: Optional[float] = None,
    ) -> WanderDecision:
        self._ensure_seed(now)
        self._tel = telemetry
        if not self._started:
            self._started = True
            self._state_entered = now
            self._begin_cruise_segment(now, event=True)

        if nearness is None:
            return WanderDecision(
                axes=RcAxes(),
                state=self._state,
                sub_state="wait_depth",
            )

        left, mid, right = self._zones_ctrl.zone_nearness(nearness)
        zones = (left, mid, right)
        new_depth = self._note_depth(depth_ts)

        # 任意状态：danger 三段式（需新深度帧证据，避免冻帧空转计时）
        danger = max(zones)
        if self._state not in (DANGER_HOLD, WANDER_RETREAT):
            if new_depth and danger > self.p.danger_thresh:
                self._resume_after_danger = self._state
                self._danger_since = now
                self._danger_hold_depth_s = 0.0
                self._enter(DANGER_HOLD, now)
                return self._pack(
                    RcAxes(), zones, now, event="DANGER_HOLD",
                    sub=f"danger={danger:.2f}",
                )
            if new_depth:
                self._danger_since = None
                self._danger_hold_depth_s = 0.0

        if self._state == DANGER_HOLD:
            return self._step_danger_hold(zones, now, danger, new_depth)
        if self._state == WANDER_RETREAT:
            return self._step_retreat(zones, now, danger, new_depth, depth_ts)
        if self._state == WANDER_PANO:
            return self._step_pano(zones, telemetry, now, new_depth, mid)
        if self._state == WANDER_TURN:
            return self._step_turn(zones, telemetry, now)
        if self._state == WANDER_VERIFY:
            return self._step_verify(zones, now, new_depth, mid, depth_ts)
        return self._step_cruise(zones, now, new_depth, mid)

    # ── 状态实现 ──────────────────────────────────────────────

    def _step_cruise(
        self,
        zones: tuple[float, float, float],
        now: float,
        new_depth: bool,
        mid: float,
    ) -> WanderDecision:
        left, _, right = zones
        event = self._pop_event()

        # 遇障转向确认（连续深度帧）
        if mid > self.p.turn_thresh:
            if new_depth:
                self._turn_confirm += 1
            if self._turn_confirm >= self.p.turn_confirm_frames:
                return self._start_obstacle_turn(zones, now)
        else:
            if new_depth:
                self._turn_confirm = 0

        # 段结束：可选 free turn，否则开新段
        if now - self._seg_started >= self._seg_dur:
            if self._rng.random() < self.p.free_turn_prob:
                return self._start_free_turn(zones, now)
            self._begin_cruise_segment(now, event=True)
            event = self._pop_event() or event

        pitch = self._seg_pitch
        throttle = self._cruise_throttle(zones, now, mid)
        # 高度子段内 pitch 减半（坑 #5）
        if self._alt_throttle != 0 and now < self._alt_until:
            pitch = max(0, pitch // 2)

        return self._pack(
            RcAxes(pitch=pitch, throttle=throttle),
            zones, now, event=event,
            sub=f"p={pitch} M{mid:.2f}",
        )

    def _step_turn(
        self,
        zones: tuple[float, float, float],
        telemetry: dict[str, str],
        now: float,
    ) -> WanderDecision:
        event = self._pop_event()
        yaw_cmd = int(self._turn_dir * self.p.yaw_speed)
        cur_yaw = self._read_yaw(telemetry)

        # 只累加实际发 yaw 杆的时间（DANGER 暂停不计入）
        if self._turn_last_tick is not None:
            self._turn_yaw_elapsed += max(0.0, now - self._turn_last_tick)
        self._turn_last_tick = now

        if cur_yaw is None:
            self._yaw_dead_reckon = True
            est = abs(yaw_cmd) * self.p.yaw_dead_reckon_dps_per_unit * self._turn_yaw_elapsed
            done = est >= self._turn_deg
        else:
            if self._turn_start_yaw is None:
                self._turn_start_yaw = cur_yaw
            # 有符号进度：仅命令方向计入到位
            progress = _yaw_delta(cur_yaw, self._turn_start_yaw) * self._turn_dir
            done = progress >= (self._turn_deg - self.p.turn_arrive_tol_deg)

        if done:
            return self._enter_verify(zones, now, event=event, sub="turn_done")

        return self._pack(
            RcAxes(yaw=yaw_cmd), zones, now, event=event,
            sub=f"turn_{'R' if self._turn_dir > 0 else 'L'}{self._turn_deg:.0f}",
            yaw_mode="yaw_dead_reckon" if self._yaw_dead_reckon else "",
        )

    def _enter_verify(
        self,
        zones: tuple[float, float, float],
        now: float,
        *,
        event: str = "",
        sub: str = "verify",
        mark_pano_done: bool = False,
    ) -> WanderDecision:
        self._turn_end_ts = now
        self._verify_clear = 0
        self._turn_last_tick = None
        if mark_pano_done:
            self._pano_done_at = now
        self._enter(WANDER_VERIFY, now)
        return self._pack(
            RcAxes(), zones, now, event=event, sub=sub,
            yaw_mode="yaw_dead_reckon" if self._yaw_dead_reckon else "",
        )

    def _step_verify(
        self,
        zones: tuple[float, float, float],
        now: float,
        new_depth: bool,
        mid: float,
        depth_ts: Optional[float],
    ) -> WanderDecision:
        # 超时 → 同向追加转角
        if now - self._state_entered >= self.p.verify_timeout_s:
            return self._start_retry_turn(zones, now)

        # 只认 turn_end_ts 之后的新深度帧
        ts_ok = depth_ts is not None and depth_ts > self._turn_end_ts
        if new_depth and ts_ok:
            if mid < self.p.clear_thresh:
                self._verify_clear += 1
            else:
                self._verify_clear = 0

        if self._verify_clear >= self.p.verify_frames:
            self._enter(WANDER_CRUISE, now)
            self._begin_cruise_segment(now, event=True)
            return self._pack(
                RcAxes(), zones, now, event=self._pop_event(),
                sub="verify_ok",
            )

        return self._pack(
            RcAxes(), zones, now,
            sub=f"verify {self._verify_clear}/{self.p.verify_frames} M{mid:.2f}",
        )

    def _step_danger_hold(
        self,
        zones: tuple[float, float, float],
        now: float,
        danger: float,
        new_depth: bool,
    ) -> WanderDecision:
        if new_depth and danger <= self.p.danger_thresh:
            self._danger_since = None
            self._danger_hold_depth_s = 0.0
            # 恢复 TURN 时丢弃暂停时长，避免 dead-reckon 空转到位
            if self._resume_after_danger == WANDER_TURN:
                self._turn_last_tick = now
            self._enter(self._resume_after_danger, now)
            return self._pack(RcAxes(), zones, now, sub="danger_clear")

        if new_depth and danger > self.p.danger_thresh:
            # 用深度帧间隔累加（无上一帧则按 hold 粒度计一小步）
            if self._danger_since is not None:
                dt = max(0.0, now - self._danger_since)
                # 单帧间隔上限，避免深度断流后一帧灌进超长 hold
                self._danger_hold_depth_s += min(dt, self.p.danger_hold_s)
            self._danger_since = now

        held = self._danger_hold_depth_s
        if held >= self.p.danger_hold_s:
            self.retreats_total += 1
            self._retreat_started = now
            self._retreat_end_ts = 0.0
            self._enter(WANDER_RETREAT, now)
            return self._pack(
                RcAxes(pitch=-abs(self.p.retreat_pitch)),
                zones, now, event="RETREAT",
                sub="to_retreat",
            )
        return self._pack(
            RcAxes(), zones, now, event="DANGER_HOLD",
            sub=f"hold {held:.2f}/{self.p.danger_hold_s:.2f}",
        )

    def _step_retreat(
        self,
        zones: tuple[float, float, float],
        now: float,
        danger: float,
        new_depth: bool,
        depth_ts: Optional[float],
    ) -> WanderDecision:
        if now - self._retreat_started < self.p.retreat_s:
            return self._pack(
                RcAxes(pitch=-abs(self.p.retreat_pitch)),
                zones, now, sub="retreating",
            )
        if self._retreat_end_ts <= 0.0:
            self._retreat_end_ts = now
        # 后退结束后必须看到 retreat 完成后的新深度，才判定 abort / 转向
        ts_ok = depth_ts is not None and depth_ts > self._retreat_end_ts
        if not (new_depth and ts_ok):
            return self._pack(RcAxes(), zones, now, sub="post_retreat_wait_depth")
        if danger > self.p.danger_thresh:
            return WanderDecision(
                axes=RcAxes(),
                state=WANDER_RETREAT,
                zones=zones,
                abort_reason="wander_danger",
                sub_state="post_retreat_danger",
            )
        return self._start_obstacle_turn(zones, now)

    def _step_pano(
        self,
        zones: tuple[float, float, float],
        telemetry: dict[str, str],
        now: float,
        new_depth: bool,
        mid: float,
    ) -> WanderDecision:
        event = self._pop_event()
        cur_yaw = self._read_yaw(telemetry)
        yaw_cmd = self.p.pano_yaw_speed

        if self._pano_phase == "scan":
            if self._pano_start_t is None:
                self._pano_start_t = now
            if self._pano_start_yaw is None and cur_yaw is not None:
                self._pano_start_yaw = cur_yaw
                self._pano_last_yaw = cur_yaw
                self._pano_travel_deg = 0.0
            if cur_yaw is not None and self._pano_last_yaw is not None:
                # 有符号累加；噪声死区以下不推进 last_yaw（避免慢转被滤掉）
                step = _yaw_delta(cur_yaw, self._pano_last_yaw)
                if abs(step) >= self.p.pano_step_deadband_deg:
                    self._pano_travel_deg += step
                    self._pano_last_yaw = cur_yaw
            if new_depth and cur_yaw is not None:
                self._pano_samples.append((cur_yaw, mid))

            start_t = self._pano_start_t if self._pano_start_t is not None else now
            timed_out = (now - start_t) >= self.p.pano_timeout_s
            scanned = False
            if cur_yaw is not None and self._pano_start_yaw is not None:
                scanned = (
                    abs(self._pano_travel_deg) >= self.p.pano_complete_deg
                    and (now - start_t) >= self.p.pano_min_s
                )
            else:
                rate = abs(yaw_cmd) * self.p.yaw_dead_reckon_dps_per_unit
                scanned = (now - start_t) * rate >= 360.0
            scanned = scanned or timed_out

            if scanned:
                if self._pano_samples:
                    best_yaw, _ = min(self._pano_samples, key=lambda x: x[1])
                elif cur_yaw is not None:
                    best_yaw = cur_yaw
                else:
                    best_yaw = 0.0
                self._pano_target_yaw = best_yaw
                self._pano_phase = "seek"
                self._pano_seek_started = now
                self._pending_event = f"PANO({best_yaw:.1f})"
                self.panos_total += 1
                event = self._pop_event()

            return self._pack(
                RcAxes(yaw=yaw_cmd), zones, now, event=event,
                sub=f"pano_scan n={len(self._pano_samples)}",
            )

        # seek：转到选定朝向 → 必须进 VERIFY（P0-1）
        seek_t0 = self._pano_seek_started if self._pano_seek_started is not None else now
        if cur_yaw is None or self._pano_target_yaw is None:
            if now - seek_t0 > 2.0:
                return self._enter_verify(
                    zones, now, event=event, sub="pano_seek_timeout", mark_pano_done=True,
                )
            return self._pack(RcAxes(yaw=yaw_cmd), zones, now, event=event, sub="pano_seek")

        err = _yaw_delta(self._pano_target_yaw, cur_yaw)
        if abs(err) <= self.p.turn_arrive_tol_deg:
            return self._enter_verify(
                zones, now, event=event, sub="pano_done", mark_pano_done=True,
            )
        yaw = int(np.sign(err) * self.p.pano_yaw_speed) or self.p.pano_yaw_speed
        return self._pack(RcAxes(yaw=yaw), zones, now, event=event, sub=f"pano_seek e={err:.0f}")

    # ── 转向 / 段 / 角落 ──────────────────────────────────────

    def _start_obstacle_turn(
        self, zones: tuple[float, float, float], now: float
    ) -> WanderDecision:
        self._prune_corner_window(now)
        # PANO 完成后 corner_window_s 内再遇障 → abort（与旧转向时间戳解耦）
        if (
            self._pano_done_at is not None
            and now - self._pano_done_at <= self.p.corner_window_s
        ):
            self.corner_aborts += 1
            return WanderDecision(
                axes=RcAxes(),
                state=self._state,
                zones=zones,
                abort_reason="wander_cornered",
                sub_state="cornered",
            )

        # 第 corner_max_turns 次遇障转向 → 全景选向
        if len(self._obstacle_turn_times) + 1 >= self.p.corner_max_turns:
            self._obstacle_turn_times.append(now)
            self.turns_total += 1  # 计遇障触发；PANO 本身另计 panos_total
            self._pano_phase = "scan"
            self._pano_samples = []
            self._pano_start_yaw = None
            self._pano_last_yaw = None
            self._pano_travel_deg = 0.0
            self._pano_target_yaw = None
            self._pano_start_t = now
            self._pano_seek_started = None
            self._enter(WANDER_PANO, now)
            return self._pack(
                RcAxes(yaw=self.p.pano_yaw_speed),
                zones, now, event="",
                sub="to_pano",
            )

        left, _, right = zones
        direction = self._pick_turn_dir(left, right)
        deg = self._rng.uniform(self.p.turn_min_deg, self.p.turn_max_deg)
        self._obstacle_turn_times.append(now)
        return self._arm_turn(zones, now, direction, deg, "obstacle")

    def _start_free_turn(
        self, zones: tuple[float, float, float], now: float
    ) -> WanderDecision:
        left, _, right = zones
        direction = self._pick_turn_dir(left, right)
        deg = self._rng.uniform(self.p.free_turn_min_deg, self.p.free_turn_max_deg)
        return self._arm_turn(zones, now, direction, deg, "free")

    def _start_retry_turn(
        self, zones: tuple[float, float, float], now: float
    ) -> WanderDecision:
        deg = self._rng.uniform(self.p.turn_min_deg, self.p.turn_max_deg)
        # 同向追加
        self._turn_deg = deg
        self._turn_reason = "retry"
        self._turn_start_yaw = None
        self._turn_start_t = now
        self._turn_yaw_elapsed = 0.0
        self._turn_last_tick = now
        self._yaw_dead_reckon = False
        self._turn_confirm = 0
        self.turns_total += 1
        dir_ch = "R" if self._turn_dir > 0 else "L"
        self._pending_event = f"TURN({dir_ch},{deg:.0f},retry)"
        self._enter(WANDER_TURN, now)
        return self._pack(
            RcAxes(yaw=int(self._turn_dir * self.p.yaw_speed)),
            zones, now, event=self._pop_event(),
            sub="retry_turn",
        )

    def _arm_turn(
        self,
        zones: tuple[float, float, float],
        now: float,
        direction: int,
        deg: float,
        reason: str,
    ) -> WanderDecision:
        self._turn_dir = 1 if direction >= 0 else -1
        self._turn_deg = float(deg)
        self._turn_reason = reason
        # 在 arm 帧锁定起始 yaw，避免下一帧才采样导致转角少算一拍
        self._turn_start_yaw = self._read_yaw(self._tel)
        self._turn_start_t = now
        self._turn_yaw_elapsed = 0.0
        self._turn_last_tick = now
        self._yaw_dead_reckon = self._turn_start_yaw is None
        self._turn_confirm = 0
        self.turns_total += 1
        dir_ch = "R" if self._turn_dir > 0 else "L"
        self._pending_event = f"TURN({dir_ch},{deg:.0f},{reason})"
        self._enter(WANDER_TURN, now)
        return self._pack(
            RcAxes(yaw=int(self._turn_dir * self.p.yaw_speed)),
            zones, now, event=self._pop_event(),
            sub=f"arm_{reason}",
        )

    def _pick_turn_dir(self, left: float, right: float) -> int:
        """偏向开阔侧：nearness 更小 = 更开阔。返回 +1 右 / -1 左。"""
        # 开阔侧：right 更小 → 向右转（+）；left 更小 → 向左转（-）
        open_is_right = right <= left
        open_dir = 1 if open_is_right else -1
        closed_dir = -open_dir
        if self._rng.random() < self.p.turn_open_bias:
            return open_dir
        return closed_dir

    def _begin_cruise_segment(self, now: float, event: bool = False) -> None:
        p = self.p
        lo, hi = int(p.cruise_pitch_min), int(p.cruise_pitch_max)
        if hi < lo:
            lo, hi = hi, lo
        self._seg_pitch = self._rng.randint(lo, hi)
        self._seg_dur = self._rng.uniform(p.segment_s_min, p.segment_s_max)
        self._seg_started = now
        self._turn_confirm = 0
        # 高度微调子段（仅标记；执行时仍要求 mid < clear）
        if self._rng.random() < p.alt_change_prob:
            sign = 1 if self._rng.random() < 0.5 else -1
            self._alt_throttle = sign * abs(p.alt_throttle)
            dur = self._rng.uniform(p.alt_segment_s_min, p.alt_segment_s_max)
            self._alt_until = now + dur
        else:
            self._alt_throttle = 0
            self._alt_until = 0.0
        if event:
            self._pending_event = f"SEG({self._seg_pitch},{self._seg_dur:.1f})"

    def _prune_corner_window(self, now: float) -> None:
        w = self.p.corner_window_s
        self._obstacle_turn_times = [t for t in self._obstacle_turn_times if now - t <= w]
        if self._pano_done_at is not None and now - self._pano_done_at > w:
            self._pano_done_at = None

    # ── 高度 / 遥测 / 深度帧 ──────────────────────────────────

    def _cruise_throttle(
        self, zones: tuple[float, float, float], now: float, mid: float
    ) -> int:
        if self._h_latch_zero:
            return 0
        if self._alt_throttle == 0 or now >= self._alt_until:
            return 0
        # 只在前方开阔时做高度微调
        if mid >= self.p.clear_thresh:
            return 0
        return int(self._alt_throttle)

    def _apply_h_clamp(
        self, throttle: int, telemetry: dict[str, str], now: float
    ) -> int:
        h = self._read_h(telemetry)

        # 闩锁中：连续读到 h 满 h_missing_s → 解除（主人裁定可恢复）
        if self._h_latch_zero:
            if h is None:
                self._h_recover_since = None
                return 0
            if self._h_recover_since is None:
                self._h_recover_since = now
            if now - self._h_recover_since < self.p.h_missing_s:
                return 0
            self._h_latch_zero = False
            self._h_recover_since = None
            self._h_missing_since = None

        if h is None:
            self._h_recover_since = None
            if self._h_missing_since is None:
                self._h_missing_since = now
            elif now - self._h_missing_since >= self.p.h_missing_s:
                self._h_latch_zero = True
            return 0
        self._h_missing_since = None
        # 单向钳制：出带只禁止继续偏离，允许回到巡航带
        if h >= self.p.h_max_cm and throttle > 0:
            return 0
        if h <= self.p.h_min_cm and throttle < 0:
            return 0
        return throttle

    def _read_yaw(self, telemetry: dict[str, str]) -> Optional[float]:
        try:
            return float(telemetry.get("yaw"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def _read_h(self, telemetry: dict[str, str]) -> Optional[float]:
        try:
            return float(telemetry.get("h"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def _note_depth(self, depth_ts: Optional[float]) -> bool:
        """深度帧是否相对上一帧更新（按 ts 变化计）。"""
        if depth_ts is None:
            return False
        if self._last_depth_ts is None or depth_ts != self._last_depth_ts:
            self._last_depth_ts = depth_ts
            return True
        return False

    # ── 组装 ──────────────────────────────────────────────────

    def _enter(self, state: str, now: float) -> None:
        self._state = state
        self._state_entered = now

    def _pop_event(self) -> str:
        e = self._pending_event
        self._pending_event = ""
        return e

    def _pack(
        self,
        axes: RcAxes,
        zones: tuple[float, float, float],
        now: float,
        event: str = "",
        sub: str = "",
        yaw_mode: str = "",
    ) -> WanderDecision:
        # 安全不变量：除 RETREAT 外 pitch≥0、roll≡0
        roll = 0
        pitch = int(axes.pitch)
        yaw = int(axes.yaw)
        throttle = int(axes.throttle)
        if self._state != WANDER_RETREAT:
            pitch = max(0, pitch)
        else:
            pitch = -abs(self.p.retreat_pitch) if pitch < 0 else 0

        throttle = self._apply_h_clamp(throttle, self._tel, now)

        return WanderDecision(
            axes=RcAxes(roll=roll, pitch=pitch, throttle=throttle, yaw=yaw),
            state=self._state,
            zones=zones,
            event=event,
            turn_end_ts=self._turn_end_ts,
            yaw_mode=yaw_mode,
            sub_state=sub,
        )
