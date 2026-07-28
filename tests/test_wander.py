"""WanderPolicy / FSM 漫游路径回归（合成近度图）。

对照：docs/design/2026-07-27-wander-explore-design.md §9.1
修复工单：docs/dev-notes/2026-07-28-wander-defect-ticket.md
"""

from __future__ import annotations

import numpy as np
import pytest

from tt_control.avoidance import AvoidanceController
from tt_control.avoidance_fsm import AvoidanceFSM, FsmParams
from tt_control.wander import (
    DANGER_HOLD,
    WANDER_CRUISE,
    WANDER_PANO,
    WANDER_RETREAT,
    WANDER_TURN,
    WANDER_VERIFY,
    WanderParams,
    WanderPolicy,
)


def _grid(val, shape=(96, 128)):
    return np.full(shape, val, dtype=np.float32)


def _wall(mid_val=0.65, left=0.20, right=0.45, shape=(96, 128)):
    n = _grid(0.10, shape)
    h, w = shape
    third = w // 3
    n[:, :third] = left
    n[:, third : 2 * third] = mid_val
    n[:, 2 * third :] = right
    return n


def _tel(bat=80, h=120, yaw=0):
    d = {"bat": str(bat), "yaw": str(yaw)}
    if h is not None:
        d["h"] = str(h)
    return d


def _policy(**kw) -> WanderPolicy:
    base = dict(seed=42, free_turn_prob=0.0, alt_change_prob=0.0)
    base.update(kw)
    return WanderPolicy(WanderParams(**base), seed=42)


def test_open_cruise_pitch_in_range_roll_zero():
    pol = _policy(cruise_pitch_min=12, cruise_pitch_max=25, segment_s_min=5, segment_s_max=5)
    d = pol.decide(_grid(0.15), _tel(), now=1.0, depth_ts=1.0)
    assert d.state == WANDER_CRUISE
    assert d.axes.roll == 0
    assert 12 <= d.axes.pitch <= 25


def test_invariant_roll_zero_pitch_neg_only_retreat():
    """全状态遍历：roll≡0；pitch<0 仅 RETREAT 且时长受限。"""
    pol = _policy(
        turn_confirm_frames=1,
        danger_hold_s=0.15,
        retreat_s=0.4,
        verify_frames=2,
        segment_s_min=100,
        segment_s_max=100,
        corner_max_turns=1,
        pano_complete_deg=40.0,
        pano_min_s=0.0,
        pano_step_deadband_deg=0.0,
    )
    t, dts, yaw = 10.0, 10.0, 0.0
    states_seen: set[str] = set()

    def step(near, y=None):
        nonlocal t, dts, yaw
        t += 0.2
        dts += 0.2
        if y is not None:
            yaw = y
        d = pol.decide(near, _tel(yaw=yaw), now=t, depth_ts=dts)
        states_seen.add(d.state)
        assert d.axes.roll == 0
        if d.state != WANDER_RETREAT:
            assert d.axes.pitch >= 0
        return d

    step(_grid(0.2))
    d = step(_wall(0.70, left=0.55, right=0.15))
    assert d.state == WANDER_PANO
    for i in range(30):
        d = step(_grid(0.12), y=(yaw + 15) % 360)
        if d.state == WANDER_VERIFY:
            break
    assert WANDER_PANO in states_seen
    assert WANDER_VERIFY in states_seen or d.state == WANDER_VERIFY

    pol2 = _policy(danger_hold_s=0.15, retreat_s=0.4)
    t2, dts2 = 100.0, 100.0
    pol2.decide(_grid(0.2), _tel(), now=t2, depth_ts=dts2)
    t2 += 0.1; dts2 += 0.1
    pol2.decide(_grid(0.90), _tel(), now=t2, depth_ts=dts2)
    t2 += 0.2; dts2 += 0.2
    d = pol2.decide(_grid(0.90), _tel(), now=t2, depth_ts=dts2)
    assert d.state == WANDER_RETREAT and d.axes.pitch < 0
    retreat_start = t2
    while t2 - retreat_start < 0.35:
        t2 += 0.1; dts2 += 0.1
        d = pol2.decide(_grid(0.90), _tel(), now=t2, depth_ts=dts2)
        assert d.axes.roll == 0
        if d.state == WANDER_RETREAT:
            assert d.axes.pitch < 0


def test_single_frame_spike_no_turn():
    pol = _policy(turn_confirm_frames=2, segment_s_min=50, segment_s_max=50)
    t, dts = 1.0, 1.0
    pol.decide(_grid(0.2), _tel(), now=t, depth_ts=dts)
    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70), _tel(), now=t, depth_ts=dts)
    assert d.state == WANDER_CRUISE
    d = pol.decide(_wall(0.70), _tel(), now=t + 0.05, depth_ts=dts)
    assert d.state == WANDER_CRUISE


def test_turn_yaw_sign_follows_open_side():
    pol = _policy(
        turn_confirm_frames=1,
        turn_open_bias=1.0,
        segment_s_min=50,
        segment_s_max=50,
    )
    t, dts = 1.0, 1.0
    pol.decide(_grid(0.2), _tel(), now=t, depth_ts=dts)
    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(mid_val=0.70, left=0.15, right=0.55), _tel(), now=t, depth_ts=dts)
    assert d.state == WANDER_TURN
    assert d.axes.pitch == 0
    assert d.axes.yaw < 0


def test_verify_rejects_stale_depth_before_turn_end_ts():
    pol = _policy(
        turn_confirm_frames=1,
        turn_min_deg=40,
        turn_max_deg=40,
        turn_arrive_tol_deg=5,
        verify_frames=2,
        clear_thresh=0.40,
        segment_s_min=50,
        segment_s_max=50,
    )
    t, dts = 10.0, 10.0
    pol.decide(_grid(0.2), _tel(yaw=0), now=t, depth_ts=dts)
    t += 0.2
    dts += 0.2
    # 右侧更开阔 → 右转；yaw 正向增加
    pol.decide(_wall(0.70, left=0.55, right=0.15), _tel(yaw=0), now=t, depth_ts=dts)
    for yaw in (20, 40, 50):
        t += 0.2
        dts += 0.05
        d = pol.decide(_wall(0.70, left=0.55, right=0.15), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_VERIFY
    turn_end = d.turn_end_ts
    for _ in range(5):
        t += 0.2
        d = pol.decide(_grid(0.10), _tel(yaw=50), now=t, depth_ts=turn_end)
        assert d.state == WANDER_VERIFY
    for i in range(3):
        t += 0.2
        dts = turn_end + 0.05 + i * 0.1
        d = pol.decide(_grid(0.10), _tel(yaw=50), now=t, depth_ts=dts)
    assert d.state == WANDER_CRUISE


def test_verify_timeout_retries_same_direction():
    pol = _policy(
        turn_confirm_frames=1,
        turn_min_deg=30,
        turn_max_deg=30,
        verify_timeout_s=1.0,
        verify_frames=10,
        segment_s_min=50,
        segment_s_max=50,
    )
    t, dts = 1.0, 1.0
    pol.decide(_grid(0.2), _tel(yaw=0), now=t, depth_ts=dts)
    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70, left=0.1, right=0.6), _tel(yaw=0), now=t, depth_ts=dts)
    assert d.state == WANDER_TURN
    turn_dir_yaw = d.axes.yaw
    # 左转：yaw 负向
    for yaw in (-15, -30, -40):
        t += 0.2
        dts += 0.2
        d = pol.decide(_wall(0.70, left=0.1, right=0.6), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_VERIFY
    t += 1.1
    dts += 1.1
    d = pol.decide(_wall(0.55), _tel(yaw=40), now=t, depth_ts=dts)
    assert d.state == WANDER_TURN
    assert np.sign(d.axes.yaw) == np.sign(turn_dir_yaw)
    assert "retry" in d.event


def test_danger_hold_retreat_abort():
    pol = _policy(danger_hold_s=0.3, retreat_s=0.4, danger_thresh=0.78)
    t, dts = 1.0, 1.0
    pol.decide(_grid(0.2), _tel(), now=t, depth_ts=dts)
    t += 0.1
    dts += 0.1
    d = pol.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)
    assert d.state == DANGER_HOLD
    t += 0.1
    dts += 0.1
    d = pol.decide(_grid(0.2), _tel(), now=t, depth_ts=dts)
    assert d.state == WANDER_CRUISE
    t += 0.1
    dts += 0.1
    pol.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)
    t += 0.35
    dts += 0.35
    d = pol.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)
    assert d.state == WANDER_RETREAT
    t += 0.45
    dts += 0.45
    d = pol.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)
    assert d.abort_reason == ""
    t += 0.1
    dts = t + 0.01
    d = pol.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)
    assert d.abort_reason == "wander_danger"


def test_pano_enters_verify_not_cruise():
    """P0-1: PANO seek 完成后必须进 VERIFY，旧深度不放行。"""
    pol = _policy(
        turn_confirm_frames=1,
        corner_max_turns=1,
        segment_s_min=50,
        segment_s_max=50,
        pano_complete_deg=40.0,
        pano_min_s=0.0,
        pano_step_deadband_deg=0.0,
        verify_frames=2,
        clear_thresh=0.40,
    )
    t, dts, yaw = 1.0, 1.0, 0.0
    pol.decide(_grid(0.2), _tel(yaw=yaw), now=t, depth_ts=dts)
    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_PANO
    # scan
    for i in range(10):
        yaw = (yaw + 15) % 360
        t += 0.3
        dts += 0.3
        d = pol.decide(_grid(0.10 + i * 0.01), _tel(yaw=yaw), now=t, depth_ts=dts)
        if pol._pano_phase == "seek":
            break
    assert pol._pano_phase == "seek"
    # seek 到位
    assert pol._pano_target_yaw is not None
    yaw = pol._pano_target_yaw
    t += 0.2
    dts += 0.2
    d = pol.decide(_grid(0.10), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_VERIFY
    assert d.axes.pitch == 0
    turn_end = d.turn_end_ts
    for _ in range(3):
        t += 0.2
        d = pol.decide(_grid(0.10), _tel(yaw=yaw), now=t, depth_ts=turn_end)
        assert d.state == WANDER_VERIFY
        assert d.axes.pitch == 0
    for i in range(3):
        t += 0.2
        dts = turn_end + 0.05 + i * 0.1
        d = pol.decide(_grid(0.10), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_CRUISE


def test_dead_reckon_turn_not_credited_during_danger_hold():
    """P0-2: DANGER 暂停不计入 dead-reckon 转角。"""
    pol = _policy(
        turn_confirm_frames=1,
        turn_min_deg=120,
        turn_max_deg=120,
        yaw_speed=40,
        yaw_dead_reckon_dps_per_unit=1.0,
        danger_hold_s=0.4,
        segment_s_min=50,
        segment_s_max=50,
    )
    t = 10.0
    pol.decide(_grid(0.2), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
    t += 0.2
    d = pol.decide(_wall(0.70), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
    assert d.state == WANDER_TURN
    t += 0.2
    d = pol.decide(_grid(0.90), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
    assert d.state == DANGER_HOLD
    t += 5.0
    d = pol.decide(_grid(0.20), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
    assert d.state == WANDER_TURN
    t += 0.04
    d = pol.decide(_grid(0.20), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
    assert d.state == WANDER_TURN
    # 继续发杆直到真实累计够 120°（40 deg/s → 3s）
    for _ in range(80):
        t += 0.05
        d = pol.decide(_grid(0.20), {"bat": "80", "h": "120"}, now=t, depth_ts=t)
        if d.state == WANDER_VERIFY:
            break
    assert d.state == WANDER_VERIFY
    assert pol._turn_yaw_elapsed >= 120.0 / 40.0 - 0.1


def test_corner_pano_then_abort():
    """P1-7: 默认 corner_window_s=20；PANO 完成后窗口内再遇障 abort。"""
    pol = _policy(
        turn_confirm_frames=1,
        corner_max_turns=2,
        corner_window_s=20.0,
        turn_min_deg=20,
        turn_max_deg=20,
        turn_arrive_tol_deg=5,
        verify_frames=2,
        segment_s_min=50,
        segment_s_max=50,
        pano_complete_deg=40.0,
        pano_min_s=0.0,
        pano_step_deadband_deg=0.0,
    )
    t, dts, yaw = 1.0, 1.0, 0.0
    pol.decide(_grid(0.2), _tel(yaw=yaw), now=t, depth_ts=dts)

    # 第 1 次正常转向 + VERIFY
    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70, left=0.55, right=0.15), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_TURN
    yaw = yaw + 25
    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70, left=0.55, right=0.15), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_VERIFY
    for _ in range(3):
        t += 0.2
        dts += 0.2
        d = pol.decide(_grid(0.15), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_CRUISE

    # 第 2 次 → PANO
    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_PANO
    for i in range(40):
        if pol._pano_phase == "seek" and pol._pano_target_yaw is not None:
            yaw = pol._pano_target_yaw
        else:
            yaw = (yaw + 15) % 360
        t += 0.4
        dts += 0.4
        d = pol.decide(_grid(0.10), _tel(yaw=yaw), now=t, depth_ts=dts)
        if d.state == WANDER_VERIFY:
            break
    assert d.state == WANDER_VERIFY
    for i in range(3):
        t += 0.2
        dts += 0.2
        d = pol.decide(_grid(0.10), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_CRUISE

    # 窗口内再遇障 → abort
    t += 1.0
    dts += 1.0
    d = pol.decide(_wall(0.70), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.abort_reason == "wander_cornered"


def test_corner_abort_expires_after_window():
    pol = _policy(
        turn_confirm_frames=1,
        corner_max_turns=99,
        corner_window_s=5.0,
        segment_s_min=50,
        segment_s_max=50,
    )
    pol._pano_done_at = 100.0
    d = pol._start_obstacle_turn((0.2, 0.7, 0.5), now=104.0)
    assert d.abort_reason == "wander_cornered"
    pol2 = _policy(turn_confirm_frames=1, corner_max_turns=99, corner_window_s=5.0,
                   segment_s_min=50, segment_s_max=50)
    pol2._pano_done_at = 100.0
    d2 = pol2._start_obstacle_turn((0.2, 0.7, 0.5), now=106.0)
    assert d2.abort_reason == ""
    assert d2.state == WANDER_TURN


def test_pano_travel_ignores_yaw_noise():
    """P1-6: 悬停噪声不得累计假行程。"""
    pol = _policy(pano_complete_deg=340.0, pano_step_deadband_deg=0.5)
    pol._state = WANDER_PANO
    pol._pano_phase = "scan"
    pol._pano_start_t = None
    pol._pano_start_yaw = None
    rng = np.random.default_rng(0)
    t = 0.0
    for _ in range(300):
        t += 1.0 / 24.0
        noisy = float(rng.normal(0.0, 0.5))
        pol.decide(_grid(0.15), _tel(yaw=noisy), now=t, depth_ts=t)
    assert abs(pol._pano_travel_deg) < 30.0
    assert pol._state == WANDER_PANO


def test_pano_real_rotation_completes():
    pol = _policy(
        pano_complete_deg=340.0,
        pano_min_s=0.5,
        pano_step_deadband_deg=0.5,
        corner_max_turns=1,
        turn_confirm_frames=1,
        segment_s_min=50,
        segment_s_max=50,
    )
    t, dts, yaw = 1.0, 1.0, 0.0
    pol.decide(_grid(0.2), _tel(yaw=yaw), now=t, depth_ts=dts)
    t += 0.2
    dts += 0.2
    pol.decide(_wall(0.70), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert pol._state == WANDER_PANO
    for i in range(40):
        yaw = (yaw + 12) % 360
        t += 0.4
        dts += 0.4
        d = pol.decide(_grid(0.12), _tel(yaw=yaw), now=t, depth_ts=dts)
        if pol._pano_phase == "seek" or d.state == WANDER_VERIFY:
            break
    assert pol._pano_phase == "seek" or d.state == WANDER_VERIFY


def test_height_band_corrective_and_missing():
    """P1-4 / P1-5: 单向回带；缺失按时间闩锁。"""
    pol = WanderPolicy(
        WanderParams(
            seed=7,
            alt_change_prob=1.0,
            alt_throttle=25,
            alt_segment_s_min=5,
            alt_segment_s_max=5,
            segment_s_min=10,
            segment_s_max=10,
            free_turn_prob=0.0,
            h_min_cm=80,
            h_max_cm=200,
            h_missing_s=1.0,
            clear_thresh=0.40,
        ),
        seed=7,
    )
    t, dts = 1.0, 1.0
    pol.decide(_grid(0.15), _tel(h=120), now=t, depth_ts=dts)
    pol._alt_throttle = 25
    pol._alt_until = t + 20
    # 带内透传
    d = pol.decide(_grid(0.15), _tel(h=120), now=t + 0.1, depth_ts=dts + 0.1)
    assert d.axes.throttle == 25
    # 跌破下界：允许爬升
    d = pol.decide(_grid(0.15), _tel(h=60), now=t + 0.2, depth_ts=dts + 0.2)
    assert d.axes.throttle == 25
    # 跌破下界：禁止下降
    pol._alt_throttle = -25
    d = pol.decide(_grid(0.15), _tel(h=60), now=t + 0.3, depth_ts=dts + 0.3)
    assert d.axes.throttle == 0
    # 超上界：禁止爬升，允许下降
    pol._alt_throttle = 25
    d = pol.decide(_grid(0.15), _tel(h=250), now=t + 0.4, depth_ts=dts + 0.4)
    assert d.axes.throttle == 0
    pol._alt_throttle = -25
    d = pol.decide(_grid(0.15), _tel(h=250), now=t + 0.5, depth_ts=dts + 0.5)
    assert d.axes.throttle == -25

    # 缺失 h：短于 h_missing_s 不闩
    pol2 = WanderPolicy(
        WanderParams(seed=8, alt_change_prob=1.0, h_missing_s=1.0, free_turn_prob=0.0,
                     segment_s_min=50, segment_s_max=50),
        seed=8,
    )
    pol2.decide(_grid(0.15), _tel(h=120), now=1.0, depth_ts=1.0)
    pol2._alt_throttle = 25
    pol2._alt_until = 100
    t = 1.0
    for i in range(10):  # 10/24 ≈ 0.42s < 1.0
        t += 1.0 / 24.0
        pol2.decide(_grid(0.15), {"bat": "80", "yaw": "0"}, now=t, depth_ts=1.0)
    assert pol2._h_latch_zero is False
    # 拉长到 >= 1s
    while t < 2.2:
        t += 1.0 / 24.0
        pol2.decide(_grid(0.15), {"bat": "80", "yaw": "0"}, now=t, depth_ts=1.0)
    assert pol2._h_latch_zero is True
    # 刚恢复瞬间仍闩（需连续读满 h_missing_s）
    d = pol2.decide(_grid(0.15), _tel(h=120), now=t + 0.1, depth_ts=t + 0.1)
    assert d.axes.throttle == 0
    # 连续正常满 1s → 解除闩锁，高度控制恢复
    t_rec = t + 0.1
    for _ in range(30):
        t_rec += 1.0 / 24.0
        d = pol2.decide(_grid(0.15), _tel(h=120), now=t_rec, depth_ts=t_rec)
    assert pol2._h_latch_zero is False
    assert d.axes.throttle == 25


def test_seed_reproducible_decision_stream():
    params = dict(
        seed=123,
        free_turn_prob=0.0,
        alt_change_prob=0.0,
        turn_confirm_frames=1,
        segment_s_min=2.0,
        segment_s_max=2.0,
        turn_min_deg=50,
        turn_max_deg=50,
    )

    def run():
        pol = WanderPolicy(WanderParams(**params), seed=123)
        out = []
        t, dts, yaw = 0.0, 0.0, 0.0
        for i in range(40):
            near = _wall(0.70) if 10 <= i < 14 else _grid(0.15)
            if i >= 14:
                yaw = min(80, yaw + 20)
            d = pol.decide(near, _tel(yaw=yaw), now=t, depth_ts=dts)
            out.append((d.state, d.axes.as_tuple(), d.event))
            t += 0.25
            dts += 0.25
        return out

    assert run() == run()


def test_now_zero_does_not_false_timeout():
    pol = _policy(verify_timeout_s=3.0, danger_hold_s=0.4, segment_s_min=5, segment_s_max=5)
    d = pol.decide(_grid(0.2), _tel(), now=0.0, depth_ts=0.0)
    assert d.abort_reason == ""
    assert d.state == WANDER_CRUISE


def test_pano_now_zero_keeps_start():
    pol = _policy(corner_max_turns=1, turn_confirm_frames=1, segment_s_min=50, segment_s_max=50)
    pol.decide(_grid(0.2), _tel(), now=0.0, depth_ts=0.0)
    d = pol.decide(_wall(0.70), _tel(), now=0.0, depth_ts=0.1)
    assert d.state == WANDER_PANO
    assert pol._pano_start_t == 0.0


def test_turn_arrival_requires_commanded_direction():
    """P2-8: 反方向转不算到位。"""
    pol = _policy(
        turn_confirm_frames=1,
        turn_min_deg=60,
        turn_max_deg=60,
        turn_arrive_tol_deg=5,
        turn_open_bias=1.0,
        segment_s_min=50,
        segment_s_max=50,
    )
    t, dts = 1.0, 1.0
    pol.decide(_grid(0.2), _tel(yaw=0), now=t, depth_ts=dts)
    t += 0.2
    dts += 0.2
    # 左开阔 → 左转（yaw-）
    d = pol.decide(_wall(0.70, left=0.1, right=0.6), _tel(yaw=0), now=t, depth_ts=dts)
    assert d.state == WANDER_TURN
    assert d.axes.yaw < 0
    # 故意向右转 70° —— 不应到位
    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70), _tel(yaw=70), now=t, depth_ts=dts)
    assert d.state == WANDER_TURN
    # 向左转 70° —— 到位
    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70), _tel(yaw=-70), now=t, depth_ts=dts)
    assert d.state == WANDER_VERIFY


def test_wander_orbit_mutex_raises():
    with pytest.raises(ValueError, match="互斥"):
        AvoidanceFSM(
            controller=AvoidanceController(),
            params=FsmParams(wander_mode=True, orbit_mode=True),
        )


def test_fsm_wander_branch_smoke():
    wander = _policy(segment_s_min=5, segment_s_max=5)
    fsm = AvoidanceFSM(
        controller=AvoidanceController(),
        params=FsmParams(
            wander_mode=True,
            orbit_mode=False,
            max_auto_engaged_s=120,
            depth_stale_s=20,
            min_battery_pct=5,
            max_height_cm=350,
        ),
        wander=wander,
    )
    t = 1000.0
    for i in range(5):
        d = fsm.step(_grid(0.2), _tel(), True, now=t + i * 0.2, depth_ts=t + i * 0.2)
        assert d.abort_reason == ""
        assert d.axes.roll == 0


def test_sim_smoke_ten_minutes_accelerated():
    """§9.2: 固定 seed 加速 10 分钟；转向>0；无卡死；无异常 abort。"""
    pol = WanderPolicy(
        WanderParams(
            seed=99,
            free_turn_prob=0.3,
            alt_change_prob=0.0,
            turn_confirm_frames=1,
            segment_s_min=2.0,
            segment_s_max=4.0,
            turn_min_deg=40,
            turn_max_deg=60,
            verify_frames=2,
            corner_max_turns=8,
            corner_window_s=30.0,
        ),
        seed=99,
    )
    dt = 0.2
    steps = int(600.0 / dt)  # 10 min
    yaw = 0.0
    t = 0.0
    turns = 0
    zero_run = 0.0
    max_zero_run = 0.0
    last_state = ""
    abort = ""

    for i in range(steps):
        # 简单「走廊」：每 ~8s 前方变堵一阵
        phase = (i * dt) % 12.0
        if 6.0 <= phase < 8.0:
            near = _wall(0.70, left=0.2, right=0.5)
        else:
            near = _grid(0.18)
        d = pol.decide(near, _tel(h=120, yaw=yaw), now=t, depth_ts=t)
        if d.event.startswith("TURN("):
            turns += 1
        if d.abort_reason:
            abort = d.abort_reason
            break
        # 模拟遥测 yaw：有 yaw 杆则转动
        yaw = (yaw + d.axes.yaw * dt * 1.0) % 360.0
        idle = d.axes.is_zero()
        holding = d.state in (DANGER_HOLD, WANDER_VERIFY)
        if idle and not holding:
            zero_run += dt
            max_zero_run = max(max_zero_run, zero_run)
        else:
            zero_run = 0.0
        last_state = d.state
        t += dt

    assert abort in ("", "wander_cornered", "wander_danger")
    assert turns > 0, f"expected turns>0, last={last_state}"
    assert max_zero_run < 30.0, f"stuck zero sticks for {max_zero_run:.1f}s"
