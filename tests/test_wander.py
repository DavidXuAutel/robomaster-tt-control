"""WanderPolicy / FSM 漫游路径回归（合成近度图）。"""

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
    """左开阔、中堵、右中等 —— 用于断言偏向开阔侧。"""
    n = _grid(0.10, shape)
    h, w = shape
    third = w // 3
    n[:, :third] = left
    n[:, third : 2 * third] = mid_val
    n[:, 2 * third :] = right
    return n


def _tel(bat=80, h=120, yaw=0):
    return {"bat": str(bat), "h": str(h), "yaw": str(yaw)}


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
    """全状态遍历：roll≡0；pitch<0 仅 RETREAT。"""
    pol = _policy(
        turn_confirm_frames=1,
        danger_hold_s=0.1,
        retreat_s=0.5,
        verify_frames=2,
        segment_s_min=100,
        segment_s_max=100,
    )
    t = 10.0
    dts = 10.0
    seen_neg = False
    # CRUISE
    for _ in range(3):
        d = pol.decide(_grid(0.2), _tel(yaw=0), now=t, depth_ts=dts)
        assert d.axes.roll == 0
        assert d.axes.pitch >= 0 or d.state == WANDER_RETREAT
        t += 0.2
        dts += 0.2
    # 触发转向
    for _ in range(2):
        d = pol.decide(_wall(0.65), _tel(yaw=0), now=t, depth_ts=dts)
        assert d.axes.roll == 0
        if d.state == WANDER_TURN:
            assert d.axes.pitch == 0
        t += 0.2
        dts += 0.2
    # 转够 → VERIFY
    for yaw in (30, 60, 90, 120):
        d = pol.decide(_wall(0.65), _tel(yaw=yaw), now=t, depth_ts=dts)
        assert d.axes.roll == 0
        assert d.axes.pitch >= 0 or d.state == WANDER_RETREAT
        t += 0.2
        dts += 0.2
        if d.state == WANDER_VERIFY:
            break
    # danger → hold → retreat
    pol2 = _policy(danger_hold_s=0.2, retreat_s=0.4)
    t = 100.0
    dts = 100.0
    pol2.decide(_grid(0.2), _tel(), now=t, depth_ts=dts)
    t += 0.1
    dts += 0.1
    d = pol2.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)
    assert d.state == DANGER_HOLD
    assert d.axes.roll == 0 and d.axes.pitch == 0
    t += 0.25
    dts += 0.25
    d = pol2.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)
    assert d.state == WANDER_RETREAT
    assert d.axes.roll == 0
    assert d.axes.pitch < 0
    seen_neg = True
    assert seen_neg


def test_single_frame_spike_no_turn():
    pol = _policy(turn_confirm_frames=2, segment_s_min=50, segment_s_max=50)
    t, dts = 1.0, 1.0
    pol.decide(_grid(0.2), _tel(), now=t, depth_ts=dts)
    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70), _tel(), now=t, depth_ts=dts)
    assert d.state == WANDER_CRUISE  # 单帧不够
    # 同一 depth_ts 再喂（控制环重复）也不该累加
    d = pol.decide(_wall(0.70), _tel(), now=t + 0.05, depth_ts=dts)
    assert d.state == WANDER_CRUISE


def test_turn_yaw_sign_follows_open_side():
    # 左开阔 → 应偏左转（yaw-）；bias=1.0 强制开阔侧
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
    assert d.axes.yaw < 0  # 左转


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
        free_turn_prob=0.0,
    )
    t, dts = 10.0, 10.0
    pol.decide(_grid(0.2), _tel(yaw=0), now=t, depth_ts=dts)
    t += 0.2
    dts += 0.2
    pol.decide(_wall(0.70), _tel(yaw=0), now=t, depth_ts=dts)  # arm turn
    # 转够；深度 ts 故意落后于 now（模拟推理延迟）
    for yaw in (20, 40, 50):
        t += 0.2
        dts += 0.05  # depth 慢于墙钟
        d = pol.decide(_wall(0.70), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_VERIFY
    turn_end = d.turn_end_ts
    # 转向期间/结束前采集的旧深度即使 mid 开阔也不计数
    for _ in range(5):
        t += 0.2
        d = pol.decide(_grid(0.10), _tel(yaw=50), now=t, depth_ts=turn_end)
        assert d.state == WANDER_VERIFY
    # 新鲜帧才通过
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
    for yaw in (15, 30, 40):
        t += 0.2
        dts += 0.2
        d = pol.decide(_wall(0.70), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_VERIFY
    # 超时
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
    # 单帧尖刺：进 HOLD 再清除 → 不 abort
    t += 0.1
    dts += 0.1
    d = pol.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)
    assert d.state == DANGER_HOLD
    assert d.abort_reason == ""
    t += 0.1
    dts += 0.1
    d = pol.decide(_grid(0.2), _tel(), now=t, depth_ts=dts)
    assert d.state == WANDER_CRUISE
    # 持续危险（需新深度帧累加 hold）→ RETREAT → 仍危险 → abort
    t += 0.1
    dts += 0.1
    pol.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)  # enter HOLD
    t += 0.35
    dts += 0.35
    d = pol.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)  # hold 满
    assert d.state == WANDER_RETREAT
    t += 0.45
    dts += 0.45
    d = pol.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)  # retreat 结束，等新深度
    assert d.state == WANDER_RETREAT
    assert d.abort_reason == ""
    # retreat 结束后的更新深度仍危险 → abort
    t += 0.1
    dts = t + 0.01
    d = pol.decide(_grid(0.90), _tel(), now=t, depth_ts=dts)
    assert d.abort_reason == "wander_danger"


def test_corner_pano_then_abort():
    pol = _policy(
        turn_confirm_frames=1,
        corner_max_turns=4,
        corner_window_s=30.0,
        turn_min_deg=20,
        turn_max_deg=20,
        turn_arrive_tol_deg=5,
        verify_frames=2,
        segment_s_min=50,
        segment_s_max=50,
        free_turn_prob=0.0,
        turn_open_bias=1.0,
        pano_yaw_speed=30,
    )
    t, dts = 1.0, 1.0
    yaw = 0.0
    pol.decide(_grid(0.2), _tel(yaw=yaw), now=t, depth_ts=dts)

    def finish_turn_and_verify():
        nonlocal t, dts, yaw
        start = yaw
        yaw = start + 25
        t += 0.2
        dts += 0.2
        d = pol.decide(_wall(0.70), _tel(yaw=yaw), now=t, depth_ts=dts)
        assert d.state == WANDER_VERIFY
        for _ in range(3):
            t += 0.2
            dts += 0.2
            d = pol.decide(_grid(0.15), _tel(yaw=yaw), now=t, depth_ts=dts)
        assert d.state == WANDER_CRUISE

    for _ in range(3):
        t += 0.2
        dts += 0.2
        d = pol.decide(_wall(0.70), _tel(yaw=yaw), now=t, depth_ts=dts)
        assert d.state == WANDER_TURN
        finish_turn_and_verify()

    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_PANO

    pano_start = yaw
    pol._pano_start_yaw = pano_start
    pol._pano_start_t = t
    for i in range(24):
        yaw = (pano_start + (i + 1) * 15) % 360
        t += 0.5
        dts += 0.5
        d = pol.decide(_grid(0.10 + (i % 5) * 0.02), _tel(yaw=yaw), now=t, depth_ts=dts)
        if d.state != WANDER_PANO:
            break
    for _ in range(30):
        if d.state == WANDER_CRUISE:
            break
        if pol._pano_target_yaw is not None:
            yaw = pol._pano_target_yaw
        t += 0.3
        dts += 0.3
        d = pol.decide(_grid(0.15), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.state == WANDER_CRUISE

    t += 0.2
    dts += 0.2
    d = pol.decide(_wall(0.70), _tel(yaw=yaw), now=t, depth_ts=dts)
    assert d.abort_reason == "wander_cornered"


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


def test_height_band_and_missing_h_zeros_throttle():
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
            h_missing_frames=5,
            clear_thresh=0.40,
        ),
        seed=7,
    )
    t, dts = 1.0, 1.0
    # 前方开阔 + 带内 → 可能有 throttle
    d = pol.decide(_grid(0.15), _tel(h=120), now=t, depth_ts=dts)
    # 强制开启高度子段
    pol._alt_throttle = 25
    pol._alt_until = t + 10
    d = pol.decide(_grid(0.15), _tel(h=120), now=t + 0.1, depth_ts=dts + 0.1)
    assert d.axes.throttle == 25
    # 超带 → 0
    d = pol.decide(_grid(0.15), _tel(h=250), now=t + 0.2, depth_ts=dts + 0.2)
    assert d.axes.throttle == 0
    # 连续缺 h → latch 0
    pol2 = WanderPolicy(
        WanderParams(seed=8, alt_change_prob=1.0, h_missing_frames=5, free_turn_prob=0.0),
        seed=8,
    )
    t = 1.0
    pol2._alt_throttle = 25
    pol2._alt_until = 100
    for i in range(5):
        d = pol2.decide(_grid(0.15), {"bat": "80", "yaw": "0"}, now=t + i, depth_ts=t + i)
    assert d.axes.throttle == 0
    # latch 后即使 h 恢复也保持 0
    d = pol2.decide(_grid(0.15), _tel(h=120), now=t + 10, depth_ts=t + 10)
    assert d.axes.throttle == 0


def test_now_zero_does_not_false_timeout():
    pol = _policy(verify_timeout_s=3.0, danger_hold_s=0.4, segment_s_min=5, segment_s_max=5)
    d = pol.decide(_grid(0.2), _tel(), now=0.0, depth_ts=0.0)
    assert d.abort_reason == ""
    assert d.state == WANDER_CRUISE


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
        assert "WANDER" in d.sub_state or d.sub_state == "wait_depth"
