"""POI 环绕 OrbitController / FSM 安全路径回归（合成近度图）。"""
import numpy as np

from tt_control.avoidance import OrbitController, OrbitParams
from tt_control.avoidance_fsm import AvoidanceFSM, AvoidState, FsmParams
from tt_control.avoidance import AvoidanceController, AvoidParams


def _grid(val, shape=(96, 128)):
    return np.full(shape, val, dtype=np.float32)


def _chair(mid_val=0.55, x0=45, x1=95, bg=0.08, shape=(96, 128)):
    n = _grid(bg, shape)
    n[:, x0:x1] = mid_val
    return n


def _tel(bat=80, h=80, yaw=0):
    return {"bat": str(bat), "h": str(h), "yaw": str(yaw)}


def test_lost_zeros_sticks():
    c = OrbitController()
    d = c.decide(_grid(0.05))
    assert d.state == "LOST"
    assert d.axes.is_zero()


def test_uniform_field_rejects_false_poi():
    """均匀近度不得当成椅子居中横移。"""
    c = OrbitController()
    d = c.decide(_grid(0.25))
    assert d.chair_pos is None
    assert d.state == "LOST"
    assert d.axes.is_zero()


def test_search_no_blind_pitch():
    """无可靠目标时不得按 mid 盲飞前进。"""
    c = OrbitController()
    n = _grid(0.09)
    d = c.decide(n)
    assert d.chair_pos is None
    assert d.state == "LOST"
    assert d.axes.pitch == 0


def test_danger_mid_zeros_sticks():
    c = OrbitController()
    n = _chair(0.85, x0=40, x1=90)
    d = c.decide(n)
    assert d.state == "DANGER"
    assert d.axes.is_zero()


def test_danger_side_zeros_sticks():
    """侧区过近也 DANGER（横移主碰撞面）。"""
    c = OrbitController(OrbitParams(danger_thresh=0.78))
    n = _grid(0.20)
    n[:, : n.shape[1] // 3] = 0.85  # 左区危险，中区仍低
    d = c.decide(n)
    assert d.state == "DANGER"
    assert d.axes.is_zero()


def test_orbit_strafe_when_centered():
    c = OrbitController()
    n = _chair(0.55, x0=50, x1=78)  # 偏中
    d = c.decide(n)
    assert d.state == "ORBIT"
    assert d.orbit_phase in ("strafe", "center")
    if d.orbit_phase == "strafe":
        assert d.axes.roll == OrbitParams().orbit_roll


def test_fsm_orbit_danger_aborts_auto():
    ctrl = AvoidanceController(AvoidParams(clear_thresh=0.35, cruise_speed=30))
    fsm = AvoidanceFSM(
        controller=ctrl,
        params=FsmParams(
            orbit_mode=True,
            orbit_enter_nearness=0.35,
            max_approach_s=60,
            max_auto_engaged_s=120,
            depth_stale_s=20,
            min_battery_pct=5,
        ),
    )
    t = 1000.0
    r = fsm.step(_chair(0.55), _tel(), True, now=t)
    assert fsm._orbit_active
    danger = _chair(0.85)
    r2 = fsm.step(danger, _tel(), True, now=t + 0.1)
    assert r2.state == AvoidState.HOVER
    assert r2.abort_reason == "orbit_danger"
    assert r2.axes.is_zero()


def test_fsm_orbit_lost_grace_zero_then_abort():
    ctrl = AvoidanceController(AvoidParams(clear_thresh=0.35, cruise_speed=30))
    fsm = AvoidanceFSM(
        controller=ctrl,
        params=FsmParams(
            orbit_mode=True,
            orbit_enter_nearness=0.35,
            orbit_lost_timeout_s=1.0,
            max_approach_s=60,
            max_auto_engaged_s=120,
            depth_stale_s=20,
            min_battery_pct=5,
        ),
    )
    t = 2000.0
    fsm.step(_chair(0.55), _tel(), True, now=t)
    assert fsm._orbit_active
    lost = _grid(0.04)
    r1 = fsm.step(lost, _tel(), True, now=t + 0.1)
    assert "LOST" in r1.sub_state
    assert r1.axes.is_zero()
    assert r1.abort_reason == ""
    r2 = fsm.step(lost, _tel(), True, now=t + 1.2)
    assert r2.state == AvoidState.HOVER
    assert r2.abort_reason == "orbit_lost"


def test_fsm_reset_clears_orbit_latch():
    ctrl = AvoidanceController(AvoidParams(clear_thresh=0.35))
    fsm = AvoidanceFSM(
        controller=ctrl,
        params=FsmParams(
            orbit_mode=True,
            orbit_enter_nearness=0.35,
            max_approach_s=60,
            max_auto_engaged_s=120,
            depth_stale_s=20,
            min_battery_pct=5,
        ),
    )
    t = 3000.0
    fsm.step(_chair(0.55), _tel(), True, now=t)
    assert fsm._orbit_active
    fsm.reset()
    assert not fsm._orbit_active
    assert fsm._auto_engaged is None
    # 再挂应从干净 APPROACH 起步，不立刻 orbit（需 mid 再触发）
    far = _grid(0.10)
    r = fsm.step(far, _tel(), True, now=t + 1.0)
    assert r.state == AvoidState.APPROACH
    assert not fsm._orbit_active


def test_offcenter_target_uses_chair_column_not_mid():
    """椅子偏一侧时，距离应用椅子列近度，不能因 mid=背景而猛冲。"""
    c = OrbitController()
    n = _grid(0.08)
    # 椅子在最右侧，中区仍是背景
    n[:, 100:125] = 0.55
    d = c.decide(n)
    assert d.chair_pos is not None and d.chair_pos > 0.3
    # 目标近度≈0.55≈target，pitch 应接近 0（允许微调）
    assert abs(d.axes.pitch) <= 10


def test_fsm_approach_holds_on_side_danger():
    """进环绕前侧障危险时不得强行 cruise 前冲。"""
    ctrl = AvoidanceController(AvoidParams(clear_thresh=0.35, estop_thresh=0.82, cruise_speed=30))
    fsm = AvoidanceFSM(
        controller=ctrl,
        params=FsmParams(
            orbit_mode=True,
            orbit_enter_nearness=0.50,  # mid 未达进入阈值
            max_approach_s=60,
            max_auto_engaged_s=120,
            depth_stale_s=20,
            min_battery_pct=5,
        ),
    )
    n = _grid(0.10)
    n[:, : n.shape[1] // 3] = 0.90  # 左区危险，中区仍低
    r = fsm.step(n, _tel(), True, now=50.0)
    assert not fsm._orbit_active
    assert r.axes.is_zero()
    assert "approach_hold" in r.sub_state


def test_fsm_approach_holds_at_orbit_danger_band():
    """侧区落在 orbit danger(0.78) 与 estop(0.82) 之间也必须 hold。"""
    ctrl = AvoidanceController(AvoidParams(clear_thresh=0.35, estop_thresh=0.82, cruise_speed=30))
    fsm = AvoidanceFSM(
        controller=ctrl,
        params=FsmParams(
            orbit_mode=True,
            orbit_enter_nearness=0.50,
            max_approach_s=60,
            max_auto_engaged_s=120,
            depth_stale_s=20,
            min_battery_pct=5,
        ),
    )
    n = _grid(0.10)
    n[:, : n.shape[1] // 3] = 0.79  # 旧逻辑会 cruise，新逻辑必须 hold
    r = fsm.step(n, _tel(), True, now=50.0)
    assert r.axes.is_zero()
    assert "approach_hold" in r.sub_state


def test_fsm_reset_clears_depth_ts():
    ctrl = AvoidanceController()
    fsm = AvoidanceFSM(
        controller=ctrl,
        params=FsmParams(orbit_mode=True, depth_stale_s=3.0, min_battery_pct=5),
    )
    fsm.step(_grid(0.10), _tel(), True, now=10.0)
    assert fsm._last_depth_ts > 0
    fsm.reset()
    assert fsm._last_depth_ts == 0.0


def test_fsm_now_zero_does_not_clear_orbit_latch():
    """回归：_auto_engaged 不可用 0.0 哨兵，否则 now=0 仿真每帧清 orbit → 退回 CRUISE 前冲。"""
    ctrl = AvoidanceController(AvoidParams(clear_thresh=0.35))
    fsm = AvoidanceFSM(
        controller=ctrl,
        params=FsmParams(
            orbit_mode=True,
            orbit_enter_nearness=0.35,
            orbit_lost_timeout_s=1.0,
            max_approach_s=60,
            max_auto_engaged_s=120,
            depth_stale_s=20,
            min_battery_pct=5,
        ),
    )
    t = 0.0
    fsm.step(_chair(0.55), _tel(), True, now=t)
    assert fsm._orbit_active
    assert fsm._auto_engaged == 0.0  # 合法挂载时刻可以是 0
    r = fsm.step(_grid(0.04), _tel(), True, now=t + 0.1)
    assert fsm._orbit_active
    assert "LOST" in r.sub_state
    assert r.axes.is_zero()
    assert r.abort_reason == ""
    r2 = fsm.step(_grid(0.04), _tel(), True, now=t + 1.2)
    assert r2.abort_reason == "orbit_lost"
