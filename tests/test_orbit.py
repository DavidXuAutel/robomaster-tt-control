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


def _decide_locked(ctrl: OrbitController, nearness, frames: int | None = None):
    """跑满 acquire_frames，返回最后一帧决策。"""
    n = frames if frames is not None else ctrl.p.acquire_frames
    d = None
    for _ in range(n):
        d = ctrl.decide(nearness)
    return d


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
    d = _decide_locked(c, n)
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
    d = _decide_locked(c, n)
    assert d.state == "ORBIT"
    assert d.orbit_phase in ("strafe", "center")
    if d.orbit_phase == "strafe":
        assert d.axes.roll == OrbitParams().orbit_roll


def test_acquire_is_not_lost_episode():
    """首帧获取中应为 ACQUIRE，不得记成 LOST。"""
    c = OrbitController(OrbitParams(acquire_frames=3))
    d = c.decide(_chair(0.55, x0=50, x1=78))
    assert d.state == "ACQUIRE"
    assert d.axes.is_zero()
    assert d.reject_reason.startswith("acquiring")


def test_broad_target_tracks_after_acquire():
    """宽目标获取后用 ROI 跟踪，不应因峰均比下降误 LOST。"""
    c = OrbitController(OrbitParams(acquire_frames=2, track_peak_ratio=1.35))
    peaked = _chair(0.64, x0=50, x1=78)
    d0 = _decide_locked(c, peaked)
    assert d0.state == "ORBIT"
    assert c._tracked
    # 变宽：全图峰均比会 <2，但跟踪 ROI 应仍锁住
    broad = _chair(0.64, x0=30, x1=100)
    d1 = c.decide(broad)
    assert d1.state == "ORBIT"
    assert d1.chair_pos is not None


def test_pre_orbit_no_latch_keeps_approach():
    """mid 过进入阈值但目标未锁定时，不得 latch，应继续靠近。"""
    ctrl = AvoidanceController(AvoidParams(clear_thresh=0.35, approach_pitch=25, cruise_speed=30))
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
    # 均匀近场：M 高但峰均比不够 → 拒识
    flat = _grid(0.40)
    r = fsm.step(flat, _tel(), True, now=5000.0)
    assert not fsm._orbit_active
    assert r.abort_reason == ""
    assert r.axes.pitch == 25
    assert "pre_orbit" in r.sub_state
    # 再喂几帧仍不应 orbit_lost
    r2 = fsm.step(flat, _tel(), True, now=5001.5)
    assert not fsm._orbit_active
    assert r2.abort_reason == ""
    assert r2.axes.pitch == 25


def test_pre_orbit_latches_after_lock():
    """锁定成功后才 latch，随后可环绕。"""
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
    t = 5100.0
    for i in range(3):
        r = fsm.step(_chair(0.55, x0=50, x1=78), _tel(), True, now=t + i * 0.05)
    assert fsm._orbit_active
    assert "ORBIT" in r.sub_state or r.axes.roll != 0 or r.axes.yaw != 0 or r.axes.pitch != 0


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
    # acquire_frames=2：多喂几帧完成锁定
    for i in range(3):
        fsm.step(_chair(0.55), _tel(), True, now=t + i * 0.05)
    assert fsm._orbit_active
    danger = _chair(0.85)
    r2 = fsm.step(danger, _tel(), True, now=t + 0.2)
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
    for i in range(3):
        fsm.step(_chair(0.55), _tel(), True, now=t + i * 0.05)
    assert fsm._orbit_active
    lost = _grid(0.04)
    r1 = fsm.step(lost, _tel(), True, now=t + 0.2)
    assert "LOST" in r1.sub_state
    assert r1.axes.is_zero()
    assert r1.abort_reason == ""
    r2 = fsm.step(lost, _tel(), True, now=t + 1.3)
    assert r2.state == AvoidState.HOVER
    assert r2.abort_reason == "orbit_lost"


def test_fsm_lost_flicker_does_not_reset_timeout():
    """偶发一帧检出不得清零 LOST episode；超时仍 abort。"""
    ctrl = AvoidanceController(AvoidParams(clear_thresh=0.35, cruise_speed=30))
    fsm = AvoidanceFSM(
        controller=ctrl,
        params=FsmParams(
            orbit_mode=True,
            orbit_enter_nearness=0.35,
            orbit_lost_timeout_s=1.0,
            orbit_relock_frames=5,
            max_approach_s=60,
            max_auto_engaged_s=120,
            depth_stale_s=20,
            min_battery_pct=5,
        ),
    )
    t = 4000.0
    for i in range(3):
        fsm.step(_chair(0.55), _tel(), True, now=t + i * 0.05)
    assert fsm._orbit_active
    lost = _grid(0.04)
    fsm.step(lost, _tel(), True, now=t + 0.2)
    # 中间闪一帧有效 → 应进入 reacq，零杆，且不清除 episode
    r_flash = fsm.step(_chair(0.55), _tel(), True, now=t + 0.4)
    assert r_flash.abort_reason == ""
    assert r_flash.axes.is_zero()
    assert fsm._orbit_lost_since is not None
    assert "reacq" in r_flash.sub_state or "LOST" in r_flash.sub_state
    # 再失锁，总 episode >1s → abort
    r_abort = fsm.step(lost, _tel(), True, now=t + 1.3)
    assert r_abort.abort_reason == "orbit_lost"


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
    for i in range(3):
        fsm.step(_chair(0.55), _tel(), True, now=t + i * 0.05)
    assert fsm._orbit_active
    fsm.reset()
    assert not fsm._orbit_active
    assert fsm._auto_engaged is None
    far = _grid(0.10)
    r = fsm.step(far, _tel(), True, now=t + 1.0)
    assert r.state == AvoidState.APPROACH
    assert not fsm._orbit_active


def test_offcenter_target_uses_chair_column_not_mid():
    """椅子偏一侧时，距离应用椅子列近度，不能因 mid=背景而猛冲。"""
    c = OrbitController(OrbitParams(target_nearness=0.55, acquire_frames=2))
    n = _grid(0.08)
    n[:, 100:125] = 0.55
    d = _decide_locked(c, n)
    assert d.chair_pos is not None and d.chair_pos > 0.3
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
    n[:, : n.shape[1] // 3] = 0.79
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
    for i in range(3):
        fsm.step(_chair(0.55), _tel(), True, now=t + i * 0.05)
    assert fsm._orbit_active
    assert fsm._auto_engaged == 0.0
    r = fsm.step(_grid(0.04), _tel(), True, now=t + 0.2)
    assert fsm._orbit_active
    assert "LOST" in r.sub_state
    assert r.axes.is_zero()
    assert r.abort_reason == ""
    r2 = fsm.step(_grid(0.04), _tel(), True, now=t + 1.3)
    assert r2.abort_reason == "orbit_lost"
