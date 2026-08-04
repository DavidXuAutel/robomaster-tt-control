"""M11：DogControlArbiter 状态机与安全不变量（方案 §4.5 的 I1–I10）。

这组测试是「两个通道绝不同时下发速度」的软件证明。任何一条红失败都意味着
真机上可能出现双源控制 —— 直接摔狗，不许 xfail、不许放宽。
"""

import time

import pytest

from adapters.dog_arbiter import (
    PREFLIGHT_BATTERY_LOW,
    PREFLIGHT_CONFIDENCE_LOW,
    PREFLIGHT_CONFIDENCE_UNAVAILABLE,
    PREFLIGHT_CONTROLLER_BUSY,
    ArbiterRejected,
    ArbiterState,
    DogControlArbiter,
    LeaseOwner,
)
from adapters.dog_unitree import LoopbackTransport, SpeedLimits, UnitreeSportClient
from adapters.topsee_client import TopseeClient
from tests.fixtures.topsee_fake import FakeTopseeServer

ROBOT = "B2000397"


@pytest.fixture
def srv():
    with FakeTopseeServer() as s:
        yield s


@pytest.fixture
def client(srv):
    c = TopseeClient(
        srv.base_url, account=srv.state.account, password=srv.state.password, timeout_s=5.0
    )
    c.login()
    return c


@pytest.fixture
def unitree():
    u = UnitreeSportClient(LoopbackTransport(), limits=SpeedLimits(1.0, 0.6, 1.0))
    u.connect()
    return u


def _arb(client, unitree, **kw):
    kw.setdefault("robot_id", ROBOT)
    kw.setdefault("controller_state", "02")
    a = DogControlArbiter(client, unitree, **kw)
    return a


def _to_wam(arb):
    """走完 IDLE → … → WAM_ACTIVE，返回租约 token。"""
    arb.ack_confidence(0.95, by="tester")
    arb.acquire_for_mission("m1")
    arb.request_wam()
    return arb.ack_human_mode_switch(by="tester")


# ---------- 正常路径 ----------


def test_happy_path_reaches_wam_active(client, unitree):
    arb = _arb(client, unitree)
    token = _to_wam(arb)
    assert arb.state is ArbiterState.WAM_ACTIVE
    assert arb.owner is LeaseOwner.WAM
    assert token and arb.lease_token == token
    arb.move(0.5, 0.0, 0.2, token=token)
    assert unitree.move_calls == 1


def test_handover_releases_everything(client, unitree):
    arb = _arb(client, unitree)
    _to_wam(arb)
    arb.begin_handover_to_mission()
    assert arb.state is ArbiterState.IDLE
    assert arb.no_owner
    assert arb.lease_token is None
    assert unitree.lease_token is None


# ---------- I1：命令所有者唯一 ----------


def test_i1_channels_never_both_enabled(client, unitree):
    arb = _arb(client, unitree)
    seen = set()
    for step in (
        lambda: arb.ack_confidence(0.95, by="t"),
        lambda: arb.acquire_for_mission("m1"),
        lambda: arb.request_wam(),
        lambda: arb.ack_human_mode_switch(by="t"),
        lambda: arb.begin_handover_to_mission(),
    ):
        step()
        assert arb.has_single_owner, f"{arb.state} 下双通道同时可下发"
        assert not (arb.topsee_cmd_enabled and arb.unitree_cmd_enabled)
        seen.add(arb.state)
    assert ArbiterState.WAM_ACTIVE in seen and ArbiterState.MISSION_NAV in seen


def test_i1_mission_nav_blocks_unitree(client, unitree):
    arb = _arb(client, unitree)
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m1")
    assert arb.topsee_cmd_enabled is True
    assert arb.unitree_cmd_enabled is False
    with pytest.raises(ArbiterRejected):
        arb.move(0.3, 0.0, 0.0, token="anything")
    assert unitree.move_calls == 0


def test_i1_wam_active_blocks_topsee(client, unitree):
    arb = _arb(client, unitree)
    _to_wam(arb)
    assert arb.allow_topsee_cmd() is False


# ---------- I2：无主状态白名单 ----------


def test_i2_no_owner_only_in_idle_hold_fault(client, unitree):
    arb = _arb(client, unitree)
    assert arb.state is ArbiterState.IDLE and arb.no_owner
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m1")
    assert not arb.no_owner
    arb.safe_hold("test")
    assert arb.state is ArbiterState.SAFE_HOLD and arb.no_owner
    arb.resume_from_hold(by="t")
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m2")
    arb.fault("fall_detected")
    assert arb.state is ArbiterState.FAULT and arb.no_owner


# ---------- I3：必须经人工确认才能进 WAM ----------


def test_i3_cannot_jump_to_wam_without_waiting_state(client, unitree):
    arb = _arb(client, unitree)
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m1")
    with pytest.raises(ArbiterRejected, match="非法起始状态"):
        arb.ack_human_mode_switch(by="t")
    assert arb.state is ArbiterState.MISSION_NAV
    assert unitree.lease_token is None


def test_i3_human_ack_requires_operator_name(client, unitree):
    arb = _arb(client, unitree)
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m1")
    arb.request_wam()
    with pytest.raises(ValueError, match="确认人"):
        arb.ack_human_mode_switch(by="")


def test_i3_dds_authority_probe_failure_blocks_wam(client):
    """G11：DDS 命令是否会被边缘侧独占尚未验证，探测失败必须挡住。"""
    transport = LoopbackTransport(state_available=False)
    u = UnitreeSportClient(transport)
    u.connect()
    arb = _arb(client, u)
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m1")
    arb.request_wam()
    with pytest.raises(ArbiterRejected, match="DDS 权限探测失败"):
        arb.ack_human_mode_switch(by="t")
    assert arb.state is ArbiterState.SAFE_HOLD
    assert arb.lease_token is None
    assert arb.human_mode_ack is False


# ---------- I4/I5：租约 ----------


def test_i4_lease_revoked_on_every_exit(client, unitree):
    for exit_call in ("begin_handover_to_mission", "force_release", "safe_hold", "fault"):
        arb = _arb(client, unitree)
        token = _to_wam(arb)
        assert arb.lease_token == token
        fn = getattr(arb, exit_call)
        fn() if exit_call == "begin_handover_to_mission" else fn("reason")
        assert arb.lease_token is None, exit_call
        assert unitree.lease_token is None, exit_call


def test_i5_wrong_token_rejected_at_both_layers(client, unitree):
    arb = _arb(client, unitree)
    token = _to_wam(arb)
    with pytest.raises(ArbiterRejected, match="token"):
        arb.move(0.2, 0.0, 0.0, token=token + "x")
    assert unitree.move_calls == 0
    arb.move(0.2, 0.0, 0.0, token=token)
    assert unitree.move_calls == 1


def test_i6_lease_ttl_expiry_auto_releases(client, unitree):
    arb = _arb(client, unitree, lease_ttl_s=10.0)
    t0 = 1000.0
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m1", now=t0)
    arb.request_wam(now=t0)
    arb.ack_human_mode_switch(by="t", now=t0)
    arb.tick(now=t0 + 9.0)
    assert arb.state is ArbiterState.WAM_ACTIVE
    arb.tick(now=t0 + 10.1)
    assert arb.state is ArbiterState.IDLE
    assert arb.lease_token is None


def test_renew_lease_extends_deadline(client, unitree):
    arb = _arb(client, unitree, lease_ttl_s=10.0)
    t0 = 1000.0
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m1", now=t0)
    arb.request_wam(now=t0)
    token = arb.ack_human_mode_switch(by="t", now=t0)
    arb.renew_lease(token, now=t0 + 9.0)
    arb.tick(now=t0 + 15.0)
    assert arb.state is ArbiterState.WAM_ACTIVE


def test_renew_with_bad_token_rejected(client, unitree):
    arb = _arb(client, unitree)
    _to_wam(arb)
    with pytest.raises(ArbiterRejected):
        arb.renew_lease("nope")


# ---------- I7：命令看门狗 ----------


def test_i7_cmd_watchdog_forces_zero_speed(client, unitree):
    arb = _arb(client, unitree, cmd_watchdog_s=0.05)
    token = _to_wam(arb)
    arb.move(0.8, 0.0, 0.0, token=token)
    assert unitree.last_cmd == (0.8, 0.0, 0.0)
    time.sleep(0.12)
    arb.tick()
    assert arb.watchdog_trips == 1
    assert arb.state is ArbiterState.SAFE_HOLD
    assert unitree.last_cmd == (0.0, 0.0, 0.0)


def test_watchdog_quiet_when_last_cmd_is_zero(client, unitree):
    arb = _arb(client, unitree, cmd_watchdog_s=0.05)
    token = _to_wam(arb)
    arb.move(0.0, 0.0, 0.0, token=token)
    time.sleep(0.12)
    arb.tick()
    assert arb.watchdog_trips == 0
    assert arb.state is ArbiterState.WAM_ACTIVE


# ---------- I8：abort 路径不许失败 ----------


def test_i8_force_release_never_raises_even_when_platform_dies(srv, client, unitree):
    arb = _arb(client, unitree)
    _to_wam(arb)
    # 平台在 WAM 段中途开始全线报错，abort 仍必须走完
    srv.state.force_error["/robotTask/stopTask"] = (1, "boom")
    srv.state.force_error["/state/updateControllerUser"] = (1, "boom")
    arb.force_release("mission_abort")  # 不抛
    assert arb.state is ArbiterState.IDLE
    assert arb.no_owner and arb.lease_token is None


def test_i8_force_release_survives_dead_motion_channel(client):
    u = UnitreeSportClient(LoopbackTransport())
    u.connect()
    arb = _arb(client, u)
    token = _to_wam(arb)
    arb.move(0.5, 0, 0, token=token)
    u.transport.raise_on_call = True  # type: ignore[attr-defined]
    arb.force_release("panic")  # StopMove/Damp 都炸也不抛
    assert arb.state is ArbiterState.IDLE


# ---------- I9：平台急停不在常规路径 ----------


def test_i9_platform_estop_not_in_normal_stop_path(client, unitree):
    """F18：平台急停≈断电，会让狗直接坐下/摔，绝不能当常规停止手段。"""
    arb = _arb(client, unitree)
    assert "emergency" not in " ".join(arb.normal_stop_methods)
    assert "estop" not in " ".join(arb.normal_stop_methods)
    assert set(arb.normal_stop_methods) == {
        "unitree_stop_move",
        "unitree_damp",
        "topsee_stop_task",
    }


# ---------- I10：preflight 门禁 ----------


def test_i10_confidence_unavailable_blocks_mission(client, unitree):
    """G4：平台无置信度接口。既没 provider 又没人工确认 → 必须拒绝。"""
    arb = _arb(client, unitree)
    with pytest.raises(ArbiterRejected, match=PREFLIGHT_CONFIDENCE_UNAVAILABLE):
        arb.acquire_for_mission("m1")
    assert arb.state is ArbiterState.IDLE
    assert arb.preflight_ok is False


def test_confidence_below_gate_blocks_mission(client, unitree):
    arb = _arb(client, unitree, min_confidence=0.9)
    arb.ack_confidence(0.5, by="t")
    with pytest.raises(ArbiterRejected, match=PREFLIGHT_CONFIDENCE_LOW):
        arb.acquire_for_mission("m1")


def test_confidence_ack_expires(client, unitree):
    """长期趴下会丢定位（手册 §5.6），人工确认不能一劳永逸。"""
    arb = _arb(client, unitree, confidence_ack_ttl_s=0.05)
    arb.ack_confidence(0.99, by="t")
    time.sleep(0.12)
    with pytest.raises(ArbiterRejected, match=PREFLIGHT_CONFIDENCE_UNAVAILABLE):
        arb.acquire_for_mission("m1")


def test_confidence_ack_requires_operator(client, unitree):
    arb = _arb(client, unitree)
    with pytest.raises(ValueError, match="确认人"):
        arb.ack_confidence(0.99, by="")


def test_confidence_provider_preferred_over_manual_ack(client, unitree):
    arb = _arb(client, unitree, confidence_provider=lambda: 0.1)
    arb.ack_confidence(0.99, by="t")
    with pytest.raises(ArbiterRejected, match=PREFLIGHT_CONFIDENCE_LOW):
        arb.acquire_for_mission("m1")


def test_confidence_provider_exception_falls_back_not_passes(client, unitree):
    def boom():
        raise RuntimeError("no api")

    arb = _arb(client, unitree, confidence_provider=boom)
    with pytest.raises(ArbiterRejected, match=PREFLIGHT_CONFIDENCE_UNAVAILABLE):
        arb.acquire_for_mission("m1")


def test_battery_gate(client, unitree):
    arb = _arb(client, unitree, min_battery_pct=25.0, battery_provider=lambda: 10.0)
    arb.ack_confidence(0.95, by="t")
    with pytest.raises(ArbiterRejected, match=PREFLIGHT_BATTERY_LOW):
        arb.acquire_for_mission("m1")


def test_controller_busy_blocks_mission(srv, client, unitree):
    srv.state.controller_busy = True
    arb = _arb(client, unitree)
    arb.ack_confidence(0.95, by="t")
    with pytest.raises(ArbiterRejected, match=PREFLIGHT_CONTROLLER_BUSY):
        arb.acquire_for_mission("m1")
    assert arb.state is ArbiterState.IDLE


def test_preflight_stops_preexisting_task(srv, client, unitree):
    """避免双调度源：抢权前先把平台在跑的任务停掉。"""
    srv.state.current_task = {"pointsId": "p1", "currentState": "行进中"}
    arb = _arb(client, unitree)
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m1")
    assert srv.state.stop_calls == [ROBOT]


def test_controller_takeover_skipped_when_values_unknown(srv, client, unitree):
    """G7：state/force 取值未抓包确认时跳过抢权，但绝不假装成功。"""
    arb = DogControlArbiter(client, unitree, robot_id=ROBOT)
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m1")
    assert srv.state.controller_calls == []
    assert arb.state is ArbiterState.MISSION_NAV


# ---------- FAULT 不自动恢复 ----------


def test_fault_requires_manual_reset(client, unitree):
    """手册 §5.7：摔倒必须人工物理介入，程序不许自愈。"""
    arb = _arb(client, unitree)
    _to_wam(arb)
    arb.fault("fall_detected")
    arb.tick()
    assert arb.state is ArbiterState.FAULT
    with pytest.raises(ArbiterRejected):
        arb.acquire_for_mission("m2")
    with pytest.raises(ValueError, match="操作人"):
        arb.reset_after_fault(by="")
    arb.reset_after_fault(by="operator")
    assert arb.state is ArbiterState.IDLE


def test_reset_after_fault_only_from_fault(client, unitree):
    arb = _arb(client, unitree)
    with pytest.raises(ArbiterRejected):
        arb.reset_after_fault(by="t")


def test_request_wam_platform_failure_goes_safe_hold(srv, client, unitree):
    srv.state.force_error["/robotTask/stopTask"] = (1, "boom")
    arb = _arb(client, unitree)
    arb.ack_confidence(0.95, by="t")
    arb.acquire_for_mission("m1")
    assert arb.request_wam() is ArbiterState.SAFE_HOLD
    assert arb.no_owner


# ---------- 审计 ----------


def test_lease_log_records_grant_and_release(client, unitree):
    arb = _arb(client, unitree)
    _to_wam(arb)
    arb.begin_handover_to_mission()
    actions = [r["action"] for r in arb.lease_log]
    assert actions == ["acquire", "await_human_mode", "grant", "handover"]
    assert all("owner" in r and "state" in r for r in arb.lease_log)


def test_illegal_transitions_counted(client, unitree):
    arb = _arb(client, unitree)
    for _ in range(3):
        with pytest.raises(ArbiterRejected):
            arb.move(0.1, 0, 0, token="x")
    assert arb.illegal_transitions == 3
