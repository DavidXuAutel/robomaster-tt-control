"""M7c/M9：TopseeNav 三态到点、TopseePerception、TopseeGas。

最重要的一组是「E1 失败特征」：派单后平台查不到任务时，必须通过
poll_fault() 如实报 nav_task_not_tracked，而不是静默返回 False 让
supervisor 的 stage_timeout_s 把它掩盖成「狗走得慢」。
"""

import time

import pytest

from adapters.dog_topsee import (
    FAULT_GOAL_UNRESOLVED,
    FAULT_NAV_TIMEOUT,
    FAULT_SEND_REJECTED,
    FAULT_STATUS_UNRECOGNIZED,
    FAULT_TASK_NOT_TRACKED,
    NavStatus,
    TopseeGas,
    TopseeNav,
    TopseePerception,
)
from adapters.gas_ledger import REASON_SOURCE_UNAVAILABLE, GasCalibrationLedger
from adapters.topsee_client import TopseeClient
from tests.fixtures.topsee_fake import FakeTopseeServer

ROBOT = "B2000397"
PID = "快速打点-1785465994716"


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


def _nav(client, **kw):
    kw.setdefault("robot_id", ROBOT)
    kw.setdefault("autostart_poller", False)
    return TopseeNav(client, **kw)


def _pump(nav):
    """手动刷一次缓存，替代后台线程（测试要确定性）。"""
    nav.cache.refresh_once()


# ---------- 到点三态 ----------


def test_arrived_via_state_whitelist(srv, client):
    srv.state.current_task = {"pointsId": PID, "currentState": "已到达"}
    nav = _nav(client, arrived_states=("已到达",), enroute_states=("行进中",))
    assert nav.goto_goal(PID) is True
    _pump(nav)
    assert nav.poll_status() is NavStatus.ARRIVED
    assert nav.is_arrived() is True
    assert nav.poll_fault() is None


def test_enroute_state_does_not_count_as_unknown(srv, client):
    srv.state.current_task = {"pointsId": PID, "currentState": "行进中"}
    nav = _nav(client, arrived_states=("已到达",), enroute_states=("行进中",), unknown_tolerance=2)
    nav.goto_goal(PID)
    for _ in range(5):
        _pump(nav)
        assert nav.poll_status() is NavStatus.EN_ROUTE
    assert nav.unknown_count == 0
    assert nav.poll_fault() is None


def test_unknown_state_eventually_reports_fault(srv, client):
    """G2：状态字符串无枚举。对不上白名单就必须如实上报，不许静默。"""
    srv.state.current_task = {"pointsId": PID, "currentState": "某个我们没见过的状态"}
    nav = _nav(client, arrived_states=("已到达",), unknown_tolerance=3)
    nav.goto_goal(PID)
    for _ in range(3):
        _pump(nav)
        assert nav.poll_status() is NavStatus.UNKNOWN
    assert nav.poll_fault() == FAULT_STATUS_UNRECOGNIZED
    assert nav.poll_fault() is None  # 取走即清零，避免重复上报


def test_e1_failure_signature_task_not_tracked(srv, client):
    """E1 的致命假设：单点派单可能压根不产生可查询任务。"""
    srv.state.current_task = None
    nav = _nav(client, unknown_tolerance=2)
    assert nav.goto_goal(PID) is True
    for _ in range(2):
        _pump(nav)
        assert nav.poll_status() is NavStatus.UNKNOWN
    assert nav.poll_fault() == FAULT_TASK_NOT_TRACKED


def test_arrived_via_distance_when_platform_strings_useless(srv, client):
    """白名单为空（E2 未落地）时，距离判据是唯一可靠证据。"""
    srv.state.current_task = {"pointsId": PID, "currentState": "???"}
    pose = {"xy": (10.0, 10.0)}
    nav = _nav(
        client,
        goal_pose_resolver=lambda label: (10.2, 10.1),
        pose_source=lambda: (pose["xy"][0], pose["xy"][1], 0.0),
        arrive_radius_m=0.5,
    )
    nav.goto_goal(PID)
    _pump(nav)
    assert nav.poll_status() is NavStatus.ARRIVED

    pose["xy"] = (0.0, 0.0)
    assert nav.poll_status() is NavStatus.UNKNOWN


def test_nav_timeout_keyword_detected(srv, client):
    """F17：路线被挡且未设检修区时平台报「未找到有效路径」。"""
    srv.state.current_task = {"pointsId": PID, "currentState": "导航超时，未找到有效路径"}
    nav = _nav(client, arrived_states=("已到达",))
    nav.goto_goal(PID)
    _pump(nav)
    assert nav.poll_status() is NavStatus.UNKNOWN
    assert nav.poll_fault() == FAULT_NAV_TIMEOUT


def test_never_polled_cache_is_unknown_not_arrived(srv, client):
    """后台线程还没成功取过一次时，绝不能因为「没看到反例」就判到达。"""
    srv.state.current_task = {"pointsId": PID, "currentState": "已到达"}
    nav = _nav(client, arrived_states=("已到达",))
    nav.goto_goal(PID)
    assert nav.poll_status() is NavStatus.UNKNOWN  # 未 pump


def test_stale_cache_is_unknown_not_arrived(srv, client):
    srv.state.current_task = {"pointsId": PID, "currentState": "已到达"}
    nav = _nav(client, arrived_states=("已到达",), stale_s=0.05)
    nav.goto_goal(PID)
    _pump(nav)
    assert nav.poll_status() is NavStatus.ARRIVED
    time.sleep(0.15)
    assert nav.poll_status() is NavStatus.UNKNOWN


def test_task_cleared_means_arrived_is_opt_in(srv, client):
    srv.state.current_task = {"pointsId": PID, "currentState": "x"}
    nav = _nav(client, task_cleared_means_arrived=True)
    nav.goto_goal(PID)
    _pump(nav)
    nav.poll_status()  # 先观察到「有任务」
    srv.state.current_task = None
    _pump(nav)
    assert nav.poll_status() is NavStatus.ARRIVED


def test_task_cleared_default_is_not_arrived(srv, client):
    """任务消失也可能是被取消，默认不许当成到达。"""
    srv.state.current_task = {"pointsId": PID, "currentState": "x"}
    nav = _nav(client)
    nav.goto_goal(PID)
    _pump(nav)
    nav.poll_status()
    srv.state.current_task = None
    _pump(nav)
    assert nav.poll_status() is NavStatus.UNKNOWN


# ---------- 派单前置 ----------


def test_unresolvable_goal_rejected(srv, client):
    nav = _nav(client, goal_resolver=lambda label: None)
    assert nav.goto_goal("wp_region_x_staging") is False
    assert nav.poll_fault() == FAULT_GOAL_UNRESOLVED
    assert srv.state.navigate_calls == []


def test_goal_resolver_maps_label_to_points_id(srv, client):
    nav = _nav(client, goal_resolver=lambda label: PID)
    assert nav.goto_goal("wp_region_x_staging") is True
    assert nav.points_id == PID
    assert srv.state.navigate_calls == [(ROBOT, PID)]


def test_arbiter_veto_blocks_navigate(srv, client):
    class Denying:
        def allow_topsee_cmd(self):
            return False

    nav = _nav(client, arbiter=Denying())
    assert nav.goto_goal(PID) is False
    assert nav.poll_fault() == FAULT_SEND_REJECTED
    assert srv.state.navigate_calls == []


def test_send_navigate_business_error_returns_false(srv, client):
    srv.state.force_error["/point/sendNavigate"] = (1, "点位不属于当前地图")
    nav = _nav(client)
    assert nav.goto_goal(PID) is False
    assert nav.poll_fault() == FAULT_SEND_REJECTED


def test_poll_interval_floor_enforced(client):
    with pytest.raises(ValueError, match="0.5s"):
        _nav(client, poll_interval_s=0.1)


# ---------- cancel ----------


def test_cancel_calls_stop_task(srv, client):
    nav = _nav(client)
    nav.goto_goal(PID)
    nav.cancel()
    assert srv.state.stop_calls == [ROBOT]


def test_cancel_swallows_platform_errors(srv, client):
    """abort 路径不许因为平台报错而抛出去。"""
    srv.state.force_error["/robotTask/stopTask"] = (1, "boom")
    nav = _nav(client)
    nav.goto_goal(PID)
    nav.cancel()  # 不抛
    assert nav.cancel_calls == 1


# ---------- Perception ----------


def test_local_vision_requires_detector(client):
    with pytest.raises(ValueError, match="detector"):
        TopseePerception(client, robot_id=ROBOT, mode="local_vision")


def test_local_vision_uses_injected_detector(client):
    p = TopseePerception(
        client,
        robot_id=ROBOT,
        mode="local_vision",
        detector=lambda label: {"confidence": 0.87, "evidence_uri": "file://x.jpg"},
    )
    hit = p.search_target("object_a")
    assert hit == {"confidence": 0.87, "evidence_uri": "file://x.jpg"}


def test_perception_rejects_forbidden_keys(client):
    """禁止有人把位姿塞进感知返回值——那会在 validate_event 里炸掉整条 mission。"""
    p = TopseePerception(
        client,
        robot_id=ROBOT,
        mode="local_vision",
        detector=lambda label: {
            "confidence": 0.9,
            "evidence_uri": "e",
            "pose_xyz": [1, 2, 3],
        },
    )
    with pytest.raises(ValueError, match="禁止字段"):
        p.search_target("object_a")


def test_alarm_uri_mode_extracts_image_url(srv, client):
    srv.state.alarm_rows = [
        {"alarmName": "无关告警", "rstImage": "http://x/1.jpg"},
        {"alarmName": "object_a 异常", "rstImage": "http://x/2.jpg"},
    ]
    p = TopseePerception(client, robot_id=ROBOT, mode="alarm_uri", autostart_poller=False)
    assert p.cache is not None
    p.cache.refresh_once()
    hit = p.search_target("object_a")
    assert hit is not None and hit["evidence_uri"] == "http://x/2.jpg"
    assert p.search_target("object_zzz") is None


def test_alarm_uri_stale_returns_none(srv, client):
    srv.state.alarm_rows = [{"alarmName": "object_a", "rstImage": "http://x/2.jpg"}]
    p = TopseePerception(
        client, robot_id=ROBOT, mode="alarm_uri", alarm_stale_s=0.05, autostart_poller=False
    )
    assert p.cache is not None
    assert p.search_target("object_a") is None  # 从未刷新
    p.cache.refresh_once()
    assert p.search_target("object_a") is not None
    time.sleep(0.15)
    assert p.search_target("object_a") is None  # 过期证据不作数


def test_perception_bad_mode_rejected(client):
    with pytest.raises(ValueError, match="local_vision"):
        TopseePerception(client, robot_id=ROBOT, mode="platform_algo")


# ---------- Gas ----------


def test_gas_sample_maps_history_rows(srv, client):
    srv.state.gas_rows = [
        {"type": "CH4", "value": 1.2, "unit": "%LEL", "alarmState": "ok"},
        {"gasType": "H2S", "avg": 0.0, "unit": "ppm"},
        {"garbage": True},
    ]
    g = TopseeGas(client, robot_id=ROBOT)
    readings = g.sample(60.0)
    assert readings == [
        {"channel": "CH4", "value": 1.2, "unit": "%LEL", "alarm_state": "ok"},
        {"channel": "H2S", "value": 0.0, "unit": "ppm", "alarm_state": "unknown"},
    ]


def test_gas_sample_empty_window_returns_empty_not_fabricated(srv, client):
    srv.state.gas_rows = []
    g = TopseeGas(client, robot_id=ROBOT)
    assert g.sample(60.0) == []


def test_gas_sample_platform_error_returns_empty(srv, client):
    srv.state.force_error["/gas/getGasHistory"] = (1, "boom")
    g = TopseeGas(client, robot_id=ROBOT)
    assert g.sample(60.0) == []


def test_gas_connected_uses_recent_history_as_proxy(srv, client):
    g = TopseeGas(client, robot_id=ROBOT)
    srv.state.gas_rows = []
    assert g.is_connected() is False
    srv.state.gas_rows = [{"type": "CH4", "value": 0.0, "unit": "%LEL"}]
    assert g.is_connected() is True


def test_gas_connected_probe_overrides(client):
    g = TopseeGas(client, robot_id=ROBOT, connected_probe=lambda: True)
    assert g.is_connected() is True


def test_gas_without_ledger_is_source_unavailable(client):
    """F13：平台没有标定数据源，不许伪造时间戳。"""
    g = TopseeGas(client, robot_id=ROBOT)
    assert g.calibration_at() == 0.0
    assert g.calibration_reason() == REASON_SOURCE_UNAVAILABLE


def test_gas_with_ledger_reads_calibration(client):
    ledger = GasCalibrationLedger.from_dict(
        {
            "sensors": [
                {"robot_id": ROBOT, "calibrated_at": "2026-07-20T09:30:00+08:00"}
            ]
        }
    )
    g = TopseeGas(client, robot_id=ROBOT, ledger=ledger)
    assert g.calibration_at() > 0.0
    assert g.calibration_reason() == "calibration_stale"  # 2026-07-20 早于 max_age
