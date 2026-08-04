"""M10：宇树运动通道（速度安全盒、租约闩、权限探测）。

方案 §6.4 / F30：遥控器档位的软限位在 DDS Move() 直控路径上**不生效**，
所以安全盒必须由我们实现。这里的测试就是那道限位的回归。
"""

import pytest

from adapters.dog_unitree import (
    API_DAMP,
    API_MOVE,
    API_STOP_MOVE,
    LoopbackTransport,
    SpeedLimits,
    SportPose,
    UnitreeAuthorityError,
    UnitreeLimitError,
    UnitreeNotConnected,
    UnitreeSportClient,
)


def _client(**kw):
    t = LoopbackTransport(**kw)
    u = UnitreeSportClient(t, limits=SpeedLimits(1.0, 0.6, 1.0))
    u.connect()
    return u, t


def test_move_requires_connect():
    u = UnitreeSportClient(LoopbackTransport())
    u.lease_token = "tok"
    with pytest.raises(UnitreeNotConnected):
        u.move(0.1, 0.0, 0.0, lease_token="tok")


def test_move_requires_matching_lease_token():
    u, t = _client()
    u.lease_token = "good"
    with pytest.raises(UnitreeAuthorityError):
        u.move(0.1, 0.0, 0.0, lease_token="bad")
    with pytest.raises(UnitreeAuthorityError):
        u.move(0.1, 0.0, 0.0, lease_token="")
    assert t.calls_of(API_MOVE) == []
    u.move(0.1, 0.0, 0.0, lease_token="good")
    assert len(t.calls_of(API_MOVE)) == 1


def test_speed_box_clamps_by_default():
    u, t = _client()
    u.lease_token = "tok"
    u.move(9.9, -9.9, 9.9, lease_token="tok")
    sent = t.calls_of(API_MOVE)[-1]
    assert sent == {"vx": 1.0, "vy": -0.6, "vyaw": 1.0}
    assert u.clamp_events == 1


def test_speed_box_can_raise_instead():
    t = LoopbackTransport()
    u = UnitreeSportClient(t, limits=SpeedLimits(1.0, 0.6, 1.0), clamp_instead_of_raise=False)
    u.connect()
    u.lease_token = "tok"
    with pytest.raises(UnitreeLimitError):
        u.move(5.0, 0.0, 0.0, lease_token="tok")
    assert t.calls_of(API_MOVE) == []


def test_default_limits_are_conservative():
    """B2 整机能到 5 m/s；默认必须远低于它，放宽只能是显式决定。"""
    d = SpeedLimits()
    assert d.max_vx <= 1.0 and d.max_vy <= 1.0 and d.max_vyaw <= 1.5


def test_in_box_speed_not_counted_as_clamp():
    u, t = _client()
    u.lease_token = "tok"
    u.move(0.9, 0.5, 0.9, lease_token="tok")
    assert u.clamp_events == 0
    assert t.calls_of(API_MOVE)[-1] == {"vx": 0.9, "vy": 0.5, "vyaw": 0.9}


def test_stop_and_damp_need_no_lease():
    """安全动作永远允许——不能因为租约过期就停不下来。"""
    u, t = _client()
    u.lease_token = None
    u.stop_move()
    u.damp()
    assert len(t.calls_of(API_STOP_MOVE)) == 1
    assert len(t.calls_of(API_DAMP)) == 1
    assert u.last_cmd == (0.0, 0.0, 0.0)


def test_sport_state_parsed():
    u, t = _client()
    t.teleport(3.0, -4.0, 1.57)
    p = u.get_sport_state()
    assert isinstance(p, SportPose)
    assert (round(p.x, 3), round(p.y, 3)) == (3.0, -4.0)
    assert round(p.yaw, 2) == 1.57
    assert u.pose_xy_yaw()[:2] == (3.0, -4.0)


def test_sport_state_none_when_unavailable():
    u, _ = _client(state_available=False)
    assert u.get_sport_state() is None
    assert u.pose_xy_yaw() is None


def test_probe_authority_true_when_state_readable():
    u, _ = _client()
    assert u.probe_dds_authority() is True


def test_probe_authority_false_without_state():
    """G11：命令发得出去但读不到状态，不能算拿到控制权。"""
    u, _ = _client(state_available=False)
    assert u.probe_dds_authority() is False


def test_probe_authority_false_when_call_fails():
    u, t = _client()
    t.raise_on_call = True
    assert u.probe_dds_authority() is False


def test_close_stops_motion_first():
    u, t = _client()
    u.lease_token = "tok"
    u.move(0.5, 0.0, 0.0, lease_token="tok")
    u.close()
    assert len(t.calls_of(API_STOP_MOVE)) == 1
    assert t.connected is False


def test_dds_transport_missing_sdk_raises_not_silently_faked():
    """真机路径绝不许在缺 SDK 时退化成 loopback 假数据。"""
    from adapters.dog_unitree import DdsTransport

    t = DdsTransport(interface="eth0")
    try:
        import unitree_sdk2py  # noqa: F401
    except ImportError:
        with pytest.raises(UnitreeNotConnected, match="unitree_sdk2py"):
            t.connect()
    else:  # pragma: no cover — 装了 SDK 的机器上跳过
        pytest.skip("本机装了 unitree_sdk2py")


def test_body_and_gait_commands_forwarded():
    u, t = _client()
    u.stand_up()
    u.switch_gait(1)
    u.body_height(0.32)
    u.speed_level(1)
    u.move_to_pos(1.0, 2.0, 0.5)
    sent = [api for api, _ in t.calls]
    assert sorted(sent) == sorted([1004, 1011, 1013, 1015, 1036])
