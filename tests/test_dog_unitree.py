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


def test_dds_sample_cache_and_stale_detection():
    """D0 Q1：subscriber 写入带 t_mono 的样本；>500ms → dds_stale。"""
    import time

    from adapters.dog_unitree import (
        DDS_STALE_S,
        TOPIC_LOW_STATE,
        TOPIC_SPORT_STATE,
        DdsTransport,
        low_state_to_dict,
        sport_state_to_dict,
    )

    t = DdsTransport(interface="eth0")
    # 不走 connect（无 SDK）；直接测缓存契约
    t._ingest(
        TOPIC_SPORT_STATE,
        sport_state_to_dict(
            {
                "position": [1.0, 2.0, 0.3],
                "velocity": [0.1, 0.0, 0.0],
                "yaw_speed": 0.05,
                "imu_state": {"rpy": [0.0, 0.0, 1.2]},
            }
        ),
    )
    t._ingest(TOPIC_LOW_STATE, low_state_to_dict({"bms_state": {"soc": 77}}))
    msg = t.read(TOPIC_SPORT_STATE)
    assert msg is not None and msg["position"][0] == 1.0
    assert "t_mono" in msg
    assert t.sample_age_s(TOPIC_SPORT_STATE) is not None
    assert t.sample_age_s(TOPIC_SPORT_STATE) < DDS_STALE_S

    # 伪造过期样本
    old = dict(msg)
    old["t_mono"] = time.monotonic() - 0.8
    t._latest[TOPIC_SPORT_STATE] = old
    assert t.sample_age_s(TOPIC_SPORT_STATE) > DDS_STALE_S

    u = UnitreeSportClient(t)
    u._connected = True  # 跳过 transport.connect
    assert u.is_dds_stale() is True
    assert u.get_low_state()["bms_state"]["soc"] == 77.0


def test_loopback_not_stale_when_state_available():
    u, _ = _client()
    assert u.is_dds_stale() is False


def test_loopback_stale_when_state_unavailable():
    u, _ = _client(state_available=False)
    assert u.is_dds_stale() is True


def test_body_and_gait_commands_forwarded():
    u, t = _client()
    u.stand_up()
    u.switch_gait(1)
    u.body_height(0.32)
    u.speed_level(1)
    u.move_to_pos(1.0, 2.0, 0.5)
    sent = [api for api, _ in t.calls]
    assert sorted(sent) == sorted([1004, 1011, 1013, 1015, 1036])


# ---------- 机型能力表（2026-08-07 实测 unitree_sdk2py 的欠账回归） ----------


class _FakeSport:
    """只实现 go2/b2 共有方法的假 SportClient。b2 独有方法故意缺席。"""

    def __init__(self):
        self.calls = []

    def _rec(self, name, *args):
        self.calls.append((name, args))
        return 0

    def Move(self, vx, vy, vyaw):
        return self._rec("Move", vx, vy, vyaw)

    def StopMove(self):
        return self._rec("StopMove")

    def Damp(self):
        return self._rec("Damp")

    def StandUp(self):
        return self._rec("StandUp")

    def SpeedLevel(self, level):
        return self._rec("SpeedLevel", level)


def _dds(family):
    from adapters.dog_unitree import DdsTransport

    t = DdsTransport(interface="eth0", family=family)
    t._sport = _FakeSport()  # 跳过真机 connect，只测能力表派发
    return t


def test_dds_transport_defaults_to_b2_not_go2():
    """目标机是 B2。默认值绝不能是「手边容易跑通的那个」。"""
    from adapters.dog_unitree import DEFAULT_FAMILY, DdsTransport

    assert DEFAULT_FAMILY == "b2"
    assert DdsTransport(interface="eth0").family == "b2"


def test_dds_transport_rejects_unknown_family():
    from adapters.dog_unitree import DdsTransport

    with pytest.raises(ValueError, match="未知机型"):
        DdsTransport(interface="eth0", family="g1")


def test_go2_rejects_b2_only_apis_cleanly():
    """go2 的 SportClient 没有 SwitchGait/BodyHeight/MoveToPos。

    实测坐实：过去导入 go2 客户端却调这三个方法，真机上是 AttributeError。
    现在必须是可读的 UnitreeError。
    """
    from adapters.dog_unitree import (
        API_BODY_HEIGHT,
        API_MOVE_TO_POS,
        API_SWITCH_GAIT,
        UnitreeError,
    )

    t = _dds("go2")
    for api in (API_SWITCH_GAIT, API_BODY_HEIGHT, API_MOVE_TO_POS):
        with pytest.raises(UnitreeError, match="不支持"):
            t.call(api, {"gait": 1, "height": 0.3, "x": 0.0, "y": 0.0, "yaw": 0.0})


def test_common_apis_dispatch_on_both_families():
    from adapters.dog_unitree import API_MOVE, API_STOP_MOVE

    for family in ("go2", "b2"):
        t = _dds(family)
        t.call(API_MOVE, {"vx": 0.1, "vy": 0.0, "vyaw": 0.0})
        t.call(API_STOP_MOVE, {})
        assert [n for n, _ in t._sport.calls] == ["Move", "StopMove"]


def test_b2_only_api_errors_when_sdk_lacks_method():
    """能力表说支持、SDK 却没有 → 报版本不一致，不是静默失败。"""
    from adapters.dog_unitree import API_SWITCH_GAIT, UnitreeError

    t = _dds("b2")  # _FakeSport 故意没有 SwitchGait
    with pytest.raises(UnitreeError, match="SDK 版本"):
        t.call(API_SWITCH_GAIT, {"gait": 1})


def test_unknown_api_id_rejected():
    from adapters.dog_unitree import UnitreeError

    with pytest.raises(UnitreeError, match="不支持"):
        _dds("b2").call(9999, {})


# ---------- 状态契约：设备时间戳 + 本体感受全量 ----------


def test_sport_state_preserves_device_stamp():
    """t_mono 是接收时刻，测不出「狗侧卡住」；必须另存设备钟。"""
    from adapters.dog_unitree import sport_state_to_dict

    d = sport_state_to_dict({"stamp": {"sec": 1700000000, "nanosec": 500_000_000}})
    assert d["t_device"] == pytest.approx(1700000000.5)
    assert d["t_mono"] != d["t_device"]
    # 无 stamp 时如实为 None，不许拿接收时刻冒充设备钟
    assert sport_state_to_dict({})["t_device"] is None


def test_sport_state_keeps_full_imu_and_foot_force():
    from adapters.dog_unitree import sport_state_to_dict

    d = sport_state_to_dict(
        {
            "imu_state": {
                "rpy": [0.1, 0.2, 0.3],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "gyroscope": [0.01, 0.02, 0.03],
                "accelerometer": [0.0, 0.0, 9.8],
                "temperature": 41.0,
            },
            "foot_force": [10, 20, 30, 40],
        }
    )
    imu = d["imu_state"]
    assert imu["accelerometer"] == [0.0, 0.0, 9.8]
    assert imu["gyroscope"] == [0.01, 0.02, 0.03]
    assert len(imu["quaternion"]) == 4
    assert imu["temperature"] == 41.0
    assert d["foot_force"] == [10.0, 20.0, 30.0, 40.0]


def test_low_state_keeps_full_proprioception():
    """v2 §数据契约点名要 12 电机 + IMU + 足底力；只留 q 会打穿 D3。"""
    from adapters.dog_unitree import _MOTOR_FIELDS, low_state_to_dict

    d = low_state_to_dict(
        {
            "bms_state": {"soc": 55},
            "motor_state": [{"q": 0.5, "dq": 0.1, "tau_est": 2.0, "temperature": 40, "lost": 0}]
            * 12,
            "foot_force": [1, 2, 3, 4],
            "tick": 123456,
        }
    )
    assert len(d["motor_state"]) == 12
    for f in _MOTOR_FIELDS:
        assert f in d["motor_state"][0], f"电机字段缺 {f}"
    assert d["motor_state"][0]["tau_est"] == 2.0
    assert d["foot_force"] == [1.0, 2.0, 3.0, 4.0]
    assert d["tick"] == 123456
    assert d["bms_state"]["soc"] == 55.0


def test_loopback_and_dds_share_the_same_read_contract():
    """loopback 缺字段而真机有 → 测试全绿但真机炸。两边键集必须一致。"""
    from adapters.dog_unitree import (
        TOPIC_LOW_STATE,
        TOPIC_SPORT_STATE,
        low_state_to_dict,
        sport_state_to_dict,
    )

    _, t = _client()
    assert set(t.read(TOPIC_SPORT_STATE)) == set(sport_state_to_dict({}))
    assert set(t.read(TOPIC_LOW_STATE)) == set(low_state_to_dict({}))
