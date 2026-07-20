"""键位映射与 RcAxes 单元测试。"""
from tt_control.control import RcAxes, map_key


def test_rcaxes_zero_and_tuple():
    assert RcAxes().is_zero()
    assert RcAxes(pitch=10).as_tuple() == (0, 10, 0, 0)
    assert not RcAxes(yaw=5).is_zero()


def test_map_basic_keys():
    assert map_key(ord("t")).kind == "takeoff"
    assert map_key(ord("l")).kind == "land"
    assert map_key(ord(" ")).kind == "hover"
    assert map_key(ord("h")).kind == "toggle_help"
    assert map_key(ord("v")).kind == "auto_toggle"  # 统一后:v=避障 ARM
    assert map_key(ord("x")).kind == "quit"
    assert map_key(27).kind == "emergency"


def test_map_rc_axes():
    spd = 40
    assert map_key(ord("w"), spd).axes.pitch == spd
    assert map_key(ord("s"), spd).axes.pitch == -spd
    assert map_key(ord("a"), spd).axes.roll == -spd
    assert map_key(ord("d"), spd).axes.roll == spd
    assert map_key(ord("r"), spd).axes.throttle == spd
    assert map_key(ord("f"), spd).axes.throttle == -spd
    assert map_key(ord("q"), spd).axes.yaw == -spd
    assert map_key(ord("e"), spd).axes.yaw == spd
