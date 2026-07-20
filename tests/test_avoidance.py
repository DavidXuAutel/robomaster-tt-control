"""避障控制律 AvoidanceController 回归测试(合成近度图,无需 GPU/服务)。"""
import numpy as np

from tt_control.avoidance import AvoidanceController, AvoidParams
from tt_control.control import RcAxes


def _grid(val, shape=(60, 90)):
    return np.full(shape, val, dtype=np.float32)


def test_clear_cruises_forward():
    c = AvoidanceController()
    d = c.decide(_grid(0.10))
    assert d.state == "CRUISE"
    assert d.axes.pitch == AvoidParams().cruise_speed
    assert d.axes.yaw == 0 and d.axes.roll == 0


def test_obstacle_left_turns_right():
    c = AvoidanceController()
    n = _grid(0.10)
    n[:, : n.shape[1] // 3] = 0.9          # 左区很近
    d = c.decide(n)
    assert d.state in ("TURN_R", "TURN_L")
    assert d.state == "TURN_R"             # 左更挡 → 向右绕
    assert d.axes.yaw > 0


def test_boxed_in_hovers():
    c = AvoidanceController()
    d = c.decide(_grid(0.95))              # 三区全近
    assert d.state == "BLOCKED"
    assert d.axes.is_zero()


def test_commit_hysteresis_holds_then_resets():
    c = AvoidanceController()
    n = _grid(0.10); n[:, : n.shape[1] // 3] = 0.9
    first = c.decide(n).state
    # 即使中区略变,承诺方向不翻转
    n2 = _grid(0.50)
    assert c.decide(n2).state == first
    c.reset()
    assert c._commit == 0
