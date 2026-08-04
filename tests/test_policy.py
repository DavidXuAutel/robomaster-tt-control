"""策略层测试:MockAvoidPolicy 反应 + 工厂。"""
import numpy as np
import cv2

from tt_control.control import RcAxes
from tt_control.policy import MockAvoidPolicy, ScriptedPolicy, create_policy


def _frame_with_red(cx):
    img = np.full((720, 960, 3), 40, dtype=np.uint8)
    cv2.circle(img, (cx, 360), 90, (40, 40, 220), -1)  # BGR 红
    return img


def _blank():
    return np.full((720, 960, 3), 40, dtype=np.uint8)


def test_factory_names():
    assert create_policy("mock").name == "mock"
    assert create_policy("scripted").name == "scripted"
    assert isinstance(create_policy("mock"), MockAvoidPolicy)


def test_mock_forward_when_clear():
    p = MockAvoidPolicy()
    ax = p.decide(_blank(), {})
    assert isinstance(ax, RcAxes)
    assert ax.pitch > 0            # 无障碍 → 前进
    assert ax.roll == 0


def test_mock_avoids_near_obstacle():
    p = MockAvoidPolicy()
    # 大红块居左 → 应向右(roll>0)避开
    left = p.decide(_frame_with_red(300), {})
    assert left.roll > 0
    # 居右 → 向左(roll<0)
    right = p.decide(_frame_with_red(660), {})
    assert right.roll < 0


def test_scripted_runs_then_hovers():
    p = ScriptedPolicy(plan=[(0.05, RcAxes(pitch=30))])
    ax = p.decide(_blank(), {})
    assert ax.pitch == 30
    import time; time.sleep(0.08)
    assert p.decide(_blank(), {}).is_zero()   # 航线走完 → 悬停
