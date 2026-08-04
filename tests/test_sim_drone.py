"""SimDrone / SimVideo 运动学与状态测试。"""
import time

import numpy as np

from tt_control.sim_drone import SimDrone, SimVideo
from tt_control.mujoco_twin import _parse_pad_pose, MujocoPadTwin


def _spin(drone, secs=0.5):
    time.sleep(secs)


def test_state_keys_parseable():
    d = SimDrone()
    d.connect()
    d.start_state_listener()
    d.takeoff()
    _spin(d, 0.4)
    st = d.state
    for k in ("mid", "x", "y", "z", "yaw", "h", "bat"):
        assert k in st, f"missing {k}"
    # 起飞后位于垫子上方 → 可解析出位姿
    assert _parse_pad_pose(st) is not None
    d.close()


def test_takeoff_land_height():
    d = SimDrone()
    d.connect(); d.start_state_listener()
    assert d.height_cm() == 0
    d.takeoff(); _spin(d, 0.3)
    assert d.height_cm() > 50           # 起飞后有高度
    d.land(); _spin(d, 0.3)
    assert d.height_cm() <= 5           # 降落后接近 0
    d.close()


def test_forward_moves_x():
    d = SimDrone()
    d.connect(); d.start_state_listener()
    d.takeoff(); _spin(d, 0.2)
    x0 = float(d.state["x"])
    d.rc(0, 60, 0, 0)                   # pitch 前进(yaw=0 → 沿 +x)
    _spin(d, 0.8)
    x1 = float(d.state["x"])
    assert x1 - x0 > 5, f"x should grow: {x0}->{x1}"
    d.close()


def test_pad_stitch_continuous_across_handoff():
    """3 张卡直线支线:换卡时轨迹应连续(不跳回各卡原点)。"""
    tw = MujocoPadTwin(get_state=lambda: {}, headless=True, stitch_pads=True)

    # 1 号卡:全局系 = 该卡局部系,起点在原点
    assert tw._to_global(0.0, 0.0, 1) == (0.0, 0.0)
    # 飞机沿 +x 飞到 1 号卡局部 +0.45m
    tw._last_xy = (0.45, 0.0)
    assert tw._to_global(0.45, 0.0, 1) == (0.45, 0.0)

    # 换到 2 号卡(约摆在全局 0.5m 处):此刻 2 号卡局部读数 ~ -0.05m
    gx, gy = tw._to_global(-0.05, 0.0, 2)
    assert abs(gx - 0.45) < 1e-9 and abs(gy) < 1e-9   # 连续,未跳回 ~0
    # 继续飞过 2 号卡,全局 x 应继续增大
    tw._last_xy = (gx, gy)
    gx2, _ = tw._to_global(0.10, 0.0, 2)
    assert gx2 > 0.45

    # 换到 3 号卡后再回看 2 号卡(往返):偏移应复用,仍连续
    tw._last_xy = (gx2, 0.0)
    tw._to_global(-0.05, 0.0, 3)
    back_x, _ = tw._to_global(0.10, 0.0, 2)   # 2 号卡偏移已标定过
    assert abs(back_x - gx2) < 1e-9


def test_pad_stitch_disabled_is_identity():
    """stitch 关闭(默认)时,单卡/仿真行为不变:局部即全局。"""
    tw = MujocoPadTwin(get_state=lambda: {}, headless=True)
    tw._last_xy = (5.0, 5.0)
    assert tw._to_global(0.3, -0.2, 1) == (0.3, -0.2)
    assert tw._to_global(0.3, -0.2, 7) == (0.3, -0.2)


def test_sim_video_frame():
    v = SimVideo(); v.start()
    f = v.read()
    assert isinstance(f, np.ndarray)
    assert f.shape == (720, 960, 3)
    # 含红色障碍(R 通道高)
    assert int((f[:, :, 2] > 180).sum()) > 500
    v.stop()
