"""AUTO 看门狗纯逻辑：挂载超时 / 感知失联 / 挂载后宽限期。"""

from tt_control.auto_safety import AutoWatchdog


def test_not_engaged_never_fires():
    wd = AutoWatchdog()
    assert wd.check(now=1000.0, engaged_since=None, last_depth_ts=999.0) is None


def test_normal_fresh_depth_ok():
    wd = AutoWatchdog(max_engaged_s=30, depth_stale_s=1.5)
    # 挂载 5s、深度 0.2s 前 → 正常
    assert wd.check(now=105.0, engaged_since=100.0, last_depth_ts=104.8) is None


def test_max_engaged_timeout():
    wd = AutoWatchdog(max_engaged_s=30, depth_stale_s=1.5)
    reason = wd.check(now=131.0, engaged_since=100.0, last_depth_ts=130.9)
    assert reason and "engaged" in reason


def test_depth_stale():
    wd = AutoWatchdog(max_engaged_s=30, depth_stale_s=1.5)
    # 挂载才 5s(未超时),但深度 2s 前(超 1.5s)→ 失联
    reason = wd.check(now=105.0, engaged_since=100.0, last_depth_ts=103.0)
    assert reason and "stale" in reason


def test_no_depth_grace_then_fire():
    wd = AutoWatchdog(max_engaged_s=30, depth_stale_s=1.5)
    # 挂载后 1s 仍无深度 → 宽限期内不报
    assert wd.check(now=101.0, engaged_since=100.0, last_depth_ts=None) is None
    # 挂载后 2s 仍无深度 → 判失联
    reason = wd.check(now=102.0, engaged_since=100.0, last_depth_ts=None)
    assert reason and "no depth" in reason
