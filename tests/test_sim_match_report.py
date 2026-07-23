"""仿真可信度匹配报告：动作重放与真机位姿一致→误差近零；位姿失配→误差显著。"""

import json
import math

from tt_control.control import RcAxes
from tt_control.episode_recorder import EpisodeRecorder
from tt_control.sim_drone import VMAX, VZMAX, YAWRATE
import sim_match_report


def _write_episode(tmp_path, real_follows_action: bool):
    """造一段前飞 episode：动作恒为 pitch=25。

    real_follows_action=True 时真机位姿按同一运动学积分(应与仿真重放吻合)；
    False 时真机原地不动(与"前飞"动作严重失配)。
    """
    rec = EpisodeRecorder(tmp_path, record_hz=0.0)
    dt = 0.1
    x = y = 0.0
    z = 1.0
    yaw = 0.0
    for i in range(30):
        a, b, c, d = (0, 25, 0, 0)  # 纯前飞
        if i > 0 and real_follows_action:
            vx = (b / 100.0) * VMAX
            vy = (a / 100.0) * VMAX
            yr = math.radians(yaw)
            x += (vx * math.cos(yr) - vy * math.sin(yr)) * dt
            y += (vx * math.sin(yr) + vy * math.cos(yr)) * dt
            z = max(0.0, z + (c / 100.0) * VZMAX * dt)
            yaw = (yaw + (d / 100.0) * YAWRATE * dt) % 360.0
        state = {
            "mid": "1", "x": f"{x * 100:.2f}", "y": f"{y * 100:.2f}",
            "z": f"{z * 100:.2f}", "yaw": f"{yaw:.2f}", "h": f"{z * 100:.2f}",
            "bat": "80",
        }
        rec.capture(
            t_mono=i * dt, rgb=_img(), state=state,
            act=RcAxes(roll=a, pitch=b, throttle=c, yaw=d), ctrl_state="MANUAL",
        )
    return rec.close()


def _img():
    import numpy as np
    return np.zeros((16, 16, 3), dtype="uint8")


def test_faithful_replay_near_zero(tmp_path):
    ep = _write_episode(tmp_path, real_follows_action=True)
    report = sim_match_report.make_report(ep)
    m = report["metrics"]
    assert m["comparable_frames"] >= 20
    # 真机位姿由同一运动学生成 → 重放几乎重合
    assert m["endpoint_error_m"] < 0.02, m
    assert m["mae_m"]["x"] < 0.02
    assert (ep / "match_report.json").is_file()
    # 报告内容自洽
    saved = json.loads((ep / "match_report.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["endpoint_error_m"] == m["endpoint_error_m"]


def test_mismatch_flags_error(tmp_path):
    ep = _write_episode(tmp_path, real_follows_action=False)
    report = sim_match_report.make_report(ep)
    m = report["metrics"]
    # 动作说前飞、真机没动 → 仿真重放会明显偏离（~0.15m/s × 2.9s ≈ 0.44m）
    assert m["endpoint_error_m"] > 0.4, m
    assert m["path_len_real_m"] < 0.05  # 真机几乎没动
    assert m["path_len_sim_m"] > 0.4    # 仿真按动作走了
