"""episode 录制器：结构 / 字段 / 深度去重 / 限流 / SimDrone 端到端。"""

import csv
import json
import time

import numpy as np

from tt_control.control import RcAxes
from tt_control.episode_recorder import EpisodeRecorder


class _Depth:
    """鸭子类型兼容 depth_backend.DepthFrame（只用到 nearness / ts）。"""

    def __init__(self, grid, ts):
        self.nearness = grid
        self.ts = ts


def _mp4_frame_count(path) -> int:
    import av
    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(video=0))


def test_recorder_writes_episode(tmp_path):
    rec = EpisodeRecorder(tmp_path, meta_base={"env": "testroom"}, record_hz=0.0)
    grid = np.random.rand(96, 128).astype(np.float32)
    depth = _Depth(grid, ts=1.0)  # 同一 ts → 应只落盘一份 npy
    state = {
        "mid": "1", "x": "10", "y": "0", "z": "100", "yaw": "5", "h": "100",
        "vgx": "3", "vgy": "0", "vgz": "0", "pitch": "1", "roll": "0", "bat": "88",
    }
    for i in range(5):
        ok = rec.capture(
            t_mono=float(i), rgb=np.zeros((60, 80, 3), np.uint8),
            depth=depth, depth_rtt_ms=100.0, state=state,
            act=RcAxes(pitch=25), ctrl_state="CRUISE", zones=(0.1, 0.2, 0.3),
        )
        assert ok
    path = rec.close()

    assert (path / "meta.json").is_file()
    assert (path / "frames.csv").is_file()
    assert (path / "video.mp4").is_file()
    # 视频帧数与 CSV 行数严格 1:1
    assert _mp4_frame_count(path / "video.mp4") == 5
    # 深度按 ts 去重：5 帧共享同一深度 → 只 1 个 npy
    assert len(list((path / "depth").glob("*.npy"))) == 1

    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_frames"] == 5
    assert meta["n_depth_frames"] == 1
    assert meta["action_source"] == "avoidance"  # 全 CRUISE=自动
    assert meta["battery_start_pct"] == 88.0
    assert meta["depth_grid"] == [96, 128]
    assert meta["env"] == "testroom"
    assert meta["outcome"] == "completed"
    assert meta["video"]["frames"] == 5

    rows = list(csv.DictReader((path / "frames.csv").open(encoding="utf-8")))
    assert len(rows) == 5
    assert rows[0]["ctrl_state"] == "CRUISE"
    assert rows[0]["act_pitch"] == "25"
    assert float(rows[0]["near_mid"]) == 0.2
    assert rows[0]["has_depth"] == "1"


def test_recorder_throttle(tmp_path):
    rec = EpisodeRecorder(tmp_path, record_hz=10.0)  # 间隔 0.1s
    img = np.zeros((10, 10, 3), np.uint8)
    assert rec.capture(t_mono=0.0, rgb=img)
    assert not rec.capture(t_mono=0.05, rgb=img)  # 未到间隔 → 跳过
    assert rec.capture(t_mono=0.10, rgb=img)
    path = rec.close()
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_frames"] == 2


def test_recorder_outcome_and_abort(tmp_path):
    rec = EpisodeRecorder(tmp_path, record_hz=0.0)
    rec.capture(t_mono=0.0, rgb=np.zeros((8, 8, 3), np.uint8), ctrl_state="MANUAL")
    rec.set_outcome("aborted", "emergency")
    path = rec.close()
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    assert meta["outcome"] == "aborted"
    assert meta["abort_reason"] == "emergency"
    assert meta["action_source"] == "manual"


def test_recorder_end_to_end_sim(tmp_path):
    """用真实 SimDrone 状态 + SimVideo 帧驱动录制器（无 GUI，等价 --sim 采集）。"""
    from tt_control.sim_drone import SimDrone, SimVideo

    drone = SimDrone()
    drone.connect()
    drone.start_state_listener()
    video = SimVideo()
    video.start()
    drone.takeoff()
    drone.rc(0, 25, 0, 0)  # 前飞

    rec = EpisodeRecorder(tmp_path, meta_base={"sim": True}, record_hz=0.0)
    t0 = time.time()
    while time.time() - t0 < 0.5:
        frame = video.read()
        rec.capture(
            t_mono=time.time(), rgb=frame, state=drone.state,
            act=RcAxes(pitch=25), ctrl_state="MANUAL",
        )
        time.sleep(0.02)
    path = rec.close()
    drone.land()
    drone.close()
    video.stop()

    rows = list(csv.DictReader((path / "frames.csv").open(encoding="utf-8")))
    assert len(rows) >= 5
    assert rows[-1]["height_cm"] != ""      # SimDrone 状态已写入
    assert _mp4_frame_count(path / "video.mp4") == len(rows)  # 视频帧=CSV 行
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    assert meta["action_source"] == "manual"
    assert meta["sim"] is True
