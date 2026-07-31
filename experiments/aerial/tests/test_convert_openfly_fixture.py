import json
from pathlib import Path

import numpy as np
import pytest

from experiments.aerial.convert_openfly_to_lerobot import (
    convert_trajectory,
    write_lerobot_dataset,
)


FIXTURE = Path(__file__).parent / "fixtures" / "mini_openfly"


def _trajectory() -> dict:
    ann = json.loads((FIXTURE / "annotation.json").read_text())
    return ann[0] if isinstance(ann, list) else ann["trajectories"][0]


def test_convert_drops_padding_and_emits_4d():
    frames = convert_trajectory(_trajectory(), FIXTURE, action_source="pos_delta_v1")
    assert len(frames) >= 2
    for fr in frames:
        assert fr["action"].shape == (4,)
        assert fr["observation.state"].shape == (4,)
        assert fr["observation.images.ego"].ndim == 3
        assert fr["meta.action_source"] == "pos_delta_v1"
        assert isinstance(fr["task"], str) and len(fr["task"]) > 0


def test_convert_uses_pose_deltas_and_omits_last_frame():
    frames = convert_trajectory(_trajectory(), FIXTURE)

    assert len(frames) == 3
    np.testing.assert_allclose(frames[0]["action"], [3.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        frames[2]["action"],
        [0.0, 0.0, 0.0, np.pi / 6],
    )
    assert frames[0]["action"].dtype == np.float32
    assert frames[0]["observation.state"].dtype == np.float32
    assert frames[0]["observation.images.ego"].dtype == np.uint8


def test_convert_drops_openfly_padding_actions():
    traj = _trajectory()
    traj["action"] = [1, -1, -2, 0]

    frames = convert_trajectory(traj, FIXTURE)

    assert len(frames) == 1
    np.testing.assert_array_equal(
        frames[0]["observation.state"],
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )


def test_convert_accepts_mixed_3d_and_4d_pos_rows():
    traj = _trajectory()
    # OpenFly sometimes stores [x,y,z,yaw] rows mixed with [x,y,z].
    traj["pos"] = [
        [0.0, 0.0, 1.0, 0.0],
        [3.0, 0.0, 1.0],
        [6.0, 0.0, 1.0, 0.0],
        [6.0, 0.0, 1.0, np.pi / 6],
    ]
    traj["yaw"] = [9.0, 9.0, 9.0, 9.0]  # should be ignored when pos is 4D

    frames = convert_trajectory(traj, FIXTURE)

    assert len(frames) == 3
    np.testing.assert_allclose(frames[0]["observation.state"], [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(
        frames[2]["observation.state"],
        [6.0, 0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        frames[2]["action"],
        [0.0, 0.0, 0.0, np.pi / 6],
        atol=1e-5,
    )


def test_convert_rejects_non_v1_action_source():
    with pytest.raises(ValueError, match="pos_delta_v1"):
        convert_trajectory(_trajectory(), FIXTURE, action_source="primitive")


def test_stop_relabel_zeros_near_goal_and_terminal_frames():
    # Fixture start positions are 6, 3, 0 m from the goal [6,0,1].
    frames = convert_trajectory(_trajectory(), FIXTURE, stop_relabel_radius=4.0)

    assert len(frames) == 3
    # frame0 starts 6 m out (>= radius) → keep its forward delta.
    np.testing.assert_allclose(frames[0]["action"], [3.0, 0.0, 0.0, 0.0])
    # frame1 starts 3 m out (< radius) → zeroed to stop.
    np.testing.assert_allclose(frames[1]["action"], [0.0, 0.0, 0.0, 0.0])
    # frame2 is terminal → always stop.
    np.testing.assert_allclose(frames[2]["action"], [0.0, 0.0, 0.0, 0.0])
    for fr in frames:
        assert fr["action"].dtype == np.float32


def test_stop_relabel_forces_terminal_stop_even_outside_radius():
    # Radius 0 means no frame is "near" the goal, but the terminal frame is
    # still zeroed (the episode must command stop at its end).
    frames = convert_trajectory(_trajectory(), FIXTURE, stop_relabel_radius=0.0)

    np.testing.assert_allclose(frames[0]["action"], [3.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(frames[1]["action"], [3.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(frames[2]["action"], [0.0, 0.0, 0.0, 0.0])


def test_stop_relabel_none_is_noop():
    baseline = convert_trajectory(_trajectory(), FIXTURE)
    relabeled = convert_trajectory(_trajectory(), FIXTURE, stop_relabel_radius=None)
    for a, b in zip(baseline, relabeled):
        np.testing.assert_array_equal(a["action"], b["action"])


def test_stop_relabel_rejects_negative_radius():
    with pytest.raises(ValueError, match="non-negative"):
        convert_trajectory(_trajectory(), FIXTURE, stop_relabel_radius=-1.0)


def test_stop_relabel_delta_maps_to_stop_primitive():
    # The zeroed action must round-trip to OpenFly primitive 0 (stop) via the
    # same classifier the CE loss uses — this is what wires stop supervision in.
    from experiments.aerial.collapse_fix.labels import delta_nearest_with_dist

    frames = convert_trajectory(_trajectory(), FIXTURE, stop_relabel_radius=4.0)
    prim, dist = delta_nearest_with_dist(frames[2]["action"].astype(float))
    assert prim == 0
    assert dist == 0.0


def test_write_lerobot_dataset_stores_task_language(tmp_path):
    frames = convert_trajectory(_trajectory(), FIXTURE)
    out_root = tmp_path / "dataset"

    write_lerobot_dataset([frames], out_root, repo_id="local/mini-openfly")

    tasks = [
        json.loads(line)
        for line in (out_root / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    assert any(item["task"] == frames[0]["task"] for item in tasks)
    info = json.loads((out_root / "meta" / "info.json").read_text())
    assert info["features"]["action"]["shape"] == [4]
    assert info["features"]["observation.state"]["shape"] == [4]
