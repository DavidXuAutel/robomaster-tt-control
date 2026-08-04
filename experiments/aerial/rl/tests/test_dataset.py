"""Offline tests for episode persistence + the data-quality gate.

Two things must hold: (1) an episode round-trips to ``.npz`` losslessly, and
(2) ``assert_nontrivial`` flags the silent-but-fatal collections (frozen
renderer / dead API control / black frames) while passing a varied episode.
"""
import numpy as np
import pytest

from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.buffer import Transition
from experiments.aerial.rl.env.obs import Observation


def _obs(pos, frame_val, depth=None):
    """An Observation whose RGB is a constant plane at ``frame_val`` + noise."""
    state = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.1], dtype=np.float32)
    rgb = np.full((8, 8, 3), int(frame_val) % 256, dtype=np.uint8)
    # bake in per-pixel spread so rgb_std is non-trivial
    rgb[0, 0, 0] = (int(frame_val) + 40) % 256
    return Observation(rgb=rgb, state=state, depth=depth)


def _moving_episode(n=6, with_depth=False):
    """A healthy episode: drone moves, frames change, some reward flows."""
    trans = []
    for i in range(n):
        depth = np.full((8, 8), 5.0 + i, np.float32) if with_depth else None
        obs = _obs([float(i), 0.0, 2.0], frame_val=10 + i * 20, depth=depth)
        nxt = _obs([float(i + 1), 0.0, 2.0], frame_val=10 + (i + 1) * 20, depth=depth)
        trans.append(Transition(obs=obs, action=np.ones(4) * 0.5,
                                reward=1.0 if i % 2 == 0 else 0.0,
                                done=(i == n - 1), next_obs=nxt))
    return trans


def _frozen_episode(n=6):
    """Frozen renderer / dead control: identical frame + stationary pose."""
    return [
        Transition(obs=_obs([0.0, 0.0, 2.0], frame_val=30),
                   action=np.zeros(4), reward=0.0, done=(i == n - 1))
        for i in range(n)
    ]


# --- round-trip --------------------------------------------------------------

def test_episode_arrays_shapes_and_dtypes():
    arr = ds.episode_arrays(_moving_episode(n=5))
    assert arr["rgb"].shape == (5, 8, 8, 3) and arr["rgb"].dtype == np.uint8
    assert arr["proprio"].shape == (5, 4)
    assert arr["actions"].shape == (5, 4)
    assert arr["rewards"].shape == (5,)
    assert arr["dones"].dtype == np.bool_
    assert "depth" not in arr  # none supplied


def test_write_episode_round_trip(tmp_path):
    ep = _moving_episode(n=4)
    path = ds.write_episode(tmp_path, 7, ep)
    assert path.name == "episode_00007.npz"
    loaded = np.load(path)
    ref = ds.episode_arrays(ep)
    np.testing.assert_array_equal(loaded["rgb"], ref["rgb"])
    np.testing.assert_allclose(loaded["proprio"], ref["proprio"])
    np.testing.assert_allclose(loaded["actions"], ref["actions"])


def test_depth_stored_only_when_every_frame_has_it():
    assert "depth" in ds.episode_arrays(_moving_episode(with_depth=True))
    # mixed: last frame missing depth -> whole channel dropped
    ep = _moving_episode(n=3, with_depth=True)
    ep[-1].obs.depth = None
    assert "depth" not in ds.episode_arrays(ep)


def test_empty_episode_raises():
    with pytest.raises(ValueError):
        ds.episode_arrays([])


# --- quality gate ------------------------------------------------------------

def test_moving_episode_is_nontrivial():
    rep = ds.quality_report(_moving_episode())
    assert ds.assert_nontrivial(rep) == []
    assert rep["path_length_m"] > 0
    assert rep["rgb_frame_variation"] > ds.MIN_FRAME_VARIATION
    assert rep["reward_sum"] > 0


def test_frozen_renderer_flagged():
    fails = ds.assert_nontrivial(ds.quality_report(_frozen_episode()))
    joined = " ".join(fails)
    assert "frozen" in joined      # identical frames
    assert "did not move" in joined  # stationary pose


def test_black_frames_flagged():
    # constant zero RGB + tiny motion => low std trips the black-frame guard
    ep = [
        Transition(obs=Observation(rgb=np.zeros((8, 8, 3), np.uint8),
                                   state=np.array([float(i), 0, 2, 0, 0, 0, 0], np.float32)),
                   action=np.zeros(4), reward=0.0, done=(i == 3))
        for i in range(4)
    ]
    fails = ds.assert_nontrivial(ds.quality_report(ep))
    assert any("constant" in f or "black" in f for f in fails)


def test_flat_reward_is_not_a_hard_failure():
    # a moving episode with all-zero reward is legitimate (stationary target) —
    # must NOT be a hard failure as long as frames vary + drone moves.
    ep = _moving_episode()
    for t in ep:
        t.reward = 0.0
    assert ds.assert_nontrivial(ds.quality_report(ep)) == []


# --- summaries ---------------------------------------------------------------

def test_manifest_and_quality_summary_written(tmp_path):
    reps = [ds.quality_report(_moving_episode()) for _ in range(2)]
    ds.write_manifest(tmp_path, [{"file": "e0.npz"}], meta={"backend": "mock"})
    summ = ds.write_quality_summary(tmp_path, reps)
    assert (tmp_path / "manifest.json").exists()
    assert summ.name == "QUALITY_SUMMARY.json"
    import json
    data = json.loads(summ.read_text())
    assert data["episodes"] == 2
    assert data["total_steps"] == sum(r["steps"] for r in reps)
