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


def _instant_crash_episode(n=1):
    """A short episode that ends in a collision — a bad spawn / instant crash.

    ``collided`` rides on the Observation; a 1-step episode has no path to
    difference, so it's the only signal that distinguishes this from a healthy
    short run.
    """
    trans = []
    for i in range(n):
        obs = _obs([float(i), 0.0, 2.0], frame_val=10 + i * 20)
        obs.collided = True
        trans.append(Transition(obs=obs, action=np.ones(4) * 0.5,
                                reward=-10.0, done=True))
    return trans


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


# --- load round-trip ---------------------------------------------------------

def test_load_episode_round_trip(tmp_path):
    ep = _moving_episode(n=4, with_depth=True)
    path = ds.write_episode(tmp_path, 0, ep)
    loaded = ds.load_episode(path)
    assert len(loaded) == 4
    ref = ds.episode_arrays(ep)
    got = ds.episode_arrays(loaded)
    np.testing.assert_array_equal(got["rgb"], ref["rgb"])
    np.testing.assert_allclose(got["actions"], ref["actions"])
    np.testing.assert_allclose(got["rewards"], ref["rewards"])
    np.testing.assert_array_equal(got["collided"], ref["collided"])
    assert "depth" in got


def test_load_dataset_skips_quarantined(tmp_path):
    ds.write_episode(tmp_path, 0, _moving_episode(n=5))
    ds.write_episode(tmp_path, 1, _instant_crash_episode(n=1))
    ds.write_episode(tmp_path, 2, _moving_episode(n=5))
    kept = ds.load_dataset(tmp_path)                       # skip_quarantined=True
    assert len(kept) == 2                                  # crash omitted
    allofthem = ds.load_dataset(tmp_path, skip_quarantined=False)
    assert len(allofthem) == 3


def test_terminal_collision_reloads_on_next_obs_not_pre_step(tmp_path):
    """``collided`` is a POST-step event. On reload the terminal pre-step obs
    must stay clean and the contact must land on next_obs — the old reload
    smeared it onto obs, which would corrupt any pre-step collision read."""
    ep = _moving_episode(n=4)
    ep[-1].next_obs.collided = True  # contact results from the last action
    path = ds.write_episode(tmp_path, 0, ep)
    loaded = ds.load_episode(path)
    assert loaded[-1].obs.collided is False, "terminal pre-step obs must be clean"
    assert loaded[-1].next_obs.collided is True, "contact belongs on next_obs"
    assert all(t.obs.collided is False for t in loaded), "no pre-step obs collided"
    # round-trip: re-serialized collided mask is unchanged (post-step semantics).
    np.testing.assert_array_equal(
        ds.episode_arrays(loaded)["collided"], ds.episode_arrays(ep)["collided"]
    )


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


# --- quarantine (instant crash) ---------------------------------------------

def test_instant_crash_is_quarantined_not_hard_failed():
    # 1-step episode ending in collision: path is structurally 0 so the
    # move/frozen checks can't see it — but it IS an instant crash.
    rep = ds.quality_report(_instant_crash_episode(n=1))
    assert ds.assert_nontrivial(rep) == []            # NOT a hard failure
    reasons = ds.quarantine_reasons(rep)
    assert reasons and "instant crash" in reasons[0]  # but quarantined


def test_two_step_crash_is_quarantined():
    rep = ds.quality_report(_instant_crash_episode(n=2))
    assert ds.quarantine_reasons(rep)                 # steps<=SPAWN_COLLISION_MAX_STEPS


def test_healthy_moving_episode_not_quarantined():
    # frames vary + moved + no early collision => usable.
    assert ds.quarantine_reasons(ds.quality_report(_moving_episode(n=6))) == []


def test_late_collision_not_quarantined():
    # a long run that happens to collide at the end is real training data,
    # NOT a spawn artifact — must not be quarantined.
    ep = _moving_episode(n=6)
    ep[-1].obs.collided = True
    assert ds.quarantine_reasons(ds.quality_report(ep)) == []


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
    assert data["quarantined"] == 0 and data["usable"] == 2


def test_quality_summary_counts_quarantined(tmp_path):
    reps = [ds.quality_report(_moving_episode()),
            ds.quality_report(_instant_crash_episode(n=1))]
    summ = ds.write_quality_summary(tmp_path, reps)
    import json
    data = json.loads(summ.read_text())
    assert data["episodes"] == 2
    assert data["quarantined"] == 1 and data["usable"] == 1


def test_load_episode_round_trip(tmp_path):
    ep = _moving_episode(n=5, with_depth=True)
    path = ds.write_episode(tmp_path, 3, ep)
    loaded = ds.load_episode(path)
    assert len(loaded) == 5
    np.testing.assert_array_equal(loaded[0].obs.rgb, ep[0].obs.rgb)
    np.testing.assert_allclose(loaded[2].obs.proprio4(), ep[2].obs.proprio4())
    np.testing.assert_allclose(loaded[1].action, ep[1].action)
    assert loaded[4].done == ep[4].done
    assert loaded[0].obs.depth is not None


def test_load_dataset_skips_quarantined(tmp_path):
    ds.write_episode(tmp_path, 0, _moving_episode(n=6))
    ds.write_episode(tmp_path, 1, _instant_crash_episode(n=1))
    eps = ds.load_dataset(tmp_path, skip_quarantined=True)
    assert len(eps) == 1
    assert len(eps[0]) == 6
    all_eps = ds.load_dataset(tmp_path, skip_quarantined=False)
    assert len(all_eps) == 2


def test_quarantine_catches_legacy_short_negative_return(tmp_path):
    # Legacy dataset_v0 wrote pre-step obs.collided (false on frame 0) even when
    # the step ended in a crash — reward_sum is the remaining signal.
    ep = [
        Transition(
            obs=_obs([0.0, 0.0, 2.0], frame_val=30),
            action=np.zeros(4),
            reward=-10.0,
            done=True,
            next_obs=_obs([0.0, 0.0, 2.0], frame_val=30),  # next also clean in legacy
        )
    ]
    reasons = ds.quarantine_reasons(ds.quality_report(ep))
    assert reasons and "instant crash" in reasons[0]


# --- schema v2: IMU / velocity / timestamps ----------------------------------

def _obs_v2(pos, vel, frame_val, t, imu=True, depth=None):
    """A schema-v2 Observation: full velocity triple + IMU dict + timestamp."""
    state = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], 0.1], dtype=np.float32)
    rgb = np.full((8, 8, 3), int(frame_val) % 256, dtype=np.uint8)
    rgb[0, 0, 0] = (int(frame_val) + 40) % 256
    imu_dict = {"ang_vel": [0.0, 0.1, 0.2], "lin_acc": [0.0, 0.0, 9.807]} if imu else {}
    return Observation(rgb=rgb, state=state, depth=depth, imu=imu_dict, t=t)


def _v2_episode(n=5, with_depth=False, drop_imu_at=None):
    trans = []
    for i in range(n):
        depth = np.full((8, 8), 5.0 + i, np.float32) if with_depth else None
        imu = drop_imu_at != i
        obs = _obs_v2([float(i), 0.0, 2.0], [1.0, 0.0, 0.0], 10 + i * 20,
                      t=0.5 * i, imu=imu, depth=depth)
        nxt = _obs_v2([float(i + 1), 0.0, 2.0], [1.0, 0.0, 0.0], 10 + (i + 1) * 20,
                      t=0.5 * (i + 1), imu=imu, depth=depth)
        trans.append(Transition(obs=obs, action=np.ones(4) * 0.5, reward=0.0,
                                done=(i == n - 1), next_obs=nxt))
    return trans


def test_schema_v2_arrays_present_with_shapes():
    arr = ds.episode_arrays(_v2_episode(n=5))
    assert arr["vel"].shape == (5, 3) and arr["vel"].dtype == np.float32
    assert arr["imu_ang_vel"].shape == (5, 3)
    assert arr["imu_lin_acc"].shape == (5, 3)
    assert arr["imu_present"].shape == (5,) and arr["imu_present"].dtype == np.bool_
    assert arr["timestamps"].shape == (5,)
    np.testing.assert_allclose(arr["vel"][0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(arr["timestamps"], [0.0, 0.5, 1.0, 1.5, 2.0])
    assert arr["imu_present"].all()


def test_schema_v2_missing_imu_is_nan_row_and_mask_false():
    arr = ds.episode_arrays(_v2_episode(n=4, drop_imu_at=2))
    assert not arr["imu_present"][2]
    assert bool(np.all(np.isnan(arr["imu_ang_vel"][2])))
    assert bool(np.all(np.isnan(arr["imu_lin_acc"][2])))
    # the present frames are finite
    assert np.isfinite(arr["imu_ang_vel"][0]).all()


def test_schema_v2_round_trip_restores_vel_imu_time(tmp_path):
    ep = _v2_episode(n=4, with_depth=True)
    path = ds.write_episode(tmp_path, 1, ep)
    loaded = ds.load_episode(path)
    # velocity restored into state[3:6] (v1 padded zeros)
    np.testing.assert_allclose(loaded[0].obs.velocity, [1.0, 0.0, 0.0], atol=1e-6)
    # IMU dict restored
    assert loaded[2].obs.imu["ang_vel"] == pytest.approx([0.0, 0.1, 0.2])
    assert loaded[2].obs.imu["lin_acc"] == pytest.approx([0.0, 0.0, 9.807])
    # timestamp restored
    assert loaded[3].obs.t == pytest.approx(1.5)


def test_schema_v2_dropped_imu_frame_reloads_empty(tmp_path):
    ep = _v2_episode(n=4, drop_imu_at=1)
    path = ds.write_episode(tmp_path, 0, ep)
    loaded = ds.load_episode(path)
    assert loaded[1].obs.imu == {}          # NaN sentinel -> empty, not [nan,...]
    assert loaded[0].obs.imu != {}


def test_legacy_v1_npz_still_loads_with_fallbacks(tmp_path):
    # Emulate a legacy npz that predates schema v2: only the v1 keys on disk.
    ep = _moving_episode(n=3)
    ref = ds.episode_arrays(ep)
    legacy = {k: ref[k] for k in ("rgb", "proprio", "actions", "rewards", "dones", "collided")}
    path = tmp_path / "episode_00000.npz"
    np.savez_compressed(path, **legacy)
    loaded = ds.load_episode(path)
    assert len(loaded) == 3
    np.testing.assert_allclose(loaded[0].obs.velocity, [0.0, 0.0, 0.0])  # zero-pad fallback
    assert loaded[0].obs.imu == {}
    assert loaded[0].obs.t == 0.0


def test_quality_report_imu_presence():
    rep_v2 = ds.quality_report(_v2_episode(n=4))
    assert rep_v2["has_imu"] and rep_v2["imu_present_frac"] == 1.0
    rep_v1 = ds.quality_report(_moving_episode(n=4))
    assert rep_v1["has_imu"] and rep_v1["imu_present_frac"] == 0.0  # empty dicts
