"""Offline tests for the V0 perception-training adapter (torch-free).

Two things must hold: (1) ``windows_to_perception_arrays`` stacks the
supervision-only channels (depth GT / IMU / velocity / timestamps) with the same
obs-aligned ``[B, L, ...]`` contract as ``wm_data``, dropping depth all-or-nothing;
(2) ``dt_from_timestamps`` yields a positive, glitch-robust per-step dt for VIO
integration.
"""
import numpy as np
import pytest

from experiments.aerial.rl.buffer import Transition
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl import perception_data as pd


def _obs(pos, vel, frame_val, t, imu=True, depth=None):
    state = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], 0.1], dtype=np.float32)
    rgb = np.full((8, 8, 3), int(frame_val) % 256, dtype=np.uint8)
    imu_dict = {"ang_vel": [0.0, 0.1, 0.2], "lin_acc": [0.0, 0.0, 9.807]} if imu else {}
    return Observation(rgb=rgb, state=state, depth=depth, imu=imu_dict, t=t)


def _window(n=4, with_depth=False, drop_imu_at=None, t0=0.0, dt=0.5):
    trans = []
    for i in range(n):
        depth = np.full((8, 8), 5.0 + i, np.float32) if with_depth else None
        obs = _obs([float(i), 0.0, 2.0], [1.0, 0.0, 0.0], 10 + i,
                   t=t0 + dt * i, imu=(drop_imu_at != i), depth=depth)
        trans.append(Transition(obs=obs, action=np.ones(4) * 0.5, reward=0.0,
                                done=(i == n - 1), next_obs=obs))
    return trans


def test_perception_arrays_shapes():
    windows = [_window(n=4), _window(n=4)]
    out = pd.windows_to_perception_arrays(windows)
    assert out["rgb"].shape == (2, 4, 8, 8, 3)
    assert out["position"].shape == (2, 4, 3)
    assert out["vel"].shape == (2, 4, 3)
    assert out["imu_ang_vel"].shape == (2, 4, 3)
    assert out["imu_lin_acc"].shape == (2, 4, 3)
    assert out["imu_present"].shape == (2, 4) and out["imu_present"].dtype == np.bool_
    assert out["timestamps"].shape == (2, 4)
    np.testing.assert_allclose(out["vel"][0, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(out["position"][0, 3], [3.0, 0.0, 2.0])


def test_perception_depth_all_or_nothing():
    assert "depth" in pd.windows_to_perception_arrays([_window(with_depth=True)])
    # one window without depth -> channel dropped for the whole batch
    mixed = [_window(with_depth=True), _window(with_depth=False)]
    assert "depth" not in pd.windows_to_perception_arrays(mixed)


def test_perception_missing_imu_nan_and_mask():
    out = pd.windows_to_perception_arrays([_window(n=4, drop_imu_at=2)])
    assert not out["imu_present"][0, 2]
    assert bool(np.all(np.isnan(out["imu_ang_vel"][0, 2])))
    assert np.isfinite(out["imu_ang_vel"][0, 0]).all()


def test_perception_empty_and_ragged_raise():
    with pytest.raises(ValueError):
        pd.windows_to_perception_arrays([])
    with pytest.raises(ValueError):
        pd.windows_to_perception_arrays([_window(n=4), _window(n=3)])


# --- dt_from_timestamps ------------------------------------------------------

def test_dt_regular_spacing():
    ts = np.array([[0.0, 0.5, 1.0, 1.5]])
    dt = pd.dt_from_timestamps(ts)
    np.testing.assert_allclose(dt, [[0.5, 0.5, 0.5, 0.5]], atol=1e-6)


def test_dt_glitch_replaced_by_median():
    # a duplicated timestamp (dt=0) and a backward step (dt<0) -> replaced by median
    ts = np.array([[0.0, 0.5, 0.5, 0.4, 1.5]])
    dt = pd.dt_from_timestamps(ts)
    assert (dt > 0).all()
    # valid deltas {0.5 (=0.5-0.0), 1.1 (=1.5-0.4)}; median 0.8 fills bad slots + slot 0
    assert dt[0, 0] == pytest.approx(0.8)
    assert dt[0, 2] == pytest.approx(0.8)
    assert dt[0, 3] == pytest.approx(0.8)
    np.testing.assert_allclose(dt[0, 1], 0.5)  # a valid delta is preserved


def test_dt_all_zero_timestamps_fall_back_to_hz():
    ts = np.zeros((2, 4), dtype=np.float32)
    dt = pd.dt_from_timestamps(ts, fallback_hz=8.0)
    np.testing.assert_allclose(dt, np.full((2, 4), 1.0 / 8.0), atol=1e-6)


def test_dt_single_frame():
    dt = pd.dt_from_timestamps(np.array([[3.0]]), fallback_hz=10.0)
    assert dt.shape == (1, 1)
    assert dt[0, 0] == pytest.approx(0.1)
