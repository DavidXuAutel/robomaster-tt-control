"""Replay-window → stacked-array adapter for the V0 perception pillars.

This is the deliberate SIBLING of ``wm_data.windows_to_arrays``. The world-model
adapter emits only the policy/WM-visible channels (RGB + proprio4) so the §1.2
RGB-only boundary is preserved. The perception heads, by contrast, are trained on
the *supervision-only* signals — GT depth for the [1b] multi-frame depth head, and
IMU + velocity + timestamps for the [1c] windowed VIO. Keeping them in a separate
module (never fed into the WM) is the structural guarantee that depth/IMU/velocity
GT cannot leak into the policy graph.

Torch-free (numpy + stdlib), so the reshape + dt math is fully unit-testable on
this GPU-less host before the H100 trainer wraps it in ``torch.from_numpy``.

Output arrays are obs-aligned along the window axis (index ``t`` = observation
``t``):

    rgb          [B, L, H, W, 3] uint8   visual input for both pillars
    position     [B, L, 3]       float32 (x, y, z) — VIO Δposition GT anchor
    vel          [B, L, 3]       float32 GT velocity (VIO regression target)
    imu_ang_vel  [B, L, 3]       float32 gyro (VIO input); NaN where absent
    imu_lin_acc  [B, L, 3]       float32 accel (VIO input); NaN where absent
    imu_present  [B, L]          bool    frame carried inertial data
    timestamps   [B, L]          float32 wall-clock capture time (s)
    depth        [B, L, H, W]    float32 GT depth ([1b] target) — ONLY if every
                                         frame in every window has it

``depth`` follows the same all-or-nothing rule as ``dataset.episode_arrays`` /
``wm_data``: present only when *every* frame carries it, so a partial-depth batch
never yields a ragged channel.
"""
from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np

from experiments.aerial.rl.buffer import Episode
from experiments.aerial.rl.wm_data import _validate


def _imu_triple(imu, key: str) -> np.ndarray:
    """One IMU triple as [3] f32, NaN when the frame's dict lacks it.

    Mirrors ``dataset._imu_row`` at the window layer so a frame whose IMU was an
    RPC miss (empty dict) surfaces as NaN + ``imu_present=False`` rather than a
    silent zero the VIO loss would treat as a real measurement.
    """
    vec = imu.get(key) if isinstance(imu, dict) else None
    if vec is None:
        return np.full(3, np.nan, dtype=np.float32)
    return np.asarray(vec, dtype=np.float32).reshape(3)


def windows_to_perception_arrays(windows: List[Episode]) -> Dict[str, np.ndarray]:
    """Stack ``List[List[Transition]]`` into perception-training arrays.

    See the module docstring for emitted keys/shapes. Raises ``ValueError`` on
    empty or ragged input (same contract as ``wm_data.windows_to_arrays``).
    """
    length = _validate(windows)

    def _obs(w: Episode, t: int):
        return w[t].obs

    rgb = np.stack(
        [np.stack([_obs(w, t).rgb for t in range(length)], axis=0) for w in windows],
        axis=0,
    ).astype(np.uint8, copy=False)
    position = np.stack(
        [np.stack([np.asarray(_obs(w, t).position, np.float32) for t in range(length)], axis=0)
         for w in windows],
        axis=0,
    ).astype(np.float32, copy=False)
    vel = np.stack(
        [np.stack([np.asarray(_obs(w, t).velocity, np.float32) for t in range(length)], axis=0)
         for w in windows],
        axis=0,
    ).astype(np.float32, copy=False)
    imu_ang_vel = np.stack(
        [np.stack([_imu_triple(_obs(w, t).imu, "ang_vel") for t in range(length)], axis=0)
         for w in windows],
        axis=0,
    ).astype(np.float32, copy=False)
    imu_lin_acc = np.stack(
        [np.stack([_imu_triple(_obs(w, t).imu, "lin_acc") for t in range(length)], axis=0)
         for w in windows],
        axis=0,
    ).astype(np.float32, copy=False)
    imu_present = np.asarray(
        [[bool(_obs(w, t).imu) for t in range(length)] for w in windows], dtype=np.bool_
    )
    timestamps = np.asarray(
        [[float(_obs(w, t).t) for t in range(length)] for w in windows], dtype=np.float32
    )

    out: Dict[str, np.ndarray] = {
        "rgb": rgb,
        "position": position,
        "vel": vel,
        "imu_ang_vel": imu_ang_vel,
        "imu_lin_acc": imu_lin_acc,
        "imu_present": imu_present,
        "timestamps": timestamps,
    }

    if all(_obs(w, t).depth is not None for w in windows for t in range(length)):
        out["depth"] = np.stack(
            [np.stack([np.asarray(_obs(w, t).depth, np.float32) for t in range(length)], axis=0)
             for w in windows],
            axis=0,
        ).astype(np.float32, copy=False)

    return out


def dt_from_timestamps(timestamps: np.ndarray, *, fallback_hz: float = 8.0) -> np.ndarray:
    """Per-step dt (seconds) from a ``[..., L]`` timestamp array, for VIO integration.

    ``dt[..., t] = ts[..., t] - ts[..., t-1]`` with ``dt[..., 0]`` set to the
    per-window median of the valid deltas (or ``1/fallback_hz`` when a window has
    no usable delta — e.g. legacy npz with all-zero timestamps). Non-positive or
    non-finite deltas are treated as invalid and replaced by that same per-window
    median, so a clock glitch or a duplicated timestamp can't drive a divide-by-dt
    to infinity downstream.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.shape[-1] < 1:
        raise ValueError("timestamps must have length >= 1 along the last axis")
    fallback = 1.0 / float(fallback_hz)

    # L == 1: no deltas to derive dt from — every step falls back to the nominal.
    if ts.shape[-1] == 1:
        return np.full(ts.shape, fallback, dtype=np.float32)

    diffs = np.diff(ts, axis=-1)                       # [..., L-1]
    valid = np.isfinite(diffs) & (diffs > 0.0)

    # Per-window median of the VALID deltas (invalid -> NaN so nanmedian skips
    # them); a window with no valid delta yields NaN, replaced by the fallback.
    masked = np.where(valid, diffs, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slice
        med = np.nanmedian(masked, axis=-1)            # [...]
    med = np.where(np.isfinite(med), med, fallback)

    dt = np.empty_like(ts)
    med_b = np.expand_dims(med, axis=-1)
    dt[..., 1:] = np.where(valid, diffs, med_b)        # glitchy deltas -> median
    dt[..., 0] = med                                   # first step has no delta
    return dt.astype(np.float32, copy=False)
