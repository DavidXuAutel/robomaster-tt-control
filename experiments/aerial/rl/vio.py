"""Local windowed VIO math (spec [1c]) — torch-free, Mac-testable.

This module is the **metric / integration layer**, not a learned network. It
turns supervision channels (velocity / IMU-optional / timestamps / depth) into:

  - integrated metric displacement over a window
  - a VIO scale proxy from ‖Δp‖
  - a depth-implied scale from predicted (or GT) depth change vs motion
  - ``scale_relative_error`` for V0 signal ③ (frozen §4.1)

A learned VIO head (if added later) must emit quantities that these helpers can
score; do not put GT vel into the policy graph (use ``perception_data`` only).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from experiments.aerial.rl.perception_data import dt_from_timestamps


def integrate_velocity(
    vel: np.ndarray,
    timestamps: np.ndarray,
    *,
    fallback_hz: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate ``vel [..., L, 3]`` with glitch-robust ``dt`` → positions.

    Returns ``(pos, dt)`` where ``pos[..., 0, :] = 0`` and
    ``pos[..., t] = pos[..., t-1] + vel[..., t-1] * dt[..., t]`` (left Riemann,
    matching "velocity reported at the start of the interval").
    """
    vel = np.asarray(vel, dtype=np.float64)
    if vel.ndim < 2 or vel.shape[-1] != 3:
        raise ValueError(f"vel must be [..., L, 3], got {vel.shape}")
    dt = dt_from_timestamps(timestamps, fallback_hz=fallback_hz).astype(np.float64)
    if dt.shape != vel.shape[:-1]:
        raise ValueError(f"timestamps/dt shape {dt.shape} incompatible with vel {vel.shape}")

    pos = np.zeros_like(vel)
    for t in range(1, vel.shape[-2]):
        pos[..., t, :] = pos[..., t - 1, :] + vel[..., t - 1, :] * dt[..., t : t + 1]
    return pos.astype(np.float32), dt.astype(np.float32)


def window_motion_m(pos: np.ndarray) -> np.ndarray:
    """Net displacement magnitude ‖pos[..., -1] - pos[..., 0]‖ → ``[...,]``."""
    pos = np.asarray(pos, dtype=np.float64)
    delta = pos[..., -1, :] - pos[..., 0, :]
    return np.linalg.norm(delta, axis=-1).astype(np.float32)


def depth_median(depth: np.ndarray) -> np.ndarray:
    """Per-frame median depth over spatial dims → ``[..., L]``.

    ``depth`` is ``[..., L, H, W]`` (or already ``[..., L]``). Non-finite / ≤0
    pixels are ignored; an all-invalid frame yields NaN.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim >= 3:
        spatial = tuple(range(d.ndim - 2, d.ndim))  # last two = H,W if present
        # If shape is [..., L, H, W], median over H,W.
        if d.ndim >= 4 or (d.ndim == 3 and d.shape[-1] != 3):
            flat = d.reshape(*d.shape[:-2], -1)
        else:
            flat = d.reshape(*d.shape[:-1], -1)
    else:
        flat = d
    masked = np.where(np.isfinite(flat) & (flat > 0), flat, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmedian(masked, axis=-1).astype(np.float32)


def scale_from_motion(motion_m: np.ndarray) -> np.ndarray:
    """VIO scale proxy: net metric motion (metres). Identity mapping for now."""
    return np.asarray(motion_m, dtype=np.float32)


def scale_from_depth_change(
    depth_med: np.ndarray,
    motion_m: np.ndarray,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    """Depth-implied scale: |Δ median_depth| aligned with motion magnitude.

    For a translating camera looking roughly along the motion, apparent depth
    change correlates with metric motion. We use
    ``s_D = |d_last - d_first|`` as the depth-side length scale to compare
    against ``s_VIO = motion_m`` (both in metres when depth is metric GT).
    """
    d = np.asarray(depth_med, dtype=np.float64)
    if d.shape[-1] < 2:
        return np.full(d.shape[:-1], np.nan, dtype=np.float32)
    s_d = np.abs(d[..., -1] - d[..., 0])
    # If depth is relative, ratio still comparable window-to-window; gate uses
    # relative error between s_d and motion, so both must be finite & > eps.
    _ = eps
    _ = motion_m
    return s_d.astype(np.float32)


def scale_relative_error(
    scale_depth: np.ndarray,
    scale_vio: np.ndarray,
    *,
    eps: float = 1e-3,
    min_motion_m: float = 0.5,
    motion_m: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Per-window |s_D - s_VIO| / max(s_VIO, eps); masks low-motion windows.

    Frozen §4.1 ③: pass if **median** of valid errors ≤ 0.25.
    """
    s_d = np.asarray(scale_depth, dtype=np.float64)
    s_v = np.asarray(scale_vio, dtype=np.float64)
    if s_d.shape != s_v.shape:
        raise ValueError(f"scale shape mismatch {s_d.shape} vs {s_v.shape}")

    if motion_m is None:
        motion_m = s_v
    motion = np.asarray(motion_m, dtype=np.float64)
    valid = (
        np.isfinite(s_d)
        & np.isfinite(s_v)
        & np.isfinite(motion)
        & (motion >= float(min_motion_m))
        & (s_v >= float(eps))
    )
    rel = np.full(s_d.shape, np.nan, dtype=np.float64)
    denom = np.maximum(s_v, float(eps))
    rel[valid] = np.abs(s_d[valid] - s_v[valid]) / denom[valid]
    return {
        "rel_err": rel.astype(np.float32),
        "valid": valid,
        "median_rel_err": (
            np.float32(np.nanmedian(rel[valid])) if np.any(valid) else np.float32(np.nan)
        ),
        "n_valid": np.int32(np.count_nonzero(valid)),
    }


def window_scale_report(
    vel: np.ndarray,
    timestamps: np.ndarray,
    depth: np.ndarray,
    *,
    fallback_hz: float = 8.0,
    min_motion_m: float = 0.5,
    eps: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """End-to-end §4.1 ③ helper on perception-shaped arrays."""
    pos, _dt = integrate_velocity(vel, timestamps, fallback_hz=fallback_hz)
    motion = window_motion_m(pos)
    s_vio = scale_from_motion(motion)
    d_med = depth_median(depth)
    s_d = scale_from_depth_change(d_med, motion, eps=eps)
    err = scale_relative_error(
        s_d, s_vio, eps=eps, min_motion_m=min_motion_m, motion_m=motion
    )
    return {
        "pos": pos,
        "motion_m": motion,
        "scale_vio": s_vio,
        "scale_depth": s_d,
        "depth_median": d_med,
        **err,
    }
