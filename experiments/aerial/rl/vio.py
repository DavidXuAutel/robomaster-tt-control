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


def depth_median(
    depth: np.ndarray,
    *,
    max_depth_m: Optional[float] = 200.0,
    min_depth_m: Optional[float] = None,
) -> np.ndarray:
    """Per-frame median depth over spatial dims, preserving the frame axis.

    Contract (frozen §4.1 ③ inputs): ``depth`` is either a spatial stack
    ``[..., L, H, W]`` (ndim ≥ 3 → the last two axes are H,W) or already
    per-frame ``[..., L]`` (ndim ≤ 2 → returned unchanged). Either way the frame
    axis L survives, since ``scale_from_depth_change`` differences ``[..., -1]``
    against ``[..., 0]`` along it. Non-finite / ≤0 pixels are ignored; an
    all-invalid frame yields NaN.

    ``max_depth_m`` matches ``v0_metrics.depth_absrel`` / DepthHead train: outdoor
    AirSim DepthPlanar sky/far-plane fill (often >1 km) must not dominate the
    median, or ``ŝ_D = |d_last − d_first|`` measures sky flicker instead of
    navigational scale (GT unmasked median-Δ routinely hundreds of metres).
    Pass ``None`` to keep every finite positive pixel.

    ``min_depth_m`` (dated §4.1 revision 2026-08-05): when set, also drops pixels
    nearer than the floor. ③ uses the navigational band
    ``[scale_depth_min_m, scale_depth_max_m]`` (default 1–40 m) so open-horizon
    medians (~100 m+) cannot zero out ``ŝ_D``. AbsRel / DepthHead keep the
    historical ``max_depth_m=200`` ceiling with no floor.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim >= 3:
        # Flatten the trailing H,W grid into one axis, then median over it.
        flat = d.reshape(*d.shape[:-2], -1)
        valid = np.isfinite(flat) & (flat > 0)
        if max_depth_m is not None:
            valid &= flat <= float(max_depth_m)
        if min_depth_m is not None:
            valid &= flat >= float(min_depth_m)
        masked = np.where(valid, flat, np.nan)
        with np.errstate(all="ignore"):
            return np.nanmedian(masked, axis=-1).astype(np.float32)
    # Already per-frame: no spatial grid to reduce; just sanitise ≤0 / non-finite.
    valid = np.isfinite(d) & (d > 0)
    if max_depth_m is not None:
        valid &= d <= float(max_depth_m)
    if min_depth_m is not None:
        valid &= d >= float(min_depth_m)
    return np.where(valid, d, np.nan).astype(np.float32)


def scale_from_motion(motion_m: np.ndarray) -> np.ndarray:
    """VIO scale proxy: net metric motion (metres). Identity mapping for now."""
    return np.asarray(motion_m, dtype=np.float32)


def scale_from_depth_change(depth_med: np.ndarray) -> np.ndarray:
    """Depth-implied length scale: ``s_D = |d_last - d_first|`` over the frame axis.

    For a translating camera looking roughly along the motion, apparent depth
    change correlates with metric motion, so ``s_D`` is compared against
    ``s_VIO = motion_m`` (both in metres when depth is metric GT). The
    motion/eps handling lives in ``scale_relative_error``; this is purely the
    depth-side length. NOTE: only physically meaningful on windows with a
    forward motion component (frozen §4.1 ③ applicability note).
    """
    d = np.asarray(depth_med, dtype=np.float64)
    if d.shape[-1] < 2:
        return np.full(d.shape[:-1], np.nan, dtype=np.float32)
    return np.abs(d[..., -1] - d[..., 0]).astype(np.float32)


def scale_relative_error(
    scale_depth: np.ndarray,
    scale_vio: np.ndarray,
    *,
    eps: float = 1e-3,
    min_motion_m: float = 0.5,
    motion_m: Optional[np.ndarray] = None,
    support_ratio: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Per-window |s_D - s_VIO| / max(s_VIO, eps); masks low-motion windows.

    Frozen §4.1 ③: pass if **median** of valid errors ≤ 0.25.

    ``support_ratio`` (dated §4.1 revision 2026-08-05): additionally require
    ``s_D ≥ support_ratio * motion``. Wall-parallel / open-horizon cruises leave
    the |Δ median depth| proxy dead (ŝ_D≈0 → rel≈1) even when heading-aligned;
    those windows are not an applicability domain for the proxy (same spirit as
    the forward-motion note). Default 0.0 preserves legacy behaviour for unit
    tests; the gate pins 0.5.
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
    if float(support_ratio) > 0.0:
        valid = valid & (s_d >= float(support_ratio) * motion)
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
    max_depth_m: Optional[float] = 200.0,
    min_depth_m: Optional[float] = None,
    support_ratio: float = 0.0,
) -> Dict[str, np.ndarray]:
    """End-to-end §4.1 ③ helper on perception-shaped arrays."""
    pos, _dt = integrate_velocity(vel, timestamps, fallback_hz=fallback_hz)
    motion = window_motion_m(pos)
    s_vio = scale_from_motion(motion)
    d_med = depth_median(depth, max_depth_m=max_depth_m, min_depth_m=min_depth_m)
    s_d = scale_from_depth_change(d_med)
    err = scale_relative_error(
        s_d,
        s_vio,
        eps=eps,
        min_motion_m=min_motion_m,
        motion_m=motion,
        support_ratio=support_ratio,
    )
    return {
        "pos": pos,
        "motion_m": motion,
        "scale_vio": s_vio,
        "scale_depth": s_d,
        "depth_median": d_med,
        **err,
    }
