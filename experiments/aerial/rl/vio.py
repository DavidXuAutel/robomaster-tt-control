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


# --------------------------------------------------------------------------- #
# Reprojection scale-consistency estimator (candidate §4.1 revision 2026-08-10) #
#                                                                             #
# The band-median proxy above (ŝ_D = |Δ median depth|) equates "change in     #
# band-median depth" with "forward displacement", true only for fronto-parallel #
# axial translation into a band-filling surface. Band pixel churn injects a    #
# per-window bias that eats the whole 0.25 budget: on local approach corpora   #
# the GT-oracle (perfect depth) forward-only median is ~0.24 — no depth model  #
# can win. This estimator replaces aggregate medians with per-pixel geometric  #
# correspondence: backproject frame-0 depth to 3D via intrinsics, transform by #
# the GT metric pose (Δposition, Δyaw), reproject into frame N, and compare the #
# predicted planar depth against the observed depth at the matched pixel. With #
# GT depth + GT pose this is a geometric identity → GT-oracle → 0 (validated at #
# ~0.005 on d12/d18/d25 probe corpora, vs 0.19–0.36 for the band-median proxy). #
# It therefore returns the 0.25 budget to the depth model's real error; the     #
# 0.25 red line is UNCHANGED — what is fixed is that the proxy wasn't measuring #
# scale. Feeding predicted D̂ (instead of GT) tests whether D̂'s metric scale is #
# consistent with metric camera motion, which is exactly V0 signal ③.          #
#                                                                             #
# NOT yet wired into the authoritative gate — dormant until a dated §4.1        #
# spec revision + re-freeze. Old helpers above are untouched for A/B.           #
# --------------------------------------------------------------------------- #
def intrinsics_from_hfov(
    width: int, height: int, hfov_deg: float = 90.0
) -> Tuple[float, float, float, float]:
    """Pinhole (fx, fy, cx, cy) from image size + horizontal FOV.

    AirSim square capture (224×224, hfov 90°) → fx=fy=112, cx=cy=111.5.
    Vertical FOV assumed equal for a square sensor (fy=fx).
    """
    fx = (float(width) / 2.0) / np.tan(np.radians(float(hfov_deg)) / 2.0)
    fy = fx * (float(height) / float(width)) if width != height else fx
    return float(fx), float(fx if height == width else fy), (width - 1) / 2.0, (height - 1) / 2.0


def _R_body_to_world(yaw: float) -> np.ndarray:
    """AirSim NED yaw (about Down axis): body(fwd,right,down) → world(N,E,D)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def reproject_pair_rel_err(
    depth0: np.ndarray,
    pos0: np.ndarray,
    yaw0: float,
    depth_n: np.ndarray,
    pos_n: np.ndarray,
    yaw_n: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth_m: float = 1.0,
    max_depth_m: float = 40.0,
    min_matched: int = 50,
) -> float:
    """Robust-median relative depth error between two posed frames.

    Backproject navigational-band pixels of ``depth0`` to 3D (camera 0 → world),
    transform into camera N by the GT metric pose, reproject to pixels, and
    compare the predicted planar depth (Z in camera N) against the observed
    ``depth_n`` at the matched pixel. Camera is body-forward: camera axes
    (right X, down Y, forward Z) map to body (fwd x, right y, down z) as
    Xc=y_body, Yc=z_body, Zc=x_body; DepthPlanar is Zc. Returns NaN if fewer
    than ``min_matched`` band-valid correspondences survive bounds/occlusion.
    """
    d0 = np.asarray(depth0, dtype=np.float64)
    dn = np.asarray(depth_n, dtype=np.float64)
    hh, ww = d0.shape[-2], d0.shape[-1]
    vv, uu = np.mgrid[0:hh, 0:ww]
    m = np.isfinite(d0) & (d0 >= min_depth_m) & (d0 <= max_depth_m)
    if int(np.count_nonzero(m)) < int(min_matched):
        return float("nan")
    u = uu[m].astype(np.float64)
    v = vv[m].astype(np.float64)
    z = d0[m]
    # backproject: pixel → camera 0 → body 0 (fwd,right,down) → world
    body0 = np.stack([z, (u - cx) / fx * z, (v - cy) / fy * z], axis=1)
    xw = np.asarray(pos0, dtype=np.float64)[None, :] + body0 @ _R_body_to_world(yaw0).T
    # world → body N (multiply by R on the right == R^T on the left)
    body_n = (xw - np.asarray(pos_n, dtype=np.float64)[None, :]) @ _R_body_to_world(yaw_n)
    z_pred = body_n[:, 0]                                  # forward = planar depth in N
    good = z_pred > 1e-3
    if int(np.count_nonzero(good)) < int(min_matched):
        return float("nan")
    un = cx + fx * body_n[good, 1] / z_pred[good]
    vn = cy + fy * body_n[good, 2] / z_pred[good]
    zp = z_pred[good]
    ui, vi = np.round(un).astype(int), np.round(vn).astype(int)
    inb = (ui >= 0) & (ui < ww) & (vi >= 0) & (vi < hh)
    if int(np.count_nonzero(inb)) < int(min_matched):
        return float("nan")
    z_obs = dn[vi[inb], ui[inb]]
    zp = zp[inb]
    ok = np.isfinite(z_obs) & (z_obs >= min_depth_m) & (z_obs <= max_depth_m)
    if int(np.count_nonzero(ok)) < int(min_matched):
        return float("nan")
    return float(np.median(np.abs(zp[ok] - z_obs[ok]) / z_obs[ok]))


def reproject_scale_error(
    depth: np.ndarray,
    positions: np.ndarray,
    yaw: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth_m: float = 1.0,
    max_depth_m: float = 40.0,
    min_motion_m: float = 0.5,
    min_matched: int = 50,
) -> Dict[str, np.ndarray]:
    """Batched ③ reprojection estimator over windows ``[B, L, H, W]``.

    ``positions [B,L,3]`` and ``yaw [B,L]`` are the GT metric camera poses
    (proprio). Reprojects each window's first frame into its last frame. Pass =
    median of valid ``rel_err`` ≤ ``scale_rel_err_max`` (0.25, unchanged); the
    estimator's own GT-oracle bias is ~0.005, so the budget belongs to D̂ error.
    """
    depth = np.asarray(depth, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    if depth.ndim != 4:
        raise ValueError(f"depth must be [B,L,H,W], got {depth.shape}")
    b = depth.shape[0]
    rel = np.full(b, np.nan, dtype=np.float64)
    valid = np.zeros(b, dtype=bool)
    for i in range(b):
        c0, cn = positions[i, 0], positions[i, -1]
        motion = float(np.linalg.norm(cn - c0))
        if motion < float(min_motion_m):
            continue
        r = reproject_pair_rel_err(
            depth[i, 0], c0, float(yaw[i, 0]),
            depth[i, -1], cn, float(yaw[i, -1]),
            fx=fx, fy=fy, cx=cx, cy=cy,
            min_depth_m=min_depth_m, max_depth_m=max_depth_m, min_matched=min_matched,
        )
        if np.isfinite(r):
            rel[i] = r
            valid[i] = True
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
