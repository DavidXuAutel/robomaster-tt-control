"""Numerical sanity checks for T2 probe details.

API-readable is not enough: IMU all-zeros, depth all-inf, or a single Scene
grab must not be marked as capability PASS. These helpers are pure (no AirSim)
so they can be unit-tested offline.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence, Tuple


def _finite(xs: Iterable[Any]) -> bool:
    try:
        return all(math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def _mag(xs: Sequence[Any]) -> float:
    return math.sqrt(sum(float(x) ** 2 for x in xs))


def imu_ok(detail: dict) -> Tuple[bool, str]:
    """Reject non-finite or near-zero linear acceleration (API stub / dead sensor)."""
    ang = detail.get("ang_vel") or []
    lin = detail.get("lin_acc") or []
    if len(ang) < 3 or len(lin) < 3:
        return False, "imu vectors shorter than 3"
    if not (_finite(ang) and _finite(lin)):
        return False, "non-finite imu values"
    mag = _mag(lin)
    # Hover / ground: |a| ~ g; CV stub often returns exact zeros.
    if mag < 0.5:
        return False, f"lin_acc near zero (mag={mag:.4f}) — readable but not usable"
    return True, f"lin_acc_mag={mag:.3f}"


def altitude_ok(detail: dict, key: str = "altitude") -> Tuple[bool, str]:
    """Barometer/GPS altitude must be finite (zero may be valid at sea level)."""
    if key not in detail and "alt" in detail:
        key = "alt"
    if key not in detail:
        return False, f"missing {key}"
    try:
        v = float(detail[key])
    except (TypeError, ValueError):
        return False, f"non-numeric {key}={detail[key]!r}"
    if not math.isfinite(v):
        return False, f"non-finite {key}={v}"
    return True, f"{key}={v:.3f}"


def depth_ok(detail: dict) -> Tuple[bool, str]:
    """Dense depth with enough finite pixels and non-trivial dynamic range."""
    if not detail.get("dense"):
        return False, "not dense (n_floats != w*h)"
    n = int(detail.get("n_floats") or 0)
    n_fin = int(detail.get("n_finite") or 0)
    if n <= 0:
        return False, "empty depth"
    frac = n_fin / float(n)
    if frac < 0.5:
        return False, f"too few finite depths ({frac:.1%} < 50%)"
    dmin = detail.get("finite_min")
    dmax = detail.get("finite_max")
    dstd = detail.get("finite_std")
    try:
        span = float(dmax) - float(dmin)
        std = float(dstd)
    except (TypeError, ValueError):
        return False, "missing finite_min/max/std"
    if not (math.isfinite(span) and math.isfinite(std)):
        return False, "non-finite depth stats"
    if span < 0.5 and std < 0.1:
        return False, f"depth nearly constant (span={span:.3f}, std={std:.3f})"
    return True, f"finite={frac:.0%} span={span:.2f}m std={std:.2f}"


def continuous_ok(detail: dict) -> Tuple[bool, str]:
    """L2f: monotonic timestamps, min fps, and some temporal change after motion."""
    if not detail.get("monotonic"):
        return False, "timestamps not monotonic"
    fps = float(detail.get("fps") or 0.0)
    min_fps = float(detail.get("min_fps_required") or 5.0)
    if fps < min_fps:
        return False, f"fps={fps:.2f} < required {min_fps:.1f}"
    if not detail.get("frames_differ"):
        return False, "consecutive frames identical even after motion cue — no temporal signal"
    return True, f"fps={fps:.2f} mean_abs_diff={detail.get('mean_abs_diff')}"


def depth_rate_ok(detail: dict) -> Tuple[bool, str]:
    """L2d-rate: dense-depth CAPTURE is fast enough + monotonic for V0 collection.

    Distinct from ``depth_ok`` (which gates one frame's density / dynamic range):
    this gates the *capture rate*. The cross-net DepthPlanar path runs ~0.7 Hz —
    plenty for a one-shot sanity grab, far too slow for the per-frame depth the V0
    [1b] depth head / [1c] VIO need. Running the probe on the 4090 loopback should
    clear tens of Hz. Without this gate a "Fork A" verdict certifies only that
    depth *exists*, not that it is fast enough to collect a V0 perception dataset.
    """
    if not detail.get("monotonic"):
        return False, "depth timestamps not monotonic"
    fps = float(detail.get("fps") or 0.0)
    min_fps = float(detail.get("min_fps_required") or 5.0)
    if fps < min_fps:
        return (
            False,
            f"depth fps={fps:.2f} < required {min_fps:.1f} — cross-net link? "
            "collect on the renderer host (127.0.0.1)",
        )
    return True, f"depth fps={fps:.2f}"
