"""Collapse-fix labeling helpers (v3.2 scheme B)."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from experiments.aerial.openfly_actions import OPENFLY_PRIMITIVES, delta_to_nearest_primitive

# Minority classes: turn / vertical / stop — exempt from d_max filtering.
MINORITY_PRIMITIVE_IDS = frozenset({0, 2, 3, 4, 5, 6, 7})
FORWARD_PRIMITIVE_IDS = frozenset({1, 8, 9})

YAW_SCALE = 3.0 / (np.pi / 6)


def scaled_l2(delta: np.ndarray, prim: Sequence[float]) -> float:
    d = np.asarray(delta, dtype=np.float64).reshape(4)
    p = np.asarray(prim, dtype=np.float64).reshape(4)
    scale = np.array([1.0, 1.0, 1.0, YAW_SCALE], dtype=np.float64)
    return float(np.linalg.norm((d - p) * scale))


def delta_nearest_with_dist(delta: np.ndarray) -> Tuple[int, float]:
    """Return (nearest_primitive_id, scaled_L2_distance)."""
    d = np.asarray(delta, dtype=np.float64).reshape(4)
    best_id, best_dist = 0, float("inf")
    for pid, prim in OPENFLY_PRIMITIVES.items():
        dist = scaled_l2(d, prim)
        if dist < best_dist:
            best_dist, best_id = dist, pid
    return best_id, best_dist


def build_ce_mask(
    prim_ids: np.ndarray,
    nearest_dists: np.ndarray,
    *,
    d_max: float,
    is_pad: Optional[np.ndarray] = None,
) -> np.ndarray:
    """True where sample should enter CE loss.

    - padding excluded
    - forward classes filtered by d_max
    - minority classes (turn/vertical/stop) always kept when not pad
    """
    prim_ids = np.asarray(prim_ids, dtype=np.int64).reshape(-1)
    nearest_dists = np.asarray(nearest_dists, dtype=np.float64).reshape(-1)
    if prim_ids.shape != nearest_dists.shape:
        raise ValueError("prim_ids and nearest_dists shape mismatch")
    keep = np.ones(prim_ids.shape[0], dtype=bool)
    if is_pad is not None:
        keep &= ~np.asarray(is_pad, dtype=bool).reshape(-1)
    for i, (pid, dist) in enumerate(zip(prim_ids, nearest_dists)):
        if not keep[i]:
            continue
        if int(pid) in MINORITY_PRIMITIVE_IDS:
            continue
        if dist > float(d_max):
            keep[i] = False
    return keep


def relabel_stop_on_trajectory(
    positions: np.ndarray,
    prim_ids: np.ndarray,
    *,
    goal: np.ndarray,
    r_stop: float = 20.0,
    force_last_stop: bool = True,
) -> np.ndarray:
    """Relabel frames within r_stop of goal as stop(0); optionally force last frame stop."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (T,3), got {pos.shape}")
    out = np.asarray(prim_ids, dtype=np.int64).copy().reshape(-1)
    if out.shape[0] != pos.shape[0]:
        raise ValueError("prim_ids length must match positions")
    g = np.asarray(goal, dtype=np.float64).reshape(3)
    dist = np.linalg.norm(pos - g[None, :], axis=1)
    out[dist < float(r_stop)] = 0
    if force_last_stop and out.size:
        out[-1] = 0
    return out


def prim_ids_from_action_chunk(
    actions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map (T,4) continuous actions → (prim_ids, dists) per step."""
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"actions must be (T,4), got {a.shape}")
    ids, dists = [], []
    for row in a:
        pid, dist = delta_nearest_with_dist(row)
        ids.append(pid)
        dists.append(dist)
    return np.asarray(ids, dtype=np.int64), np.asarray(dists, dtype=np.float64)


def class_weights_from_counts(counts: Iterable[int], n_class: int = 10) -> np.ndarray:
    """Inverse-frequency weights, normalized to mean 1."""
    c = np.asarray(list(counts), dtype=np.float64)
    if c.shape != (n_class,):
        raise ValueError(f"counts must have length {n_class}")
    c = np.maximum(c, 1.0)
    w = c.sum() / (n_class * c)
    return w / w.mean()


# Re-export for callers that already import delta_to_nearest_primitive
__all__ = [
    "MINORITY_PRIMITIVE_IDS",
    "FORWARD_PRIMITIVE_IDS",
    "delta_nearest_with_dist",
    "delta_to_nearest_primitive",
    "build_ce_mask",
    "relabel_stop_on_trajectory",
    "prim_ids_from_action_chunk",
    "class_weights_from_counts",
]
