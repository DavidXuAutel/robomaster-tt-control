from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

# OpenFly discrete IDs (official):
# 0 stop, 1 fwd3, 2 left30, 3 right30, 4 up3, 5 down3, 6 left3, 7 right3, 8 fwd6, 9 fwd9
OPENFLY_PRIMITIVES: Dict[int, Tuple[float, float, float, float]] = {
    0: (0.0, 0.0, 0.0, 0.0),
    1: (3.0, 0.0, 0.0, 0.0),
    2: (0.0, 0.0, 0.0, math.pi / 6),
    3: (0.0, 0.0, 0.0, -math.pi / 6),
    4: (0.0, 0.0, 3.0, 0.0),
    5: (0.0, 0.0, -3.0, 0.0),
    6: (0.0, 3.0, 0.0, 0.0),   # body +Y strafe left (confirm vs OpenFly bridge axes)
    7: (0.0, -3.0, 0.0, 0.0),
    8: (6.0, 0.0, 0.0, 0.0),
    9: (9.0, 0.0, 0.0, 0.0),
}

_PADDING = {-1, -2}


def is_padding_action(action_id: int) -> bool:
    return int(action_id) in _PADDING


def primitive_to_delta(action_id: int) -> np.ndarray:
    aid = int(action_id)
    if is_padding_action(aid):
        raise ValueError(f"padding action {aid} has no delta")
    if aid not in OPENFLY_PRIMITIVES:
        raise KeyError(f"unknown OpenFly action id {aid}")
    return np.asarray(OPENFLY_PRIMITIVES[aid], dtype=np.float64)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def pos_yaw_to_body_delta(
    pos0: np.ndarray,
    yaw0: float,
    pos1: np.ndarray,
    yaw1: float,
) -> np.ndarray:
    """World (pos,yaw) pair → body-frame (dx, dy, dz, dyaw)."""
    p0 = np.asarray(pos0, dtype=np.float64).reshape(3)
    p1 = np.asarray(pos1, dtype=np.float64).reshape(3)
    d_world = p1 - p0
    c, s = math.cos(yaw0), math.sin(yaw0)
    # body_x forward, body_y left
    dx = c * d_world[0] + s * d_world[1]
    dy = -s * d_world[0] + c * d_world[1]
    dz = d_world[2]
    dyaw = wrap_angle(float(yaw1) - float(yaw0))
    return np.array([dx, dy, dz, dyaw], dtype=np.float64)


def delta_to_nearest_primitive(delta: np.ndarray) -> int:
    """Map continuous 4D to nearest OpenFly primitive (L2 on scaled units)."""
    d = np.asarray(delta, dtype=np.float64).reshape(4)
    # scale yaw so ~30deg ~ comparable to ~3m
    scale = np.array([1.0, 1.0, 1.0, 3.0 / (math.pi / 6)], dtype=np.float64)
    best_id, best_dist = 0, float("inf")
    for pid, prim in OPENFLY_PRIMITIVES.items():
        p = np.asarray(prim, dtype=np.float64)
        dist = float(np.linalg.norm((d - p) * scale))
        if dist < best_dist:
            best_dist, best_id = dist, pid
    return best_id
