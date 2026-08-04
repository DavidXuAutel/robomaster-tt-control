"""Observation contract for the aerial RL env.

Design boundary (spec §1.2 — external perception is RGB-ONLY):

  POLICY / WORLD-MODEL INPUTS
    * ``rgb``      monocular ego RGB (224x224), the only exteroceptive input
    * ``proprio4`` (x, y, z, yaw) — matches the FastWAM ``state`` dim = 4

  SUPERVISION / REWARD-ONLY (NOT fed to the policy)
    * ``depth``    dense metric depth GT — target for the [1b] depth head + a
                   collision-cost signal; the deployed policy predicts it, it is
                   never handed the ground truth.
    * ``imu``      angular velocity / linear acceleration — VIO/[1c] supervision.
    * ``collided`` ground-truth contact — reward/termination signal only.

Keeping the split explicit here is the guardrail that stops depth/IMU GT from
silently leaking into the policy graph later.

World frame — ``state`` and any goal are **+up ENU-style** (z increases upward,
matching the OpenFly ascend primitive #4 = +dz and ``apply_body_delta``'s
``wz = dz``). AirSim reports NED (z down); the real env negates z / vz at the
readback boundary (``airsim_env.observe_state``) and negates z when setting the
pose, so real and mock observations share one convention. Nothing downstream
(reward progress, heuristic vertical control) should ever see raw NED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class Observation:
    """One environment observation.

    ``state`` is the full kinematic vector [x, y, z, vx, vy, vz, yaw] in the
    **+up world frame** (z up); the real env converts AirSim NED on the way in
    (see module docstring). ``proprio4()`` slices the 4-D (x, y, z, yaw) the
    FastWAM policy/world-model actually consume.
    """

    rgb: np.ndarray                      # [H, W, 3] uint8 — policy input
    state: np.ndarray                    # [7] float32 x,y,z,vx,vy,vz,yaw
    collided: bool = False               # supervision / termination only
    depth: Optional[np.ndarray] = None   # [H, W] float32 — supervision only
    imu: Dict[str, Any] = field(default_factory=dict)  # supervision only
    t: float = 0.0                       # wall-clock capture time (s)
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rgb = np.ascontiguousarray(np.asarray(self.rgb, dtype=np.uint8))
        self.state = np.asarray(self.state, dtype=np.float32).reshape(-1)
        if self.state.shape[0] != 7:
            raise ValueError(
                f"state must be [x,y,z,vx,vy,vz,yaw] (7,), got {self.state.shape}"
            )
        if self.depth is not None:
            self.depth = np.asarray(self.depth, dtype=np.float32)

    @property
    def position(self) -> np.ndarray:
        return self.state[:3].astype(np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6].astype(np.float64)

    @property
    def yaw(self) -> float:
        return float(self.state[6])

    def proprio4(self) -> np.ndarray:
        """The 4-D state the FastWAM policy consumes: (x, y, z, yaw)."""
        return np.array(
            [self.state[0], self.state[1], self.state[2], self.state[6]],
            dtype=np.float32,
        )


def depth_sanity_detail(depth: Optional[np.ndarray]) -> Dict[str, Any]:
    """Build a ``sim_verify.lib.sanity.depth_ok``-compatible detail dict.

    Lets the env reuse the already-validated depth gate (dense + enough finite
    pixels + non-trivial dynamic range) instead of re-deriving the checks.
    """
    if depth is None:
        return {"dense": False, "n_floats": 0, "n_finite": 0}
    arr = np.asarray(depth, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    finite = arr[np.isfinite(arr)]
    detail: Dict[str, Any] = {
        "dense": n > 0,
        "n_floats": n,
        "n_finite": int(finite.size),
    }
    if finite.size:
        detail["finite_min"] = float(finite.min())
        detail["finite_max"] = float(finite.max())
        detail["finite_std"] = float(finite.std())
    else:
        detail["finite_min"] = detail["finite_max"] = detail["finite_std"] = None
    return detail
