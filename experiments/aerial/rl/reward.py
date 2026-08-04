"""Composite navigation reward (spec §4.5).

The spec gives the *shape*, not a scalar:

    reward = w_prog * progress
           - w_coll * collision_risk
           - w_man  * maneuver_cost

  * ``progress``       = decrease in distance-to-goal since the last step (m).
  * ``collision_risk`` = 1.0 on real contact (``obs.collided``); otherwise the
    world-model's predicted ``p_coll`` in ∈[0,1] (imagined rollouts).
  * ``maneuver_cost``  = L2 norm of the executed body delta (aggressive-maneuver
    penalty; keeps motions smooth).

``NavigationReward`` is stateful: it remembers the previous distance so the
collector can call ``.step(obs, action)`` once per env step. ``reward_terms``
is a pure function for the imagination side (given explicit distances / p_coll).

Arrival radius: ``RewardConfig.success_dist_m`` is the *online* arrival /
termination gate and defaults to ``OPENFLY_SUCCESS_DIST_M`` only as a fallback —
the training entrypoint sets a tighter value (see ``configs/aerial_rl.yaml``).
The 20 m ``OPENFLY_SUCCESS_DIST_M`` is the loose *eval SR metric* radius and is
intentionally NOT reused as the per-step termination threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from experiments.aerial.eval.metrics import OPENFLY_SUCCESS_DIST_M
from experiments.aerial.rl.env.obs import Observation


@dataclass
class RewardConfig:
    w_progress: float = 1.0
    w_collision: float = 10.0
    w_maneuver: float = 0.01
    success_dist_m: float = float(OPENFLY_SUCCESS_DIST_M)
    success_bonus: float = 10.0


def reward_terms(
    progress: float,
    collision_risk: float,
    maneuver_cost: float,
    cfg: RewardConfig = RewardConfig(),
) -> Dict[str, float]:
    """Pure term breakdown + scalar reward. Used by both real and imagined paths."""
    r = (
        cfg.w_progress * float(progress)
        - cfg.w_collision * float(collision_risk)
        - cfg.w_maneuver * float(maneuver_cost)
    )
    return {
        "reward": float(r),
        "progress": float(progress),
        "collision_risk": float(collision_risk),
        "maneuver_cost": float(maneuver_cost),
    }


class NavigationReward:
    """Stateful per-episode reward: progress toward ``goal`` − risk − maneuver."""

    def __init__(self, goal: Optional[np.ndarray], cfg: Optional[RewardConfig] = None) -> None:
        self.cfg = cfg or RewardConfig()
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float64).reshape(3)
        self._prev_dist: Optional[float] = None

    def reset(self, goal: Optional[np.ndarray], start_pos: np.ndarray) -> None:
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float64).reshape(3)
        self._prev_dist = self._dist(np.asarray(start_pos, dtype=np.float64).reshape(3))

    def _dist(self, pos: np.ndarray) -> Optional[float]:
        if self._goal is None:
            return None
        return float(np.linalg.norm(pos - self._goal))

    def step(
        self,
        obs: Observation,
        action: np.ndarray,
        p_coll: Optional[float] = None,
    ) -> tuple[float, bool, Dict[str, float]]:
        """Return ``(reward, done, terms)`` for one real env transition."""
        pos = obs.position
        dist = self._dist(pos)
        progress = 0.0
        if dist is not None and self._prev_dist is not None:
            progress = self._prev_dist - dist
        self._prev_dist = dist

        collision_risk = 1.0 if obs.collided else float(p_coll or 0.0)
        maneuver_cost = float(np.linalg.norm(np.asarray(action, dtype=np.float64)))
        terms = reward_terms(progress, collision_risk, maneuver_cost, self.cfg)

        arrived = dist is not None and dist < self.cfg.success_dist_m
        if arrived:
            terms["reward"] += self.cfg.success_bonus
        done = bool(obs.collided or arrived)
        terms["arrived"] = float(arrived)
        return terms["reward"], done, terms
