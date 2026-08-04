"""Batched imagined rollout over a ``LatentDynamics`` (Plan-A parallelism).

This is where sample *volume* comes from: instead of many real envs (the
renderer is single-consumer), we roll a batch of latent states forward through
the fast world model. ``imagine`` takes a batch of encoded start states ``z0``
and an imagination policy, rolls ``horizon`` steps, and returns per-step
latents / actions / rewards / p_coll / done masks for the RL update (V4).

The horizon is capped (spec §9): multi-step WM error compounds, so until the WM
is shown non-divergent the rollout length stays bounded. ``done`` masking stops
reward accrual after a trajectory terminates.

Pure numpy; works with ``StubLatentDynamics`` for offline tests. Imagination
policies expose ``act_latent(z) -> [4]`` (fallback: ``act(z)``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from experiments.aerial.rl.dynamics import LatentDynamics
from experiments.aerial.rl.reward import RewardConfig, reward_terms

MAX_IMAGINATION_HORIZON = 15  # §9 safety cap until WM error shown non-divergent


@dataclass
class ImaginedRollout:
    z: np.ndarray            # [B, H+1, latent_dim]
    actions: np.ndarray      # [B, H, 4]
    rewards: np.ndarray      # [B, H]
    p_coll: np.ndarray       # [B, H]
    progress: np.ndarray     # [B, H]
    done: np.ndarray         # [B, H] bool (cumulative)

    @property
    def returns(self) -> np.ndarray:
        """Per-trajectory summed reward (done steps already zero-masked)."""
        return self.rewards.sum(axis=1)


def _act_latent(policy: Any, z: np.ndarray) -> np.ndarray:
    fn = getattr(policy, "act_latent", None) or getattr(policy, "act", None)
    if not callable(fn):
        raise TypeError("imagination policy must implement act_latent(z) or act(z)")
    return np.asarray(fn(z), dtype=np.float64).reshape(4)


def imagine(
    dynamics: LatentDynamics,
    policy: Any,
    z0_batch: np.ndarray,
    horizon: int,
    *,
    reward_cfg: Optional[RewardConfig] = None,
    max_horizon: int = MAX_IMAGINATION_HORIZON,
) -> ImaginedRollout:
    """Roll ``z0_batch`` forward ``horizon`` steps through ``dynamics``."""
    cfg = reward_cfg or RewardConfig()
    z0 = np.atleast_2d(np.asarray(z0_batch, dtype=np.float64))
    batch, latent_dim = z0.shape
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if horizon > max_horizon:
        raise ValueError(
            f"horizon {horizon} exceeds cap {max_horizon} (spec §9: WM multi-step "
            "error is unbounded until validated); raise max_horizon explicitly"
        )

    zs = np.zeros((batch, horizon + 1, latent_dim), dtype=np.float64)
    acts = np.zeros((batch, horizon, 4), dtype=np.float64)
    rews = np.zeros((batch, horizon), dtype=np.float64)
    pcs = np.zeros((batch, horizon), dtype=np.float64)
    progs = np.zeros((batch, horizon), dtype=np.float64)
    dones = np.zeros((batch, horizon), dtype=bool)
    zs[:, 0] = z0

    alive = np.ones(batch, dtype=bool)
    for t in range(horizon):
        for b in range(batch):
            if not alive[b]:
                zs[b, t + 1] = zs[b, t]
                dones[b, t] = True
                continue
            a = _act_latent(policy, zs[b, t])
            out = dynamics.step(zs[b, t], a)
            zs[b, t + 1] = np.asarray(out.z_next, dtype=np.float64).reshape(latent_dim)
            acts[b, t] = a
            pcs[b, t] = out.p_coll
            progs[b, t] = out.progress
            maneuver = float(np.linalg.norm(a))
            r = reward_terms(out.progress, out.p_coll, maneuver, cfg)["reward"]
            # Mirror NavigationReward.step: arrival earns the same success bonus,
            # so imagined and real returns are on one scale (spec reward §4.5).
            if getattr(out, "arrived", False):
                r += cfg.success_bonus
            rews[b, t] = r
            if out.done:
                alive[b] = False
                dones[b, t] = True

    return ImaginedRollout(z=zs, actions=acts, rewards=rews, p_coll=pcs, progress=progs, done=dones)
