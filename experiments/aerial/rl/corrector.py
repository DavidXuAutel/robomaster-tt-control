"""``SerialCorrectorLoop`` — Plan-A orchestration (collect → WM → imagine-RL).

One serial pass per iteration:

    collector.collect(...)                 # V0: real, runnable now
    [GATE V1] dynamics.update(windows)     # world-model training — no-op stub
    [GATE V4] imagine + policy/value update # RL in imagination — no-op stub

The two learning stages are real methods guarded by ``enable_wm_update`` /
``enable_policy_update`` flags that default OFF (spec ladder: "未过关不叠加下一阶段").
Running the loop today exercises the full V0 collection path and cleanly no-ops
the gated stages, logging why. Flip a flag (once its milestone passes) and the
insertion point is already wired: WM training consumes buffer windows; the RL
update consumes ``imagine(...)`` trajectories.

``smoke=True`` runs exactly one collection pass and returns its stats — the
on-4090 smoke test entrypoint.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.collector import CollectStats, RolloutCollector
from experiments.aerial.rl.dynamics import LatentDynamics
from experiments.aerial.rl.imagination import imagine

logger = logging.getLogger(__name__)


@dataclass
class CorrectorConfig:
    iterations: int = 10
    episodes_per_iter: int = 1
    # V1/V4 gates — OFF until each milestone passes.
    enable_wm_update: bool = False
    enable_policy_update: bool = False
    strict_gates: bool = False        # raise (vs skip+log) if a gate is off
    # WM-training window sampling (used once enable_wm_update flips on).
    wm_batch: int = 32
    wm_window: int = 8
    # Imagination-RL params (used once enable_policy_update flips on).
    imagine_batch: int = 64
    imagine_horizon: int = 10
    smoke: bool = False


@dataclass
class IterationReport:
    collect: CollectStats
    wm: Dict[str, Any] = field(default_factory=dict)
    rl: Dict[str, Any] = field(default_factory=dict)


class SerialCorrectorLoop:
    def __init__(
        self,
        collector: RolloutCollector,
        buffer: ReplayBuffer,
        dynamics: LatentDynamics,
        *,
        imagination_policy: Optional[Any] = None,
        config: Optional[CorrectorConfig] = None,
        episodes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.collector = collector
        self.buffer = buffer
        self.dynamics = dynamics
        self.imagination_policy = imagination_policy
        self.config = config or CorrectorConfig()
        self.episodes = episodes

    def run(self) -> List[IterationReport]:
        if self.config.smoke:
            stats = self.collector.collect(1, episodes=self.episodes)
            logger.info("smoke collect: %d steps @ %.1f Hz", stats.steps, stats.achieved_hz)
            return [IterationReport(collect=stats, wm={"skipped": True}, rl={"skipped": True})]

        reports: List[IterationReport] = []
        for it in range(self.config.iterations):
            stats = self.collector.collect(self.config.episodes_per_iter, episodes=self.episodes)
            wm = self._update_world_model()
            rl = self._update_policy()
            logger.info(
                "iter %d: %d steps @ %.1f Hz | wm=%s | rl=%s",
                it, stats.steps, stats.achieved_hz, wm.get("status", "?"), rl.get("status", "?"),
            )
            reports.append(IterationReport(collect=stats, wm=wm, rl=rl))
        return reports

    # -- GATE V1: world-model training -----------------------------------
    def _update_world_model(self) -> Dict[str, Any]:
        if not self.config.enable_wm_update:
            msg = "world-model training is V1-gated (enable_wm_update=False)"
            if self.config.strict_gates:
                raise RuntimeError(msg)
            return {"status": "skipped", "reason": msg}
        try:
            windows = self.buffer.sample_windows(self.config.wm_batch, self.config.wm_window)
        except ValueError as exc:
            return {"status": "skipped", "reason": f"insufficient data: {exc}"}
        result = self.dynamics.update(windows)
        return {"status": "updated", **result}

    # -- GATE V4: imagination actor-critic update ------------------------
    def _update_policy(self) -> Dict[str, Any]:
        if not self.config.enable_policy_update:
            msg = "imagination RL update is V4-gated (enable_policy_update=False)"
            if self.config.strict_gates:
                raise RuntimeError(msg)
            return {"status": "skipped", "reason": msg}
        if self.imagination_policy is None:
            return {"status": "skipped", "reason": "no imagination_policy provided"}
        # Encode a batch of real start states, imagine forward, then hand the
        # trajectories to the (future) actor-critic optimizer. The optimizer
        # itself is the V4 deliverable; here we only produce its inputs.
        try:
            transitions = self.buffer.sample(self.config.imagine_batch)
        except ValueError as exc:
            return {"status": "skipped", "reason": f"insufficient data: {exc}"}
        z0 = np.stack([self.dynamics.encode(t.obs) for t in transitions], axis=0)
        rollout = imagine(
            self.dynamics, self.imagination_policy, z0, self.config.imagine_horizon,
        )
        # >>> V4 INSERTION POINT: actor_critic.update(rollout) <<<
        return {
            "status": "imagined",
            "batch": int(z0.shape[0]),
            "horizon": int(self.config.imagine_horizon),
            "mean_return": float(rollout.returns.mean()),
            "note": "trajectories produced; actor-critic update is the V4 deliverable",
        }
