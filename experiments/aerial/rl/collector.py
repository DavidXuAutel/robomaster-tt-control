"""``RolloutCollector`` — the serial real-env worker (Plan A).

One env instance (the renderer is single-consumer) driven at ~``step_hz``: reset
→ loop {policy → step → reward} → push a full episode to the ``ReplayBuffer``.
This is the only place that touches the real renderer; sample *volume* for
learning comes from imagination, not from parallel envs.

Policy dispatch is duck-typed (mirrors ``collect_dagger._predict_delta``):

  1. ``policy.act(obs) -> [4] body delta``            (RL / continuous policy)
  2. ``policy.predict_delta(rgb, state, instr)``      (delta-native policy)
  3. ``policy.predict_primitive(rgb, state, instr)``  → ``primitive_to_delta``
     (the existing ``FastWAMAerialPolicy`` / ``ReplayPolicy`` primitive path)

Achieved Hz is measured and logged every episode; a warning fires if it drops
below the configured target so the ~30 Hz Plan-A assumption is validated on real
hardware rather than assumed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from experiments.aerial.openfly_actions import primitive_to_delta
from experiments.aerial.rl.buffer import Episode, ReplayBuffer, Transition
from experiments.aerial.rl.env.action import DEFAULT_STEP_HZ, body_delta_limits, clip_body_delta
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.reward import NavigationReward, RewardConfig
from experiments.aerial.rl.safety import SafetyShield

logger = logging.getLogger(__name__)


def act_delta(
    policy: Any,
    obs: Observation,
    instruction: str,
    limits: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resolve any supported policy to a finite, clipped 4-D body delta.

    ``limits`` is the per-step displacement cap for the env's control rate
    (``body_delta_limits(dt)``); defaults to the 30 Hz continuous cap. NOTE: a
    discrete-primitive policy returns a macro-sized delta (e.g. fwd 9 m) which
    this clips to a single per-step increment — driving macro primitives
    faithfully needs a multi-step executor, out of scope for the V0 skeleton.
    """
    act = getattr(policy, "act", None)
    if callable(act):
        # Hand the RGB-only view, never the full Observation: depth/IMU/velocity/
        # collision GT must not reach the policy graph (spec §1.2 boundary).
        raw = act(obs.policy_view())
    else:
        predict_delta = getattr(policy, "predict_delta", None)
        if callable(predict_delta):
            raw = predict_delta(obs.rgb, obs.proprio4(), instruction)
        else:
            primitive = int(policy.predict_primitive(obs.rgb, obs.proprio4(), instruction))
            raw = primitive_to_delta(primitive)
    return clip_body_delta(np.asarray(raw, dtype=np.float64), limits)


@dataclass
class CollectStats:
    episodes: int = 0
    steps: int = 0
    seconds: float = 0.0
    interventions: int = 0
    # Episodes dropped at reset because the vehicle spawned already colliding
    # (spawn-inside-geometry). Not counted in `episodes`; never reach the buffer.
    skipped: int = 0
    returns: List[float] = field(default_factory=list)

    @property
    def achieved_hz(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0


class RolloutCollector:
    def __init__(
        self,
        env: Any,
        policy: Any,
        buffer: ReplayBuffer,
        *,
        reward_cfg: Optional[RewardConfig] = None,
        safety: Optional[SafetyShield] = None,
        max_steps: int = 200,
        target_hz: float = 30.0,
        on_episode: Optional[Callable[[Episode, CollectStats], None]] = None,
        skip_reset_collision: bool = True,
    ) -> None:
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.reward_cfg = reward_cfg or RewardConfig()
        self.safety = safety
        self.max_steps = int(max_steps)
        self.target_hz = float(target_hz)
        # Drop episodes whose reset spawns the vehicle already in collision
        # (inside geometry): no action has been taken, so it's a spawn artifact,
        # not a learnable trajectory. Skipped before any step / buffer write.
        self.skip_reset_collision = bool(skip_reset_collision)
        # Optional sink invoked with every completed episode (e.g. persist to
        # disk). None -> collector stays purely in-memory (offline tests / V0).
        self.on_episode = on_episode

    def collect_episode(self, episode: Optional[Dict[str, Any]] = None) -> tuple[Episode, CollectStats]:
        instruction = str((episode or {}).get("gpt_instruction", ""))
        obs = self.env.reset(episode)
        # Entry guard: a vehicle already colliding at reset spawned inside
        # geometry. Skip before any step so it never pollutes the buffer/dataset
        # as a 1-step instant crash. (`collided` is populated at reset by both
        # backends — airsim_env.observe() / mock bounds check.)
        if self.skip_reset_collision and bool(getattr(obs, "collided", False)):
            logger.warning(
                "reset spawned in collision — skipping episode "
                "(spawn-inside-geometry; start pose may need resampling)"
            )
            return [], CollectStats(episodes=0, skipped=1)
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        reward = NavigationReward(getattr(self.env, "goal", None), self.reward_cfg)
        reward.reset(getattr(self.env, "goal", None), obs.position)

        transitions: List[Transition] = []
        stats = CollectStats(episodes=1)
        # Per-step displacement cap for this env's control rate (keeps the clip
        # consistent with what env.step will apply).
        step_hz = float(getattr(getattr(self.env, "config", None), "step_hz", DEFAULT_STEP_HZ))
        limits = body_delta_limits(1.0 / step_hz)
        t_start = time.perf_counter()

        for _ in range(self.max_steps):
            action = act_delta(self.policy, obs, instruction, limits)
            intervened = False
            # Safety shield sits ABOVE the learned policy (spec §2#6). Stub
            # returns no-override today; when wired it swaps in a safe action.
            #
            # V0: no online WM, so the shield sees obs only (D̂ / τ / p_coll all
            # absent -> ThresholdSafetyShield safely never fires). V1 TODO: once
            # the fast latent WM steps online, pass its DynamicsOutput here as
            # should_override(obs, wm_out=...) so the p_coll trigger is live —
            # that path is unit-tested (test_followups) but not yet exercised in
            # this collection loop.
            if self.safety is not None and self.safety.should_override(obs):
                action = clip_body_delta(self.safety.override_action(obs), limits)
                intervened = True

            next_obs, info = self.env.step(action)
            r, done, terms = reward.step(next_obs, action)
            transitions.append(
                Transition(
                    obs=obs, action=action, reward=r, done=done,
                    next_obs=next_obs,
                    info={**info, **terms, "intervention": intervened},
                )
            )
            stats.steps += 1
            stats.interventions += int(intervened)
            obs = next_obs
            if done:
                break

        stats.seconds = time.perf_counter() - t_start
        stats.returns.append(float(sum(t.reward for t in transitions)))
        if self.target_hz > 0 and stats.achieved_hz < self.target_hz * 0.8:
            logger.warning(
                "collector achieved %.1f Hz (< %.1f Hz target) over %d steps",
                stats.achieved_hz, self.target_hz, stats.steps,
            )
        self.buffer.add_episode(transitions)
        if self.on_episode is not None:
            self.on_episode(transitions, stats)
        return transitions, stats

    def collect(self, num_episodes: int = 1, episodes: Optional[List[Dict[str, Any]]] = None) -> CollectStats:
        total = CollectStats()
        for i in range(int(num_episodes)):
            ep = None
            if episodes:
                ep = episodes[i % len(episodes)]
            _, s = self.collect_episode(ep)
            total.episodes += s.episodes
            total.steps += s.steps
            total.seconds += s.seconds
            total.interventions += s.interventions
            total.skipped += s.skipped
            total.returns.extend(s.returns)
        return total
