"""Mac (torch-free) tests for the V0 ②/④ rollout runners + gate assembly.

② is exercised on the mock env (goal-seeking ``HeuristicPolicy`` genuinely
closes distance, random does not). The mock has no obstacles, so ④ is exercised
on a purpose-built wall stub with a pessimistic GT-proxy depth predictor — the
point of the Mac test is the *wiring* (predictor → obs.info → shield override →
fewer near-collision steps), not a renderer-grade physics check. The real ④
pass happens on airsim with a trained depth head.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.aerial.rl import v0_metrics as metrics
from experiments.aerial.rl import v0_rollout_eval as rollout
from experiments.aerial.rl._v0_gate import assemble_verdict
from experiments.aerial.rl.env.mock_env import MockAirSimDroneEnv, MockEnvConfig
from experiments.aerial.rl.env.obs import Observation
from experiments.aerial.rl.train_rl import HeuristicPolicy


# --------------------------------------------------------------------------- #
# ② progress-vs-random on the mock env                                         #
# --------------------------------------------------------------------------- #
def test_signal2_heuristic_beats_random_on_mock():
    env = MockAirSimDroneEnv(MockEnvConfig(step_hz=5.0))
    starts = rollout.make_start_episodes(16, seed=0)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    rnd = rollout.RandomActionPolicy(seed=0)
    prog = rollout.run_progress_eval(env, policy, rnd, starts, max_steps=200)

    s2 = metrics.check_progress_vs_random(
        prog["policy_progress_sums"], prog["random_progress_sums"],
        prog["policy_final_dists"], prog["random_final_dists"],
    )
    assert s2["ok"], s2
    # Sanity: the goal-seeker really does close more distance than random.
    assert s2["mean_progress_policy"] > s2["mean_progress_random"]
    assert s2["mean_final_dist_policy"] < s2["mean_final_dist_random"]


# --------------------------------------------------------------------------- #
# ④ shield-on cuts near-collision on a wall stub                               #
# --------------------------------------------------------------------------- #
class _WallEnv:
    """Minimal obstacle env: a wall at ``wall_x``; GT depth = distance to it.

    Kinematics are 1-D along +x (yaw fixed): each step advances by the commanded
    forward delta, clamped so the vehicle cannot pass through the wall. ``depth``
    is a full field whose min equals ``wall_x - pos_x`` (clipped ≥ 0.05), so the
    GT near-collision mask and a GT-proxy predictor can both read it.
    """

    def __init__(self, *, wall_x: float = 10.0, step_hz: float = 5.0, size: int = 8) -> None:
        self.config = type("C", (), {"step_hz": float(step_hz)})()
        self._wall_x = float(wall_x)
        self._size = int(size)
        self._pos = np.zeros(3, dtype=np.float64)
        self._goal = np.array([30.0, 0.0, 0.0], dtype=np.float64)

    @property
    def goal(self) -> Optional[np.ndarray]:
        return self._goal

    def _depth_min(self) -> float:
        return max(self._wall_x - float(self._pos[0]), 0.05)

    def _observe(self) -> Observation:
        d = np.full((self._size, self._size), self._depth_min(), dtype=np.float32)
        collided = bool(self._pos[0] >= self._wall_x)
        state = np.array(
            [self._pos[0], self._pos[1], self._pos[2], 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        return Observation(rgb=np.zeros((self._size, self._size, 3), dtype=np.uint8),
                           state=state, depth=d, collided=collided, info={})

    def reset(self, episode: Optional[Dict[str, Any]] = None) -> Observation:
        if episode is not None:
            self._pos = np.asarray(episode["pos"], dtype=np.float64)[0].copy()
            self._goal = np.asarray(episode["pos"], dtype=np.float64)[-1].copy()
        else:
            self._pos = np.zeros(3, dtype=np.float64)
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[Observation, Dict[str, Any]]:
        # Advance forward (body +x with yaw 0), clamp at the wall.
        self._pos[0] = min(self._pos[0] + float(action[0]), self._wall_x)
        return self._observe(), {"cmd": np.asarray(action, dtype=np.float64).tolist()}


class _PessimisticGTDepthPredictor:
    """GT-proxy predictor with a lookahead margin (stops before the near zone).

    Returns ``min(GT depth) - margin`` so the shield (threshold 1.5) triggers
    while the true depth is still well above 1.5 — the way a real depth head +
    safety margin would brake *before* the vehicle enters the near-collision
    band. Reads GT depth only inside the runner's scoring path (never a policy
    input), which is exactly what the V0 supervision boundary allows.
    """

    def __init__(self, margin: float = 1.6) -> None:
        self.margin = float(margin)

    def reset(self) -> None:
        return None

    def predict_min(self, obs: Observation) -> Optional[float]:
        d = np.asarray(obs.depth, dtype=np.float64)
        finite = d[np.isfinite(d) & (d > 0)]
        if finite.size == 0:
            return None
        return float(np.min(finite)) - self.margin


def test_signal4_shield_reduces_near_collision_on_wall():
    env = _WallEnv(wall_x=10.0, step_hz=5.0)
    starts = rollout.make_start_episodes(8, seed=0)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    predictor = _PessimisticGTDepthPredictor(margin=1.6)

    masks = rollout.run_shield_eval(
        env, policy, predictor, starts,
        near_collision_depth_m=1.5, max_steps=60,
    )
    s4 = metrics.check_shield_effectiveness(
        interventions_on=masks["interventions_on"],
        collided_on=masks["collided_on"],
        near_coll_on=masks["near_coll_on"],
        near_coll_off=masks["near_coll_off"],
    )
    assert s4["ok"], s4
    # Shield-off plows into the wall band; shield-on brakes before it.
    assert s4["near_coll_rate_off"] > 0.0
    assert s4["near_coll_rate_on"] < s4["near_coll_rate_off"]


def test_signal4_degenerate_on_obstacle_free_mock():
    """Mock has no obstacles (depth ramp min ≈ 1.0 always < 1.5) → ④ must NOT
    spuriously pass: on/off near-rates are ~equal so the ratio fails."""
    env = MockAirSimDroneEnv(MockEnvConfig(step_hz=5.0))
    starts = rollout.make_start_episodes(4, seed=0)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    predictor = _PessimisticGTDepthPredictor(margin=0.0)  # honest, but no obstacle

    masks = rollout.run_shield_eval(
        env, policy, predictor, starts, near_collision_depth_m=1.5, max_steps=20,
    )
    s4 = metrics.check_shield_effectiveness(
        interventions_on=masks["interventions_on"],
        collided_on=masks["collided_on"],
        near_coll_on=masks["near_coll_on"],
        near_coll_off=masks["near_coll_off"],
    )
    assert not s4["ok"], s4  # honest degenerate: no real obstacle to avoid


# --------------------------------------------------------------------------- #
# gate assembly: depth pillar cannot be bypassed                               #
# --------------------------------------------------------------------------- #
def _ok() -> Dict[str, Any]:
    return {"ok": True}


def test_assemble_verdict_requires_depth_pillar():
    green = assemble_verdict(s1abc=_ok(), s1d=_ok(), s2=_ok(), s3=_ok(), s4=_ok())
    assert green["ok"], green
    # ①a–c green but ①d (depth AbsRel) failing → whole gate fails.
    no_d = assemble_verdict(
        s1abc=_ok(), s1d={"ok": False, "reason": "no depth head"},
        s2=_ok(), s3=_ok(), s4=_ok(),
    )
    assert not no_d["ok"], no_d
    assert no_d["passed"]["1"] is False
