"""Mac (torch-free) tests for the V0 ②/④ rollout runners + gate assembly.

② is exercised on the mock env (goal-seeking ``HeuristicPolicy`` genuinely
closes distance, random does not). The mock has no obstacles, so ④ is exercised
on a purpose-built wall stub with a pessimistic GT-proxy depth predictor — the
point of the Mac test is the *wiring* (predictor → obs.info → shield override →
fewer near-collision steps), not a renderer-grade physics check. The real ④
pass happens on airsim with a trained depth head.
"""
from __future__ import annotations

import json
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


def test_episode_geom_diag_flags_backward_retreat():
    # 晚¹³ read-only telemetry: a blind backward retreat (body −x with a forward-only
    # sensor) into an unsensed rear wall shows up as along_heading_on < 0 with the
    # collision on the terminal step — the signal that separates it from a too-close
    # spawn (which would show start_*_min < standoff). Neither field entered the
    # policy graph; this is pure proprio+GT post-processing.
    from experiments.aerial.rl.buffer import Transition

    def _o(x, collided=False):
        state = np.array([x, 0, 0, 0, 0, 0, 0.0], np.float32)  # yaw 0 → heading +x
        return Observation(
            rgb=np.zeros((4, 4, 3), np.uint8), state=state, collided=collided,
            depth=np.full((4, 4), 5.0, np.float32),
            imu={"ang_vel": [0, 0, 0], "lin_acc": [0, 0, 9.807]}, info={},
        )

    ep_on = [
        Transition(obs=_o(0.0), action=np.array([-3, 0, 0, 0]), reward=0.0,
                   done=False, next_obs=_o(-1.0), info={"intervention": True}),
        Transition(obs=_o(-1.0), action=np.array([-3, 0, 0, 0]), reward=0.0,
                   done=True, next_obs=_o(-2.0, collided=True),
                   info={"intervention": True}),
    ]
    epi = {"pos": np.stack([np.zeros(3), np.array([5.0, 0, 0])]),
           "yaw": np.array([0.0, 0.0])}
    d = rollout._episode_geom_diag(ep_on, None, epi)
    assert d["along_heading_on"] < 0.0, d          # net travel was BACKWARD
    assert d["coll_first_on"] == 1, d              # collided on the terminal step
    assert d["interv_first"] == 0, d
    assert d["start_full_min"] == 5.0 and d["start_fwd_min"] == 5.0, d
    assert d["len_off"] == -1 and d["coll_first_off"] == -1, d


class _SpawnCollisionEnv:
    """Reset spawns already in collision for the first ``collide_resets`` resets,
    then clear — models the live renderer's non-deterministic reset (the same
    start can spawn embedded on one attempt and clear on the next).
    """

    def __init__(self, *, collide_resets: int, step_hz: float = 5.0, size: int = 8) -> None:
        self.config = type("C", (), {"step_hz": float(step_hz)})()
        self._collide_resets = int(collide_resets)
        self._resets = 0
        self._pos = np.zeros(3, dtype=np.float64)
        self._goal = np.array([30.0, 0.0, 0.0], dtype=np.float64)
        self._spawn_collided = False

    @property
    def goal(self) -> Optional[np.ndarray]:
        return self._goal

    def _observe(self, collided: bool) -> Observation:
        state = np.array([self._pos[0], self._pos[1], self._pos[2], 0, 0, 0, 0.0], np.float32)
        return Observation(
            rgb=np.zeros((8, 8, 3), np.uint8), state=state,
            depth=np.full((8, 8), 5.0, np.float32), collided=collided, info={},
        )

    def reset(self, episode: Optional[Dict[str, Any]] = None) -> Observation:
        self._pos = np.zeros(3, dtype=np.float64)
        self._spawn_collided = self._resets < self._collide_resets
        self._resets += 1
        return self._observe(self._spawn_collided)

    def step(self, action: np.ndarray) -> tuple[Observation, Dict[str, Any]]:
        self._pos[0] += float(action[0])
        return self._observe(False), {}


def test_run_one_resilient_drops_persistent_spawn_collision():
    # A start that spawns embedded on EVERY reset (no clear resample) is an invalid
    # ④ trial: no pre-contact window, so it must be DROPPED (None), not counted as a
    # shielded collision. Auditable via drop_stats["spawn_collision"].
    env = _SpawnCollisionEnv(collide_resets=99)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    stats: Dict[str, int] = {}
    ep = rollout._run_one_resilient(
        env, policy, rollout.make_start_episodes(1, seed=0)[0],
        max_steps=20, reward_cfg=None, retry_sleep_s=0.0, drop_stats=stats,
    )
    assert ep is None, ep
    assert stats.get("spawn_collision") == 1, stats
    assert stats.get("health", 0) == 0, stats


def test_run_one_resilient_resamples_spawn_collision_then_succeeds():
    # The renderer reset is non-deterministic: the first reset spawns embedded, the
    # retry lands clear. The resample must RECOVER the start (preserving N + giving a
    # genuine pre-contact window), not drop it.
    env = _SpawnCollisionEnv(collide_resets=1)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    stats: Dict[str, int] = {}
    ep = rollout._run_one_resilient(
        env, policy, rollout.make_start_episodes(1, seed=0)[0],
        max_steps=20, reward_cfg=None, retry_sleep_s=0.0, drop_stats=stats,
    )
    assert ep is not None and len(ep) > 0, ep
    assert not bool(getattr(ep[0].obs, "collided", False)), ep[0].obs  # clear spawn
    assert stats == {}, stats  # recovered → nothing dropped


def test_shield_eval_reports_spawn_collision_drops():
    # run_shield_eval surfaces the drop accounting so a FAIL→(no-contact) shift is
    # never a silent truncation: every embedded start is counted, not hidden.
    env = _SpawnCollisionEnv(collide_resets=99)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    predictor = _PessimisticGTDepthPredictor(margin=1.6)
    masks = rollout.run_shield_eval(
        env, policy, predictor, rollout.make_start_episodes(3, seed=0),
        near_collision_depth_m=1.5, max_steps=20,
    )
    # Every start drops (both arms embedded) → no scored episodes, drops surfaced.
    assert masks["interventions_on"] == [], masks
    assert masks["spawn_collision_drops"] >= 3, masks
    assert masks["health_drops"] == 0, masks


class _JitterEnv:
    """First ``jitter_resets`` resets produce a physically-impossible single-step
    position jump on the first step (proprio teleport-jitter — the live reset left
    the z coordinate unsettled for a frame); later resets fly normally. Mirrors
    _SpawnCollisionEnv so the resample-then-recover / persistent-drop paths are
    exercised identically for the jitter guard.
    """

    def __init__(self, *, jitter_resets: int, jump_m: float = 20.0,
                 step_hz: float = 5.0, size: int = 8) -> None:
        self.config = type("C", (), {"step_hz": float(step_hz)})()
        self._jitter_resets = int(jitter_resets)
        self._jump_m = float(jump_m)
        self._resets = 0
        self._pos = np.zeros(3, dtype=np.float64)
        self._goal = np.array([30.0, 0.0, 0.0], dtype=np.float64)
        self._jitter = False
        self._stepped = False

    @property
    def goal(self) -> Optional[np.ndarray]:
        return self._goal

    def _observe(self, collided: bool) -> Observation:
        state = np.array([self._pos[0], self._pos[1], self._pos[2], 0, 0, 0, 0.0], np.float32)
        return Observation(
            rgb=np.zeros((8, 8, 3), np.uint8), state=state,
            depth=np.full((8, 8), 5.0, np.float32), collided=collided, info={},
        )

    def reset(self, episode: Optional[Dict[str, Any]] = None) -> Observation:
        self._pos = np.zeros(3, dtype=np.float64)
        self._jitter = self._resets < self._jitter_resets
        self._resets += 1
        self._stepped = False
        return self._observe(False)

    def step(self, action: np.ndarray) -> tuple[Observation, Dict[str, Any]]:
        if self._jitter and not self._stepped:
            self._stepped = True
            self._pos[2] += self._jump_m  # impossible ~20 m z jump in one step
            return self._observe(True), {}
        self._stepped = True
        self._pos[0] += float(action[0])
        return self._observe(False), {}


def test_run_one_resilient_drops_persistent_proprio_jitter():
    # A start that teleport-jitters on EVERY reset is an invalid trial (not real
    # flight, no legitimate pre-contact window) → DROP (None), auditable via
    # drop_stats["proprio_jitter"]. Same class as persistent spawn-collision (晚¹⁷).
    env = _JitterEnv(jitter_resets=99)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    stats: Dict[str, int] = {}
    ep = rollout._run_one_resilient(
        env, policy, rollout.make_start_episodes(1, seed=0)[0],
        max_steps=20, reward_cfg=None, retry_sleep_s=0.0, drop_stats=stats,
    )
    assert ep is None, ep
    assert stats.get("proprio_jitter") == 1, stats
    assert stats.get("spawn_collision", 0) == 0, stats


def test_run_one_resilient_resamples_proprio_jitter_then_succeeds():
    # The reset is non-deterministic: the first spawn jitters, the retry lands a
    # clean roll. The resample must RECOVER the start (preserve N), not drop it.
    env = _JitterEnv(jitter_resets=1)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    stats: Dict[str, int] = {}
    ep = rollout._run_one_resilient(
        env, policy, rollout.make_start_episodes(1, seed=0)[0],
        max_steps=20, reward_cfg=None, retry_sleep_s=0.0, drop_stats=stats,
    )
    assert ep is not None and len(ep) > 0, ep
    assert rollout._max_step_travel(ep) <= rollout._MAX_STEP_TRAVEL_M, ep
    assert stats == {}, stats  # recovered → nothing dropped


def test_jitter_guard_no_false_positive_on_genuine_roll():
    # A real forward approach never jumps > cap in one 5 Hz step, so the guard must
    # NOT fire on it — otherwise valid ④ data would be dropped. Positive control.
    env = _WallEnv(wall_x=10.0, step_hz=5.0)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    stats: Dict[str, int] = {}
    ep = rollout._run_one_resilient(
        env, policy, rollout.make_start_episodes(1, seed=0)[0],
        max_steps=40, reward_cfg=None, retry_sleep_s=0.0, drop_stats=stats,
    )
    assert ep is not None and len(ep) > 0, ep
    assert rollout._max_step_travel(ep) <= rollout._MAX_STEP_TRAVEL_M, ep
    assert stats == {}, stats


def test_shield_eval_reports_proprio_jitter_drops():
    # run_shield_eval surfaces proprio_jitter_drops so a jitter-driven drop is
    # auditable, never a silent truncation.
    env = _JitterEnv(jitter_resets=99)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    predictor = _PessimisticGTDepthPredictor(margin=1.6)
    masks = rollout.run_shield_eval(
        env, policy, predictor, rollout.make_start_episodes(3, seed=0),
        near_collision_depth_m=1.5, max_steps=20,
    )
    assert masks["interventions_on"] == [], masks
    assert masks["proprio_jitter_drops"] >= 3, masks
    assert masks["spawn_collision_drops"] == 0, masks


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


class _ScanEnv:
    """Teleport-only env for the obstacle-facing scan: forward depth depends on
    the (start position, yaw) an episode resets to.

    A single obstacle sits ``obs_dist`` metres ahead of exactly one candidate
    position (index ``obstacle_pos_idx``) when facing yaw 0; every other pose is
    open (forward depth = ``far``). Full-field min is ``floor_m`` everywhere
    (a benign 'ground' well beyond the near-zone) so start_clearance never trips.
    """

    def __init__(self, *, obstacle_at, obs_dist=8.0, far=60.0, floor_m=30.0, size=8,
                 obstacle_yaw=0.0):
        self._obstacle_at = np.asarray(obstacle_at, dtype=np.float64)
        self._obs_dist = float(obs_dist)
        self._far = float(far)
        self._floor = float(floor_m)
        self._size = int(size)
        self._obstacle_yaw = float(obstacle_yaw)
        self._pos = np.zeros(3)
        self._yaw = 0.0
        self._goal = None

    @property
    def goal(self):
        return self._goal

    def reset(self, episode=None):
        pos = np.asarray(episode["pos"], dtype=np.float64)
        self._pos = pos[0].copy()
        self._goal = pos[-1].copy()
        self._yaw = float(np.asarray(episode["yaw"]).reshape(-1)[0])
        facing = abs(self._yaw - self._obstacle_yaw) < 1e-6  # obstacle only along this heading
        at_obstacle = bool(np.allclose(self._pos, self._obstacle_at, atol=1e-6))
        fwd = self._obs_dist if (facing and at_obstacle) else self._far
        # Central pixel carries the forward reading; borders carry the floor.
        d = np.full((self._size, self._size), self._floor, dtype=np.float32)
        c = self._size // 2
        d[c, c] = fwd
        return Observation(rgb=np.zeros((self._size, self._size, 3), np.uint8),
                           state=np.array([self._pos[0], self._pos[1], self._pos[2],
                                           0, 0, 0, self._yaw], dtype=np.float32),
                           depth=d, collided=False, info={})


def test_make_obstacle_facing_episodes_keeps_only_forward_obstacle():
    obstacle_pos = np.array([10.0, 0.0, 20.0])
    cand = np.array([[10.0, 0.0, 20.0], [500.0, 0.0, 20.0], [-300.0, 40.0, 15.0]])
    env = _ScanEnv(obstacle_at=obstacle_pos, obs_dist=8.0)
    eps, diag = rollout.make_obstacle_facing_episodes(
        env, 4, cand, seed=0, goal_dist_m=30.0,
        obstacle_min_m=5.0, obstacle_max_m=25.0,
    )
    # Only the (obstacle position, yaw 0) pose has a mid-range forward obstacle.
    assert diag["accepted"] == 1, diag
    assert len(eps) == 1
    start = eps[0]["pos"][0]
    goal = eps[0]["pos"][-1]
    assert np.allclose(start, obstacle_pos), start
    # goal is 30 m straight ahead along the accepted heading (yaw 0 → +x).
    assert np.allclose(goal, obstacle_pos + np.array([30.0, 0.0, 0.0])), goal
    assert 5.0 <= diag["accepted_fwd_depth_m"]["min"] <= 25.0, diag


def test_make_obstacle_facing_episodes_reports_zero_in_open_scene():
    """No obstacle anywhere → accepted 0 (caller fails ④ closed, no false pass)."""
    cand = np.array([[0.0, 0.0, 20.0], [50.0, 50.0, 20.0]])
    env = _ScanEnv(obstacle_at=np.array([9999.0, 0.0, 0.0]))  # never matched
    eps, diag = rollout.make_obstacle_facing_episodes(env, 4, cand, seed=0)
    assert eps == []
    assert diag["accepted"] == 0
    assert diag["rejections"]["open_ahead"] == diag["scanned"]


def test_make_obstacle_facing_episodes_uses_recorded_yaw_off_grid():
    """A head-on obstacle along an OFF-GRID heading (0.3 rad ≈ 17°, not in the
    8-yaw 45° grid) is missed by the grid alone but found when the recorded
    approach yaw is supplied — the 2026-08-11 probe_no_hit fix.
    """
    obstacle_pos = np.array([10.0, 0.0, 20.0])
    off_grid_yaw = 0.3  # radians; nearest grid yaw (0) is ~17° off → grid misses
    cand = np.array([[10.0, 0.0, 20.0]])
    env = _ScanEnv(obstacle_at=obstacle_pos, obs_dist=8.0, obstacle_yaw=off_grid_yaw)

    # Grid only (no recorded yaw): none of 0/45/…/315° align → open_ahead, 0 kept.
    eps_grid, diag_grid = rollout.make_obstacle_facing_episodes(env, 4, cand, seed=0)
    assert diag_grid["accepted"] == 0, diag_grid

    # With the recorded approach yaw, the scan tries it first → obstacle found.
    eps, diag = rollout.make_obstacle_facing_episodes(
        env, 4, cand, seed=0, candidate_yaws=np.array([off_grid_yaw]),
    )
    assert diag["accepted"] == 1, diag
    assert np.isclose(float(eps[0]["yaw"].reshape(-1)[0]), off_grid_yaw), eps[0]


def test_episode_masks_collided_reads_post_step_obs():
    """④ ``collided`` is a post-step event: it lands on ``next_obs`` of the
    terminal transition, never on any ``obs``. Reading pre-step ``tr.obs`` (the
    old bug) made the mask all-False and vacuously passed intervention-before-
    contact. The mask must be True exactly on the terminal step."""
    from experiments.aerial.rl.buffer import Transition

    def _obs(collided: bool) -> Observation:
        return Observation(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            state=np.zeros(7, dtype=np.float32),
            depth=np.full((2, 2), 5.0, dtype=np.float32),
            collided=collided,
            info={},
        )

    # 3 steps; contact happens on the last step (only next_obs carries it).
    ep = [
        Transition(obs=_obs(False), action=np.zeros(3), reward=0.0, done=False,
                   next_obs=_obs(False), info={"intervention": True}),
        Transition(obs=_obs(False), action=np.zeros(3), reward=0.0, done=False,
                   next_obs=_obs(False), info={"intervention": True}),
        Transition(obs=_obs(False), action=np.zeros(3), reward=0.0, done=True,
                   next_obs=_obs(True), info={"intervention": False}),
    ]
    masks = rollout._episode_masks(ep, near_collision_depth_m=1.5)
    assert masks["collided"] == [False, False, True], masks["collided"]
    # With a real contact on step 2 and interventions on steps 0-1, the ④
    # intervention-before-contact sub-metric is now exercised (not vacuous).
    s4 = metrics.check_shield_effectiveness(
        interventions_on=[masks["intervention"]],
        collided_on=[masks["collided"]],
        near_coll_on=[masks["near"]],
        near_coll_off=[masks["near"]],
    )
    assert s4["n_contact_episodes"] == 1, s4
    assert s4["intervention_before_contact_frac"] == 1.0, s4


class _NoDepthEnv(_WallEnv):
    """Wall kinematics but obs never carries depth (grab_depth=false case)."""

    def _observe(self) -> Observation:
        collided = bool(self._pos[0] >= self._wall_x)
        state = np.array([self._pos[0], self._pos[1], self._pos[2], 0, 0, 0, 0],
                         dtype=np.float32)
        return Observation(rgb=np.zeros((self._size, self._size, 3), dtype=np.uint8),
                           state=state, depth=None, collided=collided, info={})


def test_shield_eval_reports_zero_depth_steps_without_depth():
    """④'s near-collision mask is GT-depth-driven. When the env yields no depth
    (grab_depth=false), run_shield_eval must report depth_steps==0 so the gate
    can fail ④ closed instead of reading the all-False near mask as 'safe'."""
    env = _NoDepthEnv(wall_x=10.0, step_hz=5.0)
    starts = rollout.make_start_episodes(4, seed=0)
    policy = HeuristicPolicy(goal_getter=lambda: env.goal)
    predictor = _PessimisticGTDepthPredictor(margin=1.6)
    masks = rollout.run_shield_eval(
        env, policy, predictor, starts, near_collision_depth_m=1.5, max_steps=20,
    )
    assert masks["depth_steps"] == 0, masks["depth_steps"]
    # every near mask entry is False (no depth to threshold)
    assert not any(any(e) for e in masks["near_coll_on"])
    assert not any(any(e) for e in masks["near_coll_off"])


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


# --------------------------------------------------------------------------- #
# split evaluation (plan B): --signals subsets → --emit partials → --merge     #
# --------------------------------------------------------------------------- #
def test_merge_partials_reproduces_single_host_verdict(tmp_path):
    from experiments.aerial.rl import _v0_gate as gate

    sig1 = {"ok": True, "abc": _ok(), "d": _ok()}
    part_13 = {"partial": True, "signals": {"1": sig1, "3": _ok()}}
    part_24 = {"partial": True, "signals": {"2": _ok(), "4": _ok()}}
    p13 = tmp_path / "part_13.json"
    p24 = tmp_path / "part_24.json"
    p13.write_text(json.dumps(part_13))
    p24.write_text(json.dumps(part_24))

    merged = gate._merge_partials([p13, p24])
    assert set(merged) == {"1", "2", "3", "4"}
    verdict = metrics.aggregate_v0_verdict(merged)
    assert verdict["ok"], verdict


def test_merge_missing_signal_is_not_a_pass(tmp_path):
    from experiments.aerial.rl import _v0_gate as gate

    p13 = tmp_path / "part_13.json"
    p13.write_text(json.dumps({"signals": {"1": {"ok": True}, "3": {"ok": True}}}))
    merged = gate._merge_partials([p13])
    verdict = metrics.aggregate_v0_verdict(merged)
    assert not verdict["ok"], verdict  # ②/④ absent → cannot pass


def test_aggregate_rejects_non_bool_ok():
    """``bool("false")`` is True. aggregate_v0_verdict must not coerce a
    string/int 'ok' into a pass — a partial round-tripped through a hand-edited
    JSON with ok="false" would otherwise flip a failing signal green."""
    all_true = {k: {"ok": True} for k in ("1", "2", "3", "4")}
    assert metrics.aggregate_v0_verdict(all_true)["ok"] is True

    with_string = dict(all_true)
    with_string["3"] = {"ok": "false"}  # truthy string
    v = metrics.aggregate_v0_verdict(with_string)
    assert v["ok"] is False, v
    assert "non-bool" in v["reason"], v


def test_parse_signals_subset_and_default():
    from experiments.aerial.rl import _v0_gate as gate

    assert gate._parse_signals(None) == {"1", "2", "3", "4"}
    assert gate._parse_signals("1,3") == {"1", "3"}
    assert gate._parse_signals(" 2 , 4 ") == {"2", "4"}


def test_merge_cli_exits_nonzero_when_incomplete(tmp_path):
    from experiments.aerial.rl import _v0_gate as gate

    p = tmp_path / "part.json"
    p.write_text(json.dumps({"signals": {"1": {"ok": True}, "3": {"ok": True}}}))
    assert gate.main(["--merge", str(p)]) == 1


def test_rollout_signals_fail_closed_on_non_airsim_backend(tmp_path):
    """②/④ must not be scored authoritatively on a mock/analytic env: the
    goal-seeker trivially beats random and there are no real obstacles, which is
    the false-pass class that invalidated the single-pillar checkpoint. Default
    config ships backend:mock → ②/④ come back FAIL with an 'airsim' reason and
    the CLI exits non-zero, without ever building the env."""
    from experiments.aerial.rl import _v0_gate as gate

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("env:\n  backend: mock\n")
    out = tmp_path / "part2.json"
    rc = gate.main(["--signals", "2", "--config", str(cfg), "--emit", str(out)])
    assert rc == 1
    blob = json.loads(out.read_text())
    s2 = blob["signals"]["2"]
    assert s2["ok"] is False
    assert "airsim" in s2["reason"]
    assert s2["backend"] == "mock"


def test_signal1abc_fails_on_missing_recon_ent_keys(tmp_path):
    """①a–c is recon-monotonicity ∧ no-collapse ∧ loss-drop. A learning log with
    only ``loss`` (recon/entropy keys absent) must FAIL — the old pass-safe
    defaults (recon 0, ent 1) let a WM green-light ①a–c on the loss drop alone,
    the exact single-pillar shortcut that invalidated wm_step_5000."""
    from experiments.aerial.rl import _v0_gate as gate

    log = tmp_path / "loss_only.jsonl"
    log.write_text("\n".join(
        json.dumps({"loss": 1.0 - 0.05 * i}) for i in range(20)
    ))
    res = gate._signal1abc_from_log(log, metrics.DEFAULT_THRESHOLDS)
    assert res["ok"] is False, res
    assert "missing" in res["reason"], res

    # Full log (loss + recon + ent) still evaluates on the real curves.
    full = tmp_path / "full.jsonl"
    full.write_text("\n".join(
        json.dumps({"loss": 1.0 - 0.04 * i, "recon_err": 1.0 - 0.03 * i,
                    "post_entropy_frac": 0.9})
        for i in range(20)
    ))
    ok = gate._signal1abc_from_log(full, metrics.DEFAULT_THRESHOLDS)
    assert "missing" not in ok.get("reason", ""), ok


# --------------------------------------------------------------------------- #
# ③ diagnostic: forward-motion window selection (pure math)                     #
# --------------------------------------------------------------------------- #
def test_forwardness_separates_forward_lateral_climb():
    from experiments.aerial.rl._v0_gate import _forwardness

    # heading = +x (yaw 0) for all three windows, L=4.
    yaw = np.zeros((3, 4), dtype=np.float64)
    dvec = np.array(
        [
            [5.0, 0.0, 0.0],   # forward along heading  → |cos| ≈ 1
            [0.0, 5.0, 0.0],   # pure lateral (strafe)  → |cos| ≈ 0
            [0.0, 0.0, 5.0],   # pure climb             → |cos| ≈ 0
        ],
        dtype=np.float64,
    )
    f = _forwardness(dvec, yaw)
    assert f[0] > 0.95
    assert f[1] < 0.05
    assert f[2] < 0.05


def test_forwardness_backward_is_axis_aligned():
    from experiments.aerial.rl._v0_gate import _forwardness

    # moving backward along heading still changes |Δ median depth| → keep it.
    f = _forwardness(np.array([[-5.0, 0.0, 0.0]]), np.zeros((1, 4)))
    assert f[0] > 0.95


# --------------------------------------------------------------------------- #
# probe-verify accept must match the ④ eval's FULL-FIELD near mask, not a      #
# centre crop (2026-08-11: head-on collisions whose contact geometry sat       #
# outside the 0.3 centre crop were all rejected → accepted 0).                 #
# --------------------------------------------------------------------------- #
class _ApproachEnv:
    """Steps a probe straight toward a frontal wall. The wall reads as a
    high-but-in-range depth in the CENTRE crop (floors at ~1.6 m, never < 1.5)
    while a corner pixel — outside the centre crop, inside the full field —
    drops below 1.5 m as the drone closes. This is the real failure geometry:
    a central-crop probe rejects it, a full-field probe (matching the eval)
    accepts it. Advances ``travel`` by the horizontal action magnitude/step.
    """

    class _Cfg:
        step_hz = 5.0

    def __init__(self, *, wall_at_m=8.0, size=8):
        self.config = self._Cfg()
        self._wall = float(wall_at_m)
        self._size = int(size)
        self._pos = np.zeros(3)
        self._yaw = 0.0
        self._goal = None
        self._travel = 0.0

    @property
    def goal(self):
        return self._goal

    def _obs(self):
        rem = max(self._wall - self._travel, 0.0)
        d = np.full((self._size, self._size), 60.0, dtype=np.float32)
        c = self._size // 2
        d[c, c] = float(max(rem + 1.6, 0.05))   # centre crop floors at ~1.6 m
        d[0, 0] = float(max(rem, 0.05))          # corner (full-field only) → < 1.5
        collided = rem <= 0.3
        state = np.array([self._pos[0], self._pos[1], self._pos[2],
                          0.0, 0.0, 0.0, self._yaw], dtype=np.float32)
        return Observation(rgb=np.zeros((self._size, self._size, 3), np.uint8),
                           state=state, depth=d, collided=collided, info={})

    def reset(self, episode=None):
        pos = np.asarray(episode["pos"], dtype=np.float64)
        self._pos = pos[0].copy()
        self._goal = pos[-1].copy()
        self._yaw = float(np.asarray(episode["yaw"]).reshape(-1)[0])
        self._travel = 0.0
        return self._obs()

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        d = float(np.linalg.norm(a[:2]))
        self._travel += d
        self._pos = self._pos + np.array([np.cos(self._yaw), np.sin(self._yaw), 0.0]) * d
        return self._obs(), {}


def test_probe_accepts_on_full_field_not_centre_crop():
    env = _ApproachEnv(wall_at_m=8.0)
    cand = np.array([[0.0, 0.0, 20.0]])
    policy = HeuristicPolicy(goal_getter=lambda: getattr(env, "goal", None))
    eps, diag = rollout.make_obstacle_facing_episodes(
        env, 1, cand, seed=0, center_frac=0.3,
        probe_policy=policy, probe_near_m=1.5, probe_steps=40,
    )
    # Full-field reaches < 1.5 (corner) → accepted, even though the centre crop
    # never drops below ~1.6 (would have been rejected by the old central test).
    assert diag["accepted"] == 1, diag
    assert diag["probe"]["reached_full_m"]["min"] < 1.5, diag["probe"]
    assert diag["probe"]["reached_fwd_m"]["min"] >= 1.5, diag["probe"]
