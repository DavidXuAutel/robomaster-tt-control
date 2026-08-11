"""Rollout runners for V0 signals ② (progress-vs-random) and ④ (shield on/off).

Env-agnostic: takes an env + a policy (+ an optional depth predictor for ④).
Torch-free — the depth predictor is duck-typed (``predict_min`` / optional
``reset``), so H100 passes the real ``DepthMinPredictor`` and Mac tests pass a
GT-proxy. Feeds the pure scorers in :mod:`v0_metrics`.

Pairing is the whole point of these two signals (frozen §4.1 ②/④): the same set
of start episodes drives policy-vs-random and shield-on-vs-shield-off, so the
margin/ratio is a controlled comparison rather than two independent samples.

The mock env has no obstacle geometry (depth is a fixed ramp), so ④ is only
*meaningfully* passable on a renderer with obstacles (airsim); on mock it still
runs and reports honest degenerate numbers. ② is meaningful on mock because the
goal-seeking ``HeuristicPolicy`` genuinely closes distance while random does not.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.aerial.rl.buffer import Episode, ReplayBuffer
from experiments.aerial.rl.collector import RolloutCollector
from experiments.aerial.rl.reward import RewardConfig
from experiments.aerial.rl.safety import ThresholdSafetyShield


class RandomActionPolicy:
    """U(-1, 1) body-delta baseline (frozen §4.1 ② ``random``).

    The collector clips to ``body_delta_limits(dt)`` before stepping, so this
    matches the spec's "random = U(-1,1) clip to body_delta_limits" exactly —
    the clip lives in the collector, not here.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(int(seed))

    def reset(self) -> None:
        return None

    def act(self, view: Any) -> np.ndarray:  # view = PolicyObservation (unused)
        return self._rng.uniform(-1.0, 1.0, size=4).astype(np.float64)


def make_start_episodes(
    n: int,
    *,
    seed: int = 0,
    goal_dist_m: float = 30.0,
    cruise_alt_m: float = 20.0,
) -> List[Dict[str, np.ndarray]]:
    """N deterministic *airborne, level* start/goal episode dicts.

    ``pos`` = [start, goal]; ``env.reset(episode)`` reads ``pos[0]`` as the start
    and ``pos[-1]`` as the goal (see ``MockAirSimDroneEnv.reset`` / airsim path).
    Positions are +up world — the airsim ``reset`` negates z into NED. Starts sit
    at a fixed cruising altitude over the origin and goals are ``goal_dist_m``
    away *in the same horizontal plane*: a pure level-navigation task. This
    geometry is deliberate — the earlier ``start=zeros`` + vertical-goal draw
    produced two harness bugs that FAILED ②/④ on their first real airsim run:

      * A ground start (z=0) teleports onto the PlayerStart floor. The airsim
        ``reset`` does NOT take off, so ``has_collided`` latches on ground
        contact and every episode reads as a spawn crash (near_coll_rate_off=0,
        contacts sticky) — not a frontal-obstacle signal.
      * A goal with a vertical component (``direction[2]``) put ~half the goals
        below ground (z<0), unreachable, pinning final-distance high.

    ``cruise_alt_m`` defaults to ≈ the median start altitude (19.5 m) of the
    proven ``dataset_v1_rgb`` collection — an altitude the drone is known to hold
    and fly clear at in ``env_airsim_16``, and low enough that 30 m of level
    flight meets buildings (so ④ has real obstacles to avoid). Headings are drawn
    once from a seeded RNG so policy/random and shield on/off see identical
    starts.
    """
    rng = np.random.default_rng(int(seed))
    episodes: List[Dict[str, np.ndarray]] = []
    start = np.array([0.0, 0.0, float(cruise_alt_m)], dtype=np.float64)
    for _ in range(int(n)):
        heading = rng.normal(size=2)  # horizontal only — level navigation
        norm = float(np.linalg.norm(heading))
        heading = heading / norm if norm > 1e-9 else np.array([1.0, 0.0])
        goal = start + np.array([heading[0], heading[1], 0.0]) * float(goal_dist_m)
        episodes.append(
            {"pos": np.stack([start, goal]), "yaw": np.array([0.0, 0.0])}
        )
    return episodes


def _forward_min_depth(depth: np.ndarray, *, center_frac: float) -> float:
    """Min finite+positive depth over the central ``center_frac`` box (forward).

    The front camera faces body-forward (the episode yaw), so the image centre
    is the flight direction. Restricting the obstacle test to a centre crop is
    what makes "is there something *ahead*" distinct from "is there ground far
    below / a wall off to the side" — a full-field min at cruise altitude is
    almost always the ground, which never triggers a 1.5 m near-collision.
    """
    h, w = depth.shape[-2], depth.shape[-1]
    cf = float(np.clip(center_frac, 0.05, 1.0))
    dh, dw = int(h * cf), int(w * cf)
    r0, c0 = (h - dh) // 2, (w - dw) // 2
    crop = np.asarray(depth[r0 : r0 + dh, c0 : c0 + dw], dtype=np.float64)
    finite = crop[np.isfinite(crop) & (crop > 0)]
    return float(np.min(finite)) if finite.size else float("inf")


def _full_min_depth(depth: np.ndarray) -> float:
    d = np.asarray(depth, dtype=np.float64)
    finite = d[np.isfinite(d) & (d > 0)]
    return float(np.min(finite)) if finite.size else float("inf")


def make_obstacle_facing_episodes(
    env: Any,
    n: int,
    candidate_positions: np.ndarray,
    *,
    seed: int = 0,
    goal_dist_m: float = 30.0,
    yaw_candidates_deg: Optional[List[float]] = None,
    obstacle_min_m: float = 5.0,
    obstacle_max_m: float = 25.0,
    start_clearance_m: float = 3.0,
    center_frac: float = 0.5,
    max_scans: Optional[int] = None,
) -> tuple[List[Dict[str, np.ndarray]], Dict[str, Any]]:
    """Build N obstacle-facing start/goal episodes by *live scanning* the renderer.

    ④ needs the policy to actually approach an obstacle when the shield is off,
    so the start/goal geometry must point at real geometry. ``env_airsim_16`` is
    mostly OPEN at cruise altitude (a level goal over the origin meets nothing →
    ``near_coll_rate_off == 0`` → ④ vacuously fails), so we cannot synthesise
    starts blindly like :func:`make_start_episodes`. Instead:

      1. Take ``candidate_positions`` (+up world) — trajectory positions from a
         real collection (``--rollout-dataset``); the drone flew *there*, so the
         renderer's own geometry is nearby by construction. Using the exact
         collection coordinates side-steps every world-frame ambiguity: we never
         reason about where obstacles "should" be, we teleport and *ask*.
      2. For each (position, yaw) candidate, teleport (``env.reset``) and read GT
         depth. Keep it only when the *forward* depth (central crop) sits in
         ``[obstacle_min_m, obstacle_max_m]`` — a real obstacle in the flight
         path, mid-range so the goal-seeker still makes ② progress before it
         arrives — AND the full-field min clears ``start_clearance_m`` (not
         spawned already inside the 1.5 m near-zone) AND the spawn did not latch
         a collision. The goal is ``goal_dist_m`` straight along that heading, so
         a goal-seeking policy flies *into* the obstacle (shield-off → near-
         collision; shield-on → brake) and a random policy does not.

    This is a ②/④ *harness* geometry fix (selecting valid episodes for the
    shield comparison), NOT a §4.1 change: env / thresholds / model / flags are
    untouched, and the obstacle-selection bounds here are episode filters, not
    gate thresholds. Returns ``(episodes, diag)``; ``diag`` reports the scan so a
    short run surfaces "found K/N obstacle-facing starts" up front rather than
    after a 40-min blind rollout.
    """
    rng = np.random.default_rng(int(seed))
    cand = np.asarray(candidate_positions, dtype=np.float64).reshape(-1, 3)
    if cand.shape[0] == 0:
        raise ValueError("make_obstacle_facing_episodes: no candidate_positions")
    yaws = (
        [float(y) for y in yaw_candidates_deg]
        if yaw_candidates_deg is not None
        else [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    )
    # (position, yaw) scan order: shuffle positions, and each position tries its
    # yaws in a shuffled order, so the accepted set spreads across the map rather
    # than exhausting one spot. Deterministic under ``seed``.
    order = rng.permutation(cand.shape[0])
    pairs: List[tuple[int, float]] = []
    for pi in order:
        ys = list(yaws)
        rng.shuffle(ys)
        for y in ys:
            pairs.append((int(pi), float(y)))
    budget = int(max_scans) if max_scans is not None else len(pairs)

    episodes: List[Dict[str, np.ndarray]] = []
    accepted_fwd: List[float] = []
    n_scanned = 0
    rej = {"no_depth": 0, "spawn_collision": 0, "too_close": 0,
           "open_ahead": 0, "obstacle_ok": 0, "reset_error": 0}
    for pi, yaw_deg in pairs:
        if len(episodes) >= int(n) or n_scanned >= budget:
            break
        n_scanned += 1
        start = cand[pi].copy()
        yaw = math.radians(yaw_deg)
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        goal = start + heading * float(goal_dist_m)
        epi = {"pos": np.stack([start, goal]),
               "yaw": np.array([yaw, yaw], dtype=np.float64)}
        # A reset can raise on airsim (health-check on a bad pose, transient RPC).
        # Skip that candidate rather than aborting the whole scan — one unlucky
        # teleport must not lose the other 15 obstacle-facing starts.
        try:
            obs = env.reset(epi)
        except Exception:  # noqa: BLE001
            rej["reset_error"] += 1
            continue
        if getattr(obs, "collided", False):
            rej["spawn_collision"] += 1
            continue
        depth = getattr(obs, "depth", None)
        if depth is None:
            rej["no_depth"] += 1
            continue
        if _full_min_depth(depth) < float(start_clearance_m):
            rej["too_close"] += 1
            continue
        fwd = _forward_min_depth(depth, center_frac=center_frac)
        if not (float(obstacle_min_m) <= fwd <= float(obstacle_max_m)):
            rej["open_ahead"] += 1
            continue
        rej["obstacle_ok"] += 1
        accepted_fwd.append(fwd)
        episodes.append(epi)

    diag = {
        "requested": int(n),
        "accepted": len(episodes),
        "scanned": n_scanned,
        "candidates": int(cand.shape[0]),
        "rejections": rej,
        "accepted_fwd_depth_m": {
            "min": float(np.min(accepted_fwd)) if accepted_fwd else None,
            "max": float(np.max(accepted_fwd)) if accepted_fwd else None,
            "mean": float(np.mean(accepted_fwd)) if accepted_fwd else None,
        },
        "params": {
            "goal_dist_m": float(goal_dist_m),
            "obstacle_min_m": float(obstacle_min_m),
            "obstacle_max_m": float(obstacle_max_m),
            "start_clearance_m": float(start_clearance_m),
            "center_frac": float(center_frac),
        },
    }
    return episodes, diag


def _goal_of(env: Any) -> Optional[np.ndarray]:
    g = getattr(env, "goal", None)
    return None if g is None else np.asarray(g, dtype=np.float64).reshape(3)


def _run_one(
    env: Any,
    policy: Any,
    episode: Dict[str, np.ndarray],
    *,
    max_steps: int,
    reward_cfg: Optional[RewardConfig],
    shield: Any = None,
    depth_predictor: Any = None,
) -> Episode:
    """Collect a single episode with a throwaway in-memory buffer."""
    buf = ReplayBuffer(capacity_episodes=2, seed=0)
    col = RolloutCollector(
        env,
        policy,
        buf,
        reward_cfg=reward_cfg,
        safety=shield,
        max_steps=int(max_steps),
        target_hz=0.0,
        depth_predictor=depth_predictor,
        skip_reset_collision=False,
    )
    ep, _stats = col.collect_episode(episode)
    return ep


def run_progress_eval(
    env: Any,
    policy: Any,
    random_policy: Any,
    start_episodes: List[Dict[str, np.ndarray]],
    *,
    max_steps: int = 200,
    reward_cfg: Optional[RewardConfig] = None,
) -> Dict[str, List[float]]:
    """Signal ② inputs: paired progress_sum + final_dist for policy vs random.

    ``progress_sum`` telescopes to ``‖g − p_0‖ − ‖g − p_final‖`` (the sum of the
    per-step ``NavigationReward`` progress terms), computed here from positions
    so it is independent of reward-shaping weights.
    """
    out = {
        "policy_progress_sums": [],
        "random_progress_sums": [],
        "policy_final_dists": [],
        "random_final_dists": [],
    }
    for epi in start_episodes:
        for tag, pol in (("policy", policy), ("random", random_policy)):
            if hasattr(pol, "reset"):
                pol.reset()
            ep = _run_one(env, pol, epi, max_steps=max_steps, reward_cfg=reward_cfg)
            goal = _goal_of(env)
            if goal is None or not ep:
                # No goal or empty episode → non-informative; record neutral 0s.
                out[f"{tag}_progress_sums"].append(0.0)
                out[f"{tag}_final_dists"].append(float("nan"))
                continue
            start_pos = np.asarray(ep[0].obs.position, dtype=np.float64)
            final_pos = np.asarray(ep[-1].next_obs.position, dtype=np.float64)
            init_d = float(np.linalg.norm(goal - start_pos))
            final_d = float(np.linalg.norm(goal - final_pos))
            out[f"{tag}_progress_sums"].append(init_d - final_d)
            out[f"{tag}_final_dists"].append(final_d)
    return out


def _episode_masks(
    ep: Episode,
    *,
    near_collision_depth_m: float,
) -> Dict[str, List[bool]]:
    """Per-step boolean sequences for ④ from a collected episode.

    ``intervention`` comes from the stored transition info; ``collided`` and the
    GT-depth near-collision flag come from the (supervision-only) full obs — GT
    depth never entered the policy graph, it is read here only to score.

    ``collided`` is a *post-step* event: the vehicle contacts an obstacle as a
    result of the action, so it only ever appears on ``next_obs`` (and, because
    the collector breaks the episode on ``done``, only on the terminal step).
    Reading the pre-step ``tr.obs`` here would make ``collided`` all-False for
    every episode and silently vacuous the ④ intervention-before-contact check.
    """
    interv, coll, near = [], [], []
    depth_steps = 0
    for tr in ep:
        interv.append(bool(tr.info.get("intervention", False)))
        post = tr.next_obs if tr.next_obs is not None else tr.obs
        coll.append(bool(getattr(post, "collided", False)))
        obs = tr.obs
        d = getattr(obs, "depth", None)
        if d is None:
            near.append(False)
            continue
        d = np.asarray(d, dtype=np.float64)
        finite = d[np.isfinite(d) & (d > 0)]
        if finite.size:
            depth_steps += 1
        near.append(bool(finite.size and float(np.min(finite)) < float(near_collision_depth_m)))
    # ``depth_steps`` = steps that actually carried a usable GT depth field. The
    # near-collision mask is GT-depth-driven; a run with grab_depth=false yields
    # depth_steps==0 → an all-False near mask that must NOT be read as "no
    # obstacle nearby" (see run_shield_eval's fail-closed guard).
    return {"intervention": interv, "collided": coll, "near": near,
            "depth_steps": depth_steps}


def run_shield_eval(
    env: Any,
    policy: Any,
    depth_predictor_on: Any,
    start_episodes: List[Dict[str, np.ndarray]],
    *,
    near_collision_depth_m: float = 1.5,
    max_steps: int = 200,
    reward_cfg: Optional[RewardConfig] = None,
) -> Dict[str, List[List[bool]]]:
    """Signal ④ inputs: paired shield-on vs shield-off per-step boolean masks.

    ``depth_predictor_on`` fills ``obs.info['depth_min_pred']`` so the
    ``ThresholdSafetyShield`` (``min_depth_m = near_collision_depth_m``) can
    trigger; the shield-off side installs neither shield nor predictor. Same
    ``start_episodes`` both sides. GT depth drives the near-collision mask on
    both sides identically, so the ratio is a controlled comparison.
    """
    shield = ThresholdSafetyShield(min_depth_m=float(near_collision_depth_m))
    interventions_on: List[List[bool]] = []
    collided_on: List[List[bool]] = []
    near_coll_on: List[List[bool]] = []
    near_coll_off: List[List[bool]] = []
    depth_steps = 0

    for epi in start_episodes:
        if hasattr(policy, "reset"):
            policy.reset()
        ep_on = _run_one(
            env, policy, epi, max_steps=max_steps, reward_cfg=reward_cfg,
            shield=shield, depth_predictor=depth_predictor_on,
        )
        m_on = _episode_masks(ep_on, near_collision_depth_m=near_collision_depth_m)
        interventions_on.append(m_on["intervention"])
        collided_on.append(m_on["collided"])
        near_coll_on.append(m_on["near"])
        depth_steps += int(m_on["depth_steps"])

        if hasattr(policy, "reset"):
            policy.reset()
        ep_off = _run_one(
            env, policy, epi, max_steps=max_steps, reward_cfg=reward_cfg,
            shield=None, depth_predictor=None,
        )
        m_off = _episode_masks(ep_off, near_collision_depth_m=near_collision_depth_m)
        near_coll_off.append(m_off["near"])
        depth_steps += int(m_off["depth_steps"])

    return {
        "interventions_on": interventions_on,
        "collided_on": collided_on,
        "near_coll_on": near_coll_on,
        "near_coll_off": near_coll_off,
        # 0 → no episode carried GT depth (grab_depth=false). The near-collision
        # masks are then vacuously all-False; the caller must fail ④ closed
        # rather than mistaking "no depth" for "no obstacle ever near".
        "depth_steps": depth_steps,
    }
