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

import logging
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.aerial.rl.buffer import Episode, ReplayBuffer
from experiments.aerial.rl.collector import RolloutCollector
from experiments.aerial.rl.reward import RewardConfig
from experiments.aerial.rl.safety import ThresholdSafetyShield

logger = logging.getLogger(__name__)

# Substrings that mark a *transient renderer / sensor* reset failure (a flaky
# depth/IMU frame on spawn) — as opposed to a code bug. On these we retry the
# reset a few times and then skip the start, rather than crashing the whole gate.
# The obstacle-facing ④ starts are precisely the ones most likely to trip the
# "depth nearly constant" health guard (a wall filling the FOV is near-constant
# depth), so one bad frame must not nuke a 40-min run. The guard stays STRICT:
# a bad frame is never *scored*, only retried/skipped.
_TRANSIENT_RESET_MARKERS = (
    "sanity failed",
    "no depth data on reset",
    "no imu data on reset",
    "renderer/sensors unavailable",
)


def _run_one_resilient(
    env: Any,
    policy: Any,
    episode: Dict[str, np.ndarray],
    *,
    max_steps: int,
    reward_cfg: Optional[RewardConfig],
    shield: Any = None,
    depth_predictor: Any = None,
    retries: int = 2,
    retry_sleep_s: float = 0.5,
) -> Optional[Episode]:
    """``_run_one`` with retry-then-skip on transient reset/health failures.

    Returns the episode, or ``None`` if the reset kept failing its health guard
    (renderer produced a degenerate depth/IMU frame). A ``None`` tells the caller
    to DROP this start from both arms so the pairing (and same-N) is preserved.
    Non-transient errors (real bugs) propagate unchanged — we never mask those.
    """
    last: Optional[BaseException] = None
    for attempt in range(int(retries) + 1):
        try:
            return _run_one(
                env, policy, episode, max_steps=max_steps, reward_cfg=reward_cfg,
                shield=shield, depth_predictor=depth_predictor,
            )
        except RuntimeError as exc:
            msg = str(exc).lower()
            if not any(m in msg for m in _TRANSIENT_RESET_MARKERS):
                raise  # a real error, not a flaky frame — do not swallow it
            last = exc
            logger.warning(
                "rollout reset/health failure (attempt %d/%d), retrying: %s",
                attempt + 1, int(retries) + 1, exc,
            )
            if attempt < int(retries) and retry_sleep_s > 0:
                time.sleep(float(retry_sleep_s))
    logger.warning("skipping start after %d reset/health failures: %s", int(retries) + 1, last)
    return None


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
    candidate_yaws: Optional[np.ndarray] = None,
    obstacle_min_m: float = 5.0,
    obstacle_max_m: float = 25.0,
    start_clearance_m: float = 3.0,
    center_frac: float = 0.5,
    max_scans: Optional[int] = None,
    probe_policy: Any = None,
    probe_steps: int = 24,
    probe_near_m: Optional[float] = None,
    reward_cfg: Optional[RewardConfig] = None,
    preserve_order: bool = False,
    log_every: int = 0,
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
      3. PROBE-VERIFY (when ``probe_policy`` given): the central-crop forward
         depth is only a *proxy* — a wide cone accepts starts where the obstacle
         merely clips the field of view, and the straight-line goal-seeker
         (``HeuristicPolicy`` steers on proprio only, never avoids) threads past
         it with >1.5 m clearance → ``near_coll_rate_off == 0`` and ④ is
         unscorable (observed 2026-08-11: 16/16 accepted on the proxy, all
         near_coll_off==0). So after the proxy passes, roll ``probe_policy`` from
         this start for ``probe_steps`` (shield OFF, no predictor) and keep the
         start ONLY if the GT-depth min actually drops below ``probe_near_m``
         (= near-collision depth). Because the shield-OFF eval arm later runs the
         SAME policy from the SAME start, this makes ``near_coll_rate_off > 0`` by
         construction — ④'s ratio becomes measurable instead of NaN. ② still
         passes on these starts: even blocked, the goal-seeker out-progresses the
         random policy before it stalls at the obstacle.

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
    # Per-candidate RECORDED approach heading (radians→deg), aligned row-for-row
    # with ``candidate_positions``. On a head-on approach corpus the logged yaw
    # points the camera straight at the obstacle, so the straight-line probe rams
    # it dead-on; tried FIRST for each position (see loop below). ``inf``/absent →
    # fall back to the grid. Backward-compatible: callers that pass no yaws (and
    # the unit tests) keep the pure 8-yaw grid behaviour.
    cand_yaw_deg = (
        np.degrees(np.asarray(candidate_yaws, dtype=np.float64).reshape(-1))
        if candidate_yaws is not None else None
    )
    if cand_yaw_deg is not None and cand_yaw_deg.shape[0] != cand.shape[0]:
        raise ValueError(
            "candidate_yaws must align with candidate_positions "
            f"({cand_yaw_deg.shape[0]} vs {cand.shape[0]})"
        )
    # (position, yaw) scan order: each position tries its yaws in a shuffled
    # order (spread across headings). Positions are shuffled by default so the
    # accepted set spreads across the map; but when ``preserve_order`` is set the
    # caller has pre-ranked candidates (e.g. nearest-geometry-first from the
    # collection's own depth) and we walk them in that order so the scan hits the
    # obstacle-rich spots first instead of exhausting ``max_scans`` on open
    # cruise corridors. Deterministic under ``seed`` either way.
    order = np.arange(cand.shape[0]) if preserve_order else rng.permutation(cand.shape[0])
    pairs: List[tuple[int, float]] = []
    for pi in order:
        ys = list(yaws)
        rng.shuffle(ys)
        # Recorded approach heading first: the 8-yaw grid is ≤22.5° off the true
        # heading, so a mid-range frontal obstacle clips the FOV edge → proxy_ok
        # but the straight-line probe threads past (2026-08-11 on the head-on
        # corpus: proxy_ok=19 / probe_no_hit=19 / accepted=0). The logged yaw is
        # the heading the drone actually flew into the obstacle → probe rams it.
        if cand_yaw_deg is not None and math.isfinite(cand_yaw_deg[pi]):
            ys = [float(cand_yaw_deg[pi])] + ys
        for y in ys:
            pairs.append((int(pi), float(y)))
    budget = int(max_scans) if max_scans is not None else len(pairs)

    episodes: List[Dict[str, np.ndarray]] = []
    accepted_fwd: List[float] = []
    probe_hit_depths: List[float] = []
    n_scanned = 0
    do_probe = probe_policy is not None and probe_near_m is not None and int(probe_steps) > 0
    rej = {"no_depth": 0, "spawn_collision": 0, "too_close": 0,
           "open_ahead": 0, "proxy_ok": 0, "probe_no_hit": 0,
           "obstacle_ok": 0, "reset_error": 0}
    for pi, yaw_deg in pairs:
        if len(episodes) >= int(n) or n_scanned >= budget:
            break
        n_scanned += 1
        if int(log_every) and n_scanned % int(log_every) == 0:
            print(f"[v0-gate] scan progress: scanned={n_scanned}/{budget} "
                  f"accepted={len(episodes)}/{int(n)} proxy_ok={rej['proxy_ok']} "
                  f"probe_no_hit={rej['probe_no_hit']} open_ahead={rej['open_ahead']}",
                  flush=True)
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
        rej["proxy_ok"] += 1
        # Confirm the straight-line goal-seeker actually enters the near-zone from
        # this start (proxy accepts wide-cone hits it threads past). Roll it a few
        # steps shield-OFF and require GT-depth min < probe_near_m; else drop.
        if do_probe:
            try:
                probe_ep = _run_one(
                    env, probe_policy, epi, max_steps=int(probe_steps),
                    reward_cfg=reward_cfg, shield=None, depth_predictor=None,
                )
            except Exception:  # noqa: BLE001
                rej["reset_error"] += 1
                continue
            # Require the FORWARD (central-crop) depth to drop below near_m — i.e.
            # the straight-line policy rams a frontal obstacle head-on. A full-field
            # min here accepts a side-graze or the ground sinking into view (the
            # docstring's "full-field min at cruise is almost always the ground");
            # such hits do NOT reproduce on the ④ shield-off arm under cross-net
            # RPC timing jitter (observed 2026-08-11: fwd=13.4 m accepted, probe
            # "hit" 1.08 m off-axis, eval near_coll_off==0). A head-on frontal wall
            # is jitter-robust and makes both the eval near mask (full-field <1.5)
            # and the ④ ratio reproduce by construction.
            probe_min = float("inf")
            for tr in probe_ep:
                d = getattr(tr.obs, "depth", None)
                if d is None:
                    continue
                fwd_min = _forward_min_depth(np.asarray(d, dtype=np.float64), center_frac=center_frac)
                if math.isfinite(fwd_min):
                    probe_min = min(probe_min, float(fwd_min))
            if not (probe_min < float(probe_near_m)):
                rej["probe_no_hit"] += 1
                continue
            probe_hit_depths.append(probe_min)
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
        "probe": {
            "enabled": bool(do_probe),
            "near_m": float(probe_near_m) if probe_near_m is not None else None,
            "steps": int(probe_steps),
            "hits": len(probe_hit_depths),
            "hit_depth_m": {
                "min": float(np.min(probe_hit_depths)) if probe_hit_depths else None,
                "max": float(np.max(probe_hit_depths)) if probe_hit_depths else None,
                "mean": float(np.mean(probe_hit_depths)) if probe_hit_depths else None,
            },
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
        # Run both arms first; a transient reset/health failure on EITHER arm
        # drops the whole start (skip=True) so policy/random stay paired at same N.
        arms: Dict[str, Any] = {}
        skip = False
        for tag, pol in (("policy", policy), ("random", random_policy)):
            if hasattr(pol, "reset"):
                pol.reset()
            ep = _run_one_resilient(env, pol, epi, max_steps=max_steps, reward_cfg=reward_cfg)
            if ep is None:  # reset kept failing its health guard → skip the start
                skip = True
                break
            arms[tag] = ep
        if skip:
            continue
        goal = _goal_of(env)
        for tag in ("policy", "random"):
            ep = arms[tag]
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
    shield_trigger_depth_m: float = 3.0,
    max_steps: int = 200,
    reward_cfg: Optional[RewardConfig] = None,
) -> Dict[str, List[List[bool]]]:
    """Signal ④ inputs: paired shield-on vs shield-off per-step boolean masks.

    ``depth_predictor_on`` fills ``obs.info['depth_min_pred']`` so the
    ``ThresholdSafetyShield`` can trigger; the shield-off side installs neither
    shield nor predictor. Same ``start_episodes`` both sides. GT depth drives the
    near-collision mask on both sides identically, so the ratio is a controlled
    comparison.

    The shield's trigger ``shield_trigger_depth_m`` (reaction standoff, default
    3.0 m) is DECOUPLED from the ④a near-collision metric ``near_collision_depth_m``
    (frozen 1.5 m). Reacting at the 1.5 m band boundary is structurally too late
    under an optimistic predictor — it parks the vehicle in the band (ratio
    inversion) and triggers after contact (before_frac<0.5). A >metric standoff
    lets the (continuous-retreat) shield leave the band and intervene before
    contact. Metric masks still use ``near_collision_depth_m``. See frozen-spec
    ④a re-freeze 2026-08-11.
    """
    shield = ThresholdSafetyShield(min_depth_m=float(shield_trigger_depth_m))
    interventions_on: List[List[bool]] = []
    collided_on: List[List[bool]] = []
    near_coll_on: List[List[bool]] = []
    near_coll_off: List[List[bool]] = []
    depth_steps = 0

    for epi in start_episodes:
        if hasattr(policy, "reset"):
            policy.reset()
        # The shield instance is reused across episodes and latches on first
        # breach — clear the latch so each episode starts un-engaged.
        if hasattr(shield, "reset"):
            shield.reset()
        ep_on = _run_one_resilient(
            env, policy, epi, max_steps=max_steps, reward_cfg=reward_cfg,
            shield=shield, depth_predictor=depth_predictor_on,
        )

        if hasattr(policy, "reset"):
            policy.reset()
        ep_off = _run_one_resilient(
            env, policy, epi, max_steps=max_steps, reward_cfg=reward_cfg,
            shield=None, depth_predictor=None,
        )

        # A transient reset/health failure on EITHER arm drops the whole start,
        # so on/off stay paired at the same N (the ratio is a controlled compare).
        if ep_on is None or ep_off is None:
            continue

        m_on = _episode_masks(ep_on, near_collision_depth_m=near_collision_depth_m)
        interventions_on.append(m_on["intervention"])
        collided_on.append(m_on["collided"])
        near_coll_on.append(m_on["near"])
        depth_steps += int(m_on["depth_steps"])

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
