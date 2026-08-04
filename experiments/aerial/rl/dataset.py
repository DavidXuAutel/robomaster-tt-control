"""Persist collected episodes + score their quality (V1 WM-training input).

The V0 collector holds episodes only in the in-memory ``ReplayBuffer``; the
world-model milestone (V1) needs them on disk. This module turns an episode
(list of ``Transition``) into:

  * a compressed ``.npz`` — the heavy per-frame tensors (RGB stack, proprio,
    actions, rewards, done/collided masks, optional depth). One file per episode.
  * a ``quality_report`` — cheap statistics that answer the only question a raw
    collection can't assume: *is this data non-trivial?* A frozen renderer
    (identical frames), a drone that never moved (API control silently off), or
    an all-zero reward channel are all invisible to "it ran without error" but
    fatal to WM training. ``assert_nontrivial`` turns the dangerous ones into
    hard failures.

Pure numpy + json; no torch / cv2 / AirSim, so it is fully unit-testable and the
writer runs anywhere the collector does.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiments.aerial.rl.buffer import Transition
from experiments.aerial.rl.env.obs import Observation, depth_sanity_detail

# A frozen renderer / dead API-control run is the failure we most need to catch.
MIN_FRAME_VARIATION = 1e-3   # mean |Δ| between consecutive RGB frames (uint8 scale)
MIN_RGB_STD = 1.0            # pixel std across the episode (all-black/constant guard)
MIN_PATH_LENGTH_M = 1e-3     # summed |Δposition| — did the vehicle move at all?

# An episode that ends in a collision after <=this many steps is an instant
# crash — a bad spawn / start pose, not a learnable trajectory. Quarantined
# (excluded from the usable set) rather than hard-failed: for steps<=1 the path
# length is structurally 0 (a lone position can't be differenced), so `collided`
# is the ONLY valid discriminator here — the frozen/moved path checks can't see it.
SPAWN_COLLISION_MAX_STEPS = 2
# Run level: a handful of instant crashes is expected, but if more than this
# fraction of a collection is quarantined the start-pose distribution is broken
# and the whole run should fail.
MAX_QUARANTINE_FRACTION = 0.2


def episode_arrays(transitions: Sequence[Transition]) -> Dict[str, np.ndarray]:
    """Stack an episode's transitions into per-field arrays for ``.npz`` storage."""
    if not transitions:
        raise ValueError("cannot serialize an empty episode")
    arrays: Dict[str, np.ndarray] = {
        "rgb": np.stack([t.obs.rgb for t in transitions]),                 # [N,H,W,3] u8
        "proprio": np.stack([t.obs.proprio4() for t in transitions]),      # [N,4] f32
        "actions": np.stack([np.asarray(t.action, np.float32) for t in transitions]),  # [N,4]
        "rewards": np.asarray([t.reward for t in transitions], np.float32),
        "dones": np.asarray([t.done for t in transitions], np.bool_),
        "collided": np.asarray([bool(t.obs.collided) for t in transitions], np.bool_),
    }
    # Depth is optional per-step (grab_depth); store it only if every frame has it.
    if all(t.obs.depth is not None for t in transitions):
        arrays["depth"] = np.stack([np.asarray(t.obs.depth, np.float32) for t in transitions])
    return arrays


def write_episode(out_dir: Path, index: int, transitions: Sequence[Transition]) -> Path:
    """Write one episode to ``out_dir/episode_{index:05d}.npz``; return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{index:05d}.npz"
    np.savez_compressed(path, **episode_arrays(transitions))
    return path


def load_episode(path: Path) -> List[Transition]:
    """Rehydrate an episode written by ``write_episode``.

    Velocity is not stored (proprio is 4-D); reconstructed ``state`` pads
    ``vx,vy,vz = 0``. ``next_obs`` is the following frame's obs (last step
    duplicates obs). Enough for buffer window sampling + stub WM bring-up;
    not a bit-exact clone of the live ``Transition`` graph.
    """
    path = Path(path)
    raw = np.load(path)
    rgb = raw["rgb"]
    proprio = raw["proprio"]
    actions = raw["actions"]
    rewards = raw["rewards"]
    dones = raw["dones"]
    collided = raw["collided"]
    depth = raw["depth"] if "depth" in raw.files else None
    n = int(rgb.shape[0])
    if n == 0:
        raise ValueError(f"empty episode file: {path}")

    def _obs_at(i: int) -> Observation:
        x, y, z, yaw = (float(v) for v in proprio[i])
        state = np.array([x, y, z, 0.0, 0.0, 0.0, yaw], dtype=np.float32)
        d = None if depth is None else np.asarray(depth[i], dtype=np.float32)
        return Observation(
            rgb=np.asarray(rgb[i], dtype=np.uint8),
            state=state,
            collided=bool(collided[i]),
            depth=d,
        )

    transitions: List[Transition] = []
    for i in range(n):
        obs = _obs_at(i)
        next_obs = _obs_at(i + 1) if i + 1 < n else obs
        transitions.append(
            Transition(
                obs=obs,
                action=np.asarray(actions[i], dtype=np.float32),
                reward=float(rewards[i]),
                done=bool(dones[i]),
                next_obs=next_obs,
            )
        )
    return transitions


def load_dataset(
    out_dir: Path,
    *,
    skip_quarantined: bool = True,
) -> List[List[Transition]]:
    """Load every ``episode_*.npz`` under ``out_dir`` (sorted by name).

    When ``skip_quarantined`` is true, instant-crash episodes are omitted so a
    V0 smoke corpus can still exercise the load→buffer→stub-WM path without
    feeding spawn-collision junk into window sampling.
    """
    out_dir = Path(out_dir)
    paths = sorted(out_dir.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode_*.npz under {out_dir}")
    episodes: List[List[Transition]] = []
    for p in paths:
        ep = load_episode(p)
        if skip_quarantined and quarantine_reasons(quality_report(ep)):
            continue
        episodes.append(ep)
    return episodes


def quality_report(transitions: Sequence[Transition]) -> Dict[str, Any]:
    """Cheap non-triviality statistics over one episode (JSON-serializable)."""
    arr = episode_arrays(transitions)
    rgb = arr["rgb"].astype(np.float32)
    pos = arr["proprio"][:, :3].astype(np.float64)
    rewards = arr["rewards"].astype(np.float64)

    frame_var = (
        float(np.mean(np.abs(np.diff(rgb, axis=0)))) if rgb.shape[0] > 1 else 0.0
    )
    path_len = (
        float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        if pos.shape[0] > 1 else 0.0
    )
    report: Dict[str, Any] = {
        "steps": int(rgb.shape[0]),
        "rgb_frame_variation": frame_var,      # ~0 => renderer frozen
        "rgb_std": float(rgb.std()),           # ~0 => constant/black image
        "path_length_m": path_len,             # ~0 => vehicle never moved
        "pos_min": pos.min(axis=0).tolist(),
        "pos_max": pos.max(axis=0).tolist(),
        "reward_sum": float(rewards.sum()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "reward_nonzero_frac": float(np.mean(rewards != 0.0)),
        "collisions": int(arr["collided"].sum()),
        "has_depth": "depth" in arr,
    }
    if "depth" in arr:
        # Reuse the validated depth gate on the mean frame.
        report["depth"] = depth_sanity_detail(arr["depth"].mean(axis=0))
    return report


def assert_nontrivial(report: Dict[str, Any]) -> List[str]:
    """Return a list of HARD failures (empty == data is usable for WM training).

    These are the silent-but-fatal conditions: a frozen renderer, a constant
    image, or a vehicle that never moved. A flat reward channel is a *warning*
    (a legitimately stationary episode can be all-zero), not a hard failure.
    """
    failures: List[str] = []
    if report["steps"] <= 0:
        failures.append("empty episode (0 steps)")
    if report["rgb_frame_variation"] < MIN_FRAME_VARIATION and report["steps"] > 1:
        failures.append(
            f"RGB frames barely change ({report['rgb_frame_variation']:.2e} "
            f"< {MIN_FRAME_VARIATION}) — renderer may be frozen"
        )
    if report["rgb_std"] < MIN_RGB_STD:
        failures.append(
            f"RGB nearly constant (std {report['rgb_std']:.2e} < {MIN_RGB_STD}) "
            "— black / blank frames"
        )
    if report["path_length_m"] < MIN_PATH_LENGTH_M and report["steps"] > 1:
        failures.append(
            f"vehicle did not move (path {report['path_length_m']:.2e} m) "
            "— API control may be off"
        )
    return failures


def quarantine_reasons(report: Dict[str, Any]) -> List[str]:
    """Soft exclusions: episodes that ran cleanly but aren't usable training data.

    Distinct from ``assert_nontrivial`` (silent-but-fatal, always a hard fail): a
    quarantined episode's collection succeeded — it's just untrustworthy. The
    canonical case is the *instant crash*: a 1–2 step episode ending in a
    collision (spawn-inside-geometry or a start pose facing a wall). The path
    check in ``assert_nontrivial`` can't catch these — a 1-step episode has a
    structurally-zero path — so ``collided`` is the discriminator. These are
    excluded per-episode; the *run-level* gate (``MAX_QUARANTINE_FRACTION``)
    decides whether there are enough of them to condemn the whole collection.
    """
    reasons: List[str] = []
    if report["collisions"] > 0 and 0 < report["steps"] <= SPAWN_COLLISION_MAX_STEPS:
        reasons.append(
            f"instant crash: collision within {report['steps']} step(s) "
            "— suspected bad spawn / start pose"
        )
    return reasons


def write_manifest(out_dir: Path, entries: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> Path:
    """Write the dataset index (``manifest.json``) listing every episode + meta."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    payload = {"meta": meta or {}, "episodes": entries}
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_quality_summary(out_dir: Path, per_episode: List[Dict[str, Any]]) -> Path:
    """Aggregate per-episode reports into ``QUALITY_SUMMARY.json`` (tracked in git)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "QUALITY_SUMMARY.json"

    def _agg(key: str) -> Dict[str, float]:
        vals = [float(r[key]) for r in per_episode]
        return {"min": min(vals), "max": max(vals), "mean": float(np.mean(vals))}

    quarantined = int(sum(1 for r in per_episode if quarantine_reasons(r)))
    summary = {
        "episodes": len(per_episode),
        "quarantined": quarantined,           # instant-crash / bad-spawn episodes
        "usable": len(per_episode) - quarantined,
        "total_steps": int(sum(r["steps"] for r in per_episode)),
        "total_collisions": int(sum(r["collisions"] for r in per_episode)),
        "rgb_frame_variation": _agg("rgb_frame_variation"),
        "path_length_m": _agg("path_length_m"),
        "reward_sum": _agg("reward_sum"),
        "any_depth": any(r["has_depth"] for r in per_episode),
    } if per_episode else {"episodes": 0}
    path.write_text(json.dumps(summary, indent=2))
    return path
