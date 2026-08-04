"""V0 four-signal numeric thresholds + pure-numpy scorers (frozen §4.1).

Torch-free so Mac CI can lock the gate math before H100 has depth/VIO weights.
``_v0_gate.py`` imports these thresholds; changing a number here requires a
dated revision of the frozen spec §4.1 table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from experiments.aerial.rl import vio as vio_lib


@dataclass(frozen=True)
class V0GateThresholds:
    """Pinned numbers from frozen spec §4.1 (2026-08-04 evaluation patch)."""

    # ① learning
    loss_drop_ratio: float = 0.98
    collapse_entropy_frac: float = 0.10
    depth_absrel_max: float = 0.30
    # ② progress vs random
    n_eval_episodes: int = 16
    progress_margin: float = 5.0
    dist_margin_m: float = 3.0
    # ③ scale
    min_motion_m: float = 0.5
    scale_rel_err_max: float = 0.25
    scale_eps: float = 1e-3
    # ④ near-collision / shield
    near_collision_depth_m: float = 1.5
    intervention_before_contact_min: float = 0.50
    near_coll_rate_ratio_max: float = 0.80


DEFAULT_THRESHOLDS = V0GateThresholds()


def check_learning_curves(
    losses: Sequence[float],
    recons: Sequence[float],
    ent_fracs: Sequence[float],
    *,
    thr: V0GateThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Signal ①a–c (same logic as ``_wm_train_validate._check_learning``)."""
    losses_a = np.asarray(losses, dtype=np.float64)
    recons_a = np.asarray(recons, dtype=np.float64)
    ents = np.asarray(ent_fracs, dtype=np.float64)
    if losses_a.size == 0:
        return {"ok": False, "reason": "empty loss curve"}
    if not np.all(np.isfinite(losses_a)):
        return {"ok": False, "reason": "non-finite loss"}
    k = max(1, losses_a.size // 10)
    first, last = float(np.mean(losses_a[:k])), float(np.mean(losses_a[-k:]))
    recon_first, recon_last = float(np.mean(recons_a[:k])), float(np.mean(recons_a[-k:]))
    min_ent = float(np.min(ents)) if ents.size else 0.0
    loss_ok = last < first * thr.loss_drop_ratio
    recon_ok = recon_last <= recon_first
    collapse_ok = min_ent >= thr.collapse_entropy_frac
    return {
        "ok": bool(loss_ok and recon_ok and collapse_ok),
        "loss_first": first,
        "loss_last": last,
        "loss_ok": loss_ok,
        "recon_first": recon_first,
        "recon_last": recon_last,
        "recon_ok": recon_ok,
        "min_ent": min_ent,
        "collapse_ok": collapse_ok,
    }


def depth_absrel(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    max_depth_m: Optional[float] = 200.0,
) -> float:
    """Median AbsRel = median(|pred-gt|/gt) over finite positive gt pixels.

    ``max_depth_m`` excludes outdoor AirSim far-plane / sky fill (often >1 km)
    so the metric matches navigational near/mid field used by DepthHead train
    and ``depth_min_pred`` safety. Pass ``None`` to score all finite pixels.
    """
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    g = np.asarray(gt, dtype=np.float64).reshape(-1)
    m = np.isfinite(p) & np.isfinite(g) & (g > 1e-6)
    if max_depth_m is not None:
        m &= g <= float(max_depth_m)
    if not np.any(m):
        return float("nan")
    return float(np.median(np.abs(p[m] - g[m]) / g[m]))


def check_depth_absrel(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    thr: V0GateThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Signal ①d — required once a depth head + GT depth corpus exist."""
    val = depth_absrel(pred, gt)
    if not np.isfinite(val):
        return {"ok": False, "absrel": val, "reason": "no valid depth pixels"}
    return {"ok": bool(val <= thr.depth_absrel_max), "absrel": val}


def check_progress_vs_random(
    policy_progress_sums: Sequence[float],
    random_progress_sums: Sequence[float],
    policy_final_dists: Sequence[float],
    random_final_dists: Sequence[float],
    *,
    thr: V0GateThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Signal ② — pass if progress margin OR distance margin holds."""
    pp = np.asarray(policy_progress_sums, dtype=np.float64)
    rp = np.asarray(random_progress_sums, dtype=np.float64)
    pd = np.asarray(policy_final_dists, dtype=np.float64)
    rd = np.asarray(random_final_dists, dtype=np.float64)
    if min(pp.size, rp.size, pd.size, rd.size) == 0:
        return {"ok": False, "reason": "empty eval arrays"}
    mean_pp, mean_rp = float(np.mean(pp)), float(np.mean(rp))
    mean_pd, mean_rd = float(np.mean(pd)), float(np.mean(rd))
    progress_ok = mean_pp >= mean_rp + thr.progress_margin
    dist_ok = mean_pd <= mean_rd - thr.dist_margin_m
    return {
        "ok": bool(progress_ok or dist_ok),
        "mean_progress_policy": mean_pp,
        "mean_progress_random": mean_rp,
        "progress_ok": progress_ok,
        "mean_final_dist_policy": mean_pd,
        "mean_final_dist_random": mean_rd,
        "dist_ok": dist_ok,
        "n": int(min(pp.size, rp.size)),
    }


def check_scale_consistency(
    vel: np.ndarray,
    timestamps: np.ndarray,
    depth: np.ndarray,
    *,
    thr: V0GateThresholds = DEFAULT_THRESHOLDS,
    fallback_hz: float = 8.0,
) -> Dict[str, Any]:
    """Signal ③ via ``vio.window_scale_report``."""
    report = vio_lib.window_scale_report(
        vel,
        timestamps,
        depth,
        fallback_hz=fallback_hz,
        min_motion_m=thr.min_motion_m,
        eps=thr.scale_eps,
    )
    med = float(report["median_rel_err"])
    n_valid = int(report["n_valid"])
    if n_valid == 0 or not np.isfinite(med):
        return {"ok": False, "median_rel_err": med, "n_valid": n_valid,
                "reason": "no motion windows ≥ min_motion_m"}
    return {
        "ok": bool(med <= thr.scale_rel_err_max),
        "median_rel_err": med,
        "n_valid": n_valid,
        "rel_err": report["rel_err"],
    }


def near_collision_frame_mask(
    depth_min: np.ndarray,
    *,
    thr: V0GateThresholds = DEFAULT_THRESHOLDS,
) -> np.ndarray:
    """Per-step near-collision when GT min depth < threshold."""
    d = np.asarray(depth_min, dtype=np.float64)
    return np.isfinite(d) & (d < thr.near_collision_depth_m)


def check_shield_effectiveness(
    *,
    interventions_on: Sequence[Sequence[bool]],
    collided_on: Sequence[Sequence[bool]],
    near_coll_on: Sequence[Sequence[bool]],
    near_coll_off: Sequence[Sequence[bool]],
    thr: V0GateThresholds = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    """Signal ④ — intervention-before-contact + near-coll rate ratio.

    Each argument is a list of per-episode boolean sequences (length = steps).
    """
    if not interventions_on:
        return {"ok": False, "reason": "empty shield-on episodes"}

    before = []
    for interv, coll in zip(interventions_on, collided_on):
        interv_a = np.asarray(interv, dtype=bool)
        coll_a = np.asarray(coll, dtype=bool)
        if not np.any(coll_a):
            continue
        first_c = int(np.argmax(coll_a))
        if not np.any(interv_a):
            before.append(False)
            continue
        first_i = int(np.argmax(interv_a))
        before.append(first_i < first_c)
    if before:
        before_frac = float(np.mean(before))
    else:
        before_frac = 1.0  # no contacts → vacuously OK on this sub-metric

    def _rate(episodes: Sequence[Sequence[bool]]) -> float:
        total = sum(len(e) for e in episodes)
        if total == 0:
            return float("nan")
        hits = sum(int(np.count_nonzero(e)) for e in episodes)
        return hits / total

    rate_on = _rate(near_coll_on)
    rate_off = _rate(near_coll_off)
    if not np.isfinite(rate_on) or not np.isfinite(rate_off) or rate_off <= 0:
        ratio_ok = False
        ratio = float("nan")
    else:
        ratio = rate_on / rate_off
        ratio_ok = ratio <= thr.near_coll_rate_ratio_max

    before_ok = before_frac >= thr.intervention_before_contact_min
    return {
        "ok": bool(before_ok and ratio_ok),
        "intervention_before_contact_frac": before_frac,
        "before_ok": before_ok,
        "near_coll_rate_on": rate_on,
        "near_coll_rate_off": rate_off,
        "near_coll_rate_ratio": ratio,
        "ratio_ok": ratio_ok,
        "n_contact_episodes": len(before),
    }


def aggregate_v0_verdict(
    results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Combine signal dicts ``{"1": ..., "2": ..., "3": ..., "4": ...}``."""
    keys = ("1", "2", "3", "4")
    missing = [k for k in keys if k not in results]
    if missing:
        return {"ok": False, "passed": {}, "reason": f"missing signals {missing}"}
    passed = {k: bool(results[k].get("ok")) for k in keys}
    return {
        "ok": all(passed.values()),
        "passed": passed,
        "thresholds": asdict(DEFAULT_THRESHOLDS),
        "details": dict(results),
    }
