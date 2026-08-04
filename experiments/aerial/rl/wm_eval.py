"""Open-loop imagination-fidelity evaluation (V1 "多步 rollout 达标", §7).

The V1 gate's *floor* — non-divergence (§9) — is already validated by
``_wm_train_validate``. This module fills the rest of "多步 rollout 达标":
does the world model, rolled open-loop from a real start state under the
*recorded* actions, actually **track** the real trajectory, not merely stay
bounded?

Pass口径 (user decision, §1.5 — our 4-D kinematic SEARCH regime has no paper
number, so it is BASELINE-RELATIVE + BOUNDED-GROWTH, never a magic absolute):

  * REWARD head — per-horizon MAE must beat a trivial constant-mean predictor,
    and the error must grow no faster than ~linearly in the horizon (bounded
    compounding). 1-step error must itself be small vs that baseline.
  * P_COLL head — trajectory-level separation (max predicted p_coll over the
    horizon) between colliding and non-colliding trajectories, measured by
    AUROC, must clear a margin over chance (a constant prior scores 0.5).
  * DONE head — per-step done accuracy must beat the majority-class baseline.

Everything here is pure numpy and works with ``StubLatentDynamics``, so the
harness + verdict logic are unit-testable on the GPU-less dev host. The torch
checkpoint eval (``_wm_fidelity_eval``) imports these same functions on the
H100 so the metric math has a single source of truth.

ALIGNMENT: rollout step ``t`` calls ``step(z_t, a_t)`` with the recorded action
``window[t].action`` and its heads are compared to the recorded consequence at
``window[t]`` (reward ``r_t``, ``obs.collided``, ``done``). There is a ±1-step
ambiguity in exactly when a contact is *labeled* vs *predicted*; the p_coll
metric is deliberately trajectory-level (max over the horizon) so it is robust
to that timing, and the ambiguity is noted in the fidelity doc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiments.aerial.rl.dynamics import LatentDynamics
from experiments.aerial.rl.imagination import MAX_IMAGINATION_HORIZON

# -- pass thresholds (project-tuned for OUR regime, §1.5 — NOT paper numbers) --
#: WM reward MAE must be below the constant-mean baseline on at least this
#: fraction of horizons (baseline-relative, not an absolute tolerance).
REWARD_BEAT_FRAC = 0.8
#: bounded compounding: err(H) <= err(1) * (1 + GROWTH_SLOPE_TOL * (H-1)).
#: 1.0 = "no worse than linear growth" with a 1× slack; > that reads as the
#: multi-step error compounding faster than linearly = not "达标".
GROWTH_SLOPE_TOL = 1.0
#: trajectory-level collision separation floor (0.5 = chance).
PCOLL_AUROC_MIN = 0.65
#: done accuracy must beat the majority-class baseline by at least this margin.
DONE_ACC_MARGIN = 0.0

_EPS = 1e-8


@dataclass
class RolloutTrace:
    """One open-loop rollout's predicted heads vs the recorded ground truth.

    All arrays are length ``H`` (the number of stepped predictions) except
    ``latent_norm`` which is ``H+1`` (includes the encoded start). ``valid``
    masks out horizons past the first real termination, where comparing an
    imagined step to a non-existent real step is meaningless.
    """

    reward_pred: np.ndarray      # [H]
    p_coll_pred: np.ndarray      # [H]
    done_pred: np.ndarray        # [H] bool
    reward_real: np.ndarray      # [H]
    collided_real: np.ndarray    # [H] bool
    done_real: np.ndarray        # [H] bool
    latent_norm: np.ndarray      # [H+1]
    valid: np.ndarray            # [H] bool — horizon still within the real episode


def open_loop_rollout(
    dynamics: LatentDynamics,
    window: Sequence[Any],
    *,
    horizon: Optional[int] = None,
    max_horizon: int = MAX_IMAGINATION_HORIZON,
) -> RolloutTrace:
    """Encode ``window[0].obs`` then step with the recorded actions.

    Rolls ``min(horizon or len(window), max_horizon, len(window))`` steps —
    never past the recorded actions, and never past the §9 cap. The real
    trajectory may terminate before the horizon; steps at/after the first real
    ``done`` are marked ``valid=False`` (the ``done`` step itself stays valid so
    the done head is scored on the actual termination).
    """
    n = len(window)
    if n < 1:
        raise ValueError("window must have >= 1 transition")
    H = min(horizon if horizon is not None else n, max_horizon, n)
    if H < 1:
        raise ValueError("resolved horizon must be >= 1")

    z = np.asarray(dynamics.encode(window[0].obs), dtype=np.float64)
    reward_pred = np.zeros(H)
    p_coll_pred = np.zeros(H)
    done_pred = np.zeros(H, dtype=bool)
    reward_real = np.zeros(H)
    collided_real = np.zeros(H, dtype=bool)
    done_real = np.zeros(H, dtype=bool)
    latent_norm = np.zeros(H + 1)
    latent_norm[0] = float(np.linalg.norm(z))

    first_done = None
    for t in range(H):
        tr = window[t]
        out = dynamics.step(z, np.asarray(tr.action, dtype=np.float64).reshape(4))
        z = np.asarray(out.z_next, dtype=np.float64).reshape(-1)
        reward_pred[t] = float(out.progress)
        p_coll_pred[t] = float(out.p_coll)
        done_pred[t] = bool(out.done)
        reward_real[t] = float(tr.reward)
        collided_real[t] = bool(getattr(tr.obs, "collided", False))
        done_real[t] = bool(tr.done)
        latent_norm[t + 1] = float(np.linalg.norm(z))
        if first_done is None and bool(tr.done):
            first_done = t

    valid = np.ones(H, dtype=bool)
    if first_done is not None:
        valid[first_done + 1:] = False  # the done step itself stays valid
    return RolloutTrace(
        reward_pred=reward_pred, p_coll_pred=p_coll_pred, done_pred=done_pred,
        reward_real=reward_real, collided_real=collided_real, done_real=done_real,
        latent_norm=latent_norm, valid=valid,
    )


# -- metric primitives -------------------------------------------------------
def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (enough for Mann-Whitney AUROC)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average of 1-based positions i..j
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank-sum (Mann-Whitney U) identity; ties handled.

    Returns NaN when a class is absent (AUROC is undefined) — callers treat
    NaN as "not enough signal to judge", not as a pass.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(scores)
    sum_pos = float(ranks[labels].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probs.size == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def _masked_mae_curve(pred: np.ndarray, real: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-horizon MAE, averaging only over trajectories valid at that horizon."""
    err = np.abs(pred - real)
    err = np.where(valid, err, np.nan)
    with np.errstate(invalid="ignore"):
        curve = np.nanmean(err, axis=0)
    return curve  # [H]; NaN at horizons no trajectory reaches


def growth_bounded(err_curve: np.ndarray, slope_tol: float = GROWTH_SLOPE_TOL) -> bool:
    """True if the error grows no faster than ~linearly with the horizon.

    Compares the deepest finite horizon to the 1-step error: bounded when
    ``err(H) <= err(1) * (1 + slope_tol * (H - 1))``. A flat or shrinking curve
    trivially passes; a super-linearly compounding curve fails.
    """
    curve = np.asarray(err_curve, dtype=float)
    finite = np.where(np.isfinite(curve))[0]
    if finite.size == 0:
        return False
    h0 = int(finite[0])
    h_last = int(finite[-1])
    base = curve[h0]
    if not np.isfinite(base) or base <= _EPS:
        return True  # near-zero 1-step error: nothing to compound
    span = h_last - h0
    ceiling = base * (1.0 + slope_tol * span)
    return bool(curve[h_last] <= ceiling + _EPS)


# -- aggregate + verdict -----------------------------------------------------
def aggregate(traces: List[RolloutTrace]) -> Dict[str, Any]:
    """Stack per-trajectory traces into per-horizon curves + scalar summaries."""
    if not traces:
        raise ValueError("no rollout traces to aggregate")
    H = max(tr.reward_pred.shape[0] for tr in traces)

    def _pad(a: np.ndarray, fill: float) -> np.ndarray:
        out = np.full(H, fill, dtype=float)
        out[: a.shape[0]] = a
        return out

    rp = np.stack([_pad(t.reward_pred, np.nan) for t in traces])
    rr = np.stack([_pad(t.reward_real, np.nan) for t in traces])
    vv = np.stack([_pad(t.valid.astype(float), 0.0) for t in traces]).astype(bool)

    wm_mae = _masked_mae_curve(rp, rr, vv)
    # constant-mean baseline: predict the global mean real reward everywhere.
    global_mean = float(np.nanmean(np.where(vv, rr, np.nan)))
    mean_pred = np.full_like(rr, global_mean)
    base_mae = _masked_mae_curve(mean_pred, rr, vv)
    # persistence baseline (reference only, not gated — degenerate at t=0).
    persist_pred = np.repeat(rr[:, :1], H, axis=1)
    persist_mae = _masked_mae_curve(persist_pred, rr, vv)

    # trajectory-level collision separation.
    traj_label = np.array([bool(np.any(t.collided_real[t.valid])) for t in traces])
    traj_score = np.array([
        float(np.max(t.p_coll_pred[t.valid])) if np.any(t.valid) else 0.0
        for t in traces
    ])
    coll_auroc = auroc(traj_score, traj_label)

    # per-step done accuracy over valid steps + majority-class baseline.
    dp = np.concatenate([t.done_pred[t.valid] for t in traces]) if traces else np.array([])
    dr = np.concatenate([t.done_real[t.valid] for t in traces]) if traces else np.array([])
    if dr.size:
        done_acc = float(np.mean(dp == dr))
        p1 = float(np.mean(dr))
        majority_acc = max(p1, 1.0 - p1)
    else:
        done_acc = majority_acc = float("nan")

    latent_norm_max = float(max(np.max(t.latent_norm) for t in traces))
    return {
        "horizon": H,
        "n_traj": len(traces),
        "wm_reward_mae": wm_mae,
        "baseline_reward_mae": base_mae,
        "persistence_reward_mae": persist_mae,
        "reward_global_mean": global_mean,
        "coll_auroc": coll_auroc,
        "coll_traj_pos": int(traj_label.sum()),
        "coll_traj_neg": int((~traj_label).sum()),
        "done_acc": done_acc,
        "done_majority_acc": majority_acc,
        "latent_norm_max": latent_norm_max,
    }


def fidelity_verdict(agg: Dict[str, Any]) -> Dict[str, Any]:
    """Baseline-relative + bounded-growth PASS/FAIL over the aggregated curves."""
    wm = np.asarray(agg["wm_reward_mae"], dtype=float)
    base = np.asarray(agg["baseline_reward_mae"], dtype=float)
    both = np.isfinite(wm) & np.isfinite(base)

    # reward: beat the constant-mean baseline on >= REWARD_BEAT_FRAC of horizons,
    # 1-step error below baseline, and bounded compounding.
    if both.any():
        beat = (wm[both] < base[both])
        beat_frac = float(np.mean(beat))
        first = int(np.where(both)[0][0])
        one_step_ok = bool(wm[first] < base[first] + _EPS)
    else:
        beat_frac, one_step_ok = 0.0, False
    growth_ok = growth_bounded(wm)
    reward_ok = bool(beat_frac >= REWARD_BEAT_FRAC and one_step_ok and growth_ok)

    # collision: trajectory-level separation over chance. NaN (a class absent)
    # is "insufficient signal" -> not a pass.
    au = agg["coll_auroc"]
    coll_ok = bool(np.isfinite(au) and au >= PCOLL_AUROC_MIN)

    # done: beat majority-class baseline.
    da, ma = agg["done_acc"], agg["done_majority_acc"]
    done_ok = bool(np.isfinite(da) and np.isfinite(ma) and da >= ma + DONE_ACC_MARGIN)

    passed = reward_ok and coll_ok and done_ok
    return {
        "reward_ok": reward_ok,
        "reward_beat_frac": beat_frac,
        "reward_growth_ok": growth_ok,
        "coll_ok": coll_ok,
        "done_ok": done_ok,
        "passed": passed,
    }


def evaluate(
    dynamics: LatentDynamics,
    windows: Sequence[Sequence[Any]],
    *,
    horizon: Optional[int] = None,
    max_horizon: int = MAX_IMAGINATION_HORIZON,
) -> Dict[str, Any]:
    """End-to-end: rollout every window, aggregate, and render the verdict."""
    traces = [
        open_loop_rollout(dynamics, w, horizon=horizon, max_horizon=max_horizon)
        for w in windows if len(w) >= 1
    ]
    agg = aggregate(traces)
    verdict = fidelity_verdict(agg)
    return {"agg": agg, "verdict": verdict}
