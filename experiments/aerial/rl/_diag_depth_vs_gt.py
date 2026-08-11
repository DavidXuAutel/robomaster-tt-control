"""READ-ONLY diagnostic: is predicted D̂ optimistic (over-reads GT) near obstacles?

Answers the ④ late-trigger hypothesis WITHOUT a rollout / AirSim. The shield
triggers on the predictor's full-field min D̂ < ``trigger_m`` (3.0); the frozen
④a near-collision metric flags a frame when GT full-field min < ``near_m`` (1.5).
If D̂ systematically reads FARTHER than GT in the near band, the 3.0 m trigger
only fires once GT is already inside the 1.5 m band → contact before the shield
reacts (observed: 4/7 shield-ON episodes still collided, before_frac=0.25).

This loads a collection corpus (npz with stored GT depth + RGB), runs the SAME
``DepthMinPredictor`` the shield uses (identical n-frame window + left-pad warm-up),
pairs D̂ against GT per frame, and bins by GT depth.

It touches NOTHING governed: no gate/spec/config/threshold/weight change, no env,
no flags — it only READS the checkpoint and the corpus and prints statistics.

    python -m experiments.aerial.rl._diag_depth_vs_gt \
      --dataset <corpus dir with depth> --depth-ckpt <DA3 head> --device cuda \
      --emit artifacts/diag_depth_vs_gt.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import deque
from typing import Any, Dict, List

import numpy as np

from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.depth_predictor import DepthMinPredictor
from experiments.aerial.rl.v0_rollout_eval import _forward_min_depth, _full_min_depth


# GT bin edges (m); the [near, trigger] bin is the reaction window that must fire.
_BIN_EDGES = [0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, float("inf")]


def _bin_label(lo: float, hi: float) -> str:
    hi_s = "inf" if hi == float("inf") else f"{hi:g}"
    return f"[{lo:g},{hi_s})"


def _summarize(gt: np.ndarray, dhat: np.ndarray, *, trigger_m: float) -> List[Dict[str, Any]]:
    """Per-GT-bin stats of D̂: count, D̂ percentiles, P(D̂<trigger), median AbsRel."""
    out: List[Dict[str, Any]] = []
    for lo, hi in zip(_BIN_EDGES[:-1], _BIN_EDGES[1:]):
        m = (gt >= lo) & (gt < hi)
        n = int(m.sum())
        if n == 0:
            out.append({"gt_bin": _bin_label(lo, hi), "n": 0})
            continue
        d = dhat[m]
        g = gt[m]
        absrel = np.abs(d - g) / np.clip(g, 1e-6, None)
        out.append({
            "gt_bin": _bin_label(lo, hi),
            "n": n,
            "dhat_p10": round(float(np.percentile(d, 10)), 3),
            "dhat_p50": round(float(np.percentile(d, 50)), 3),
            "dhat_p90": round(float(np.percentile(d, 90)), 3),
            "dhat_mean": round(float(np.mean(d)), 3),
            # Fraction the shield WOULD trigger on (D̂ < trigger_m) for GT in this bin.
            "p_trigger": round(float(np.mean(d < trigger_m)), 3),
            # Fraction D̂ reads FARTHER than GT (optimistic → late).
            "p_overread": round(float(np.mean(d > g)), 3),
            "median_absrel_min": round(float(np.median(absrel)), 3),
        })
    return out


def _print_table(title: str, rows: List[Dict[str, Any]], *, trigger_m: float) -> None:
    print(f"\n=== {title} (trigger D̂<{trigger_m:g} m) ===")
    print(f"{'gt_bin':>12} {'n':>5} {'D̂p10':>7} {'D̂p50':>7} {'D̂p90':>7} "
          f"{'D̂mean':>7} {'P(trig)':>8} {'P(over)':>8} {'AbsRelmin':>10}")
    for r in rows:
        if r["n"] == 0:
            print(f"{r['gt_bin']:>12} {0:>5}")
            continue
        print(f"{r['gt_bin']:>12} {r['n']:>5} {r['dhat_p10']:>7.3f} {r['dhat_p50']:>7.3f} "
              f"{r['dhat_p90']:>7.3f} {r['dhat_mean']:>7.3f} {r['p_trigger']:>8.3f} "
              f"{r['p_overread']:>8.3f} {r['median_absrel_min']:>10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="corpus dir (episode_*.npz with depth)")
    ap.add_argument("--depth-ckpt", required=True, help="DepthHead checkpoint (.pt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--center-frac", type=float, default=0.3,
                    help="forward central-crop fraction (matches scan proxy)")
    ap.add_argument("--trigger-m", type=float, default=3.0, help="shield reaction standoff")
    ap.add_argument("--near-m", type=float, default=1.5, help="frozen ④a near-collision depth")
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--emit", default=None, help="write JSON summary here")
    args = ap.parse_args()

    import torch  # noqa: F401  (runs on GPU host; imported here, not at module load)

    pred = DepthMinPredictor.from_checkpoint(Path(args.depth_ckpt), device=args.device)
    model = pred._model
    if model is None:
        raise SystemExit("checkpoint produced no model")
    n_frames = int(pred.n_frames)

    episodes = ds.load_dataset(Path(args.dataset), skip_quarantined=True)
    if args.max_episodes:
        episodes = episodes[: int(args.max_episodes)]

    gt_full: List[float] = []
    dhat_full: List[float] = []
    gt_fwd: List[float] = []
    dhat_fwd: List[float] = []
    n_no_depth = 0
    n_frames_total = 0

    for ep in episodes:
        hist: deque = deque(maxlen=n_frames)
        for t in ep:
            rgb = np.asarray(t.obs.rgb, dtype=np.uint8)
            hist.append(rgb)
            depth_gt = getattr(t.obs, "depth", None)
            if depth_gt is None:
                n_no_depth += 1
                continue
            frames = list(hist)
            while len(frames) < n_frames:
                frames.insert(0, frames[0])  # left-pad warm-up, identical to predict_min
            stack = np.stack(frames[-n_frames:], axis=0)  # [L,H,W,3]
            tensor = torch.from_numpy(stack).unsqueeze(0)  # [1,L,H,W,3]
            with torch.no_grad():
                dmap_t, _ = model.predict_from_window(tensor.to(args.device))
            dmap = np.squeeze(dmap_t.squeeze(0).detach().float().cpu().numpy())

            g_full = _full_min_depth(depth_gt)
            d_full = _full_min_depth(dmap)
            if np.isfinite(g_full) and np.isfinite(d_full):
                gt_full.append(g_full)
                dhat_full.append(d_full)
            if dmap.ndim == 2:  # forward crop needs a 2-D map
                g_fwd = _forward_min_depth(depth_gt, center_frac=args.center_frac)
                d_fwd = _forward_min_depth(dmap, center_frac=args.center_frac)
                if np.isfinite(g_fwd) and np.isfinite(d_fwd):
                    gt_fwd.append(g_fwd)
                    dhat_fwd.append(d_fwd)
            n_frames_total += 1

    gt_full_a = np.asarray(gt_full)
    dhat_full_a = np.asarray(dhat_full)
    gt_fwd_a = np.asarray(gt_fwd)
    dhat_fwd_a = np.asarray(dhat_fwd)

    full_rows = _summarize(gt_full_a, dhat_full_a, trigger_m=args.trigger_m)
    fwd_rows = _summarize(gt_fwd_a, dhat_fwd_a, trigger_m=args.trigger_m) if gt_fwd_a.size else []

    # Headline late-trigger numbers on the SHIELD basis (full-field), which is what
    # the shield actually reads and what ④a masks on.
    def _recall(gt_a: np.ndarray, dhat_a: np.ndarray, lo: float, hi: float) -> Dict[str, Any]:
        m = (gt_a >= lo) & (gt_a < hi)
        n = int(m.sum())
        return {"n": n, "p_trigger": (round(float(np.mean(dhat_a[m] < args.trigger_m)), 3) if n else None)}

    headline = {
        # In the near band (GT<1.5): shield should already be latched → want ~1.0.
        "near_band_GT_lt_1.5": _recall(gt_full_a, dhat_full_a, 0.0, args.near_m),
        # Reaction window [1.5,3.0): shield must fire HERE to brake before the band.
        "reaction_window_GT_1.5_3.0": _recall(gt_full_a, dhat_full_a, args.near_m, args.trigger_m),
    }

    print(f"\n[diag] episodes={len(episodes)} frames_scored={n_frames_total} "
          f"no_depth_frames={n_no_depth} n_frames_window={n_frames}")
    print(f"[diag] full-field pairs={gt_full_a.size} forward-crop pairs={gt_fwd_a.size}")
    _print_table("D̂ vs GT — FULL-FIELD min (shield basis + ④a mask basis)",
                 full_rows, trigger_m=args.trigger_m)
    if fwd_rows:
        _print_table("D̂ vs GT — FORWARD central-crop min (obstacle-ahead view)",
                     fwd_rows, trigger_m=args.trigger_m)
    print("\n[diag] HEADLINE (full-field, shield basis):")
    print(f"  near band  GT<{args.near_m:g}      : {json.dumps(headline['near_band_GT_lt_1.5'])}")
    print(f"  reaction   GT[{args.near_m:g},{args.trigger_m:g}) : "
          f"{json.dumps(headline['reaction_window_GT_1.5_3.0'])}  "
          f"(low P(trig) here == late trigger == the ④ failure)")

    if args.emit:
        payload = {
            "dataset": str(args.dataset),
            "depth_ckpt": str(args.depth_ckpt),
            "trigger_m": args.trigger_m,
            "near_m": args.near_m,
            "center_frac": args.center_frac,
            "episodes": len(episodes),
            "frames_scored": n_frames_total,
            "full_field": full_rows,
            "forward_crop": fwd_rows,
            "headline": headline,
        }
        Path(args.emit).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit).write_text(json.dumps(payload, indent=2))
        print(f"\n[diag] wrote {args.emit}")


if __name__ == "__main__":  # pragma: no cover
    main()
