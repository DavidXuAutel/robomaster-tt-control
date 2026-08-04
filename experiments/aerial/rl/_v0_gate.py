"""V0 four-signal gate entrypoint (frozen §4 / §4.1).

Mac-testable scorers live in :mod:`v0_metrics` / :mod:`vio`. This module is the
H100-facing CLI skeleton: it refuses desynced V0 RGB-only corpora, loads a
schema-v2 dataset when present, and evaluates whichever signals have enough
inputs. Signals that still need a depth head / live shield eval report
``ok=False`` with an explicit reason (V0 requires all four — no silent skip).

    # metric self-check (no GPU):
    python -m experiments.aerial.rl._v0_gate --self-check

    # later, on H100 with depth-capable corpus + predictions:
    python -m experiments.aerial.rl._v0_gate \\
        --dataset .../artifacts/dataset_v0_depth \\
        --learning-log .../wm_train.jsonl

Does **not** flip ``configs/aerial_rl.yaml`` gates — that happens only after
this process exits 0 (human / follow-up commit).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl import v0_metrics as metrics


def _refuse_rgb_only_desync(root: Path, allow: bool) -> None:
    """Refuse over-commanded / incomplete corpora (aligned with ``_refuse_v0``)."""
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    meta = manifest.get("meta") or {}
    step_hz = float(meta.get("step_hz", 0) or 0)
    grab_depth = bool(meta.get("grab_depth", False))
    if not allow:
        if grab_depth and step_hz > 6.5:
            print(
                f"[v0-gate] REFUSE: step_hz={step_hz} with grab_depth exceeds the "
                "measured 4090-local depth closed-loop ceiling (~6.2 Hz).",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if step_hz > 8.5:
            print(
                f"[v0-gate] REFUSE: step_hz={step_hz} looks like the dt-desynced "
                "cross-net RGB smoke corpus — not a V0 training set.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if not grab_depth:
            print(
                "[v0-gate] REFUSE: manifest grab_depth=false — V0 four-signal gate "
                "needs schema-v2 depth/IMU corpus from 4090-local collect. "
                "Pass --allow-incomplete only to exercise scorers on synthetic inputs.",
                file=sys.stderr,
            )
            raise SystemExit(2)


def _signal1_from_log(path: Optional[Path], thr: metrics.V0GateThresholds) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"ok": False, "reason": "missing --learning-log (① needs train curves)"}
    losses: List[float] = []
    recons: List[float] = []
    ents: List[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        losses.append(float(row["loss"]))
        recons.append(float(row.get("recon_err", row.get("recon", 0.0))))
        ents.append(float(row.get("post_entropy_frac", row.get("ent", 1.0))))
    return metrics.check_learning_curves(losses, recons, ents, thr=thr)


def _signal3_from_dataset(root: Path, thr: metrics.V0GateThresholds) -> Dict[str, Any]:
    """Use GT depth + GT vel as a scale-consistency smoke when no depth head yet.

    When a depth head exists, callers should pass predicted depth via
    ``--depth-pred-npz``. Using GT depth here only validates the *metric
    plumbing* (motion windows, median Abs scale); for a real V0 pass the depth
    channel must be model predictions.
    """
    episodes = ds.load_dataset(root, skip_quarantined=True)
    if not episodes:
        return {"ok": False, "reason": "no episodes"}
    # Build a single batch of windows from long-enough episodes.
    windows = []
    for ep in episodes:
        if len(ep) < 8:
            continue
        windows.append(ep[:8])
        if len(windows) >= 16:
            break
    if not windows:
        return {"ok": False, "reason": "no episode with >=8 steps"}
    from experiments.aerial.rl.perception_data import windows_to_perception_arrays

    arr = windows_to_perception_arrays(windows)
    if "depth" not in arr:
        return {"ok": False, "reason": "dataset has no depth channel"}
    return metrics.check_scale_consistency(
        arr["vel"], arr["timestamps"], arr["depth"], thr=thr
    )


def _self_check(thr: metrics.V0GateThresholds) -> int:
    """Synthetic fixtures — locks §4.1 numbers without AirSim."""
    # ① pass
    losses = [10.0] * 50 + [5.0] * 50
    recons = [0.2] * 50 + [0.1] * 50
    ents = [0.5] * 100
    s1 = metrics.check_learning_curves(losses, recons, ents, thr=thr)
    assert s1["ok"], s1

    # ② pass via progress margin
    s2 = metrics.check_progress_vs_random(
        [20.0] * 16, [10.0] * 16, [30.0] * 16, [30.0] * 16, thr=thr
    )
    assert s2["ok"] and s2["progress_ok"], s2

    # ③ pass: depth change matches motion
    B, L, H, W = 4, 8, 4, 4
    ts = np.linspace(0.0, 1.0, L, dtype=np.float32)
    timestamps = np.stack([ts] * B, axis=0)
    # constant +x velocity 1 m/s → ~1 m motion over 1 s
    vel = np.zeros((B, L, 3), dtype=np.float32)
    vel[..., 0] = 1.0
    # depth decreases by ~1 m across the window
    depth = np.ones((B, L, H, W), dtype=np.float32) * 5.0
    for t in range(L):
        depth[:, t] = 5.0 - ts[t]
    s3 = metrics.check_scale_consistency(vel, timestamps, depth, thr=thr)
    assert s3["ok"], s3

    # ④ pass
    interventions = [[False, True, False, False]]
    collided = [[False, False, False, True]]
    near_on = [[False, False, False, False]]
    near_off = [[True, True, False, False]]
    s4 = metrics.check_shield_effectiveness(
        interventions_on=interventions,
        collided_on=collided,
        near_coll_on=near_on,
        near_coll_off=near_off,
        thr=thr,
    )
    assert s4["ok"], s4

    verdict = metrics.aggregate_v0_verdict({"1": s1, "2": s2, "3": s3, "4": s4})
    assert verdict["ok"], verdict
    print("[v0-gate] self-check PASS")
    from dataclasses import asdict
    print(json.dumps({"thresholds": asdict(thr), "verdict": verdict["passed"]}, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None, help="schema-v2 episode dir")
    p.add_argument("--learning-log", default=None, help="jsonl with loss/recon/ent")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args(argv)
    thr = metrics.DEFAULT_THRESHOLDS

    if args.self_check:
        return _self_check(thr)

    if not args.dataset:
        print("[v0-gate] FAIL: --dataset required (or --self-check)", file=sys.stderr)
        return 2

    root = Path(args.dataset)
    _refuse_rgb_only_desync(root, args.allow_incomplete)

    results: Dict[str, Dict[str, Any]] = {}
    results["1"] = _signal1_from_log(
        Path(args.learning_log) if args.learning_log else None, thr
    )
    # ② / ④ need live rollouts — not available offline without a runner hook.
    results["2"] = {
        "ok": False,
        "reason": "live progress-vs-random eval not wired yet "
                  "(needs mock/airsim runner; thresholds locked in v0_metrics)",
    }
    results["3"] = _signal3_from_dataset(root, thr)
    results["4"] = {
        "ok": False,
        "reason": "shield on/off eval not wired yet "
                  "(CLI-only; do not flip default yaml)",
    }

    verdict = metrics.aggregate_v0_verdict(results)
    print(json.dumps(verdict, indent=2, default=str))
    print(f"[v0-gate] {'PASS' if verdict['ok'] else 'FAIL'}")
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
