"""V0 four-signal gate entrypoint (frozen §4 / §4.1).

Mac-testable scorers live in :mod:`v0_metrics` / :mod:`vio`; the paired rollout
runners in :mod:`v0_rollout_eval`. This module is the H100-facing CLI: it refuses
desynced / RGB-only corpora, and assembles the four signals.

**Depth pillar is enforced** (the failure mode that invalidated the old single-
pillar checkpoint): signal ① = ①a–c *and* ①d (holdout depth AbsRel), and ③
consumes the depth head's **predicted** D̂ — not GT depth. Without ``--depth-ckpt``
both ①d and ③ FAIL, so the whole gate FAILs. GT depth alone can never pass V0.

    # metric self-check (no GPU):
    python -m experiments.aerial.rl._v0_gate --self-check

    # on H100 with a depth-capable corpus + a trained depth head:
    python -m experiments.aerial.rl._v0_gate \\
        --dataset .../artifacts/dataset_v0_local_depth \\
        --learning-log .../wm_ckpt/wm_train.jsonl \\
        --depth-ckpt .../depth_ckpt/depth_step_2000.pt \\
        --rollout-eval --device cuda

Does **not** flip ``configs/aerial_rl.yaml`` gates — that happens only after this
process exits 0 (human / follow-up commit).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# --------------------------------------------------------------------------- #
# Signal assembly (pure — Mac-unit-testable by injecting the per-signal dicts) #
# --------------------------------------------------------------------------- #
def assemble_verdict(
    *,
    s1abc: Dict[str, Any],
    s1d: Dict[str, Any],
    s2: Dict[str, Any],
    s3: Dict[str, Any],
    s4: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine the four signals with ① = ①a–c ∧ ①d (depth pillar folded in)."""
    sig1 = {
        "ok": bool(s1abc.get("ok") and s1d.get("ok")),
        "abc": s1abc,
        "d": s1d,
    }
    return metrics.aggregate_v0_verdict({"1": sig1, "2": s2, "3": s3, "4": s4})


def _signal1abc_from_log(
    path: Optional[Path], thr: metrics.V0GateThresholds
) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"ok": False, "reason": "missing --learning-log (①a–c needs train curves)"}
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


def _signal1d(
    pred_depth: Optional[np.ndarray],
    gt_depth: Optional[np.ndarray],
    thr: metrics.V0GateThresholds,
) -> Dict[str, Any]:
    """①d — holdout depth AbsRel. FAIL (not SKIP) when no depth head / GT.

    Frozen §4.1: "否则 ①d=SKIP（整门 FAIL——V0 需要深度柱）". We surface that as an
    explicit FAIL so the aggregate can never pass without the depth pillar.
    """
    if pred_depth is None:
        return {"ok": False, "reason": "no depth head predictions (pass --depth-ckpt); V0 needs the depth pillar"}
    if gt_depth is None:
        return {"ok": False, "reason": "dataset carries no GT depth to score ①d against"}
    return metrics.check_depth_absrel(pred_depth, gt_depth, thr=thr)


def _signal3(
    pred_depth: Optional[np.ndarray],
    vel: Optional[np.ndarray],
    timestamps: Optional[np.ndarray],
    thr: metrics.V0GateThresholds,
) -> Dict[str, Any]:
    """③ — D̂-scale vs VIO. Must use PREDICTED depth; GT-only cannot pass V0."""
    if pred_depth is None:
        return {"ok": False, "reason": "③ needs predicted D̂ (pass --depth-ckpt); GT depth is plumbing-only"}
    if vel is None or timestamps is None:
        return {"ok": False, "reason": "dataset missing vel/timestamps for VIO scale"}
    return metrics.check_scale_consistency(vel, timestamps, pred_depth, thr=thr)


# --------------------------------------------------------------------------- #
# Depth-head inference over dataset windows (lazy torch — H100 only)          #
# --------------------------------------------------------------------------- #
def _sample_windows(root: Path, *, window: int, max_windows: int) -> List[Any]:
    episodes = ds.load_dataset(root, skip_quarantined=True)
    windows: List[Any] = []
    for ep in episodes:
        if len(ep) < window:
            continue
        windows.append(ep[:window])
        if len(windows) >= max_windows:
            break
    return windows


def _predict_depth_over_windows(
    ckpt_path: Path,
    windows: List[Any],
    *,
    device: str = "cpu",
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Run the trained ``_DepthHead`` on each frame of each window (lazy torch).

    Returns ``(perception_arrays, pred_depth[B, L, H, W])``. Predicting per frame
    (sliding the head's n_frames window to end at ``t``) gives a full-window depth
    stack so ③ can measure the *depth change* across the window from D̂, not GT.
    """
    import torch  # lazy: H100 only

    from experiments.aerial.rl.dynamics_torch import _DepthHead
    from experiments.aerial.rl.perception_data import windows_to_perception_arrays

    payload = torch.load(str(ckpt_path), map_location="cpu")
    model = _DepthHead(
        image_size=int(payload.get("image_size", 224)),
        n_frames=int(payload.get("n_frames", 4)),
        base=int(payload.get("base", 32)),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval().to(device)

    arr = windows_to_perception_arrays(windows)
    rgb = torch.from_numpy(np.ascontiguousarray(arr["rgb"]))  # [B, L, H, W, 3]
    B, L = int(rgb.shape[0]), int(rgb.shape[1])
    n = model.n_frames
    preds = np.empty((B, L, model.image_size, model.image_size), dtype=np.float32)
    with torch.no_grad():
        for t in range(L):
            lo = max(0, t - n + 1)
            sub = rgb[:, lo : t + 1]  # pack_rgb_nhwc left-pads if shorter than n
            d, _ = model.predict_from_window(sub.to(device))
            preds[:, t] = d.cpu().numpy()
    return arr, preds


# --------------------------------------------------------------------------- #
# ② / ④ via paired rollouts (mock env by default; airsim for a real ④ pass)   #
# --------------------------------------------------------------------------- #
def _signals_2_4_from_rollouts(
    config_path: Path,
    thr: metrics.V0GateThresholds,
    *,
    depth_ckpt: Optional[Path],
    device: str,
    n_episodes: int,
    max_steps: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import yaml

    from experiments.aerial.rl import v0_rollout_eval as rollout
    from experiments.aerial.rl.reward import RewardConfig
    from experiments.aerial.rl.train_rl import HeuristicPolicy, _build_env

    cfg = yaml.safe_load(config_path.read_text()) or {}
    env = _build_env(cfg.get("env", {}) or {})
    reward_cfg = RewardConfig(**(cfg.get("reward", {}) or {})) if cfg.get("reward") else None
    starts = rollout.make_start_episodes(int(n_episodes), seed=int(seed))

    # ② progress-vs-random (goal-seeking HeuristicPolicy is the V0 baseline policy).
    policy = HeuristicPolicy(goal_getter=lambda: getattr(env, "goal", None))
    rnd = rollout.RandomActionPolicy(seed=int(seed))
    prog = rollout.run_progress_eval(
        env, policy, rnd, starts, max_steps=int(max_steps), reward_cfg=reward_cfg
    )
    s2 = metrics.check_progress_vs_random(
        prog["policy_progress_sums"], prog["random_progress_sums"],
        prog["policy_final_dists"], prog["random_final_dists"], thr=thr,
    )

    # ④ shield on/off — requires the real depth head to fill depth_min_pred.
    if depth_ckpt is None:
        s4 = {"ok": False, "reason": "④ needs --depth-ckpt (shield reads predicted D̂); depth pillar enforced"}
        return s2, s4
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor

    predictor = DepthMinPredictor.from_checkpoint(depth_ckpt, device=device)
    masks = rollout.run_shield_eval(
        env, policy, predictor, starts,
        near_collision_depth_m=thr.near_collision_depth_m,
        max_steps=int(max_steps), reward_cfg=reward_cfg,
    )
    s4 = metrics.check_shield_effectiveness(
        interventions_on=masks["interventions_on"],
        collided_on=masks["collided_on"],
        near_coll_on=masks["near_coll_on"],
        near_coll_off=masks["near_coll_off"],
        thr=thr,
    )
    return s2, s4


def _self_check(thr: metrics.V0GateThresholds) -> int:
    """Synthetic fixtures — locks §4.1 numbers + the depth-pillar aggregation."""
    s1abc = metrics.check_learning_curves(
        [10.0] * 50 + [5.0] * 50, [0.2] * 50 + [0.1] * 50, [0.5] * 100, thr=thr
    )
    assert s1abc["ok"], s1abc
    # ①d with predicted depth close to GT.
    gt = np.ones((4, 4, 4), dtype=np.float32) * 10.0
    pred = gt * 1.1
    s1d = _signal1d(pred, gt, thr)
    assert s1d["ok"], s1d
    s2 = metrics.check_progress_vs_random(
        [20.0] * 16, [10.0] * 16, [30.0] * 16, [30.0] * 16, thr=thr
    )
    assert s2["ok"], s2
    # ③ predicted depth-change matches motion.
    B, L, H, W = 4, 8, 4, 4
    ts = np.linspace(0.0, 1.0, L, dtype=np.float32)
    timestamps = np.stack([ts] * B, axis=0)
    vel = np.zeros((B, L, 3), dtype=np.float32)
    vel[..., 0] = 1.0
    depth = np.ones((B, L, H, W), dtype=np.float32) * 5.0
    for t in range(L):
        depth[:, t] = 5.0 - ts[t]
    s3 = _signal3(depth, vel, timestamps, thr)
    assert s3["ok"], s3
    s4 = metrics.check_shield_effectiveness(
        interventions_on=[[False, True, False, False]],
        collided_on=[[False, False, False, True]],
        near_coll_on=[[False, False, False, False]],
        near_coll_off=[[True, True, False, False]],
        thr=thr,
    )
    assert s4["ok"], s4

    verdict = assemble_verdict(s1abc=s1abc, s1d=s1d, s2=s2, s3=s3, s4=s4)
    assert verdict["ok"], verdict
    # Depth pillar removed → whole gate must fail even with ①a–c/②/③/④ green.
    no_depth = assemble_verdict(
        s1abc=s1abc, s1d=_signal1d(None, None, thr), s2=s2, s3=s3, s4=s4
    )
    assert not no_depth["ok"], no_depth
    print("[v0-gate] self-check PASS (incl. depth-pillar enforcement)")
    print(json.dumps({"thresholds": asdict(thr), "verdict": verdict["passed"]}, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None, help="schema-v2 episode dir")
    p.add_argument("--learning-log", default=None, help="jsonl with loss/recon/ent")
    p.add_argument("--depth-ckpt", default=None, help="trained _DepthHead .pt (①d/③/④ depth pillar)")
    p.add_argument("--config", default="configs/aerial_rl.yaml", help="for ②/④ rollout env")
    p.add_argument("--rollout-eval", action="store_true", help="run ②/④ paired rollouts")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-eval-episodes", type=int, default=None, help="override N (default §4.1 = 16)")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=16)
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

    # Depth-head predictions (①d + ③). Absent → depth pillar FAILs both.
    pred_depth: Optional[np.ndarray] = None
    gt_depth: Optional[np.ndarray] = None
    vel: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    if args.depth_ckpt:
        windows = _sample_windows(root, window=int(args.window), max_windows=int(args.max_windows))
        if not windows:
            print(f"[v0-gate] FAIL: no episode >= {args.window} steps for depth eval", file=sys.stderr)
            return 1
        arr, pred_full = _predict_depth_over_windows(
            Path(args.depth_ckpt), windows, device=args.device
        )
        pred_depth = pred_full
        gt_depth = arr.get("depth")
        vel = arr.get("vel")
        timestamps = arr.get("timestamps")

    thr_eff = thr
    if args.n_eval_episodes is not None:
        from dataclasses import replace

        thr_eff = replace(thr, n_eval_episodes=int(args.n_eval_episodes))

    results: Dict[str, Dict[str, Any]] = {}
    s1abc = _signal1abc_from_log(
        Path(args.learning_log) if args.learning_log else None, thr
    )
    # ①d scores the head's prediction at the last frame of each window vs GT.
    s1d = _signal1d(
        None if pred_depth is None else pred_depth[:, -1],
        None if gt_depth is None else gt_depth[:, -1],
        thr,
    )
    s3 = _signal3(pred_depth, vel, timestamps, thr)

    if args.rollout_eval:
        s2, s4 = _signals_2_4_from_rollouts(
            Path(args.config), thr_eff,
            depth_ckpt=Path(args.depth_ckpt) if args.depth_ckpt else None,
            device=args.device,
            n_episodes=thr_eff.n_eval_episodes,
            max_steps=int(args.max_steps),
            seed=int(args.seed),
        )
    else:
        s2 = {"ok": False, "reason": "pass --rollout-eval to run ② progress-vs-random"}
        s4 = {"ok": False, "reason": "pass --rollout-eval (+ --depth-ckpt) to run ④ shield on/off"}

    verdict = assemble_verdict(s1abc=s1abc, s1d=s1d, s2=s2, s3=s3, s4=s4)
    results = verdict["details"]  # for the printed dump
    print(json.dumps(verdict, indent=2, default=str))
    print(f"[v0-gate] {'PASS' if verdict['ok'] else 'FAIL'}")
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
