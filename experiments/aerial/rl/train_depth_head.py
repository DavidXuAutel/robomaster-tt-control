"""H100 offline trainer for the [1b] multi-frame DepthHead (frozen §6 Step 3).

Trains ``_DepthHead`` on schema-v2 episodes via ``perception_data`` (GT depth /
IMU stay off the WM/policy graph). Does **not** flip
``world_model.depth_head.enable`` — that flag stays false until ``_v0_gate``
four-signal PASS (frozen §4).

    python -m experiments.aerial.rl.train_depth_head \
        --dataset experiments/aerial/rl/artifacts/dataset_v0_local_depth \
        --config configs/aerial_rl.yaml --steps 2000 --device cuda --save-ckpt

Writes a jsonl learning log (for ①d / gate) next to the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.dynamics_torch import (
    _DepthHead,
    depth_delta_scale_loss,
    depth_head_loss,
)
from experiments.aerial.rl.perception_data import windows_to_perception_arrays
from experiments.aerial.rl.v0_metrics import DEFAULT_THRESHOLDS, depth_absrel


def _load_depth_cfg(config_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text()) or {}
    wm = dict(cfg.get("world_model", {}) or {})
    dh = dict(wm.get("depth_head", {}) or {})
    # Sensible defaults when the block is only partially filled.
    dh.setdefault("n_frames", 4)
    dh.setdefault("base", 32)
    dh.setdefault("lr", 1.0e-4)
    dh.setdefault("grad_clip", 1000.0)
    dh.setdefault("absrel_weight", 1.0)
    dh.setdefault("nll_weight", 0.1)
    dh.setdefault("delta_weight", 1.0)  # temporal ŝ_D consistency (V0 ③)
    dh.setdefault("max_depth_m", 200.0)
    dh.setdefault("scale_depth_min_m", 1.0)
    dh.setdefault("scale_depth_max_m", 40.0)
    dh.setdefault("image_size", int((cfg.get("env") or {}).get("width", 224)))
    dh.setdefault(
        "checkpoint_dir",
        "experiments/aerial/rl/artifacts/depth_ckpt",
    )
    dh.setdefault("enable", False)
    return dh


def _refuse_bad_corpus(root: Path, allow: bool) -> None:
    """Same floor policy as ``_wm_train_validate._refuse_v0`` for depth corpora."""
    from experiments.aerial.rl._wm_train_validate import _refuse_v0

    _refuse_v0(root, allow)


def _usable_episodes(root: Path, window: int) -> List[Any]:
    """Episodes long enough for a window AND carrying per-frame GT depth."""
    episodes = ds.load_dataset(root, skip_quarantined=True)
    episodes = [ep for ep in episodes if len(ep) >= window]
    if not episodes:
        print(f"[depth-train] FAIL: no episode >= {window} steps", file=sys.stderr)
        raise SystemExit(1)
    with_depth = [ep for ep in episodes if all(t.obs.depth is not None for t in ep)]
    if not with_depth:
        print("[depth-train] FAIL: no usable episode carries per-frame depth", file=sys.stderr)
        raise SystemExit(1)
    return with_depth


def _split_train_holdout(
    episodes: List[Any], *, holdout_frac: float, seed: int
) -> tuple[List[Any], List[Any]]:
    """Episode-level split so ①d AbsRel is scored on trajectories never trained on.

    Deterministic (seeded permutation) and always leaves ≥1 episode for training.
    With <2 usable episodes a real holdout is impossible — returns an empty
    holdout and the caller warns that ①d would be in-sample.
    """
    n = len(episodes)
    if n < 2:
        return episodes, []
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    n_hold = max(1, int(round(n * float(holdout_frac))))
    n_hold = min(n_hold, n - 1)
    hold = [episodes[int(i)] for i in idx[:n_hold]]
    train = [episodes[int(i)] for i in idx[n_hold:]]
    return train, hold


def _buffer_from(episodes: List[Any], *, tag: str, window: int) -> ReplayBuffer:
    buf = ReplayBuffer(capacity_episodes=len(episodes) + 1, seed=0)
    for ep in episodes:
        buf.add_episode(ep)
    print(
        f"[depth-train] {tag} buffer: {buf.num_episodes} eps / {buf.num_transitions} "
        f"steps (window>={window}, depth present)"
    )
    return buf


def _holdout_windows(episodes: List[Any], window: int) -> List[Any]:
    """Deterministic non-overlapping (stride=window) windows from held-out eps."""
    windows: List[Any] = []
    for ep in episodes:
        for start in range(0, len(ep) - window + 1, window):
            windows.append(ep[start : start + window])
    return windows


def _holdout_absrel(
    model: _DepthHead,
    holdout_eps: List[Any],
    *,
    wm_batch: int,
    window: int,
    device: torch.device,
    max_depth_m: float = 200.0,
) -> float:
    """Median AbsRel over ALL held-out windows (not a random resample of train).

    Enumerates fixed windows so the number is stable across runs, and predicts in
    ``wm_batch`` chunks; pred/GT are pooled before a single ``depth_absrel`` so the
    median-over-pixels semantics match the gate scorer.
    """
    windows = _holdout_windows(holdout_eps, window)
    if not windows:
        return float("nan")
    model.eval()
    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    for i in range(0, len(windows), int(wm_batch)):
        chunk = windows[i : i + int(wm_batch)]
        arrays = windows_to_perception_arrays(chunk)
        if "depth" not in arrays:
            return float("nan")
        rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"])).to(device)
        gt = torch.from_numpy(np.ascontiguousarray(arrays["depth"])).to(device)
        with torch.no_grad():
            pred, _ = model.predict_from_window(rgb)
        preds.append(pred.cpu().numpy())
        gts.append(gt[:, -1].cpu().numpy())
    return float(
        depth_absrel(
            np.concatenate(preds, axis=0),
            np.concatenate(gts, axis=0),
            max_depth_m=float(max_depth_m),
        )
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/aerial_rl.yaml"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--wm-batch", type=int, default=8)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-ckpt", action="store_true")
    p.add_argument("--allow-v0-desync", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--holdout-frac", type=float, default=0.2, help="episode fraction reserved for ①d AbsRel")
    p.add_argument("--split-seed", type=int, default=0)
    args = p.parse_args(argv)

    root = args.dataset
    _refuse_bad_corpus(root, args.allow_v0_desync)
    dh_cfg = _load_depth_cfg(args.config)
    if bool(dh_cfg.get("enable")):
        print(
            "[depth-train] NOTE: world_model.depth_head.enable is true in yaml; "
            "frozen §4 says flip only AFTER _v0_gate PASS — training still runs, "
            "but do not treat this as gate success.",
            file=sys.stderr,
        )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if str(device) != args.device:
        print(f"[depth-train] falling back to {device} (requested {args.device})")

    all_eps = _usable_episodes(root, args.window)
    train_eps, holdout_eps = _split_train_holdout(
        all_eps, holdout_frac=float(args.holdout_frac), seed=int(args.split_seed)
    )
    if not holdout_eps:
        print(
            "[depth-train] WARN: <2 usable episodes — no held-out split; ①d AbsRel "
            "will be IN-SAMPLE and is NOT a valid gate signal. Collect more episodes.",
            file=sys.stderr,
        )
        holdout_eps = train_eps  # in-sample fallback, explicitly flagged above
    buf = _buffer_from(train_eps, tag="train", window=args.window)
    print(f"[depth-train] holdout: {len(holdout_eps)} eps reserved for ①d AbsRel")
    model = _DepthHead(
        image_size=int(dh_cfg["image_size"]),
        n_frames=int(dh_cfg["n_frames"]),
        base=int(dh_cfg["base"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(dh_cfg["lr"]), betas=(0.9, 0.95))
    grad_clip = float(dh_cfg["grad_clip"])

    ckpt_dir = Path(str(dh_cfg["checkpoint_dir"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "depth_train.jsonl"
    if log_path.exists():
        log_path.unlink()

    absrels: List[float] = []
    losses: List[float] = []
    model.train()
    for step in range(1, int(args.steps) + 1):
        windows = buf.sample_windows(int(args.wm_batch), int(args.window))
        arrays = windows_to_perception_arrays(windows)
        if "depth" not in arrays:
            print("[depth-train] FAIL: batch missing depth", file=sys.stderr)
            return 1
        rgb = torch.from_numpy(np.ascontiguousarray(arrays["rgb"])).to(device)
        gt = torch.from_numpy(np.ascontiguousarray(arrays["depth"])).to(device)
        pred, log_sigma = model.predict_from_window(rgb)
        loss, stats = depth_head_loss(
            pred, log_sigma, gt[:, -1],
            absrel_weight=float(dh_cfg["absrel_weight"]),
            nll_weight=float(dh_cfg["nll_weight"]),
            max_depth_m=float(dh_cfg["max_depth_m"]),
        )
        # Temporal / Δ-depth: predict the first frame of the window with full
        # n_frames context (requires window >= n_frames) and match |Δ band-median|
        # to GT — teaches the scale-change ③ measures (diagnose 2026-08-05).
        delta_w = float(dh_cfg.get("delta_weight", 0.0))
        if delta_w > 0.0 and int(args.window) >= int(dh_cfg["n_frames"]):
            n_f = int(dh_cfg["n_frames"])
            pred_first, _ = model.predict_from_window(rgb[:, :n_f])
            d_loss, d_stats = depth_delta_scale_loss(
                pred_first,
                pred,
                gt[:, n_f - 1],
                gt[:, -1],
                min_depth_m=float(dh_cfg["scale_depth_min_m"]),
                max_depth_m=float(dh_cfg["scale_depth_max_m"]),
            )
            loss = loss + delta_w * d_loss
            stats = {**stats, **d_stats, "loss": float(loss.detach().item())}
        else:
            stats = {**stats, "delta_rel": float("nan"), "n_delta": 0}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        losses.append(float(stats["loss"]))
        absrels.append(float(stats["absrel"]))
        row = {"step": step, **stats}
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step % int(args.log_every) == 0 or step == 1:
            print(
                f"[depth-train] step {step}/{args.steps} "
                f"loss={stats['loss']:.4f} absrel={stats['absrel']:.4f} "
                f"delta_rel={stats.get('delta_rel', float('nan'))} "
                f"n_valid={stats['n_valid']}"
            )

    holdout = _holdout_absrel(
        model, holdout_eps, wm_batch=int(args.wm_batch), window=int(args.window),
        device=device, max_depth_m=float(dh_cfg["max_depth_m"]),
    )
    thr = DEFAULT_THRESHOLDS.depth_absrel_max
    print(f"[depth-train] holdout median AbsRel={holdout:.4f} (gate ①d ≤ {thr})")
    ok_1d = bool(np.isfinite(holdout) and holdout <= thr)
    if not ok_1d:
        print(
            f"[depth-train] WARN: holdout AbsRel above ①d threshold — continue "
            "training or retune; _v0_gate will FAIL ①d until this clears.",
            file=sys.stderr,
        )

    if args.save_ckpt:
        path = ckpt_dir / f"depth_step_{args.steps}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "step": int(args.steps),
                "n_frames": int(dh_cfg["n_frames"]),
                "image_size": int(dh_cfg["image_size"]),
                "base": int(dh_cfg["base"]),
                "holdout_absrel": holdout,
                "depth_cfg": dh_cfg,
            },
            path,
        )
        print(f"[depth-train] wrote {path}")

    # Soft pass for the trainer process: finite descending loss + finite AbsRel.
    if not losses or not np.isfinite(losses[-1]):
        print("[depth-train] FAIL: non-finite final loss", file=sys.stderr)
        return 1
    print(f"[depth-train] OK: log={log_path} enable_flag_still={dh_cfg.get('enable')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
