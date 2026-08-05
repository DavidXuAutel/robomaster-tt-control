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
    # Keep delta << AbsRel/SILog: delta_weight=1.0 from-scratch collapsed AbsRel
    # (0.98 / 0.70 archived 2026-08-05). Prefer finetune from canonical PASS ckpt.
    dh.setdefault("delta_weight", 0.1)
    dh.setdefault("delta_min_gt_m", 0.5)  # approach gate: ŝ_gt ≥ this
    dh.setdefault("delta_support_ratio", 0.6)  # match §4.1 scale_support_ratio
    dh.setdefault("approach_oversample", 4)  # candidate pool / batch for Δ bias
    dh.setdefault("max_depth_m", 200.0)
    dh.setdefault("scale_depth_min_m", 1.0)
    dh.setdefault("scale_depth_max_m", 40.0)
    # Decoder-only / freeze-encoder Δ-finetune: keep AbsRel-good encoder features
    # fixed; train decoder (depth head) only so scale can move without AbsRel
    # regression. CLI --freeze-encoder overrides; yaml default stays false.
    dh.setdefault("freeze_encoder", False)
    dh.setdefault("image_size", int((cfg.get("env") or {}).get("width", 224)))
    dh.setdefault(
        "checkpoint_dir",
        "experiments/aerial/rl/artifacts/depth_ckpt",
    )
    dh.setdefault("enable", False)
    return dh


def _apply_freeze_encoder(model: _DepthHead, freeze: bool) -> list:
    """Freeze ``model.encoder`` and return the AdamW param list.

    When ``freeze`` is True, encoder ``requires_grad=False`` and only
    ``model.decoder`` params are returned (structurally blocks encoder AbsRel
    drift). When False, all parameters remain trainable and are returned.
    """
    if not freeze:
        for p in model.parameters():
            p.requires_grad = True
        return list(model.parameters())
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = True
    return [p for p in model.decoder.parameters() if p.requires_grad]


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


def _band_mean_np(
    depth: np.ndarray, *, min_depth_m: float, max_depth_m: float
) -> np.ndarray:
    """Numpy sibling of ``_band_spatial_mean`` for approach scoring (no torch)."""
    flat = np.asarray(depth, dtype=np.float64).reshape(depth.shape[0], -1)
    valid = np.isfinite(flat) & (flat >= float(min_depth_m)) & (flat <= float(max_depth_m))
    masked = np.where(valid, flat, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(masked, axis=-1).astype(np.float32)


def _sample_approach_biased_windows(
    buf: ReplayBuffer,
    batch: int,
    window: int,
    *,
    oversample: int,
    min_depth_m: float,
    max_depth_m: float,
    min_gt_delta_m: float,
    support_ratio: float,
    n_frames: int = 1,
) -> List[Any]:
    """Prefer windows with alive GT ŝ_D (approach geometry for Δ-depth).

    Draws ``batch * oversample`` candidates, ranks by GT |Δ band-mean|, and
    keeps the top ``batch`` that pass the approach gate when possible. Falls
    back to uniform sampling when the pool has no approach-alive windows so
    AbsRel training never stalls.

    Scoring uses the same depth endpoints as ``depth_delta_scale_loss`` in the
    train loop: ``depth[:, n_frames-1]`` vs ``depth[:, -1]`` (not ``[:, 0]``).
    """
    batch = int(batch)
    oversample = max(1, int(oversample))
    n_cand = max(batch, batch * oversample)
    candidates = buf.sample_windows(n_cand, int(window))
    if oversample <= 1 or n_cand == batch:
        return candidates[:batch]
    arrays = windows_to_perception_arrays(candidates)
    # Approach-bias needs both GT depth (to score |Δ|) and position (to gate on
    # motion). If either is absent, fall back to a uniform sample so AbsRel still
    # trains rather than KeyError-ing on the missing field.
    if "depth" not in arrays or "position" not in arrays:
        return candidates[:batch]
    depth = arrays["depth"]
    # Align with Δ-loss: pred_first = predict(rgb[:, :n_f]) → GT [:, n_f-1] vs [:, -1].
    n_f = max(1, min(int(n_frames), int(window)))
    g0 = _band_mean_np(depth[:, n_f - 1], min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    g1 = _band_mean_np(depth[:, -1], min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    s_gt = np.abs(g1.astype(np.float64) - g0.astype(np.float64))
    pos = arrays["position"]
    # Match the Δ-loss motion interval [n_f-1, -1] (not the full window) so the
    # sampler's support gate agrees with depth_delta_scale_loss's.
    motion = np.linalg.norm(pos[:, -1] - pos[:, n_f - 1], axis=-1).astype(np.float64)
    alive = np.isfinite(s_gt) & (s_gt >= float(min_gt_delta_m))
    if float(support_ratio) > 0.0:
        alive &= np.isfinite(motion) & (s_gt >= float(support_ratio) * motion)
    scores = np.where(alive, s_gt, -1.0)
    order = np.argsort(-scores)
    picked = [candidates[int(i)] for i in order[:batch]]
    if not any(scores[int(i)] >= 0.0 for i in order[:batch]):
        # No approach-alive candidates — keep uniform so AbsRel still trains.
        return candidates[:batch]
    return picked


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
    p.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="Finetune from an existing DepthHead ckpt (prefer canonical AbsRel-PASS)",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override yaml checkpoint_dir (write FAIL candidates outside canonical)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override yaml lr (finetune often uses 3e-5)",
    )
    p.add_argument(
        "--delta-weight",
        type=float,
        default=None,
        help="Override yaml delta_weight for this run",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit overwriting an existing ckpt / clobbering depth_train.jsonl "
             "in the target dir. Off by default so a finetune run cannot silently "
             "replace the canonical AbsRel-PASS checkpoint.",
    )
    p.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=None,
        help="Freeze encoder; AdamW only on decoder (depth head). Preserves "
             "AbsRel-good features while Δ-finetuning scale. Overrides yaml.",
    )
    p.add_argument(
        "--no-freeze-encoder",
        action="store_true",
        help="Force full-model finetune even if yaml freeze_encoder=true.",
    )
    args = p.parse_args(argv)

    root = args.dataset
    _refuse_bad_corpus(root, args.allow_v0_desync)
    dh_cfg = _load_depth_cfg(args.config)
    if args.delta_weight is not None:
        print(f"[depth-train] NOTE: --delta-weight {args.delta_weight} overrides "
              f"yaml delta_weight={dh_cfg.get('delta_weight')}", file=sys.stderr)
        dh_cfg["delta_weight"] = float(args.delta_weight)
    if args.lr is not None:
        print(f"[depth-train] NOTE: --lr {args.lr} overrides yaml lr={dh_cfg.get('lr')}",
              file=sys.stderr)
        dh_cfg["lr"] = float(args.lr)
    if args.no_freeze_encoder:
        dh_cfg["freeze_encoder"] = False
    elif args.freeze_encoder:
        dh_cfg["freeze_encoder"] = True
    # The Δ term needs window STRICTLY > n_frames to have a non-degenerate Δ
    # interval. A finetune whose whole purpose is ③/Δ must not silently run for
    # hours with the term disabled — fail fast at setup instead.
    if float(dh_cfg.get("delta_weight", 0.0)) > 0.0 and int(args.window) <= int(dh_cfg["n_frames"]):
        print(f"[depth-train] FAIL: delta_weight>0 needs --window > n_frames "
              f"(got window={args.window}, n_frames={dh_cfg['n_frames']}); at "
              "window==n_frames the Δ interval collapses to a single frame (Δ≡0)",
              file=sys.stderr)
        return 1
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
    if args.init_ckpt is not None:
        ckpt_path = Path(args.init_ckpt)
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        # strict=True is deliberate: refuse to finetune from a checkpoint whose
        # architecture doesn't match the configured DepthHead (n_frames / base /
        # image_size). On mismatch load_state_dict raises — there is no
        # (missing, unexpected) tuple to report (that only comes back non-empty
        # with strict=False), so surface a clean, actionable FAIL instead.
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            print(f"[depth-train] FAIL: --init-ckpt {ckpt_path} arch mismatch vs "
                  f"configured DepthHead (n_frames={dh_cfg['n_frames']} "
                  f"base={dh_cfg['base']} image_size={dh_cfg['image_size']}): {e}",
                  file=sys.stderr)
            return 1
        prior = blob.get("holdout_absrel") if isinstance(blob, dict) else None
        print(f"[depth-train] init from {ckpt_path} (prior_holdout={prior})")
    freeze_enc = bool(dh_cfg.get("freeze_encoder", False))
    trainable = _apply_freeze_encoder(model, freeze_enc)
    if not trainable:
        print("[depth-train] FAIL: no trainable params after freeze_encoder",
              file=sys.stderr)
        return 1
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print(
        f"[depth-train] freeze_encoder={freeze_enc}: "
        f"trainable={n_train}/{n_all} params "
        f"(encoder={n_enc} frozen={freeze_enc})"
    )
    opt = torch.optim.AdamW(trainable, lr=float(dh_cfg["lr"]), betas=(0.9, 0.95))
    grad_clip = float(dh_cfg["grad_clip"])

    ckpt_dir = Path(str(args.checkpoint_dir or dh_cfg["checkpoint_dir"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "depth_train.jsonl"
    # Finetune runs default their save dir to the canonical checkpoint_dir. A
    # naive save + blind log-unlink would silently replace the AbsRel-PASS
    # canonical ckpt (and destroy its training record) with an unvalidated run.
    # Give finetune runs a distinct '_ft' stem and refuse to clobber anything
    # pre-existing unless --overwrite is explicit; point re-runs at a fresh
    # --checkpoint-dir instead.
    stem = f"depth_step_{args.steps}"
    if args.init_ckpt:
        stem += "_ft"
    if freeze_enc:
        stem += "_head"  # decoder-only / freeze-encoder run
    save_path = ckpt_dir / f"{stem}.pt"
    if args.save_ckpt:
        if args.init_ckpt is not None and save_path.resolve() == Path(args.init_ckpt).resolve():
            print(f"[depth-train] FAIL: save path {save_path} == --init-ckpt source; "
                  "refusing to overwrite the checkpoint being finetuned from",
                  file=sys.stderr)
            return 1
        if save_path.exists() and not args.overwrite:
            print(f"[depth-train] FAIL: {save_path} already exists; pass --overwrite "
                  "or a fresh --checkpoint-dir (won't clobber a canonical ckpt)",
                  file=sys.stderr)
            return 1
    if log_path.exists():
        if not args.overwrite:
            print(f"[depth-train] FAIL: {log_path} already exists; pass --overwrite "
                  "or a fresh --checkpoint-dir (won't destroy an existing train log)",
                  file=sys.stderr)
            return 1
        log_path.unlink()
    print(
        f"[depth-train] recipe: lr={dh_cfg['lr']} delta_weight={dh_cfg['delta_weight']} "
        f"delta_min_gt_m={dh_cfg.get('delta_min_gt_m')} "
        f"approach_oversample={dh_cfg.get('approach_oversample')} "
        f"freeze_encoder={freeze_enc} "
        f"ckpt_dir={ckpt_dir}"
    )

    absrels: List[float] = []
    losses: List[float] = []
    model.train()
    for step in range(1, int(args.steps) + 1):
        windows = _sample_approach_biased_windows(
            buf,
            int(args.wm_batch),
            int(args.window),
            oversample=int(dh_cfg.get("approach_oversample", 1)),
            min_depth_m=float(dh_cfg["scale_depth_min_m"]),
            max_depth_m=float(dh_cfg["scale_depth_max_m"]),
            min_gt_delta_m=float(dh_cfg.get("delta_min_gt_m", 0.5)),
            support_ratio=float(dh_cfg.get("delta_support_ratio", 0.0)),
            n_frames=int(dh_cfg["n_frames"]),
        )
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
        # n_frames context and match |Δ band-mean| to GT on approach-alive rows
        # — teaches ③ without drowning AbsRel. Requires window STRICTLY > n_frames:
        # at window == n_frames, gt[:, n_f-1] and gt[:, -1] are the same frame, so
        # Δ is identically 0 and the loss is a degenerate no-op. Also needs
        # position (motion gate); skip the term for a batch that lacks it.
        delta_w = float(dh_cfg.get("delta_weight", 0.0))
        if (delta_w > 0.0 and int(args.window) > int(dh_cfg["n_frames"])
                and "position" in arrays):
            n_f = int(dh_cfg["n_frames"])
            pred_first, _ = model.predict_from_window(rgb[:, :n_f])
            pos = torch.from_numpy(np.ascontiguousarray(arrays["position"])).to(device)
            # Motion must span the SAME interval as the depth Δ: pred_first/gt
            # are at frame n_f-1, pred_last/gt at frame -1. Using full-window
            # motion pos[-1]-pos[0] overstates ‖Δp‖ and makes the support gate
            # (s_gt ≥ support_ratio·‖Δp‖) reject valid approach windows.
            motion_m = torch.linalg.norm(pos[:, -1] - pos[:, n_f - 1], dim=-1)
            d_loss, d_stats = depth_delta_scale_loss(
                pred_first,
                pred,
                gt[:, n_f - 1],
                gt[:, -1],
                min_depth_m=float(dh_cfg["scale_depth_min_m"]),
                max_depth_m=float(dh_cfg["scale_depth_max_m"]),
                min_gt_delta_m=float(dh_cfg.get("delta_min_gt_m", 0.5)),
                motion_m=motion_m,
                support_ratio=float(dh_cfg.get("delta_support_ratio", 0.0)),
            )
            loss = loss + delta_w * d_loss
            stats = {**stats, **d_stats, "loss": float(loss.detach().item())}
        else:
            stats = {**stats, "delta_rel": float("nan"), "n_delta": 0}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
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
                f"n_delta={stats.get('n_delta', 0)} "
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
        path = save_path
        torch.save(
            {
                "model": model.state_dict(),
                "step": int(args.steps),
                "n_frames": int(dh_cfg["n_frames"]),
                "image_size": int(dh_cfg["image_size"]),
                "base": int(dh_cfg["base"]),
                "holdout_absrel": holdout,
                "depth_cfg": dh_cfg,
                "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None,
                "freeze_encoder": freeze_enc,
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
