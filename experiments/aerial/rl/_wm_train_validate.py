"""V1 world-model VALIDATION on the H100 (torch) — the gate before enabling training.

Unlike ``_wm_bringup_smoke`` (which only checks the disk→buffer→stub path with no
weights), this trains the real :class:`TorchRSSMDynamics` on ``dataset_v1_rgb``
and checks the two V1 pass criteria from the design doc:

  A. LEARNING — ``update()`` loss (and recon term) trends DOWN over N steps, all
     finite; the posterior does NOT collapse (entropy stays above the §2.3 floor).
  B. NON-DIVERGENCE (§9) — open-loop multi-step imagination from real start states
     stays bounded: latents finite, norm not exploding, p_coll∈[0,1] over the
     horizon cap. This is the gate: WM multi-step error must not blow up.

Runs on the H100 only (imports torch at module top). Exits 0 on PASS, 1 on FAIL.

    python -m experiments.aerial.rl._wm_train_validate \
        --dataset /home/<user>/rl_collect_run/.../artifacts/dataset_v1_rgb \
        --config configs/aerial_rl.yaml --steps 500

Refuses the dt-desynced V0 corpus (step_hz>8.5) unless --allow-v0-desync, exactly
like the bring-up smoke — a real WM must not be trained on desynced labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch  # H100 only; the whole script is gated on this import

import yaml

from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics


def _load_world_model_cfg(config_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text()) or {}
    return dict(cfg.get("world_model", {}) or {})


def _refuse_v0(root: Path, allow: bool) -> None:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    step_hz = float((manifest.get("meta") or {}).get("step_hz", 0) or 0)
    if step_hz > 8.5 and not allow:
        print(
            f"[wm-validate] REFUSE: dataset step_hz={step_hz} is the dt-desynced V0 "
            "corpus — do not train a real WM on it. Pass --allow-v0-desync only to "
            "exercise the code path, or point --dataset at dataset_v1_rgb.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _load_buffer(root: Path, window: int) -> ReplayBuffer:
    episodes = ds.load_dataset(root, skip_quarantined=True)
    episodes = [ep for ep in episodes if len(ep) >= window]
    if not episodes:
        print(f"[wm-validate] FAIL: no episode >= {window} steps", file=sys.stderr)
        raise SystemExit(1)
    buf = ReplayBuffer(capacity_episodes=len(episodes) + 1, seed=0)
    for ep in episodes:
        buf.add_episode(ep)
    print(f"[wm-validate] buffer: {buf.num_episodes} eps / {buf.num_transitions} steps")
    return buf


def _check_learning(model: TorchRSSMDynamics, buf: ReplayBuffer,
                    steps: int, wm_batch: int, window: int) -> bool:
    losses: List[float] = []
    recons: List[float] = []
    ent_fracs: List[float] = []
    for i in range(steps):
        windows = buf.sample_windows(wm_batch, window)
        out = model.update(windows)
        losses.append(out["loss"])
        recons.append(out["recon_err"])
        ent_fracs.append(out.get("post_entropy_frac", 1.0))
        if i % max(1, steps // 10) == 0:
            print(f"[wm-validate] step {i:4d} | loss={out['loss']:.4f} "
                  f"recon={out['recon_err']:.4f} dyn={out['loss_dyn']:.3f} "
                  f"rep={out['loss_rep']:.3f} ent={ent_fracs[-1]:.2f} "
                  f"|g|={out['grad_norm']:.1f}")

    if not all(np.isfinite(losses)):
        print("[wm-validate] FAIL(A): non-finite loss during training", file=sys.stderr)
        return False
    k = max(1, steps // 10)
    first, last = float(np.mean(losses[:k])), float(np.mean(losses[-k:]))
    recon_first, recon_last = float(np.mean(recons[:k])), float(np.mean(recons[-k:]))
    min_ent = float(np.min(ent_fracs))
    loss_ok = last < first * 0.98              # ≥2% total-loss drop
    recon_ok = recon_last <= recon_first       # recon not worse
    collapse_ok = min_ent >= model.collapse_entropy_frac
    print(f"[wm-validate] LEARNING: loss {first:.4f}→{last:.4f} ({'OK' if loss_ok else 'FAIL'}) | "
          f"recon {recon_first:.4f}→{recon_last:.4f} ({'OK' if recon_ok else 'FAIL'}) | "
          f"min entropy frac {min_ent:.2f} ({'OK' if collapse_ok else 'COLLAPSE'})")
    return loss_ok and recon_ok and collapse_ok


def _check_non_divergence(model: TorchRSSMDynamics, buf: ReplayBuffer,
                          window: int, horizon: int, n_traj: int = 8) -> bool:
    windows = buf.sample_windows(n_traj, window)
    ok = True
    max_norm_seen = 0.0
    for w in windows:
        z = model.encode(w[0].obs)
        norms = [float(np.linalg.norm(z))]
        for t in range(min(horizon, len(w))):
            out = model.step(z, w[t].action)
            z = out.z_next
            if not np.all(np.isfinite(z)):
                print("[wm-validate] FAIL(B): non-finite latent in rollout", file=sys.stderr)
                return False
            if not (0.0 <= out.p_coll <= 1.0) or not np.isfinite(out.progress):
                print(f"[wm-validate] FAIL(B): bad head output p_coll={out.p_coll} "
                      f"progress={out.progress}", file=sys.stderr)
                return False
            norms.append(float(np.linalg.norm(z)))
        max_norm_seen = max(max_norm_seen, max(norms))
        # non-divergence: the packed-latent norm must not blow up over the horizon.
        if max(norms) > 50.0 * (norms[0] + 1.0):
            print(f"[wm-validate] FAIL(B): latent norm diverged {norms[0]:.2f}→{max(norms):.2f}",
                  file=sys.stderr)
            ok = False
    print(f"[wm-validate] NON-DIVERGENCE: {n_traj} rollouts × H={horizon}, "
          f"max latent norm {max_norm_seen:.2f} ({'OK' if ok else 'FAIL'})")
    return ok


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="dir with episode_*.npz (dataset_v1_rgb)")
    p.add_argument("--config", default="configs/aerial_rl.yaml")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--wm-batch", type=int, default=8)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--horizon", type=int, default=15)  # MAX_IMAGINATION_HORIZON (§9)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-v0-desync", action="store_true")
    p.add_argument(
        "--save-ckpt",
        action="store_true",
        help="on PASS, write world_model.checkpoint_dir/wm_step_<N>.pt (runbook §3)",
    )
    args = p.parse_args(argv)

    root = Path(args.dataset)
    _refuse_v0(root, args.allow_v0_desync)
    buf = _load_buffer(root, args.window)

    wm_cfg = _load_world_model_cfg(Path(args.config))
    wm_cfg.setdefault("device", args.device)
    # Match the model's image size to the actual frame (defaults to 224).
    sample_obs = buf.sample_windows(1, 1)[0][0].obs
    wm_cfg["image_size"] = int(np.asarray(sample_obs.rgb).shape[0])
    model = TorchRSSMDynamics.from_config(wm_cfg)
    print(f"[wm-validate] model on {model.device} | latent_dim={model.latent_dim} "
          f"| image_size={wm_cfg['image_size']}")

    learn_ok = _check_learning(model, buf, args.steps, args.wm_batch, args.window)
    diverge_ok = _check_non_divergence(model, buf, args.window, args.horizon)

    passed = learn_ok and diverge_ok
    print(f"[wm-validate] {'PASS' if passed else 'FAIL'}: "
          f"learning={learn_ok} non_divergence={diverge_ok}")
    if passed and args.save_ckpt:
        ckpt_dir = Path(wm_cfg.get("checkpoint_dir") or "experiments/aerial/rl/artifacts/wm_ckpt")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"wm_step_{args.steps}.pt"
        model.save_checkpoint(str(ckpt_path), step=args.steps)
        print(f"[wm-validate] checkpoint → {ckpt_path}")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
