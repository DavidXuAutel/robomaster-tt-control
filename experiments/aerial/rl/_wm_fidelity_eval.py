"""V1 imagination-FIDELITY eval on the H100 (torch) — "多步 rollout 达标" (§7).

``_wm_train_validate`` already proved the WM *learns* and its rollout is
*non-divergent* (§9). That is only the FLOOR of the §7 V1 criterion "多步
rollout 达标". This script fills the rest: rolled open-loop from real start
states under the *recorded* actions, does the WM **track** the real trajectory?

Verdict口径 (user decision, §1.5): BASELINE-RELATIVE + BOUNDED-GROWTH, no magic
absolute number — see ``wm_eval`` for the exact thresholds. Gate = the reward /
p_coll / done heads clear their baselines (``wm_eval.fidelity_verdict``) AND the
train-only decoder's multi-step recon does not blow up with the horizon.

    python -m experiments.aerial.rl._wm_fidelity_eval \
        --dataset /home/<user>/rl_collect_run/.../artifacts/dataset_v1_rgb \
        --ckpt    .../artifacts/wm_ckpt/wm_step_5000.pt \
        --config  configs/aerial_rl.yaml --heldout-frac 0.25 --horizon 15

HELD-OUT DISCIPLINE (read this): the 5000-step checkpoint from §3 was trained on
ALL episodes, so ``--heldout-frac 0`` measures IN-SAMPLE fidelity — a lower
bound only ("if it fails in-sample it definitely fails"). For an HONEST gate,
retrain the WM with the same held-out split excluded, then eval here with
``--heldout-frac`` matching. This script logs loudly which regime it ran in.

Runs on the H100 only (imports torch at module top). Exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch  # H100 only; the whole script is gated on this import
import yaml

from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl import wm_eval
from experiments.aerial.rl._wm_train_validate import _refuse_v0, _load_world_model_cfg
from experiments.aerial.rl.dynamics_torch import TorchRSSMDynamics


def _heldout_split(episodes: List[Any], frac: float) -> List[Any]:
    """Deterministic tail split — the last ceil(frac*N) episodes are held out."""
    n = len(episodes)
    if frac <= 0.0:
        print("[fidelity] WARNING: --heldout-frac=0 → IN-SAMPLE eval (lower bound "
              "only; the ckpt saw these episodes). Retrain with a held-out split "
              "for an honest gate.", file=sys.stderr)
        return episodes
    k = max(1, math.ceil(frac * n))
    held = episodes[n - k:]
    print(f"[fidelity] held-out split: {k}/{n} episodes (tail); train saw the rest")
    return held


def _make_windows(episodes: List[Any], horizon: int, n_starts: int) -> List[Sequence[Any]]:
    """Deterministic evenly-spaced start points per episode (no random sampling)."""
    windows: List[Sequence[Any]] = []
    for ep in episodes:
        if len(ep) < 2:
            continue
        last_start = max(0, len(ep) - 2)
        if n_starts <= 1:
            starts = [0]
        else:
            starts = sorted(set(int(round(i * last_start / (n_starts - 1)))
                                for i in range(n_starts)))
        for s in starts:
            windows.append(ep[s: s + horizon + 1])  # +1 real frame for recon depth
    if not windows:
        print("[fidelity] FAIL: no window >= 2 steps in held-out set", file=sys.stderr)
        raise SystemExit(1)
    print(f"[fidelity] {len(windows)} rollouts from {len(episodes)} episodes")
    return windows


@torch.no_grad()
def _recon_curve(model: TorchRSSMDynamics, windows: Sequence[Sequence[Any]],
                 horizon: int) -> Dict[str, Any]:
    """Multi-step open-loop decoder recon MSE vs the real frame at each depth.

    depth 0 = reconstruct the encoded start frame (posterior floor); depth d =
    decode the d-steps-imagined latent and compare to the real frame d steps
    ahead. The train-only decoder is never stepped online — here it is a
    diagnostic that the imagined latent still *carries the scene* H steps out.
    """
    model.eval()
    D = min(horizon, wm_eval.MAX_IMAGINATION_HORIZON)
    sums = np.zeros(D + 1)
    counts = np.zeros(D + 1)

    def _decode_mse(packed: np.ndarray, real_rgb: np.ndarray) -> float:
        feat = torch.from_numpy(np.ascontiguousarray(packed)).to(
            model.device, model.torch_dtype).reshape(1, -1)
        recon = model.decoder(feat)                       # [1, 3, H, W]
        tgt = torch.from_numpy(np.ascontiguousarray(real_rgb)).to(
            model.device, model.torch_dtype)
        tgt = tgt.permute(2, 0, 1).unsqueeze(0) / 255.0    # [1, 3, H, W]
        return float(torch.mean((recon - tgt) ** 2).item())

    for w in windows:
        z = np.asarray(model.encode(w[0].obs), dtype=np.float64)
        sums[0] += _decode_mse(z, w[0].obs.rgb); counts[0] += 1
        for d in range(1, D + 1):
            if d - 1 >= len(w):
                break
            out = model.step(z, np.asarray(w[d - 1].action, dtype=np.float64).reshape(4))
            z = np.asarray(out.z_next, dtype=np.float64)
            if d < len(w):
                sums[d] += _decode_mse(z, w[d].obs.rgb); counts[d] += 1

    with np.errstate(invalid="ignore"):
        curve = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    growth_ok = wm_eval.growth_bounded(curve)
    return {"recon_mse_curve": curve, "recon_growth_ok": bool(growth_ok)}


def _print_report(agg: Dict[str, Any], verdict: Dict[str, Any], recon: Dict[str, Any]) -> None:
    wm = agg["wm_reward_mae"]; base = agg["baseline_reward_mae"]
    persist = agg["persistence_reward_mae"]; rc = recon["recon_mse_curve"]
    print("\n[fidelity] per-horizon (h: wm_mae | mean-base | persist | recon_mse)")
    for h in range(agg["horizon"]):
        rcv = rc[h] if h < len(rc) else float("nan")
        print(f"  h={h:2d}: {wm[h]:8.4f} | {base[h]:8.4f} | {persist[h]:8.4f} | {rcv:8.5f}")
    print(f"[fidelity] reward: beat_frac={verdict['reward_beat_frac']:.2f} "
          f"growth_ok={verdict['reward_growth_ok']} -> {'OK' if verdict['reward_ok'] else 'FAIL'}")
    print(f"[fidelity] p_coll: AUROC={agg['coll_auroc']:.3f} "
          f"(+{agg['coll_traj_pos']}/-{agg['coll_traj_neg']} traj) "
          f"-> {'OK' if verdict['coll_ok'] else 'FAIL'}")
    print(f"[fidelity] done  : acc={agg['done_acc']:.3f} vs majority "
          f"{agg['done_majority_acc']:.3f} -> {'OK' if verdict['done_ok'] else 'FAIL'}")
    print(f"[fidelity] recon : growth_ok={recon['recon_growth_ok']} "
          f"(latent_norm_max={agg['latent_norm_max']:.2f})")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="dir with episode_*.npz (dataset_v1_rgb)")
    p.add_argument("--ckpt", required=True, help="trained WM checkpoint (wm_step_5000.pt)")
    p.add_argument("--config", default="configs/aerial_rl.yaml")
    p.add_argument("--heldout-frac", type=float, default=0.25,
                   help="tail fraction of episodes held out; 0 = in-sample (lower bound)")
    p.add_argument("--horizon", type=int, default=15)  # MAX_IMAGINATION_HORIZON (§9)
    p.add_argument("--n-starts", type=int, default=1, help="rollout start points per episode")
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-v0-desync", action="store_true")
    args = p.parse_args(argv)

    if args.horizon > wm_eval.MAX_IMAGINATION_HORIZON:
        print(f"[fidelity] FAIL: --horizon {args.horizon} exceeds §9 cap "
              f"{wm_eval.MAX_IMAGINATION_HORIZON}", file=sys.stderr)
        return 1

    root = Path(args.dataset)
    _refuse_v0(root, args.allow_v0_desync)
    episodes = ds.load_dataset(root, skip_quarantined=True)
    if not episodes:
        print("[fidelity] FAIL: no episodes loaded", file=sys.stderr)
        return 1
    held = _heldout_split(episodes, args.heldout_frac)
    windows = _make_windows(held, args.horizon, args.n_starts)

    wm_cfg = _load_world_model_cfg(Path(args.config))
    wm_cfg.setdefault("device", args.device)
    sample_obs = held[0][0].obs
    wm_cfg["image_size"] = int(np.asarray(sample_obs.rgb).shape[0])
    model = TorchRSSMDynamics.from_config(wm_cfg)
    payload = model.load_checkpoint(args.ckpt)
    print(f"[fidelity] loaded ckpt step={payload.get('step')} on {model.device} "
          f"| latent_dim={model.latent_dim} | image_size={wm_cfg['image_size']}")

    out = wm_eval.evaluate(model, windows, horizon=args.horizon)
    agg, verdict = out["agg"], out["verdict"]
    recon = _recon_curve(model, windows, args.horizon)
    _print_report(agg, verdict, recon)

    passed = bool(verdict["passed"] and recon["recon_growth_ok"])
    tag = "PASS" if passed else "FAIL"
    if args.heldout_frac <= 0.0:
        tag += " (IN-SAMPLE — lower bound, not the honest gate)"
    print(f"[fidelity] {tag}: reward={verdict['reward_ok']} coll={verdict['coll_ok']} "
          f"done={verdict['done_ok']} recon={recon['recon_growth_ok']}")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
