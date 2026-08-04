"""WM bring-up smoke: load on-disk episodes → buffer → stub V1 gate (no training).

Validates the V0→V1 *data path* without claiming a trainable world model:

  1. ``load_dataset`` rehydrates ``episode_*.npz``
  2. Quarantined instant-crash episodes are skipped (smoke corpus hygiene)
  3. ``ReplayBuffer.sample_windows`` succeeds
  4. ``StubLatentDynamics.update`` returns the V1-gated skip marker (no weights)
  5. ``encode`` + ``step`` run on a real loaded obs/action (stub imagination)

dataset_v0 was collected at commanded 12 Hz / achieved ~8 Hz (dt-desync) — this
smoke may use it for pipeline verification only. Do NOT flip
``enable_wm_update`` to a real trainer on this corpus; re-collect at step_hz=8
for V1 training data.

    python -m experiments.aerial.rl._wm_bringup_smoke \\
        --dataset experiments/aerial/rl/artifacts/dataset_v0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.aerial.rl.buffer import ReplayBuffer
from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.dynamics import StubLatentDynamics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default="experiments/aerial/rl/artifacts/dataset_v0",
        help="directory with episode_*.npz (+ optional QUALITY_SUMMARY.json)",
    )
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--wm-batch", type=int, default=4)
    p.add_argument(
        "--allow-v0-desync",
        action="store_true",
        help="required to run against dataset_v0 (12 Hz labels / ~8 Hz wall) — "
             "documents that this is pipeline smoke, not V1 training",
    )
    args = p.parse_args(argv)
    root = Path(args.dataset)

    summary_path = root / "QUALITY_SUMMARY.json"
    meta_note = ""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        meta_note = (
            f"summary episodes={summary.get('episodes')} "
            f"usable≈{summary.get('usable', '?')} "
            f"path_mean={summary.get('path_length_m', {}).get('mean', '?')}"
        )

    # Hard policy: refuse to pretend this is a V1 train set without the flag.
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        step_hz = float((manifest.get("meta") or {}).get("step_hz", 0) or 0)
        if step_hz > 8.5 and not args.allow_v0_desync:
            print(
                f"[wm-bringup] REFUSE: dataset step_hz={step_hz} looks like the "
                "dt-desynced V0 smoke corpus. Pass --allow-v0-desync to exercise "
                "the load path only, or re-collect at step_hz=8 for V1 training.",
                file=sys.stderr,
            )
            return 2

    print(f"[wm-bringup] loading {root} {meta_note}")
    episodes = ds.load_dataset(root, skip_quarantined=True)
    if not episodes:
        print("[wm-bringup] FAIL: no usable episodes after quarantine filter", file=sys.stderr)
        return 1

    # Drop leftovers too short for the requested WM window (legacy short crashes
    # that quarantine didn't catch, or truncated rolls).
    long_enough = [ep for ep in episodes if len(ep) >= args.window]
    if not long_enough:
        print(
            f"[wm-bringup] FAIL: no episode has >= {args.window} steps "
            f"(loaded {len(episodes)} after quarantine)",
            file=sys.stderr,
        )
        return 1
    if len(long_enough) < len(episodes):
        print(
            f"[wm-bringup] dropped {len(episodes) - len(long_enough)} short "
            f"episode(s) (< {args.window} steps)"
        )
    episodes = long_enough

    buf = ReplayBuffer(capacity_episodes=max(1000, len(episodes) + 1), seed=0)
    for ep in episodes:
        bad = ds.assert_nontrivial(ds.quality_report(ep))
        if bad:
            print(f"[wm-bringup] FAIL nontrivial after load: {bad}", file=sys.stderr)
            return 1
        buf.add_episode(ep)

    print(
        f"[wm-bringup] buffer: {buf.num_episodes} eps / {buf.num_transitions} steps "
        f"(quarantine skipped)"
    )

    window = min(args.window, min(len(ep) for ep in episodes))
    if window < 2:
        print("[wm-bringup] FAIL: usable episodes too short for windows", file=sys.stderr)
        return 1

    windows = buf.sample_windows(args.wm_batch, window)
    print(f"[wm-bringup] sampled {len(windows)} windows × {window}")

    dyn = StubLatentDynamics(latent_dim=8)
    update_out = dyn.update(windows)
    if not update_out.get("skipped"):
        print(
            f"[wm-bringup] FAIL: stub update must stay V1-gated skipped, got {update_out}",
            file=sys.stderr,
        )
        return 1
    print(f"[wm-bringup] stub.update → skipped OK ({update_out.get('reason')})")

    # One encode/step against a real loaded frame (still stub dynamics).
    t0 = windows[0][0]
    z = dyn.encode(t0.obs)
    out = dyn.step(z, t0.action)
    print(
        f"[wm-bringup] stub encode/step OK | z_dim={z.shape[0]} "
        f"p_coll={out.p_coll:.3f} progress={out.progress:.3f} done={out.done}"
    )

    print("[wm-bringup] OK: disk→buffer→window→stub-V1-gate path verified")
    if manifest_path.exists():
        step_hz = float((manifest.get("meta") or {}).get("step_hz", 0) or 0)
        if step_hz > 8.5:
            print(
                "[wm-bringup] NOTE: this looks like the V0 smoke corpus "
                "(step_hz>8.5) — do not train a real WM on it."
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
