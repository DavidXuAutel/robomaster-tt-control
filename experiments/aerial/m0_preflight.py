#!/usr/bin/env python3
"""CPU-side M0 wiring checks (not a substitute for the 50-step GPU gate)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lerobot-root",
        default="data/openfly_lerobot/train_subset",
        help="LeRobot dataset root passed to verify_aerial_source.py",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verify_aerial_source.py (e.g. when only smoke data is present)",
    )
    args = parser.parse_args()

    root = _repo_root()
    cfg_dir = str((root / "configs").resolve())
    ok = True

    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=aerial_joint_1cam_1e-4"])

    print(
        "[preflight] action_dim=%s cams=%s max_steps=%s"
        % (
            cfg.data.train.processor.action_output_dim,
            cfg.data.train.processor.num_output_cameras,
            cfg.max_steps,
        )
    )
    print("[preflight] text_cache=%s" % cfg.data.train.text_embedding_cache_dir)

    if not args.skip_verify:
        verify_cmd = [
            sys.executable,
            str(root / "experiments/aerial/verify_aerial_source.py"),
            "--lerobot-root",
            str(root / args.lerobot_root),
            "--sample-size",
            "5",
        ]
        print("[preflight] running:", " ".join(verify_cmd))
        result = subprocess.run(verify_cmd, cwd=root, check=False)
        if result.returncode != 0:
            ok = False
            print("[preflight] verify_aerial_source FAILED")

    try:
        ds = instantiate(cfg.data.train)
        print("[preflight] dataset_len=%s" % len(ds))
        _ = ds[0]
        print("[preflight] sample_ok=True")
    except Exception as exc:
        ok = False
        print("[preflight] sample_ok=False (%s: %s)" % (type(exc).__name__, exc))

    if ok:
        print("[preflight] PASS wiring checks (GPU 50-step train still required for M0)")
        return 0

    print(
        "[preflight] BLOCKED: fix dataset/text cache on a Linux CUDA host, "
        "then run experiments/aerial/README.md"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
