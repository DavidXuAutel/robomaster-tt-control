#!/usr/bin/env python3
"""Compute d_max (p90 nearest-prim distance) on train_subset actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.aerial.collapse_fix.labels import (
    FORWARD_PRIMITIVE_IDS,
    MINORITY_PRIMITIVE_IDS,
    delta_nearest_with_dist,
)
from experiments.aerial.openfly_actions import OPENFLY_PRIMITIVES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/openfly_lerobot/train_subset"),
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/collapse_fix_dmax.json"))
    parser.add_argument("--percentile", type=float, default=90.0)
    args = parser.parse_args()

    files = sorted(args.dataset.glob("data/*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.dataset}")

    dists_all: list[float] = []
    dists_fwd: list[float] = []
    dists_min: list[float] = []
    counts = np.zeros(10, dtype=np.int64)

    for path in files:
        df = pd.read_parquet(path)
        for row in df["action"].values:
            pid, dist = delta_nearest_with_dist(np.asarray(row, dtype=np.float64))
            counts[pid] += 1
            dists_all.append(dist)
            if pid in FORWARD_PRIMITIVE_IDS:
                dists_fwd.append(dist)
            if pid in MINORITY_PRIMITIVE_IDS:
                dists_min.append(dist)

    d_max = float(np.percentile(dists_fwd if dists_fwd else dists_all, args.percentile))
    out = {
        "n_frames": len(dists_all),
        "d_max_p90_forward": d_max,
        "percentile": args.percentile,
        "counts": {str(i): int(counts[i]) for i in range(10)},
        "names": {str(i): list(OPENFLY_PRIMITIVES[i]) for i in range(10)},
        "mean_dist_minority": float(np.mean(dists_min)) if dists_min else None,
        "mean_dist_forward": float(np.mean(dists_fwd)) if dists_fwd else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
