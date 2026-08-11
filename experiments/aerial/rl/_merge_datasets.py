"""Merge several collection dirs into one by copying + re-indexing episodes.

Each source is a dir of self-contained ``episode_*.npz`` (see ``dataset.py``);
``load_dataset`` only globs ``episode_*.npz`` sorted by name, so a merge is just
a renumbered copy. Provenance (which source each episode came from) is written to
``merge_manifest.json`` in the output dir. Nothing governed changes — this only
copies data files.

    python -m experiments.aerial.rl._merge_datasets \
      --src <dirA> --src <dirB> --out <mergedDir> [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", action="append", required=True,
                    help="source collection dir (repeatable, order preserved)")
    ap.add_argument("--out", required=True, help="output merged dir")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into a non-empty output dir")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.glob("episode_*.npz")) and not args.overwrite:
        raise SystemExit(f"{out} already has episodes; pass --overwrite to replace")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("episode_*.npz"):
        old.unlink()

    provenance = []
    idx = 0
    for src in args.src:
        sdir = Path(src)
        eps = sorted(sdir.glob("episode_*.npz"))
        if not eps:
            raise SystemExit(f"no episode_*.npz under {sdir}")
        for ep in eps:
            dst = out / f"episode_{idx:05d}.npz"
            shutil.copy2(ep, dst)
            provenance.append({"index": idx, "src": str(sdir), "orig": ep.name})
            idx += 1
        print(f"[merge] {sdir}: {len(eps)} eps")

    manifest = out / "merge_manifest.json"
    manifest.write_text(json.dumps(
        {"sources": args.src, "total": idx, "episodes": provenance}, indent=2))
    print(f"[merge] TOTAL -> {out} : {idx} eps  (manifest: {manifest})")


if __name__ == "__main__":  # pragma: no cover
    main()
