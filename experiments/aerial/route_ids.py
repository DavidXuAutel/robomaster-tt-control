from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


def route_id(episode: dict[str, Any]) -> str:
    for key in ("trajectory_id", "traj_id", "id", "uuid"):
        if episode.get(key):
            return str(episode[key])
    blob = json.dumps(
        {
            "scene": episode.get("scene_id", episode.get("scene")),
            "image_path": episode.get("image_path"),
            "instruction": episode.get("gpt_instruction"),
            "start": (episode.get("pos") or [None])[0],
            "goal": (episode.get("pos") or [None])[-1],
        },
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def assert_disjoint(a: set[str], b: set[str]) -> None:
    overlap = sorted(a & b)
    if overlap:
        raise ValueError(f"heldout/collection overlap: {overlap[:10]}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_collection40(
    source_path: Path,
    heldout_path: Path,
    out_ann_path: Path,
    out_manifest_path: Path,
    *,
    seed: int = 42,
    n: int = 40,
) -> dict[str, Any]:
    source = json.loads(Path(source_path).read_text())
    heldout = json.loads(Path(heldout_path).read_text())
    held_ids = {route_id(e) for e in heldout}
    candidates = [e for e in source if route_id(e) not in held_ids]
    if len(candidates) < n:
        raise ValueError(f"need {n} collection routes, have {len(candidates)}")
    rng = random.Random(seed)
    chosen = rng.sample(candidates, n)
    assert_disjoint(held_ids, {route_id(e) for e in chosen})
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
    out_ann_path.write_text(json.dumps(chosen, indent=2))
    manifest = {
        "seed": seed,
        "n": n,
        "heldout_path": str(heldout_path),
        "source_path": str(source_path),
        "out_ann_path": str(out_ann_path),
        "heldout_sha256": sha256_file(Path(heldout_path)),
        "source_sha256": sha256_file(Path(source_path)),
        "heldout_route_ids": sorted(held_ids),
        "collection_route_ids": [route_id(e) for e in chosen],
    }
    out_manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
