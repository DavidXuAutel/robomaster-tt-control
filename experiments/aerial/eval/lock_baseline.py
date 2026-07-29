from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from experiments.aerial.orchestration.state import write_status

ORCHESTRATION_VERSION = "aerial-b0-b1-orchestration/1"
_CANDIDATE_KEYS = ("step", "checkpoint", "metrics_path", "mean_ne", "sha256")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_READ_CHUNK_BYTES = 8192


def load_metrics(metrics_path: Path) -> float:
    if not metrics_path.is_file():
        raise ValueError(f"metrics file not found: {metrics_path}")
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metrics JSON: {metrics_path}") from exc
    try:
        ne = float(data["NE"])
        n = float(data["n"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"metrics missing NE or n: {metrics_path}") from exc
    if not math.isfinite(ne):
        raise ValueError(f"non-finite NE in metrics: {metrics_path}")
    if not math.isfinite(n) or n < 1:
        raise ValueError(f"invalid n in metrics: {metrics_path}")
    return ne


def validate_sha256(sha256: str) -> None:
    if not _SHA256_RE.match(sha256):
        raise ValueError(f"invalid sha256: {sha256!r}")


def validate_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if not path.is_file():
        raise ValueError(f"checkpoint not found: {path}")
    return path


def compute_checkpoint_sha256(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_sha256(checkpoint: Path, expected: str) -> None:
    validate_sha256(expected)
    actual = compute_checkpoint_sha256(checkpoint)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"sha256 mismatch for checkpoint {checkpoint}: expected {expected}, got {actual}"
        )


def validate_candidate(candidate: dict) -> dict:
    try:
        step = int(candidate["step"])
        checkpoint = str(candidate["checkpoint"])
        metrics_path = str(candidate["metrics_path"])
        sha256 = str(candidate["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid candidate fields: {candidate!r}") from exc
    if "mean_ne" not in candidate:
        raise ValueError(f"missing mean_ne for step {step}")
    mean_ne = float(candidate["mean_ne"])
    if not math.isfinite(mean_ne):
        raise ValueError(f"non-finite mean_ne for step {step}")

    ckpt_path = validate_checkpoint_path(checkpoint)
    verify_checkpoint_sha256(ckpt_path, sha256)
    metrics_ne = load_metrics(Path(metrics_path))
    if mean_ne != metrics_ne:
        raise ValueError(
            f"mean_ne mismatch for step {step}: candidate has {mean_ne}, metrics has {metrics_ne}"
        )

    return {
        "step": step,
        "checkpoint": checkpoint,
        "metrics_path": metrics_path,
        "mean_ne": mean_ne,
        "sha256": sha256,
    }


def validate_candidates_unique(candidates: list[dict]) -> None:
    steps: set[int] = set()
    checkpoints: set[str] = set()
    metrics_paths: set[str] = set()
    for candidate in candidates:
        step = int(candidate["step"])
        checkpoint = str(candidate["checkpoint"])
        metrics_path = str(candidate["metrics_path"])
        if step in steps:
            raise ValueError(f"duplicate step: {step}")
        if checkpoint in checkpoints:
            raise ValueError(f"duplicate checkpoint: {checkpoint}")
        if metrics_path in metrics_paths:
            raise ValueError(f"duplicate metrics_path: {metrics_path}")
        steps.add(step)
        checkpoints.add(checkpoint)
        metrics_paths.add(metrics_path)


def validate_candidates(candidates: list[dict]) -> list[dict]:
    if not candidates:
        raise ValueError("no candidates")
    validated = [validate_candidate(candidate) for candidate in candidates]
    validate_candidates_unique(validated)
    return validated


def select_baseline(candidates: list[dict]) -> dict:
    validated = validate_candidates(candidates)
    return sorted(validated, key=lambda c: (float(c["mean_ne"]), -int(c["step"])))[0]


def _candidate_matches(left: dict, right: dict) -> bool:
    return all(left[key] == right[key] for key in _CANDIDATE_KEYS)


def _assert_chosen_in_candidates(chosen: dict, candidates: list[dict]) -> None:
    for candidate in candidates:
        if _candidate_matches(candidate, chosen):
            return
    raise ValueError("chosen candidate not in candidates list")


def build_lock_manifest(
    chosen: dict,
    *,
    candidates: list[dict],
    stamp: str,
    selection_time: str | None = None,
) -> dict:
    validated = validate_candidates(candidates)
    chosen_validated = validate_candidate(chosen)
    _assert_chosen_in_candidates(chosen_validated, validated)
    if selection_time is None:
        selection_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    baseline = float(chosen_validated["mean_ne"])
    return {
        "stamp": stamp,
        "checkpoint": chosen_validated["checkpoint"],
        "sha256": chosen_validated["sha256"],
        "metrics_path": chosen_validated["metrics_path"],
        "baseline_mean_ne": baseline,
        "s1_ne": 0.8 * baseline,
        "candidates": validated,
        "selection_rule": "min_mean_ne_tie_later_step",
        "selection_time": selection_time,
        "orchestration_version": ORCHESTRATION_VERSION,
    }


def parse_candidate_json(spec: str) -> dict:
    try:
        raw = json.loads(spec)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid candidate JSON: {spec!r}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"invalid candidate JSON: {spec!r}")
    try:
        step = int(raw["step"])
        checkpoint = str(raw["checkpoint"])
        metrics_path = str(raw["metrics_path"])
        sha256 = str(raw["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid candidate JSON: {spec!r}") from exc
    mean_ne = load_metrics(Path(metrics_path))
    return validate_candidate(
        {
            "step": step,
            "checkpoint": checkpoint,
            "metrics_path": metrics_path,
            "mean_ne": mean_ne,
            "sha256": sha256,
        }
    )


def write_lock_manifest(path: Path, manifest: dict) -> None:
    write_status(path, manifest)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock B0 baseline from eval candidates",
        allow_abbrev=False,
    )
    parser.add_argument("--stamp", required=True)
    parser.add_argument(
        "--candidate-json",
        action="append",
        required=True,
        metavar="JSON",
        help="Candidate object JSON: {step, checkpoint, metrics_path, sha256}",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    candidates = validate_candidates([parse_candidate_json(spec) for spec in args.candidate_json])
    chosen = select_baseline(candidates)
    manifest = build_lock_manifest(chosen, candidates=candidates, stamp=args.stamp)
    write_lock_manifest(args.out, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
