from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

from experiments.aerial.orchestration.checkpoint import is_complete_checkpoint
from experiments.aerial.orchestration.eval_queue import enqueue

B1_STEPS = tuple(range(250, 5001, 250))


def default_b1_metrics_path(results_root: Path, stamp: str, step: int) -> Path:
    return results_root / f"b1_{stamp}" / f"step_{step:06d}_seen20" / "metrics.json"


def _ensure_sha256_sidecar(pt: Path) -> None:
    sidecar = Path(str(pt) + ".sha256")
    if sidecar.is_file():
        return
    digest = hashlib.sha256()
    with pt.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    sidecar.write_text(digest.hexdigest() + f"  {pt.name}\n", encoding="utf-8")


def _mirror_checkpoint(source: Path, shared_weights_dir: Path) -> Path:
    shared_weights_dir.mkdir(parents=True, exist_ok=True)
    destination = shared_weights_dir / source.name
    source_sidecar = Path(str(source) + ".sha256")
    expected = source_sidecar.read_text(encoding="utf-8").split()[0]
    destination_sidecar = Path(str(destination) + ".sha256")
    if destination.is_file() and destination_sidecar.is_file():
        if destination_sidecar.read_text(encoding="utf-8").split()[0] == expected:
            return destination

    temporary = shared_weights_dir / f".{source.name}.{os.getpid()}.part"
    try:
        shutil.copyfile(source, temporary)
        digest = hashlib.sha256()
        with temporary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"shared checkpoint SHA256 mismatch: {source}")
        temporary.replace(destination)
        sidecar_tmp = Path(str(destination_sidecar) + f".{os.getpid()}.part")
        sidecar_tmp.write_text(
            expected + f"  {destination.name}\n", encoding="utf-8"
        )
        sidecar_tmp.replace(destination_sidecar)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _mirror_dataset_stats(weights_dir: Path, shared_weights_dir: Path) -> Path | None:
    """Place dataset_stats.json beside the mirrored checkpoint.

    Eval resolves the normalizer from `checkpoint.parent` / `checkpoint.parent.parent`;
    without it the policy skips action denormalization and only ever emits yaw
    primitives, so the drone never changes position.
    """
    for candidate in (
        weights_dir / "dataset_stats.json",
        weights_dir.parent / "dataset_stats.json",
        weights_dir.parent.parent / "dataset_stats.json",
    ):
        if not candidate.is_file():
            continue
        shared_weights_dir.mkdir(parents=True, exist_ok=True)
        destination = shared_weights_dir / "dataset_stats.json"
        payload = candidate.read_bytes()
        if destination.is_file() and destination.read_bytes() == payload:
            return destination
        temporary = shared_weights_dir / f".dataset_stats.json.{os.getpid()}.part"
        temporary.write_bytes(payload)
        temporary.replace(destination)
        return destination
    return None


def discover_b1_checkpoints(
    weights_dir: Path,
    *,
    steps: Sequence[int] = B1_STEPS,
    min_bytes: int = 1_000_000_000,
    settle_s: float = 5.0,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for step in steps:
        pt = weights_dir / f"step_{int(step):06d}.pt"
        if not pt.is_file():
            continue
        try:
            size1 = pt.stat().st_size
        except FileNotFoundError:
            continue
        if size1 < min_bytes:
            continue
        if settle_s > 0:
            time.sleep(settle_s)
            try:
                if pt.stat().st_size != size1:
                    continue
            except FileNotFoundError:
                continue
        _ensure_sha256_sidecar(pt)
        if not is_complete_checkpoint(pt, settle_s=0.0, min_bytes=min_bytes):
            continue
        found.append(
            {
                "step": int(step),
                "checkpoint": str(pt.resolve()),
                "sha256_path": str(Path(str(pt) + ".sha256").resolve()),
            }
        )
    return found


def build_b1_eval_job(
    *,
    stamp: str,
    step: int,
    checkpoint: str,
    results_root: Path,
    ann: Path,
    openfly_root: Path,
    task: str = "aerial_joint_b1_joint",
) -> dict[str, Any]:
    out_metrics = default_b1_metrics_path(results_root, stamp, step)
    return {
        "id": f"b1-{stamp}-step_{step:06d}",
        "kind": "b1",
        "checkpoint": checkpoint,
        "out_metrics": str(out_metrics),
        "task": task,
        "ann": str(ann),
        "openfly_root": str(openfly_root),
        "seed": 42,
        "max_steps": 100,
        "max_episodes": 20,
    }


def enqueue_ready_b1_jobs(
    *,
    stamp: str,
    weights_dir: Path,
    shared_weights_dir: Path | None = None,
    queue_dir: Path,
    results_root: Path,
    ann: Path,
    openfly_root: Path,
    task: str = "aerial_joint_b1_joint",
    steps: Sequence[int] = B1_STEPS,
    min_bytes: int = 1_000_000_000,
    settle_s: float = 5.0,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for item in discover_b1_checkpoints(
        weights_dir, steps=steps, min_bytes=min_bytes, settle_s=settle_s
    ):
        checkpoint = Path(str(item["checkpoint"]))
        if shared_weights_dir is not None:
            checkpoint = _mirror_checkpoint(checkpoint, shared_weights_dir)
            if _mirror_dataset_stats(weights_dir, shared_weights_dir) is None:
                raise FileNotFoundError(
                    "missing dataset_stats.json for "
                    f"{weights_dir}; eval would run without action denormalization"
                )
        job = build_b1_eval_job(
            stamp=stamp,
            step=int(item["step"]),
            checkpoint=str(checkpoint.resolve()),
            results_root=results_root,
            ann=ann,
            openfly_root=openfly_root,
            task=task,
        )
        Path(job["out_metrics"]).parent.mkdir(parents=True, exist_ok=True)
        enqueue(queue_dir, job)
        jobs.append(job)
    return jobs


def watch_and_enqueue(
    *,
    stamp: str,
    weights_dir: Path,
    shared_weights_dir: Path | None = None,
    queue_dir: Path,
    results_root: Path,
    ann: Path,
    openfly_root: Path,
    task: str = "aerial_joint_b1_joint",
    steps: Sequence[int] = B1_STEPS,
    poll_s: float = 60.0,
    min_bytes: int = 1_000_000_000,
    settle_s: float = 5.0,
    once: bool = False,
) -> None:
    while True:
        jobs = enqueue_ready_b1_jobs(
            stamp=stamp,
            weights_dir=weights_dir,
            shared_weights_dir=shared_weights_dir,
            queue_dir=queue_dir,
            results_root=results_root,
            ann=ann,
            openfly_root=openfly_root,
            task=task,
            steps=steps,
            min_bytes=min_bytes,
            settle_s=settle_s,
        )
        print(f"enqueued={len(jobs)}")
        for job in jobs:
            print(job["id"], job["checkpoint"], job["out_metrics"])
        if once:
            return
        time.sleep(poll_s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch B1 checkpoints and enqueue evals")
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--shared-weights-dir", type=Path)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--openfly-root", type=Path, required=True)
    parser.add_argument("--task", default="aerial_joint_b1_joint")
    parser.add_argument("--steps", default=",".join(str(s) for s in B1_STEPS))
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--min-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--settle-s", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    steps = tuple(int(p.strip()) for p in args.steps.split(",") if p.strip())
    watch_and_enqueue(
        stamp=args.stamp,
        weights_dir=args.weights_dir,
        shared_weights_dir=args.shared_weights_dir,
        queue_dir=args.queue_dir,
        results_root=args.results_root,
        ann=args.ann,
        openfly_root=args.openfly_root,
        task=args.task,
        steps=steps,
        poll_s=args.poll_s,
        min_bytes=args.min_bytes,
        settle_s=args.settle_s,
        once=args.once,
    )


if __name__ == "__main__":
    main()
