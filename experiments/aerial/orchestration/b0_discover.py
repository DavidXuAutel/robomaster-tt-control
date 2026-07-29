from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from experiments.aerial.orchestration.checkpoint import is_complete_checkpoint
from experiments.aerial.orchestration.eval_queue import enqueue

B0_STEPS = (1000, 2000, 3000, 4000, 5000)
_STEP_RE = re.compile(r"^step_(\d{6})\.pt$")


def discover_b0_checkpoints(
    weights_dir: Path,
    *,
    steps: tuple[int, ...] = B0_STEPS,
    min_bytes: int = 1_000_000_000,
    settle_s: float = 5.0,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for step in steps:
        pt = weights_dir / f"step_{step:06d}.pt"
        if not is_complete_checkpoint(pt, settle_s=settle_s, min_bytes=min_bytes):
            continue
        found.append(
            {
                "step": step,
                "checkpoint": str(pt.resolve()),
                "sha256_path": str(Path(str(pt) + ".sha256").resolve()),
            }
        )
    return found


def default_metrics_path(results_root: Path, step: int) -> Path:
    if step == 1000:
        legacy = results_root / "step_001000_seen20" / "metrics.json"
        if legacy.is_file():
            return legacy
    return results_root / f"b0_step_{step:06d}_seen20" / "metrics.json"


def build_b0_eval_job(
    *,
    step: int,
    checkpoint: str,
    results_root: Path,
    ann: Path,
    openfly_root: Path,
    task: str = "aerial_joint_b1_joint",
) -> dict[str, Any]:
    out_metrics = default_metrics_path(results_root, step)
    return {
        "id": f"b0-step_{step:06d}",
        "kind": "b0",
        "checkpoint": checkpoint,
        "out_metrics": str(out_metrics),
        "task": task,
        "ann": str(ann),
        "openfly_root": str(openfly_root),
        "seed": 42,
        "max_steps": 100,
        "max_episodes": 20,
    }


def enqueue_b0_eval_jobs(
    *,
    weights_dir: Path,
    queue_dir: Path,
    results_root: Path,
    ann: Path,
    openfly_root: Path,
    task: str = "aerial_joint_b1_joint",
    min_bytes: int = 1_000_000_000,
    settle_s: float = 5.0,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for item in discover_b0_checkpoints(
        weights_dir, min_bytes=min_bytes, settle_s=settle_s
    ):
        job = build_b0_eval_job(
            step=int(item["step"]),
            checkpoint=str(item["checkpoint"]),
            results_root=results_root,
            ann=ann,
            openfly_root=openfly_root,
            task=task,
        )
        Path(job["out_metrics"]).parent.mkdir(parents=True, exist_ok=True)
        enqueue(queue_dir, job)
        jobs.append(job)
    return jobs


def wait_for_final_checkpoint(
    weights_dir: Path,
    *,
    final_step: int = 5000,
    poll_s: float = 60.0,
    min_bytes: int = 1_000_000_000,
    settle_s: float = 5.0,
    max_wait_s: float | None = None,
) -> Path:
    import time

    pt = weights_dir / f"step_{final_step:06d}.pt"
    waited = 0.0
    while True:
        if is_complete_checkpoint(pt, settle_s=settle_s, min_bytes=min_bytes):
            return pt
        if max_wait_s is not None and waited >= max_wait_s:
            raise TimeoutError(f"timed out waiting for {pt}")
        time.sleep(poll_s)
        waited += poll_s


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover/enqueue B0 eval checkpoints")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--openfly-root", type=Path, required=True)
    parser.add_argument("--task", default="aerial_joint_b1_joint")
    parser.add_argument("--min-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--settle-s", type=float, default=5.0)
    parser.add_argument("--wait-final", action="store_true")
    parser.add_argument("--poll-s", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.wait_final:
        wait_for_final_checkpoint(
            args.weights_dir,
            poll_s=args.poll_s,
            min_bytes=args.min_bytes,
            settle_s=args.settle_s,
        )
    jobs = enqueue_b0_eval_jobs(
        weights_dir=args.weights_dir,
        queue_dir=args.queue_dir,
        results_root=args.results_root,
        ann=args.ann,
        openfly_root=args.openfly_root,
        task=args.task,
        min_bytes=args.min_bytes,
        settle_s=args.settle_s,
    )
    print(f"enqueued={len(jobs)}")
    for job in jobs:
        print(job["id"], job["checkpoint"], job["out_metrics"])


if __name__ == "__main__":
    main()
