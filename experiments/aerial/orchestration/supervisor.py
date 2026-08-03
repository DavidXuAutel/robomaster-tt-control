from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from experiments.aerial.eval.lock_baseline import (
    build_lock_manifest,
    select_baseline,
    validate_candidates,
    write_lock_manifest,
)
from experiments.aerial.orchestration.b0_discover import B0_STEPS, default_metrics_path
from experiments.aerial.orchestration.b1_discover import B1_STEPS, default_b1_metrics_path
from experiments.aerial.orchestration.eval_queue import metrics_valid
from experiments.aerial.orchestration.gates import evaluate_b1_gates, queue_is_idle
from experiments.aerial.orchestration.state import Phase, read_status, write_status


def advance_phase(
    status: dict,
    *,
    queue_dir: Optional[Path] = None,
    b1_results_root: Optional[Path] = None,
    stamp: Optional[str] = None,
) -> dict:
    """Pure helper: compute next phase payload from current status + queue/metrics."""
    phase = status.get("phase", Phase.WAIT_B0_COMPLETE.value)
    if phase == Phase.EVAL_B0_CHECKPOINTS.value:
        if queue_dir is None or not queue_is_idle(queue_dir):
            return status
        return {**status, "phase": Phase.LOCK_BASELINE.value}
    if phase == Phase.LOCK_BASELINE.value:
        return {**status, "phase": Phase.B1_GATES.value}
    if phase == Phase.B1_GATES.value and status.get("gates_passed"):
        return {**status, "phase": Phase.RUN_B1_TRAIN.value}
    if phase == Phase.RUN_B1_TRAIN.value and status.get("b1_train_started"):
        return {**status, "phase": Phase.EVAL_B1_CHECKPOINTS.value}
    if phase == Phase.EVAL_B1_CHECKPOINTS.value:
        effective_stamp = stamp or str(status.get("stamp", ""))
        if b1_results_root is None or not effective_stamp:
            return status
        if not b1_metrics_ready(results_root=b1_results_root, stamp=effective_stamp):
            return status
        return {**status, "phase": Phase.S1_REPORT.value}
    if phase == Phase.S1_REPORT.value and status.get("s1_report_written"):
        return {**status, "phase": Phase.DONE.value}
    return status


def b1_metrics_ready(
    *,
    results_root: Path,
    stamp: str,
    steps: Sequence[int] = B1_STEPS,
) -> bool:
    return all(
        metrics_valid(default_b1_metrics_path(results_root, stamp, int(step)))
        for step in steps
    )


def _read_sha256_sidecar(checkpoint: Path) -> str:
    sidecar = Path(str(checkpoint) + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing sha256 sidecar: {sidecar}")
    text = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if len(text) != 64:
        raise ValueError(f"malformed sha256 sidecar: {sidecar}")
    return text.lower()


def build_lock_candidates(
    *,
    weights_dir: Path,
    results_root: Path,
    steps: Sequence[int] = B0_STEPS,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for step in steps:
        checkpoint = weights_dir / f"step_{int(step):06d}.pt"
        metrics_path = default_metrics_path(results_root, int(step))
        if not checkpoint.is_file():
            raise ValueError(f"missing checkpoint for step {step}: {checkpoint}")
        if not metrics_path.is_file():
            raise ValueError(f"missing metrics for step {step}: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        candidates.append(
            {
                "step": int(step),
                "checkpoint": str(checkpoint.resolve()),
                "metrics_path": str(metrics_path.resolve()),
                "mean_ne": float(metrics["NE"]),
                "sha256": _read_sha256_sidecar(checkpoint),
            }
        )
    return candidates


def lock_baseline_from_results(
    *,
    stamp: str,
    weights_dir: Path,
    results_root: Path,
    out: Path,
    steps: Sequence[int] = B0_STEPS,
) -> dict[str, Any]:
    candidates = validate_candidates(
        build_lock_candidates(
            weights_dir=weights_dir,
            results_root=results_root,
            steps=steps,
        )
    )
    chosen = select_baseline(candidates)
    manifest = build_lock_manifest(chosen, candidates=candidates, stamp=stamp)
    write_lock_manifest(out, manifest)
    return manifest


def _parse_steps(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --steps")
    return tuple(int(p) for p in parts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B0→B1 orchestration supervisor helpers")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--stamp", required=True)
    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--init", action="store_true")
    sub.add_argument("--set-phase", choices=[p.value for p in Phase])
    sub.add_argument("--advance-from-eval-queue", action="store_true")
    sub.add_argument("--advance-b1-eval", action="store_true")
    sub.add_argument("--mark-b1-train-started", action="store_true")
    sub.add_argument("--mark-s1-report", action="store_true")
    sub.add_argument("--lock-baseline", action="store_true")
    sub.add_argument("--run-b1-gates", action="store_true")
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--steps", default="1000,2000,3000,4000,5000")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--s1-pass", choices=["true", "false"])
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument("--collection-source", type=Path)
    parser.add_argument("--heldout-ann", type=Path)
    parser.add_argument("--oracle-json", type=Path)
    parser.add_argument("--correction-root", type=Path)
    parser.add_argument("--ft-manifest", type=Path)
    parser.add_argument("--smoke-status", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.init:
        write_status(
            args.status,
            {"phase": Phase.WAIT_B0_COMPLETE.value, "stamp": args.stamp},
        )
        print(Phase.WAIT_B0_COMPLETE.value)
        return
    if args.set_phase is not None:
        payload = read_status(args.status)
        payload["phase"] = args.set_phase
        payload["stamp"] = args.stamp
        write_status(args.status, payload)
        print(args.set_phase)
        return
    if args.advance_from_eval_queue:
        if args.queue_dir is None:
            raise SystemExit("--queue-dir required")
        status = read_status(args.status)
        status.setdefault("stamp", args.stamp)
        status.setdefault("phase", Phase.EVAL_B0_CHECKPOINTS.value)
        next_status = advance_phase(status, queue_dir=args.queue_dir)
        write_status(args.status, next_status)
        print(next_status["phase"])
        return
    if args.mark_b1_train_started:
        status = read_status(args.status)
        status["stamp"] = args.stamp
        status["b1_train_started"] = True
        status["phase"] = Phase.RUN_B1_TRAIN.value
        next_status = advance_phase(status)
        write_status(args.status, next_status)
        print(next_status["phase"])
        return
    if args.advance_b1_eval:
        if args.results_root is None:
            raise SystemExit("--results-root required")
        status = read_status(args.status)
        status.setdefault("stamp", args.stamp)
        status.setdefault("phase", Phase.EVAL_B1_CHECKPOINTS.value)
        next_status = advance_phase(
            status,
            b1_results_root=args.results_root,
            stamp=args.stamp,
        )
        write_status(args.status, next_status)
        print(next_status["phase"])
        return
    if args.mark_s1_report:
        if args.s1_pass is None:
            raise SystemExit("--s1-pass required")
        status = read_status(args.status)
        status.update(
            {
                "stamp": args.stamp,
                "phase": Phase.S1_REPORT.value,
                "s1_report_written": True,
                "s1_pass": args.s1_pass == "true",
            }
        )
        if args.report_path is not None:
            status["s1_report_path"] = str(args.report_path)
        next_status = advance_phase(status)
        write_status(args.status, next_status)
        print(next_status["phase"])
        return
    if args.lock_baseline:
        if args.weights_dir is None or args.results_root is None or args.out is None:
            raise SystemExit("--weights-dir, --results-root, and --out required")
        try:
            steps = _parse_steps(args.steps)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        try:
            manifest = lock_baseline_from_results(
                stamp=args.stamp,
                weights_dir=args.weights_dir,
                results_root=args.results_root,
                out=args.out,
                steps=steps,
            )
        except ValueError as exc:
            blocked = {
                "phase": Phase.BLOCKED.value,
                "stamp": args.stamp,
                "failed_gate": "baseline_lock",
                "reason": str(exc),
            }
            write_status(args.status, blocked)
            print(Phase.BLOCKED.value)
            raise SystemExit(2) from exc
        status = read_status(args.status)
        status.update(
            {
                "phase": Phase.B1_GATES.value,
                "stamp": args.stamp,
                "baseline_checkpoint": manifest["checkpoint"],
                "baseline_mean_ne": manifest["baseline_mean_ne"],
                "s1_ne": manifest["s1_ne"],
            }
        )
        write_status(args.status, status)
        print(Phase.B1_GATES.value)
        return
    required = [
        args.lock_path,
        args.collection_source,
        args.heldout_ann,
        args.oracle_json,
        args.correction_root,
        args.ft_manifest,
        args.smoke_status,
    ]
    if any(v is None for v in required):
        raise SystemExit("B1 gate paths are required")
    payload = evaluate_b1_gates(
        stamp=args.stamp,
        lock_path=args.lock_path,
        collection_source=args.collection_source,
        heldout_ann=args.heldout_ann,
        oracle_json=args.oracle_json,
        correction_root=args.correction_root,
        ft_manifest=args.ft_manifest,
        smoke_status=args.smoke_status,
    )
    write_status(args.status, payload)
    print(payload["phase"])
    if payload["phase"] == Phase.BLOCKED.value:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
