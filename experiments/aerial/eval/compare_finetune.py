from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

FLAT_NE_TOLERANCE = 1e-9
FAILURE_BINS = [
    "improved_but_below_s1_margin",
    "flat",
    "regressed",
    "quantization_gap",
]


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _episode_nes(metrics: Mapping[str, Any]) -> dict[str, float]:
    raw_episodes = metrics.get("episodes", [])
    if raw_episodes is None:
        return {}
    if not isinstance(raw_episodes, list):
        raise ValueError("metrics 'episodes' must be a list")

    result: dict[str, float] = {}
    for index, episode in enumerate(raw_episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"episodes[{index}] must be an object")
        episode_id = episode.get("episode_id", episode.get("id", episode.get("route_id")))
        if episode_id is None:
            episode_id = str(index)
        ne = episode.get("NE", episode.get("navigation_error"))
        if ne is None:
            raise ValueError(f"episode {episode_id!r} has no NE/navigation_error")
        key = str(episode_id)
        if key in result:
            raise ValueError(f"duplicate episode id {key!r}")
        result[key] = _finite_float(ne, f"episode {key} NE")
    return result


def _aggregate_run(metrics: Mapping[str, Any]) -> dict[str, Any]:
    episodes = _episode_nes(metrics)
    if "NE" in metrics:
        mean_ne = _finite_float(metrics["NE"], "NE")
    elif episodes:
        mean_ne = statistics.fmean(episodes.values())
    else:
        raise ValueError("metrics must contain NE or non-empty episodes")

    summary: dict[str, Any] = {
        "mean_NE": mean_ne,
        "median_NE": statistics.median(episodes.values()) if episodes else mean_ne,
        "SR": _finite_float(metrics["SR"], "SR") if "SR" in metrics else None,
        "SPL": _finite_float(metrics["SPL"], "SPL") if "SPL" in metrics else None,
    }
    return summary


def _quantization_stats(metrics: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    values = metrics.get("quantization_gap_l2")
    if values is None:
        return None
    if isinstance(values, Mapping):
        return dict(values)
    if not isinstance(values, list):
        raise ValueError("quantization_gap_l2 must be a list or stats object")
    finite = [
        _finite_float(value, f"quantization_gap_l2[{index}]")
        for index, value in enumerate(values)
    ]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "max": max(finite),
    }


def _diagnosis(best_mean_ne: float, s1_ne: float) -> dict[str, Any]:
    return {
        "best_mean_NE": best_mean_ne,
        "s1_threshold_NE": s1_ne,
        "failure_bins": list(FAILURE_BINS),
        "instructions": (
            "Populate each failure bin from held-out episode deltas and quantization "
            "gap evidence before proposing another experiment."
        ),
        "auto_expand_data": False,
        "start_unseen": False,
    }


def compare_metrics(
    baseline: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    baseline_mean_ne: float,
    s1_ne: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")

    locked_baseline_ne = _finite_float(baseline_mean_ne, "baseline_mean_ne")
    locked_s1_ne = _finite_float(s1_ne, "s1_ne")
    baseline_summary = _aggregate_run(baseline)
    baseline_episodes = _episode_nes(baseline)
    candidate_reports: dict[str, dict[str, Any]] = {}

    for step, metrics in candidates.items():
        step_key = str(step)
        summary = _aggregate_run(metrics)
        candidate_episodes = _episode_nes(metrics)
        shared_ids = sorted(set(baseline_episodes) & set(candidate_episodes))
        deltas = {
            episode_id: candidate_episodes[episode_id] - baseline_episodes[episode_id]
            for episode_id in shared_ids
        }
        counts = {"improve": 0, "flat": 0, "regress": 0}
        for delta in deltas.values():
            if delta < -FLAT_NE_TOLERANCE:
                counts["improve"] += 1
            elif delta > FLAT_NE_TOLERANCE:
                counts["regress"] += 1
            else:
                counts["flat"] += 1

        summary["per_episode_deltas"] = deltas
        summary["delta_counts"] = counts
        quantization = _quantization_stats(metrics)
        if quantization is not None:
            summary["quantization_gap_l2"] = quantization
        candidate_reports[step_key] = summary

    best_step = min(
        candidate_reports,
        key=lambda step: (candidate_reports[step]["mean_NE"], _step_sort_key(step)),
    )
    best_mean_ne = candidate_reports[best_step]["mean_NE"]
    passed = best_mean_ne <= locked_s1_ne
    report: dict[str, Any] = {
        "locked_baseline_NE": locked_baseline_ne,
        "s1_threshold_NE": locked_s1_ne,
        "baseline": baseline_summary,
        "candidates": candidate_reports,
        "best_step": best_step,
        "best_mean_NE": best_mean_ne,
        "s1_pass": passed,
    }
    if not passed:
        report["diagnosis"] = _diagnosis(best_mean_ne, locked_s1_ne)
    return report


def _step_sort_key(step: str) -> tuple[int, Any]:
    try:
        return (0, int(step))
    except ValueError:
        return (1, step)


def summarize(
    *,
    baseline_ne: float,
    cand: Mapping[str, float],
    s1_ne: float,
) -> dict[str, Any]:
    locked_baseline_ne = _finite_float(baseline_ne, "baseline_ne")
    baseline = {"NE": locked_baseline_ne}
    candidates = {
        str(step): {"NE": _finite_float(ne, f"candidate {step} NE")}
        for step, ne in cand.items()
    }
    return compare_metrics(
        baseline,
        candidates,
        baseline_mean_ne=locked_baseline_ne,
        s1_ne=s1_ne,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _lock_thresholds(manifest: Mapping[str, Any]) -> tuple[float, float]:
    values: dict[str, float] = {}
    for key in ("baseline_mean_ne", "s1_ne"):
        if key not in manifest:
            raise ValueError(f"lock manifest missing required field {key!r}")
        value = manifest[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"lock manifest field {key!r} must be a number")
        try:
            values[key] = _finite_float(value, f"lock manifest {key}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"lock manifest field {key!r} must be finite") from exc
    expected_s1_ne = 0.8 * values["baseline_mean_ne"]
    if not math.isclose(
        values["s1_ne"],
        expected_s1_ne,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "lock manifest s1_ne must equal 0.8 * baseline_mean_ne: "
            f"expected {expected_s1_ne}, got {values['s1_ne']}"
        )
    return values["baseline_mean_ne"], values["s1_ne"]


def _candidate_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be STEP=PATH")
    step, path = value.split("=", 1)
    if not step or not path:
        raise argparse.ArgumentTypeError("candidate must be STEP=PATH")
    return step, Path(path)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare held-out B0 fine-tune metrics")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--lock-manifest", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        type=_candidate_arg,
        metavar="STEP=PATH",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--diagnosis-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    lock_manifest = _read_json(args.lock_manifest)
    locked_baseline_ne, s1_ne = _lock_thresholds(lock_manifest)
    baseline = _read_json(args.baseline)
    baseline_ne = _aggregate_run(baseline)["mean_NE"]
    if not math.isclose(
        baseline_ne,
        locked_baseline_ne,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "locked baseline NE mismatch: "
            f"expected {locked_baseline_ne}, got {baseline_ne}"
        )

    candidates: dict[str, dict[str, Any]] = {}
    for step, path in args.candidate:
        if step in candidates:
            raise ValueError(f"duplicate candidate step {step!r}")
        candidates[step] = _read_json(path)

    report = compare_metrics(
        baseline,
        candidates,
        baseline_mean_ne=locked_baseline_ne,
        s1_ne=s1_ne,
    )
    _write_json(args.out, report)
    if not report["s1_pass"]:
        diagnosis_path = (
            args.diagnosis_out
            if args.diagnosis_out is not None
            else args.out.with_name("ft_s1_failure_diagnosis.json")
        )
        _write_json(diagnosis_path, report["diagnosis"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["s1_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
