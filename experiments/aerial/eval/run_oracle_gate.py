from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from experiments.aerial.eval.metrics import OPENFLY_SUCCESS_DIST_M
from experiments.aerial.eval.run_closed_loop import (
    Bridge,
    _episode_env_name,
    build_bridge,
    build_policy,
    load_annotation,
    normalize_episode_poses,
)
from experiments.aerial.openfly_actions import (
    delta_to_nearest_primitive,
    is_stop_delta,
)
from experiments.aerial.path_expert import PathExpert
from experiments.aerial.takeover import freeze_thresholds

ORACLE_MIN_SR = 0.80
ORACLE_MAX_MEDIAN_NE = 20.0


@dataclass(frozen=True)
class OracleEpisodeResult:
    success: bool
    navigation_error: float
    cross_tracks: tuple[float, ...]
    projection_failures: int
    steps: int


def oracle_gate_passes(
    sr: float,
    median_ne: float,
    projection_failures: int,
) -> bool:
    return (
        float(sr) >= ORACLE_MIN_SR
        and float(median_ne) < ORACLE_MAX_MEDIAN_NE
        and int(projection_failures) == 0
    )


def summarize_oracle_results(
    results: Sequence[OracleEpisodeResult],
    *,
    pilot_episodes: int,
) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one oracle result is required")
    pilot_count = min(max(0, int(pilot_episodes)), len(results))
    pilot_cross_tracks = [
        cross_track
        for result in results[:pilot_count]
        for cross_track in result.cross_tracks
    ]
    sr = sum(result.success for result in results) / len(results)
    median_ne = float(np.median([result.navigation_error for result in results]))
    projection_failures = sum(result.projection_failures for result in results)
    cross_track_p95 = (
        float(np.percentile(pilot_cross_tracks, 95)) if pilot_cross_tracks else 0.0
    )
    return {
        "SR": float(sr),
        "median_NE": median_ne,
        "projection_failures": int(projection_failures),
        "cross_track_p95": cross_track_p95,
        "n": len(results),
        "pilot_episodes": pilot_count,
        "passed": oracle_gate_passes(sr, median_ne, projection_failures),
    }


def run_oracle_episode(
    bridge: Bridge,
    expert: PathExpert,
    episode: dict[str, Any],
    *,
    max_steps: int,
    shadow_policy: Any = None,
) -> OracleEpisodeResult:
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    positions, _ = normalize_episode_poses(episode)
    goal = positions[-1]
    bridge.reset(episode)
    expert.reset(episode)
    if shadow_policy is not None:
        reset_shadow = getattr(shadow_policy, "reset", None)
        if callable(reset_shadow):
            reset_shadow()

    cross_tracks: list[float] = []
    projection_failures = 0
    steps = 0
    while steps < max_steps:
        state = np.asarray(bridge.state(), dtype=np.float64).reshape(4)
        if shadow_policy is not None:
            rgb = np.asarray(bridge.render(), dtype=np.uint8)
            instruction = str(episode.get("gpt_instruction", ""))
            predict_delta = getattr(shadow_policy, "predict_delta", None)
            if callable(predict_delta):
                predict_delta(rgb, state, instruction)
            else:
                shadow_policy.predict_primitive(rgb, state, instruction)
        try:
            label = expert.label(state[:3], float(state[3]))
        except (FloatingPointError, RuntimeError, ValueError):
            projection_failures += 1
            break
        action = np.asarray(label.action, dtype=np.float64).reshape(4)
        if not np.isfinite(action).all() or not np.isfinite(label.cross_track_m):
            projection_failures += 1
            break
        cross_tracks.append(float(label.cross_track_m))
        if is_stop_delta(action):
            break
        step_delta = getattr(bridge, "step_delta", None)
        if callable(step_delta):
            step_delta(action)
        else:
            bridge.step(delta_to_nearest_primitive(action))
        steps += 1

    final_pos = np.asarray(bridge.state(), dtype=np.float64).reshape(4)[:3]
    navigation_error = float(np.linalg.norm(final_pos - goal))
    return OracleEpisodeResult(
        success=navigation_error < OPENFLY_SUCCESS_DIST_M,
        navigation_error=navigation_error,
        cross_tracks=tuple(cross_tracks),
        projection_failures=projection_failures,
        steps=steps,
    )


def evaluate_oracle_episodes(
    episodes: Sequence[dict[str, Any]],
    *,
    bridge_name: str,
    max_steps: int,
    pilot_episodes: int,
    openfly_root: Optional[Path] = None,
    seed: int = 42,
    shadow_checkpoint: Optional[Path] = None,
    shadow_task: str = "aerial_joint_b0_novideo",
) -> dict[str, Any]:
    results: list[OracleEpisodeResult] = []
    shadow_policy = None
    if shadow_checkpoint is not None:
        if shadow_checkpoint.name != "step_004000.pt":
            raise ValueError("B0 shadow checkpoint must be named step_004000.pt")
        shadow_policy = build_policy(
            "fastwam",
            episodes[0],
            checkpoint=shadow_checkpoint,
            task=shadow_task,
            seed=seed,
        )
    for index, episode in enumerate(episodes):
        bridge = build_bridge(
            bridge_name,
            openfly_root=openfly_root,
            env_name=_episode_env_name(episode),
            seed=seed,
        )
        try:
            results.append(
                run_oracle_episode(
                    bridge,
                    PathExpert(),
                    episode,
                    max_steps=max_steps,
                    shadow_policy=(
                        shadow_policy if index < max(0, int(pilot_episodes)) else None
                    ),
                )
            )
        finally:
            bridge.close()
    return summarize_oracle_results(results, pilot_episodes=pilot_episodes)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_collection_manifest(
    path: Path,
    *,
    annotation_path: Path,
    report: dict[str, Any],
) -> None:
    existing: dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("collection manifest must be a JSON object")
        existing = loaded
    existing.update(
        {
            "collection_source": {
                "path": str(annotation_path),
                "sha256": _sha256(annotation_path),
            },
            "oracle_gate": report,
            "pilot": {
                "episodes": int(report["pilot_episodes"]),
                "cross_track_p95": float(report["cross_track_p95"]),
            },
            "thresholds": asdict(
                freeze_thresholds(float(report["cross_track_p95"]))
            ),
        }
    )
    _write_json_atomic(path, existing)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PathExpert-only collection oracle and freeze pilot thresholds."
    )
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("oracle_gate.json"))
    parser.add_argument(
        "--collection-manifest",
        type=Path,
        default=Path("collection_manifest.json"),
    )
    parser.add_argument("--bridge", choices=("mock", "openfly"), default="openfly")
    parser.add_argument("--openfly-root", type=Path)
    parser.add_argument("--max-episodes", type=int, default=40)
    parser.add_argument("--pilot-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shadow-checkpoint",
        type=Path,
        help="Optional B0 step_004000.pt; predictions run label-only.",
    )
    parser.add_argument("--shadow-task", default="aerial_joint_b0_novideo")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    episodes = load_annotation(args.ann)[: max(0, int(args.max_episodes))]
    if not episodes:
        raise ValueError(f"no episodes found in {args.ann}")
    report = evaluate_oracle_episodes(
        episodes,
        bridge_name=str(args.bridge),
        max_steps=int(args.max_steps),
        pilot_episodes=int(args.pilot_episodes),
        openfly_root=args.openfly_root,
        seed=int(args.seed),
        shadow_checkpoint=args.shadow_checkpoint,
        shadow_task=str(args.shadow_task),
    )
    _write_json_atomic(args.out, report)
    _write_collection_manifest(
        args.collection_manifest,
        annotation_path=args.ann,
        report=report,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
