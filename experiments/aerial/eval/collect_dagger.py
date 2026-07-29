from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

from experiments.aerial.eval.run_closed_loop import (
    Bridge,
    Policy,
    _episode_env_name,
    build_bridge,
    build_policy,
    load_annotation,
)
from experiments.aerial.openfly_actions import (
    clip_body_delta,
    delta_to_nearest_primitive,
    primitive_to_delta,
)
from experiments.aerial.path_expert import PathExpert
from experiments.aerial.takeover import TakeoverConfig, TakeoverController


@dataclass(frozen=True)
class CollectionResult:
    status: str
    reason: str
    frames: int


def _predict_delta(
    policy: Policy,
    rgb: np.ndarray,
    state: np.ndarray,
    instruction: str,
) -> np.ndarray:
    predict_delta = getattr(policy, "predict_delta", None)
    if callable(predict_delta):
        raw_action = predict_delta(rgb, state, instruction)
    else:
        raw_action = primitive_to_delta(
            int(policy.predict_primitive(rgb, state, instruction))
        )
    action = clip_body_delta(np.asarray(raw_action, dtype=np.float64))
    if not np.isfinite(action).all():
        raise ValueError("policy action must contain only finite values")
    return action


def _execute_delta(bridge: Bridge, action: np.ndarray) -> None:
    step_delta = getattr(bridge, "step_delta", None)
    if callable(step_delta):
        step_delta(action)
    else:
        bridge.step(delta_to_nearest_primitive(action))


def _write_jsonl_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _remove_truncated_images(
    out_path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    for record in records:
        image_path = out_path.parent / str(record["observation.images.ego"])
        image_path.unlink(missing_ok=True)


def collect_episode(
    bridge: Bridge,
    policy: Policy,
    expert: PathExpert,
    controller: TakeoverController,
    episode: dict[str, Any],
    out_path: Path,
    *,
    max_steps: int,
    abort_tail_frames: int = 0,
) -> CollectionResult:
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if abort_tail_frames < 0:
        raise ValueError("abort_tail_frames must be non-negative")

    bridge.reset(episode)
    expert.reset(episode)
    reset_policy = getattr(policy, "reset", None)
    if callable(reset_policy):
        reset_policy()

    instruction = str(episode.get("gpt_instruction", "")).strip()
    if not instruction:
        raise ValueError("episode gpt_instruction must be non-empty")

    image_dir = out_path.parent / f"{out_path.stem}_frames"
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    status = "completed"
    reason = "max_steps"

    for step in range(max_steps):
        rgb = np.asarray(bridge.render(), dtype=np.uint8)
        state = np.asarray(bridge.state(), dtype=np.float64).reshape(4)
        label = expert.label(state[:3], float(state[3]))
        policy_delta = _predict_delta(policy, rgb, state, instruction)
        decision = controller.step(label.cross_track_m, label.progress_m)

        if decision.mode == "abort":
            status = "failed"
            reason = decision.reason
            if abort_tail_frames:
                removed = records[-abort_tail_frames:]
                _remove_truncated_images(out_path, removed)
                del records[-abort_tail_frames:]
            break

        expert_action = np.asarray(label.action, dtype=np.float64).reshape(4)
        if not np.isfinite(expert_action).all():
            raise ValueError("expert action must contain only finite values")
        executed = expert_action if decision.mode == "expert" else policy_delta

        image_path = image_dir / f"frame{step:06d}.png"
        Image.fromarray(rgb, mode="RGB").save(image_path)
        records.append(
            {
                "observation.images.ego": str(image_path.relative_to(out_path.parent)),
                "observation.state": state.astype(np.float32).tolist(),
                "action": expert_action.astype(np.float32).tolist(),
                "task": instruction,
                "intervention": bool(decision.intervene),
                "executed_action": executed.astype(np.float32).tolist(),
                "mode": decision.mode,
                "reason": decision.reason,
                "cross_track_m": float(label.cross_track_m),
                "progress_m": float(label.progress_m),
                "meta.action_source": "pos_delta_v1",
            }
        )
        _execute_delta(bridge, executed)

    _write_jsonl_atomic(out_path, records)
    return CollectionResult(status=status, reason=reason, frames=len(records))


def _write_manifest_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def collect_episodes(
    episodes: Sequence[dict[str, Any]],
    out_dir: Path,
    *,
    bridge_name: str,
    policy_name: str,
    config: TakeoverConfig,
    max_steps: int,
    abort_tail_frames: int = 0,
    checkpoint: Optional[Path] = None,
    openfly_root: Optional[Path] = None,
    seed: int = 0,
    task: str = "aerial_joint_1cam_1e-4",
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "completed": [],
        "failed": [],
        "thresholds": asdict(config),
    }
    manifest_path = out_dir / "manifest.json"

    for index, episode in enumerate(episodes):
        episode_name = f"ep{index:03d}"
        bridge: Optional[Bridge] = None
        try:
            bridge = build_bridge(
                bridge_name,
                openfly_root=openfly_root,
                env_name=_episode_env_name(episode),
                seed=seed,
            )
            policy = build_policy(
                policy_name,
                episode,
                checkpoint=checkpoint,
                task=task,
                seed=seed,
            )
            result = collect_episode(
                bridge,
                policy,
                PathExpert(),
                TakeoverController(config),
                episode,
                out_dir / f"{episode_name}.jsonl",
                max_steps=max_steps,
                abort_tail_frames=abort_tail_frames,
            )
            destination = "completed" if result.status == "completed" else "failed"
            manifest[destination].append(
                {
                    "episode": episode_name,
                    "reason": result.reason,
                    "frames": result.frames,
                }
            )
        except Exception as exc:
            manifest["failed"].append(
                {"episode": episode_name, "reason": str(exc), "frames": 0}
            )
        finally:
            if bridge is not None:
                bridge.close()
        _write_manifest_atomic(manifest_path, manifest)

    return manifest


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DAgger correction episodes.")
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bridge", choices=("mock", "openfly"), default="mock")
    parser.add_argument("--policy", choices=("replay", "fastwam"), default="replay")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--openfly-root", type=Path)
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--abort-tail-frames", type=int, default=0)
    parser.add_argument("--takeover-m", type=float, required=True)
    parser.add_argument("--release-m", type=float, required=True)
    parser.add_argument("--abort-m", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="aerial_joint_1cam_1e-4")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    episodes = load_annotation(args.ann)[: max(0, int(args.max_episodes))]
    if not episodes:
        raise ValueError(f"no episodes found in {args.ann}")
    config = TakeoverConfig(
        takeover_m=float(args.takeover_m),
        release_m=float(args.release_m),
        abort_m=float(args.abort_m),
    )
    collect_episodes(
        episodes,
        args.out,
        bridge_name=args.bridge,
        policy_name=args.policy,
        config=config,
        max_steps=int(args.max_steps),
        abort_tail_frames=int(args.abort_tail_frames),
        checkpoint=args.checkpoint,
        openfly_root=args.openfly_root,
        seed=int(args.seed),
        task=str(args.task),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
