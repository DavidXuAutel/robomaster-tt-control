from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, Sequence

import numpy as np

from experiments.aerial.eval.metrics import OPENFLY_SUCCESS_DIST_M, compute_sr_ne_spl
from experiments.aerial.openfly_actions import primitive_to_delta, wrap_angle

REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_root() -> Path:
    return REPO_ROOT


def load_annotation(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"annotation must be a JSON list, got {type(data).__name__}")
    return data


def apply_body_delta(
    pos: np.ndarray,
    yaw: float,
    delta: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Apply body-frame (dx, dy, dz, dyaw) to world pose."""
    dx, dy, dz, dyaw = (float(v) for v in np.asarray(delta, dtype=np.float64).reshape(4))
    c, s = math.cos(yaw), math.sin(yaw)
    wx = c * dx - s * dy
    wy = s * dx + c * dy
    wz = dz
    new_pos = np.asarray(pos, dtype=np.float64).reshape(3) + np.array([wx, wy, wz], dtype=np.float64)
    new_yaw = wrap_angle(yaw + dyaw)
    return new_pos, new_yaw


def expert_path_length(positions: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(positions, dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    diffs = pts[1:] - pts[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())


def trajectory_path_length(positions: Sequence[np.ndarray]) -> float:
    if len(positions) < 2:
        return 0.0
    total = 0.0
    for idx in range(1, len(positions)):
        total += float(np.linalg.norm(positions[idx] - positions[idx - 1]))
    return total


class Bridge(Protocol):
    def reset(self, episode: dict[str, Any]) -> None: ...

    def render(self) -> np.ndarray: ...

    def state(self) -> np.ndarray: ...

    def step(self, primitive_id: int) -> None: ...

    def close(self) -> None: ...


class MockBridge:
    """Kinematic bridge using OpenFly primitive deltas (no AirSim)."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = int(seed)
        self._pos = np.zeros(3, dtype=np.float64)
        self._yaw = 0.0
        self._goal_pos = np.zeros(3, dtype=np.float64)
        self._instruction = ""

    def reset(self, episode: dict[str, Any]) -> None:
        positions = np.asarray(episode["pos"], dtype=np.float64)
        yaws = np.asarray(episode["yaw"], dtype=np.float64).reshape(-1)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"episode pos must be (N, 3), got {positions.shape}")
        self._pos = positions[0].copy()
        self._yaw = float(yaws[0])
        self._goal_pos = positions[-1].copy()
        self._instruction = str(episode.get("gpt_instruction", ""))

    def render(self) -> np.ndarray:
        # Deterministic pseudo-RGB from pose (no image files required offline).
        x, y, z = self._pos
        r = int((math.sin(x * 0.1 + self._seed) * 0.5 + 0.5) * 255) % 256
        g = int((math.sin(y * 0.1 + self._seed) * 0.5 + 0.5) * 255) % 256
        b = int((math.sin(z * 0.1 + self._seed) * 0.5 + 0.5) * 255) % 256
        return np.full((64, 64, 3), [r, g, b], dtype=np.uint8)

    def state(self) -> np.ndarray:
        return np.array([self._pos[0], self._pos[1], self._pos[2], self._yaw], dtype=np.float32)

    def step(self, primitive_id: int) -> None:
        if int(primitive_id) == 0:
            return
        delta = primitive_to_delta(int(primitive_id))
        self._pos, self._yaw = apply_body_delta(self._pos, self._yaw, delta)

    def close(self) -> None:
        return


class OpenFlyBridge:
    """Thin wrapper around OpenFly AirsimBridge (Linux + AirSim required)."""

    def __init__(self, openfly_root: Path, env_name: str) -> None:
        self._openfly_root = openfly_root.resolve()
        self._env_name = env_name
        self._bridge: Any = None
        self._pos = np.zeros(3, dtype=np.float64)
        self._yaw = 0.0

    def _ensure_bridge(self) -> Any:
        if self._bridge is not None:
            return self._bridge
        sim_dir = self._openfly_root / "scripts" / "sim"
        if not sim_dir.is_dir():
            raise FileNotFoundError(
                f"OpenFly scripts/sim not found under {self._openfly_root}. "
                "Clone https://github.com/SHAILAB-IPEC/OpenFly-Platform.git"
            )
        if str(sim_dir) not in sys.path:
            sys.path.insert(0, str(sim_dir))
        try:
            from airsim_bridge import AirsimBridge  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Failed to import OpenFly AirsimBridge. Install OpenFly deps "
                "(airsim, opencv-python) on a Linux GPU host."
            ) from exc
        self._bridge = AirsimBridge(self._env_name)
        return self._bridge

    def reset(self, episode: dict[str, Any]) -> None:
        positions = np.asarray(episode["pos"], dtype=np.float64)
        yaws = np.asarray(episode["yaw"], dtype=np.float64).reshape(-1)
        self._pos = positions[0].copy()
        self._yaw = float(yaws[0])
        bridge = self._ensure_bridge()
        bridge.set_drone_pos(
            float(self._pos[0]),
            float(self._pos[1]),
            float(self._pos[2]),
            0.0,
            math.degrees(self._yaw),
            0.0,
        )

    def render(self) -> np.ndarray:
        bridge = self._ensure_bridge()
        bgr = bridge.get_camera_data("color")
        rgb = np.asarray(bgr[..., ::-1], dtype=np.uint8)
        return rgb

    def state(self) -> np.ndarray:
        return np.array([self._pos[0], self._pos[1], self._pos[2], self._yaw], dtype=np.float32)

    def step(self, primitive_id: int) -> None:
        if int(primitive_id) == 0:
            return
        delta = primitive_to_delta(int(primitive_id))
        self._pos, self._yaw = apply_body_delta(self._pos, self._yaw, delta)
        bridge = self._ensure_bridge()
        bridge.set_drone_pos(
            float(self._pos[0]),
            float(self._pos[1]),
            float(self._pos[2]),
            0.0,
            math.degrees(self._yaw),
            0.0,
        )

    def close(self) -> None:
        self._bridge = None


class Policy(Protocol):
    def predict_primitive(
        self,
        obs_rgb: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> int: ...


class ReplayPolicy:
    """Replay expert primitive ids from annotation (offline smoke / wiring)."""

    def __init__(self, actions: Sequence[int]) -> None:
        self._actions = [int(a) for a in actions]
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def predict_primitive(
        self,
        obs_rgb: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> int:
        del obs_rgb, state, instruction
        if self._cursor >= len(self._actions):
            return 0
        action = self._actions[self._cursor]
        self._cursor += 1
        return int(action)


def _episode_env_name(episode: dict[str, Any]) -> str:
    image_path = str(episode.get("image_path", ""))
    if not image_path:
        return "env_airsim_16"
    first = image_path.split("/")[0]
    if first.startswith("env_"):
        return first
    scene_id = str(episode.get("scene_id", episode.get("scene", "")))
    if scene_id.startswith("env_"):
        return scene_id
    return "env_airsim_16"


@dataclass
class EpisodeResult:
    success: bool
    path_length: float
    shortest_length: float
    navigation_error: float
    steps: int


def run_episode(
    bridge: Bridge,
    policy: Policy,
    episode: dict[str, Any],
    *,
    max_steps: int,
) -> EpisodeResult:
    positions = np.asarray(episode["pos"], dtype=np.float64)
    goal_pos = positions[-1]
    shortest_length = expert_path_length(positions)

    bridge.reset(episode)
    if hasattr(policy, "reset"):
        policy.reset()

    instruction = str(episode.get("gpt_instruction", ""))
    visited = [bridge.state()[:3].astype(np.float64).copy()]
    steps = 0
    primitive = -1

    while steps < max_steps:
        rgb = bridge.render()
        state = bridge.state()
        primitive = int(policy.predict_primitive(rgb, state, instruction))
        if primitive == 0:
            break
        bridge.step(primitive)
        visited.append(bridge.state()[:3].astype(np.float64).copy())
        steps += 1

    final_pos = bridge.state()[:3].astype(np.float64)
    navigation_error = float(np.linalg.norm(final_pos - goal_pos))
    success = navigation_error < OPENFLY_SUCCESS_DIST_M
    path_length = trajectory_path_length(visited)

    return EpisodeResult(
        success=success,
        path_length=path_length,
        shortest_length=shortest_length,
        navigation_error=navigation_error,
        steps=steps if primitive != 0 else steps,
    )


def build_policy(
    policy_name: str,
    episode: dict[str, Any],
    *,
    checkpoint: Optional[Path] = None,
    task: str = "aerial_joint_1cam_1e-4",
    device: Optional[str] = None,
    seed: Optional[int] = None,
) -> Policy:
    if policy_name == "replay":
        actions = [int(a) for a in episode.get("action", [])]
        return ReplayPolicy(actions)

    if policy_name != "fastwam":
        raise ValueError(f"unknown policy {policy_name!r}; expected replay or fastwam")

    if checkpoint is None:
        raise ValueError("--checkpoint is required when --policy fastwam")

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from experiments.aerial.eval.policy_fastwam import FastWAMAerialPolicy
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    cfg_dir = str((_repo_root() / "configs").resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"task={task}"])

    eval_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.model, model_dtype=torch.float32, device=eval_device)
    model.load_checkpoint(str(checkpoint))
    model = model.to(eval_device).eval()

    stats_candidates = [
        checkpoint.parent / "dataset_stats.json",
        checkpoint.parent.parent / "dataset_stats.json",
    ]
    stats_path = next((p for p in stats_candidates if p.is_file()), None)
    processor = None
    if stats_path is not None:
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))

    action_horizon = int(cfg.data.train.num_frames) - 1
    return FastWAMAerialPolicy(
        model=model,
        processor=processor,
        action_horizon=action_horizon,
        replan_steps=1,
        seed=seed,
        rand_device=eval_device,
    )


def build_bridge(
    bridge_name: str,
    *,
    openfly_root: Optional[Path] = None,
    env_name: Optional[str] = None,
    seed: int = 0,
) -> Bridge:
    if bridge_name == "mock":
        return MockBridge(seed=seed)
    if bridge_name != "openfly":
        raise ValueError(f"unknown bridge {bridge_name!r}; expected mock or openfly")
    if openfly_root is None:
        raise ValueError("--openfly-root is required when --bridge openfly")
    return OpenFlyBridge(openfly_root, env_name or "env_airsim_16")


def _episode_id(episode: dict[str, Any], index: int) -> str:
    for key in ("route_id", "episode_id", "id"):
        if key in episode and episode[key] is not None:
            return str(episode[key])
    return str(index)


def evaluate_episodes(
    episodes: Sequence[dict[str, Any]],
    *,
    bridge_name: str,
    policy_name: str,
    max_steps: int,
    checkpoint: Optional[Path] = None,
    openfly_root: Optional[Path] = None,
    seed: int = 0,
    task: str = "aerial_joint_1cam_1e-4",
) -> dict[str, Any]:
    successes: list[bool] = []
    path_lengths: list[float] = []
    shortest_lengths: list[float] = []
    nes: list[float] = []
    episode_records: list[dict[str, Any]] = []

    for index, episode in enumerate(episodes):
        env_name = _episode_env_name(episode)
        bridge = build_bridge(
            bridge_name,
            openfly_root=openfly_root,
            env_name=env_name,
            seed=seed,
        )
        policy = build_policy(
            policy_name,
            episode,
            checkpoint=checkpoint,
            task=task,
            seed=seed,
        )
        try:
            result = run_episode(bridge, policy, episode, max_steps=max_steps)
        finally:
            bridge.close()

        successes.append(result.success)
        path_lengths.append(result.path_length)
        shortest_lengths.append(result.shortest_length)
        nes.append(result.navigation_error)
        episode_records.append(
            {
                "episode_id": _episode_id(episode, index),
                "success": result.success,
                "NE": result.navigation_error,
                "path_length": result.path_length,
                "shortest_length": result.shortest_length,
                "steps": result.steps,
            }
        )

    metrics = compute_sr_ne_spl(successes, path_lengths, shortest_lengths, nes)
    metrics["n"] = float(len(episodes))
    metrics["episodes"] = episode_records
    return metrics


def write_metrics(metrics: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "episodes":
            payload[key] = value
        else:
            payload[key] = float(value)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenFly closed-loop eval runner (M1a)")
    parser.add_argument("--ann", type=Path, required=True, help="OpenFly annotation JSON (e.g. seen.json)")
    parser.add_argument("--out", type=Path, required=True, help="Output metrics JSON path")
    parser.add_argument("--bridge", choices=("mock", "openfly"), default="mock")
    parser.add_argument("--policy", choices=("replay", "fastwam"), default="replay")
    parser.add_argument("--openfly-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="aerial_joint_1cam_1e-4")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    episodes = load_annotation(args.ann)[: max(0, int(args.max_episodes))]
    if not episodes:
        raise ValueError(f"no episodes found in {args.ann}")

    metrics = evaluate_episodes(
        episodes,
        bridge_name=args.bridge,
        policy_name=args.policy,
        max_steps=int(args.max_steps),
        checkpoint=args.checkpoint,
        openfly_root=args.openfly_root,
        seed=int(args.seed),
        task=str(args.task),
    )
    write_metrics(metrics, args.out)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
