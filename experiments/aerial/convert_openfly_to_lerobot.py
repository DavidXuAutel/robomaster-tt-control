from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from experiments.aerial.openfly_actions import (
    is_padding_action,
    pos_yaw_to_body_delta,
)


ACTION_SOURCE = "pos_delta_v1"
DEFAULT_FPS = 10


def _image_file(image_root: Path, image_path: str, frame_id: object) -> Path:
    base = image_root / image_path
    frame_name = str(frame_id)
    candidates = [base / frame_name]
    if not Path(frame_name).suffix:
        candidates.extend(
            base / f"{frame_name}{suffix}" for suffix in (".png", ".jpg", ".jpeg")
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"image for frame {frame_name!r} not found under {base}; "
        f"tried {[str(path) for path in candidates]}"
    )


def _normalize_pos_yaw(traj: dict) -> tuple[np.ndarray, np.ndarray]:
    """Normalize OpenFly pose rows to (N,3) xyz + (N,) yaw.

    Some OpenFly trajectories mix 3D ``[x,y,z]`` with 4D ``[x,y,z,yaw]`` pose
    rows in the same ``pos`` list. Prefer the embedded yaw when present.
    """
    raw_pos = list(traj["pos"])
    raw_yaw = list(traj["yaw"])
    if len(raw_pos) != len(raw_yaw):
        raise ValueError(
            f"pos and yaw length mismatch: {len(raw_pos)} vs {len(raw_yaw)}"
        )

    positions: list[list[float]] = []
    yaws: list[float] = []
    for index, row in enumerate(raw_pos):
        values = list(row)
        if len(values) == 3:
            positions.append([float(values[0]), float(values[1]), float(values[2])])
            yaws.append(float(raw_yaw[index]))
        elif len(values) == 4:
            positions.append([float(values[0]), float(values[1]), float(values[2])])
            yaws.append(float(values[3]))
        else:
            raise ValueError(
                f"pos[{index}] must have length 3 or 4, got {len(values)}: {values!r}"
            )

    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(yaws, dtype=np.float64).reshape(-1),
    )


def convert_trajectory(
    traj: dict,
    image_root: Path,
    *,
    action_source: str = ACTION_SOURCE,
    stop_relabel_radius: float | None = None,
) -> list[dict]:
    """Convert one OpenFly trajectory into aligned LeRobot frame records.

    When ``stop_relabel_radius`` is not None, any emitted frame whose *start*
    position lies within that many metres of the episode goal — plus the
    terminal frame unconditionally — has its action zeroed to ``[0,0,0,0]``.
    A zero body-delta is the OpenFly ``stop`` primitive (id 0), so this injects
    ``stop`` supervision through the existing CE label pipeline
    (``delta_nearest_with_dist`` maps zero delta → primitive 0) with no
    torch-side changes. The goal is the end position of the last non-padding
    transition, i.e. where the (successful) demo actually ends up. Default
    ``None`` is a no-op, preserving the raw-delta dataset.
    """
    if action_source != ACTION_SOURCE:
        raise ValueError(f"action_source must be {ACTION_SOURCE!r}, got {action_source!r}")
    if stop_relabel_radius is not None and float(stop_relabel_radius) < 0:
        raise ValueError("stop_relabel_radius must be non-negative")

    positions, yaws = _normalize_pos_yaw(traj)
    actions = np.asarray(traj["action"]).reshape(-1)
    frame_ids = list(traj["index_list"])
    task = str(traj["gpt_instruction"]).strip()
    image_path = str(traj["image_path"])

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"pos must have shape (N, 3), got {positions.shape}")
    frame_count = positions.shape[0]
    if len(yaws) != frame_count or len(frame_ids) != frame_count:
        raise ValueError("pos, yaw, and index_list must have the same length")
    if len(actions) < max(0, frame_count - 1):
        raise ValueError("action must contain an entry for every transition")
    if not task:
        raise ValueError("gpt_instruction must be non-empty")

    scene_id = str(traj.get("scene_id", traj.get("scene", "")))
    frames: list[dict] = []
    emitted_start_pos: list[np.ndarray] = []
    last_end_pos: np.ndarray | None = None
    for index in range(frame_count - 1):
        if is_padding_action(actions[index]):
            continue

        image_file = _image_file(Path(image_root), image_path, frame_ids[index])
        with Image.open(image_file) as source_image:
            image = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
        state = np.concatenate((positions[index], [yaws[index]])).astype(np.float32)
        action = pos_yaw_to_body_delta(
            positions[index],
            yaws[index],
            positions[index + 1],
            yaws[index + 1],
        ).astype(np.float32)
        frames.append(
            {
                "observation.images.ego": image,
                "observation.state": state,
                "action": action,
                "task": task,
                "meta.scene_id": scene_id,
                "meta.action_source": ACTION_SOURCE,
            }
        )
        emitted_start_pos.append(positions[index])
        last_end_pos = positions[index + 1]

    if stop_relabel_radius is not None and frames:
        radius = float(stop_relabel_radius)
        goal = np.asarray(last_end_pos, dtype=np.float64)
        for frame, start_pos in zip(frames, emitted_start_pos):
            dist = float(np.linalg.norm(np.asarray(start_pos, dtype=np.float64) - goal))
            if dist < radius:
                frame["action"] = np.zeros(4, dtype=np.float32)
        # Terminal frame always commands stop, even if it starts >= radius away.
        frames[-1]["action"] = np.zeros(4, dtype=np.float32)

    return frames


def write_lerobot_dataset(
    episodes: Iterable[list[dict]],
    out_root: Path,
    repo_id: str,
) -> None:
    """Write converted episodes using FastWAM's vendored LeRobot v2 writer."""
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    episode_list = [episode for episode in episodes if episode]
    if not episode_list:
        raise ValueError("episodes must contain at least one non-empty episode")

    sample_image = np.asarray(episode_list[0][0]["observation.images.ego"])
    if sample_image.ndim != 3 or sample_image.shape[2] != 3:
        raise ValueError(f"ego images must have shape (H, W, 3), got {sample_image.shape}")
    height, width, _ = sample_image.shape
    features = {
        "observation.images.ego": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["x", "y", "z", "yaw"],
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["dx", "dy", "dz", "dyaw"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=DEFAULT_FPS,
        features=features,
        root=Path(out_root),
        robot_type="aerial",
        use_videos=False,
        is_compute_episode_stats_image=True,
    )

    for episode in episode_list:
        for record in episode:
            task = str(record["task"])
            frame = {
                "observation.images.ego": np.asarray(
                    record["observation.images.ego"], dtype=np.uint8
                ),
                "observation.state": np.asarray(
                    record["observation.state"], dtype=np.float32
                ),
                "action": np.asarray(record["action"], dtype=np.float32),
            }

            frame_index = dataset.episode_buffer["size"]
            episode_index = dataset.episode_buffer["episode_index"]
            image_file = dataset._get_image_file_path(
                episode_index,
                "observation.images.ego",
                frame_index,
            )
            image_file.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame["observation.images.ego"], mode="RGB").save(image_file)
            dataset.add_frame(frame, task=[task, task, "successful", "successful"])
        dataset.save_episode()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OpenFly trajectories to LeRobot v2.")
    parser.add_argument("--ann", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-trajs", type=int)
    parser.add_argument(
        "--stop-relabel-radius",
        type=float,
        default=None,
        help=(
            "If set (metres), zero the action of frames within this distance of "
            "the episode goal plus the terminal frame, injecting stop (primitive "
            "0) supervision. Recommended: 20.0 (OPENFLY_SUCCESS_DIST_M). Default: "
            "off (raw deltas)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    annotation = json.loads(args.ann.read_text())
    trajectories = annotation if isinstance(annotation, list) else annotation["trajectories"]
    if args.max_trajs is not None:
        if args.max_trajs < 0:
            raise ValueError("--max-trajs must be non-negative")
        trajectories = trajectories[: args.max_trajs]

    episodes = [
        convert_trajectory(
            traj,
            args.image_root,
            action_source=ACTION_SOURCE,
            stop_relabel_radius=args.stop_relabel_radius,
        )
        for traj in trajectories
    ]
    write_lerobot_dataset(episodes, args.out, repo_id=args.out.name)


if __name__ == "__main__":
    main()
