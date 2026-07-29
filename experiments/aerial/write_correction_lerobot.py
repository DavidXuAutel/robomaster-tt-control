from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

from experiments.aerial.convert_openfly_to_lerobot import (
    ACTION_SOURCE,
    write_lerobot_dataset,
)


DEFAULT_OUTPUT = Path("data/openfly_lerobot/b0_dagger_correction")


def _load_record_image(episode_path: Path, value: object) -> np.ndarray:
    image_path = Path(str(value))
    if not image_path.is_absolute():
        image_path = episode_path.parent / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"correction image not found: {image_path}")
    with Image.open(image_path) as source:
        return np.asarray(source.convert("RGB"), dtype=np.uint8)


def _validate_vector(
    record: dict[str, Any],
    key: str,
    *,
    line_number: int,
) -> np.ndarray:
    value = np.asarray(record.get(key), dtype=np.float32)
    if value.shape != (4,):
        raise ValueError(
            f"{key} at {line_number} must have shape (4,), got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{key} at {line_number} must contain only finite values")
    return value


def load_correction_episode(episode_path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with episode_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"record at {line_number} must be a JSON object")
            task = str(record.get("task", "")).strip()
            if not task:
                raise ValueError(f"task at {line_number} must be non-empty")
            state = _validate_vector(
                record, "observation.state", line_number=line_number
            )
            action = _validate_vector(record, "action", line_number=line_number)
            image = _load_record_image(
                episode_path, record.get("observation.images.ego")
            )
            frames.append(
                {
                    "observation.images.ego": image,
                    "observation.state": state,
                    "action": action,
                    "task": task,
                    "meta.action_source": ACTION_SOURCE,
                }
            )
    return frames


def write_correction_dataset(
    input_root: Path,
    out_root: Path = DEFAULT_OUTPUT,
    *,
    repo_id: Optional[str] = None,
) -> int:
    episode_paths = sorted(input_root.glob("ep*.jsonl"))
    if not episode_paths:
        raise ValueError(f"no ep*.jsonl correction episodes found under {input_root}")
    episodes = [load_correction_episode(path) for path in episode_paths]
    non_empty_count = sum(bool(episode) for episode in episodes)
    if non_empty_count == 0:
        raise ValueError("correction episodes contain no frames")
    write_lerobot_dataset(
        episodes,
        out_root,
        repo_id=repo_id or out_root.name,
    )
    return non_empty_count


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write DAgger corrections as a LeRobot v2.1 dataset."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    write_correction_dataset(args.input, args.out, repo_id=args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
