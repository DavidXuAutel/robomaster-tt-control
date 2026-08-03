from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


LOGGER = logging.getLogger(__name__)
DEFAULT_REPO = "IPEC-COMMUNITY/OpenFly"


@dataclass(frozen=True)
class SelectedTrajectory:
    source_index: int
    trajectory: dict[str, Any]


def _safe_relative_path(value: str) -> Path:
    posix_path = PurePosixPath(value)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"dataset path must be relative and cannot contain '..': {value!r}")
    return Path(*posix_path.parts)


def _annotation_trajectories(annotation: Any) -> list[dict[str, Any]]:
    trajectories = annotation if isinstance(annotation, list) else annotation["trajectories"]
    if not isinstance(trajectories, list):
        raise ValueError("annotation must be a list or contain a 'trajectories' list")
    return trajectories


def select_trajectories(
    trajectories: Sequence[dict[str, Any]],
    *,
    env_prefixes: Sequence[str],
    target_count: int,
    seed: int,
) -> list[SelectedTrajectory]:
    """Filter by image-path prefix and deterministically sample source trajectories."""
    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    if not env_prefixes:
        raise ValueError("env_prefixes must contain at least one prefix")

    candidates = [
        SelectedTrajectory(index, trajectory)
        for index, trajectory in enumerate(trajectories)
        if str(trajectory.get("image_path", "")).startswith(tuple(env_prefixes))
    ]
    count = min(target_count, len(candidates))
    selected = random.Random(seed).sample(candidates, count)
    return sorted(selected, key=lambda item: item.source_index)


def _copy_annotations(
    source_root: Path,
    out_root: Path,
    split_ann: Mapping[str, str],
) -> None:
    for relative_name in split_ann.values():
        relative_path = _safe_relative_path(relative_name)
        source = source_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"local annotation not found: {source}")
        destination = out_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _download_annotations(
    repo_id: str,
    out_root: Path,
    split_ann: Mapping[str, str],
) -> None:
    from huggingface_hub import hf_hub_download

    for filename in split_ann.values():
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=out_root,
        )


def _copy_image_folders(
    source_root: Path,
    out_root: Path,
    image_paths: Sequence[str],
) -> None:
    for image_path in image_paths:
        relative_path = _safe_relative_path(image_path)
        source = source_root / relative_path
        if not source.is_dir():
            raise FileNotFoundError(f"local image folder not found: {source}")
        destination = out_root / relative_path
        shutil.copytree(source, destination, dirs_exist_ok=True)


def trajectory_parquet_filename(image_path: str) -> str:
    relative_path = _safe_relative_path(image_path)
    return PurePosixPath("traj", *relative_path.parts).with_suffix(".parquet").as_posix()


def write_embedded_images(
    rows: Sequence[Mapping[str, Any]],
    frame_ids: Sequence[object],
    destination: Path,
) -> None:
    """Extract selected embedded PNG payloads from OpenFly parquet rows."""
    requested = {str(frame_id) for frame_id in frame_ids}
    written: set[str] = set()
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_id = str(row["image_id"])
        if image_id not in requested:
            continue
        image = row["image"]
        payload = image.get("bytes") if isinstance(image, Mapping) else image
        if not isinstance(payload, bytes):
            raise ValueError(f"embedded image {image_id!r} has no byte payload")
        filename = image_id if Path(image_id).suffix else f"{image_id}.png"
        (destination / filename).write_bytes(payload)
        written.add(image_id)
    missing = sorted(requested - written)
    if missing:
        raise ValueError(f"trajectory parquet is missing requested frames: {missing[:5]}")


def _download_trajectory_images(
    repo_id: str,
    out_root: Path,
    selected: Sequence[SelectedTrajectory],
) -> None:
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    staging_root = (out_root / ".openfly_download").resolve()
    shutil.rmtree(staging_root, ignore_errors=True)
    for item in selected:
        image_path = str(item.trajectory["image_path"])
        parquet_file = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=trajectory_parquet_filename(image_path),
                repo_type="dataset",
                local_dir=staging_root,
            )
        )
        rows = parquet.read_table(
            parquet_file,
            columns=["image_id", "image"],
        ).to_pylist()
        write_embedded_images(
            rows,
            item.trajectory["index_list"],
            out_root / _safe_relative_path(image_path),
        )
        parquet_file.unlink()
    shutil.rmtree(staging_root, ignore_errors=True)


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def download_subset(
    config: Mapping[str, Any],
    *,
    max_trajs: int | None = None,
    local_source: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Download or locally copy a path-filtered OpenFly training subset."""
    repo_id = str(config.get("hf_repo", DEFAULT_REPO))
    out_root = Path(config["out_raw"]).expanduser()
    split_ann = dict(config["split_ann"])
    for required_split in ("train", "seen", "unseen"):
        if required_split not in split_ann:
            raise ValueError(f"split_ann is missing required split {required_split!r}")

    out_root.mkdir(parents=True, exist_ok=True)
    if local_source is None:
        _download_annotations(repo_id, out_root, split_ann)
    else:
        _copy_annotations(Path(local_source), out_root, split_ann)

    train_path = out_root / _safe_relative_path(split_ann["train"])
    trajectories = _annotation_trajectories(json.loads(train_path.read_text()))
    target_count = int(config["target_train_trajs"])
    if max_trajs is not None:
        if max_trajs < 0:
            raise ValueError("--max-trajs must be non-negative")
        target_count = min(target_count, max_trajs)

    selected = select_trajectories(
        trajectories,
        env_prefixes=[str(prefix) for prefix in config["env_prefixes"]],
        target_count=target_count,
        seed=int(config.get("seed", 42)),
    )
    if target_count and not selected:
        raise ValueError("no trajectories matched the configured env_prefixes")

    annotation_dir = out_root / "Annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    subset_path = annotation_dir / "subset_train.json"
    subset_path.write_text(
        json.dumps([item.trajectory for item in selected], indent=2) + "\n"
    )

    image_paths = sorted(
        {str(item.trajectory["image_path"]) for item in selected}
    )
    if not dry_run:
        if local_source is None:
            _download_trajectory_images(repo_id, out_root, selected)
        else:
            _copy_image_folders(Path(local_source), out_root, image_paths)

    manifest = dict(config)
    manifest.update(
        {
            "hf_repo": repo_id,
            "out_raw": str(out_root),
            "selected_train_traj_indices": [
                item.source_index for item in selected
            ],
            "selected_train_image_paths": image_paths,
            "selected_train_trajs": len(selected),
            "subset_annotation": "Annotation/subset_train.json",
            "dry_run": dry_run,
            "total_local_bytes": _directory_bytes(out_root),
        }
    )
    manifest_path = out_root / "subset_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    LOGGER.info(
        "selected %d trajectories in %d image folders; local data size %.2f MiB",
        len(selected),
        len(image_paths),
        manifest["total_local_bytes"] / (1024 * 1024),
    )
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a path-filtered OpenFly training subset."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-trajs", type=int)
    parser.add_argument(
        "--local-source",
        type=Path,
        help="Use a local OpenFly-style root instead of Hugging Face.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write selected annotations and manifest without image folders.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text())
    manifest = download_subset(
        config,
        max_trajs=args.max_trajs,
        local_source=args.local_source,
        dry_run=args.dry_run,
    )
    LOGGER.info("wrote manifest to %s", manifest)


if __name__ == "__main__":
    main()
