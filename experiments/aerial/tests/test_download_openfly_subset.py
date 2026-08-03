import json
from pathlib import Path

import yaml

from experiments.aerial import download_openfly_subset as downloader
from experiments.aerial.download_openfly_subset import (
    download_subset,
    select_trajectories,
    trajectory_parquet_filename,
    write_embedded_images,
)


def _trajectory(image_path: str, frame: str = "000000") -> dict:
    return {
        "image_path": image_path,
        "index_list": [frame],
        "gpt_instruction": "Fly forward.",
        "action": [0],
        "pos": [[0.0, 0.0, 1.0]],
        "yaw": [0.0],
    }


def test_select_trajectories_filters_prefix_and_samples_deterministically():
    trajectories = [
        _trajectory("env_airsim_alpha/traj_0"),
        _trajectory("env_matterport/traj_1"),
        _trajectory("env_airsim_beta/traj_2"),
        _trajectory("env_airsim_gamma/traj_3"),
    ]

    first = select_trajectories(
        trajectories,
        env_prefixes=["env_airsim_"],
        target_count=2,
        seed=17,
    )
    second = select_trajectories(
        trajectories,
        env_prefixes=["env_airsim_"],
        target_count=2,
        seed=17,
    )

    assert first == second
    assert len(first) == 2
    assert all(item.trajectory["image_path"].startswith("env_airsim_") for item in first)
    assert all(item.source_index in {0, 2, 3} for item in first)


def test_local_subset_copies_only_selected_images_and_writes_manifest(tmp_path):
    source = tmp_path / "source"
    annotation_dir = source / "Annotation"
    annotation_dir.mkdir(parents=True)
    trajectories = [
        _trajectory("env_airsim_alpha/traj_0"),
        _trajectory("env_airsim_beta/traj_1"),
        _trajectory("env_other/traj_2"),
    ]
    (annotation_dir / "train.json").write_text(json.dumps(trajectories))
    (annotation_dir / "seen.json").write_text(json.dumps([_trajectory("eval/seen")]))
    (annotation_dir / "unseen.json").write_text(json.dumps([_trajectory("eval/unseen")]))
    for trajectory in trajectories:
        image_dir = source / trajectory["image_path"]
        image_dir.mkdir(parents=True)
        (image_dir / "000000.png").write_bytes(b"image")

    out_raw = tmp_path / "raw"
    config = {
        "hf_repo": "IPEC-COMMUNITY/OpenFly",
        "target_train_trajs": 3,
        "split_ann": {
            "train": "Annotation/train.json",
            "seen": "Annotation/seen.json",
            "unseen": "Annotation/unseen.json",
        },
        "env_prefixes": ["env_airsim_"],
        "seed": 42,
        "out_raw": str(out_raw),
        "out_lerobot": str(tmp_path / "lerobot"),
    }

    manifest_path = download_subset(
        config,
        max_trajs=1,
        local_source=source,
        dry_run=False,
    )

    subset = json.loads((out_raw / "Annotation" / "subset_train.json").read_text())
    manifest = yaml.safe_load(manifest_path.read_text())
    assert len(subset) == 1
    assert (out_raw / subset[0]["image_path"] / "000000.png").is_file()
    assert not (out_raw / "env_other" / "traj_2").exists()
    assert manifest["selected_train_traj_indices"] == [0]
    assert manifest["selected_train_image_paths"] == ["env_airsim_alpha/traj_0"]
    assert (out_raw / "Annotation" / "seen.json").is_file()
    assert (out_raw / "Annotation" / "unseen.json").is_file()


def test_dry_run_writes_selection_without_copying_images(tmp_path):
    source = tmp_path / "source"
    annotation_dir = source / "Annotation"
    annotation_dir.mkdir(parents=True)
    trajectory = _trajectory("env_airsim_alpha/traj_0")
    for name, data in (
        ("train.json", [trajectory]),
        ("seen.json", []),
        ("unseen.json", []),
    ):
        (annotation_dir / name).write_text(json.dumps(data))

    out_raw = tmp_path / "raw"
    config = {
        "hf_repo": "IPEC-COMMUNITY/OpenFly",
        "target_train_trajs": 1,
        "split_ann": {
            "train": "Annotation/train.json",
            "seen": "Annotation/seen.json",
            "unseen": "Annotation/unseen.json",
        },
        "env_prefixes": ["env_airsim_"],
        "seed": 42,
        "out_raw": str(out_raw),
    }

    download_subset(config, local_source=source, dry_run=True)

    assert (out_raw / "Annotation" / "subset_train.json").is_file()
    assert not (out_raw / trajectory["image_path"]).exists()


def test_hf_annotations_download_individually_without_repo_tree_listing(
    tmp_path, monkeypatch
):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("[]")
        return str(destination)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("snapshot_download enumerates the full repository")
        ),
    )
    downloader._download_annotations(
        "IPEC-COMMUNITY/OpenFly",
        tmp_path,
        {
            "train": "Annotation/train.json",
            "seen": "Annotation/seen.json",
            "unseen": "Annotation/unseen.json",
        },
    )

    assert [call["filename"] for call in calls] == [
        "Annotation/train.json",
        "Annotation/seen.json",
        "Annotation/unseen.json",
    ]
    assert all(call["repo_type"] == "dataset" for call in calls)


def test_trajectory_parquet_filename_uses_path_filterable_repo_layout():
    assert (
        trajectory_parquet_filename("env_airsim_18/astar_data/low_short/run")
        == "traj/env_airsim_18/astar_data/low_short/run.parquet"
    )


def test_write_embedded_images_extracts_only_annotation_frames(tmp_path):
    rows = [
        {"image_id": "frame_0", "image": {"bytes": b"zero", "path": None}},
        {"image_id": "frame_1", "image": {"bytes": b"one", "path": None}},
        {"image_id": "frame_2", "image": {"bytes": b"two", "path": None}},
    ]

    write_embedded_images(rows, ["frame_0", "frame_2"], tmp_path)

    assert (tmp_path / "frame_0.png").read_bytes() == b"zero"
    assert (tmp_path / "frame_2.png").read_bytes() == b"two"
    assert not (tmp_path / "frame_1.png").exists()
