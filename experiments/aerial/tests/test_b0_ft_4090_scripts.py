from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "experiments" / "aerial" / "scripts"


def _run(script: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPTS / script), *args],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        text=True,
        capture_output=True,
        check=False,
    )


def _fake_cache(tmp_path: Path, *, smoke_complete: bool = False) -> Path:
    cache = tmp_path / "cache"
    checkpoint = cache / "model" / "step_004000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    remote_scripts = cache / "repo" / "experiments" / "aerial" / "scripts"
    remote_scripts.mkdir(parents=True)
    remote_scripts.joinpath("accelerate_zero2_opt_offload_2proc.yaml").write_text(
        SCRIPTS.joinpath("accelerate_zero2_opt_offload_2proc.yaml").read_text(),
        encoding="utf-8",
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (cache / "SHA256SUMS").write_text(
        f"{digest}  model/step_004000.pt\n", encoding="utf-8"
    )
    if smoke_complete:
        (cache / "smoke.status").write_text("COMPLETED\n", encoding="utf-8")
    return cache


def test_accelerate_config_locks_dual_gpu_zero2_cpu_offload() -> None:
    config = (SCRIPTS / "accelerate_zero2_opt_offload_2proc.yaml").read_text()
    assert "distributed_type: DEEPSPEED" in config
    assert "zero_stage: 2" in config
    assert "offload_optimizer_device: cpu" in config
    assert "num_processes: 2" in config
    assert "mixed_precision: bf16" in config


def test_sync_dry_run_builds_manifest_without_remote_access(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "step_004000.pt"
    checkpoint.write_bytes(b"weights")
    (checkpoint_dir / "dataset_stats.json").write_text("{}", encoding="utf-8")
    assets = {}
    for name in ("train_subset", "correction", "text_embeds"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "asset.bin").write_bytes(name.encode())
        assets[name] = directory
    collection_manifest = tmp_path / "collection_manifest.json"
    collection_manifest.write_text("{}", encoding="utf-8")

    result = _run(
        "sync_b0_ft_to_4090.sh",
        "--dry-run",
        env={
            "B0_CHECKPOINT": str(checkpoint),
            "TRAIN_SUBSET": str(assets["train_subset"]),
            "CORRECTION_SET": str(assets["correction"]),
            "TEXT_EMBEDS": str(assets["text_embeds"]),
            "COLLECTION_MANIFEST": str(collection_manifest),
            "CODE_ROOT": str(REPO_ROOT),
        },
    )
    assert result.returncode == 0, result.stderr
    assert "DRY RUN: no SSH or rsync commands executed" in result.stdout
    assert "a25689@10.239.121.14" in result.stdout
    assert "SSH port 30879" in result.stdout


def test_smoke_dry_run_locks_steps_checkpoint_and_single_oom_retry(tmp_path: Path) -> None:
    cache = _fake_cache(tmp_path)
    result = _run(
        "smoke_b0_ft_4090.sh",
        "--dry-run",
        env={"AERIAL_FT_CACHE": str(cache)},
    )
    assert result.returncode == 0, result.stderr
    assert "max_steps=1" in result.stdout
    assert "max_steps=10" in result.stdout
    assert "step_004000.pt" in result.stdout
    assert "OOM retries allowed: 1" in result.stdout
    assert "reduce_bucket_size=50000000" in result.stdout
    assert "peak memory gate: <23552 MiB/GPU" in result.stdout


def test_full_dry_run_requires_smoke_and_locks_outputs(tmp_path: Path) -> None:
    cache = _fake_cache(tmp_path, smoke_complete=True)
    result = _run(
        "run_b0_ft_4090.sh",
        "--dry-run",
        env={"AERIAL_FT_CACHE": str(cache)},
    )
    assert result.returncode == 0, result.stderr
    assert "max_steps=1000" in result.stdout
    assert "save_every=250" in result.stdout
    assert "resume=" in result.stdout and "step_004000.pt" in result.stdout
    for step in (250, 500, 1000):
        assert f"step_{step:06d}.pt" in result.stdout


def test_shell_scripts_parse() -> None:
    for name in (
        "sync_b0_ft_to_4090.sh",
        "smoke_b0_ft_4090.sh",
        "run_b0_ft_4090.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPTS / name)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
