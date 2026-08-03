from __future__ import annotations

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


def test_accelerate_h100_config_is_zero2_no_offload() -> None:
    config = (SCRIPTS / "accelerate_zero2_no_offload_2proc.yaml").read_text(encoding="utf-8")
    assert "distributed_type: DEEPSPEED" in config
    assert "zero_stage: 2" in config
    assert "offload_optimizer_device: none" in config
    assert "num_processes: 2" in config
    assert "mixed_precision: bf16" in config


def test_orch_b1_train_dry_run_locks_recipe(tmp_path: Path) -> None:
    cache = tmp_path / "ft_cache"
    repo = cache / "repo"
    scripts = repo / "experiments" / "aerial" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "accelerate_zero2_no_offload_2proc.yaml").write_text(
        (SCRIPTS / "accelerate_zero2_no_offload_2proc.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    model = cache / "model"
    model.mkdir()
    ckpt = model / "baseline.pt"
    ckpt.write_bytes(b"weights")
    orch = tmp_path / "orch"
    orch.mkdir()
    status = orch / "status.json"
    status.write_text(
        '{"phase":"RUN_B1_TRAIN","stamp":"s1","gates_passed":true}\n',
        encoding="utf-8",
    )
    lock = orch / "baseline_lock.manifest.json"
    lock.write_text(
        '{"checkpoint":"%s","sha256":"%s"}\n'
        % (ckpt, "0" * 64),
        encoding="utf-8",
    )
    (cache / "smoke.status").write_text("PASSED\n", encoding="utf-8")
    (cache / "SHA256SUMS").write_text("deadbeef  model/baseline.pt\n", encoding="utf-8")

    result = _run(
        "orch_b1_train.sh",
        "--dry-run",
        env={
            "AERIAL_FT_CACHE": str(cache),
            "STATUS_PATH": str(status),
            "LOCK_PATH": str(lock),
            "SKIP_MANIFEST_VERIFY": "1",
        },
    )
    assert result.returncode == 0, result.stderr + result.stdout
    out = result.stdout
    assert "lambda_video=0.0" in out
    assert "max_steps=5000" in out
    assert "save_every=250" in out
    assert "learning_rate=1e-5" in out
    assert "resume=" in out and "baseline.pt" in out
    assert "Expected:" in out and "step_005000.pt" in out


def test_orch_b1_train_dry_run_resumes_latest_weights(tmp_path: Path) -> None:
    cache = tmp_path / "ft_cache"
    repo = cache / "repo"
    scripts = repo / "experiments" / "aerial" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "accelerate_zero2_no_offload_2proc.yaml").write_text(
        (SCRIPTS / "accelerate_zero2_no_offload_2proc.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    model = cache / "model"
    model.mkdir()
    ckpt = model / "baseline.pt"
    ckpt.write_bytes(b"weights")
    run = cache / "runs" / "b1-20260727-072347-5k-2gpu-b0-to-joint-video"
    weights = run / "checkpoints" / "weights"
    weights.mkdir(parents=True)
    (weights / "step_001000.pt").write_bytes(b"ft-weights")
    orch = tmp_path / "orch"
    orch.mkdir()
    status = orch / "status.json"
    status.write_text(
        '{"phase":"RUN_B1_TRAIN","stamp":"s1","gates_passed":true}\n',
        encoding="utf-8",
    )
    lock = orch / "baseline_lock.manifest.json"
    lock.write_text(
        '{"checkpoint":"%s","sha256":"%s"}\n' % (ckpt, "0" * 64),
        encoding="utf-8",
    )
    (cache / "SHA256SUMS").write_text("deadbeef  model/baseline.pt\n", encoding="utf-8")

    result = _run(
        "orch_b1_train.sh",
        "--dry-run",
        env={
            "AERIAL_FT_CACHE": str(cache),
            "STATUS_PATH": str(status),
            "LOCK_PATH": str(lock),
            "SKIP_MANIFEST_VERIFY": "1",
        },
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "resume=" in result.stdout and "step_001000.pt" in result.stdout
    assert "max_steps=5000" in result.stdout
    assert "baseline.pt" not in result.stdout.split("resume=")[1].split()[0]


def test_orch_b1_train_refuses_without_gates(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    status.write_text(
        '{"phase":"B1_GATES","stamp":"s1","gates_passed":false}\n',
        encoding="utf-8",
    )
    result = _run(
        "orch_b1_train.sh",
        "--dry-run",
        env={
            "AERIAL_FT_CACHE": str(tmp_path / "missing"),
            "STATUS_PATH": str(status),
            "LOCK_PATH": str(tmp_path / "missing_lock.json"),
        },
    )
    assert result.returncode != 0
    assert "gates" in (result.stderr + result.stdout).lower()


def test_sync_h100_dry_run_targets_31103(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "step_002000.pt"
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
    lock = tmp_path / "baseline_lock.manifest.json"
    lock.write_text(
        '{"checkpoint":"%s","sha256":"%s"}\n' % (checkpoint, "a" * 64),
        encoding="utf-8",
    )

    result = _run(
        "sync_b0_ft_to_h100.sh",
        "--dry-run",
        env={
            "LOCK_PATH": str(lock),
            "TRAIN_SUBSET": str(assets["train_subset"]),
            "CORRECTION_SET": str(assets["correction"]),
            "TEXT_EMBEDS": str(assets["text_embeds"]),
            "COLLECTION_MANIFEST": str(collection_manifest),
            "CODE_ROOT": str(REPO_ROOT),
            "DATASET_STATS": str(checkpoint_dir / "dataset_stats.json"),
        },
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "DRY RUN: no SSH or rsync commands executed" in result.stdout
    assert "a25689@10.239.121.22" in result.stdout
    assert "SSH port 31103" in result.stdout


def test_ckpt_watch_dry_run_lists_b1_steps() -> None:
    result = _run("orch_ckpt_watch_enqueue.sh", "--dry-run", env={})
    assert result.returncode == 0, result.stderr + result.stdout
    assert "250,500,750,1000" in result.stdout
    assert "5000" in result.stdout
    for step in (250, 1000, 5000):
        assert f"--steps" in result.stdout or f"{step}" in result.stdout
    assert "5000" in result.stdout.split("enqueue steps:")[1].splitlines()[0]


def test_orch_b1_progress_summarizes_fixture(tmp_path: Path) -> None:
    stamp = "20260727-072347-5k-2gpu-b0-to-joint-video"
    cache = tmp_path / "ft_cache"
    orch = tmp_path / "orch"
    shared = tmp_path / "shared"
    run = cache / f"runs/b1-{stamp}"
    weights = run / "checkpoints" / "weights"
    shared_weights = (
        shared / "runs" / "aerial_b1_ft" / f"m1b-{stamp}" / "checkpoints" / "weights"
    )
    queue = shared / "orchestration" / "eval_queue"
    logs = cache / "logs" / "ft"
    for path in (
        orch,
        weights,
        shared_weights,
        logs,
        *(queue / name for name in ("pending", "running", "done", "failed")),
    ):
        path.mkdir(parents=True)
    (orch / "status.json").write_text(
        '{"phase":"RUN_B1_TRAIN","stamp":"%s","gates_passed":true}\n' % stamp,
        encoding="utf-8",
    )
    (cache / "ft.status").write_text("RUNNING\n", encoding="utf-8")
    (cache / "smoke.status").write_text("COMPLETED\n", encoding="utf-8")
    (orch / "ft_smoke.status").write_text("PASSED\n", encoding="utf-8")
    (weights / "step_001000.pt").write_bytes(b"x" * 64)
    (shared_weights / "step_001000.pt").write_bytes(b"x" * 64)
    (shared_weights / "step_001000.pt.sha256").write_text("abc  step_001000.pt\n")
    (queue / "pending" / "job.json").write_text("{}\n", encoding="utf-8")
    (logs / "b1-5000-step.log").write_text(
        "INFO | >>  epoch=0 step=1250/5000 loss=0.1234 loss_action=0.1234\n",
        encoding="utf-8",
    )

    result = _run(
        "orch_b1_progress.sh",
        env={
            "STAMP": stamp,
            "AERIAL_FT_CACHE": str(cache),
            "ORCH_ROOT": str(orch),
            "STATUS_PATH": str(orch / "status.json"),
            "SHARED_WEIGHTS_DIR": str(shared_weights),
            "EVAL_QUEUE_DIR": str(queue),
            "SKIP_PROCESS_CHECK": "1",
            "SKIP_GPU_CHECK": "1",
        },
    )
    assert result.returncode == 0, result.stderr + result.stdout
    out = result.stdout
    assert "phase=RUN_B1_TRAIN" in out
    assert "gates_passed=true" in out
    assert "ft.status=RUNNING" in out
    assert "step=1250/5000" in out
    assert "loss=0.1234" in out
    assert "local_ckpt step_001000=yes" in out
    assert "local_ckpt step_002000=no" in out
    assert "shared_ckpt step_001000=yes" in out
    assert "latest_local_ckpt_step=1000" in out
    assert "queue pending=1" in out


def test_shell_scripts_parse() -> None:
    for name in (
        "orch_b1_train.sh",
        "orch_ckpt_watch_enqueue.sh",
        "orch_b1_progress.sh",
        "sync_b0_ft_to_h100.sh",
    ):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPTS / name)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
