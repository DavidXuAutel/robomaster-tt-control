from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.aerial.orchestration.eval_queue import enqueue


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "aerial" / "scripts" / "orch_eval_worker.sh"


def _job(tmp_path: Path, job_id: str = "b0-step_001000") -> dict:
    return {
        "id": job_id,
        "kind": "b0",
        "checkpoint": str(tmp_path / "step_001000.pt"),
        "out_metrics": str(tmp_path / f"{job_id}.json"),
        "task": "aerial_joint_b1_joint",
        "ann": str(tmp_path / "seen.json"),
        "openfly_root": str(tmp_path / "openfly"),
        "seed": 42,
        "max_steps": 100,
        "max_episodes": 20,
    }


def _run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.aerial.orchestration.eval_queue",
            *args,
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": "."},
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_prints_locked_serial_eval_command_without_remote_access(tmp_path: Path):
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "EVAL_WORKER_LOCK": str(tmp_path / "worker.lock"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "experiments.aerial.eval.run_closed_loop" in result.stdout
    assert "--seed 42" in result.stdout
    assert "--max-episodes 20" in result.stdout
    assert "--dump-frames /path/to/frames" in result.stdout
    assert "AIRSIM_HOST=10.229.20.125" in result.stdout
    assert "AIRSIM_PORT=41451" in result.stdout
    assert "AIRSIM_ALLOW_LOCAL_LAUNCH=0" in result.stdout
    assert "ssh" not in result.stdout.lower()


def test_existing_worker_lock_rejects_second_worker(tmp_path: Path):
    lock = tmp_path / "worker.lock"
    lock.mkdir()

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "EVAL_WORKER_LOCK": str(lock)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "already running" in result.stderr


def test_eval_queue_cli_marks_success_and_failure_terminally(tmp_path: Path):
    queue = tmp_path / "queue"
    success = _job(tmp_path, "success")
    enqueue(queue, success)
    claimed = _run_module("--queue-dir", str(queue), "--claim")
    assert claimed.returncode == 0
    assert json.loads(claimed.stdout)["id"] == "success"
    Path(success["out_metrics"]).write_text('{"NE": 1.0, "n": 20}\n', encoding="utf-8")

    marked = _run_module("--queue-dir", str(queue), "--mark-done", "success")
    assert marked.returncode == 0, marked.stderr
    assert (queue / "done" / "success.json").is_file()

    failed = _job(tmp_path, "failed")
    enqueue(queue, failed)
    assert _run_module("--queue-dir", str(queue), "--claim").returncode == 0
    marked = _run_module(
        "--queue-dir", str(queue), "--mark-failed", "failed", "--error", "exit 7"
    )
    assert marked.returncode == 0, marked.stderr
    assert (queue / "failed" / "failed.json").is_file()
    assert not (queue / "done" / "failed.json").exists()
    record = json.loads((queue / "failed" / "failed.json").read_text(encoding="utf-8"))
    assert record["error"] == "exit 7"


def test_failed_eval_is_marked_failed_not_done(tmp_path: Path):
    queue = tmp_path / "queue"
    enqueue(queue, _job(tmp_path))
    env_file = tmp_path / "env.sh"
    env_file.write_text("export UNUSED_TEST_VALUE=1\n", encoding="utf-8")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == \"-c\" ]] || "
        "[[ \" $* \" == *\" experiments.aerial.orchestration.eval_queue \"* ]]; then\n"
        f"  exec {sys.executable!s} \"$@\"\n"
        "fi\n"
        "exit 7\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--once"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": ".",
            "PYTHON_BIN": str(fake_python),
            "EVAL_QUEUE_DIR": str(queue),
            "EVAL_ENV_FILE": str(env_file),
            "EVAL_WORKER_LOCK": str(tmp_path / "worker.lock"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 7
    assert (queue / "failed" / "b0-step_001000.json").is_file()
    assert not (queue / "done" / "b0-step_001000.json").exists()


def test_env_sh_must_not_steal_mark_done_onto_default_queue(tmp_path: Path):
    """Regression: sourced env.sh exporting EVAL_QUEUE_DIR used to leave jobs
    stuck in the dedicated queue's running/ because mark_done looked elsewhere.
    """
    dedicated = tmp_path / "eval_queue_collapse_fix"
    default = tmp_path / "eval_queue"
    for q in (dedicated, default):
        for sub in ("pending", "running", "done", "failed"):
            (q / sub).mkdir(parents=True)

    job = _job(tmp_path, "collapse-step_001000")
    enqueue(dedicated, job)

    env_file = tmp_path / "env.sh"
    # Mimic production env.sh which unconditionally resets EVAL_QUEUE_DIR.
    env_file.write_text(
        f"export EVAL_QUEUE_DIR={default!s}\n"
        "export AIRSIM_HOST=10.229.20.125\n",
        encoding="utf-8",
    )

    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"${1:-}\" == \"-c\" ]] || "
        "[[ \" $* \" == *\" experiments.aerial.orchestration.eval_queue \"* ]]; then\n"
        f"  exec {sys.executable!s} \"$@\"\n"
        "fi\n"
        # Successful closed-loop: write valid metrics next to --out.
        "out=\"\"\n"
        "prev=\"\"\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$prev\" == \"--out\" ]]; then out=\"$arg\"; fi\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "if [[ -n \"$out\" ]]; then\n"
        "  mkdir -p \"$(dirname \"$out\")\"\n"
        "  printf '%s\\n' '{\"NE\": 12.3, \"SR\": 0.0, \"n\": 20}' > \"$out\"\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--once"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": ".",
            "PYTHON_BIN": str(fake_python),
            "EVAL_QUEUE_DIR": str(dedicated),
            "EVAL_ENV_FILE": str(env_file),
            "EVAL_WORKER_LOCK": str(tmp_path / "worker.lock"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert (dedicated / "done" / "collapse-step_001000.json").is_file()
    assert not (dedicated / "running" / "collapse-step_001000.json").exists()
    assert list((default / "done").glob("*.json")) == []
    assert list((default / "running").glob("*.json")) == []
    assert list((default / "failed").glob("*.json")) == []
