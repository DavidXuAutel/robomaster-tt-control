from __future__ import annotations

from pathlib import Path

from experiments.aerial.orchestration.b0_discover import (
    discover_b0_checkpoints,
    enqueue_b0_eval_jobs,
)


def test_discover_only_complete_steps(tmp_path: Path) -> None:
    weights = tmp_path / "checkpoints" / "weights"
    weights.mkdir(parents=True)
    pt = weights / "step_001000.pt"
    pt.write_bytes(b"checkpoint")
    (weights / "step_001000.pt.sha256").write_text(
        "deadbeef  step_001000.pt\n",
        encoding="utf-8",
    )
    (weights / "step_002000.pt").write_bytes(b"incomplete")
    found = discover_b0_checkpoints(weights, min_bytes=1, settle_s=0.0)
    assert [c["step"] for c in found] == [1000]
    assert found[0]["checkpoint"].endswith("step_001000.pt")


def test_enqueue_b0_jobs_reuses_existing_step1000_metrics(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    pt = weights / "step_001000.pt"
    pt.write_bytes(b"ckpt")
    (weights / "step_001000.pt.sha256").write_text("abc  step_001000.pt\n", encoding="utf-8")
    metrics = tmp_path / "step_001000_seen20" / "metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"NE": 151.0, "SR": 0.0, "n": 20}\n', encoding="utf-8")
    queue = tmp_path / "queue"
    jobs = enqueue_b0_eval_jobs(
        weights_dir=weights,
        queue_dir=queue,
        results_root=tmp_path,
        ann=tmp_path / "ann.json",
        openfly_root=tmp_path / "openfly",
        min_bytes=1,
        settle_s=0.0,
    )
    assert len(jobs) == 1
    assert jobs[0]["id"] == "b0-step_001000"
    assert jobs[0]["out_metrics"] == str(metrics)
    assert (queue / "pending" / "b0-step_001000.json").is_file() or (
        queue / "done" / "b0-step_001000.json"
    ).is_file()
