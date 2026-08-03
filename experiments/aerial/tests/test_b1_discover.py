from __future__ import annotations

from pathlib import Path

from experiments.aerial.orchestration.b1_discover import (
    B1_STEPS,
    build_b1_eval_job,
    enqueue_ready_b1_jobs,
)


def test_build_b1_eval_job_paths():
    job = build_b1_eval_job(
        stamp="s1",
        step=250,
        checkpoint="/tmp/step_000250.pt",
        results_root=Path("/tmp/results"),
        ann=Path("/tmp/ann.json"),
        openfly_root=Path("/tmp/openfly"),
    )
    assert job["id"] == "b1-s1-step_000250"
    assert job["kind"] == "b1"
    assert job["out_metrics"].endswith("b1_s1/step_000250_seen20/metrics.json")
    assert job["seed"] == 42
    assert job["max_episodes"] == 20


def test_enqueue_ready_b1_jobs(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    queue = tmp_path / "queue"
    results = tmp_path / "results"
    ann = tmp_path / "ann.json"
    ann.write_text("[]\n", encoding="utf-8")
    openfly = tmp_path / "openfly"
    openfly.mkdir()
    for step in B1_STEPS:
        pt = weights / f"step_{step:06d}.pt"
        pt.write_bytes(b"x" * 100)
        (Path(str(pt) + ".sha256")).write_text("abcd\n", encoding="utf-8")
    jobs = enqueue_ready_b1_jobs(
        stamp="s1",
        weights_dir=weights,
        queue_dir=queue,
        results_root=results,
        ann=ann,
        openfly_root=openfly,
        min_bytes=10,
        settle_s=0.0,
    )
    assert len(jobs) == len(B1_STEPS)
    assert (queue / "pending").is_dir()
    assert len(list((queue / "pending").glob("*.json"))) == len(B1_STEPS)


def test_enqueue_mirrors_checkpoint_to_shared_storage(tmp_path):
    weights = tmp_path / "local" / "weights"
    weights.mkdir(parents=True)
    shared_weights = tmp_path / "shared" / "weights"
    queue = tmp_path / "queue"
    results = tmp_path / "results"
    ann = tmp_path / "ann.json"
    ann.write_text("[]\n", encoding="utf-8")
    openfly = tmp_path / "openfly"
    openfly.mkdir()
    source = weights / "step_000250.pt"
    source.write_bytes(b"checkpoint-bytes")
    (weights / "dataset_stats.json").write_text("{}\n", encoding="utf-8")

    jobs = enqueue_ready_b1_jobs(
        stamp="s1",
        weights_dir=weights,
        shared_weights_dir=shared_weights,
        queue_dir=queue,
        results_root=results,
        ann=ann,
        openfly_root=openfly,
        steps=(250,),
        min_bytes=10,
        settle_s=0.0,
    )

    mirrored = shared_weights / source.name
    assert mirrored.read_bytes() == source.read_bytes()
    assert Path(str(mirrored) + ".sha256").is_file()
    assert jobs[0]["checkpoint"] == str(mirrored.resolve())


def test_enqueue_mirrors_dataset_stats_next_to_checkpoint(tmp_path):
    """Eval resolves the normalizer from checkpoint.parent; without it the policy
    emits un-denormalized actions that never change position."""
    run_dir = tmp_path / "runs" / "b1-s1"
    weights = run_dir / "checkpoints" / "weights"
    weights.mkdir(parents=True)
    stats = run_dir / "dataset_stats.json"
    stats.write_text('{"action": {"default": {}}}\n', encoding="utf-8")
    shared_weights = tmp_path / "shared" / "weights"
    queue = tmp_path / "queue"
    results = tmp_path / "results"
    ann = tmp_path / "ann.json"
    ann.write_text("[]\n", encoding="utf-8")
    openfly = tmp_path / "openfly"
    openfly.mkdir()
    (weights / "step_000250.pt").write_bytes(b"checkpoint-bytes")

    enqueue_ready_b1_jobs(
        stamp="s1",
        weights_dir=weights,
        shared_weights_dir=shared_weights,
        queue_dir=queue,
        results_root=results,
        ann=ann,
        openfly_root=openfly,
        steps=(250,),
        min_bytes=10,
        settle_s=0.0,
    )

    mirrored_stats = shared_weights / "dataset_stats.json"
    assert mirrored_stats.is_file()
    assert mirrored_stats.read_text(encoding="utf-8") == stats.read_text(encoding="utf-8")
