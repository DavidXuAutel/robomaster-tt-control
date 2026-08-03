from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.aerial.orchestration.gates import (
    blocked_payload,
    evaluate_b1_gates,
    queue_is_idle,
)
from experiments.aerial.orchestration.state import Phase
from experiments.aerial.orchestration.supervisor import advance_phase


def test_blocked_payload_includes_phase_reason_and_failed_gate():
    payload = blocked_payload(
        stamp="20260727-072347-5k-2gpu-b0-to-joint-video",
        failed_gate="collection_source",
        reason="missing collection source; refuse fabrication from held-out",
    )
    assert payload["phase"] == Phase.BLOCKED.value
    assert payload["failed_gate"] == "collection_source"
    assert "refuse fabrication" in payload["reason"]
    assert payload["stamp"] == "20260727-072347-5k-2gpu-b0-to-joint-video"
    assert "checked_at" in payload


def test_queue_idle_requires_empty_pending_and_running(tmp_path):
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "running").mkdir()
    (queue / "done").mkdir()
    assert queue_is_idle(queue) is True
    (queue / "pending" / "job.json").write_text("{}\n", encoding="utf-8")
    assert queue_is_idle(queue) is False


def test_advance_phase_eval_to_lock_when_queue_idle(tmp_path):
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "running").mkdir()
    status = {
        "phase": Phase.EVAL_B0_CHECKPOINTS.value,
        "stamp": "s1",
    }
    next_status = advance_phase(status, queue_dir=queue)
    assert next_status["phase"] == Phase.LOCK_BASELINE.value


def test_evaluate_b1_gates_blocks_on_missing_collection(tmp_path):
    ckpt = tmp_path / "step.pt"
    ckpt.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    lock = tmp_path / "baseline_lock.manifest.json"
    lock.write_text(
        json.dumps({"checkpoint": str(ckpt), "sha256": digest}),
        encoding="utf-8",
    )
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps([{"id": "h1"}]), encoding="utf-8")
    payload = evaluate_b1_gates(
        stamp="s1",
        lock_path=lock,
        collection_source=tmp_path / "missing_collection.json",
        heldout_ann=heldout,
        oracle_json=tmp_path / "oracle.json",
        correction_root=tmp_path / "correction",
        ft_manifest=tmp_path / "manifest.sha256",
        smoke_status=tmp_path / "smoke.status",
    )
    assert payload["phase"] == Phase.BLOCKED.value
    assert payload["failed_gate"] == "collection_source"
    assert "refuse fabrication" in payload["reason"]


def test_lock_baseline_from_results_picks_min_ne(tmp_path):
    from experiments.aerial.orchestration.supervisor import lock_baseline_from_results

    weights = tmp_path / "weights"
    results = tmp_path / "results"
    weights.mkdir()
    for step, ne in ((1000, 150.0), (2000, 100.0)):
        ckpt = weights / f"step_{step:06d}.pt"
        payload = f"ckpt-{step}".encode()
        ckpt.write_bytes(payload)
        (Path(str(ckpt) + ".sha256")).write_text(
            hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8"
        )
        metrics_dir = results / f"b0_step_{step:06d}_seen20"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps({"NE": ne, "n": 20}),
            encoding="utf-8",
        )
    out = tmp_path / "baseline_lock.manifest.json"
    manifest = lock_baseline_from_results(
        stamp="s1",
        weights_dir=weights,
        results_root=results,
        out=out,
        steps=(1000, 2000),
    )
    assert manifest["baseline_mean_ne"] == 100.0
    assert manifest["checkpoint"].endswith("step_002000.pt")
    assert out.is_file()
