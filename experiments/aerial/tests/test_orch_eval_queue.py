from experiments.aerial.orchestration.eval_queue import enqueue, claim_next, mark_done, metrics_valid


def _job(job_id: str, metrics_path) -> dict:
    return {
        "id": job_id,
        "kind": "b0",
        "checkpoint": "/c.pt",
        "out_metrics": str(metrics_path),
        "task": "aerial_joint_b1_joint",
        "ann": "/a.json",
        "openfly_root": "/of",
        "seed": 42,
        "max_steps": 100,
        "max_episodes": 20,
    }


def test_fifo_claim_follows_enqueue_order_not_job_id(tmp_path):
    q = tmp_path / "queue"
    enqueue(q, _job("z", tmp_path / "z.json"))
    enqueue(q, _job("a", tmp_path / "a.json"))
    job = claim_next(q)
    assert job is not None and job["id"] == "z"


def test_skip_valid_metrics_moves_to_done_and_clears_pending(tmp_path):
    q = tmp_path / "queue"
    metrics = tmp_path / "m.json"
    metrics.write_text('{"NE": 1.0, "SR": 0.0, "n": 20}\n')
    jid = enqueue(q, _job("b0-step_001000", metrics))
    assert claim_next(q) is None
    assert list((q / "pending").glob("*.json")) == []
    assert (q / "done" / f"{jid}.json").is_file()


def test_claim_and_mark_done(tmp_path):
    q = tmp_path / "queue"
    metrics = tmp_path / "m.json"
    jid = enqueue(q, _job("b0-step_001000", metrics))
    job = claim_next(q)
    assert job is not None and job["id"] == jid
    mark_done(q, jid, {"NE": 12.3, "SR": 0.0, "n": 20.0})
    assert claim_next(q) is None
    assert (q / "done" / f"{jid}.json").is_file()


def test_enqueue_idempotent_by_job_id(tmp_path):
    q = tmp_path / "queue"
    metrics = tmp_path / "m.json"
    job = _job("b0-step_001000", metrics)
    jid1 = enqueue(q, job)
    jid2 = enqueue(q, job)
    assert jid1 == jid2
    assert len(list((q / "pending").glob("*.json"))) == 1


def test_metrics_valid_rejects_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json\n")
    assert metrics_valid(path) is False


def test_metrics_valid_rejects_nonnumeric_ne(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"NE": "oops", "n": 20}\n')
    assert metrics_valid(path) is False


def test_metrics_valid_rejects_n_below_one(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"NE": 1.0, "n": 0}\n')
    assert metrics_valid(path) is False
