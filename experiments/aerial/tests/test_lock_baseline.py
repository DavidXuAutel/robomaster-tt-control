from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pytest

from experiments.aerial.eval.lock_baseline import (
    ORCHESTRATION_VERSION,
    build_lock_manifest,
    compute_checkpoint_sha256,
    load_metrics,
    main,
    parse_candidate_json,
    select_baseline,
    validate_candidate,
    validate_candidates,
    verify_checkpoint_sha256,
    write_lock_manifest,
)
from experiments.aerial.orchestration.state import read_status


def _write_ckpt(path, content: bytes = b"weights") -> str:
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_metrics(path, ne: float, n: float = 20.0) -> None:
    path.write_text(json.dumps({"NE": ne, "n": n}) + "\n")


def _candidate_record(tmp_path, step: int, ne: float, tag: str, content: bytes | None = None) -> dict:
    ckpt = tmp_path / f"{tag}.pt"
    sha = _write_ckpt(ckpt, content if content is not None else tag.encode())
    metrics = tmp_path / f"{tag}.json"
    _write_metrics(metrics, ne)
    return {
        "step": step,
        "checkpoint": str(ckpt),
        "metrics_path": str(metrics),
        "mean_ne": ne,
        "sha256": sha,
    }


def test_select_lowest_ne_tie_breaks_later_step(tmp_path):
    candidates = [
        _candidate_record(tmp_path, 1000, 150.0, "a"),
        _candidate_record(tmp_path, 4000, 120.0, "b"),
        _candidate_record(tmp_path, 5000, 120.0, "c"),
    ]
    chosen = select_baseline(candidates)
    assert chosen["step"] == 5000
    man = build_lock_manifest(
        chosen,
        candidates=candidates,
        stamp="20260727-072347-5k-2gpu-b0-to-joint-video",
        selection_time="2026-07-27T12:00:00+00:00",
    )
    assert man["s1_ne"] == 96.0
    assert man["baseline_mean_ne"] == 120.0
    assert man["selection_time"] == "2026-07-27T12:00:00+00:00"
    assert man["orchestration_version"] == ORCHESTRATION_VERSION


def test_select_baseline_rejects_no_finite_candidates(tmp_path):
    metrics_inf = tmp_path / "inf.json"
    _write_metrics(metrics_inf, float("inf"))
    ckpt = tmp_path / "x.pt"
    sha = _write_ckpt(ckpt, b"x")
    candidates = [
        {
            "step": 1000,
            "mean_ne": float("inf"),
            "checkpoint": str(ckpt),
            "metrics_path": str(metrics_inf),
            "sha256": sha,
        }
    ]
    with pytest.raises(ValueError, match="non-finite mean_ne"):
        select_baseline(candidates)


def test_select_baseline_rejects_any_non_finite(tmp_path):
    good = _candidate_record(tmp_path, 2000, 100.0, "good")
    bad_ckpt = tmp_path / "bad.pt"
    bad_sha = _write_ckpt(bad_ckpt, b"bad")
    candidates = [
        {
            "step": 1000,
            "mean_ne": float("inf"),
            "checkpoint": str(bad_ckpt),
            "metrics_path": good["metrics_path"],
            "sha256": bad_sha,
        },
        good,
    ]
    with pytest.raises(ValueError, match="non-finite mean_ne"):
        select_baseline(candidates)


def test_select_baseline_rejects_missing_mean_ne(tmp_path):
    record = _candidate_record(tmp_path, 1000, 100.0, "a")
    del record["mean_ne"]
    with pytest.raises(ValueError, match="missing mean_ne"):
        select_baseline([record])


def test_select_baseline_rejects_mean_ne_mismatch(tmp_path):
    record = _candidate_record(tmp_path, 1000, 100.0, "a")
    record["mean_ne"] = 999.0
    with pytest.raises(ValueError, match="mean_ne mismatch"):
        select_baseline([record])


def test_select_baseline_rejects_duplicate_step(tmp_path):
    first = _candidate_record(tmp_path, 1000, 100.0, "a")
    second = _candidate_record(tmp_path, 1000, 110.0, "b")
    with pytest.raises(ValueError, match="duplicate step"):
        select_baseline([first, second])


def test_load_metrics_requires_finite_ne_and_n(tmp_path):
    good = tmp_path / "good.json"
    _write_metrics(good, 100.0)
    assert load_metrics(good) == 100.0

    bad_n = tmp_path / "bad_n.json"
    bad_n.write_text('{"NE": 100.0, "n": 0}\n')
    with pytest.raises(ValueError, match="invalid n"):
        load_metrics(bad_n)

    bad_ne = tmp_path / "bad_ne.json"
    bad_ne.write_text('{"NE": "nan", "n": 20.0}\n')
    with pytest.raises(ValueError, match="non-finite NE"):
        load_metrics(bad_ne)

    malformed = tmp_path / "bad.json"
    malformed.write_text("{not json\n")
    with pytest.raises(ValueError, match="invalid metrics JSON"):
        load_metrics(malformed)

    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="metrics file not found"):
        load_metrics(missing)


def test_verify_checkpoint_sha256_pass_and_mismatch(tmp_path):
    ckpt = tmp_path / "model.pt"
    content = b"checkpoint-bytes"
    expected = _write_ckpt(ckpt, content)
    verify_checkpoint_sha256(ckpt, expected)
    assert compute_checkpoint_sha256(ckpt) == expected

    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_checkpoint_sha256(ckpt, "0" * 64)


def test_validate_candidate_rejects_bad_sha_and_missing_paths(tmp_path):
    ckpt = tmp_path / "model.pt"
    _write_ckpt(ckpt, b"weights")
    metrics = tmp_path / "metrics.json"
    _write_metrics(metrics, 120.0)

    with pytest.raises(ValueError, match="invalid sha256"):
        validate_candidate(
            {
                "step": 5000,
                "checkpoint": str(ckpt),
                "metrics_path": str(metrics),
                "mean_ne": 120.0,
                "sha256": "tooshort",
            }
        )

    missing_ckpt = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="checkpoint not found"):
        validate_candidate(
            {
                "step": 5000,
                "checkpoint": str(missing_ckpt),
                "metrics_path": str(metrics),
                "mean_ne": 120.0,
                "sha256": "a" * 64,
            }
        )


def test_build_lock_manifest_rejects_chosen_not_in_candidates(tmp_path):
    chosen = _candidate_record(tmp_path, 5000, 120.0, "c")
    candidates = [
        _candidate_record(tmp_path, 1000, 150.0, "a"),
        _candidate_record(tmp_path, 4000, 130.0, "b"),
    ]
    with pytest.raises(ValueError, match="chosen candidate not in candidates"):
        build_lock_manifest(
            chosen,
            candidates=candidates,
            stamp="x",
            selection_time="2026-07-27T12:00:00+00:00",
        )


def test_parse_candidate_json_validates_and_loads_mean_ne(tmp_path):
    ckpt = tmp_path / "model.pt"
    sha = _write_ckpt(ckpt, b"weights")
    metrics = tmp_path / "metrics.json"
    _write_metrics(metrics, 120.0)

    parsed = parse_candidate_json(
        json.dumps(
            {
                "step": 5000,
                "checkpoint": str(ckpt),
                "metrics_path": str(metrics),
                "sha256": sha,
            }
        )
    )
    assert parsed["step"] == 5000
    assert parsed["mean_ne"] == 120.0


def test_parse_candidate_json_accepts_equals_in_paths(tmp_path):
    ckpt = tmp_path / "weird=path.pt"
    sha = _write_ckpt(ckpt, b"weights")
    metrics = tmp_path / "m=etrics.json"
    _write_metrics(metrics, 120.0)

    parsed = parse_candidate_json(
        json.dumps(
            {
                "step": 5000,
                "checkpoint": str(ckpt),
                "metrics_path": str(metrics),
                "sha256": sha,
            }
        )
    )
    assert parsed["checkpoint"] == str(ckpt)
    assert parsed["metrics_path"] == str(metrics)


def test_cli_rejects_legacy_candidate_flag(tmp_path):
    out = tmp_path / "baseline_lock.manifest.json"
    with pytest.raises(SystemExit):
        main(["--stamp", "x", "--candidate", "1000=a=b=c", "--out", str(out)])


def test_cli_requires_candidate_json(tmp_path):
    out = tmp_path / "baseline_lock.manifest.json"
    with pytest.raises(SystemExit):
        main(["--stamp", "x", "--out", str(out)])


def test_cli_writes_baseline_lock_manifest(tmp_path):
    record_a = _candidate_record(tmp_path, 1000, 150.0, "a")
    record_b = _candidate_record(tmp_path, 4000, 120.0, "b")
    record_c = _candidate_record(tmp_path, 5000, 120.0, "c")
    out = tmp_path / "baseline_lock.manifest.json"

    rc = main(
        [
            "--stamp",
            "test-stamp",
            "--candidate-json",
            json.dumps(
                {
                    "step": record_a["step"],
                    "checkpoint": record_a["checkpoint"],
                    "metrics_path": record_a["metrics_path"],
                    "sha256": record_a["sha256"],
                }
            ),
            "--candidate-json",
            json.dumps(
                {
                    "step": record_b["step"],
                    "checkpoint": record_b["checkpoint"],
                    "metrics_path": record_b["metrics_path"],
                    "sha256": record_b["sha256"],
                }
            ),
            "--candidate-json",
            json.dumps(
                {
                    "step": record_c["step"],
                    "checkpoint": record_c["checkpoint"],
                    "metrics_path": record_c["metrics_path"],
                    "sha256": record_c["sha256"],
                }
            ),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    man = json.loads(out.read_text(encoding="utf-8"))
    assert man["checkpoint"] == record_c["checkpoint"]
    assert man["s1_ne"] == 96.0
    assert man["selection_rule"] == "min_mean_ne_tie_later_step"
    assert man["orchestration_version"] == ORCHESTRATION_VERSION
    datetime.fromisoformat(man["selection_time"])


def test_write_lock_manifest_atomic_roundtrip(tmp_path):
    out = tmp_path / "baseline_lock.manifest.json"
    payload = {"stamp": "x", "baseline_mean_ne": 1.0}
    write_lock_manifest(out, payload)
    assert read_status(out) == payload
    assert list(tmp_path.glob(".status.*")) == []
