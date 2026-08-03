from __future__ import annotations

from pathlib import Path

from experiments.aerial.orchestration.b1_discover import default_b1_metrics_path
from experiments.aerial.orchestration.eval_queue import metrics_valid
from experiments.aerial.orchestration.state import Phase
from experiments.aerial.orchestration.supervisor import (
    advance_phase,
    b1_metrics_ready,
)


def test_advance_phase_eval_b1_to_s1_when_metrics_ready(tmp_path: Path):
    stamp = "s1"
    results = tmp_path / "results"
    for step in (250, 500, 1000):
        metrics = default_b1_metrics_path(results, stamp, step)
        metrics.parent.mkdir(parents=True)
        metrics.write_text('{"NE": 10.0, "n": 20}\n', encoding="utf-8")
    assert b1_metrics_ready(results_root=results, stamp=stamp) is True
    status = {"phase": Phase.EVAL_B1_CHECKPOINTS.value, "stamp": stamp}
    next_status = advance_phase(status, b1_results_root=results, stamp=stamp)
    assert next_status["phase"] == Phase.S1_REPORT.value


def test_advance_phase_s1_to_done():
    status = {
        "phase": Phase.S1_REPORT.value,
        "stamp": "s1",
        "s1_report_written": True,
        "s1_pass": False,
    }
    next_status = advance_phase(status)
    assert next_status["phase"] == Phase.DONE.value
    assert next_status["s1_pass"] is False


def test_advance_phase_eval_b1_waits_until_ready(tmp_path: Path):
    status = {"phase": Phase.EVAL_B1_CHECKPOINTS.value, "stamp": "s1"}
    next_status = advance_phase(
        status, b1_results_root=tmp_path / "results", stamp="s1"
    )
    assert next_status["phase"] == Phase.EVAL_B1_CHECKPOINTS.value
    assert metrics_valid(tmp_path / "missing.json") is False
