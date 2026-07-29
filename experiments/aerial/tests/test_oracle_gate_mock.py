import hashlib
import json
from pathlib import Path

import pytest

from experiments.aerial.eval.run_oracle_gate import (
    OracleEpisodeResult,
    main,
    oracle_gate_passes,
    run_oracle_episode,
    summarize_oracle_results,
)
from experiments.aerial.eval.run_closed_loop import MockBridge, load_annotation
from experiments.aerial.path_expert import PathExpert


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "mini_openfly"
    / "seen_mini.json"
)


@pytest.mark.parametrize(
    ("sr", "median_ne", "projection_failures", "expected"),
    [
        (0.80, 19.999, 0, True),
        (0.799, 0.0, 0, False),
        (1.0, 20.0, 0, False),
        (1.0, 0.0, 1, False),
    ],
)
def test_oracle_gate_uses_exact_pass_thresholds(
    sr, median_ne, projection_failures, expected
):
    assert oracle_gate_passes(sr, median_ne, projection_failures) is expected


def test_oracle_summary_uses_median_and_pilot_cross_track_p95():
    results = [
        OracleEpisodeResult(True, 1.0, (0.0, 1.0), 0, 2),
        OracleEpisodeResult(True, 3.0, (2.0,), 0, 1),
        OracleEpisodeResult(False, 100.0, (99.0,), 1, 0),
    ]

    report = summarize_oracle_results(results, pilot_episodes=2)

    assert report["SR"] == pytest.approx(2 / 3)
    assert report["median_NE"] == pytest.approx(3.0)
    assert report["projection_failures"] == 1
    assert report["cross_track_p95"] == pytest.approx(1.9)
    assert report["passed"] is False


def test_shadow_policy_is_label_only_and_cannot_control_oracle():
    class BadShadowPolicy:
        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls = 0

        def predict_delta(self, rgb, state, instruction):
            self.calls += 1
            return [0.0, 100.0, 0.0, 0.0]

    shadow = BadShadowPolicy()
    result = run_oracle_episode(
        MockBridge(),
        PathExpert(),
        load_annotation(FIXTURE)[0],
        max_steps=10,
        shadow_policy=shadow,
    )

    assert shadow.calls > 0
    assert result.success is True
    assert result.navigation_error == pytest.approx(0.0)


def test_oracle_cli_writes_gate_and_frozen_collection_manifest(tmp_path):
    gate_path = tmp_path / "oracle_gate.json"
    manifest_path = tmp_path / "collection_manifest.json"

    exit_code = main(
        [
            "--ann",
            str(FIXTURE),
            "--out",
            str(gate_path),
            "--collection-manifest",
            str(manifest_path),
            "--bridge",
            "mock",
            "--max-episodes",
            "3",
            "--pilot-episodes",
            "2",
            "--max-steps",
            "10",
        ]
    )

    assert exit_code == 0
    report = json.loads(gate_path.read_text())
    assert set(
        ("SR", "median_NE", "projection_failures", "cross_track_p95", "passed")
    ) <= report.keys()
    assert report["passed"] is True

    manifest = json.loads(manifest_path.read_text())
    assert manifest["collection_source"]["path"] == str(FIXTURE)
    assert manifest["collection_source"]["sha256"] == hashlib.sha256(
        FIXTURE.read_bytes()
    ).hexdigest()
    assert manifest["pilot"]["episodes"] == 2
    assert manifest["pilot"]["cross_track_p95"] == report["cross_track_p95"]
    assert manifest["thresholds"] == {
        "takeover_m": 9.0,
        "release_m": 6.0,
        "abort_m": 30.0,
        "worsen_steps": 3,
        "stall_steps": 8,
        "release_stable_steps": 3,
        "no_progress_abort_steps": 20,
    }
