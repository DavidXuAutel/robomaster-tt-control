import json
from pathlib import Path
import subprocess

import pytest

from experiments.aerial.eval.compare_finetune import (
    _lock_thresholds,
    compare_metrics,
    main,
    summarize,
)


def test_summarize_uses_dynamic_s1_threshold():
    report = summarize(
        baseline_ne=150.0,
        cand={"250": 120.01, "500": 120.0},
        s1_ne=120.0,
    )
    assert report["best_step"] == "500"
    assert report["s1_pass"] is True
    assert report["locked_baseline_NE"] == 150.0
    assert report["s1_threshold_NE"] == 120.0

    failed = summarize(
        baseline_ne=150.0,
        cand={"250": 120.01},
        s1_ne=120.0,
    )
    assert failed["s1_pass"] is False
    assert failed["diagnosis"]["s1_threshold_NE"] == 120.0
    assert failed["diagnosis"]["failure_bins"] == [
        "improved_but_below_s1_margin",
        "flat",
        "regressed",
        "quantization_gap",
    ]
    boundary = summarize(
        baseline_ne=150.0,
        cand={"1000": 120.0},
        s1_ne=120.0,
    )
    assert boundary["s1_pass"] is True


def test_compare_metrics_includes_episode_deltas_and_quantization_stats():
    baseline = {
        "NE": 30.0,
        "SR": 0.25,
        "SPL": 0.20,
        "episodes": [
            {"episode_id": "route-a", "NE": 10.0},
            {"episode_id": "route-b", "NE": 30.0},
            {"episode_id": "route-c", "NE": 50.0},
        ],
    }
    candidate = {
        "NE": 25.0,
        "SR": 0.50,
        "SPL": 0.40,
        "episodes": [
            {"episode_id": "route-a", "NE": 8.0},
            {"episode_id": "route-b", "NE": 30.0},
            {"episode_id": "route-c", "NE": 55.0},
        ],
        "quantization_gap_l2": [1.0, 2.0, 3.0],
    }

    report = compare_metrics(
        baseline,
        {"250": candidate},
        baseline_mean_ne=30.0,
        s1_ne=24.0,
    )
    run = report["candidates"]["250"]
    assert run["mean_NE"] == 25.0
    assert run["median_NE"] == 30.0
    assert run["SR"] == 0.50
    assert run["SPL"] == 0.40
    assert run["per_episode_deltas"] == {
        "route-a": -2.0,
        "route-b": 0.0,
        "route-c": 5.0,
    }
    assert run["delta_counts"] == {"improve": 1, "flat": 1, "regress": 1}
    assert run["quantization_gap_l2"] == {
        "count": 3,
        "mean": 2.0,
        "median": 2.0,
        "max": 3.0,
    }


def _write_lock_manifest(
    path: Path,
    *,
    baseline_mean_ne: object = 150.0,
    s1_ne: object = 120.0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "baseline_mean_ne": baseline_mean_ne,
                "s1_ne": s1_ne,
            }
        ),
        encoding="utf-8",
    )


def test_cli_fail_writes_report_and_diagnosis_scaffold(tmp_path: Path):
    baseline = tmp_path / "dynamic_baseline.json"
    candidate = tmp_path / "step_000250.json"
    lock_manifest = tmp_path / "baseline_lock.manifest.json"
    out = tmp_path / "ft_selection_report.json"
    diagnosis = tmp_path / "ft_s1_failure_diagnosis.json"
    baseline.write_text(json.dumps({"NE": 150.0}), encoding="utf-8")
    candidate.write_text(json.dumps({"NE": 120.01}), encoding="utf-8")
    _write_lock_manifest(lock_manifest)

    rc = main(
        [
            "--baseline",
            str(baseline),
            "--lock-manifest",
            str(lock_manifest),
            "--candidate",
            f"250={candidate}",
            "--out",
            str(out),
            "--diagnosis-out",
            str(diagnosis),
        ]
    )

    assert rc == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["s1_pass"] is False
    assert report["locked_baseline_NE"] == 150.0
    assert report["s1_threshold_NE"] == 120.0
    payload = json.loads(diagnosis.read_text(encoding="utf-8"))
    assert payload["auto_expand_data"] is False
    assert payload["start_unseen"] is False


def test_cli_exit_code_uses_manifest_s1_ne(tmp_path: Path):
    baseline = tmp_path / "dynamic_baseline.json"
    candidate = tmp_path / "step_000250.json"
    lock_manifest = tmp_path / "baseline_lock.manifest.json"
    baseline.write_text(json.dumps({"NE": 150.0}), encoding="utf-8")
    candidate.write_text(json.dumps({"NE": 119.0}), encoding="utf-8")
    _write_lock_manifest(lock_manifest, baseline_mean_ne=150.0, s1_ne=120.0)

    rc = main(
        [
            "--baseline",
            str(baseline),
            "--lock-manifest",
            str(lock_manifest),
            "--candidate",
            f"250={candidate}",
            "--out",
            str(tmp_path / "report.json"),
        ]
    )

    assert rc == 0


def test_cli_rejects_baseline_ne_that_disagrees_with_lock_manifest(tmp_path: Path):
    baseline = tmp_path / "dynamic_baseline.json"
    candidate = tmp_path / "step_000250.json"
    lock_manifest = tmp_path / "baseline_lock.manifest.json"
    baseline.write_text(json.dumps({"NE": 149.0}), encoding="utf-8")
    candidate.write_text(json.dumps({"NE": 100.0}), encoding="utf-8")
    _write_lock_manifest(lock_manifest, baseline_mean_ne=150.0)

    with pytest.raises(ValueError, match="locked baseline"):
        main(
            [
                "--baseline",
                str(baseline),
                "--lock-manifest",
                str(lock_manifest),
                "--candidate",
                f"250={candidate}",
                "--out",
                str(tmp_path / "report.json"),
            ]
        )


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"baseline_mean_ne": 150.0},
        {"s1_ne": 120.0},
        {"baseline_mean_ne": "not-a-number", "s1_ne": 120.0},
        {"baseline_mean_ne": 150.0, "s1_ne": float("nan")},
        {"baseline_mean_ne": True, "s1_ne": 120.0},
    ],
)
def test_cli_rejects_malformed_lock_manifest(tmp_path: Path, manifest: dict):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    lock_manifest = tmp_path / "baseline_lock.manifest.json"
    baseline.write_text(json.dumps({"NE": 150.0}), encoding="utf-8")
    candidate.write_text(json.dumps({"NE": 100.0}), encoding="utf-8")
    lock_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="lock manifest"):
        main(
            [
                "--baseline",
                str(baseline),
                "--lock-manifest",
                str(lock_manifest),
                "--candidate",
                f"250={candidate}",
                "--out",
                str(tmp_path / "report.json"),
            ]
        )


def test_lock_thresholds_rejects_s1_that_is_not_eighty_percent():
    with pytest.raises(ValueError, match=r"s1_ne.*0\.8"):
        _lock_thresholds({"baseline_mean_ne": 150.0, "s1_ne": 119.999})


def test_lock_thresholds_accepts_json_roundtrip_ratio():
    baseline_mean_ne = 135.94562291546043
    manifest = json.loads(
        json.dumps(
            {
                "baseline_mean_ne": baseline_mean_ne,
                "s1_ne": 0.8 * baseline_mean_ne,
            }
        )
    )

    assert _lock_thresholds(manifest) == (
        manifest["baseline_mean_ne"],
        manifest["s1_ne"],
    )


def test_eval_script_dry_run_locks_heldout_protocol(tmp_path: Path):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_ft_ckpts_seen20.sh"
    )
    syntax = subprocess.run(
        ["bash", "-n", str(script)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    lock = tmp_path / "baseline_lock.manifest.json"
    baseline = tmp_path / "baseline_metrics.json"
    baseline.write_text('{"NE": 100.0, "n": 20}\n', encoding="utf-8")
    lock.write_text(
        json.dumps(
            {
                "baseline_mean_ne": 100.0,
                "s1_ne": 80.0,
                "metrics_path": str(baseline),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(script), "--dry-run"],
        env={
            "PATH": "/usr/bin:/bin",
            "REPO_DIR": str(tmp_path / "repo"),
            "OPENFLY_ROOT": str(tmp_path / "openfly"),
            "HELDOUT_ANN": str(tmp_path / "heldout_seen20.json"),
            "LOCK_MANIFEST": str(lock),
            "FT_RUN_DIR": str(tmp_path / "ft"),
            "RESULT_DIR": str(tmp_path / "results"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("run_closed_loop") == 3
    for step in (250, 500, 1000):
        assert f"step_{step:06d}.pt" in result.stdout
    assert "--max-episodes 20" in result.stdout
    assert "--max-steps 100" in result.stdout
    assert "--seed 42" in result.stdout
    assert "--task aerial_joint_b1_joint" in result.stdout
    assert "compare_finetune" in result.stdout
    assert "--lock-manifest" in result.stdout
    assert "unseen" not in result.stdout.lower()
