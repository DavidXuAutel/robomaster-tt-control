from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.aerial.collapse_fix.labels import (
    build_ce_mask,
    delta_nearest_with_dist,
    prim_ids_from_action_chunk,
    relabel_stop_on_trajectory,
)
from experiments.aerial.collapse_fix.probe_verdict import (
    probe_sensitivity_verdict,
    verdict_from_summary_json,
)
from experiments.aerial.openfly_actions import OPENFLY_PRIMITIVES


def test_delta_nearest_exact_primitives():
    for pid, prim in OPENFLY_PRIMITIVES.items():
        got, dist = delta_nearest_with_dist(np.asarray(prim))
        assert got == pid
        assert dist == pytest.approx(0.0, abs=1e-9)


def test_build_ce_mask_exempts_minority_even_if_far():
    # stop id=0 with huge dist still kept; fwd9 with huge dist dropped
    prim = np.array([0, 9], dtype=np.int64)
    dist = np.array([100.0, 100.0], dtype=np.float64)
    keep = build_ce_mask(prim, dist, d_max=1.0)
    assert keep.tolist() == [True, False]


def test_build_ce_mask_drops_padding():
    prim = np.array([1, 1], dtype=np.int64)
    dist = np.array([0.0, 0.0], dtype=np.float64)
    keep = build_ce_mask(prim, dist, d_max=1.0, is_pad=np.array([True, False]))
    assert keep.tolist() == [False, True]


def test_relabel_stop_near_goal_and_last_frame():
    # path approaches goal then overshoots
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [19.0, 0.0, 0.0],  # within 20m of goal at 30
            [40.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    goal = np.array([30.0, 0.0, 0.0])
    prim = np.array([9, 9, 9, 9], dtype=np.int64)
    out = relabel_stop_on_trajectory(pos, prim, goal=goal, r_stop=20.0, force_last_stop=True)
    assert out[2] == 0
    assert out[-1] == 0
    assert out[0] == 9


def test_prim_ids_from_action_chunk_fwd9():
    actions = np.tile(np.array([[9.0, 0.0, 0.0, 0.0]]), (3, 1))
    ids, dists = prim_ids_from_action_chunk(actions)
    assert ids.tolist() == [9, 9, 9]
    assert np.allclose(dists, 0.0)


def test_probe_verdict_three_way():
    assert probe_sensitivity_verdict([9, 9, 9, 9]) == "insensitive"
    assert probe_sensitivity_verdict([9, 1, 2, 3]) == "sensitive"
    # empty=9, then only one other class → partial
    assert probe_sensitivity_verdict([9, 1, 1, 1]) == "partial"


def test_verdict_from_summary_json(tmp_path: Path):
    summary = tmp_path / "summary.json"
    records = [
        {"instruction": "", "primitive": 9},
        {"instruction": "go north", "primitive": 1},
        {"instruction": "go south", "primitive": 2},
        {"instruction": "hover", "primitive": 0},
    ]
    summary.write_text(json.dumps(records), encoding="utf-8")
    out = verdict_from_summary_json(summary)
    assert out["verdict"] == "sensitive"
    assert out["stage3"]["weaken_first_frame_pin"] is False
