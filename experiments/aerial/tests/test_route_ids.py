import json
from pathlib import Path

import pytest

from experiments.aerial.route_ids import assert_disjoint, build_collection40, route_id

FIX = Path(__file__).parent / "fixtures" / "split"


def test_route_id_prefers_stable_fields():
    ep = {
        "trajectory_id": "t1",
        "scene_id": "env_airsim_16",
        "gpt_instruction": "go",
        "image_path": "env_airsim_16/a",
    }
    assert route_id(ep) == "t1"


def test_assert_disjoint_raises():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint({"a", "b"}, {"b"})


def test_build_collection40_no_heldout_leak(tmp_path):
    out_ann = tmp_path / "c40.json"
    out_man = tmp_path / "c40.manifest.json"
    man = build_collection40(
        FIX / "collection_source_mini.json",
        FIX / "heldout_mini.json",
        out_ann,
        out_man,
        seed=42,
        n=3,
    )
    held = {route_id(e) for e in json.loads((FIX / "heldout_mini.json").read_text())}
    got = {route_id(e) for e in json.loads(out_ann.read_text())}
    assert held.isdisjoint(got)
    assert man["n"] == 3
    assert man["seed"] == 42
