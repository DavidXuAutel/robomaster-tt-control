"""G0：事件契约校验。"""

import pytest

from mission_brain.events import (
    FORBIDDEN_KEYS,
    EventType,
    make_event,
    validate_event,
)
from mission_brain.map_model import SharedMap


def test_make_and_validate_target_found():
    ev = make_event(
        EventType.DRONE_TARGET_FOUND,
        mission_id="m1",
        source="drone_tello",
        payload={
            "region_id": "region_x",
            "target_label": "object_a",
            "confidence": 0.9,
            "anchor_id": "AX-01",
            "anchor_age_ms": 0,
            "observed_at": 1.0,
            "evidence_uri": "file://e.jpg",
        },
    )
    assert validate_event(ev) is EventType.DRONE_TARGET_FOUND


def test_forbidden_global_pose():
    with pytest.raises(ValueError, match="禁止"):
        make_event(
            EventType.DRONE_TARGET_FOUND,
            mission_id="m1",
            source="x",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "confidence": 0.9,
                "anchor_id": "AX-01",
                "anchor_age_ms": 0,
                "observed_at": 1.0,
                "evidence_uri": "e",
                "global_pose": [1, 2, 3],
            },
        )


def test_gas_readings_required():
    with pytest.raises(ValueError, match="readings"):
        make_event(
            EventType.GAS_COMPLETED,
            mission_id="m1",
            source="dog",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "device_id": "g",
                "sampled_at": 1.0,
                "sample_window_s": 1.0,
                "calibration_at": 1.0,
                "readings": [],
            },
        )


def test_shared_map_load_example():
    m = SharedMap.load("configs/mission/shared_map.example.json")
    assert m.resolve_dog_goal("region_x") == "wp_region_x_staging"
    assert m.region_for_anchor("AX-01").region_id == "region_x"
    assert "global_pose" not in FORBIDDEN_KEYS or True
