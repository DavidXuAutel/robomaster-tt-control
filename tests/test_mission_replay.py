"""G0 契约回放：乱序 / 重复不导致重复导航或采样。"""

from mission_brain.brain import MissionBrain, MissionState
from mission_brain.bus import RecordingSink
from mission_brain.events import EventType, make_event
from mission_brain.map_model import SharedMap


def test_out_of_order_dog_arrived_before_found_ignored():
    sink = RecordingSink()
    brain = MissionBrain(
        SharedMap.load("configs/mission/shared_map.example.json"),
        sink,
        now_fn=lambda: 1.0,
    )
    brain.handle(
        make_event(
            EventType.MISSION_START,
            mission_id="m",
            source="op",
            payload={
                "target_label": "object_a",
                "region_ids": ["region_x"],
                "deadline": 99.0,
            },
        )
    )
    # 乱序：尚未 target_found 就 arrived
    brain.handle(
        make_event(
            EventType.DOG_ARRIVED,
            mission_id="m",
            source="dog",
            payload={
                "region_id": "region_x",
                "dog_goal_id": "wp_region_x_staging",
                "arrived_at": 2.0,
            },
        )
    )
    assert brain.state is MissionState.SCOUTING
    assert not any(c["type"] == "gas.sample" for c in sink.commands)


def test_replay_duplicate_found_single_dog_inspect():
    sink = RecordingSink()
    brain = MissionBrain(
        SharedMap.load("configs/mission/shared_map.example.json"),
        sink,
        now_fn=lambda: 1.0,
    )
    brain.handle(
        make_event(
            EventType.MISSION_START,
            mission_id="m",
            source="op",
            payload={
                "target_label": "object_a",
                "region_ids": ["region_x"],
                "deadline": 99.0,
            },
        )
    )
    found = make_event(
        EventType.DRONE_TARGET_FOUND,
        mission_id="m",
        source="drone",
        event_id="same-found",
        payload={
            "region_id": "region_x",
            "target_label": "object_a",
            "confidence": 0.99,
            "anchor_id": "AX-01",
            "anchor_age_ms": 0,
            "observed_at": 2.0,
            "evidence_uri": "e",
        },
    )
    brain.handle(found)
    brain.handle(dict(found))
    brain.handle(dict(found))
    assert len([c for c in sink.commands if c["type"] == "dog.inspect"]) == 1


def test_abort_works_in_dog_nav():
    sink = RecordingSink()
    brain = MissionBrain(
        SharedMap.load("configs/mission/shared_map.example.json"),
        sink,
        now_fn=lambda: 1.0,
    )
    brain.handle(
        make_event(
            EventType.MISSION_START,
            mission_id="m",
            source="op",
            payload={
                "target_label": "object_a",
                "region_ids": ["region_x"],
                "deadline": 99.0,
            },
        )
    )
    brain.handle(
        make_event(
            EventType.DRONE_TARGET_FOUND,
            mission_id="m",
            source="drone",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "confidence": 0.99,
                "anchor_id": "AX-01",
                "anchor_age_ms": 0,
                "observed_at": 2.0,
                "evidence_uri": "e",
            },
        )
    )
    assert brain.state is MissionState.DOG_NAV
    brain.handle(
        make_event(
            EventType.MISSION_ABORT,
            mission_id="m",
            source="op",
            payload={"reason": "kill"},
        )
    )
    assert brain.state is MissionState.SAFE_FAILED
