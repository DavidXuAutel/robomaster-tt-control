"""MissionBrain FSM + 幂等 / abort / 超时。"""

from mission_brain.brain import MissionBrain, MissionState
from mission_brain.bus import RecordingSink
from mission_brain.events import EventType, make_event
from mission_brain.map_model import SharedMap


def _map() -> SharedMap:
    return SharedMap.load("configs/mission/shared_map.example.json")


def _brain():
    sink = RecordingSink()
    brain = MissionBrain(
        _map(),
        sink,
        now_fn=lambda: 1000.0,
        stage_timeout_s=10.0,
        sample_window_s=2.0,
    )
    return brain, sink


def _start(brain, mid="m1", regions=None, deadline=2000.0):
    ev = make_event(
        EventType.MISSION_START,
        mission_id=mid,
        source="op",
        event_id="start-1",
        sent_at=1000.0,
        payload={
            "target_label": "object_a",
            "region_ids": regions or ["region_x"],
            "deadline": deadline,
        },
    )
    brain.handle(ev)
    return ev


def test_happy_path_dispatches_once():
    brain, sink = _brain()
    _start(brain)
    assert brain.state is MissionState.SCOUTING
    scouts = [c for c in sink.commands if c["type"] == "drone.scout"]
    assert len(scouts) == 1

    found = make_event(
        EventType.DRONE_TARGET_FOUND,
        mission_id="m1",
        source="drone_tello",
        event_id="found-1",
        payload={
            "region_id": "region_x",
            "target_label": "object_a",
            "confidence": 0.95,
            "anchor_id": "AX-01",
            "anchor_age_ms": 0,
            "observed_at": 1001.0,
            "evidence_uri": "e.jpg",
        },
    )
    brain.handle(found)
    # 重复同一 event_id 不得再派狗
    brain.handle(found)
    dogs = [c for c in sink.commands if c["type"] == "dog.inspect"]
    assert len(dogs) == 1
    assert dogs[0]["dog_goal_id"] == "wp_region_x_staging"
    assert brain.state is MissionState.DOG_NAV

    brain.handle(
        make_event(
            EventType.DOG_ARRIVED,
            mission_id="m1",
            source="dog",
            event_id="arr-1",
            payload={
                "region_id": "region_x",
                "dog_goal_id": "wp_region_x_staging",
                "arrived_at": 1002.0,
            },
        )
    )
    assert brain.state is MissionState.DOG_SEARCH

    brain.handle(
        make_event(
            EventType.DOG_TARGET_FOUND,
            mission_id="m1",
            source="dog",
            event_id="df-1",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "confidence": 0.9,
                "evidence_uri": "dog://e",
            },
        )
    )
    gases = [c for c in sink.commands if c["type"] == "gas.sample"]
    assert len(gases) == 1
    brain.handle(
        make_event(
            EventType.DOG_TARGET_FOUND,
            mission_id="m1",
            source="dog",
            event_id="df-1",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "confidence": 0.9,
                "evidence_uri": "dog://e",
            },
        )
    )
    assert len([c for c in sink.commands if c["type"] == "gas.sample"]) == 1

    brain.handle(
        make_event(
            EventType.GAS_COMPLETED,
            mission_id="m1",
            source="dog",
            event_id="gas-1",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "device_id": "g",
                "sampled_at": 1003.0,
                "sample_window_s": 2.0,
                "calibration_at": 900.0,
                "readings": [
                    {
                        "channel": "CH4",
                        "value": 0.0,
                        "unit": "%LEL",
                        "alarm_state": "ok",
                    }
                ],
            },
        )
    )
    assert brain.state is MissionState.COMPLETE
    assert any(c["type"] == "mission.completed" for c in sink.commands)


def test_abort_from_scouting():
    brain, sink = _brain()
    _start(brain)
    brain.handle(
        make_event(
            EventType.MISSION_ABORT,
            mission_id="m1",
            source="op",
            event_id="ab-1",
            payload={"reason": "operator"},
        )
    )
    assert brain.state is MissionState.SAFE_FAILED
    assert any(c["type"] == "mission.failed" for c in sink.commands)


def test_deadline_timeout():
    brain, sink = _brain()
    _start(brain, deadline=1005.0)
    brain.tick(now=1006.0)
    assert brain.state is MissionState.SAFE_FAILED


def test_duplicate_start_event_id_ignored_after_progress():
    """同一 start event_id 不应在已启动后再次 reset（已在 seen 集合）。"""
    brain, sink = _brain()
    _start(brain)
    n = len(sink.commands)
    brain.handle(
        make_event(
            EventType.MISSION_START,
            mission_id="m1",
            source="op",
            event_id="start-1",
            sent_at=1000.0,
            payload={
                "target_label": "object_a",
                "region_ids": ["region_x"],
                "deadline": 2000.0,
            },
        )
    )
    assert len(sink.commands) == n
