"""M0：Brain 防火墙 / 串行 Scout / abort mission 隔离 / 拒覆盖 start。"""

import math

from mission_brain.brain import MissionBrain, MissionState
from mission_brain.bus import RecordingSink
from mission_brain.events import EventType, make_event
from mission_brain.map_model import SharedMap


def _map() -> SharedMap:
    return SharedMap.load("configs/mission/shared_map.example.json")


def _brain(*, now: float = 1000.0, **kwargs):
    sink = RecordingSink()
    brain = MissionBrain(
        _map(),
        sink,
        now_fn=lambda: now,
        stage_timeout_s=kwargs.pop("stage_timeout_s", 10.0),
        sample_window_s=2.0,
        **kwargs,
    )
    return brain, sink


def _start(brain, mid="m1", regions=None, deadline=2000.0, event_id="start-1"):
    ev = make_event(
        EventType.MISSION_START,
        mission_id=mid,
        source="op",
        event_id=event_id,
        sent_at=1000.0,
        payload={
            "target_label": "object_a",
            "region_ids": regions or ["region_x"],
            "deadline": deadline,
        },
    )
    brain.handle(ev)
    return ev


def _found(mid="m1", **payload):
    base = {
        "region_id": "region_x",
        "target_label": "object_a",
        "confidence": 0.95,
        "anchor_id": "AX-01",
        "anchor_age_ms": 0,
        "observed_at": 1000.0,
        "evidence_uri": "e.jpg",
    }
    base.update(payload)
    return make_event(
        EventType.DRONE_TARGET_FOUND,
        mission_id=mid,
        source="drone_tello",
        payload=base,
    )


def test_b_fw_bad_anchor_zero_inspect():
    brain, sink = _brain()
    _start(brain)
    brain.handle(_found(anchor_id="TAG-999"))
    assert brain.state is MissionState.SCOUTING
    assert not any(c["type"] == "dog.inspect" for c in sink.commands)


def test_b_fw_empty_anchor_list_rejects_any():
    """空 anchor_ids 不得放行任意锚点。"""
    from mission_brain.map_model import Region, SharedMap

    sink = RecordingSink()
    m = SharedMap(
        version=1,
        frame="dog_map",
        regions={
            "region_x": Region(
                region_id="region_x",
                dog_goal_id="wp_x",
                drone_route_id="r_x",
                anchor_ids=(),
                label="empty anchors",
            )
        },
    )
    brain = MissionBrain(m, sink, now_fn=lambda: 1000.0)
    brain.handle(
        make_event(
            EventType.MISSION_START,
            mission_id="m1",
            source="op",
            payload={
                "target_label": "object_a",
                "region_ids": ["region_x"],
                "deadline": 2000.0,
            },
        )
    )
    brain.handle(_found(anchor_id="ANY"))
    assert not any(c["type"] == "dog.inspect" for c in sink.commands)


def test_b_fw_label_mismatch():
    brain, sink = _brain()
    _start(brain)
    brain.handle(_found(target_label="other"))
    assert not any(c["type"] == "dog.inspect" for c in sink.commands)


def test_b_fw_age_too_large():
    brain, sink = _brain(max_anchor_age_ms=100.0)
    _start(brain)
    brain.handle(_found(anchor_age_ms=9999))
    assert not any(c["type"] == "dog.inspect" for c in sink.commands)


def test_b_fw_future_observed():
    brain, sink = _brain(now=1000.0, clock_skew_s=0.0)
    _start(brain)
    brain.handle(_found(observed_at=1005.0))
    assert not any(c["type"] == "dog.inspect" for c in sink.commands)


def test_b_fw_nan_confidence_rejected_by_schema():
    """NaN 在事件契约层即拒绝，进不了 Brain 防火墙。"""
    import pytest

    with pytest.raises(ValueError, match="confidence"):
        _found(confidence=float("nan"))
    assert math.isnan(float("nan"))


def test_b_fw_past_deadline_now():
    # now 固定 1000，deadline 1000 → now>deadline 在 tick；found 时 now==deadline 边界：用 now>deadline
    clock = {"t": 1000.0}
    sink = RecordingSink()
    brain = MissionBrain(_map(), sink, now_fn=lambda: clock["t"], freshness_s=5.0)
    _start(brain, deadline=1000.5)
    clock["t"] = 1001.0
    brain.handle(_found(observed_at=1000.0))
    assert not any(c["type"] == "dog.inspect" for c in sink.commands)


def test_b_fw_legal_found_one_inspect():
    brain, sink = _brain()
    _start(brain)
    brain.handle(_found())
    assert len([c for c in sink.commands if c["type"] == "dog.inspect"]) == 1
    assert brain.state is MissionState.DOG_NAV


def test_b_ser_start_only_first_region():
    brain, sink = _brain()
    _start(brain, regions=["region_x", "region_y"])
    scouts = [c for c in sink.commands if c["type"] == "drone.scout"]
    assert len(scouts) == 1
    assert scouts[0]["region_id"] == "region_x"
    assert brain.active_scout_region == "region_x"


def test_b_ser_fail_then_second():
    brain, sink = _brain()
    _start(brain, regions=["region_x", "region_y"])
    brain.handle(
        make_event(
            EventType.DRONE_SCOUT_FAILED,
            mission_id="m1",
            source="drone",
            payload={"region_id": "region_x", "reason": "not_found"},
        )
    )
    scouts = [c for c in sink.commands if c["type"] == "drone.scout"]
    assert len(scouts) == 2
    assert scouts[1]["region_id"] == "region_y"
    assert brain.active_scout_region == "region_y"


def test_b_ser_found_no_second_scout():
    brain, sink = _brain()
    _start(brain, regions=["region_x", "region_y"])
    brain.handle(_found())
    scouts = [c for c in sink.commands if c["type"] == "drone.scout"]
    assert len(scouts) == 1


def test_b_ser_non_active_found_rejected():
    brain, sink = _brain()
    _start(brain, regions=["region_x", "region_y"])
    brain.handle(_found(region_id="region_y", anchor_id="AX-02"))
    assert not any(c["type"] == "dog.inspect" for c in sink.commands)
    assert brain.state is MissionState.SCOUTING


def test_b_ser_stale_failed_ignored():
    brain, sink = _brain()
    _start(brain, regions=["region_x", "region_y"])
    brain.handle(
        make_event(
            EventType.DRONE_SCOUT_FAILED,
            mission_id="m1",
            source="drone",
            payload={"region_id": "region_x", "reason": "not_found"},
        )
    )
    n = len(sink.commands)
    # 迟到的 region_x failed，当前已是 y
    brain.handle(
        make_event(
            EventType.DRONE_SCOUT_FAILED,
            mission_id="m1",
            source="drone",
            event_id="late-x",
            payload={"region_id": "region_x", "reason": "late"},
        )
    )
    assert len(sink.commands) == n
    assert brain.active_scout_region == "region_y"


def test_b_ser_region_timeout_advances():
    clock = {"t": 1000.0}
    sink = RecordingSink()
    brain = MissionBrain(
        _map(),
        sink,
        now_fn=lambda: clock["t"],
        stage_timeout_s=5.0,
    )
    _start(brain, regions=["region_x", "region_y"], deadline=5000.0)
    clock["t"] = 1006.0
    brain.tick(now=1006.0)
    scouts = [c for c in sink.commands if c["type"] == "drone.scout"]
    assert len(scouts) == 2
    assert scouts[1]["region_id"] == "region_y"
    assert brain.state is MissionState.SCOUTING


def test_b_ab_other_mission_ignored():
    brain, sink = _brain()
    _start(brain)
    brain.handle(
        make_event(
            EventType.MISSION_ABORT,
            mission_id="other",
            source="op",
            payload={"reason": "x"},
        )
    )
    assert brain.state is MissionState.SCOUTING
    assert not any(c["type"] == "mission.failed" for c in sink.commands)


def test_b_ab_self():
    brain, sink = _brain()
    _start(brain)
    brain.handle(
        make_event(
            EventType.MISSION_ABORT,
            mission_id="m1",
            source="op",
            payload={"reason": "operator"},
        )
    )
    assert brain.state is MissionState.SAFE_FAILED


def test_b_st_reject_start_while_active():
    brain, sink = _brain()
    _start(brain)
    brain.handle(
        make_event(
            EventType.MISSION_START,
            mission_id="m2",
            source="op",
            event_id="start-2",
            payload={
                "target_label": "object_a",
                "region_ids": ["region_y"],
                "deadline": 3000.0,
            },
        )
    )
    assert brain.mission_id == "m1"
    assert brain.state is MissionState.SCOUTING
    assert brain.active_scout_region == "region_x"
