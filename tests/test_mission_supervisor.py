"""M0：Supervisor abort 广播 / mission 隔离 / SAFE_FAILED 停机。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from adapters.dog_sdk import DogSdkAdapter
from adapters.dog_stub import DogStubAdapter
from adapters.drone_tello import TelloScoutAdapter
from mission_brain.brain import MissionBrain, MissionState
from mission_brain.bus import EventBus
from mission_brain.events import EventType, make_event
from mission_brain.map_model import SharedMap
from mission_brain.supervisor import MissionSupervisor


def _map() -> SharedMap:
    return SharedMap.load("configs/mission/shared_map.example.json")


class FakeNav:
    def __init__(self) -> None:
        self.goto_calls = 0
        self.cancel_calls = 0
        self.last_goal_id: Optional[str] = None
        self._arrived = False

    def goto_goal(self, dog_goal_id: str) -> bool:
        self.goto_calls += 1
        self.last_goal_id = dog_goal_id
        return True

    def is_arrived(self) -> bool:
        # cancel 后仍声称 arrived —— 用于验证 abort 锁存
        return True

    def cancel(self) -> None:
        self.cancel_calls += 1


class FakePerception:
    def search_target(self, target_label: str) -> Optional[Dict[str, Any]]:
        return None


class FakeGas:
    def is_connected(self) -> bool:
        return True

    def calibration_at(self) -> float:
        return 0.0

    def sample(self, window_s: float) -> List[Dict[str, Any]]:
        return []


class BoomDog(DogStubAdapter):
    def abort(self, reason: str) -> None:
        super().abort(reason)
        raise RuntimeError("boom-dog")


def _harness(*, boom_dog: bool = False, use_sdk: bool = False):
    bus = EventBus()
    clock = {"t": 1000.0}
    out: List[Dict[str, Any]] = []

    def emit(ev: Dict[str, Any]) -> None:
        out.append(ev)
        bus.publish(ev)

    brain = MissionBrain(
        _map(),
        emit,
        now_fn=lambda: clock["t"],
        stage_timeout_s=30.0,
        freshness_s=30.0,
    )
    scout = TelloScoutAdapter(emit, _map())
    nav = FakeNav()
    if use_sdk:
        dog = DogSdkAdapter(
            emit,
            mode="backend",
            nav=nav,
            perception=FakePerception(),
            gas=FakeGas(),
        )
    elif boom_dog:
        dog = BoomDog(emit, nav_delay_s=0.0, search_delay_s=0.0, sample_delay_s=0.0)
    else:
        dog = DogStubAdapter(
            emit, nav_delay_s=10.0, search_delay_s=10.0, sample_delay_s=10.0
        )
    sup = MissionSupervisor(bus, brain, scout=scout, dog=dog)
    sup.wire()
    return bus, brain, scout, dog, nav, sup, clock, out


def _start(sup, mid="m1", regions=None):
    return sup.publish_operator(
        make_event(
            EventType.MISSION_START,
            mission_id=mid,
            source="operator",
            payload={
                "target_label": "object_a",
                "region_ids": regions or ["region_x"],
                "deadline": 2000.0,
            },
        )
    )


def test_s_ab_01_abort_stops_dog_even_if_nav_arrived():
    _, brain, scout, dog, nav, sup, clock, out = _harness(use_sdk=True)
    _start(sup)
    # 派狗
    brain.handle(
        make_event(
            EventType.DRONE_TARGET_FOUND,
            mission_id="m1",
            source="drone_tello",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "confidence": 0.95,
                "anchor_id": "AX-01",
                "anchor_age_ms": 0,
                "observed_at": 1000.0,
                "evidence_uri": "e.jpg",
            },
        )
    )
    assert nav.goto_calls == 1
    assert brain.state is MissionState.DOG_NAV

    sup.publish_operator(
        make_event(
            EventType.MISSION_ABORT,
            mission_id="m1",
            source="operator",
            payload={"reason": "kill"},
        )
    )
    assert brain.state is MissionState.SAFE_FAILED
    assert nav.cancel_calls == 1
    assert dog.abort_count == 1
    assert scout._aborted is True

    # cancel 后 FakeNav 仍 is_arrived=True，不得再发 dog.arrived
    before = [e for e in out if e["type"] == "dog.arrived"]
    dog.tick(now=1001.0)
    dog.tick(now=1002.0)
    after = [e for e in out if e["type"] == "dog.arrived"]
    assert after == before


def test_s_ab_02_other_mission_abort_does_not_touch_executors():
    _, brain, scout, dog, _, sup, _, _ = _harness()
    _start(sup)
    assert brain.state is MissionState.SCOUTING
    sup.publish_operator(
        make_event(
            EventType.MISSION_ABORT,
            mission_id="m-other",
            source="operator",
            payload={"reason": "wrong"},
        )
    )
    assert brain.state is MissionState.SCOUTING
    assert dog.abort_count == 0
    assert scout._aborted is False


def test_s_ab_03_dog_abort_raises_scout_still_stopped():
    _, brain, scout, dog, _, sup, _, _ = _harness(boom_dog=True)
    _start(sup)
    brain.handle(
        make_event(
            EventType.DRONE_TARGET_FOUND,
            mission_id="m1",
            source="drone_tello",
            payload={
                "region_id": "region_x",
                "target_label": "object_a",
                "confidence": 0.95,
                "anchor_id": "AX-01",
                "anchor_age_ms": 0,
                "observed_at": 1000.0,
                "evidence_uri": "e.jpg",
            },
        )
    )
    sup.publish_operator(
        make_event(
            EventType.MISSION_ABORT,
            mission_id="m1",
            source="operator",
            payload={"reason": "kill"},
        )
    )
    assert brain.state is MissionState.SAFE_FAILED
    assert scout._aborted is True
    assert dog.abort_count == 1
    assert any("dog:" in e for e in sup.last_stop_errors)


def test_s_fail_01_deadline_stops_executors():
    _, brain, scout, dog, _, sup, clock, _ = _harness()
    _start(sup, mid="m1")
    clock["t"] = 2001.0
    brain.tick(now=2001.0)
    assert brain.state is MissionState.SAFE_FAILED
    assert dog.abort_count == 1
    assert scout._aborted is True


def test_untrusted_abort_via_publish_ignored():
    _, brain, scout, dog, _, sup, _, _ = _harness()
    _start(sup)
    sup.publish(
        make_event(
            EventType.MISSION_ABORT,
            mission_id="m1",
            source="untrusted",
            payload={"reason": "spoof"},
        )
    )
    assert brain.state is MissionState.SCOUTING
    assert dog.abort_count == 0
    assert scout._aborted is False


def test_foreign_mission_failed_does_not_stop():
    _, brain, scout, dog, _, sup, _, _ = _harness()
    _start(sup)
    sup.publish(
        make_event(
            EventType.MISSION_FAILED,
            mission_id="m-other",
            source=MissionBrain.SOURCE,
            payload={"stage": "abort", "reason": "spoof"},
        )
    )
    assert brain.state is MissionState.SCOUTING
    assert dog.abort_count == 0
    assert scout._aborted is False


def test_s_ok_01_single_region_happy_stub():
    bus = EventBus()
    clock = {"t": 1000.0}
    out: List[Dict[str, Any]] = []

    def emit(ev: Dict[str, Any]) -> None:
        out.append(ev)
        bus.publish(ev)

    brain = MissionBrain(
        _map(),
        emit,
        now_fn=lambda: clock["t"],
        sample_window_s=0.1,
        freshness_s=60.0,
    )
    scout = TelloScoutAdapter(emit, _map())
    dog = DogStubAdapter(
        emit, nav_delay_s=0.0, search_delay_s=0.0, sample_delay_s=0.0
    )
    sup = MissionSupervisor(bus, brain, scout=scout, dog=dog)
    sup.wire()
    _start(sup)

    import cv2
    import numpy as np

    frame = np.full((720, 960, 3), 40, dtype=np.uint8)
    cv2.circle(frame, (320, 360), 100, (230, 80, 40), -1)
    cv2.circle(frame, (640, 360), 100, (40, 40, 220), -1)
    for i in range(15):
        clock["t"] = 1000.0 + i * 0.1
        scout.process_frame(frame, now=clock["t"])
        dog.tick(now=clock["t"])
        brain.tick(now=clock["t"])
        if brain.state is MissionState.COMPLETE:
            break
    assert brain.state is MissionState.COMPLETE, (brain.state, brain.fail_reason, [e["type"] for e in out])
