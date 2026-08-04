"""本地联调：Brain + TelloScout(帧) + DogStub 经 Supervisor 跑通一条任务。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from adapters.dog_stub import DogStubAdapter
from adapters.drone_tello import TelloScoutAdapter
from mission_brain.brain import MissionBrain, MissionState
from mission_brain.bus import EventBus
from mission_brain.events import EventType, make_event
from mission_brain.map_model import SharedMap
from mission_brain.supervisor import MissionSupervisor


def _demo_frame(w: int = 960, h: int = 720) -> np.ndarray:
    """蓝锚 + 红物体（M1 解耦后 demo 帧）。"""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.circle(img, (w // 3, h // 2), 100, (230, 80, 40), -1)
    cv2.circle(img, (2 * w // 3, h // 2), 100, (40, 40, 220), -1)
    return img


def run_demo_mission(
    shared_map: SharedMap,
    *,
    region_ids: Optional[List[str]] = None,
    target_label: str = "object_a",
    wall_clock: bool = False,
) -> Dict[str, Any]:
    """跑一条完整任务；默认虚拟时间加速。"""
    bus = EventBus()
    out: List[Dict[str, Any]] = []

    def emit(ev: Dict[str, Any]) -> None:
        out.append(ev)
        bus.publish(ev)

    t0 = time.time()
    # 用可变时钟，随 now 推进，满足 Brain 新鲜度检查
    clock = {"t": t0}

    brain = MissionBrain(
        shared_map,
        emit,
        now_fn=lambda: clock["t"],
        stage_timeout_s=30.0,
        sample_window_s=1.0,
        freshness_s=30.0,
    )
    scout = TelloScoutAdapter(emit, shared_map, target_label=target_label)
    dog = DogStubAdapter(
        emit,
        nav_delay_s=0.0,
        search_delay_s=0.0,
        sample_delay_s=0.0,
    )
    sup = MissionSupervisor(bus, brain, scout=scout, dog=dog)
    sup.wire()

    regions = region_ids or [shared_map.region_ids()[0]]
    start = make_event(
        EventType.MISSION_START,
        mission_id="demo-mission-1",
        source="operator",
        sent_at=t0,
        payload={
            "target_label": target_label,
            "region_ids": regions,
            "deadline": t0 + 60.0,
        },
    )
    sup.publish_operator(start)

    frame = _demo_frame()
    now = t0
    for _ in range(20):
        now += 0.1
        clock["t"] = now
        scout.process_frame(frame, now=now)
        dog.tick(now=now)
        brain.tick(now=now)
        if wall_clock:
            time.sleep(0.05)
        if brain.state in (MissionState.COMPLETE, MissionState.SAFE_FAILED):
            break

    return {
        "state": brain.state.value,
        "active_region_id": brain.active_region_id,
        "events": [e["type"] for e in out],
        "fail_reason": brain.fail_reason,
    }
