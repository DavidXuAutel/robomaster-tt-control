"""G2 合成契约 runner：场景帧 → Scout →（可选）Brain，断言事件序列。

仅 synthetic_contract；不冒充真机 recorded_tello。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from adapters.drone_tello import TelloScoutAdapter
from mission_brain.brain import MissionBrain, MissionState
from mission_brain.bus import EventBus
from mission_brain.events import EventType, make_event
from mission_brain.map_model import SharedMap
from mission_brain.supervisor import MissionSupervisor


@dataclass
class G2Scene:
    scene_id: str
    frames: List[np.ndarray]
    expect: str  # dispatch_once | no_dispatch | abort_failed
    region_ids: List[str] = field(default_factory=lambda: ["region_x"])
    need_anchor_frames: int = 3
    anchor_mode: str = "color"
    apriltag_detector: Any = None
    abort_after_frame: Optional[int] = None
    notes: str = ""


@dataclass
class G2Result:
    scene_id: str
    ok: bool
    events: List[str]
    detail: str
    ledger: str = "synthetic_contract"


def paint_blue_red(
    w: int = 320,
    h: int = 240,
    *,
    blue: bool = True,
    red: bool = True,
    red_radius: int = 55,
    brightness: int = 30,
) -> np.ndarray:
    img = np.full((h, w, 3), brightness, dtype=np.uint8)
    if blue:
        cv2.circle(img, (w // 3, h // 2), 55, (230, 80, 40), -1)
    if red:
        cv2.circle(img, (2 * w // 3, h // 2), red_radius, (40, 40, 230), -1)
    return img


def run_g2_scene(
    scene: G2Scene,
    shared_map: SharedMap,
    *,
    evidence_dir: str,
    with_brain: bool = True,
) -> G2Result:
    bus = EventBus()
    out: List[Dict[str, Any]] = []
    clock = {"t": 1000.0}

    def emit(ev: Dict[str, Any]) -> None:
        out.append(ev)
        bus.publish(ev)

    brain = MissionBrain(
        shared_map,
        emit,
        now_fn=lambda: clock["t"],
        freshness_s=60.0,
        stage_timeout_s=100.0,
    )
    scout = TelloScoutAdapter(
        emit,
        shared_map,
        evidence_dir=evidence_dir,
        need_anchor_frames=scene.need_anchor_frames,
        anchor_mode=scene.anchor_mode,
        apriltag_detector=scene.apriltag_detector,
    )
    dog = None
    # 用轻量 stub 仅在 dispatch 路径；no_dispatch 场景不需要狗
    from adapters.dog_stub import DogStubAdapter

    dog = DogStubAdapter(emit, nav_delay_s=99.0, search_delay_s=99.0)
    sup = MissionSupervisor(bus, brain, scout=scout, dog=dog)
    sup.wire()

    if with_brain:
        sup.publish_operator(
            make_event(
                EventType.MISSION_START,
                mission_id=f"g2-{scene.scene_id}",
                source="operator",
                payload={
                    "target_label": "object_a",
                    "region_ids": list(scene.region_ids),
                    "deadline": 5000.0,
                },
            )
        )
    else:
        scout.begin_scout(
            make_event(
                EventType.DRONE_SCOUT,
                mission_id=f"g2-{scene.scene_id}",
                source="mission_brain",
                payload={
                    "region_id": scene.region_ids[0],
                    "drone_route_id": "r",
                    "target_label": "object_a",
                    "deadline": 5000.0,
                },
            )
        )

    for i, frame in enumerate(scene.frames):
        clock["t"] = 1000.0 + i * 0.1
        if scene.abort_after_frame is not None and i == scene.abort_after_frame:
            if with_brain:
                sup.publish_operator(
                    make_event(
                        EventType.MISSION_ABORT,
                        mission_id=f"g2-{scene.scene_id}",
                        source="operator",
                        payload={"reason": "g2_abort"},
                    )
                )
        scout.process_frame(frame, now=clock["t"])
        brain.tick(now=clock["t"])

    # 负例终点：帧用尽仍无 dispatch → scout_failed（当前区）
    if scene.expect == "no_dispatch" and with_brain:
        if brain.state is MissionState.SCOUTING and brain.active_scout_region:
            scout.report_scout_failed(brain.active_scout_region, "scene_exhausted")
            # report emits via emit→bus→brain
            pass

    types = [e["type"] for e in out]
    inspect_n = types.count("dog.inspect")
    found_n = types.count("drone.target_found")

    if scene.expect == "dispatch_once":
        ok = inspect_n == 1 and found_n == 1
        detail = f"inspect={inspect_n} found={found_n}"
    elif scene.expect == "no_dispatch":
        ok = inspect_n == 0
        detail = f"inspect={inspect_n} state={brain.state.value}"
    elif scene.expect == "abort_failed":
        ok = brain.state is MissionState.SAFE_FAILED and inspect_n == 0
        detail = f"state={brain.state.value} inspect={inspect_n}"
    else:
        ok = False
        detail = f"unknown expect {scene.expect}"

    # evidence 若有 found 必须可解码
    for e in out:
        if e.get("type") == "drone.target_found":
            uri = Path(str(e.get("evidence_uri", "")))
            if not uri.is_file():
                ok = False
                detail += "; missing evidence"
            else:
                img = cv2.imread(str(uri))
                if img is None:
                    ok = False
                    detail += "; evidence undecodable"

    return G2Result(scene_id=scene.scene_id, ok=ok, events=types, detail=detail)


def run_all(
    scenes: Sequence[G2Scene],
    shared_map: SharedMap,
    evidence_root: str,
) -> List[G2Result]:
    root = Path(evidence_root)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for sc in scenes:
        results.append(
            run_g2_scene(sc, shared_map, evidence_dir=str(root / sc.scene_id))
        )
    return results
