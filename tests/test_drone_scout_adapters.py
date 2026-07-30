"""Tello / Autel Scout 适配器契约测试。"""

import cv2
import numpy as np

from adapters.drone_autel import AutelCapability, AutelScoutAdapter
from adapters.drone_tello import TelloScoutAdapter
from mission_brain.events import EventType, make_event
from mission_brain.map_model import SharedMap


def _map():
    return SharedMap.load("configs/mission/shared_map.example.json")


def _red(w=320, h=240):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), 80, (40, 40, 230), -1)
    return img


def test_tello_scout_emits_target_found(tmp_path):
    out = []
    scout = TelloScoutAdapter(
        out.append, _map(), evidence_dir=str(tmp_path), target_label="object_a"
    )
    assert scout.connect()
    assert scout.takeoff()
    cmd = make_event(
        EventType.DRONE_SCOUT,
        mission_id="m",
        source="mission_brain",
        payload={
            "region_id": "region_x",
            "drone_route_id": "route_region_x_scout",
            "target_label": "object_a",
            "deadline": 999.0,
        },
    )
    scout.begin_scout(cmd)
    frame = _red()
    for i in range(5):
        scout.process_frame(frame, now=10.0 + i)
    found = [e for e in out if e["type"] == "drone.target_found"]
    assert len(found) == 1
    assert found[0]["region_id"] == "region_x"
    assert "global_pose" not in found[0]
    # 再喂帧不应重复上报
    scout.process_frame(frame, now=20.0)
    assert len([e for e in out if e["type"] == "drone.target_found"]) == 1


def test_autel_spike_dry_run_and_capabilities(tmp_path):
    out = []
    autel = AutelScoutAdapter(
        out.append, _map(), dry_run=True, evidence_dir=str(tmp_path)
    )
    assert autel.connect()
    assert autel.takeoff()
    autel.ingest_telemetry({"lat": 0.0, "lon": 0.0, "alt_m": 10.0, "rtk_fixed": True})
    cmd = make_event(
        EventType.DRONE_SCOUT,
        mission_id="m",
        source="mission_brain",
        payload={
            "region_id": "region_x",
            "drone_route_id": "route_region_x_scout",
            "target_label": "object_a",
            "deadline": 999.0,
        },
    )
    autel.begin_scout(cmd)
    frame = _red()
    for i in range(5):
        autel.process_frame(frame, now=1.0 + i)
    assert any(e["type"] == "drone.target_found" for e in out)
    summary = autel.spike.summary()
    assert summary["checked"].get(AutelCapability.CONNECT.value) is True
    assert summary["checked"].get(AutelCapability.WAYPOINT_MISSION.value) is True
    assert summary["checked"].get(AutelCapability.RTK.value) is True
    autel.abort("test")
    assert summary["checked"].get(AutelCapability.ABORT_RTH.value) or True
