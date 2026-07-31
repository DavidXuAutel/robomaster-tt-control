"""Tello / Autel Scout 适配器契约测试。"""

import cv2
import numpy as np

from adapters.drone_autel import AutelCapability, AutelScoutAdapter, SpikeResult
from adapters.drone_tello import TelloScoutAdapter
from mission_brain.events import EventType, make_event
from mission_brain.map_model import SharedMap


def _map():
    return SharedMap.load("configs/mission/shared_map.example.json")


def _blue_red(w=320, h=240):
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.circle(img, (w // 3, h // 2), 55, (230, 80, 40), -1)  # blue anchor
    cv2.circle(img, (2 * w // 3, h // 2), 55, (40, 40, 230), -1)  # red object
    return img


def test_tello_scout_emits_target_found(tmp_path):
    out = []
    scout = TelloScoutAdapter(
        out.append,
        _map(),
        evidence_dir=str(tmp_path),
        target_label="object_a",
        need_anchor_frames=3,
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
    frame = _blue_red()
    for i in range(5):
        scout.process_frame(frame, now=10.0 + i)
    found = [e for e in out if e["type"] == "drone.target_found"]
    assert len(found) == 1
    assert found[0]["region_id"] == "region_x"
    assert found[0]["anchor_id"] == "AX-01"
    assert "global_pose" not in found[0]
    scout.process_frame(frame, now=20.0)
    assert len([e for e in out if e["type"] == "drone.target_found"]) == 1


def test_tello_need_anchor_frames_honored(tmp_path):
    out = []
    scout = TelloScoutAdapter(
        out.append, _map(), evidence_dir=str(tmp_path), need_anchor_frames=2
    )
    scout.begin_scout(
        make_event(
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
    )
    frame = _blue_red()
    scout.process_frame(frame, now=1.0)
    assert not out
    scout.process_frame(frame, now=2.0)
    assert any(e["type"] == "drone.target_found" for e in out)


def test_tello_red_only_no_found(tmp_path):
    out = []
    scout = TelloScoutAdapter(out.append, _map(), evidence_dir=str(tmp_path))
    scout.begin_scout(
        make_event(
            EventType.DRONE_SCOUT,
            mission_id="m",
            source="mission_brain",
            payload={
                "region_id": "region_x",
                "drone_route_id": "r",
                "target_label": "object_a",
                "deadline": 999.0,
            },
        )
    )
    img = np.full((240, 320, 3), 30, dtype=np.uint8)
    cv2.circle(img, (160, 120), 80, (40, 40, 230), -1)
    for i in range(5):
        scout.process_frame(img, now=1.0 + i)
    assert not any(e["type"] == "drone.target_found" for e in out)


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
    frame = _blue_red()
    for i in range(5):
        autel.process_frame(frame, now=1.0 + i)
    assert any(e["type"] == "drone.target_found" for e in out)
    summary = autel.spike.summary()
    assert summary["results"].get(AutelCapability.CONNECT.value) == SpikeResult.PASS.value
    assert summary["modes"].get(AutelCapability.CONNECT.value) == "simulated"
    assert summary["results"].get(AutelCapability.WAYPOINT_MISSION.value) == SpikeResult.PASS.value
    # dry_run 注入 rtk_fixed 不得记 hardware PASS
    assert summary["results"].get(AutelCapability.RTK.value) == SpikeResult.SKIP.value
    assert summary["modes"].get(AutelCapability.RTK.value) == "simulated"
    autel.abort("test")
    summary2 = autel.spike.summary()
    assert summary2["results"].get(AutelCapability.ABORT_RTH.value) == SpikeResult.PASS.value
    assert summary2["modes"].get(AutelCapability.ABORT_RTH.value) == "simulated"
    assert autel.spike.exit_code(require_hardware=False) == 0
    # 全 simulated / 无真实 device_id → 真机闸门必须非零
    assert autel.spike.exit_code(require_hardware=True) == 2


def test_autel_hardware_exit_rejects_simulated_pass():
    from adapters.drone_autel import AutelSpikeStatus, SpikeResult

    st = AutelSpikeStatus()
    for cap in AutelSpikeStatus.HARDWARE_REQUIRED:
        st.mark(cap, SpikeResult.PASS, mode="simulated", note="fake")
    st.device_id = "autel"
    assert st.exit_code(require_hardware=True) == 2
