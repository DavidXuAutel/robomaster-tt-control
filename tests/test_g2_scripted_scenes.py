"""M2：synthetic_contract ≥20 场景（参数化生成，非真机账本）。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mission_brain.g2_runner import G2Scene, paint_blue_red, run_all, run_g2_scene
from mission_brain.map_model import SharedMap


def _map():
    return SharedMap.load("configs/mission/shared_map.example.json")


class _FakeTag:
    def __init__(self, tid: int):
        self.tag_id = tid
        self.decision_margin = 20.0


class _FakeDet:
    def __init__(self, tid: int):
        self.tid = tid

    def detect(self, gray):
        return [_FakeTag(self.tid)]


def _build_scenes() -> list[G2Scene]:
    scenes: list[G2Scene] = []
    happy = paint_blue_red()
    for i in range(1, 6):
        scenes.append(
            G2Scene(
                scene_id=f"S0{i}_happy",
                frames=[happy] * 5,
                expect="dispatch_once",
            )
        )
    blue_only = paint_blue_red(red=False)
    for i, sid in enumerate(("S06", "S07", "S08"), start=6):
        scenes.append(
            G2Scene(
                scene_id=f"{sid}_anchor_only",
                frames=[blue_only] * 5,
                expect="no_dispatch",
            )
        )
    red_only = paint_blue_red(blue=False)
    for sid in ("S09", "S10"):
        scenes.append(
            G2Scene(
                scene_id=f"{sid}_object_only",
                frames=[red_only] * 5,
                expect="no_dispatch",
            )
        )
    # streak < need_frames：只喂 2 帧，need=3
    scenes.append(
        G2Scene(
            scene_id="S11_short_streak",
            frames=[happy] * 2,
            expect="no_dispatch",
            need_anchor_frames=3,
        )
    )
    scenes.append(
        G2Scene(
            scene_id="S12_short_streak",
            frames=[happy] * 1,
            expect="no_dispatch",
            need_anchor_frames=3,
        )
    )
    # 低置信：极小红点
    tiny = paint_blue_red(red_radius=2)
    scenes.append(
        G2Scene(scene_id="S13_low_conf", frames=[tiny] * 5, expect="no_dispatch")
    )
    scenes.append(
        G2Scene(scene_id="S14_low_conf", frames=[tiny] * 5, expect="no_dispatch")
    )
    # S15 错误 Tag（图像内容由 FakeDet 决定）
    scenes.append(
        G2Scene(
            scene_id="S15_wrong_tag",
            frames=[np.full((240, 320, 3), 30, dtype=np.uint8)] * 5,
            expect="no_dispatch",
            anchor_mode="apriltag",
            apriltag_detector=_FakeDet(1),  # TAG-1，当前区要 TAG-0
            need_anchor_frames=1,
        )
    )
    scenes.append(
        G2Scene(
            scene_id="S16_duplicate",
            frames=[happy] * 10,
            expect="dispatch_once",
        )
    )
    scenes.append(
        G2Scene(
            scene_id="S17_abort",
            frames=[happy] * 5,
            expect="abort_failed",
            abort_after_frame=1,
        )
    )
    # S18：先耗尽区 x，再区 y 成功
    scenes.append(
        G2Scene(
            scene_id="S18_serial_regions",
            frames=[red_only] * 3 + [paint_blue_red()] * 5,
            expect="dispatch_once",
            region_ids=["region_x", "region_y"],
            notes="runner 内对第一区 scene 中段发 scout_failed",
        )
    )
    # S19 暗光：预先写死 no_dispatch
    dark = paint_blue_red(brightness=5, red_radius=8)
    scenes.append(
        G2Scene(
            scene_id="S19_dark_predeclared_no",
            frames=[dark] * 5,
            expect="no_dispatch",
        )
    )
    # S20 噪声小红斑
    noise = np.full((240, 320, 3), 30, dtype=np.uint8)
    cv2.circle(noise, (10, 10), 2, (40, 40, 230), -1)
    scenes.append(
        G2Scene(scene_id="S20_noise_red", frames=[noise] * 5, expect="no_dispatch")
    )
    assert len(scenes) >= 20
    return scenes


def test_g2_real_apriltag_wrong_tag_no_dispatch(tmp_path):
    """真库 + 官方 Tag1 图：侦察 region_x(要 TAG-0) 不得派狗。"""
    pytest.importorskip("pupil_apriltags")
    from pathlib import Path

    fix = Path(__file__).resolve().parent / "fixtures/mission/apriltags/tag36h11_00001.png"
    g = cv2.imread(str(fix), cv2.IMREAD_GRAYSCALE)
    big = cv2.resize(g, (200, 200), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((400, 400), 255, dtype=np.uint8)
    canvas[100:300, 100:300] = big
    frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    scene = G2Scene(
        scene_id="S15b_real_wrong_tag",
        frames=[frame] * 5,
        expect="no_dispatch",
        anchor_mode="apriltag",
        need_anchor_frames=1,
    )
    r = run_g2_scene(scene, _map(), evidence_dir=str(tmp_path / "S15b"))
    assert r.ok, (r.detail, r.events)


def test_synthetic_contract_20(tmp_path):
    scenes = _build_scenes()
    # S18 特殊路径：手动跑以在中途 fail 第一区
    special = [s for s in scenes if s.scene_id.startswith("S18")][0]
    others = [s for s in scenes if s is not special]

    results = run_all(others, _map(), str(tmp_path / "g2"))
    # S18
    from adapters.dog_stub import DogStubAdapter
    from mission_brain.brain import MissionBrain, MissionState
    from mission_brain.bus import EventBus
    from mission_brain.events import EventType, make_event
    from mission_brain.supervisor import MissionSupervisor
    from adapters.drone_tello import TelloScoutAdapter

    bus = EventBus()
    out = []
    clock = {"t": 1000.0}

    def emit(ev):
        out.append(ev)
        bus.publish(ev)

    m = _map()
    brain = MissionBrain(m, emit, now_fn=lambda: clock["t"], freshness_s=60.0)
    scout = TelloScoutAdapter(emit, m, evidence_dir=str(tmp_path / "S18"))
    dog = DogStubAdapter(emit, nav_delay_s=99.0)
    sup = MissionSupervisor(bus, brain, scout=scout, dog=dog)
    sup.wire()
    mid = "g2-S18"
    sup.publish_operator(
        make_event(
            EventType.MISSION_START,
            mission_id=mid,
            source="operator",
            payload={
                "target_label": "object_a",
                "region_ids": ["region_x", "region_y"],
                "deadline": 5000.0,
            },
        )
    )
    red_only = paint_blue_red(blue=False)
    for i in range(3):
        clock["t"] = 1000 + i
        scout.process_frame(red_only, now=clock["t"])
    scout.report_scout_failed("region_x", "not_found")
    happy = paint_blue_red()
    for i in range(5):
        clock["t"] = 1010 + i
        scout.process_frame(happy, now=clock["t"])
        brain.tick(now=clock["t"])
    inspect = [e for e in out if e["type"] == "dog.inspect"]
    assert len(inspect) == 1
    assert inspect[0]["region_id"] == "region_y"

    failed = [r for r in results if not r.ok]
    assert not failed, [(r.scene_id, r.detail, r.events) for r in failed]
    assert all(r.ledger == "synthetic_contract" for r in results)
    assert len(results) + 1 >= 20
