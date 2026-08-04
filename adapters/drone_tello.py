"""Tello ScoutAdapter：蓝锚 + 红物体解耦，上报 drone.target_found。

真机飞行仍走现有 main.py / wander；本适配器负责侦察任务契约侧。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from adapters.drone_base import EmitFn, ScoutAdapter
from mission_brain.detect import (
    PRESET_ANCHOR_BLUE,
    PRESET_OBJECT_A_RED,
    MarkerSpec,
    RegionConfirmer,
    detect_marker,
)
from mission_brain.events import EventType, make_event
from mission_brain.map_model import SharedMap

logger = logging.getLogger(__name__)


class TelloScoutAdapter(ScoutAdapter):
    name = "drone_tello"

    def __init__(
        self,
        emit: EmitFn,
        shared_map: SharedMap,
        *,
        target_label: str = "object_a",
        evidence_dir: Optional[str] = None,
        need_anchor_frames: int = 3,
        anchor_mode: str = "color",
        object_spec: MarkerSpec = PRESET_OBJECT_A_RED,
        anchor_color_spec: MarkerSpec = PRESET_ANCHOR_BLUE,
        apriltag_detector: Any = None,
    ) -> None:
        super().__init__(emit, source=self.name)
        self.map = shared_map
        self.default_target_label = target_label
        self.evidence_dir = Path(evidence_dir or "data/mission_evidence")
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.need_anchor_frames = int(need_anchor_frames)
        self.anchor_mode = anchor_mode
        self.object_spec = object_spec
        self.anchor_color_spec = anchor_color_spec
        self.apriltag_detector = apriltag_detector
        self._connected = False
        self._airborne = False
        self._aborted = False
        self._confirmer: Optional[RegionConfirmer] = None
        self._target_label = target_label
        self._reported = False

    def connect(self) -> bool:
        self._connected = True
        logger.info("TelloScoutAdapter ready (control path via existing Tello client)")
        return True

    def takeoff(self) -> bool:
        if not self._connected:
            return False
        self._airborne = True
        return True

    def land(self) -> bool:
        self._airborne = False
        return True

    def abort(self, reason: str) -> None:
        self._aborted = True
        self.active_scout = None
        self._airborne = False
        logger.warning("Tello scout abort: %s", reason)

    def begin_scout(self, command: Mapping[str, Any]) -> None:
        self.mission_id = str(command["mission_id"])
        self._target_label = str(command.get("target_label", self.default_target_label))
        self._aborted = False
        self._reported = False
        region_id = str(command["region_id"])
        region = self.map.get(region_id)
        anchor_map = {a: region_id for a in region.anchor_ids} or {
            "AX-01": region_id
        }
        # 颜色模式：取第一个非 TAG- 锚点作 color_anchor_id；否则 AX-01
        color_anchor = next(
            (a for a in region.anchor_ids if not str(a).startswith("TAG-")),
            region.anchor_ids[0] if region.anchor_ids else "AX-01",
        )
        tag_ids = [
            int(str(a).split("-", 1)[1])
            for a in region.anchor_ids
            if str(a).startswith("TAG-") and str(a).split("-", 1)[1].isdigit()
        ]
        self._confirmer = RegionConfirmer(
            anchor_map,
            need_frames=self.need_anchor_frames,
            mode=self.anchor_mode,
            color_anchor_id=str(color_anchor),
            color_spec=self.anchor_color_spec,
            detector=self.apriltag_detector,
            allowed_tag_ids=tag_ids or None,
        )
        self.active_scout = dict(command)
        logger.info(
            "begin scout region=%s route=%s mode=%s frames=%s",
            region_id,
            command.get("drone_route_id"),
            self.anchor_mode,
            self.need_anchor_frames,
        )

    def process_frame(self, frame: np.ndarray, now: Optional[float] = None) -> None:
        if self._aborted or self.active_scout is None or self._reported:
            return
        if self._confirmer is None or self.mission_id is None:
            return
        t = float(now if now is not None else time.time())
        region_hit = self._confirmer.update(frame)
        if region_hit is None:
            return
        region_id, anchor = region_hit
        obj_spec = MarkerSpec(
            label=self._target_label,
            kind=self.object_spec.kind,
            hsv_ranges=self.object_spec.hsv_ranges,
            tag_ids=self.object_spec.tag_ids,
            min_area_ratio=self.object_spec.min_area_ratio,
            min_confidence=self.object_spec.min_confidence,
        )
        det = detect_marker(frame, obj_spec)
        if det is None or det.confidence < obj_spec.min_confidence:
            return
        evidence_uri = self._save_evidence(frame, region_id, t)
        self._reported = True
        self._emit(
            make_event(
                EventType.DRONE_TARGET_FOUND,
                mission_id=self.mission_id,
                source=self.source,
                causation_id=str(self.active_scout.get("event_id")),
                sent_at=t,
                payload={
                    "region_id": region_id,
                    "target_label": self._target_label,
                    "confidence": float(det.confidence),
                    "anchor_id": anchor.anchor_id,
                    "anchor_age_ms": int(anchor.age_ms),
                    "observed_at": t,
                    "evidence_uri": evidence_uri,
                },
            )
        )

    def report_scout_failed(self, region_id: str, reason: str) -> None:
        if self.mission_id is None:
            return
        self._emit(
            make_event(
                EventType.DRONE_SCOUT_FAILED,
                mission_id=self.mission_id,
                source=self.source,
                payload={"region_id": region_id, "reason": reason},
            )
        )

    def _save_evidence(self, frame: np.ndarray, region_id: str, t: float) -> str:
        name = f"tello_{region_id}_{int(t * 1000)}.jpg"
        path = self.evidence_dir / name
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError(f"failed to write evidence: {path}")
        return str(path)
