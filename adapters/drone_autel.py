"""Autel 4T Scout 适配尖刺：与 Tello 同一 Scout 契约。

真机 SDK（Mobile SDK / 控制器工作流）因机型固件而异；本模块冻结接口与
状态机，连接/图传在未配置时返回明确未实现，便于 G0/G2 并行推进。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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


class AutelCapability(str, Enum):
    CONNECT = "connect"
    TAKEOFF = "takeoff"
    LAND = "land"
    ABORT_RTH = "abort_rth"
    WAYPOINT_MISSION = "waypoint_mission"
    LIVE_FRAME = "live_frame"
    TELEMETRY = "telemetry"
    RTK = "rtk"


class SpikeResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    NOT_RUN = "NOT_RUN"


@dataclass
class AutelSpikeStatus:
    """尖刺验收：四态 + simulated|hardware，禁止用 dry_run 冒充真机。"""

    results: Dict[str, str] = field(default_factory=dict)
    modes: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    device_id: str = ""
    checked_at: Dict[str, float] = field(default_factory=dict)

    def mark(
        self,
        cap: AutelCapability,
        result: SpikeResult,
        *,
        mode: str,
        note: str = "",
        device_id: str = "",
    ) -> None:
        if mode not in ("simulated", "hardware"):
            raise ValueError("spike mode 必须是 simulated 或 hardware")
        if result is SpikeResult.PASS and mode == "hardware" and not device_id:
            raise ValueError("hardware PASS 必须提供 device_id")
        self.results[cap.value] = result.value
        self.modes[cap.value] = mode
        self.checked_at[cap.value] = time.time()
        if device_id:
            self.device_id = device_id
        if note:
            self.notes.append(f"{cap.value}:{result.value}:{mode}:{note}")

    def summary(self) -> Dict[str, Any]:
        return {
            "results": dict(self.results),
            "modes": dict(self.modes),
            "notes": list(self.notes),
            "device_id": self.device_id,
            "checked_at": dict(self.checked_at),
        }

    # 真机验收至少覆盖这些能力
    HARDWARE_REQUIRED = (
        AutelCapability.CONNECT,
        AutelCapability.TAKEOFF,
        AutelCapability.LAND,
        AutelCapability.ABORT_RTH,
        AutelCapability.LIVE_FRAME,
    )

    def exit_code(self, *, require_hardware: bool = False) -> int:
        """机读：FAIL→1；require_hardware 时须全部关键项 hardware PASS 且有 device_id，否则→2。"""
        if SpikeResult.FAIL.value in self.results.values():
            return 1
        if not require_hardware:
            return 0
        if not self.device_id or self.device_id == "autel":
            # 默认占位不算真机身份
            return 2
        for cap in self.HARDWARE_REQUIRED:
            if self.results.get(cap.value) != SpikeResult.PASS.value:
                return 2
            if self.modes.get(cap.value) != "hardware":
                return 2
        return 0


class AutelScoutAdapter(ScoutAdapter):
    """道通侦察适配器尖刺。

    - 默认 dry_run=True：用注入帧跑与 Tello 相同的发现逻辑（契约对齐）
    - dry_run=False：要求注入 sdk_backend（真机桥）
    """

    name = "drone_autel"

    def __init__(
        self,
        emit: EmitFn,
        shared_map: SharedMap,
        *,
        dry_run: bool = True,
        sdk_backend: Optional[Any] = None,
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
        self.dry_run = bool(dry_run)
        self.sdk = sdk_backend
        self.default_target_label = target_label
        self.evidence_dir = Path(evidence_dir or "data/mission_evidence")
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.need_anchor_frames = int(need_anchor_frames)
        self.anchor_mode = anchor_mode
        self.object_spec = object_spec
        self.anchor_color_spec = anchor_color_spec
        self.apriltag_detector = apriltag_detector
        self.spike = AutelSpikeStatus()
        self._connected = False
        self._airborne = False
        self._aborted = False
        self._confirmer: Optional[RegionConfirmer] = None
        self._target_label = target_label
        self._reported = False
        self.last_telemetry: Dict[str, Any] = {}

    def connect(self) -> bool:
        if self.dry_run:
            self._connected = True
            self.spike.mark(
                AutelCapability.CONNECT,
                SpikeResult.PASS,
                mode="simulated",
                note="dry_run",
            )
            return True
        if self.sdk is None:
            self.spike.mark(
                AutelCapability.CONNECT,
                SpikeResult.NOT_RUN,
                mode="hardware",
                note="no sdk_backend",
            )
            return False
        ok = bool(self.sdk.connect())
        self._connected = ok
        self.spike.mark(
            AutelCapability.CONNECT,
            SpikeResult.PASS if ok else SpikeResult.FAIL,
            mode="hardware",
            note="sdk_backend",
            device_id=str(getattr(self.sdk, "device_id", "autel")),
        )
        return ok

    def takeoff(self) -> bool:
        if not self._connected:
            return False
        if self.dry_run:
            self._airborne = True
            self.spike.mark(
                AutelCapability.TAKEOFF, SpikeResult.PASS, mode="simulated", note="dry_run"
            )
            return True
        ok = bool(self.sdk.takeoff())
        self._airborne = ok
        self.spike.mark(
            AutelCapability.TAKEOFF,
            SpikeResult.PASS if ok else SpikeResult.FAIL,
            mode="hardware",
            device_id=str(getattr(self.sdk, "device_id", "autel")),
        )
        return ok

    def land(self) -> bool:
        if self.dry_run:
            self._airborne = False
            self.spike.mark(
                AutelCapability.LAND, SpikeResult.PASS, mode="simulated", note="dry_run"
            )
            return True
        if self.sdk is None:
            self.spike.mark(
                AutelCapability.LAND, SpikeResult.NOT_RUN, mode="hardware", note="no sdk"
            )
            return False
        ok = bool(self.sdk.land())
        self._airborne = not ok
        self.spike.mark(
            AutelCapability.LAND,
            SpikeResult.PASS if ok else SpikeResult.FAIL,
            mode="hardware",
            device_id=str(getattr(self.sdk, "device_id", "autel")),
        )
        return ok

    def abort(self, reason: str) -> None:
        self._aborted = True
        self.active_scout = None
        if self.dry_run:
            self._airborne = False
            self.spike.mark(
                AutelCapability.ABORT_RTH,
                SpikeResult.PASS,
                mode="simulated",
                note=f"dry_run:{reason}",
            )
            return
        if self.sdk is not None and hasattr(self.sdk, "rth"):
            self.sdk.rth()
            self.spike.mark(
                AutelCapability.ABORT_RTH,
                SpikeResult.PASS,
                mode="hardware",
                note=reason,
                device_id=str(getattr(self.sdk, "device_id", "autel")),
            )
        else:
            self.spike.mark(
                AutelCapability.ABORT_RTH,
                SpikeResult.NOT_RUN,
                mode="hardware",
                note="no rth",
            )
        self._airborne = False
        logger.warning("Autel scout abort: %s", reason)

    def start_waypoint_mission(self, route_id: str) -> bool:
        """尖刺：航点/KMZ 任务入口（dry_run 仅 simulated）。"""
        if self.dry_run:
            self.spike.mark(
                AutelCapability.WAYPOINT_MISSION,
                SpikeResult.PASS,
                mode="simulated",
                note=f"route={route_id}",
            )
            return True
        if self.sdk is None or not hasattr(self.sdk, "start_mission"):
            self.spike.mark(
                AutelCapability.WAYPOINT_MISSION,
                SpikeResult.NOT_RUN,
                mode="hardware",
                note="no sdk mission",
            )
            return False
        ok = bool(self.sdk.start_mission(route_id))
        self.spike.mark(
            AutelCapability.WAYPOINT_MISSION,
            SpikeResult.PASS if ok else SpikeResult.FAIL,
            mode="hardware",
            note=route_id,
            device_id=str(getattr(self.sdk, "device_id", "autel")),
        )
        return ok

    def ingest_telemetry(self, telem: Mapping[str, Any]) -> None:
        """尖刺：接收 GPS/RTK/高度/健康。不含点云。"""
        self.last_telemetry = dict(telem)
        mode = "simulated" if self.dry_run else "hardware"
        self.spike.mark(
            AutelCapability.TELEMETRY,
            SpikeResult.PASS,
            mode=mode,
            note="dict ingest",
            device_id=("" if self.dry_run else str(getattr(self.sdk, "device_id", "autel"))),
        )
        if telem.get("rtk_fixed"):
            # dry_run 注入 rtk_fixed 只算 simulated，不得冒充硬件 RTK
            self.spike.mark(
                AutelCapability.RTK,
                SpikeResult.PASS if not self.dry_run else SpikeResult.SKIP,
                mode=mode,
                note="rtk_fixed field",
                device_id=("" if self.dry_run else str(getattr(self.sdk, "device_id", "autel"))),
            )
        elif AutelCapability.RTK.value not in self.spike.results:
            self.spike.mark(
                AutelCapability.RTK,
                SpikeResult.NOT_RUN,
                mode=mode,
                note="no rtk_fixed",
            )

    def begin_scout(self, command: Mapping[str, Any]) -> None:
        self.mission_id = str(command["mission_id"])
        self._target_label = str(command.get("target_label", self.default_target_label))
        self._aborted = False
        self._reported = False
        region_id = str(command["region_id"])
        region = self.map.get(region_id)
        anchor_map = {a: region_id for a in region.anchor_ids} or {"AX-01": region_id}
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
        self.start_waypoint_mission(str(command.get("drone_route_id", "")))

    def process_frame(self, frame: np.ndarray, now: Optional[float] = None) -> None:
        self.spike.mark(
            AutelCapability.LIVE_FRAME,
            SpikeResult.PASS,
            mode="simulated" if self.dry_run else "hardware",
            note="frame ingested",
            device_id=("" if self.dry_run else str(getattr(self.sdk, "device_id", "autel"))),
        )
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
        path = self.evidence_dir / f"autel_{region_id}_{int(t * 1000)}.jpg"
        ok = cv2.imwrite(str(path), frame)
        evidence_uri = str(path) if ok else f"autel://evidence/{region_id}/{int(t * 1000)}"
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
