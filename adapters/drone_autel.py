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

import numpy as np

from adapters.drone_base import EmitFn, ScoutAdapter
from mission_brain.detect import RegionConfirmer, find_color_blob
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


@dataclass
class AutelSpikeStatus:
    """尖刺验收清单：每项 True 表示已在真机验证。"""

    checked: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def mark(self, cap: AutelCapability, ok: bool, note: str = "") -> None:
        self.checked[cap.value] = bool(ok)
        if note:
            self.notes.append(f"{cap.value}: {note}")

    def summary(self) -> Dict[str, Any]:
        return {"checked": dict(self.checked), "notes": list(self.notes)}


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
    ) -> None:
        super().__init__(emit, source=self.name)
        self.map = shared_map
        self.dry_run = bool(dry_run)
        self.sdk = sdk_backend
        self.default_target_label = target_label
        self.evidence_dir = Path(evidence_dir or "data/mission_evidence")
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
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
            self.spike.mark(AutelCapability.CONNECT, True, "dry_run stub")
            return True
        if self.sdk is None:
            self.spike.mark(AutelCapability.CONNECT, False, "no sdk_backend")
            return False
        ok = bool(self.sdk.connect())
        self._connected = ok
        self.spike.mark(AutelCapability.CONNECT, ok, "sdk_backend")
        return ok

    def takeoff(self) -> bool:
        if not self._connected:
            return False
        if self.dry_run:
            self._airborne = True
            self.spike.mark(AutelCapability.TAKEOFF, True, "dry_run")
            return True
        ok = bool(self.sdk.takeoff())
        self._airborne = ok
        self.spike.mark(AutelCapability.TAKEOFF, ok)
        return ok

    def land(self) -> bool:
        if self.dry_run:
            self._airborne = False
            self.spike.mark(AutelCapability.LAND, True, "dry_run")
            return True
        if self.sdk is None:
            self.spike.mark(AutelCapability.LAND, False, "no sdk")
            return False
        ok = bool(self.sdk.land())
        self._airborne = not ok
        self.spike.mark(AutelCapability.LAND, ok)
        return ok

    def abort(self, reason: str) -> None:
        self._aborted = True
        self.active_scout = None
        if self.dry_run:
            self._airborne = False
            self.spike.mark(AutelCapability.ABORT_RTH, True, f"dry_run:{reason}")
            return
        if self.sdk is not None and hasattr(self.sdk, "rth"):
            self.sdk.rth()
            self.spike.mark(AutelCapability.ABORT_RTH, True, reason)
        else:
            self.spike.mark(AutelCapability.ABORT_RTH, False, "no rth")
        self._airborne = False
        logger.warning("Autel scout abort: %s", reason)

    def start_waypoint_mission(self, route_id: str) -> bool:
        """尖刺：航点/KMZ 任务入口（dry_run 仅记录）。"""
        if self.dry_run:
            self.spike.mark(
                AutelCapability.WAYPOINT_MISSION, True, f"dry_run route={route_id}"
            )
            return True
        if self.sdk is None or not hasattr(self.sdk, "start_mission"):
            self.spike.mark(AutelCapability.WAYPOINT_MISSION, False, "no sdk mission")
            return False
        ok = bool(self.sdk.start_mission(route_id))
        self.spike.mark(AutelCapability.WAYPOINT_MISSION, ok, route_id)
        return ok

    def ingest_telemetry(self, telem: Mapping[str, Any]) -> None:
        """尖刺：接收 GPS/RTK/高度/健康。不含点云。"""
        self.last_telemetry = dict(telem)
        self.spike.mark(AutelCapability.TELEMETRY, True)
        if telem.get("rtk_fixed"):
            self.spike.mark(AutelCapability.RTK, True, "rtk_fixed")

    def begin_scout(self, command: Mapping[str, Any]) -> None:
        self.mission_id = str(command["mission_id"])
        self._target_label = str(command.get("target_label", self.default_target_label))
        self._aborted = False
        self._reported = False
        region_id = str(command["region_id"])
        region = self.map.get(region_id)
        anchor_map = {a: region_id for a in region.anchor_ids} or {"AX-01": region_id}
        color_anchor = region.anchor_ids[0] if region.anchor_ids else "AX-01"
        self._confirmer = RegionConfirmer(
            anchor_map,
            need_frames=3,
            mode="color",
            color_anchor_id=color_anchor,
        )
        self.active_scout = dict(command)
        self.start_waypoint_mission(str(command.get("drone_route_id", "")))

    def process_frame(self, frame: np.ndarray, now: Optional[float] = None) -> None:
        self.spike.mark(AutelCapability.LIVE_FRAME, True, "frame ingested")
        if self._aborted or self.active_scout is None or self._reported:
            return
        if self._confirmer is None or self.mission_id is None:
            return
        t = float(now if now is not None else time.time())
        region_hit = self._confirmer.update(frame)
        if region_hit is None:
            return
        region_id, anchor = region_hit
        det = find_color_blob(frame, label=self._target_label)
        if det is None or det.confidence < 0.55:
            return
        evidence_uri = f"autel://evidence/{region_id}/{int(t * 1000)}"
        # dry_run 不强制写盘；若需要可与 Tello 相同落盘
        try:
            import cv2

            path = self.evidence_dir / f"autel_{region_id}_{int(t * 1000)}.jpg"
            cv2.imwrite(str(path), frame)
            evidence_uri = str(path)
        except Exception:
            pass
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
