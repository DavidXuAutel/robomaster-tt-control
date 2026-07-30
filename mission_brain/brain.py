"""MissionBrain：确定性任务 FSM。

IDLE → SCOUTING → DOG_NAV → DOG_SEARCH → GAS_SAMPLE → COMPLETE
失败/超时/abort → SAFE_FAILED

不下发速度杆量；只发任务级命令事件。
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

from mission_brain.events import EventType, make_event, validate_event
from mission_brain.map_model import SharedMap

logger = logging.getLogger(__name__)

EmitFn = Callable[[Dict[str, Any]], None]


class MissionState(str, Enum):
    IDLE = "IDLE"
    SCOUTING = "SCOUTING"
    DOG_NAV = "DOG_NAV"
    DOG_SEARCH = "DOG_SEARCH"
    GAS_SAMPLE = "GAS_SAMPLE"
    COMPLETE = "COMPLETE"
    SAFE_FAILED = "SAFE_FAILED"


class MissionBrain:
    """规则式任务编排。幂等：同一 event_id 只处理一次。"""

    SOURCE = "mission_brain"
    DEFAULT_SAMPLE_WINDOW_S = 5.0
    DEFAULT_STAGE_TIMEOUT_S = 120.0

    def __init__(
        self,
        shared_map: SharedMap,
        emit: EmitFn,
        *,
        now_fn: Optional[Callable[[], float]] = None,
        sample_window_s: float = DEFAULT_SAMPLE_WINDOW_S,
        stage_timeout_s: float = DEFAULT_STAGE_TIMEOUT_S,
        min_confidence: float = 0.6,
    ) -> None:
        self.map = shared_map
        self._emit = emit
        self._now = now_fn or time.time
        self.sample_window_s = float(sample_window_s)
        self.stage_timeout_s = float(stage_timeout_s)
        self.min_confidence = float(min_confidence)

        self.state = MissionState.IDLE
        self.mission_id: Optional[str] = None
        self.target_label: Optional[str] = None
        self.region_ids: List[str] = []
        self.deadline: Optional[float] = None
        self.active_region_id: Optional[str] = None
        self.last_evidence_uri: str = ""
        self.fail_reason: Optional[str] = None

        self._seen_event_ids: Set[str] = set()
        self._stage_entered_at: Optional[float] = None
        self._dispatched_dog_inspect = False
        self._dispatched_gas_sample = False
        self._scout_commands: Set[str] = set()  # region_id already scouted
        self._scout_failures: Set[str] = set()

    def reset(self) -> None:
        self.state = MissionState.IDLE
        self.mission_id = None
        self.target_label = None
        self.region_ids = []
        self.deadline = None
        self.active_region_id = None
        self.last_evidence_uri = ""
        self.fail_reason = None
        self._seen_event_ids.clear()
        self._stage_entered_at = None
        self._dispatched_dog_inspect = False
        self._dispatched_gas_sample = False
        self._scout_commands.clear()
        self._scout_failures.clear()

    def handle(self, event: Mapping[str, Any]) -> None:
        et = validate_event(event)
        eid = str(event["event_id"])
        if eid in self._seen_event_ids:
            logger.debug("ignore duplicate event_id=%s type=%s", eid, et.value)
            return
        self._seen_event_ids.add(eid)

        if et is EventType.MISSION_ABORT:
            self._fail("abort", str(event.get("reason", "abort")), causation_id=eid)
            return

        if et is EventType.HEARTBEAT:
            return

        if self.state in (MissionState.COMPLETE, MissionState.SAFE_FAILED):
            # 终态只接受新 mission.start（换任务）
            if et is not EventType.MISSION_START:
                return

        if et is EventType.MISSION_START:
            self._on_mission_start(event)
            return

        if self.mission_id is None or event.get("mission_id") != self.mission_id:
            logger.warning("drop event for other/missing mission: %s", event.get("type"))
            return

        if et is EventType.DRONE_TARGET_FOUND:
            self._on_drone_target_found(event)
        elif et is EventType.DRONE_SCOUT_FAILED:
            self._on_drone_scout_failed(event)
        elif et is EventType.DOG_ARRIVED:
            self._on_dog_arrived(event)
        elif et is EventType.DOG_TARGET_FOUND:
            self._on_dog_target_found(event)
        elif et is EventType.DOG_INSPECT_FAILED:
            self._fail("dog_inspect", str(event["reason"]), causation_id=eid)
        elif et is EventType.GAS_COMPLETED:
            self._on_gas_completed(event)
        elif et is EventType.GAS_FAILED:
            self._fail("gas", str(event["reason"]), causation_id=eid)

    def tick(self, now: Optional[float] = None) -> None:
        """超时检查；由外部时钟驱动。"""
        if self.state in (
            MissionState.IDLE,
            MissionState.COMPLETE,
            MissionState.SAFE_FAILED,
        ):
            return
        t = float(now if now is not None else self._now())
        if self.deadline is not None and t > float(self.deadline):
            self._fail("deadline", "mission deadline exceeded")
            return
        if (
            self._stage_entered_at is not None
            and (t - self._stage_entered_at) > self.stage_timeout_s
        ):
            self._fail(self.state.value.lower(), f"stage timeout in {self.state.value}")

    # --- transitions ---

    def _enter(self, state: MissionState) -> None:
        self.state = state
        self._stage_entered_at = self._now()

    def _on_mission_start(self, event: Mapping[str, Any]) -> None:
        self.reset()
        self._seen_event_ids.add(str(event["event_id"]))
        self.mission_id = str(event["mission_id"])
        self.target_label = str(event["target_label"])
        self.region_ids = [str(r) for r in event["region_ids"]]
        self.deadline = float(event["deadline"])
        for rid in self.region_ids:
            self.map.get(rid)  # 启动前校验地图
        self._enter(MissionState.SCOUTING)
        for rid in self.region_ids:
            self._emit_scout(rid, causation_id=str(event["event_id"]))

    def _emit_scout(self, region_id: str, *, causation_id: str) -> None:
        if region_id in self._scout_commands:
            return
        assert self.mission_id and self.target_label
        region = self.map.get(region_id)
        self._scout_commands.add(region_id)
        self._emit(
            make_event(
                EventType.DRONE_SCOUT,
                mission_id=self.mission_id,
                source=self.SOURCE,
                causation_id=causation_id,
                payload={
                    "region_id": region_id,
                    "drone_route_id": region.drone_route_id,
                    "target_label": self.target_label,
                    "deadline": self.deadline,
                },
            )
        )

    def _on_drone_target_found(self, event: Mapping[str, Any]) -> None:
        if self.state is not MissionState.SCOUTING:
            return
        conf = float(event["confidence"])
        if conf < self.min_confidence:
            logger.info("ignore low-confidence find conf=%.2f", conf)
            return
        region_id = str(event["region_id"])
        if region_id not in self.region_ids:
            return
        # 地图锚点一致性（若上报了 anchor）
        anchor_id = str(event["anchor_id"])
        region = self.map.get(region_id)
        if region.anchor_ids and anchor_id not in region.anchor_ids:
            logger.warning(
                "anchor %s not in region %s anchors; still accept region_id",
                anchor_id,
                region_id,
            )
        self.active_region_id = region_id
        self.last_evidence_uri = str(event.get("evidence_uri", ""))
        self._enter(MissionState.DOG_NAV)
        self._dispatch_dog_inspect(causation_id=str(event["event_id"]))

    def _dispatch_dog_inspect(self, *, causation_id: str) -> None:
        if self._dispatched_dog_inspect:
            return
        assert self.mission_id and self.target_label and self.active_region_id
        region = self.map.get(self.active_region_id)
        self._dispatched_dog_inspect = True
        self._emit(
            make_event(
                EventType.DOG_INSPECT,
                mission_id=self.mission_id,
                source=self.SOURCE,
                causation_id=causation_id,
                payload={
                    "region_id": self.active_region_id,
                    "dog_goal_id": region.dog_goal_id,
                    "target_label": self.target_label,
                    "evidence_uri": self.last_evidence_uri,
                    "deadline": self.deadline,
                },
            )
        )

    def _on_drone_scout_failed(self, event: Mapping[str, Any]) -> None:
        if self.state is not MissionState.SCOUTING:
            return
        # 全部候选区失败才 fail
        failed_region = str(event["region_id"])
        self._scout_commands.add(failed_region)
        self._scout_failures.add(failed_region)
        if self._scout_failures.issuperset(self.region_ids):
            self._fail(
                "scout",
                "all regions scout_failed",
                causation_id=str(event["event_id"]),
            )

    def _on_dog_arrived(self, event: Mapping[str, Any]) -> None:
        if self.state is not MissionState.DOG_NAV:
            return
        if str(event["region_id"]) != self.active_region_id:
            return
        self._enter(MissionState.DOG_SEARCH)

    def _on_dog_target_found(self, event: Mapping[str, Any]) -> None:
        if self.state is not MissionState.DOG_SEARCH:
            return
        if str(event["region_id"]) != self.active_region_id:
            return
        conf = float(event["confidence"])
        if conf < self.min_confidence:
            return
        self.last_evidence_uri = str(event.get("evidence_uri", self.last_evidence_uri))
        self._enter(MissionState.GAS_SAMPLE)
        self._dispatch_gas_sample(causation_id=str(event["event_id"]))

    def _dispatch_gas_sample(self, *, causation_id: str) -> None:
        if self._dispatched_gas_sample:
            return
        assert self.mission_id and self.target_label and self.active_region_id
        self._dispatched_gas_sample = True
        self._emit(
            make_event(
                EventType.GAS_SAMPLE,
                mission_id=self.mission_id,
                source=self.SOURCE,
                causation_id=causation_id,
                payload={
                    "region_id": self.active_region_id,
                    "target_label": self.target_label,
                    "sample_window_s": self.sample_window_s,
                    "deadline": self.deadline,
                },
            )
        )

    def _on_gas_completed(self, event: Mapping[str, Any]) -> None:
        if self.state is not MissionState.GAS_SAMPLE:
            return
        if str(event["region_id"]) != self.active_region_id:
            return
        assert self.mission_id
        self._enter(MissionState.COMPLETE)
        self._emit(
            make_event(
                EventType.MISSION_COMPLETED,
                mission_id=self.mission_id,
                source=self.SOURCE,
                causation_id=str(event["event_id"]),
                payload={"completed_at": self._now()},
            )
        )

    def _fail(self, stage: str, reason: str, *, causation_id: Optional[str] = None) -> None:
        if self.state in (MissionState.COMPLETE, MissionState.SAFE_FAILED):
            return
        self.fail_reason = reason
        mid = self.mission_id or "unknown"
        self._enter(MissionState.SAFE_FAILED)
        self._emit(
            make_event(
                EventType.MISSION_FAILED,
                mission_id=mid,
                source=self.SOURCE,
                causation_id=causation_id,
                payload={"stage": stage, "reason": reason},
            )
        )
