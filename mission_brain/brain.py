"""MissionBrain：确定性任务 FSM。

IDLE → SCOUTING → DOG_NAV → DOG_SEARCH → GAS_SAMPLE → COMPLETE
失败/超时/abort → SAFE_FAILED

不下发速度杆量；只发任务级命令事件。
"""

from __future__ import annotations

import logging
import math
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

from mission_brain.events import EventType, make_event, validate_event
from mission_brain.map_model import SharedMap

logger = logging.getLogger(__name__)

EmitFn = Callable[[Dict[str, Any]], None]

_ACTIVE_STATES = frozenset(
    {
        "SCOUTING",
        "DOG_NAV",
        "DOG_SEARCH",
        "GAS_SAMPLE",
    }
)


class MissionState(str, Enum):
    IDLE = "IDLE"
    SCOUTING = "SCOUTING"
    DOG_NAV = "DOG_NAV"
    DOG_SEARCH = "DOG_SEARCH"
    GAS_SAMPLE = "GAS_SAMPLE"
    COMPLETE = "COMPLETE"
    SAFE_FAILED = "SAFE_FAILED"


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


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
        max_anchor_age_ms: float = 5000.0,
        freshness_s: float = 5.0,
        clock_skew_s: float = 1.0,
    ) -> None:
        self.map = shared_map
        self._emit = emit
        self._now = now_fn or time.time
        self.sample_window_s = float(sample_window_s)
        self.stage_timeout_s = float(stage_timeout_s)
        self.min_confidence = float(min_confidence)
        self.max_anchor_age_ms = float(max_anchor_age_ms)
        self.freshness_s = float(freshness_s)
        self.clock_skew_s = float(clock_skew_s)

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
        self._mission_started_at: Optional[float] = None
        self._dispatched_dog_inspect = False
        self._dispatched_gas_sample = False

        self._scout_index: int = -1
        self.active_scout_region: Optional[str] = None
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
        self._mission_started_at = None
        self._dispatched_dog_inspect = False
        self._dispatched_gas_sample = False
        self._scout_index = -1
        self.active_scout_region = None
        self._scout_failures.clear()

    def handle(self, event: Mapping[str, Any]) -> None:
        et = validate_event(event)
        eid = str(event["event_id"])

        if et is EventType.MISSION_ABORT:
            self._on_abort(event)
            return

        if eid in self._seen_event_ids:
            logger.debug("ignore duplicate event_id=%s type=%s", eid, et.value)
            return
        self._seen_event_ids.add(eid)

        if et is EventType.HEARTBEAT:
            return

        if self.state in (MissionState.COMPLETE, MissionState.SAFE_FAILED):
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
            self._fail("deadline", "deadline")
            return
        if (
            self._stage_entered_at is not None
            and (t - self._stage_entered_at) > self.stage_timeout_s
        ):
            if self.state is MissionState.SCOUTING:
                self._on_scout_region_timeout()
            else:
                self._fail(self.state.value.lower(), "stage_timeout")

    # --- transitions ---

    def _enter(self, state: MissionState) -> None:
        self.state = state
        self._stage_entered_at = self._now()

    def _on_abort(self, event: Mapping[str, Any]) -> None:
        eid = str(event["event_id"])
        if self.mission_id is None or self.state is MissionState.IDLE:
            return
        if str(event.get("mission_id")) != self.mission_id:
            logger.info(
                "ignore abort for other mission got=%s active=%s",
                event.get("mission_id"),
                self.mission_id,
            )
            return
        if eid in self._seen_event_ids:
            return
        self._seen_event_ids.add(eid)
        self._fail("abort", str(event.get("reason", "abort")), causation_id=eid)

    def _on_mission_start(self, event: Mapping[str, Any]) -> None:
        if self.state.value in _ACTIVE_STATES:
            logger.warning(
                "reject mission.start while active state=%s mission=%s",
                self.state.value,
                self.mission_id,
            )
            return

        now = float(self._now())
        deadline = float(event["deadline"])
        if not _finite(deadline) or now > deadline:
            # 过期 start：进入失败终态（无 scout）
            self.reset()
            self._seen_event_ids.add(str(event["event_id"]))
            self.mission_id = str(event["mission_id"])
            self.deadline = deadline
            self._fail("start", "deadline", causation_id=str(event["event_id"]))
            return

        self.reset()
        self._seen_event_ids.add(str(event["event_id"]))
        self.mission_id = str(event["mission_id"])
        self.target_label = str(event["target_label"])
        self.region_ids = [str(r) for r in event["region_ids"]]
        self.deadline = deadline
        self._mission_started_at = now
        if not self.region_ids:
            self._fail("start", "no_regions", causation_id=str(event["event_id"]))
            return
        try:
            for rid in self.region_ids:
                self.map.get(rid)
        except KeyError:
            self._fail("start", "unknown_region", causation_id=str(event["event_id"]))
            return
        self._enter(MissionState.SCOUTING)
        self._scout_index = 0
        self._emit_scout(self.region_ids[0], causation_id=str(event["event_id"]))

    def _emit_scout(self, region_id: str, *, causation_id: str) -> None:
        assert self.mission_id and self.target_label
        region = self.map.get(region_id)
        self.active_scout_region = region_id
        self._stage_entered_at = self._now()
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

    def _advance_scout(self, *, causation_id: str, failed_region: str) -> None:
        self._scout_failures.add(failed_region)
        nxt = self._scout_index + 1
        if nxt >= len(self.region_ids):
            self._fail(
                "scout",
                "all_regions_exhausted",
                causation_id=causation_id,
            )
            return
        self._scout_index = nxt
        self._emit_scout(self.region_ids[nxt], causation_id=causation_id)

    def _on_scout_region_timeout(self) -> None:
        if self.active_scout_region is None:
            self._fail("scout", "stage_timeout")
            return
        self._advance_scout(
            causation_id="stage-timeout",
            failed_region=self.active_scout_region,
        )

    def _firewall_ok(self, event: Mapping[str, Any]) -> bool:
        now = float(self._now())
        if self.deadline is not None and now > float(self.deadline):
            logger.info("reject found: now past deadline")
            return False
        conf = event.get("confidence")
        age = event.get("anchor_age_ms")
        observed = event.get("observed_at")
        if not (_finite(conf) and _finite(age) and _finite(observed)):
            logger.info("reject found: non-finite numeric field")
            return False
        conf_f = float(conf)
        age_f = float(age)
        obs_f = float(observed)
        if conf_f < self.min_confidence:
            logger.info("ignore low-confidence find conf=%.2f", conf_f)
            return False
        if age_f < 0 or age_f > self.max_anchor_age_ms:
            logger.info("reject found: bad anchor_age_ms=%s", age_f)
            return False
        if obs_f > now + self.clock_skew_s:
            logger.info("reject found: observed_at in future")
            return False
        if (now - obs_f) > self.freshness_s:
            logger.info("reject found: stale observation")
            return False
        if self.deadline is not None and obs_f > float(self.deadline):
            logger.info("reject found: observed_at past deadline")
            return False
        region_id = str(event["region_id"])
        if region_id != self.active_scout_region:
            logger.info(
                "reject found: region %s != active scout %s",
                region_id,
                self.active_scout_region,
            )
            return False
        if region_id not in self.region_ids:
            return False
        if str(event.get("target_label")) != self.target_label:
            logger.info("reject found: target_label mismatch")
            return False
        anchor_id = str(event["anchor_id"])
        region = self.map.get(region_id)
        if anchor_id not in region.anchor_ids:
            logger.info(
                "reject found: anchor %s not in region %s anchors=%s",
                anchor_id,
                region_id,
                region.anchor_ids,
            )
            return False
        return True

    def _on_drone_target_found(self, event: Mapping[str, Any]) -> None:
        if self.state is not MissionState.SCOUTING:
            return
        if not self._firewall_ok(event):
            return
        region_id = str(event["region_id"])
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
        failed_region = str(event["region_id"])
        if failed_region != self.active_scout_region:
            logger.info(
                "ignore scout_failed for non-active region %s (active=%s)",
                failed_region,
                self.active_scout_region,
            )
            return
        self._advance_scout(
            causation_id=str(event["event_id"]),
            failed_region=failed_region,
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
        conf = event.get("confidence")
        if not _finite(conf) or float(conf) < self.min_confidence:
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
