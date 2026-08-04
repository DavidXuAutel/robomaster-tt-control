"""进程内 MissionSupervisor：唯一取消所有者 + 按 EventType 路由。

约定单线程 tick；不做多线程消息中间件。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Set

from adapters.dog_base import DogAdapter
from adapters.drone_base import ScoutAdapter
from mission_brain.brain import MissionBrain, MissionState
from mission_brain.bus import EventBus
from mission_brain.events import EventType

logger = logging.getLogger(__name__)


class MissionSupervisor:
    """Brain + Scout + Dog 的线性化入口。"""

    def __init__(
        self,
        bus: EventBus,
        brain: MissionBrain,
        *,
        scout: Optional[ScoutAdapter] = None,
        dog: Optional[DogAdapter] = None,
        operator_sources: Optional[Set[str]] = None,
    ) -> None:
        self.bus = bus
        self.brain = brain
        self.scout = scout
        self.dog = dog
        self.operator_sources = set(operator_sources or {"operator", "op"})
        self._wired = False
        self._stop_done_for_mission: Optional[str] = None
        self.last_stop_errors: list[str] = []

    def wire(self) -> None:
        if self._wired:
            return
        self.bus.subscribe(None, self._route)
        self._wired = True

    def publish_operator(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        et = str(event.get("type", ""))
        if et not in (EventType.MISSION_START.value, EventType.MISSION_ABORT.value):
            raise ValueError(f"operator may only publish start/abort, got {et}")
        src = str(event.get("source", ""))
        if src not in self.operator_sources:
            raise ValueError(f"untrusted operator source: {src}")
        self.wire()
        return self.bus.publish(event)

    def publish(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        """Agent / 测试注入事件（非 operator 专用校验）。"""
        self.wire()
        return self.bus.publish(event)

    def _route(self, event: Mapping[str, Any]) -> None:
        et = str(event.get("type", ""))
        src = str(event.get("source", ""))

        if et == EventType.MISSION_ABORT.value:
            if src not in self.operator_sources:
                logger.warning("drop abort from untrusted source=%s", src)
                return
            self._handle_abort(event)
            return

        if et == EventType.MISSION_START.value:
            if src not in self.operator_sources:
                logger.warning("drop start from untrusted source=%s", src)
                return
            self.brain.handle(event)
            # 新任务启动成功后允许再次 stop
            if self.brain.state is MissionState.SCOUTING:
                self._stop_done_for_mission = None
            return

        if et == EventType.MISSION_FAILED.value:
            mid = str(event.get("mission_id", ""))
            if (
                src == MissionBrain.SOURCE
                and mid
                and mid == self.brain.mission_id
                and self.brain.state is MissionState.SAFE_FAILED
            ):
                self._stop_executors_once(
                    mid,
                    reason=str(event.get("reason", "mission_failed")),
                )
            return

        if et in (
            EventType.DRONE_SCOUT.value,
            EventType.DOG_INSPECT.value,
            EventType.GAS_SAMPLE.value,
        ):
            if src != MissionBrain.SOURCE:
                return
            if self.scout is not None and et == EventType.DRONE_SCOUT.value:
                self.scout.on_brain_event(event)
            if self.dog is not None and et in (
                EventType.DOG_INSPECT.value,
                EventType.GAS_SAMPLE.value,
            ):
                self.dog.on_brain_event(event)
            return

        if et.startswith("drone.") or et.startswith("dog.") or et.startswith("gas."):
            self.brain.handle(event)

    def _handle_abort(self, event: Mapping[str, Any]) -> None:
        mid = str(event.get("mission_id", ""))
        active = self.brain.mission_id
        if (
            not active
            or self.brain.state in (MissionState.IDLE, MissionState.COMPLETE)
            or mid != active
        ):
            # 仍交给 Brain 做忽略/日志；绝不碰执行端
            self.brain.handle(event)
            return
        reason = str(event.get("reason", "abort"))
        self.brain.handle(event)
        self._stop_executors_once(mid, reason=reason)

    def _stop_executors_once(self, mission_id: str, *, reason: str) -> None:
        if not mission_id:
            return
        if self._stop_done_for_mission == mission_id:
            return
        self._stop_done_for_mission = mission_id
        self.last_stop_errors = []
        if self.dog is not None:
            try:
                self.dog.abort(reason)
            except Exception as exc:  # noqa: BLE001 — best-effort stop
                self.last_stop_errors.append(f"dog:{exc}")
                logger.exception("dog.abort failed during stop")
        if self.scout is not None:
            try:
                self.scout.abort(reason)
            except Exception as exc:  # noqa: BLE001
                self.last_stop_errors.append(f"scout:{exc}")
                logger.exception("scout.abort failed during stop")
