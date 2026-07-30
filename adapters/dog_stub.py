"""机器狗 stub：用时间线模拟 navigate → reacquire → gas。

真机前用此完成 G0/G3/G4 契约；SDK 实现见 dog_sdk.py。
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from adapters.dog_base import DogAdapter, EmitFn
from mission_brain.events import EventType, make_event

logger = logging.getLogger(__name__)


class DogStubPhase(str, Enum):
    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    SEARCHING = "SEARCHING"
    WAIT_GAS_CMD = "WAIT_GAS_CMD"
    SAMPLING = "SAMPLING"
    DONE = "DONE"
    FAILED = "FAILED"


class DogStubAdapter(DogAdapter):
    name = "dog_stub"

    def __init__(
        self,
        emit: EmitFn,
        *,
        nav_delay_s: float = 0.5,
        search_delay_s: float = 0.3,
        sample_delay_s: float = 0.2,
        find_confidence: float = 0.9,
        gas_device_id: str = "gas_rs485_stub",
        force_sensor_disconnect: bool = False,
        calibration_stale: bool = False,
        fail_nav: bool = False,
        fail_search: bool = False,
    ) -> None:
        super().__init__(emit, source=self.name)
        self.nav_delay_s = nav_delay_s
        self.search_delay_s = search_delay_s
        self.sample_delay_s = sample_delay_s
        self.find_confidence = find_confidence
        self.gas_device_id = gas_device_id
        self.force_sensor_disconnect = force_sensor_disconnect
        self.calibration_stale = calibration_stale
        self.fail_nav = fail_nav
        self.fail_search = fail_search

        self.phase = DogStubPhase.IDLE
        self._phase_t0: Optional[float] = None
        self._inspect: Optional[Dict[str, Any]] = None
        self._gas_cmd: Optional[Dict[str, Any]] = None
        self._arrived = False
        self._found = False
        self._sampled = False
        self.calibration_at: float = time.time()

    def reset(self) -> None:
        self.phase = DogStubPhase.IDLE
        self._phase_t0 = None
        self._inspect = None
        self._gas_cmd = None
        self._arrived = False
        self._found = False
        self._sampled = False
        self.mission_id = None

    def begin_inspect(self, command: Mapping[str, Any]) -> None:
        self.mission_id = str(command["mission_id"])
        self._inspect = dict(command)
        self._arrived = False
        self._found = False
        self._sampled = False
        self._gas_cmd = None
        self.phase = DogStubPhase.NAVIGATING
        # 相位时钟在首次 tick(now=...) 时锁定，避免墙钟与虚拟时间错位
        self._phase_t0 = None
        logger.info(
            "dog stub navigate to %s (region %s)",
            command.get("dog_goal_id"),
            command.get("region_id"),
        )

    def begin_gas_sample(self, command: Mapping[str, Any]) -> None:
        if self.phase not in (DogStubPhase.WAIT_GAS_CMD, DogStubPhase.SEARCHING):
            # 允许在已找到后接收
            if not self._found:
                return
        self._gas_cmd = dict(command)
        self.phase = DogStubPhase.SAMPLING
        self._phase_t0 = None
        self._sampled = False

    def abort(self, reason: str) -> None:
        self.phase = DogStubPhase.FAILED
        logger.warning("dog stub abort: %s", reason)

    def tick(self, now: Optional[float] = None) -> None:
        t = float(now if now is not None else time.time())
        if self.phase is DogStubPhase.NAVIGATING:
            self._tick_nav(t)
        elif self.phase is DogStubPhase.SEARCHING:
            self._tick_search(t)
        elif self.phase is DogStubPhase.SAMPLING:
            self._tick_sample(t)

    def _tick_nav(self, t: float) -> None:
        assert self._inspect and self.mission_id
        if self._phase_t0 is None:
            self._phase_t0 = t
        if t - self._phase_t0 < self.nav_delay_s:
            return
        if self.fail_nav:
            self.phase = DogStubPhase.FAILED
            self._emit(
                make_event(
                    EventType.DOG_INSPECT_FAILED,
                    mission_id=self.mission_id,
                    source=self.source,
                    sent_at=t,
                    payload={
                        "region_id": self._inspect["region_id"],
                        "stage": "nav",
                        "reason": "nav_failed",
                    },
                )
            )
            return
        if self._arrived:
            return
        self._arrived = True
        self._emit(
            make_event(
                EventType.DOG_ARRIVED,
                mission_id=self.mission_id,
                source=self.source,
                sent_at=t,
                causation_id=str(self._inspect.get("event_id")),
                payload={
                    "region_id": self._inspect["region_id"],
                    "dog_goal_id": self._inspect["dog_goal_id"],
                    "arrived_at": t,
                },
            )
        )
        self.phase = DogStubPhase.SEARCHING
        self._phase_t0 = None

    def _tick_search(self, t: float) -> None:
        assert self._inspect and self.mission_id
        if self._phase_t0 is None:
            self._phase_t0 = t
        if t - self._phase_t0 < self.search_delay_s:
            return
        if self.fail_search:
            self.phase = DogStubPhase.FAILED
            self._emit(
                make_event(
                    EventType.DOG_INSPECT_FAILED,
                    mission_id=self.mission_id,
                    source=self.source,
                    sent_at=t,
                    payload={
                        "region_id": self._inspect["region_id"],
                        "stage": "search",
                        "reason": "target_not_found",
                    },
                )
            )
            return
        if self._found:
            return
        self._found = True
        self._emit(
            make_event(
                EventType.DOG_TARGET_FOUND,
                mission_id=self.mission_id,
                source=self.source,
                sent_at=t,
                causation_id=str(self._inspect.get("event_id")),
                payload={
                    "region_id": self._inspect["region_id"],
                    "target_label": self._inspect["target_label"],
                    "confidence": float(self.find_confidence),
                    "evidence_uri": f"dog://reacquire/{self._inspect['region_id']}",
                },
            )
        )
        # emit 可能重入 begin_gas_sample→SAMPLING；勿覆盖已推进的相位
        if self.phase is DogStubPhase.SEARCHING:
            self.phase = DogStubPhase.WAIT_GAS_CMD

    def _tick_sample(self, t: float) -> None:
        assert self._gas_cmd and self.mission_id
        if self._phase_t0 is None:
            self._phase_t0 = t
        if t - self._phase_t0 < self.sample_delay_s:
            return
        if self._sampled:
            return
        self._sampled = True
        region_id = str(self._gas_cmd["region_id"])
        if self.force_sensor_disconnect:
            self.phase = DogStubPhase.FAILED
            self._emit(
                make_event(
                    EventType.GAS_FAILED,
                    mission_id=self.mission_id,
                    source=self.source,
                    sent_at=t,
                    payload={"region_id": region_id, "reason": "sensor_disconnected"},
                )
            )
            return
        if self.calibration_stale:
            self.phase = DogStubPhase.FAILED
            self._emit(
                make_event(
                    EventType.GAS_FAILED,
                    mission_id=self.mission_id,
                    source=self.source,
                    sent_at=t,
                    payload={"region_id": region_id, "reason": "calibration_stale"},
                )
            )
            return
        readings: List[Dict[str, Any]] = [
            {
                "channel": "CH4",
                "value": 0.0,
                "unit": "%LEL",
                "alarm_state": "ok",
            },
            {
                "channel": "H2S",
                "value": 0.0,
                "unit": "PPM",
                "alarm_state": "ok",
            },
            {
                "channel": "C3H8",
                "value": 0.0,
                "unit": "%LEL",
                "alarm_state": "ok",
            },
        ]
        self._emit(
            make_event(
                EventType.GAS_COMPLETED,
                mission_id=self.mission_id,
                source=self.source,
                sent_at=t,
                causation_id=str(self._gas_cmd.get("event_id")),
                payload={
                    "region_id": region_id,
                    "target_label": self._gas_cmd["target_label"],
                    "device_id": self.gas_device_id,
                    "sampled_at": t,
                    "sample_window_s": float(self._gas_cmd["sample_window_s"]),
                    "calibration_at": float(self.calibration_at),
                    "readings": readings,
                },
            )
        )
        self.phase = DogStubPhase.DONE
