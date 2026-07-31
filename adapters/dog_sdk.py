"""机器狗 SDK 适配层：统一 dog.inspect / gas.sample 契约。

真机时注入 NavBackend + GasBackend（厂商 SDK / ROS2 / RS485）。
未注入时回退 DogStubAdapter，保证开发与 G0 不阻塞。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Protocol

from adapters.dog_base import DogAdapter, EmitFn
from adapters.dog_stub import DogStubAdapter
from mission_brain.events import EventType, make_event

logger = logging.getLogger(__name__)


class NavBackend(Protocol):
    def goto_goal(self, dog_goal_id: str) -> bool: ...

    def is_arrived(self) -> bool: ...

    def cancel(self) -> None: ...


class PerceptionBackend(Protocol):
    def search_target(self, target_label: str) -> Optional[Dict[str, Any]]:
        """返回 {confidence, evidence_uri} 或 None。"""
        ...


class GasBackend(Protocol):
    def is_connected(self) -> bool: ...

    def calibration_at(self) -> float: ...

    def sample(self, window_s: float) -> List[Dict[str, Any]]:
        """readings: channel/value/unit/alarm_state。"""
        ...


class DogSdkAdapter(DogAdapter):
    """真机入口。backend 齐全时走 SDK；否则包装 stub。"""

    name = "dog_sdk"

    def __init__(
        self,
        emit: EmitFn,
        *,
        nav: Optional[NavBackend] = None,
        perception: Optional[PerceptionBackend] = None,
        gas: Optional[GasBackend] = None,
        stub: Optional[DogStubAdapter] = None,
        calibration_max_age_s: float = 7 * 24 * 3600,
    ) -> None:
        super().__init__(emit, source=self.name)
        self.nav = nav
        self.perception = perception
        self.gas = gas
        self.calibration_max_age_s = calibration_max_age_s
        self._use_stub = nav is None or perception is None or gas is None
        self._stub = stub or DogStubAdapter(
            emit,
            nav_delay_s=0.0,
            search_delay_s=0.0,
            sample_delay_s=0.0,
        )
        if self._use_stub:
            logger.info("DogSdkAdapter: backends incomplete → using stub")

        self._inspect: Optional[Dict[str, Any]] = None
        self._gas_cmd: Optional[Dict[str, Any]] = None
        self._nav_started = False
        self._arrived_emitted = False
        self._found_emitted = False
        self._sample_done = False
        self._aborted = False
        self.abort_count = 0

    def begin_inspect(self, command: Mapping[str, Any]) -> None:
        if self._use_stub:
            self._stub.begin_inspect(command)
            return
        self._aborted = False
        self.mission_id = str(command["mission_id"])
        self._inspect = dict(command)
        self._nav_started = False
        self._arrived_emitted = False
        self._found_emitted = False
        self._sample_done = False
        self._gas_cmd = None
        ok = self.nav.goto_goal(str(command["dog_goal_id"]))
        self._nav_started = ok
        if not ok:
            self._emit(
                make_event(
                    EventType.DOG_INSPECT_FAILED,
                    mission_id=self.mission_id,
                    source=self.source,
                    payload={
                        "region_id": command["region_id"],
                        "stage": "nav",
                        "reason": "goto_goal_rejected",
                    },
                )
            )

    def begin_gas_sample(self, command: Mapping[str, Any]) -> None:
        if self._use_stub:
            self._stub.begin_gas_sample(command)
            return
        if self._aborted:
            return
        self._gas_cmd = dict(command)
        self._sample_done = False

    def abort(self, reason: str) -> None:
        if self._use_stub:
            self._stub.abort(reason)
            return
        self._aborted = True
        self.abort_count += 1
        self._inspect = None
        self._gas_cmd = None
        self._nav_started = False
        if self.nav is not None:
            try:
                self.nav.cancel()
            except Exception:  # noqa: BLE001 — latch already set
                logger.exception("nav.cancel failed; abort latch kept")
        logger.warning("dog sdk abort: %s", reason)

    def tick(self, now: Optional[float] = None) -> None:
        if self._use_stub:
            self._stub.tick(now)
            return
        if self._aborted:
            return
        t = float(now if now is not None else time.time())
        if self._inspect and self._nav_started and not self._arrived_emitted:
            if self.nav.is_arrived():
                self._arrived_emitted = True
                self._emit(
                    make_event(
                        EventType.DOG_ARRIVED,
                        mission_id=self.mission_id or "",
                        source=self.source,
                        sent_at=t,
                        payload={
                            "region_id": self._inspect["region_id"],
                            "dog_goal_id": self._inspect["dog_goal_id"],
                            "arrived_at": t,
                        },
                    )
                )
        if self._arrived_emitted and not self._found_emitted and self._inspect:
            hit = self.perception.search_target(str(self._inspect["target_label"]))
            if hit is not None:
                self._found_emitted = True
                self._emit(
                    make_event(
                        EventType.DOG_TARGET_FOUND,
                        mission_id=self.mission_id or "",
                        source=self.source,
                        sent_at=t,
                        payload={
                            "region_id": self._inspect["region_id"],
                            "target_label": self._inspect["target_label"],
                            "confidence": float(hit["confidence"]),
                            "evidence_uri": str(hit["evidence_uri"]),
                        },
                    )
                )
        if self._gas_cmd and not self._sample_done:
            self._do_gas(t)

    def _do_gas(self, t: float) -> None:
        assert self._gas_cmd and self.mission_id
        region_id = str(self._gas_cmd["region_id"])
        if not self.gas.is_connected():
            self._sample_done = True
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
        cal = float(self.gas.calibration_at())
        if (t - cal) > self.calibration_max_age_s:
            self._sample_done = True
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
        readings = self.gas.sample(float(self._gas_cmd["sample_window_s"]))
        self._sample_done = True
        self._emit(
            make_event(
                EventType.GAS_COMPLETED,
                mission_id=self.mission_id,
                source=self.source,
                sent_at=t,
                payload={
                    "region_id": region_id,
                    "target_label": self._gas_cmd["target_label"],
                    "device_id": "gas_rs485",
                    "sampled_at": t,
                    "sample_window_s": float(self._gas_cmd["sample_window_s"]),
                    "calibration_at": cal,
                    "readings": readings,
                },
            )
        )
