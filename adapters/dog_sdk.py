"""机器狗 SDK 适配层：统一 dog.inspect / gas.sample 契约。

显式 mode:
  - stub: 明确使用 DogStubAdapter
  - backend: 必须注入 nav+perception+gas，缺一即报错（禁止隐式回退）

三个 backend 协议保持 v1 签名不变。为了让真机适配器（TopseeNav / TopseeGas）
能如实上报平台侧的失败原因，这里用 getattr 探测两个**可选**扩展钩子：

  nav.poll_fault()        -> Optional[str]  导航失败原因
  gas.calibration_reason() -> Optional[str]  标定门禁失败的细分原因

没实现这两个钩子的 backend（如 FakeNav / FakeGas）行为完全不变。
背景见 docs/design/2026-08-03-dog-integration-plan.md §5.1、§5.4。
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
    """真机入口；mode 必须显式指定。"""

    name = "dog_sdk"

    def __init__(
        self,
        emit: EmitFn,
        *,
        mode: str = "stub",
        nav: Optional[NavBackend] = None,
        perception: Optional[PerceptionBackend] = None,
        gas: Optional[GasBackend] = None,
        stub: Optional[DogStubAdapter] = None,
        calibration_max_age_s: float = 7 * 24 * 3600,
    ) -> None:
        super().__init__(emit, source=self.name)
        if mode not in ("stub", "backend"):
            raise ValueError("DogSdkAdapter.mode 必须是 stub 或 backend")
        self.mode = mode
        self.nav = nav
        self.perception = perception
        self.gas = gas
        self.calibration_max_age_s = calibration_max_age_s
        self._use_stub = mode == "stub"
        if mode == "backend":
            missing = [
                n
                for n, b in (("nav", nav), ("perception", perception), ("gas", gas))
                if b is None
            ]
            if missing:
                raise ValueError(
                    f"mode=backend 缺少 backend: {', '.join(missing)}（禁止隐式 stub）"
                )
        self._stub = stub or DogStubAdapter(
            emit,
            nav_delay_s=0.0,
            search_delay_s=0.0,
            sample_delay_s=0.0,
        )
        logger.info("DogSdkAdapter mode=%s", self.mode)

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
            fault = self._nav_fault()
            if fault:
                # 平台状态对不上白名单/查不到任务/导航超时，都不能靠 supervisor
                # 的 stage_timeout_s 兜底掩盖成「狗走得慢」
                self._nav_started = False
                self._emit(
                    make_event(
                        EventType.DOG_INSPECT_FAILED,
                        mission_id=self.mission_id or "",
                        source=self.source,
                        sent_at=t,
                        payload={
                            "region_id": self._inspect["region_id"],
                            "stage": "nav",
                            "reason": fault,
                        },
                    )
                )
                return
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

    def _nav_fault(self) -> Optional[str]:
        """探测 nav 的可选 poll_fault 钩子。未实现则永远返回 None。"""
        hook = getattr(self.nav, "poll_fault", None)
        if hook is None:
            return None
        try:
            fault = hook()
        except Exception:  # noqa: BLE001 — 诊断钩子不许影响主流程
            logger.exception("nav.poll_fault failed; ignored")
            return None
        return str(fault) if fault else None

    def _calibration_reason(self, fallback: str) -> str:
        """探测 gas 的可选 calibration_reason 钩子，拿不到就用旧 reason。"""
        hook = getattr(self.gas, "calibration_reason", None)
        if hook is None:
            return fallback
        try:
            reason = hook()
        except Exception:  # noqa: BLE001 — 同上
            logger.exception("gas.calibration_reason failed; ignored")
            return fallback
        return str(reason) if reason else fallback

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
                    payload={
                        "region_id": region_id,
                        "reason": self._calibration_reason("calibration_stale"),
                    },
                )
            )
            return
        readings = self.gas.sample(float(self._gas_cmd["sample_window_s"]))
        if not readings:
            # 平台只能按时间窗回查历史，窗口内没数据是正常结果；
            # 绝不编造读数去凑 GAS_COMPLETED 的非空校验
            self._sample_done = True
            self._emit(
                make_event(
                    EventType.GAS_FAILED,
                    mission_id=self.mission_id,
                    source=self.source,
                    sent_at=t,
                    payload={"region_id": region_id, "reason": "no_gas_data"},
                )
            )
            return
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
