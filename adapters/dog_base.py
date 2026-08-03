"""机器狗适配器协议：导航到 dog_goal → 重找 A → 气检。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, Optional

EmitFn = Callable[[Dict[str, Any]], None]


class DogAdapter(ABC):
    name: str = "dog"

    def __init__(self, emit: EmitFn, *, source: Optional[str] = None) -> None:
        self._emit = emit
        self.source = source or self.name
        self.mission_id: Optional[str] = None

    def on_brain_event(self, event: Mapping[str, Any]) -> None:
        # 取消仅由 MissionSupervisor 拥有，不在此监听 mission.abort
        et = str(event.get("type", ""))
        if et == "dog.inspect":
            self.begin_inspect(event)
        elif et == "gas.sample":
            self.begin_gas_sample(event)

    @abstractmethod
    def begin_inspect(self, command: Mapping[str, Any]) -> None:
        ...

    @abstractmethod
    def begin_gas_sample(self, command: Mapping[str, Any]) -> None:
        ...

    @abstractmethod
    def abort(self, reason: str) -> None:
        ...

    @abstractmethod
    def tick(self, now: Optional[float] = None) -> None:
        """推进 stub/仿真时间线或轮询 SDK 状态。"""
        ...
