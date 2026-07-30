"""Scout 平台协议：只上报任务事件，不下发伪全局坐标。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

EmitFn = Callable[[Dict[str, Any]], None]


class ScoutAdapter(ABC):
    """无人机侦察适配器公共接口。"""

    name: str = "scout"

    def __init__(self, emit: EmitFn, *, source: Optional[str] = None) -> None:
        self._emit = emit
        self.source = source or self.name
        self.mission_id: Optional[str] = None
        self.active_scout: Optional[Dict[str, Any]] = None

    def on_brain_event(self, event: Mapping[str, Any]) -> None:
        et = str(event.get("type", ""))
        if et == "drone.scout":
            self.begin_scout(event)
        elif et == "mission.abort":
            self.abort(str(event.get("reason", "abort")))

    @abstractmethod
    def begin_scout(self, command: Mapping[str, Any]) -> None:
        ...

    @abstractmethod
    def abort(self, reason: str) -> None:
        ...

    @abstractmethod
    def process_frame(self, frame: np.ndarray, now: Optional[float] = None) -> None:
        """处理一帧；满足条件时 emit drone.target_found。"""
        ...

    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def takeoff(self) -> bool:
        ...

    @abstractmethod
    def land(self) -> bool:
        ...
