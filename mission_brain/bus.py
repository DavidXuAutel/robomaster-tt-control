"""进程内事件总线（契约回放 / 单机联调）。生产可换成 MQTT。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional

from mission_brain.events import validate_event

Listener = Callable[[Mapping[str, Any]], None]


class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Listener]] = defaultdict(list)
        self._any: List[Listener] = []
        self.history: List[Dict[str, Any]] = []

    def subscribe(self, event_type: Optional[str], listener: Listener) -> None:
        if event_type is None:
            self._any.append(listener)
        else:
            self._subs[event_type].append(listener)

    def publish(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        validate_event(event)
        stored = dict(event)
        self.history.append(stored)
        for fn in list(self._subs.get(str(stored["type"]), [])):
            fn(stored)
        for fn in list(self._any):
            fn(stored)
        return stored

    def clear_history(self) -> None:
        self.history.clear()

    def of_type(self, event_type: str) -> List[Dict[str, Any]]:
        return [e for e in self.history if e.get("type") == event_type]


class RecordingSink:
    """收集 Brain 发出的命令，便于断言幂等。"""

    def __init__(self) -> None:
        self.commands: List[Dict[str, Any]] = []

    def __call__(self, event: Mapping[str, Any]) -> None:
        self.commands.append(dict(event))

    def types(self) -> List[str]:
        return [str(c["type"]) for c in self.commands]

    def reset(self) -> None:
        self.commands.clear()
