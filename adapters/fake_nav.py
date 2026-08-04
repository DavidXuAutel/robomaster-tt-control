"""可观测 FakeNav（及最小 Perception/Gas double）— G1 软件预埋，非真机。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class FakeNav:
    def __init__(
        self,
        *,
        arrive_after_ticks: int = 1,
        reject_goto: bool = False,
        raise_on_cancel: bool = False,
    ) -> None:
        self.arrive_after_ticks = int(arrive_after_ticks)
        self.reject_goto = reject_goto
        self.raise_on_cancel = raise_on_cancel
        self.goto_calls = 0
        self.cancel_calls = 0
        self.last_goal_id: Optional[str] = None
        self._ticks = 0
        self._goto_ok = False

    def goto_goal(self, dog_goal_id: str) -> bool:
        self.goto_calls += 1
        self.last_goal_id = dog_goal_id
        self._ticks = 0
        if self.reject_goto:
            self._goto_ok = False
            return False
        self._goto_ok = True
        return True

    def is_arrived(self) -> bool:
        if not self._goto_ok:
            return False
        self._ticks += 1
        return self._ticks >= self.arrive_after_ticks

    def cancel(self) -> None:
        self.cancel_calls += 1
        self._goto_ok = False
        if self.raise_on_cancel:
            raise RuntimeError("cancel boom")


class FakePerception:
    def __init__(self, hit: Optional[Dict[str, Any]] = None) -> None:
        self.hit = hit
        self.calls = 0

    def search_target(self, target_label: str) -> Optional[Dict[str, Any]]:
        self.calls += 1
        if self.hit is None:
            return None
        return dict(self.hit)


class FakeGas:
    def __init__(
        self,
        *,
        connected: bool = True,
        calibration_at: float = 0.0,
        readings: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._connected = connected
        self._calibration_at = calibration_at
        self._readings = readings or [
            {"channel": "CH4", "value": 0.0, "unit": "%LEL", "alarm_state": "ok"}
        ]
        self.sample_calls = 0

    def is_connected(self) -> bool:
        return self._connected

    def calibration_at(self) -> float:
        return float(self._calibration_at)

    def sample(self, window_s: float) -> List[Dict[str, Any]]:
        self.sample_calls += 1
        return list(self._readings)
