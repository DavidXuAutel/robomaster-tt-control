"""键盘 → 飞行指令映射。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RcAxes:
    roll: int = 0
    pitch: int = 0
    throttle: int = 0
    yaw: int = 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.roll, self.pitch, self.throttle, self.yaw

    def is_zero(self) -> bool:
        return self.roll == self.pitch == self.throttle == self.yaw == 0


@dataclass
class KeyAction:
    kind: str  # takeoff | land | emergency | hover | rc | toggle_help | connect_toggle | quit | none
    axes: RcAxes = field(default_factory=RcAxes)


HELP_TEXT = [
    "1. CONNECT  2. T takeoff  3. hold WASD/RF/QE move",
    "Arrow keys also move | SPACE hover | L land",
    "ESC emergency | C disconnect | H help | X quit",
]

# OpenCV waitKeyEx 方向键（常见值）
_KEY_LEFT = {81, 2, 2424832}
_KEY_UP = {82, 0, 2490368}
_KEY_RIGHT = {83, 3, 2555904}
_KEY_DOWN = {84, 1, 2621440}


def map_key(key: int, speed: int = 40) -> KeyAction:
    """OpenCV waitKey / waitKeyEx 返回值 → 动作。"""
    if key is None or key < 0 or key == 255:
        return KeyAction("none")

    raw = key
    k = key & 0xFF
    ch = chr(k).lower() if 32 <= k < 127 else ""

    if k == 27:  # Esc
        return KeyAction("emergency")
    if ch == "c":
        return KeyAction("connect_toggle")
    if ch == "t":
        return KeyAction("takeoff")
    if ch == "l":
        return KeyAction("land")
    if ch == " " or k == 32:
        return KeyAction("hover")
    if ch == "h":
        return KeyAction("toggle_help")
    if ch == "x":
        return KeyAction("quit")

    axes = RcAxes()
    if ch == "a" or raw in _KEY_LEFT:
        axes.roll = -speed
    elif ch == "d" or raw in _KEY_RIGHT:
        axes.roll = speed
    elif ch == "w" or raw in _KEY_UP:
        axes.pitch = speed
    elif ch == "s" or raw in _KEY_DOWN:
        axes.pitch = -speed
    elif ch == "r":
        axes.throttle = speed
    elif ch == "f":
        axes.throttle = -speed
    elif ch == "q":
        axes.yaw = -speed
    elif ch == "e":
        axes.yaw = speed
    else:
        return KeyAction("none")

    return KeyAction("rc", axes=axes)
