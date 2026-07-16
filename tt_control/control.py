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
    kind: str  # takeoff | land | emergency | hover | rc | toggle_help | quit | none
    axes: RcAxes = field(default_factory=RcAxes)


HELP_TEXT = [
    "T takeoff | L land | ESC emergency",
    "W/S forward/back  A/D left/right",
    "R/F up/down       Q/E yaw",
    "SPACE hover       H help  X quit",
]


def map_key(key: int, speed: int = 40) -> KeyAction:
    """OpenCV waitKey 返回值 → 动作。"""
    if key < 0:
        return KeyAction("none")

    k = key & 0xFF
    ch = chr(k).lower() if 32 <= k < 127 else ""

    if k == 27:  # Esc
        return KeyAction("emergency")
    if ch == "t":
        return KeyAction("takeoff")
    if ch == "l":
        return KeyAction("land")
    if ch == " " or k == 32:
        return KeyAction("hover")
    if ch == "h":
        return KeyAction("toggle_help")
    # X 退出（Q 留给偏航）
    if ch == "x":
        return KeyAction("quit")

    axes = RcAxes()
    if ch == "a":
        axes.roll = -speed
    elif ch == "d":
        axes.roll = speed
    elif ch == "w":
        axes.pitch = speed
    elif ch == "s":
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
