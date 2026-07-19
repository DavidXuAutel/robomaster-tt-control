"""可插拔实时推理接口。默认透传，后续替换为你的模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InferenceEvent:
    """推理后端产生的高层事件；由 App 统一做飞行安全校验。"""

    kind: str  # takeoff | land
    confidence: float
    detail: str = ""


class InferenceBackend(ABC):
    @abstractmethod
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """输入 BGR 帧，返回叠加/处理后的 BGR 帧。"""

    def drain_events(self) -> list[InferenceEvent]:
        """取走自上次调用以来产生的事件。普通推理后端默认无事件。"""
        return []

    @property
    def status_text(self) -> str:
        return ""

    @property
    def training_supported(self) -> bool:
        return False

    @property
    def active_training_label(self) -> str:
        return ""

    def toggle_training(self, label: str) -> str:
        return "Training is not supported by this backend"

    def save_training_profile(self) -> str:
        return "Training is not supported by this backend"

    def close(self) -> None:
        """释放模型资源。"""


class PassthroughBackend(InferenceBackend):
    """占位：原样返回，保证采帧与显示管线可跑通。"""

    def infer(self, frame: np.ndarray) -> np.ndarray:
        return frame


def create_backend(name: str = "passthrough") -> InferenceBackend:
    name = (name or "passthrough").lower()
    if name in ("passthrough", "none", "identity"):
        return PassthroughBackend()
    if name in ("gesture", "gestures", "hand"):
        # 延迟导入，保证不启用手势时无需安装 mediapipe。
        from tt_control.gesture_control import MediaPipeGestureBackend

        return MediaPipeGestureBackend()
    raise ValueError(f"未知推理后端: {name}（请实现 InferenceBackend 并在此处注册）")
