"""可插拔实时推理接口。默认透传，后续替换为你的模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InferenceBackend(ABC):
    @abstractmethod
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """输入 BGR 帧，返回叠加/处理后的 BGR 帧。"""


class PassthroughBackend(InferenceBackend):
    """占位：原样返回，保证采帧与显示管线可跑通。"""

    def infer(self, frame: np.ndarray) -> np.ndarray:
        return frame


def create_backend(name: str = "passthrough") -> InferenceBackend:
    name = (name or "passthrough").lower()
    if name in ("passthrough", "none", "identity"):
        return PassthroughBackend()
    raise ValueError(f"未知推理后端: {name}（请实现 InferenceBackend 并在此处注册）")
