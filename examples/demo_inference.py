"""示例：自定义推理后端（复制后改模型逻辑）。"""

from __future__ import annotations

import cv2
import numpy as np

from tt_control.inference import InferenceBackend


class DemoOverlayBackend(InferenceBackend):
    """演示用：在画面上画中心十字与标签，不改变飞行逻辑。"""

    def infer(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        h, w = out.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(
            out,
            "DemoInference",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        return out
