"""Depth Anything V2 感知后端（瘦客户端）。

推理跑在远端 GPU 服务(server/da_v2_service.py，默认 4090)：
  infer(frame) → POST JPEG → 收到按帧归一化的「近度网格」→ 叠图 + 缓存最新深度。

只依赖标准库 urllib + numpy + opencv（主 venv 已有），不引入 torch。
近度约定：值越大越近/越挡路（服务端已做帧内分位数归一化，见服务脚本）。
"""

from __future__ import annotations

import logging
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from tt_control.avoidance import AvoidanceController, AvoidDecision
from tt_control.inference import InferenceBackend

logger = logging.getLogger(__name__)

DEFAULT_SERVICE = "http://10.229.20.125:8899/depth"


@dataclass
class DepthFrame:
    nearness: np.ndarray  # 小网格，float32，值越大越近
    ts: float


class DepthServiceError(RuntimeError):
    pass


class DepthAnythingBackend(InferenceBackend):
    def __init__(
        self,
        service_url: str = DEFAULT_SERVICE,
        controller: Optional[AvoidanceController] = None,
        jpeg_quality: int = 80,
        timeout: float = 2.0,
        min_interval: float = 0.0,
        overlay: bool = True,
    ) -> None:
        self.service_url = service_url
        self.controller = controller  # 仅用于叠图标注「此刻会输出什么杆量」
        self.jpeg_quality = int(jpeg_quality)
        self.timeout = timeout
        self.min_interval = min_interval  # >0 时限流，两次请求间复用上一帧深度
        self.overlay = overlay

        self._lock = threading.Lock()
        self._latest: Optional[DepthFrame] = None
        self._last_req = 0.0
        self._infer_ms = 0.0
        self._err: str = ""
        self._probed = False

    # --- 感知 ---
    def _request_depth(self, frame: np.ndarray) -> np.ndarray:
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise DepthServiceError("JPEG 编码失败")
        req = urllib.request.Request(
            self.service_url,
            data=buf.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except (urllib.error.URLError, OSError) as e:
            raise DepthServiceError(f"连接 {self.service_url} 失败: {e}") from e
        if len(raw) < 8:
            raise DepthServiceError(f"响应过短: {len(raw)}B")
        h, w = struct.unpack("<II", raw[:8])
        expect = 8 + h * w * 2
        if len(raw) != expect:
            raise DepthServiceError(f"响应长度不符: {len(raw)} != {expect}")
        grid = np.frombuffer(raw[8:], dtype=np.float16).reshape(h, w).astype(np.float32)
        return grid

    def latest_depth(self) -> Optional[DepthFrame]:
        with self._lock:
            return self._latest

    @property
    def last_error(self) -> str:
        return self._err

    def infer(self, frame: np.ndarray) -> np.ndarray:
        now = time.time()
        need = self.min_interval <= 0.0 or (now - self._last_req) >= self.min_interval
        if need:
            self._last_req = now
            try:
                t0 = time.time()
                grid = self._request_depth(frame)
                self._infer_ms = (time.time() - t0) * 1000.0
                with self._lock:
                    self._latest = DepthFrame(nearness=grid, ts=now)
                self._err = ""
                self._probed = True
            except DepthServiceError as e:
                self._err = str(e)
                # 首帧就连不上直接抛，避免静默失败；后续偶发错误只记录、复用上一帧
                if not self._probed:
                    raise
                logger.warning("depth infer error: %s", e)

        if not self.overlay:
            return frame
        return self._draw(frame)

    # --- 叠图 ---
    def _draw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        depth = self.latest_depth()
        if depth is None:
            cv2.putText(
                frame,
                (self._err or "waiting depth service...")[:60],
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            return frame

        near = depth.nearness
        big = cv2.resize(near, (w, h), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap((np.clip(big, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        frame[:] = cv2.addWeighted(frame, 0.6, heat, 0.4, 0.0)

        # 左/中/右三区分隔线
        for i in (1, 2):
            x = w * i // 3
            cv2.line(frame, (x, 0), (x, h), (255, 255, 255), 1)

        decision: Optional[AvoidDecision] = None
        if self.controller is not None:
            decision = self.controller.decide(near)
        line = f"infer {self._infer_ms:.0f}ms"
        if decision is not None:
            line = f"{decision.as_hud()}  {line}"
        cv2.rectangle(frame, (0, h - 30), (w, h), (25, 25, 25), -1)
        cv2.putText(
            frame,
            line[:80],
            (10, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60, 255, 120),
            2,
            cv2.LINE_AA,
        )
        return frame
