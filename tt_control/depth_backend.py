"""Depth Anything V2 感知后端（瘦客户端）。

推理跑在远端 GPU 服务(server/da_v2_service.py，默认 4090)：
  infer(frame) → POST JPEG → 收到按帧归一化的「近度网格」→ 叠图 + 缓存最新深度。

只依赖标准库 urllib + numpy + opencv（主 venv 已有），不引入 torch。
近度约定：值越大越近/越挡路（服务端已做帧内分位数归一化，见服务脚本）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from tt_control.avoidance import AvoidanceController, AvoidDecision
from tt_control.async_infer import AsyncInferWorker
from tt_control.inference import InferenceBackend

logger = logging.getLogger(__name__)

# 仅供 offline_avoidance 等显式工具默认；飞行入口 main.py 不得静默回落至此
DEFAULT_SERVICE = "https://depth.david-x.com/depth"


@dataclass
class DepthFrame:
    nearness: np.ndarray  # 小网格，float32，值越大越近
    ts: float  # 帧采集墙钟


class DepthAnythingBackend(InferenceBackend):
    def __init__(
        self,
        service_url: str,
        controller: Optional[AvoidanceController] = None,
        jpeg_quality: int = 80,
        timeout: float = 15.0,
        min_interval: float = 0.0,
        overlay: bool = True,
    ) -> None:
        if not service_url:
            raise ValueError(
                "DepthAnythingBackend 需要显式 service_url"
                "（--depth-service 或 --start-depth-service）"
            )
        self.service_url = service_url
        self.controller = controller  # 仅用于叠图标注「此刻会输出什么杆量」
        self.jpeg_quality = int(jpeg_quality)
        self.overlay = overlay

        self._worker = AsyncInferWorker(
            service_url=service_url,
            timeout=timeout,
            jpeg_quality=jpeg_quality,
        )
        self._worker.start()
        self._err: str = ""

    # --- 感知 ---

    def latest_depth(self, max_age_s: float = 5.0) -> Optional[DepthFrame]:
        """主线程调用：返回最新推理结果（非阻塞）。

        result.ts 为帧采集时刻。冻图反复推理不会刷新该时间戳，
        超过 max_age_s 返回 None，让看门狗走「无深度」路径。
        """
        result = self._worker.latest_result()
        if result is None:
            return None
        if time.time() - result.ts > max_age_s:
            return None
        return DepthFrame(nearness=result.nearness, ts=result.ts)

    @property
    def infer_ms(self) -> float:
        """最近一次感知往返耗时（ms），录制器记录 depth_rtt_ms 用。"""
        return self._worker.stats.rtt_ms

    @property
    def last_error(self) -> str:
        return self._worker.stats.last_error

    def infer(self, frame: np.ndarray, frame_ts: Optional[float] = None) -> np.ndarray:
        """主线程调用：喂帧给异步工作线程（非阻塞），返回叠图帧。

        frame_ts: 帧采集墙钟；不传则用当前时间（无法检测冻图）。
        """
        self._worker.feed_frame(frame, ts=frame_ts)

        if not self.overlay:
            return frame
        return self._draw(frame)

    def close(self) -> None:
        """停止异步工作线程。"""
        self._worker.stop()

    @property
    def status_text(self) -> str:
        s = self._worker.stats
        url = self.service_url
        if len(url) > 40:
            url = url[:37] + "..."
        parts = [url, f"RTT {s.rtt_ms:.0f}ms"]
        if s.consecutive_errors > 0:
            parts.append(f"err={s.consecutive_errors}")
        if s.last_frame_age_ms > 0:
            parts.append(f"age={s.last_frame_age_ms:.0f}ms")
        return " | ".join(parts)

    # --- 叠图 ---
    def _draw(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        depth = self.latest_depth()
        if depth is None:
            cv2.putText(
                frame,
                (self.last_error or "waiting depth service...")[:60],
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
        line = f"infer {self.infer_ms:.0f}ms"
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
