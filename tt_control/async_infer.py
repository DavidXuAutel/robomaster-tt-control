"""异步深度推理工作线程。

与 DepthAnythingBackend 配合：主线程喂帧（非阻塞），工作线程异步请求
远端 GPU 服务，结果线程安全地写回。控制循环只读 latest_result()，不阻塞。

用法:
    worker = AsyncInferWorker("http://10.229.20.125:8899/depth")
    worker.start()
    worker.feed_frame(frame)      # 主线程每帧调用
    result = worker.latest_result()  # 主线程读取最新深度
    worker.stop()
"""

from __future__ import annotations

import logging
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cloudflare Browser Integrity Check 会拦默认 Python-urllib UA，须伪装浏览器
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


@dataclass
class DepthResult:
    """单次深度推理结果。"""
    nearness: np.ndarray   # 近度网格, float32, 值越大越近
    ts: float              # 帧采集墙钟（新鲜度以此为准，非推理完成时刻）
    rtt_ms: float          # 往返耗时
    frame_age_ms: float    # 帧采集→请求发起的延迟


@dataclass
class InferStats:
    rtt_ms: float = 0.0
    requests: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    last_error: str = ""
    last_frame_age_ms: float = 0.0


class AsyncInferWorker:
    """异步推理工作线程。

    - 取最新帧 → JPEG 编码 → POST /depth → 解析 nearness grid → 写结果。
    - max_pending=1：上一请求未返回前不发送新请求，避免服务端积压。
    - 超时或错误时保留上一帧有效结果，记录统计。
    """

    def __init__(
        self,
        service_url: str,
        timeout: float = 2.0,
        jpeg_quality: int = 80,
    ) -> None:
        self._service_url = service_url
        self._timeout = timeout
        self._jpeg_quality = int(jpeg_quality)

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0
        self._latest_result: Optional[DepthResult] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = InferStats()

    # ── 生命周期 ──────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="async-infer")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ── 主线程接口 ────────────────────────────────────────────

    def feed_frame(self, frame: np.ndarray, ts: Optional[float] = None) -> None:
        """主线程调用：喂入最新帧（非阻塞）。"""
        if ts is None:
            ts = time.time()
        with self._lock:
            # 拷贝避免工作线程编码时帧数据被覆盖
            self._latest_frame = frame.copy()
            self._latest_frame_ts = ts

    def latest_result(self) -> Optional[DepthResult]:
        """主线程调用：读取最新推理结果（非阻塞）。"""
        with self._lock:
            return self._latest_result

    @property
    def stats(self) -> InferStats:
        with self._lock:
            return self._stats

    # ── 工作线程 ──────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            # 取最新帧
            with self._lock:
                frame = self._latest_frame
                frame_ts = self._latest_frame_ts

            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            frame_age_ms = (t0 - frame_ts) * 1000.0 if frame_ts > 0 else 0.0

            # JPEG 编码
            import cv2
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if not ok:
                self._record_error("JPEG encode failed")
                time.sleep(0.01)
                continue

            # POST 请求
            try:
                req = urllib.request.Request(
                    self._service_url,
                    data=buf.tobytes(),
                    headers={
                        "Content-Type": "image/jpeg",
                        "User-Agent": _BROWSER_UA,
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read()

                nearness = self._decode_response(raw)
                rtt_ms = (time.time() - t0) * 1000.0
                # 新鲜度锚在帧采集时刻：同一冻帧反复推理不得刷新 watchdog
                result_ts = frame_ts if frame_ts > 0 else t0

                with self._lock:
                    self._latest_result = DepthResult(
                        nearness=nearness,
                        ts=result_ts,
                        rtt_ms=rtt_ms,
                        frame_age_ms=frame_age_ms,
                    )
                    self._stats.rtt_ms = rtt_ms
                    self._stats.requests += 1
                    self._stats.consecutive_errors = 0
                    self._stats.last_error = ""
                    self._stats.last_frame_age_ms = frame_age_ms
                    # 消费掉已推理帧；无新帧时停转，避免冻图空转占满服务
                    if self._latest_frame_ts == frame_ts:
                        self._latest_frame = None

            except Exception as e:
                self._record_error(str(e))
                time.sleep(0.05)  # 错误时稍等避免疯狂重试

    def _decode_response(self, raw: bytes) -> np.ndarray:
        """解析 binary nearness grid 响应。"""
        if len(raw) < 8:
            raise ValueError(f"响应过短: {len(raw)}B")
        h, w = struct.unpack("<II", raw[:8])
        expect = 8 + h * w * 2
        if len(raw) != expect:
            raise ValueError(f"响应长度不符: {len(raw)} != {expect}")
        return np.frombuffer(raw[8:], dtype=np.float16).reshape(h, w).astype(np.float32)

    def _record_error(self, msg: str) -> None:
        with self._lock:
            self._stats.errors += 1
            self._stats.consecutive_errors += 1
            self._stats.last_error = msg
        logger.warning("async infer error: %s", msg)
