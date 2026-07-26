"""L1 契约：深度新鲜度必须锚在帧采集时刻，冻帧不得伪装新鲜。"""
from __future__ import annotations

import struct
import time
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from tt_control.async_infer import AsyncInferWorker
from tt_control.depth_backend import DepthAnythingBackend


def _nearness_payload(h: int = 4, w: int = 4) -> bytes:
    grid = np.full((h, w), 0.3, dtype=np.float16)
    return struct.pack("<II", h, w) + grid.tobytes()


class _FakeResp:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@contextmanager
def _mock_depth_http(raw: bytes | None = None):
    payload = raw if raw is not None else _nearness_payload()

    def _urlopen(req, timeout=None):  # noqa: ARG001
        return _FakeResp(payload)

    with patch("urllib.request.urlopen", side_effect=_urlopen):
        yield


def test_frozen_frame_result_ts_does_not_refresh():
    """同一 frame_ts 反复推理，result.ts 保持采集时刻。"""
    with _mock_depth_http():
        w = AsyncInferWorker("http://127.0.0.1:9/depth", timeout=1.0)
        w.start()
        try:
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            capture_ts = time.time() - 2.5
            w.feed_frame(frame, ts=capture_ts)
            deadline = time.time() + 2.0
            result = None
            while time.time() < deadline:
                result = w.latest_result()
                if result is not None:
                    break
                time.sleep(0.02)
            assert result is not None
            assert abs(result.ts - capture_ts) < 1e-6
            # 再喂同一采集时刻（模拟冻图被主循环反复 feed）
            time.sleep(0.05)
            w.feed_frame(frame, ts=capture_ts)
            time.sleep(0.15)
            result2 = w.latest_result()
            assert result2 is not None
            assert abs(result2.ts - capture_ts) < 1e-6
        finally:
            w.stop()


def test_latest_depth_stale_on_old_capture_ts():
    """采集时刻已过期 → latest_depth 返回 None。"""
    with _mock_depth_http():
        be = DepthAnythingBackend(
            service_url="http://127.0.0.1:9/depth",
            timeout=1.0,
            overlay=False,
        )
        try:
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            old_ts = time.time() - 6.0
            be.infer(frame, frame_ts=old_ts)
            deadline = time.time() + 2.0
            saw = False
            while time.time() < deadline:
                if be._worker.latest_result() is not None:
                    saw = True
                    break
                time.sleep(0.02)
            assert saw
            assert be.latest_depth(max_age_s=5.0) is None
        finally:
            be.close()


def test_backend_requires_service_url():
    try:
        DepthAnythingBackend(service_url="")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "service_url" in str(e)
