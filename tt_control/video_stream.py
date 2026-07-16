"""UDP H.264 图传接收与解码。"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Optional

import av
import numpy as np

logger = logging.getLogger(__name__)


class VideoStream:
    def __init__(self, local_ip: str, video_port: int = 11111) -> None:
        self.local_ip = local_ip
        self.video_port = video_port
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._fps = 0.0
        self._frame_count = 0
        self._fps_ts = time.time()

    @property
    def fps(self) -> float:
        return self._fps

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
        self._sock.bind((self.local_ip, self.video_port))
        self._sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("video listen %s:%s", self.local_ip, self.video_port)

    def _loop(self) -> None:
        codec = av.CodecContext.create("h264", "r")
        assert self._sock is not None
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            try:
                packets = codec.parse(data)
            except Exception:
                continue
            for packet in packets:
                try:
                    frames = codec.decode(packet)
                except Exception:
                    continue
                for frame in frames:
                    img = frame.to_ndarray(format="bgr24")
                    with self._lock:
                        self._frame = img
                    self._frame_count += 1
                    now = time.time()
                    dt = now - self._fps_ts
                    if dt >= 1.0:
                        self._fps = self._frame_count / dt
                        self._frame_count = 0
                        self._fps_ts = now

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
