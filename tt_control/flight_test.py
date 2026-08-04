"""真机手势飞行测试的追加式 JSONL 日志。"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class FlightTestRecorder:
    """每次测试创建新文件，每条事件立即 flush，异常退出也尽量保留现场。"""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.path = directory / f"gesture_flight_{stamp}.jsonl"
        self._handle = self.path.open("x", encoding="utf-8")
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._closed = False

    def record(self, event: str, **data: Any) -> None:
        payload = {
            "time": datetime.now().astimezone().isoformat(),
            "elapsed_s": round(time.monotonic() - self._started, 3),
            "event": event,
            **data,
        }
        with self._lock:
            if self._closed:
                return
            self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._handle.close()
                self._closed = True
