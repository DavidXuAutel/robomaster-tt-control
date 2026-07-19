"""Tello UDP 明文协议客户端（Tello SDK 3.0）。"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TelloClient:
    def __init__(
        self,
        local_ip: str,
        tello_ip: str = "192.168.10.1",
        cmd_port: int = 8889,
        state_port: int = 8890,
    ) -> None:
        self.local_ip = local_ip
        self.tello_addr = (tello_ip, cmd_port)
        self.state_port = state_port

        self._cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd.bind((local_ip, cmd_port))
        self._cmd.settimeout(5.0)

        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.bind((local_ip, state_port))
        self._state_sock.settimeout(1.0)

        self._lock = threading.Lock()
        self._running = False
        self._state_thread: Optional[threading.Thread] = None
        self.state: dict[str, str] = {}
        self._on_state: Optional[Callable[[dict[str, str]], None]] = None

    def start_state_listener(self, on_state: Optional[Callable[[dict[str, str]], None]] = None) -> None:
        self._on_state = on_state
        self._running = True
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()

    def _state_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._state_sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            text = data.decode(errors="ignore").strip()
            parsed: dict[str, str] = {}
            for part in text.split(";"):
                if ":" in part:
                    k, v = part.split(":", 1)
                    parsed[k] = v
            if parsed:
                self.state = parsed
                if self._on_state:
                    self._on_state(parsed)

    def send(self, cmd: str, wait_response: bool = True, timeout: float = 5.0) -> Optional[str]:
        with self._lock:
            logger.info(">>> %s", cmd)
            self._cmd.settimeout(timeout)
            try:
                self._cmd.sendto(cmd.encode("utf-8"), self.tello_addr)
            except OSError as e:
                logger.error("send failed: %s", e)
                return None
            if not wait_response:
                return None
            try:
                data, _ = self._cmd.recvfrom(2048)
                resp = data.decode(errors="ignore").strip()
                logger.info("<<< %s", resp)
                return resp
            except socket.timeout:
                logger.warning("timeout: %s", cmd)
                return None

    def connect(self) -> bool:
        return self.send("command", timeout=3.0) == "ok"

    def stream_on(self) -> bool:
        return self.send("streamon", timeout=5.0) == "ok"

    def stream_off(self) -> None:
        self.send("streamoff", timeout=3.0)

    def mission_pad_on(self, downward: bool = True) -> Optional[str]:
        """打开 Mission Pad 检测；downward=True 时仅下视。"""
        r = self.send("mon", timeout=3.0)
        if r != "ok":
            return r
        # 0=下视 1=前视 2=双向
        return self.send("mdirection 0" if downward else "mdirection 2", timeout=3.0)

    def mission_pad_off(self) -> Optional[str]:
        return self.send("moff", timeout=3.0)

    def takeoff(self) -> Optional[str]:
        return self.send("takeoff", timeout=20.0)

    def up(self, centimeters: int) -> Optional[str]:
        """按 Tello SDK 范围执行相对上升。"""
        centimeters = max(20, min(500, int(centimeters)))
        return self.send(f"up {centimeters}", timeout=15.0)

    def land(self) -> Optional[str]:
        return self.send("land", timeout=20.0)

    def emergency(self) -> None:
        self.send("emergency", wait_response=False)

    def rc(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> None:
        """a=roll, b=pitch, c=throttle, d=yaw；无应答。"""
        a = max(-100, min(100, int(a)))
        b = max(-100, min(100, int(b)))
        c = max(-100, min(100, int(c)))
        d = max(-100, min(100, int(d)))
        self.send(f"rc {a} {b} {c} {d}", wait_response=False)

    def battery(self) -> Optional[int]:
        raw = self.state.get("bat")
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                pass
        resp = self.send("battery?", timeout=3.0)
        try:
            return int(resp) if resp else None
        except ValueError:
            return None

    def height_cm(self) -> Optional[int]:
        raw = self.state.get("h")
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def close(self) -> None:
        self._running = False
        if self._state_thread and self._state_thread.is_alive():
            self._state_thread.join(timeout=1.0)
        for s in (self._cmd, self._state_sock):
            try:
                s.close()
            except OSError:
                pass
