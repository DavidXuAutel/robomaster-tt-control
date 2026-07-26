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
        self._ka_thread: Optional[threading.Thread] = None
        self.state: dict[str, str] = {}
        self._last_state_ts: float = 0.0
        self._on_state: Optional[Callable[[dict[str, str]], None]] = None

    def start_state_listener(self, on_state: Optional[Callable[[dict[str, str]], None]] = None) -> None:
        self._on_state = on_state
        self._running = True
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self._state_thread.start()
        # 保活:定期发 command,防止 Tello 空闲自动关机/断 SDK
        self._ka_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._ka_thread.start()

    def _keepalive_loop(self, interval: float = 5.0) -> None:
        """保活：从主 socket 发 command，只发不等回执。

        必须用主 socket（同一 IP:8889 元组）：RMTT 组网模式会把回复目标
        锁定到最先/最近发 command 的客户端端口，随机源端口的保活包可能
        把锁定切到已关闭的临时端口，导致主 socket 此后全部超时
        （2026-07-24 真机排障结论）。回执留在缓冲区，由 send() 发送前清空。
        """
        while self._running:
            slept = 0.0
            while slept < interval:
                if not self._running:
                    return
                time.sleep(0.5)
                slept += 0.5
            with self._lock:
                try:
                    self._cmd.sendto(b"command", self.tello_addr)
                except OSError:
                    pass

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
                self._last_state_ts = time.time()
                if self._on_state:
                    self._on_state(parsed)

    def _drain_stale(self) -> None:
        """清空命令 socket 缓冲的陈旧回执（如保活 command 的 ok）。须持有 _lock。"""
        self._cmd.settimeout(0.0)
        try:
            while True:
                self._cmd.recvfrom(2048)
        except (BlockingIOError, socket.timeout, OSError):
            pass

    def send(self, cmd: str, wait_response: bool = True, timeout: float = 5.0) -> Optional[str]:
        with self._lock:
            logger.info(">>> %s", cmd)
            self._drain_stale()
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
                self._warn_if_reply_locked()
                return None

    @property
    def state_age_s(self) -> float:
        """距上次收到状态包的秒数；从未收到返回 inf。"""
        if not self._last_state_ts:
            return float("inf")
        return time.time() - self._last_state_ts

    def _warn_if_reply_locked(self) -> None:
        """命令超时但状态流仍在更新 → 大概率是回复目标被锁到别的客户端端口。

        RMTT 组网模式下，飞机把命令回复发给最先握手的 IP:端口；若那次握手
        来自随机临时端口（如网段扫描），回复会全部丢失，而状态流(8890)不受
        影响。此时唯一恢复手段是重启飞机后用本 socket(8889)抢先握手。
        """
        if self._last_state_ts and time.time() - self._last_state_ts < 2.0:
            logger.error(
                "命令无回复但状态流正常：飞机可能已把回复锁定到其他客户端端口"
                "（如曾用随机端口扫描/发过 command）。请重启飞机后重连，"
                "并确保首个 command 从本机 %s:8889 发出", self.local_ip,
            )

    def connect(self, retries: int = 4) -> bool:
        # Tello 常丢首包,重试几次
        for _ in range(max(1, retries)):
            if self.send("command", timeout=2.5) == "ok":
                return True
        return False

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
        if self._ka_thread and self._ka_thread.is_alive():
            self._ka_thread.join(timeout=1.0)
        for s in (self._cmd, self._state_sock):
            try:
                s.close()
            except OSError:
                pass
