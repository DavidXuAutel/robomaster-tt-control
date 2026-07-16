"""统一 OpenCV 界面：连接按钮、在线状态、图传、推理、键盘操控。"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from tt_control.config import AppConfig, detect_local_ip
from tt_control.control import HELP_TEXT, RcAxes, map_key
from tt_control.inference import InferenceBackend, PassthroughBackend
from tt_control.status import is_drone_online
from tt_control.tello_client import TelloClient
from tt_control.video_stream import VideoStream

logger = logging.getLogger(__name__)

# 按钮区域相对画布右上角（在 960x720 基准上，绘制时按比例缩放）
BTN_W, BTN_H = 160, 44
BTN_MARGIN = 16
STATUS_H = 36


class ConnState(str, Enum):
    OFFLINE = "OFFLINE"
    ONLINE = "ONLINE"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


class App:
    def __init__(
        self,
        config: AppConfig,
        inference: Optional[InferenceBackend] = None,
    ) -> None:
        self.config = config
        self.inference = inference or PassthroughBackend()
        self.client: Optional[TelloClient] = None
        self.video: Optional[VideoStream] = None
        self.show_help = True
        self._last_rc = RcAxes()
        self._last_heartbeat = 0.0
        self._flying = False
        self._conn_state = ConnState.OFFLINE
        self._status_msg = ""
        self._online = False
        self._probe_stop = threading.Event()
        self._probe_thread: Optional[threading.Thread] = None
        self._btn_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._canvas_size = (960, 720)
        self._connect_lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._conn_state == ConnState.CONNECTED

    def run(self) -> int:
        logger.info(
            "start UI local_ip=%r tello=%s",
            self.config.local_ip,
            self.config.tello_ip,
        )
        self._start_probe()
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.config.window_name, self._on_mouse)
        try:
            return self._loop()
        finally:
            self._shutdown()

    def _start_probe(self) -> None:
        self._probe_stop.clear()
        self._probe_thread = threading.Thread(target=self._probe_loop, daemon=True)
        self._probe_thread.start()

    def _probe_loop(self) -> None:
        while not self._probe_stop.is_set():
            if self._conn_state in (ConnState.CONNECTING,):
                time.sleep(0.5)
                continue
            local = self.config.local_ip or detect_local_ip()
            online = is_drone_online(self.config.tello_ip, local_ip=local or "")
            self._online = online
            if self._conn_state == ConnState.CONNECTED:
                # 已连接时若长时间无状态且 ping 失败，标为异常但仍保持会话
                if not online and not (self.client and self.client.state):
                    self._status_msg = "link lost?"
            elif self._conn_state != ConnState.CONNECTING:
                self._conn_state = ConnState.ONLINE if online else ConnState.OFFLINE
            self._probe_stop.wait(2.0)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x1, y1, x2, y2 = self._btn_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            if self.connected:
                threading.Thread(target=self._disconnect, daemon=True).start()
            else:
                threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self) -> None:
        with self._connect_lock:
            if self._conn_state in (ConnState.CONNECTED, ConnState.CONNECTING):
                return
            self._conn_state = ConnState.CONNECTING
            self._status_msg = "connecting..."
            try:
                local_ip = self.config.local_ip or detect_local_ip()
                if not local_ip:
                    raise RuntimeError("未检测到 192.168.10.x，请先连接 TELLO Wi-Fi")
                self.config.local_ip = local_ip

                if self.client:
                    self.client.close()
                if self.video:
                    self.video.stop()

                self.client = TelloClient(
                    local_ip=local_ip,
                    tello_ip=self.config.tello_ip,
                    cmd_port=self.config.cmd_port,
                    state_port=self.config.state_port,
                )
                if not self.client.connect():
                    raise RuntimeError("command 未返回 ok")
                self.client.start_state_listener()
                if not self.client.stream_on():
                    raise RuntimeError("streamon 失败")
                self.video = VideoStream(local_ip, self.config.video_port)
                self.video.start()
                self._conn_state = ConnState.CONNECTED
                self._status_msg = "connected"
                self._last_heartbeat = time.time()
                logger.info("drone connected via %s", local_ip)
            except Exception as e:
                logger.exception("connect failed: %s", e)
                self._status_msg = str(e)
                self._conn_state = ConnState.ERROR
                self._cleanup_session(land=False)

    def _disconnect(self) -> None:
        with self._connect_lock:
            self._status_msg = "disconnecting..."
            self._cleanup_session(land=True)
            self._conn_state = ConnState.ONLINE if self._online else ConnState.OFFLINE
            self._status_msg = "disconnected"
            logger.info("drone disconnected")

    def _cleanup_session(self, land: bool) -> None:
        try:
            if self.client and self._flying and land:
                self.client.land()
        except Exception:
            pass
        self._flying = False
        self._last_rc = RcAxes()
        try:
            if self.client:
                self.client.rc(0, 0, 0, 0)
                self.client.stream_off()
        except Exception:
            pass
        if self.video:
            self.video.stop()
            self.video = None
        if self.client:
            self.client.close()
            self.client = None

    def _loop(self) -> int:
        blank = np.zeros((720, 960, 3), dtype=np.uint8)
        while True:
            frame = None
            if self.video and self.connected:
                frame = self.video.read()
            if frame is None:
                frame = blank.copy()
                tip = "Click CONNECT or press C"
                if self._conn_state == ConnState.CONNECTING:
                    tip = "Connecting..."
                elif self._conn_state == ConnState.CONNECTED:
                    tip = "Waiting for video..."
                elif self._conn_state == ConnState.OFFLINE:
                    tip = "Drone OFFLINE — connect TELLO Wi-Fi"
                cv2.putText(
                    frame,
                    tip,
                    (40, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 200, 255),
                    2,
                )
            else:
                try:
                    frame = self.inference.infer(frame)
                except Exception as e:
                    logger.exception("inference error: %s", e)
                    cv2.putText(
                        frame,
                        f"infer error: {e}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            self._canvas_size = (frame.shape[1], frame.shape[0])
            self._draw_ui(frame)
            cv2.imshow(self.config.window_name, frame)

            key = cv2.waitKey(1)
            if key != -1:
                action = map_key(key, self.config.rc_speed)
                if action.kind == "quit":
                    break
                if action.kind == "connect_toggle":
                    if self.connected:
                        threading.Thread(target=self._disconnect, daemon=True).start()
                    else:
                        threading.Thread(target=self._connect, daemon=True).start()
                    continue
                if self.connected:
                    self._handle_action(action)

            if self.connected and self.client:
                now = time.time()
                if now - self._last_heartbeat >= self.config.heartbeat_interval:
                    a, b, c, d = self._last_rc.as_tuple()
                    self.client.rc(a, b, c, d)
                    self._last_heartbeat = now

        return 0

    def _handle_action(self, action) -> None:
        if not self.client:
            return
        kind = action.kind
        if kind == "none":
            return
        if kind == "toggle_help":
            self.show_help = not self.show_help
            return
        if kind == "takeoff":
            self.client.takeoff()
            self._flying = True
            self._last_rc = RcAxes()
            return
        if kind == "land":
            self.client.land()
            self._flying = False
            self._last_rc = RcAxes()
            return
        if kind == "emergency":
            self.client.emergency()
            self._flying = False
            self._last_rc = RcAxes()
            return
        if kind == "hover":
            self._last_rc = RcAxes()
            self.client.rc(0, 0, 0, 0)
            self._last_heartbeat = time.time()
            return
        if kind == "rc":
            self._last_rc = action.axes
            a, b, c, d = action.axes.as_tuple()
            self.client.rc(a, b, c, d)
            self._last_heartbeat = time.time()

    def _draw_ui(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        # 状态灯 + 文字（右上）
        status = self._conn_state.value
        if self._conn_state == ConnState.CONNECTED:
            color = (40, 220, 40)
        elif self._conn_state == ConnState.ONLINE:
            color = (0, 220, 255)
        elif self._conn_state == ConnState.CONNECTING:
            color = (0, 180, 255)
        elif self._conn_state == ConnState.ERROR:
            color = (0, 0, 255)
        else:
            color = (80, 80, 220)

        sx = w - BTN_MARGIN - 200
        sy = BTN_MARGIN
        cv2.rectangle(frame, (sx, sy), (w - BTN_MARGIN, sy + STATUS_H), (30, 30, 30), -1)
        cv2.circle(frame, (sx + 18, sy + STATUS_H // 2), 8, color, -1)
        cv2.putText(
            frame,
            status,
            (sx + 36, sy + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        # Connect / Disconnect 按钮
        bx2 = w - BTN_MARGIN
        by1 = sy + STATUS_H + 10
        bx1 = bx2 - BTN_W
        by2 = by1 + BTN_H
        self._btn_rect = (bx1, by1, bx2, by2)
        btn_label = "DISCONNECT" if self.connected else "CONNECT"
        btn_color = (60, 60, 200) if self.connected else (40, 160, 40)
        if self._conn_state == ConnState.CONNECTING:
            btn_label = "..."
            btn_color = (100, 100, 100)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), btn_color, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (240, 240, 240), 1)
        tw = cv2.getTextSize(btn_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv2.putText(
            frame,
            btn_label,
            (bx1 + (BTN_W - tw) // 2, by1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 左上 HUD
        bat = self.client.state.get("bat", "?") if self.client else "?"
        alt = self.client.state.get("h", "?") if self.client else "?"
        fps = self.video.fps if self.video else 0.0
        lines = [
            f"BAT {bat}%  H {alt}cm  FPS {fps:.1f}",
            f"IP {self.config.local_ip or '-'} -> {self.config.tello_ip}",
            f"Online={self._online}  {self._status_msg}",
        ]
        if self.connected:
            lines.append(
                f"RC a={self._last_rc.roll} b={self._last_rc.pitch} "
                f"c={self._last_rc.throttle} d={self._last_rc.yaw}"
            )
        y = 28
        for text in lines:
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 255, 40), 1, cv2.LINE_AA)
            y += 26

        if self.show_help:
            help_lines = HELP_TEXT + ["C connect/disconnect  click button"]
            y = h - 12 - 22 * len(help_lines)
            for text in help_lines:
                cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                y += 22

    def _shutdown(self) -> None:
        logger.info("shutting down")
        self._probe_stop.set()
        if self._probe_thread and self._probe_thread.is_alive():
            self._probe_thread.join(timeout=2.0)
        self._cleanup_session(land=True)
        cv2.destroyAllWindows()
