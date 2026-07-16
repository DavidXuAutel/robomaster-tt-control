"""统一 OpenCV 界面：连接按钮、在线状态、图传、推理、键盘操控。"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from tt_control.config import AppConfig, detect_local_ip
from tt_control.control import HELP_TEXT, RcAxes, map_key
from tt_control.inference import InferenceBackend, PassthroughBackend
from tt_control.mujoco_twin import MujocoPadTwin
from tt_control.status import is_drone_online
from tt_control.tello_client import TelloClient
from tt_control.video_stream import VideoStream
logger = logging.getLogger(__name__)

BTN_W, BTN_H = 160, 44
BTN_MARGIN = 16
STATUS_H = 36
RC_HOLD_TIMEOUT = 0.35  # 按键松开判定（秒）
RC_SEND_HZ = 15.0


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
        self._last_rc_time = 0.0
        self._last_rc_send = 0.0
        self._flying = False
        self._conn_state = ConnState.OFFLINE
        self._status_msg = ""
        self._hint = "Click window focus, then CONNECT → T takeoff → WASD"
        self._last_key_label = ""
        self._online = False
        self._probe_stop = threading.Event()
        self._probe_thread: Optional[threading.Thread] = None
        self._buttons: Dict[str, Tuple[int, int, int, int]] = {}
        self._connect_lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._twin: Optional[MujocoPadTwin] = None
        if config.enable_mujoco:
            self._twin = MujocoPadTwin(get_state=lambda: (self.client.state if self.client else {}))

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
            if self._conn_state == ConnState.CONNECTING:
                time.sleep(0.5)
                continue
            local = self.config.local_ip or detect_local_ip()
            online = is_drone_online(self.config.tello_ip, local_ip=local or "")
            self._online = online
            if self.client and self.client.state:
                try:
                    h = int(self.client.state.get("h", "0"))
                    if h > 20:
                        self._flying = True
                    elif h <= 5 and self._flying:
                        # 可能已降落
                        pass
                except ValueError:
                    pass
            if self._conn_state == ConnState.CONNECTED:
                if not online and not (self.client and self.client.state):
                    self._status_msg = "link lost?"
            elif self._conn_state != ConnState.CONNECTING:
                self._conn_state = ConnState.ONLINE if online else ConnState.OFFLINE
            self._probe_stop.wait(2.0)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x1, y1, x2, y2) in self._buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._click_button(name)
                return

    def _click_button(self, name: str) -> None:
        if name == "connect":
            if self.connected:
                threading.Thread(target=self._disconnect, daemon=True).start()
            else:
                threading.Thread(target=self._connect, daemon=True).start()
        elif name == "takeoff":
            self._async_flight_cmd("takeoff")
        elif name == "land":
            self._async_flight_cmd("land")
        elif name == "hover":
            self._hover()

    def _connect(self) -> None:
        with self._connect_lock:
            if self._conn_state in (ConnState.CONNECTED, ConnState.CONNECTING):
                return
            self._conn_state = ConnState.CONNECTING
            self._status_msg = "connecting..."
            self._hint = "Connecting..."
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
                if self.config.enable_mission_pad:
                    pad = self.client.mission_pad_on(downward=True)
                    logger.info("mission pad on: %s", pad)
                    if pad != "ok":
                        self._hint = f"Connected; pad detect: {pad}"
                    else:
                        self._hint = "Connected — fly over Mission Pad for MuJoCo lock"
                self.video = VideoStream(local_ip, self.config.video_port)
                self.video.start()
                if self._twin:
                    if self._twin.start():
                        logger.info("MuJoCo twin started")
                    else:
                        logger.warning("MuJoCo twin failed: %s", self._twin.status)
                self._conn_state = ConnState.CONNECTED
                self._status_msg = "connected"
                if not self._hint.startswith("Connected"):
                    self._hint = "Press T or TAKEOFF, then hold WASD to fly"
                self._flying = False
                self._last_rc = RcAxes()
                logger.info("drone connected via %s", local_ip)
            except Exception as e:
                logger.exception("connect failed: %s", e)
                self._status_msg = str(e)
                self._hint = f"Connect failed: {e}"
                self._conn_state = ConnState.ERROR
                self._cleanup_session(land=False)

    def _disconnect(self) -> None:
        with self._connect_lock:
            self._status_msg = "disconnecting..."
            self._cleanup_session(land=True)
            self._conn_state = ConnState.ONLINE if self._online else ConnState.OFFLINE
            self._status_msg = "disconnected"
            self._hint = "Disconnected"
            logger.info("drone disconnected")

    def _cleanup_session(self, land: bool) -> None:
        if self._twin:
            self._twin.stop()
        try:
            if self.client and self.config.enable_mission_pad:
                self.client.mission_pad_off()
        except Exception:
            pass
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

    def _async_flight_cmd(self, kind: str) -> None:
        if not self.connected or not self.client:
            self._hint = "Connect first"
            return
        threading.Thread(target=self._run_flight_cmd, args=(kind,), daemon=True).start()

    def _run_flight_cmd(self, kind: str) -> None:
        if not self.client:
            return
        with self._cmd_lock:
            if kind == "takeoff":
                self._hint = "Taking off..."
                self._status_msg = "takeoff..."
                self._last_rc = RcAxes()
                try:
                    self.client.rc(0, 0, 0, 0)
                except Exception:
                    pass
                resp = self.client.takeoff()
                if resp == "ok":
                    self._flying = True
                    self._hint = "Airborne — hold WASD/arrows to move, SPACE hover"
                    self._status_msg = "flying"
                else:
                    self._hint = f"Takeoff failed: {resp or 'timeout'}"
                    self._status_msg = "takeoff failed"
                    # 仍可能已离地，用高度兜底
                    h = self.client.height_cm()
                    if h is not None and h > 20:
                        self._flying = True
                logger.info("takeoff result=%s flying=%s", resp, self._flying)
            elif kind == "land":
                self._hint = "Landing..."
                self._last_rc = RcAxes()
                try:
                    self.client.rc(0, 0, 0, 0)
                except Exception:
                    pass
                resp = self.client.land()
                self._flying = False
                self._hint = "Landed" if resp == "ok" else f"Land: {resp or 'timeout'}"
                self._status_msg = "landed"
                logger.info("land result=%s", resp)

    def _hover(self) -> None:
        self._last_rc = RcAxes()
        self._last_rc_time = time.time()
        if self.client and self.connected:
            self.client.rc(0, 0, 0, 0)
            self._hint = "Hover"
            self._last_key_label = "SPACE/hover"

    def _loop(self) -> int:
        blank = np.zeros((720, 960, 3), dtype=np.uint8)
        running = True
        while running:
            frame = None
            if self.video and self.connected:
                frame = self.video.read()
            if frame is None:
                frame = blank.copy()
                tip = self._hint or "Click CONNECT or press C"
                cv2.putText(
                    frame,
                    tip[:60],
                    (40, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                )
            else:
                try:
                    frame = self.inference.infer(frame)
                except Exception as e:
                    logger.exception("inference error: %s", e)

            self._draw_ui(frame)
            cv2.imshow(self.config.window_name, frame)

            key = cv2.waitKeyEx(1)
            if key != -1 and key != 255:
                action = map_key(key, self.config.rc_speed)
                if action.kind == "quit":
                    running = False
                else:
                    self._dispatch_action(action, key)

            self._update_rc_stream()

        return 0

    def _dispatch_action(self, action, key: int) -> None:
        kind = action.kind
        if kind == "none":
            return
        if kind == "toggle_help":
            self.show_help = not self.show_help
            return
        if kind == "connect_toggle":
            if self.connected:
                threading.Thread(target=self._disconnect, daemon=True).start()
            else:
                threading.Thread(target=self._connect, daemon=True).start()
            return

        if not self.connected:
            self._hint = "Not connected — press C or CONNECT first"
            return

        if kind == "takeoff":
            self._last_key_label = "T takeoff"
            self._async_flight_cmd("takeoff")
            return
        if kind == "land":
            self._last_key_label = "L land"
            self._async_flight_cmd("land")
            return
        if kind == "emergency":
            self._last_key_label = "ESC emergency"
            if self.client:
                self.client.emergency()
            self._flying = False
            self._last_rc = RcAxes()
            self._hint = "EMERGENCY stop"
            return
        if kind == "hover":
            self._hover()
            return
        if kind == "rc":
            self._last_key_label = f"key={key & 0xFF} rc{action.axes.as_tuple()}"
            if not self._flying:
                self._hint = "Not airborne — press T / TAKEOFF first (rc ignored on ground)"
                logger.info("rc ignored (not flying): %s", action.axes.as_tuple())
                return
            self._last_rc = action.axes
            self._last_rc_time = time.time()
            if self.client:
                a, b, c, d = action.axes.as_tuple()
                self.client.rc(a, b, c, d)
                self._last_rc_send = time.time()
            self._hint = f"RC {action.axes.as_tuple()}"

    def _update_rc_stream(self) -> None:
        if not (self.connected and self.client and self._flying):
            return
        now = time.time()
        # 按键松开 → 回中悬停
        if (
            not self._last_rc.is_zero()
            and self._last_rc_time > 0
            and now - self._last_rc_time > RC_HOLD_TIMEOUT
        ):
            self._last_rc = RcAxes()
            self.client.rc(0, 0, 0, 0)
            self._last_rc_send = now
            self._hint = "Key released → hover"
            return
        # 按住时高频发送 rc（Tello 需要持续杆量）
        if not self._last_rc.is_zero() and now - self._last_rc_send >= 1.0 / RC_SEND_HZ:
            a, b, c, d = self._last_rc.as_tuple()
            self.client.rc(a, b, c, d)
            self._last_rc_send = now

    def _draw_button(
        self,
        frame: np.ndarray,
        name: str,
        x1: int,
        y1: int,
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        x2, y2 = x1 + BTN_W, y1 + BTN_H
        self._buttons[name] = (x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 1)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        cv2.putText(
            frame,
            label,
            (x1 + (BTN_W - tw) // 2, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_ui(self, frame: np.ndarray) -> None:
        self._buttons.clear()
        h, w = frame.shape[:2]

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

        sx = w - BTN_MARGIN - BTN_W
        sy = BTN_MARGIN
        cv2.rectangle(frame, (sx, sy), (w - BTN_MARGIN, sy + STATUS_H), (30, 30, 30), -1)
        cv2.circle(frame, (sx + 18, sy + STATUS_H // 2), 8, color, -1)
        cv2.putText(
            frame,
            self._conn_state.value,
            (sx + 36, sy + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        by = sy + STATUS_H + 8
        conn_label = "DISCONNECT" if self.connected else "CONNECT"
        conn_color = (60, 60, 200) if self.connected else (40, 160, 40)
        if self._conn_state == ConnState.CONNECTING:
            conn_label, conn_color = "...", (100, 100, 100)
        self._draw_button(frame, "connect", sx, by, conn_label, conn_color)
        by += BTN_H + 8
        self._draw_button(frame, "takeoff", sx, by, "TAKEOFF", (0, 140, 255))
        by += BTN_H + 8
        self._draw_button(frame, "land", sx, by, "LAND", (0, 100, 220))
        by += BTN_H + 8
        self._draw_button(frame, "hover", sx, by, "HOVER", (120, 120, 120))

        bat = self.client.state.get("bat", "?") if self.client else "?"
        alt = self.client.state.get("h", "?") if self.client else "?"
        fps = self.video.fps if self.video else 0.0
        fly = "AIRBORNE" if self._flying else "GROUNDED"
        fly_color = (40, 255, 40) if self._flying else (0, 165, 255)
        lines = [
            f"BAT {bat}%  H {alt}cm  FPS {fps:.1f}  [{fly}]",
            f"IP {self.config.local_ip or '-'} -> {self.config.tello_ip}",
            f"{self._hint}",
            f"key {self._last_key_label}  RC {self._last_rc.as_tuple()}",
        ]
        if self._twin:
            lines.append(f"MuJoCo: {self._twin.status}")
        y = 28
        for i, text in enumerate(lines):
            col = fly_color if i == 0 else (40, 255, 40)
            cv2.putText(frame, text[:78], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text[:78], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)
            y += 22

        if self.show_help:
            help_lines = HELP_TEXT
            y = h - 12 - 22 * len(help_lines)
            for text in help_lines:
                cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                y += 22

    def _shutdown(self) -> None:
        logger.info("shutting down")
        self._probe_stop.set()
        if self._probe_thread and self._probe_thread.is_alive():
            self._probe_thread.join(timeout=2.0)
        self._cleanup_session(land=True)
        cv2.destroyAllWindows()
