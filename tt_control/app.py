"""统一 OpenCV 界面：图传 + 推理叠加 + 键盘操控。"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from tt_control.config import AppConfig
from tt_control.control import HELP_TEXT, RcAxes, map_key
from tt_control.inference import InferenceBackend, PassthroughBackend
from tt_control.tello_client import TelloClient
from tt_control.video_stream import VideoStream

logger = logging.getLogger(__name__)


class App:
    def __init__(
        self,
        config: AppConfig,
        inference: Optional[InferenceBackend] = None,
    ) -> None:
        if not config.local_ip:
            raise ValueError("local_ip 为空：请连接 TELLO Wi-Fi 或传入 --local-ip 192.168.10.x")
        self.config = config
        self.inference = inference or PassthroughBackend()
        self.client = TelloClient(
            local_ip=config.local_ip,
            tello_ip=config.tello_ip,
            cmd_port=config.cmd_port,
            state_port=config.state_port,
        )
        self.video = VideoStream(config.local_ip, config.video_port)
        self.show_help = True
        self._last_rc = RcAxes()
        self._last_heartbeat = 0.0
        self._flying = False

    def run(self) -> int:
        logger.info("local_ip=%s tello=%s", self.config.local_ip, self.config.tello_ip)
        if not self.client.connect():
            logger.error("无法进入 SDK 模式（command 未返回 ok）")
            self.client.close()
            return 1

        self.client.start_state_listener()
        if not self.client.stream_on():
            logger.error("streamon 失败")
            self.client.close()
            return 1

        self.video.start()
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)

        try:
            return self._loop()
        finally:
            self._shutdown()

    def _loop(self) -> int:
        blank = np.zeros((720, 960, 3), dtype=np.uint8)
        while True:
            frame = self.video.read()
            if frame is None:
                frame = blank.copy()
                cv2.putText(
                    frame,
                    "Waiting for video...",
                    (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
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
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            self._draw_hud(frame)
            cv2.imshow(self.config.window_name, frame)

            key = cv2.waitKey(1)
            if key != -1:
                action = map_key(key, self.config.rc_speed)
                if action.kind == "quit":
                    break
                self._handle_action(action)

            # 无按键时保持上次杆量；周期性发 rc 作心跳（避免 15s 无指令）
            now = time.time()
            if now - self._last_heartbeat >= self.config.heartbeat_interval:
                a, b, c, d = self._last_rc.as_tuple()
                self.client.rc(a, b, c, d)
                self._last_heartbeat = now

        return 0

    def _handle_action(self, action) -> None:
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

    def _draw_hud(self, frame: np.ndarray) -> None:
        bat = self.client.state.get("bat", "?")
        h = self.client.state.get("h", "?")
        lines = [
            f"BAT {bat}%  H {h}cm  FPS {self.video.fps:.1f}",
            f"IP {self.config.local_ip} -> {self.config.tello_ip}",
            f"RC a={self._last_rc.roll} b={self._last_rc.pitch} "
            f"c={self._last_rc.throttle} d={self._last_rc.yaw}",
        ]
        y = 28
        for text in lines:
            cv2.putText(
                frame,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 255, 40),
                1,
                cv2.LINE_AA,
            )
            y += 28

        if self.show_help:
            hh, ww = frame.shape[:2]
            y = hh - 12 - 22 * len(HELP_TEXT)
            for text in HELP_TEXT:
                cv2.putText(
                    frame,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                y += 22

    def _shutdown(self) -> None:
        logger.info("shutting down")
        try:
            self.client.rc(0, 0, 0, 0)
        except Exception:
            pass
        try:
            if self._flying:
                self.client.land()
        except Exception:
            pass
        try:
            self.client.stream_off()
        except Exception:
            pass
        self.video.stop()
        self.client.close()
        cv2.destroyAllWindows()
