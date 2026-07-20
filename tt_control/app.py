"""统一 OpenCV 界面：连接按钮、在线状态、图传、推理、键盘操控。"""

from __future__ import annotations

import logging
import pathlib
import threading
import time
from enum import Enum
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from tt_control.avoidance import AvoidanceController
from tt_control.config import AppConfig, detect_local_ip
from tt_control.control import HELP_TEXT, RcAxes, map_key
from tt_control.flight_test import FlightTestRecorder
from tt_control.inference import InferenceBackend, InferenceEvent, PassthroughBackend
from tt_control.mujoco_twin import MujocoPadTwin
from tt_control.status import is_drone_online
from tt_control.tello_client import TelloClient
from tt_control.video_stream import VideoStream
from tt_control.sim_drone import SimDrone, SimVideo
logger = logging.getLogger(__name__)

BTN_W, BTN_H = 160, 44
BTN_MARGIN = 16
STATUS_H = 36
RC_HOLD_TIMEOUT = 0.35  # 按键松开判定（秒）
RC_SEND_HZ = 15.0


def _ascii(text: str) -> str:
    """cv2.putText 只能渲染 ASCII，其余字符替换，避免画出乱码。"""
    text = text.replace("—", "-").replace("→", "->").replace("；", "; ")
    return text.encode("ascii", "replace").decode()


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
        self._flight_cmd_pending = threading.Event()
        self._last_inference_event = 0.0
        self._gesture_test_results: set[str] = set()
        self._gesture_test_complete = False
        self._gesture_banner = ""
        self._gesture_banner_color = (0, 200, 255)
        self._gesture_banner_until = 0.0
        self._flight_test_state = "DISARMED"
        self._flight_test_recorder: FlightTestRecorder | None = None
        self._flight_test_log: pathlib.Path | None = None
        self._last_test_telemetry = 0.0
        self._twin: Optional[MujocoPadTwin] = None
        if config.enable_mujoco:
            traj_dir = pathlib.Path.cwd() / "logs" / "trajectories"
            self._twin = MujocoPadTwin(
                get_state=lambda: (self.client.state if self.client else {}),
                traj_dir=traj_dir,
            )
        # 半自动视觉避障：OFF -> ARMED -> ON（需已起飞 + 深度后端）
        self._auto = "OFF"
        self._auto_hud = ""
        self._controller = AvoidanceController()
        # 深度后端持有同一 controller 用于叠图标注「此刻会输出的杆量」
        if hasattr(self.inference, "controller"):
            self.inference.controller = self._controller

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
            if self.config.sim:
                online = True
            else:
                local = self.config.local_ip or detect_local_ip()
                online = is_drone_online(self.config.tello_ip, local_ip=local or "")
            self._online = online
            # 飞行状态以起飞/降落指令为准,不用高度读数猜(避免落地后被误判为在飞、切换错乱)
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
            if self.config.gesture_flight_test:
                self._hint = "Flight test: use the takeoff gesture after TEST ARM"
                return
            self._async_flight_cmd("takeoff")
        elif name == "land":
            if self.config.gesture_flight_test and self._flight_test_state not in (
                "DISARMED", "PASSED", "FAILED",
            ):
                self._record_flight_test("manual_land_backup")
                self._flight_test_state = "ABORTING"
            self._async_flight_cmd("land")
        elif name == "hover":
            self._hover()
        elif name.startswith("train_") and name != "train_save":
            if not self.connected:
                self._hint = "Connect first, then start gesture training"
                return
            self._reset_gesture_test()
            label = name.removeprefix("train_")
            self._hint = self.inference.toggle_training(label)
        elif name == "train_save":
            self._reset_gesture_test()
            self._hint = self.inference.save_training_profile()
        elif name == "test_arm":
            if self._flight_test_state == "ARMED":
                self._disarm_flight_test("user_disarmed")
            else:
                self._arm_flight_test()
        elif name == "test_fail":
            self._fail_flight_test("FAIL button clicked")

    def _reset_gesture_test(self) -> None:
        self._gesture_test_results.clear()
        self._gesture_test_complete = False
        self._gesture_banner = ""
        self._gesture_banner_until = 0.0

    def _test_snapshot(self) -> dict:
        return {
            "test_state": self._flight_test_state,
            "connected": self.connected,
            "flying": self._flying,
            "telemetry": dict(self.client.state) if self.client else {},
            "inference": self.inference.status_text,
            "hint": self._hint,
            "video_fps": round(self.video.fps, 2) if self.video else 0.0,
        }

    def _record_flight_test(self, event: str, **data) -> None:
        if self._flight_test_recorder:
            self._flight_test_recorder.record(event, **self._test_snapshot(), **data)

    def _close_flight_test_log(self, reason: str) -> None:
        if self._flight_test_recorder:
            self._record_flight_test("recorder_closed", reason=reason)
            self._flight_test_recorder.close()
            self._flight_test_recorder = None

    def _arm_flight_test(self) -> None:
        if not self.config.gesture_flight_test:
            return
        if not (self.connected and self.client):
            self._hint = "Connect first"
            return
        if self._flight_cmd_pending.is_set() or self._flying:
            self._hint = "Cannot arm while a flight command is active or airborne"
            return
        if "waiting for hand" not in self.inference.status_text.lower():
            self._hint = "Remove hand from camera, then ARM"
            return
        try:
            battery = int(self.client.state.get("bat", "-1"))
            height = int(self.client.state.get("h", "0"))
        except ValueError:
            battery, height = -1, 0
        if battery < 50:
            reason = "telemetry not ready" if battery < 0 else f"battery {battery}% < 50%"
            self._hint = f"TEST ARM blocked: {reason}"
            return
        if height > 10:
            self._hint = f"TEST ARM blocked: reported height {height}cm"
            return

        if self._flight_test_recorder:
            self._flight_test_recorder.close()
        recorder = FlightTestRecorder(pathlib.Path.cwd() / "logs" / "gesture_flight_tests")
        self._flight_test_recorder = recorder
        self._flight_test_log = recorder.path
        self._flight_test_state = "ARMED"
        self._gesture_test_complete = False
        self._last_inference_event = 0.0
        self._gesture_banner = "FLIGHT TEST ARMED"
        self._gesture_banner_color = (0, 180, 255)
        self._gesture_banner_until = time.monotonic() + 2.5
        self._hint = "ARMED: show takeoff gesture; keep clear of propellers"
        self._record_flight_test("armed", battery=battery, height_cm=height)
        logger.info("flight test armed log=%s", recorder.path)

    def _disarm_flight_test(self, reason: str) -> None:
        if self._flying:
            self._hint = "Cannot disarm while airborne - use TEST FAIL or LAND"
            return
        self._record_flight_test("disarmed", reason=reason)
        self._flight_test_state = "DISARMED"
        self._hint = "Flight test disarmed"
        if self._flight_test_recorder:
            self._flight_test_recorder.close()
            self._flight_test_recorder = None

    def _fail_flight_test(self, reason: str) -> None:
        if not self.config.gesture_flight_test:
            return
        if self._flight_test_state in ("DISARMED", "PASSED"):
            self._hint = "No active flight test"
            return
        if self._flight_test_state == "FAILED":
            self._hint = "Failure already recorded; review the current test log"
            return
        self._record_flight_test("failed", reason=reason)
        self._flight_test_state = "FAILED"
        self._gesture_banner = "TEST FAILED - LANDING"
        self._gesture_banner_color = (0, 0, 255)
        self._gesture_banner_until = float("inf")
        self._hint = f"FAILED: {reason}; landing if airborne"
        logger.error("flight test failed: %s log=%s", reason, self._flight_test_log)
        height = 0
        if self.client:
            try:
                height = int(self.client.state.get("h", "0"))
            except ValueError:
                pass
        if (self._flying or height > 20) and not self._flight_cmd_pending.is_set():
            self._async_flight_cmd("land")
        elif not self._flying and height <= 20 and not self._flight_cmd_pending.is_set():
            self._gesture_test_complete = True
            self._close_flight_test_log("failed_on_ground")

    def _connect(self) -> None:
        with self._connect_lock:
            if self._conn_state in (ConnState.CONNECTED, ConnState.CONNECTING):
                return
            self._conn_state = ConnState.CONNECTING
            self._status_msg = "connecting..."
            self._hint = "Connecting..."
            try:
                if self.config.sim:
                    local_ip = self.config.local_ip or "sim"
                else:
                    local_ip = self.config.local_ip or detect_local_ip()
                    if not local_ip:
                        raise RuntimeError("no local IP - join drone Wi-Fi or pass --local-ip")
                self.config.local_ip = local_ip

                if self.client:
                    self.client.close()
                if self.video:
                    self.video.stop()

                if self.config.sim:
                    self.client = SimDrone(local_ip=local_ip or "sim", tello_ip=self.config.tello_ip)
                else:
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
                if self.config.sim:
                    self.video = SimVideo(local_ip, self.config.video_port)
                else:
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
        self._auto = "OFF"
        if self._flight_test_recorder:
            self._record_flight_test("session_cleanup", requested_land=land)
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
        if self._flight_cmd_pending.is_set():
            self._hint = "Flight command already running"
            return
        self._flight_cmd_pending.set()
        threading.Thread(
            target=self._run_flight_cmd_guarded,
            args=(kind,),
            daemon=True,
        ).start()

    def _run_flight_cmd_guarded(self, kind: str) -> None:
        try:
            self._run_flight_cmd(kind)
        finally:
            self._flight_cmd_pending.clear()

    def _run_flight_cmd(self, kind: str) -> None:
        if not self.client:
            return
        with self._cmd_lock:
            if kind == "takeoff":
                self._record_flight_test("command_takeoff")
                self._hint = "Taking off..."
                self._status_msg = "takeoff..."
                self._last_rc = RcAxes()
                try:
                    self.client.rc(0, 0, 0, 0)
                except Exception:
                    pass
                resp = self.client.takeoff()
                self._record_flight_test("command_takeoff_result", response=resp)
                if resp == "ok":
                    self._flying = True
                    self._status_msg = "flying"
                    if self.config.gesture_flight_test:
                        if self._flight_test_state == "FAILED":
                            land_resp = self.client.land()
                            self._record_flight_test(
                                "land_after_failure", response=land_resp
                            )
                            self._flying = land_resp != "ok"
                            self._gesture_test_complete = True
                            self._close_flight_test_log("failed_during_takeoff")
                            return
                        # Tello 的自动 takeoff 已经完成离地和定高；测试不再发送
                        # 任何额外 up/down 指令，直接在该高度悬停等待降落手势。
                        self.client.rc(0, 0, 0, 0)
                        self._flight_test_state = "HOVERING_WAIT_LAND"
                        self._hint = "Takeoff OK - hovering; show LAND gesture"
                        self._gesture_banner = "HOVERING - WAIT LAND GESTURE"
                        self._gesture_banner_color = (0, 210, 255)
                        self._gesture_banner_until = float("inf")
                        self._record_flight_test("hovering_after_takeoff")
                    else:
                        self._hint = "Airborne — hold WASD/arrows to move, SPACE hover"
                else:
                    self._hint = f"Takeoff failed: {resp or 'timeout'}"
                    self._status_msg = "takeoff failed"
                    # 仍可能已离地，用高度兜底
                    h = self.client.height_cm()
                    if h is not None and h > 20:
                        self._flying = True
                    if self.config.gesture_flight_test:
                        self._flight_test_state = "FAILED"
                        self._record_flight_test(
                            "failed",
                            reason="takeoff command failed or timed out",
                            response=resp,
                            reported_height_cm=h,
                        )
                        # 超时并不等于未离地；真机测试中无条件补发 land 更安全。
                        land_resp = self.client.land()
                        self._record_flight_test(
                            "land_after_failure", response=land_resp
                        )
                        self._flying = land_resp != "ok"
                        self._gesture_test_complete = True
                        self._close_flight_test_log("takeoff_failed")
                logger.info("takeoff result=%s flying=%s", resp, self._flying)
            elif kind == "land":
                self._record_flight_test("command_land")
                self._hint = "Landing..."
                self._last_rc = RcAxes()
                try:
                    self.client.rc(0, 0, 0, 0)
                except Exception:
                    pass
                resp = self.client.land()
                self._record_flight_test("command_land_result", response=resp)
                if resp == "ok":
                    self._flying = False
                else:
                    height = self.client.height_cm()
                    # 高度未知时按仍在空中处理，保证退出/FAIL 会继续尝试降落。
                    self._flying = height is None or height > 20
                self._hint = "Landed" if resp == "ok" else f"Land: {resp or 'timeout'}"
                self._status_msg = "landed" if resp == "ok" else "land failed"
                if self.config.gesture_flight_test and self._flight_test_state == "LANDING":
                    if resp == "ok":
                        self._flight_test_state = "PASSED"
                        self._gesture_test_complete = True
                        self._gesture_banner = "REAL FLIGHT TEST PASSED"
                        self._gesture_banner_color = (40, 210, 40)
                        self._gesture_banner_until = float("inf")
                        self._hint = "PASS: takeoff + hover + gesture land"
                        self._record_flight_test("passed")
                        self._close_flight_test_log("test_passed")
                        logger.info("real flight test PASS log=%s", self._flight_test_log)
                    else:
                        self._flight_test_state = "FAILED"
                        self._gesture_banner = "LAND FAILED - CLICK TEST FAIL"
                        self._gesture_banner_color = (0, 0, 255)
                        self._gesture_banner_until = float("inf")
                        self._record_flight_test(
                            "failed", reason="land command failed", response=resp
                        )
                elif self.config.gesture_flight_test and self._flight_test_state == "FAILED":
                    self._gesture_test_complete = True
                    self._record_flight_test(
                        "failure_landing_finished", response=resp
                    )
                    self._close_flight_test_log("failure_landing_finished")
                logger.info("land result=%s", resp)

    def _hover(self) -> None:
        self._last_rc = RcAxes()
        self._last_rc_time = time.time()
        self._auto = "OFF"  # 悬停即关闭半自动
        if self.client and self.connected:
            self.client.rc(0, 0, 0, 0)
            self._hint = "Hover"
            self._last_key_label = "SPACE/hover"

    def _toggle_auto(self) -> None:
        """V 键：首次 ARMED 确认，再按在 ARMED/ON 间切换。需已起飞 + 深度后端。"""
        if not self._flying:
            self._auto = "OFF"
            self._hint = "V ignored - take off first (auto only after airborne)"
            return
        if not hasattr(self.inference, "latest_depth"):
            self._hint = "V ignored - depth backend off (run --inference depth-anything)"
            return
        if self._auto == "OFF":
            self._auto = "ARMED"
            self._hint = "AUTO ARMED - press V again to ENGAGE"
        elif self._auto == "ARMED":
            self._auto = "ON"
            self._controller.reset()
            self._hint = "AUTO ON - WASD overrides, SPACE/ESC/L to stop"
        else:  # ON -> 暂停回 ARMED
            self._auto = "ARMED"
            self._last_rc = RcAxes()
            if self.client:
                self.client.rc(0, 0, 0, 0)
            self._hint = "AUTO paused (ARMED)"
        self._last_key_label = f"V auto={self._auto}"

    def _auto_decision(self):
        """读最新深度 → 控制律决策；无深度则 None。"""
        be = self.inference
        depth = be.latest_depth() if hasattr(be, "latest_depth") else None
        if depth is None:
            return None
        return self._controller.decide(depth.nearness)

    def _loop(self) -> int:
        blank = np.zeros((720, 960, 3), dtype=np.uint8)
        running = True
        last_draw = 0.0
        while running:
            key = cv2.waitKeyEx(5)
            if key != -1 and key != 255:
                action = map_key(key, self.config.rc_speed)
                if action.kind == "quit":
                    running = False
                    continue
                self._dispatch_action(action, key)

            self._update_rc_stream()
            self._update_flight_test_log()

            # macOS 的 Cocoa 后端异步合成，imshow 频率过高会把前后帧混叠出文字重影；
            # 渲染节流到 30fps，按键轮询仍走上面的高频 waitKeyEx
            now = time.time()
            if now - last_draw < 1.0 / 30.0:
                continue
            last_draw = now

            frame = None
            if self.video and self.connected:
                frame = self.video.read()
            if frame is None:
                frame = blank.copy()
                tip = self._hint or "Click CONNECT or press C"
                cv2.putText(
                    frame,
                    _ascii(tip)[:60],
                    (40, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                try:
                    if not self._gesture_test_complete:
                        frame = self.inference.infer(frame)
                        for event in self.inference.drain_events():
                            self._handle_inference_event(event)
                except Exception as e:
                    logger.exception("inference error: %s", e)

            self._draw_ui(frame)
            cv2.imshow(self.config.window_name, frame)

        return 0

    def _handle_inference_event(self, event: InferenceEvent) -> None:
        """把模型事件转换成飞行命令；所有安全条件只在这一层判断。"""
        now = time.monotonic()
        if now - self._last_inference_event < 1.0:
            return
        if not self.config.gesture_commands_enabled:
            self._last_inference_event = now
            self._gesture_test_results.add(event.kind)
            self._last_key_label = f"DRY-RUN {event.kind} {event.confidence:.2f}"
            label = "TAKEOFF" if event.kind == "takeoff" else "LAND"
            self._gesture_banner = f"{label} GESTURE DETECTED"
            self._gesture_banner_color = (40, 210, 255)
            self._gesture_banner_until = now + 3.0
            self._hint = f"Gesture test PASS: {event.kind}"
            logger.info("gesture dry-run %s: %s", event.kind, event.detail)
            if {"takeoff", "land"}.issubset(self._gesture_test_results):
                self._gesture_test_complete = True
                self._gesture_banner = "GESTURE TEST PASSED"
                self._gesture_banner_color = (40, 210, 40)
                self._gesture_banner_until = float("inf")
                self._hint = "PASS: takeoff + land detected; gesture inference stopped"
                logger.info("gesture dry-run test complete: takeoff + land PASS")
            return
        if not (self.connected and self.client):
            self._hint = f"Gesture {event.kind} ignored: not connected"
            return
        if self._flight_cmd_pending.is_set():
            return

        if self.config.gesture_flight_test:
            self._record_flight_test(
                "gesture_detected",
                gesture=event.kind,
                confidence=round(event.confidence, 4),
                detail=event.detail,
            )
            if event.kind == "takeoff" and self._flight_test_state != "ARMED":
                self._hint = f"Takeoff gesture ignored: test {self._flight_test_state}"
                return
            if event.kind == "land" and self._flight_test_state != "HOVERING_WAIT_LAND":
                self._hint = f"Land gesture ignored: test {self._flight_test_state}"
                return

        try:
            height = int(self.client.state.get("h", "0"))
        except ValueError:
            height = 0

        if event.kind == "takeoff":
            if self._flying or height > 20:
                self._hint = "Palm-up ignored: already airborne"
                return
            try:
                battery = int(self.client.state.get("bat", "-1"))
            except ValueError:
                battery = -1
            if battery < 30:
                reason = "battery unknown" if battery < 0 else f"battery {battery}%"
                self._hint = f"Palm-up blocked: {reason}"
                logger.warning("gesture takeoff blocked: %s", reason)
                return
            self._last_inference_event = now
            if self.config.gesture_flight_test:
                self._flight_test_state = "TAKING_OFF"
            self._last_key_label = f"GESTURE palm-up {event.confidence:.2f}"
            self._hint = "Palm-up confirmed - taking off"
            logger.info("gesture takeoff: %s", event.detail)
            self._async_flight_cmd("takeoff")
            return

        if event.kind == "land":
            if not self._flying and height <= 20:
                self._hint = "Finger-snap ignored: already grounded"
                return
            self._last_inference_event = now
            if self.config.gesture_flight_test:
                self._flight_test_state = "LANDING"
            self._last_key_label = f"GESTURE finger-snap {event.confidence:.2f}"
            self._hint = "Visual finger-snap confirmed - landing"
            logger.info("gesture land: %s", event.detail)
            self._async_flight_cmd("land")

    def _update_flight_test_log(self) -> None:
        if not self._flight_test_recorder:
            return
        now = time.monotonic()
        if now - self._last_test_telemetry >= 1.0:
            self._last_test_telemetry = now
            self._record_flight_test("telemetry")

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
            if self.config.gesture_flight_test:
                self._hint = "Flight test: use takeoff gesture after TEST ARM"
                return
            self._last_key_label = "T takeoff"
            self._async_flight_cmd("takeoff")
            return
        if kind == "land":
            self._last_key_label = "L land"
            self._auto = "OFF"
            self._async_flight_cmd("land")
            return
        if kind == "emergency":
            self._last_key_label = "ESC emergency"
            self._auto = "OFF"
            if self.client:
                self.client.emergency()
            self._flying = False
            self._last_rc = RcAxes()
            self._hint = "EMERGENCY stop"
            return
        if kind == "hover":
            self._hover()
            return
        if kind == "auto_toggle":
            self._toggle_auto()
            return
        if kind == "rc":
            if self.config.gesture_flight_test and self._flight_test_state not in (
                "DISARMED", "PASSED", "FAILED",
            ):
                self._hint = "RC movement disabled during vertical flight test"
                return
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
        # 键盘保持优先：有杆量时走键盘链路，覆盖半自动
        if not self._last_rc.is_zero():
            # 按键松开 → 回中（半自动开启时交回避障，否则悬停）
            if self._last_rc_time > 0 and now - self._last_rc_time > RC_HOLD_TIMEOUT:
                self._last_rc = RcAxes()
                self.client.rc(0, 0, 0, 0)
                self._last_rc_send = now
                self._hint = "Key released -> auto" if self._auto == "ON" else "Key released -> hover"
                return
            # 按住时高频发送 rc（Tello 需要持续杆量）
            if now - self._last_rc_send >= 1.0 / RC_SEND_HZ:
                a, b, c, d = self._last_rc.as_tuple()
                self.client.rc(a, b, c, d)
                self._last_rc_send = now
            return
        # 无键盘输入 + 半自动 ON → 由避障控制律驱动
        if self._auto == "ON" and now - self._last_rc_send >= 1.0 / RC_SEND_HZ:
            dec = self._auto_decision()
            if dec is not None:
                a, b, c, d = dec.axes.as_tuple()
                self.client.rc(a, b, c, d)
                self._last_rc_send = now
                self._auto_hud = dec.as_hud()

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
        if self.inference.training_supported:
            by += BTN_H + 8
            active = self.inference.active_training_label
            training_buttons = (
                ("takeoff", "TRAIN TAKEOFF", (120, 95, 30)),
                ("land", "TRAIN LAND", (125, 75, 75)),
                ("none", "TRAIN NONE", (100, 100, 100)),
            )
            for label, idle_text, idle_color in training_buttons:
                button_text = f"STOP {label.upper()}" if active == label else idle_text
                button_color = (30, 70, 210) if active == label else idle_color
                self._draw_button(
                    frame, f"train_{label}", sx, by, button_text, button_color
                )
                by += BTN_H + 8
            self._draw_button(
                frame, "train_save", sx, by, "SAVE PROFILE", (110, 45, 125)
            )
            by += BTN_H + 8
        if self.config.gesture_flight_test:
            arm_label = "DISARM TEST" if self._flight_test_state == "ARMED" else "TEST ARM"
            arm_color = (0, 150, 220) if self._flight_test_state == "ARMED" else (45, 135, 45)
            self._draw_button(frame, "test_arm", sx, by, arm_label, arm_color)
            by += BTN_H + 8
            self._draw_button(frame, "test_fail", sx, by, "TEST FAIL", (0, 0, 210))

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
        if self._auto != "OFF":
            lines.append(f"AUTO: {self._auto}  {self._auto_hud}")
        if self._twin:
            lines.append(f"MuJoCo: {self._twin.status}")
        if self.inference.status_text:
            lines.append(self.inference.status_text)
        if not self.config.gesture_commands_enabled:
            takeoff = "PASS" if "takeoff" in self._gesture_test_results else "WAIT"
            land = "PASS" if "land" in self._gesture_test_results else "WAIT"
            state = "COMPLETE" if self._gesture_test_complete else "RUNNING"
            lines.append(f"DRY-RUN TEST [{state}]  TAKEOFF {takeoff} | LAND {land}")
        if self.config.gesture_flight_test:
            log_name = self._flight_test_log.name if self._flight_test_log else "-"
            lines.append(f"REAL TEST [{self._flight_test_state}]  LOG {log_name}")
        lines = [_ascii(t)[:70] for t in lines]
        colors = [fly_color] + [(60, 255, 60)] * (len(lines) - 1)
        self._draw_text_panel(frame, lines, 6, 6, colors)

        if self.show_help:
            help_lines = [_ascii(t) for t in HELP_TEXT]
            line_h, pad = 28, 12
            block_h = pad * 2 + line_h * len(help_lines)
            self._draw_text_panel(
                frame,
                help_lines,
                6,
                h - 10 - block_h,
                [(235, 235, 235)] * len(help_lines),
            )
        self._draw_gesture_banner(frame)

    def _draw_gesture_banner(self, frame: np.ndarray) -> None:
        if (
            self.config.gesture_commands_enabled
            and not self.config.gesture_flight_test
            or not self._gesture_banner
            or time.monotonic() > self._gesture_banner_until
        ):
            return
        h, w = frame.shape[:2]
        x1, x2 = 24, max(300, w - BTN_W - BTN_MARGIN * 2)
        y1, y2 = max(110, h // 2 - 76), max(230, h // 2 + 76)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self._gesture_banner_color, 5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.05
        text = self._gesture_banner
        tw = cv2.getTextSize(text, font, scale, 3)[0][0]
        cv2.putText(
            frame, text, (x1 + max(12, (x2 - x1 - tw) // 2), y1 + 66),
            font, scale, self._gesture_banner_color, 3, cv2.LINE_AA,
        )
        takeoff = "PASS" if "takeoff" in self._gesture_test_results else "WAIT"
        land = "PASS" if "land" in self._gesture_test_results else "WAIT"
        detail = f"TAKEOFF: {takeoff}     LAND: {land}"
        dw = cv2.getTextSize(detail, font, 0.72, 2)[0][0]
        cv2.putText(
            frame, detail, (x1 + max(12, (x2 - x1 - dw) // 2), y1 + 118),
            font, 0.72, (245, 245, 245), 2, cv2.LINE_AA,
        )

    def _draw_text_panel(self, frame, lines, x, y_top, colors) -> None:
        """深色半透明底板 + 文字，保证任何视频背景下都可读。"""
        font, scale, line_h, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 28, 12
        h, w = frame.shape[:2]
        tw = max(cv2.getTextSize(t, font, scale, 1)[0][0] for t in lines)
        x2 = min(x + tw + pad * 2, w)
        y2 = min(y_top + pad * 2 + line_h * len(lines) - 8, h)
        x1, y1 = max(x, 0), max(y_top, 0)
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = (roi * 0.35).astype(np.uint8)
        y = y_top + pad + 16
        for text, col in zip(lines, colors):
            cv2.putText(frame, text, (x + pad, y), font, scale, col, 1, cv2.LINE_AA)
            y += line_h

    def _shutdown(self) -> None:
        logger.info("shutting down")
        self._probe_stop.set()
        if self._probe_thread and self._probe_thread.is_alive():
            self._probe_thread.join(timeout=2.0)
        self._cleanup_session(land=True)
        try:
            self.inference.close()
        except Exception:
            logger.exception("inference close failed")
        if self._flight_test_recorder:
            self._record_flight_test("recorder_closed")
            self._flight_test_recorder.close()
            self._flight_test_recorder = None
        cv2.destroyAllWindows()
