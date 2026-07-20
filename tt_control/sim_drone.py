"""离线仿真无人机 + 合成图传。

鸭子类型兼容 tello_client.TelloClient 与 video_stream.VideoStream，
使 App / MujocoPadTwin / Policy 在无真机时也能端到端跑通。
到场后把 IO 换回真机(不带 --sim)即可,下游代码零改动。
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 运动学上限(满杆)
VMAX = 0.6        # 水平速度 m/s
VZMAX = 0.5       # 垂直速度 m/s
YAWRATE = 70.0    # 偏航角速度 deg/s
TAKEOFF_Z = 1.0   # 起飞悬停高度 m
PAD_RANGE = 1.5   # 距垫原点超过该值(m)判为丢垫(mid=-1),用于触发 coast 逻辑
TICK_HZ = 30.0    # 内部积分频率


class SimDrone:
    """鸭子类型兼容 TelloClient 的运动学仿真机。

    state dict 的键与单位与 Tello SDK 保持一致(值为字符串):
    mid, x/y/z(cm,垫子局部系), yaw/pitch/roll(deg), vgx/vgy/vgz(cm/s), h(cm), bat(%)。
    """

    def __init__(
        self,
        local_ip: str = "sim",
        tello_ip: str = "192.168.10.1",
        cmd_port: int = 8889,
        state_port: int = 8890,
    ) -> None:
        self.local_ip = local_ip
        self.tello_addr = (tello_ip, cmd_port)
        self.state_port = state_port

        # 内部真值(米 / 度)
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._yaw = 0.0
        # 当前杆量目标(-100..100)
        self._rc = (0, 0, 0, 0)
        self._airborne = False
        self._bat = 90.0

        self.state: dict[str, str] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_state: Optional[Callable[[dict], None]] = None
        self._cmd_count = 0

    # ---- 生命周期(对应 TelloClient) ----
    def connect(self) -> bool:
        logger.info("[sim] connect")
        self._refresh_state()
        return True

    def start_state_listener(self, on_state: Optional[Callable[[dict], None]] = None) -> None:
        self._on_state = on_state
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._tick_loop, name="sim-drone", daemon=True)
        self._thread.start()

    def stream_on(self) -> bool:
        return True

    def stream_off(self) -> None:
        pass

    def mission_pad_on(self, downward: bool = True) -> Optional[str]:
        return "ok"

    def mission_pad_off(self) -> Optional[str]:
        return "ok"

    # ---- 飞行指令 ----
    def takeoff(self) -> Optional[str]:
        self._cmd_count += 1
        with self._lock:
            self._airborne = True
            self._z = TAKEOFF_Z
        logger.info("[sim] takeoff -> z=%.2f", TAKEOFF_Z)
        return "ok"

    def land(self) -> Optional[str]:
        self._cmd_count += 1
        with self._lock:
            self._airborne = False
            self._z = 0.0
            self._rc = (0, 0, 0, 0)
        logger.info("[sim] land")
        return "ok"

    def emergency(self) -> None:
        self._cmd_count += 1
        with self._lock:
            self._airborne = False
            self._z = 0.0
            self._rc = (0, 0, 0, 0)
        logger.info("[sim] emergency")

    def rc(self, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> None:
        """a=roll b=pitch c=throttle d=yaw；无应答(同真机)。"""
        clamp = lambda v: max(-100, min(100, int(v)))
        with self._lock:
            self._rc = (clamp(a), clamp(b), clamp(c), clamp(d))
        self._cmd_count += 1

    def send(self, cmd: str, wait_response: bool = True, timeout: float = 5.0):
        """支持真机风格的位移/转向指令(up/down/forward/back/left/right/cw/ccw),
        按当前航向即时更新位姿;其余指令返回 ok。

        注意:这是「到位即时跳变」的粗仿真(不走 rc 积分),用于离线预演真机脚本走的
        SDK 动作指令序列与响应/记录/降落逻辑,不追求轨迹平滑。平滑闭环见 rc 通道。
        """
        parts = cmd.split()
        kw = parts[0] if parts else ""
        if kw in ("up", "down", "forward", "back", "left", "right", "cw", "ccw") and len(parts) >= 2:
            try:
                val = float(parts[1])
            except ValueError:
                val = 0.0
            with self._lock:
                if kw in ("cw", "ccw"):
                    # 真机约定:cw=顺时针=航向角减小
                    self._yaw = (self._yaw + (-val if kw == "cw" else val)) % 360.0
                elif kw == "up":
                    self._z = max(0.0, self._z + val / 100.0)
                elif kw == "down":
                    self._z = max(0.0, self._z - val / 100.0)
                else:
                    m = val / 100.0
                    # 机体系分量:前后=vx_b,右为正=vy_b(与 _integrate 约定一致)
                    vx_b = m if kw == "forward" else (-m if kw == "back" else 0.0)
                    vy_b = m if kw == "right" else (-m if kw == "left" else 0.0)
                    yaw_rad = math.radians(self._yaw)
                    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
                    self._x += vx_b * cos_y - vy_b * sin_y
                    self._y += vx_b * sin_y + vy_b * cos_y
        self._cmd_count += 1
        return "ok"

    def height_cm(self) -> Optional[int]:
        return int(round(self._z * 100))

    @property
    def cmd_count(self) -> int:
        return self._cmd_count

    def close(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    # ---- 内部积分 ----
    def _tick_loop(self) -> None:
        dt = 1.0 / TICK_HZ
        last = time.time()
        while self._running:
            now = time.time()
            dt = min(0.1, now - last) or (1.0 / TICK_HZ)
            last = now
            self._integrate(dt)
            self._refresh_state()
            if self._on_state:
                try:
                    self._on_state(self.state)
                except Exception:
                    logger.exception("[sim] on_state cb error")
            time.sleep(1.0 / TICK_HZ)

    def _integrate(self, dt: float) -> None:
        with self._lock:
            a, b, c, d = self._rc
            if self._airborne:
                # 机体系速度
                vx_b = (b / 100.0) * VMAX   # pitch -> 前后
                vy_b = (a / 100.0) * VMAX   # roll  -> 左右(右为正)
                yaw_rad = math.radians(self._yaw)
                cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
                self._x += (vx_b * cos_y - vy_b * sin_y) * dt
                self._y += (vx_b * sin_y + vy_b * cos_y) * dt
                self._z = max(0.0, self._z + (c / 100.0) * VZMAX * dt)
                self._yaw = (self._yaw + (d / 100.0) * YAWRATE * dt) % 360.0
                self._bat = max(0.0, self._bat - 0.02 * dt)
            self._vx_b = (b / 100.0) * VMAX if self._airborne else 0.0
            self._vy_b = (a / 100.0) * VMAX if self._airborne else 0.0

    def _refresh_state(self) -> None:
        with self._lock:
            x, y, z, yaw = self._x, self._y, self._z, self._yaw
            bat = self._bat
            airborne = self._airborne
            vx_b = getattr(self, "_vx_b", 0.0)
            vy_b = getattr(self, "_vy_b", 0.0)
        dist = math.hypot(x, y)
        pad_locked = airborne and dist <= PAD_RANGE
        mid = 1 if pad_locked else -1
        self.state = {
            "mid": str(mid),
            "x": f"{x * 100:.0f}",
            "y": f"{y * 100:.0f}",
            "z": f"{max(z, 0.0) * 100:.0f}",
            "yaw": f"{yaw:.0f}",
            "pitch": "0",
            "roll": "0",
            "vgx": f"{vx_b * 100:.0f}",
            "vgy": f"{vy_b * 100:.0f}",
            "vgz": "0",
            "h": f"{max(z, 0.0) * 100:.0f}",
            "bat": f"{bat:.0f}",
            "templ": "0",
            "tof": f"{max(z, 0.0) * 100:.0f}",
        }


class SimVideo:
    """鸭子类型兼容 VideoStream 的合成图传。

    生成含一个移动"障碍色块"的画面,供视觉策略(如 MockAvoidPolicy)反应。
    仅用于测通图像通路,不追求真实感。
    """

    W, H = 960, 720

    def __init__(self, local_ip: str = "sim", video_port: int = 11111) -> None:
        self.local_ip = local_ip
        self.video_port = video_port
        self._running = False
        self._n = 0
        self._t0 = time.time()

    @property
    def fps(self) -> float:
        return 30.0

    def start(self) -> None:
        self._running = True
        self._t0 = time.time()

    def read(self) -> Optional[np.ndarray]:
        if not self._running:
            return None
        self._n += 1
        img = np.full((self.H, self.W, 3), 40, dtype=np.uint8)
        # 网格背景
        for gx in range(0, self.W, 80):
            cv2.line(img, (gx, 0), (gx, self.H), (55, 55, 55), 1)
        for gy in range(0, self.H, 80):
            cv2.line(img, (0, gy), (self.W, gy), (55, 55, 55), 1)
        # 移动障碍(红色实心圆),x 位置随时间正弦摆动,半径脉动
        cx = int(self.W / 2 + (self.W * 0.28) * math.sin(self._n * 0.045))
        cy = int(self.H * 0.45)
        radius = int(55 + 25 * (0.5 + 0.5 * math.sin(self._n * 0.03)))
        cv2.circle(img, (cx, cy), radius, (40, 40, 220), -1)
        cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2)
        return img

    def stop(self) -> None:
        self._running = False
