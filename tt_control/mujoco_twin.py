"""Mission Pad 局部坐标 → MuJoCo 实时数字孪生。"""

from __future__ import annotations

import logging
import math
import pathlib
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ASSET = pathlib.Path(__file__).resolve().parent / "assets" / "tello_pad_twin.xml"


def _yaw_to_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    """Z-up yaw (deg) → MuJoCo quaternion (w, x, y, z)."""
    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _parse_pad_pose(state: dict) -> Optional[tuple[float, float, float, float, int]]:
    """
    从 Tello state 解析垫子坐标系位姿。
    返回 (x_m, y_m, z_m, yaw_deg, mid)；无效时返回 None。
    """
    try:
        mid = int(float(state.get("mid", "-2")))
        x_cm = float(state.get("x", "-200"))
        y_cm = float(state.get("y", "-200"))
        z_cm = float(state.get("z", "-200"))
        yaw = float(state.get("yaw", "0"))
    except (TypeError, ValueError):
        return None

    # mid: -2 未开检测, -1 未看到垫子
    if mid < 0:
        return None
    if x_cm <= -100 or y_cm <= -100 or z_cm <= -100:
        return None

    # SDK: cm → m；z 为相对垫子高度，抬离地面一点避免穿模
    x_m = x_cm / 100.0
    y_m = y_cm / 100.0
    z_m = max(z_cm / 100.0, 0.03)
    return x_m, y_m, z_m, yaw, mid


class MujocoPadTwin:
    """被动查看器：每帧用 SDK 相对 Mission Pad 的 (x,y,z,yaw) 写 freejoint。"""

    def __init__(self, get_state: Callable[[], dict], model_path: Optional[pathlib.Path] = None) -> None:
        self._get_state = get_state
        self._model_path = model_path or ASSET
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_mid: Optional[int] = None
        self._ok = False
        self._status = "idle"

    @property
    def status(self) -> str:
        return self._status

    def start(self) -> bool:
        try:
            import mujoco  # noqa: F401
            import mujoco.viewer  # noqa: F401
        except ImportError as e:
            self._status = f"mujoco not installed: {e}"
            logger.error(self._status)
            return False
        if not self._model_path.is_file():
            self._status = f"missing model {self._model_path}"
            logger.error(self._status)
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mujoco-twin", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._status = "stopped"

    def _run(self) -> None:
        import mujoco
        import mujoco.viewer

        model = mujoco.MjModel.from_xml_path(str(self._model_path))
        data = mujoco.MjData(model)
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drone_free")
        if jnt_id < 0:
            self._status = "joint drone_free not found"
            logger.error(self._status)
            return
        qadr = model.jnt_qposadr[jnt_id]

        self._ok = True
        self._status = "viewer running (waiting for pad)"
        logger.info("MuJoCo twin started: %s", self._model_path)

        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and not self._stop.is_set():
                    state = self._get_state() or {}
                    pose = _parse_pad_pose(state)
                    if pose is not None:
                        x, y, z, yaw, mid = pose
                        qw, qx, qy, qz = _yaw_to_quat(yaw)
                        data.qpos[qadr : qadr + 7] = [x, y, z, qw, qx, qy, qz]
                        mujoco.mj_forward(model, data)
                        self._last_mid = mid
                        self._status = f"pad m{mid}  xyz=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.0f}"
                    else:
                        # 无垫子：仅用高度+姿态做弱更新（水平冻结在上次或原点上方）
                        try:
                            h_cm = float(state.get("h", "0"))
                            yaw = float(state.get("yaw", "0"))
                        except (TypeError, ValueError):
                            h_cm, yaw = 0.0, 0.0
                        if self._last_mid is None:
                            data.qpos[qadr + 2] = max(h_cm / 100.0, 0.03)
                            qw, qx, qy, qz = _yaw_to_quat(yaw)
                            data.qpos[qadr + 3 : qadr + 7] = [qw, qx, qy, qz]
                            mujoco.mj_forward(model, data)
                        self._status = "no pad lock (enable mon + fly over Mission Pad)"

                    viewer.sync()
                    time.sleep(0.05)
        except Exception as e:
            self._status = f"viewer error: {e}"
            logger.exception("MuJoCo twin failed")
        finally:
            self._ok = False
            self._status = "viewer closed"
            logger.info("MuJoCo twin stopped")
