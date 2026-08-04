"""Mission Pad 局部坐标 → MuJoCo 实时数字孪生 + 完整轨迹记录。

支持两种运行模式:
- 交互式(默认): mujoco.viewer 被动查看器 + 实时画轨迹(需图形显示)。
- 无头(headless=True): 只做位姿映射与轨迹记录/导出,不开窗口(服务器/测试用)。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import pathlib
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

ASSET = pathlib.Path(__file__).resolve().parent / "assets" / "tello_pad_twin.xml"

MIN_POINT_SPACING_M = 0.02
MAX_DRAW_SEGMENTS = 2000


@dataclass
class TrajPoint:
    t: float
    mid: int
    x: float
    y: float
    z: float
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0
    vgx: float = 0.0
    vgy: float = 0.0
    vgz: float = 0.0
    h: float = 0.0
    bat: float = 0.0
    pad_locked: bool = True


def _yaw_to_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    half = math.radians(yaw_deg) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _f(state: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(state.get(key, default))
    except (TypeError, ValueError):
        return default


def _parse_pad_pose(state: dict) -> Optional[tuple[float, float, float, float, int]]:
    try:
        mid = int(float(state.get("mid", "-2")))
        x_cm = float(state.get("x", "-200"))
        y_cm = float(state.get("y", "-200"))
        z_cm = float(state.get("z", "-200"))
        yaw = float(state.get("yaw", "0"))
    except (TypeError, ValueError):
        return None
    if mid < 0 or x_cm <= -100 or y_cm <= -100 or z_cm <= -100:
        return None
    return x_cm / 100.0, y_cm / 100.0, max(z_cm / 100.0, 0.03), yaw, mid


def _draw_trajectory(viewer, points: List[TrajPoint], mujoco) -> None:
    """用 user_scn 胶囊线段画出完整轨迹。"""
    scn = viewer.user_scn
    scn.ngeom = 0
    if len(points) < 2:
        return
    step = max(1, (len(points) - 1) // MAX_DRAW_SEGMENTS)
    idxs = list(range(0, len(points), step))
    if idxs[-1] != len(points) - 1:
        idxs.append(len(points) - 1)

    rgba_lock = np.array([0.15, 0.95, 0.35, 1.0], dtype=np.float32)
    rgba_weak = np.array([0.95, 0.75, 0.15, 0.8], dtype=np.float32)

    for a, b in zip(idxs[:-1], idxs[1:]):
        if scn.ngeom >= scn.maxgeom:
            break
        p0 = points[a]
        p1 = points[b]
        start = np.array([p0.x, p0.y, p0.z], dtype=np.float64)
        end = np.array([p1.x, p1.y, p1.z], dtype=np.float64)
        rgba = rgba_lock if (p0.pad_locked and p1.pad_locked) else rgba_weak
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            rgba,
        )
        mujoco.mjv_connector(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.01,
            start,
            end,
        )
        scn.ngeom += 1

    for pt, color in (
        (points[0], np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32)),
        (points[-1], np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32)),
    ):
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.025, 0.025, 0.025]),
            np.array([pt.x, pt.y, pt.z]),
            np.eye(3).flatten(),
            color,
        )
        scn.ngeom += 1


class MujocoPadTwin:
    """被动查看器：同步垫子系位姿，并记录/可视化完整轨迹。

    headless=True 时不开窗口，仅记录与导出(适合无显示器的服务器/自动化测试)。
    """

    def __init__(
        self,
        get_state: Callable[[], dict],
        model_path: Optional[pathlib.Path] = None,
        traj_dir: Optional[pathlib.Path] = None,
        headless: bool = False,
        stitch_pads: bool = False,
    ) -> None:
        self._get_state = get_state
        self._model_path = model_path or ASSET
        self._traj_dir = traj_dir or (pathlib.Path.cwd() / "logs" / "trajectories")
        self._headless = headless
        self._stitch = stitch_pads
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_mid: Optional[int] = None
        self._ok = False
        self._status = "idle"
        self._traj: List[TrajPoint] = []
        self._traj_lock = threading.Lock()
        self._t0 = 0.0
        self._last_xy: Optional[tuple[float, float]] = None
        self._saved_path: Optional[pathlib.Path] = None
        self._anchor: Optional[TrajPoint] = None
        # 多卡拼接:mid -> 该卡原点在全局系的平移量(仅 stitch_pads 时启用)
        self._pad_offsets: dict[int, tuple[float, float]] = {}

    @property
    def status(self) -> str:
        return self._status

    @property
    def trajectory(self) -> List[TrajPoint]:
        with self._traj_lock:
            return list(self._traj)

    @property
    def traj_count(self) -> int:
        with self._traj_lock:
            return len(self._traj)

    def clear_trajectory(self) -> None:
        with self._traj_lock:
            self._traj.clear()
        self._last_xy = None
        self._saved_path = None
        self._anchor = None
        self._pad_offsets = {}

    def _to_global(self, lx: float, ly: float, mid: int) -> tuple[float, float]:
        """把卡局部 xy 平移到统一全局系,实现多卡连续轨迹拼接。

        stitch 关闭时原样返回(单卡/仿真行为不变)。假设各卡朝向一致(火箭同向),
        故仅需平移:第一张卡定义全局原点;之后换到新卡时,用"飞机位置不会瞬移"的
        连续性,以换卡瞬间的上一全局位置反解该卡偏移量,一次标定、之后复用。
        """
        if not self._stitch:
            return lx, ly
        off = self._pad_offsets.get(mid)
        if off is None:
            if not self._pad_offsets:
                off = (0.0, 0.0)            # 第一张卡:全局系 = 该卡局部系
            else:
                bx, by = self._last_xy or (0.0, 0.0)
                off = (bx - lx, by - ly)     # 新卡:由位置连续性反解偏移
            self._pad_offsets[mid] = off
        return lx + off[0], ly + off[1]

    def start(self) -> bool:
        try:
            import mujoco  # noqa: F401
            if not self._headless:
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
        self.clear_trajectory()
        self._t0 = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mujoco-twin", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        path = self.save_trajectory()
        if path:
            self._status = f"stopped; traj saved {path.name} (n={self.traj_count})"
        else:
            self._status = "stopped"
        logger.info("MuJoCo twin stopped; points=%s file=%s", self.traj_count, path)

    def _append_point(self, pt: TrajPoint) -> None:
        with self._traj_lock:
            if self._traj and self._anchor is not None:
                a = self._anchor
                dist = math.sqrt(
                    (pt.x - a.x) ** 2 + (pt.y - a.y) ** 2 + (pt.z - a.z) ** 2
                )
                if dist < MIN_POINT_SPACING_M:
                    # 未超过最小间距:仅刷新末端最新姿态,不移动采样锚点
                    self._traj[-1] = pt
                    return
            self._traj.append(pt)
            self._anchor = pt

    def save_trajectory(self) -> Optional[pathlib.Path]:
        with self._traj_lock:
            points = list(self._traj)
        if not points:
            return None
        self._traj_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self._traj_dir / f"traj_{stamp}.csv"
        json_path = self._traj_dir / f"traj_{stamp}.json"

        fieldnames = list(asdict(points[0]).keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for p in points:
                w.writerow(asdict(p))

        meta = {
            "created": stamp,
            "frame": "mission_pad_local_m",
            "count": len(points),
            "duration_s": points[-1].t - points[0].t,
            "pad_locked_count": sum(1 for p in points if p.pad_locked),
            "start": asdict(points[0]),
            "end": asdict(points[-1]),
            "csv": str(csv_path),
        }
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._saved_path = csv_path
        logger.info("trajectory saved: %s (%d points)", csv_path, len(points))
        return csv_path

    def _update_once(self, mujoco, model, data, qadr) -> None:
        """读取一次状态并更新仿真位姿 + 记录轨迹点(不涉及绘制/窗口)。"""
        state = self._get_state() or {}
        now = time.time() - self._t0
        pitch, roll = _f(state, "pitch"), _f(state, "roll")
        vgx, vgy, vgz = _f(state, "vgx"), _f(state, "vgy"), _f(state, "vgz")
        h_cm, bat = _f(state, "h"), _f(state, "bat")

        pose = _parse_pad_pose(state)
        if pose is not None:
            lx, ly, z, yaw, mid = pose
            x, y = self._to_global(lx, ly, mid)  # 多卡拼接到全局系
            qw, qx, qy, qz = _yaw_to_quat(yaw)
            data.qpos[qadr : qadr + 7] = [x, y, z, qw, qx, qy, qz]
            mujoco.mj_forward(model, data)
            self._last_mid = mid
            self._last_xy = (x, y)
            self._append_point(
                TrajPoint(
                    t=now, mid=mid, x=x, y=y, z=z, yaw=yaw,
                    pitch=pitch, roll=roll, vgx=vgx, vgy=vgy, vgz=vgz,
                    h=h_cm, bat=bat, pad_locked=True,
                )
            )
            n = self.traj_count
            self._status = (
                f"pad m{mid} xyz=({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.0f} traj={n}"
            )
        else:
            yaw = _f(state, "yaw")
            z = max(h_cm / 100.0, 0.03)
            if self._last_xy is not None:
                x, y = self._last_xy
                qw, qx, qy, qz = _yaw_to_quat(yaw)
                data.qpos[qadr : qadr + 7] = [x, y, z, qw, qx, qy, qz]
                mujoco.mj_forward(model, data)
                self._append_point(
                    TrajPoint(
                        t=now, mid=int(_f(state, "mid", -1)), x=x, y=y, z=z, yaw=yaw,
                        pitch=pitch, roll=roll, vgx=vgx, vgy=vgy, vgz=vgz,
                        h=h_cm, bat=bat, pad_locked=False,
                    )
                )
                self._status = (
                    f"pad lost; coast xyz=({x:.2f},{y:.2f},{z:.2f}) traj={self.traj_count}"
                )
            else:
                data.qpos[qadr + 2] = z
                qw, qx, qy, qz = _yaw_to_quat(yaw)
                data.qpos[qadr + 3 : qadr + 7] = [qw, qx, qy, qz]
                mujoco.mj_forward(model, data)
                self._status = (
                    f"no pad lock; traj={self.traj_count} (fly over Mission Pad)"
                )

    def _run(self) -> None:
        import mujoco

        # 用 from_xml_string 而非 from_xml_path:MuJoCo 的 C++ 文件读取器在 Windows
        # 下打不开含非 ASCII 字符的路径(如中文目录)。由 Python 读出文本再传字符串,
        # 绕开该限制;本模型自包含(纹理/材质均 builtin),无需外部资源目录。
        model = mujoco.MjModel.from_xml_string(
            self._model_path.read_text(encoding="utf-8"))
        data = mujoco.MjData(model)
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drone_free")
        if jnt_id < 0:
            self._status = "joint drone_free not found"
            logger.error(self._status)
            return
        qadr = model.jnt_qposadr[jnt_id]

        self._ok = True

        try:
            if self._headless:
                self._status = "headless recording (waiting for pad)"
                logger.info("MuJoCo twin started (headless): %s", self._model_path)
                while not self._stop.is_set():
                    self._update_once(mujoco, model, data, qadr)
                    time.sleep(0.05)
            else:
                import mujoco.viewer
                self._status = "viewer running (waiting for pad)"
                logger.info("MuJoCo twin started: %s", self._model_path)
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    if viewer.user_scn.maxgeom < MAX_DRAW_SEGMENTS + 16:
                        logger.warning(
                            "user_scn.maxgeom=%s may clip long trajectories",
                            viewer.user_scn.maxgeom,
                        )
                    while viewer.is_running() and not self._stop.is_set():
                        self._update_once(mujoco, model, data, qadr)
                        with self._traj_lock:
                            pts = list(self._traj)
                        _draw_trajectory(viewer, pts, mujoco)
                        viewer.sync()
                        time.sleep(0.05)
        except Exception as e:
            self._status = f"viewer error: {e}"
            logger.exception("MuJoCo twin failed")
        finally:
            self._ok = False
            path = self.save_trajectory()
            mode = "headless" if self._headless else "viewer"
            self._status = (
                f"{mode} closed; traj n={self.traj_count}"
                + (f" -> {path.name}" if path else "")
            )
            logger.info("MuJoCo twin stopped")
