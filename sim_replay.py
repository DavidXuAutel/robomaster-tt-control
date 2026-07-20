#!/usr/bin/env python3
"""在 MuJoCo 3D 查看器里回放一条已记录的飞行轨迹(交给大众看的"仿真复现")。

读 trajectory CSV(twin schema:t,mid,x,y,z,yaw,...,x/y/z 单位米),按时间轴把
位姿喂给非无头的 MujocoPadTwin(查看器模式),无人机模型沿原路径飞、轨迹线实时
生长。CSV 里已是拼接好的全局坐标,故 stitch 关闭、原样重绘。

需图形显示(服务器物理显示器):
    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
      .venv/bin/python sim_replay.py logs/trajectories/traj_XXXX.csv
    # 可选:--speed 1.0 播放倍速  --loop 循环播放
"""

from __future__ import annotations

import argparse
import csv
import logging
import pathlib
import time
from typing import List, Tuple

from tt_control.mujoco_twin import MujocoPadTwin

logger = logging.getLogger("sim_replay")


def _load(csv_path: pathlib.Path) -> List[Tuple[float, float, float, float, float]]:
    """读 CSV -> [(t, x_m, y_m, z_m, yaw_deg), ...],t 归一到从 0 开始。"""
    rows: List[Tuple[float, float, float, float, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["t"]), float(r["x"]), float(r["y"]),
                             float(r["z"]), float(r["yaw"])))
            except (KeyError, TypeError, ValueError):
                continue
    if len(rows) < 2:
        raise SystemExit(f"轨迹点不足(读到 {len(rows)} 个): {csv_path}")
    t0 = rows[0][0]
    return [(t - t0, x, y, z, yaw) for (t, x, y, z, yaw) in rows]


class Replayer:
    """按墙钟时间在轨迹点间线性插值,产出 Tello 风格 state(x/y/z 用 cm)。"""

    def __init__(self, rows, speed: float = 1.0, loop: bool = False) -> None:
        self._rows = rows
        self._tmax = rows[-1][0]
        self._speed = max(0.05, speed)
        self._loop = loop
        self._t0 = time.time()

    def _pose_at(self, tt: float):
        rows = self._rows
        if tt <= rows[0][0]:
            return rows[0][1:]
        if tt >= self._tmax:
            return rows[-1][1:]
        for i in range(1, len(rows)):
            if rows[i][0] >= tt:
                t_a, *a = rows[i - 1]
                t_b, *b = rows[i]
                f = (tt - t_a) / (t_b - t_a) if t_b > t_a else 0.0
                return tuple(av + (bv - av) * f for av, bv in zip(a, b))
        return rows[-1][1:]

    @property
    def state(self) -> dict:
        elapsed = (time.time() - self._t0) * self._speed
        if self._loop and self._tmax > 0:
            elapsed = elapsed % self._tmax
        x, y, z, yaw = self._pose_at(elapsed)
        return {
            "mid": "1",  # 恒正 -> pad_locked;stitch 关闭 -> 全局坐标原样重绘
            "x": f"{x * 100:.1f}", "y": f"{y * 100:.1f}", "z": f"{z * 100:.1f}",
            "yaw": f"{yaw:.1f}", "pitch": "0", "roll": "0",
            "vgx": "0", "vgy": "0", "vgz": "0",
            "h": f"{z * 100:.0f}", "bat": "0",
        }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MuJoCo 3D 回放已记录轨迹")
    p.add_argument("csv", help="trajectory CSV 路径(twin schema)")
    p.add_argument("--speed", type=float, default=1.0, help="播放倍速(默认 1.0)")
    p.add_argument("--loop", action="store_true", help="循环播放")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rows = _load(pathlib.Path(args.csv))
    logger.info("载入 %d 点,时长 %.1fs;倍速 %.2f%s",
                len(rows), rows[-1][0], args.speed, " 循环" if args.loop else "")

    rp = Replayer(rows, speed=args.speed, loop=args.loop)
    # 回放产生的轨迹另存到 replays/ 子目录,不与真机记录混淆
    twin = MujocoPadTwin(
        get_state=lambda: rp.state,
        traj_dir=pathlib.Path("logs") / "replays",
        headless=False,
        stitch_pads=False,
    )
    if not twin.start():
        raise SystemExit(f"查看器启动失败: {twin.status}")

    logger.info("查看器已开(关闭窗口或 Ctrl-C 结束)...")
    try:
        while twin._thread and twin._thread.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("收到中断,关闭")
    finally:
        twin.stop()
    logger.info("回放结束: %s", twin.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
