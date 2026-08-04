#!/usr/bin/env python3
"""运动学闭环避障验证(第二阶段):不连飞机、不用 GPU。

轻量 2D 俯视仿真:虚拟障碍场 → 沿相机 FOV 光线投射合成「近度网格」→
AvoidanceController.decide() → 把 RC 当机体速度积分 → 看航迹是否绕开障碍到达目标。
用 opencv 渲染俯视图,输出 mp4 + PNG,打印是否到达/最小净空。

这一步验证的是「控制律本身会不会撞」,与感知后端解耦(用几何真值当近度)。
用法:
  python sim_avoidance.py --scenario slalom --out logs/sim_slalom.mp4
  python sim_avoidance.py --scenario wall
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from tt_control.avoidance import AvoidanceController, AvoidParams

logger = logging.getLogger("sim_avoid")

FOV_DEG = 82.6      # Tello 相机水平视场角
SENSE_RANGE = 4.0   # 合成近度的最大探测距离(米)
GRID_W = 128        # 近度网格宽(与服务一致)
GRID_H = 96
V_MAX = 0.6         # pitch=100 对应前进速度(m/s),半自动取小
OMEGA_MAX = 70.0    # yaw=100 对应转向角速度(deg/s)
DT = 0.1            # 控制步长(s)
DRONE_R = 0.15      # 机体半径(m),判碰撞用


@dataclass
class Circle:
    x: float
    y: float
    r: float


SCENARIOS = {
    # 单个正前方障碍
    "single": ([Circle(4.0, 0.0, 0.8)], (9.0, 0.0)),
    # 两个交错障碍,需要连续左右绕
    "slalom": ([Circle(3.5, 0.3, 0.7), Circle(6.0, -0.8, 0.7)], (9.0, 0.0)),
    # 一堵墙留个缺口
    "wall": (
        [Circle(4.0, y, 0.5) for y in (-2.0, -1.0, 1.0, 2.0)],
        (9.0, 0.0),
    ),
}


def cast_nearness(x: float, y: float, yaw: float, obs: List[Circle]) -> np.ndarray:
    """从 (x,y,yaw) 沿 FOV 投射,得到 (GRID_H, GRID_W) 近度网格(越大越近)。"""
    fov = math.radians(FOV_DEG)
    cols = np.zeros(GRID_W, dtype=np.float32)
    for c in range(GRID_W):
        ang = yaw + (c / (GRID_W - 1) - 0.5) * fov
        dx, dy = math.cos(ang), math.sin(ang)
        hit = SENSE_RANGE
        # 沿射线步进找最近障碍
        d = 0.05
        while d < SENSE_RANGE:
            px, py = x + dx * d, y + dy * d
            for o in obs:
                if (px - o.x) ** 2 + (py - o.y) ** 2 <= o.r * o.r:
                    hit = d
                    break
            else:
                d += 0.05
                continue
            break
        cols[c] = float(np.clip(1.0 - hit / SENSE_RANGE, 0.0, 1.0))
    return np.tile(cols, (GRID_H, 1))


def min_clearance(x: float, y: float, obs: List[Circle]) -> float:
    return min((math.hypot(x - o.x, y - o.y) - o.r) for o in obs)


def render(
    world_wh: Tuple[float, float],
    obs: List[Circle],
    traj: List[Tuple[float, float]],
    pose: Tuple[float, float, float],
    goal: Tuple[float, float],
    hud: str,
) -> np.ndarray:
    """俯视图:x 向右,y 向上。"""
    ww, wh = world_wh
    scale = 90
    W, H = int(ww * scale), int(wh * scale)
    img = np.full((H, W, 3), 30, np.uint8)

    def to_px(wx: float, wy: float) -> Tuple[int, int]:
        return int(wx * scale), int(H - (wy + wh / 2) * scale)

    # 网格线
    for gx in range(0, int(ww) + 1):
        cv2.line(img, to_px(gx, -wh / 2), to_px(gx, wh / 2), (50, 50, 50), 1)
    # 障碍
    for o in obs:
        cv2.circle(img, to_px(o.x, o.y), int(o.r * scale), (40, 40, 200), -1)
    # 目标
    cv2.drawMarker(img, to_px(*goal), (40, 220, 40), cv2.MARKER_STAR, 22, 2)
    # 航迹
    for a, b in zip(traj[:-1], traj[1:]):
        cv2.line(img, to_px(*a), to_px(*b), (0, 200, 255), 2)
    # 机体 + 朝向
    x, y, yaw = pose
    cv2.circle(img, to_px(x, y), int(DRONE_R * scale), (255, 255, 255), -1)
    hx, hy = x + 0.4 * math.cos(yaw), y + 0.4 * math.sin(yaw)
    cv2.line(img, to_px(x, y), to_px(hx, hy), (255, 150, 0), 2)

    cv2.rectangle(img, (0, 0), (W, 26), (20, 20, 20), -1)
    cv2.putText(img, hud[:80], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 120), 1, cv2.LINE_AA)
    return img


def main() -> int:
    p = argparse.ArgumentParser(description="运动学闭环避障验证")
    p.add_argument("--scenario", default="slalom", choices=list(SCENARIOS))
    p.add_argument("--out", default="", help="输出 mp4 路径;留空只出 PNG 末帧")
    p.add_argument("--show", action="store_true")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--cruise", type=int, default=AvoidParams.cruise_speed)
    p.add_argument("--yaw", type=int, default=AvoidParams.yaw_speed)
    p.add_argument("--stop-thresh", type=float, default=AvoidParams.stop_thresh)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    obs, goal = SCENARIOS[args.scenario]
    ctrl = AvoidanceController(
        AvoidParams(cruise_speed=args.cruise, yaw_speed=args.yaw, stop_thresh=args.stop_thresh)
    )

    x, y, yaw = 0.0, 0.0, 0.0
    traj: List[Tuple[float, float]] = [(x, y)]
    world_wh = (10.0, 6.0)
    writer = None
    outcome = "WANDER"  # 无碰撞但未穿过障碍区(朝开阔处飞走,设计允许)
    min_clr = 1e9
    blocked_steps = 0
    obs_max_x = max(o.x + o.r for o in obs)  # 障碍区最远边界

    for step in range(args.max_steps):
        near = cast_nearness(x, y, yaw, obs)
        dec = ctrl.decide(near)
        roll, pitch, _thr, yawr = dec.axes.as_tuple()

        v = (pitch / 100.0) * V_MAX
        # 真机约定:yaw 杆量为正=顺时针右转=航向角减小
        omega = math.radians(-(yawr / 100.0) * OMEGA_MAX)
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw += omega * DT
        traj.append((x, y))

        clr = min_clearance(x, y, obs)
        min_clr = min(min_clr, clr)
        blocked_steps = blocked_steps + 1 if dec.state == "BLOCKED" else 0

        hud = f"{args.scenario} step{step} {dec.state} rc{dec.axes.as_tuple()} clr{clr:.2f}m"
        if args.out or args.show:
            img = render(world_wh, obs, traj, (x, y, yaw), goal, hud)
            if args.out:
                if writer is None:
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    writer = cv2.VideoWriter(
                        args.out, cv2.VideoWriter_fourcc(*"mp4v"), int(1 / DT), (img.shape[1], img.shape[0])
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"无法打开视频写出: {args.out}")
                writer.write(img)
            if args.show:
                cv2.imshow("sim avoidance", img)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break

        if clr <= DRONE_R:
            outcome = "COLLISION"
            break
        if x > obs_max_x + 0.8 and abs(y) < world_wh[1] / 2:
            outcome = "CLEARED"  # 无碰撞穿过障碍区
            break
        if blocked_steps > 30:
            outcome = "STOPPED(blocked)"  # 被围住主动悬停,安全
            break

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    png = (args.out[:-4] if args.out.endswith(".mp4") else args.out or f"logs/sim_{args.scenario}") + ".png"
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
    cv2.imwrite(png, render(world_wh, obs, traj, (x, y, yaw), goal, f"{outcome} min_clr={min_clr:.2f}m"))

    logger.info(
        "scenario=%s outcome=%s steps=%d final=(%.2f,%.2f) min_clearance=%.2fm",
        args.scenario, outcome, len(traj) - 1, x, y, min_clr,
    )
    logger.info("plot -> %s%s", png, f"  video -> {args.out}" if args.out else "")
    return 1 if outcome == "COLLISION" else 0


if __name__ == "__main__":
    raise SystemExit(main())
