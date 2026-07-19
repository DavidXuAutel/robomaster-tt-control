#!/usr/bin/env python3
"""入口：RoboMaster TT 统一控制界面（可选 MuJoCo Mission Pad 孪生）。"""

from __future__ import annotations

import argparse
import logging
import sys

from tt_control.app import App
from tt_control.config import AppConfig, detect_local_ip
from tt_control.inference import create_backend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoboMaster TT 实时图传 + 键盘控制")
    p.add_argument(
        "--local-ip",
        default="",
        help="本机 Wi-Fi IP（默认自动检测 192.168.10.x；可稍后点 CONNECT 再连）",
    )
    p.add_argument("--tello-ip", default="192.168.10.1", help="飞机 IP")
    p.add_argument("--rc-speed", type=int, default=40, help="杆量 1-100")
    p.add_argument(
        "--inference",
        default="passthrough",
        choices=("passthrough", "gestures"),
        help="推理后端：passthrough 或 gestures（纯视觉手势控制）",
    )
    p.add_argument(
        "--mujoco",
        action="store_true",
        help="启用 MuJoCo 数字孪生（Mission Pad 局部坐标 x/y/z → 仿真机体）",
    )
    p.add_argument(
        "--no-mission-pad",
        action="store_true",
        help="连接后不自动 mon（默认会开启垫子检测）",
    )
    p.add_argument(
        "--gesture-dry-run",
        action="store_true",
        help="只显示手势识别结果，不发送起飞/降落命令",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    local_ip = args.local_ip or detect_local_ip()
    if not local_ip:
        logging.warning(
            "未检测到 192.168.10.x，界面仍会启动；请连接 TELLO Wi-Fi 后点击 CONNECT"
        )

    cfg = AppConfig(
        local_ip=local_ip,
        tello_ip=args.tello_ip,
        rc_speed=max(1, min(100, args.rc_speed)),
        enable_mujoco=args.mujoco,
        enable_mission_pad=(not args.no_mission_pad) or args.mujoco,
        gesture_commands_enabled=not args.gesture_dry_run,
    )
    backend = create_backend(args.inference)
    return App(cfg, inference=backend).run()


if __name__ == "__main__":
    raise SystemExit(main())
