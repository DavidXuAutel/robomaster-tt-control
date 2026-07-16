#!/usr/bin/env python3
"""入口：RoboMaster TT 统一控制界面。"""

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
        help="本机 Wi-Fi IP（默认自动检测 192.168.10.x）",
    )
    p.add_argument("--tello-ip", default="192.168.10.1", help="飞机 IP")
    p.add_argument("--rc-speed", type=int, default=40, help="杆量 1-100")
    p.add_argument(
        "--inference",
        default="passthrough",
        help="推理后端名（默认 passthrough，可自行扩展）",
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
        print(
            "未检测到 192.168.10.x。请先连接 TELLO Wi-Fi，或指定 --local-ip。\n"
            "注意：不要改动服务器有线网卡配置。",
            file=sys.stderr,
        )
        return 2

    cfg = AppConfig(
        local_ip=local_ip,
        tello_ip=args.tello_ip,
        rc_speed=max(1, min(100, args.rc_speed)),
    )
    backend = create_backend(args.inference)
    return App(cfg, inference=backend).run()


if __name__ == "__main__":
    raise SystemExit(main())
