#!/usr/bin/env python3
"""入口：RoboMaster TT 统一控制界面（可选 MuJoCo Mission Pad 孪生）。"""

from __future__ import annotations

import argparse
import atexit
import faulthandler
import json
import logging
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

# 原生层崩溃（segfault 等）不会进 Python 日志，faulthandler 把 C 栈写到独立文件
# （2026-07-24 真机飞行中 GUI 无迹闪退，疑似 cv2/av 重复 dylib 冲突）
_crash_log = pathlib.Path(__file__).parent / "logs" / "faulthandler.log"
_crash_log.parent.mkdir(exist_ok=True)
faulthandler.enable(open(_crash_log, "a"))

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
        help="推理后端名（passthrough / depth-anything / gestures）",
    )
    p.add_argument(
        "--depth-service",
        default="",
        help="depth-anything 后端的推理服务地址（留空用内置默认 4090 地址）",
    )
    p.add_argument(
        "--start-depth-service",
        action="store_true",
        help="自动启动本地深度推理服务(默认 127.0.0.1:8899; 退出时自动停止)",
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
    p.add_argument(
        "--gesture-flight-test",
        action="store_true",
        help="启用需手动 ARM 的真机流程：手势起飞 -> 自动起飞高度悬停 -> 手势降落",
    )
    p.add_argument(
        "--sim",
        action="store_true",
        help="离线仿真:用 SimDrone/SimVideo 代替真机(无需飞机)",
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="飞行中同步录制 episode(RGB+深度+动作+状态+时间戳)到 logs/episodes/",
    )
    p.add_argument(
        "--record-hz",
        type=float,
        default=10.0,
        help="录制采样频率(默认 10Hz,限流避免全帧落盘)",
    )
    p.add_argument("--cruise", type=int, default=25, help="避障:通畅前进杆量")
    p.add_argument("--approach-pitch", type=int, default=16,
                   help="避障:接近区前进量(调小=更原地转、少前冲,近人更安全)")
    p.add_argument("--yaw", type=int, default=35, help="避障:转向杆量")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.gesture_dry_run and args.gesture_flight_test:
        logging.error("--gesture-dry-run 与 --gesture-flight-test 不能同时使用")
        return 2
    if args.gesture_flight_test and args.inference != "gestures":
        logging.error("真机手势测试必须使用 --inference gestures")
        return 2

    # ── 自动启动本地深度推理服务 ──────────────────────────────
    _depth_proc: subprocess.Popen | None = None
    _depth_stderr_thread: threading.Thread | None = None

    def _depth_kill(proc: subprocess.Popen | None) -> None:
        """终止深度推理子进程（可安全重复调用）。"""
        if proc is None or proc.poll() is not None:
            return
        log = logging.getLogger()
        log.info("停止深度推理服务(pid=%d) ...", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log.warning("深度服务未响应终止信号, 强制杀死")
            proc.kill()
            proc.wait()
        log.info("深度推理服务已停止")

    def _sigterm_handler(signum: int, frame) -> None:
        logging.getLogger().info("收到 SIGTERM, 正在退出 ...")
        _depth_kill(_depth_proc)
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    if args.start_depth_service:
        if args.depth_service:
            logging.warning(
                "--start-depth-service 与 --depth-service 同时指定, "
                "忽略 --start-depth-service, 使用您指定的服务地址"
            )
        else:
            # 冲突检测：check 是否与显式指定的 --inference 冲突
            _inference_was_default = args.inference in ("passthrough", "none")
            if not _inference_was_default and args.inference not in (
                "depth-anything", "da-v2", "depth"
            ):
                logging.warning(
                    "--start-depth-service 与 --inference %s 同时使用, "
                    "深度模型会被加载但不会被用到",
                    args.inference,
                )

            if _inference_was_default:
                args.inference = "depth-anything"
                logging.info("--start-depth-service: 自动切换 --inference depth-anything")

            port = 8899
            service_url = f"http://127.0.0.1:{port}/depth"
            health_url = f"http://127.0.0.1:{port}/health"

            def _verify_health(url: str, timeout: float = 5.0) -> bool:
                """检验目标地址是否返回有效的深度服务 /health 响应。"""
                try:
                    with urllib.request.urlopen(url, timeout=timeout) as resp:
                        if resp.status != 200:
                            return False
                        body = json.loads(resp.read())
                        return body.get("ok") is True and "model" in body
                except Exception:
                    return False

            # 检查端口是否已被深度服务占用
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            port_busy = sock.connect_ex(("127.0.0.1", port)) == 0
            sock.close()

            if port_busy:
                if _verify_health(health_url):
                    logging.info("端口 %d 已有深度服务运行中, 接续使用", port)
                    args.depth_service = service_url
                else:
                    logging.error(
                        "端口 %d 已被占用但非深度服务(health 校验失败); "
                        "请手动关闭占用程序或使用 --depth-service 指定远端地址",
                        port,
                    )
                    return 2
            else:
                da_v2_path = (
                    pathlib.Path(__file__).resolve().parent / "server" / "da_v2_service.py"
                )
                if not da_v2_path.exists():
                    logging.error(
                        "未找到 %s, 无法启动本地深度服务; "
                        "请手动启动或使用 --depth-service 指定远端地址",
                        da_v2_path,
                    )
                    return 2

                # da_v2_service 需要 server/.venv (有 torch); 找不到则回退 sys.executable
                _server_python = (
                    da_v2_path.parent / ".venv" / "bin" / "python"
                )
                if not _server_python.exists():
                    _server_python = pathlib.Path(sys.executable)
                _depth_proc = subprocess.Popen(
                    [
                        str(_server_python),
                        str(da_v2_path),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,  # 见下方转存线程, 非阻塞
                )

                log = logging.getLogger()
                log.info("启动深度推理服务(pid=%d), 等待模型加载 ...", _depth_proc.pid)

                # 转存子进程 stderr → parent logger, 防止 pipe buffer 满阻塞
                def _drain_stderr(proc: subprocess.Popen) -> None:
                    logger = logging.getLogger("da_v2_service")
                    try:
                        for line in iter(proc.stderr.readline, b""):
                            logger.debug("stderr: %s", line.decode(errors="replace").rstrip())
                    except ValueError:
                        pass  # pipe 关闭 = 子进程已退出

                _depth_stderr_thread = threading.Thread(
                    target=_drain_stderr,
                    args=(_depth_proc,),
                    daemon=True,
                    name="depth-stderr-drain",
                )
                _depth_stderr_thread.start()

                # 轮询 /health 直至就绪
                deadline = time.monotonic() + 120.0
                ready = False
                while time.monotonic() < deadline:
                    if _depth_proc.poll() is not None:
                        # stderr 已由 _drain_stderr 线程持续消费并 log DEBUG,
                        # 这里不再直接读 pipe（避免与 drain 线程竞态）。
                        log.error(
                            "深度推理服务启动失败(exit=%d); "
                            "stderr 日志前缀 [da_v2_service], 建议添 -v 查看详情",
                            _depth_proc.returncode,
                        )
                        _depth_kill(_depth_proc)
                        _depth_proc = None
                        break
                    try:
                        with urllib.request.urlopen(health_url, timeout=2) as resp:
                            if resp.status == 200:
                                ready = True
                                break
                    except Exception:
                        log.debug("深度服务 health 尚未就绪, 继续等待 ...")
                    time.sleep(0.5)

                if ready:
                    log.info("深度推理服务就绪 ✓")
                    args.depth_service = service_url
                elif _depth_proc is not None:
                    log.error(
                        "深度推理服务启动超时(2min), 已停止; "
                        "请检查网络/模型下载状态后重试"
                    )
                    _depth_kill(_depth_proc)
                    _depth_proc = None

    # ── 注册 atexit 清理（正常退出时触发） ─────────────────
    atexit.register(_depth_kill, _depth_proc)

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
        gesture_flight_test=args.gesture_flight_test,
        sim=args.sim,
        enable_record=args.record,
        record_hz=args.record_hz,
        avoid_cruise=args.cruise,
        avoid_approach_pitch=args.approach_pitch,
        avoid_yaw=args.yaw,
    )
    # 手势后端通过 --inference gestures 选择；depth-anything 可注入远端服务地址
    kw = {}
    if args.inference in ("depth-anything", "da-v2", "depth") and args.depth_service:
        kw["service_url"] = args.depth_service
    backend = create_backend(args.inference, **kw)
    return App(cfg, inference=backend).run()


if __name__ == "__main__":
    raise SystemExit(main())
