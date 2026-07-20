"""运行配置。不修改系统有线网络，仅选用 Wi-Fi 地址与飞机通信。"""

from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass


TELLO_IP = "192.168.10.1"
CMD_PORT = 8889
STATE_PORT = 8890
VIDEO_PORT = 11111

# rc 通道默认杆量（-100 ~ 100）
RC_SPEED = 40


@dataclass
class AppConfig:
    tello_ip: str = TELLO_IP
    local_ip: str = ""
    cmd_port: int = CMD_PORT
    state_port: int = STATE_PORT
    video_port: int = VIDEO_PORT
    rc_speed: int = RC_SPEED
    window_name: str = "RoboMaster TT Control"
    heartbeat_interval: float = 5.0
    enable_mujoco: bool = False
    enable_mission_pad: bool = True
    gesture_commands_enabled: bool = True
    gesture_flight_test: bool = False
    sim: bool = False          # 使用 SimDrone/SimVideo 离线仿真(无需真机)


def detect_local_ip(preferred_prefix: str = "192.168.10.") -> str:
    """优先选择 Tello 直连网段地址；找不到则回退为空字符串。"""
    try:
        out = subprocess.check_output(["ip", "-4", "-o", "addr", "show"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        out = ""

    for line in out.splitlines():
        parts = line.split()
        if "inet" not in parts:
            continue
        addr = parts[parts.index("inet") + 1].split("/")[0]
        if addr.startswith(preferred_prefix):
            return addr

    # macOS / Windows / 无 ip 命令时：用 UDP 路由探测本机主用地址。
    # 优先 Tello 直连网段；找不到则回退为本机地址（如 station 模式下与飞机同网段）。
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((TELLO_IP, CMD_PORT))
        ip = s.getsockname()[0]
        s.close()
        if ip.startswith(preferred_prefix):
            return ip
        if ip and ip != "0.0.0.0":
            return ip
    except OSError:
        pass
    return ""
