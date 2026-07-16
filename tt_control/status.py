"""探测飞机是否在线（不改动有线网络）。"""

from __future__ import annotations

import socket
import subprocess
from typing import Optional


def ping_host(host: str, timeout_s: float = 1.0) -> bool:
    try:
        r = subprocess.run(
            ["ping", "-c", "1", "-W", str(max(1, int(timeout_s))), host],
            capture_output=True,
            text=True,
            timeout=timeout_s + 1.5,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def udp_probe(host: str, port: int = 8889, local_ip: str = "", timeout_s: float = 1.0) -> bool:
    """发送 command 探测；能收到任意应答则视为在线。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout_s)
    try:
        if local_ip:
            sock.bind((local_ip, 0))
        sock.sendto(b"command", (host, port))
        sock.recvfrom(1024)
        return True
    except OSError:
        return False
    finally:
        sock.close()


def is_drone_online(tello_ip: str, local_ip: str = "") -> bool:
    if ping_host(tello_ip):
        return True
    if local_ip:
        return udp_probe(tello_ip, local_ip=local_ip)
    return False
