"""探测飞机是否在线（不改动有线网络）。"""

from __future__ import annotations

import platform
import socket
import subprocess
from typing import Optional


def ping_host(host: str, timeout_s: float = 1.0) -> bool:
    if platform.system().lower().startswith("win"):
        # Windows: -n 次数, -w 超时(毫秒)
        cmd = ["ping", "-n", "1", "-w", str(max(1, int(timeout_s * 1000))), host]
    else:
        # Linux/macOS: -c 次数, -W 超时(秒)
        cmd = ["ping", "-c", "1", "-W", str(max(1, int(timeout_s))), host]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s + 1.5,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def udp_probe(host: str, port: int = 8889, local_ip: str = "", timeout_s: float = 1.0) -> bool:
    """发送 command 探测；能收到任意应答则视为在线。

    必须从源端口 8889 发送：RMTT 组网模式会把命令回复锁定到首个握手的
    IP:端口，随机源端口的探测包会把锁抢到即将关闭的临时端口，导致主
    连接全部超时（2026-07-24 排障结论）。8889 被占用（说明主客户端已在
    用它，探测无必要）时直接返回 False，绝不退化为随机端口。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout_s)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((local_ip or "", port))
        except OSError:
            return False  # 8889 已被 TelloClient 占用，不能从别的端口发 command
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
