"""L1 契约：emergency 不得被 takeoff/land 的命令锁堵住。"""
from __future__ import annotations

import socket
import threading
import time

from tt_control.tello_client import TelloClient


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_emergency_bypasses_command_lock():
    cmd_port = _free_udp_port()
    state_port = _free_udp_port()
    client = TelloClient(
        "127.0.0.1",
        tello_ip="127.0.0.1",
        cmd_port=cmd_port,
        state_port=state_port,
    )
    held = threading.Event()
    release = threading.Event()

    def _hold_lock() -> None:
        with client._lock:
            held.set()
            release.wait(timeout=5.0)

    t = threading.Thread(target=_hold_lock, daemon=True)
    t.start()
    assert held.wait(timeout=1.0)

    t0 = time.time()
    client.emergency()
    elapsed = time.time() - t0
    release.set()
    t.join(timeout=2.0)
    client.close()
    assert elapsed < 0.5, f"emergency blocked for {elapsed:.2f}s"


def test_emergency_packet_reaches_drone_port():
    """临时 socket 发出的 emergency 应到达飞机命令端口。"""
    cmd_port = _free_udp_port()
    state_port = _free_udp_port()
    drone = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    drone.bind(("127.0.0.1", cmd_port))
    drone.settimeout(1.0)

    # 客户端必须绑不同端口；TelloClient 会占 cmd_port，故换本地 IP 语义：
    # 用另一对端口模拟「本机命令口」与「飞机命令口」。
    local_cmd = _free_udp_port()
    local_state = _free_udp_port()
    client = TelloClient(
        "127.0.0.1",
        tello_ip="127.0.0.1",
        cmd_port=local_cmd,
        state_port=local_state,
    )
    # 把目标改到探测 socket（飞机侧）
    client.tello_addr = ("127.0.0.1", cmd_port)
    try:
        client.emergency()
        data, _ = drone.recvfrom(64)
        assert data == b"emergency"
    finally:
        client.close()
        drone.close()
