#!/usr/bin/env python3
"""RoboMaster TT 组网模式辅助脚本（仅标准库）。

用法：
  1) Mac 连接飞机热点 RMTT-xxxx 后执行：
       python station_mode.py setup
     路由器 SSID/密码 默认读 wifi_config.json（首次运行进入配置向导），
     也可用 --ssid/--password 临时覆盖。飞机回复 OK 后会自动重启并加入路由器。
  2) Mac 切回路由器 Wi-Fi 后执行：
       python station_mode.py find
     在本机所在 /24 网段广播 `command` 探测，打印飞机的局域网 IP。
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

CMD_PORT = 8889
TELLO_AP_IP = "192.168.10.1"


def _open_socket(bind_ip: str = "") -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_ip, CMD_PORT))
    sock.settimeout(5.0)
    return sock


def _send(sock: socket.socket, addr: tuple[str, int], cmd: str) -> str:
    print(f">>> {cmd}")
    sock.sendto(cmd.encode(), addr)
    try:
        data, _ = sock.recvfrom(1024)
    except socket.timeout:
        return "<timeout>"
    reply = data.decode(errors="ignore").strip()
    print(f"<<< {reply}")
    return reply


def cmd_setup(ssid: str, password: str) -> int:
    sock = _open_socket()
    try:
        addr = (TELLO_AP_IP, CMD_PORT)
        if _send(sock, addr, "command") != "ok":
            print("无法进入 SDK 模式：请确认 Mac 已连接 RMTT-xxxx 热点、飞机电量充足", file=sys.stderr)
            return 1
        reply = _send(sock, addr, f"ap {ssid} {password}")
    finally:
        sock.close()
    if reply.lower().startswith("ok"):
        print(f"成功：飞机将重启并加入路由器 [{ssid}]。约 10 秒后把 Mac 切回该 Wi-Fi，再运行: python station_mode.py find")
        return 0
    print(f"设置失败：{reply}", file=sys.stderr)
    return 1


def get_lan_ip() -> str:
    # 优先取 Wi-Fi 网卡的地址（默认路由可能被 VPN 虚拟网卡接管，不可信）
    import subprocess

    from wifi_config import detect_wifi_interface
    try:
        ip = subprocess.run(
            ["ipconfig", "getifaddr", detect_wifi_interface()],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if ip:
            return ip
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 53))
        return probe.getsockname()[0]
    except OSError:
        return ""
    finally:
        probe.close()


def find_drone(local_ip: str, timeout: float = 3.0) -> list[str]:
    """向 /24 网段多轮发送 `command`，降低 UDP 单轮扫描漏检概率。"""
    prefix = local_ip.rsplit(".", 1)[0]
    sock = _open_socket(local_ip)
    sock.settimeout(0.2)
    found: list[str] = []
    try:
        deadline = time.time() + timeout
        next_sweep = 0.0
        while time.time() < deadline:
            now = time.time()
            if now >= next_sweep:
                next_sweep = now + 0.75
                for i in range(1, 255):
                    target = f"{prefix}.{i}"
                    if target == local_ip:
                        continue
                    try:
                        sock.sendto(b"command", (target, CMD_PORT))
                    except OSError:
                        continue  # 接口切换瞬间可能 No route to host，跳过该地址
            try:
                data, addr = sock.recvfrom(1024)
            except socket.timeout:
                continue
            if data.decode(errors="ignore").strip() == "ok" and addr[0] not in found:
                found.append(addr[0])
                # 已找到即可返回，避免让飞机反复进入 SDK 模式。
                break
    finally:
        sock.close()
    return found


def cmd_find(timeout: float = 3.0) -> int:
    local_ip = get_lan_ip()
    if not local_ip:
        print("本机没有可用网络地址", file=sys.stderr)
        return 1
    prefix = local_ip.rsplit(".", 1)[0]
    print(f"本机 IP {local_ip}，扫描 {prefix}.1-254 ...")
    found = find_drone(local_ip, timeout)
    for ip in found:
        print(f"找到飞机: {ip}")

    if not found:
        print("未找到。确认飞机指示灯为组网状态、与 Mac 在同一路由器下，稍等片刻重试", file=sys.stderr)
        return 1
    print(f"\n启动控制界面:\n  python main.py --tello-ip {found[0]} --local-ip {local_ip} -v")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    sp = sub.add_parser("setup", help="直连飞机热点时执行，发送 ap 组网指令")
    sp.add_argument("--ssid", default=None, help="路由器 Wi-Fi 名（默认读 wifi_config.json）")
    sp.add_argument("--password", default=None, help="路由器 Wi-Fi 密码（默认读 wifi_config.json）")
    sub.add_parser("find", help="连接路由器 Wi-Fi 时执行，扫描飞机局域网 IP")
    args = p.parse_args()
    if args.action == "setup":
        from wifi_config import get_config
        cfg = get_config(args.ssid, args.password)
        return cmd_setup(cfg["router_ssid"], cfg["router_password"])
    return cmd_find()


if __name__ == "__main__":
    raise SystemExit(main())
