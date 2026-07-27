#!/usr/bin/env python3
"""Tello 一次性配网脚本（安全版）

在 macOS Terminal.app 中本地运行，绝不会自动切换 Wi-Fi。
只做两件事：(1) 引导你手动连 Tello 热点并发 ap 入网命令，
(2) 引导你手动切回路由器后扫描 Tello 的局域网 IP。

用法:
    python3 tello_provision.py

安全保证:
    - 不调用任何切换 Wi-Fi 的 API（你手动操作，脚本只检测）
    - ap 密码仅在发 UDP 包时使用，日志及终端输出全部脱敏
    - 每一步均有超时保护，超时退出而非死循环
    - 纯标准库，无外部依赖
    - Wi-Fi 密码经 --router-password 或交互 getpass 传入，不硬编码
"""

from __future__ import annotations

import argparse
import getpass
import socket
import subprocess
import sys
import time

# ── 配置（CLI 可覆盖；密码禁止硬编码，空则 getpass）──────────────
ROUTER_SSID = "DS"
ROUTER_PASSWORD = ""
TELLO_HOTSPOT = "TELLO-E9F873"

TELLO_AP_IP = "192.168.10.1"
CMD_PORT = 8889
HOTSPOT_PREFIX = "192.168.10."

# ── 工具函数 ──────────────────────────────────────────────────────


def _run(argv: list[str], timeout: float = 5.0) -> str:
    """运行命令，超时或失败返回空串。"""
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def detect_wifi_interface() -> str:
    """查找 macOS Wi-Fi 接口名（en0 / en1 ...），找不到回退 en0。"""
    out = _run(["networksetup", "-listallhardwareports"])
    if out:
        lines = out.splitlines()
        for i, line in enumerate(lines):
            if "Wi-Fi" in line or "AirPort" in line:
                for follow in lines[i + 1 : i + 3]:
                    if follow.startswith("Device:"):
                        return follow.split(":", 1)[1].strip()
    return "en0"


def get_ssid(iface: str) -> str:
    """获取当前 Wi-Fi SSID。先用 ipconfig，失败回退 networksetup。"""
    # 方法1: ipconfig getsummary（新版 macOS 更可靠）
    out = _run(["ipconfig", "getsummary", iface])
    if out:
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("SSID") and ":" in line:
                ssid = line.split(":", 1)[1].strip()
                if ssid:
                    return ssid
    # 方法2: networksetup（旧版 macOS）
    out = _run(["networksetup", "-getairportnetwork", iface])
    if out and ":" in out and "not associated" not in out.lower():
        ssid = out.split(":", 1)[1].strip()
        if ssid:
            return ssid
    return ""


def get_ip(iface: str) -> str:
    """获取接口 IP 地址。"""
    out = _run(["ipconfig", "getifaddr", iface])
    ip = out.strip()
    if ip:
        return ip
    # 回退：用 ifconfig 解析
    out = _run(["ifconfig", iface])
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("inet ") and "netmask" in line:
            parts = line.split()
            for p in parts:
                if "." in p and not p.startswith("0x"):
                    return p
    return ""


def print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def prompt_enter(msg: str = "按 Enter 继续") -> None:
    try:
        input(f"\n{msg}...")
    except (EOFError, KeyboardInterrupt):
        print("\n用户取消")
        sys.exit(0)


def wait_for(condition, timeout: float, interval: float = 2.0, tick_msg: str = "") -> bool:
    """轮询等待条件满足。tick_msg 为每次轮询时打印的提示。"""
    deadline = time.time() + timeout
    first = True
    while time.time() < deadline:
        result = condition()
        if result:
            return True
        if first and tick_msg:
            print(f"  {tick_msg}")
            first = False
        time.sleep(interval)
        print(".", end="", flush=True)
    print()
    return False


# ── 核心步骤 ──────────────────────────────────────────────────────


def step_check_prerequisites(iface: str) -> str:
    """打印当前网络状态，返回当前 IP（如已连路由器则可能复用）。"""
    ssid = get_ssid(iface)
    ip = get_ip(iface)
    print_header("当前网络状态")
    print(f"  Wi-Fi 接口 : {iface}")
    print(f"  当前 SSID  : {ssid or '(未关联)'}")
    print(f"  当前 IP    : {ip or '(无)'}")
    return ip


def step_connect_tello(iface: str, tello_hotspot: str) -> str:
    """引导用户连 Tello 热点，检测到后返回本机 192.168.10.x IP。"""
    print_header(f"第 1 步：连接 Tello 热点")
    print(f"  请在 Mac 菜单栏 Wi-Fi 中选择: {tello_hotspot}")
    print(f"  ⚠️  此时本会话会断网，本脚本在 Terminal 中继续运行。")
    prompt_enter("确认准备好后按 Enter")

    def _on_tello() -> bool:
        ip = get_ip(iface)
        return bool(ip and ip.startswith(HOTSPOT_PREFIX))

    if wait_for(_on_tello, timeout=120, tick_msg="等待连接 Tello 热点..."):
        ip = get_ip(iface)
        ssid = get_ssid(iface)
        print(f"\n  ✓ 已连上 Tello 热点")
        print(f"    SSID: {ssid or tello_hotspot}")
        print(f"    IP  : {ip}")
        return ip
    else:
        print("\n  ✗ 超时：2 分钟内未检测到 Tello 热点连接")
        print("    请确认 Tello 已开机（LED 闪烁），Wi-Fi 列表中可见该热点。")
        print("    确认后重新运行本脚本即可。")
        sys.exit(1)


def step_send_ap(bind_ip: str, router_ssid: str, router_password: str) -> None:
    """发 command → ap 两条 UDP 指令。密码只在发 ap 时用，日志脱敏。"""
    print_header("第 2 步：发送入网指令")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((bind_ip, CMD_PORT))
    except OSError as e:
        print(f"  ✗ 无法绑定 {bind_ip}:{CMD_PORT}: {e}")
        print("    请确认没有其他程序占用 8889 端口后重试。")
        sock.close()
        sys.exit(1)
    sock.settimeout(5.0)

    try:
        # 1) 进入 SDK 模式
        print(f"  >>> command")
        try:
            sock.sendto(b"command", (TELLO_AP_IP, CMD_PORT))
            data, _ = sock.recvfrom(1024)
            reply = data.decode(errors="ignore").strip()
        except socket.timeout:
            print("  ✗ Tello 无应答（command 超时）")
            print("    请确认 Tello 电量充足（≥30%）、未进入保护模式。")
            print("    长按 Tello 电源键 5 秒重置 Wi-Fi 后重试。")
            sys.exit(1)

        print(f"  <<< {reply}")
        if reply.lower() != "ok":
            print(f"  ✗ SDK 模式进入失败: {reply}")
            sys.exit(1)

        # 2) 发 ap 入网指令（日志脱敏）
        ap_cmd = f"ap {router_ssid} {router_password}"
        # 终端上只显示脱敏版本
        print(f"  >>> ap {router_ssid} ***")
        try:
            sock.sendto(ap_cmd.encode(), (TELLO_AP_IP, CMD_PORT))
            data, _ = sock.recvfrom(1024)
            reply = data.decode(errors="ignore").strip()
        except socket.timeout:
            print("  ✗ Tello 无应答（ap 超时）")
            print("    Tello 可能正在重启或已拒绝该指令。")
            sys.exit(1)

        # 脱敏回复（万一回复里含密码）
        safe_reply = reply.replace(router_password, "***") if router_password else reply
        print(f"  <<< {safe_reply}")

        if reply.lower().startswith("ok"):
            print()
            print(f"  ✓ 入网成功！Tello 将重启并加入 [{router_ssid}]。")
            print(f"    约 10-15 秒后进入下一步。")
        else:
            print(f"  ✗ 入网失败: {safe_reply}")
            print("    可能原因：SSID/密码错误，或 Tello 不支持该路由器加密方式。")
            sys.exit(1)
    finally:
        sock.close()


def step_reconnect_router(iface: str, router_ssid: str) -> str:
    """引导用户切回路由器，返回路由器局域网 IP。"""
    print_header(f"第 3 步：切回 [{router_ssid}]")
    print(f"  请在 Mac 菜单栏 Wi-Fi 中选择: {router_ssid}")
    print(f"  连接成功后本脚本继续扫描 Tello。")
    prompt_enter("确认连接好 [{router_ssid}] 后按 Enter")

    # 等拿到有效局域网 IP
    def _on_router() -> bool:
        ssid = get_ssid(iface)
        ip = get_ip(iface)
        valid = bool(ip and not ip.startswith(HOTSPOT_PREFIX) and not ip.startswith("169.254."))
        if ssid and ssid != router_ssid:
            print(f"\n  ⚠️  当前连接 [{ssid}] 而非 [{router_ssid}]")
        return valid

    if wait_for(_on_router, timeout=120, tick_msg="等待获取路由器 IP..."):
        ssid = get_ssid(iface)
        ip = get_ip(iface)
        print(f"\n  ✓ 已回到路由器网络")
        print(f"    SSID: {ssid or '(未知)'}")
        print(f"    IP  : {ip}")
        if ssid and ssid != router_ssid:
            print(f"  ⚠️  注意：当前连接的不是 [{router_ssid}]，是 [{ssid}]。")
            print(f"    如果 Tello 加入的是 [{router_ssid}]，则它们不在同一局域网。")
            prompt_enter("确认无误，继续扫描")
        return ip
    else:
        print("\n  ✗ 超时：2 分钟内未获取有效 IP")
        print("    请手动连接 Wi-Fi 后重新运行: python3 station_mode.py find")
        sys.exit(1)


def step_scan_tello(lan_ip: str) -> str:
    """广播扫描 Tello 局域网 IP，找不到时退出。"""
    print_header("第 4 步：扫描 Tello 局域网 IP")
    prefix = lan_ip.rsplit(".", 1)[0]
    print(f"  本机 {lan_ip}，子网 {prefix}.1-254")
    print(f"  等待 Tello 重启入网 + 多轮扫描（约 60 秒）...")

    found: str | None = None
    deadline = time.time() + 60
    attempt = 0

    while time.time() < deadline and found is None:
        attempt += 1
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((lan_ip, CMD_PORT))
            except OSError:
                # 绝不能退化为随机源端口：飞机会把命令回复锁定到首个握手端口，
                # 随机端口扫描会让后续 8889 主连接全部超时（2026-07-24 排障结论）
                sock.close()
                print(f"  ✗ 本机 {CMD_PORT} 端口被占用，无法安全扫描。")
                print(f"    请先关闭占用进程（lsof -nP -iUDP:{CMD_PORT}）后重试。")
                sys.exit(1)

            # 发送阶段：短超时，尽力发完 /24
            sock.settimeout(0.1)
            for i in range(1, 255):
                target = f"{prefix}.{i}"
                if target == lan_ip:
                    continue
                try:
                    sock.sendto(b"command", (target, CMD_PORT))
                except OSError:
                    continue

            # 收包阶段：长超时，持续等待直到收齐或超时
            sock.settimeout(0.5)
            recv_deadline = time.time() + 2.0
            while time.time() < recv_deadline:
                try:
                    data, addr = sock.recvfrom(1024)
                    if data.decode(errors="ignore").strip() == "ok":
                        found = addr[0]
                        break
                except socket.timeout:
                    continue  # 还没收完，继续等
                except OSError:
                    break
        finally:
            sock.close()

        if found is None:
            print(f"    轮次 {attempt}: 未找到，5 秒后重试...")
            time.sleep(5)

    if found:
        print(f"\n  ✓ 找到 Tello！")
        print(f"    局域网 IP: {found}")
        return found
    else:
        print("\n  ✗ 未找到 Tello。可能原因：")
        print(f"    1. Tello 还在重启中（稍等 30 秒后运行: python station_mode.py find）")
        print(f"    2. Tello 距路由器太远或信号弱")
        print(f"    3. 路由器开启了客户端隔离（AP Isolation），需在路由器管理页关闭")
        print(f"    4. Tello 不支持当前路由器加密方式")
        print(f"    5. 检查 Tello LED：应为组网慢闪而非热点快闪")
        sys.exit(1)


# ── 主流程 ─────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Tello 一次性配网脚本（安全版 — 不自动切 Wi-Fi）"
    )
    p.add_argument("--router-ssid", default="", help=f"路由器 SSID（默认: {ROUTER_SSID}）")
    p.add_argument("--router-password", default="", help="路由器密码")
    p.add_argument("--tello-hotspot", default="", help=f"Tello 热点名（默认: {TELLO_HOTSPOT}）")
    args = p.parse_args()

    # CLI 优先，否则用模块常量
    router_ssid = args.router_ssid or ROUTER_SSID
    router_password = args.router_password or ROUTER_PASSWORD
    tello_hotspot = args.tello_hotspot or TELLO_HOTSPOT

    # 如果密码仍为空（模块常量也没设），交互式输入（不回显，不进入 ps）
    if not router_password:
        if sys.stdin.isatty():
            router_password = getpass.getpass(f"[{router_ssid}] Wi-Fi 密码（输入不回显）: ")
        if not router_password:
            print("错误：未提供路由器密码", file=sys.stderr)
            return 1

    if not router_ssid:
        print("错误：未提供路由器 SSID", file=sys.stderr)
        return 1

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        Tello 一次性配网脚本（安全版）                      ║")
    print("║                                                          ║")
    print("║  本脚本不会自动切换 Wi-Fi，每一步由你手动操作。             ║")
    print("║  全程只需在 Terminal 和 Wi-Fi 菜单之间切换。               ║")
    print("║                                                          ║")
    print("║  密码仅内存使用，不写入日志。                              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 0) 查接口
    iface = detect_wifi_interface()
    current_ip = step_check_prerequisites(iface)

    # 如果已经连在路由器上，看看 Tello 是否已在局域网
    if current_ip and not current_ip.startswith(HOTSPOT_PREFIX):
        prompt_enter("按 Enter 先扫描一下 Tello 是否已在局域网内")
        prefix = current_ip.rsplit(".", 1)[0]
        print(f"\n  快速扫描 {prefix}.1-254 ...")
        # 快速单轮扫
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((current_ip, CMD_PORT))
            sock.settimeout(0.15)
            for i in range(1, 255):
                target = f"{prefix}.{i}"
                if target == current_ip:
                    continue
                try:
                    sock.sendto(b"command", (target, CMD_PORT))
                except OSError:
                    continue
            deadline = time.time() + 2.0
            while time.time() < deadline:
                try:
                    data, addr = sock.recvfrom(1024)
                    if data.decode(errors="ignore").strip() == "ok":
                        print(f"  ✓ Tello 已在局域网！IP: {addr[0]}")
                        print(f"\n  启动控制界面:")
                        print(f"    python main.py --tello-ip {addr[0]} --local-ip {current_ip} -v")
                        sock.close()
                        return 0
                except socket.timeout:
                    break
            sock.close()
            print("  未找到，继续配网流程...")
        except OSError:
            if sock is not None:
                sock.close()

    # 1) 连 Tello 热点
    tello_ip = step_connect_tello(iface, tello_hotspot)

    # 2) 发入网指令
    step_send_ap(tello_ip, router_ssid, router_password)

    # 3) 切回路由器
    lan_ip = step_reconnect_router(iface, router_ssid)

    # 4) 扫描 Tello
    drone_ip = step_scan_tello(lan_ip)

    # 完成
    print_header("✅ 配网完成")
    print(f"  启动控制界面:")
    print(f"    python main.py --tello-ip {drone_ip} --local-ip {lan_ip} -v")
    print()
    print(f"  ⚠️  本脚本含 Wi-Fi 密码明文，配网完成后建议删除:")
    print(f"    rm tello_provision.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n\n中断退出。")
        raise SystemExit(130)
