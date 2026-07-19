#!/usr/bin/env python3
"""全自动组网 + 启动控制界面。

首次运行会进入配置向导（生成本地 wifi_config.json，不会提交到仓库），
之后零参数运行：python auto_fly.py

流程（无人值守）：
  0. 若飞机已在路由器局域网内 → 直接跳到第 4 步
  1. 等待用户把 Mac Wi-Fi 连到 RMTT-xxxx 热点（检测 Wi-Fi 网卡出现 192.168.10.x）
  2. 发送 `ap <ssid> <password>` 组网指令（带重试）
  3. 自动把 Mac Wi-Fi 切回路由器，等待拿到局域网 IP
  4. 扫描找到飞机 IP（飞机重启需要时间，带重试）
  5. 后台拉起 main.py 控制界面
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

import wifi_config
from station_mode import cmd_setup, find_drone, get_lan_ip

REPO = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(REPO, ".venv", "bin", "python")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wifi_ip(iface: str) -> str:
    try:
        return subprocess.run(
            ["ipconfig", "getifaddr", iface],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def switch_wifi(iface: str, ssid: str, password: str = "") -> None:
    cmd = ["networksetup", "-setairportnetwork", iface, ssid]
    if password:
        cmd.append(password)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        log(f"networksetup: {out}")


def try_find(retries: int, interval: float) -> tuple[str, str]:
    """返回 (本机IP, 飞机IP)；找不到返回 ('', '')。"""
    for i in range(retries):
        lan = get_lan_ip()
        if lan and not lan.startswith("192.168.10."):
            found = find_drone(lan, timeout=3.0)
            if found:
                return lan, found[0]
        log(f"扫描第 {i + 1}/{retries} 次未找到飞机（本机 {lan or '无网络'}），{interval}s 后重试")
        time.sleep(interval)
    return "", ""


def launch_gui(
    tello_ip: str,
    local_ip: str,
    inference: str = "gestures",
    gesture_dry_run: bool = False,
) -> None:
    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
    gui_log = os.path.join(REPO, "logs", "gui.log")
    with open(gui_log, "a") as f:
        cmd = [PYTHON, os.path.join(REPO, "main.py"),
               "--tello-ip", tello_ip, "--local-ip", local_ip,
               "--inference", inference, "-v"]
        if gesture_dry_run:
            cmd.append("--gesture-dry-run")
        subprocess.Popen(
            cmd,
            stdout=f, stderr=f, cwd=REPO, start_new_session=True,
        )
    log(f"控制界面已启动（日志 {gui_log}）：点 CONNECT → 按 T 起飞，L 降落，Esc 紧急停桨")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ssid", default=None, help="路由器 Wi-Fi 名（默认读 wifi_config.json）")
    p.add_argument("--password", default=None, help="路由器 Wi-Fi 密码（默认读 wifi_config.json）")
    p.add_argument("--drone-ssid", default=None, help="飞机热点名（已知网络时可自动连接）")
    p.add_argument(
        "--inference",
        choices=("gestures", "passthrough"),
        default="gestures",
        help="控制界面推理后端（默认启用纯视觉手势）",
    )
    p.add_argument(
        "--gesture-dry-run",
        action="store_true",
        help="只识别和显示手势，不发送飞行命令",
    )
    p.add_argument("--wait-hotspot", type=float, default=480, help="等待连上 RMTT 热点的秒数")
    args = p.parse_args()

    cfg = wifi_config.get_config(args.ssid, args.password, args.drone_ssid)
    ssid, password = cfg["router_ssid"], cfg["router_password"]
    drone_ssid, iface = cfg["drone_ssid"], cfg["wifi_interface"]

    # 0. 飞机可能已经组网成功，直接找
    log("先检查飞机是否已在局域网内 ...")
    lan, drone = try_find(retries=1, interval=0)
    if drone:
        log(f"飞机已在局域网: {drone}，跳过组网步骤")
        launch_gui(drone, lan, args.inference, args.gesture_dry_run)
        return 0

    # 1. 等待飞机热点出现并连接（已知网络可全自动；否则等用户手动点击）
    if drone_ssid:
        log(f"等待飞机热点 [{drone_ssid}] 出现，每 15 秒自动尝试连接 ...")
    else:
        log("请把飞机开机，并在 Mac 的 Wi-Fi 菜单中手动连接 RMTT-xxxx 热点 ...")
    deadline = time.time() + args.wait_hotspot
    connected = False
    last_try = 0.0
    while time.time() < deadline:
        ip = wifi_ip(iface)
        if ip.startswith("192.168.10."):
            log(f"已连上飞机热点（本机 {ip}）")
            connected = True
            break
        if drone_ssid and time.time() - last_try >= 15:
            last_try = time.time()
            switch_wifi(iface, drone_ssid)
        time.sleep(3)
    if not connected:
        log("超时：一直没有连上飞机热点。确认飞机已长按电源键 5 秒重置 WiFi 后，重新运行本脚本")
        switch_wifi(iface, ssid, password)
        return 1

    # 2. 发组网指令
    ok = False
    for i in range(3):
        log(f"发送组网指令（第 {i + 1}/3 次）: ap {ssid} ***")
        if cmd_setup(ssid, password) == 0:
            ok = True
            break
        time.sleep(3)
    if not ok:
        log("组网指令失败：检查飞机电量（低电量会拒绝 SDK 模式）后重跑本脚本")
        return 1

    # 3. 切回路由器
    time.sleep(3)
    log(f"把 Mac Wi-Fi 切回 [{ssid}] ...")
    switch_wifi(iface, ssid, password)
    deadline = time.time() + 60
    while time.time() < deadline:
        ip = wifi_ip(iface)
        if ip and not ip.startswith("192.168.10.") and not ip.startswith("169.254."):
            log(f"已回到路由器网络（本机 {ip}）")
            break
        time.sleep(2)
    else:
        log(f"没能自动切回 [{ssid}]，请手动在 Wi-Fi 菜单选择它，脚本继续扫描 ...")

    # 4. 等飞机重启入网并扫描
    log("等待飞机重启加入路由器（约 10-30 秒）...")
    lan, drone = try_find(retries=12, interval=10)
    if not drone:
        log("扫描不到飞机。确认飞机指示灯为组网状态后重跑本脚本（会直接跳到扫描步骤）")
        return 1

    log(f"找到飞机: {drone}（本机 {lan}）")
    # 5. 拉起控制界面
    launch_gui(drone, lan, args.inference, args.gesture_dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
