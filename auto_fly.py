#!/usr/bin/env python3
"""全自动组网 + 启动控制界面。

流程（无人值守）：
  0. 若飞机已在路由器局域网内 → 直接跳到第 4 步
  1. 等待用户把 Mac Wi-Fi 连到 RMTT-xxxx 热点（检测 en0 出现 192.168.10.x）
  2. 发送 `ap <ssid> <password>` 组网指令（带重试）
  3. 自动把 Mac Wi-Fi 切回路由器，等待拿到局域网 IP
  4. 扫描找到飞机 IP（飞机重启需要时间，带重试）
  5. 后台拉起 main.py 控制界面
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from station_mode import cmd_setup, find_drone, get_lan_ip

REPO = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(REPO, ".venv", "bin", "python")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wifi_ip() -> str:
    try:
        return subprocess.run(
            ["ipconfig", "getifaddr", "en0"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def switch_wifi(ssid: str, password: str = "") -> None:
    cmd = ["networksetup", "-setairportnetwork", "en0", ssid]
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


def launch_gui(tello_ip: str, local_ip: str) -> None:
    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
    gui_log = os.path.join(REPO, "logs", "gui.log")
    with open(gui_log, "a") as f:
        subprocess.Popen(
            [PYTHON, os.path.join(REPO, "main.py"),
             "--tello-ip", tello_ip, "--local-ip", local_ip, "-v"],
            stdout=f, stderr=f, cwd=REPO, start_new_session=True,
        )
    log(f"控制界面已启动（日志 {gui_log}）：点 CONNECT → 按 T 起飞，L 降落，Esc 紧急停桨")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ssid", required=True, help="路由器 Wi-Fi 名称")
    p.add_argument("--password", required=True, help="路由器 Wi-Fi 密码")
    p.add_argument("--drone-ssid", default="", help="飞机热点名（已知网络时可自动连接）")
    p.add_argument("--wait-hotspot", type=float, default=480, help="等待连上 RMTT 热点的秒数")
    args = p.parse_args()

    # 0. 飞机可能已经组网成功，直接找
    log("先检查飞机是否已在局域网内 ...")
    lan, drone = try_find(retries=1, interval=0)
    if drone:
        log(f"飞机已在局域网: {drone}，跳过组网步骤")
        launch_gui(drone, lan)
        return 0

    # 1. 等待飞机热点出现并连接（已知网络可全自动；否则等用户手动点击）
    log(f"等待飞机热点 [{args.drone_ssid}] 出现，每 15 秒自动尝试连接 ...")
    deadline = time.time() + args.wait_hotspot
    connected = False
    last_try = 0.0
    while time.time() < deadline:
        ip = wifi_ip()
        if ip.startswith("192.168.10."):
            log(f"已连上飞机热点（本机 {ip}）")
            connected = True
            break
        if args.drone_ssid and time.time() - last_try >= 15:
            last_try = time.time()
            switch_wifi(args.drone_ssid)
        time.sleep(3)
    if not connected:
        log("超时：一直没有连上飞机热点。确认飞机已长按电源键 5 秒重置 WiFi 后，重新运行本脚本")
        switch_wifi(args.ssid, args.password)
        return 1

    # 2. 发组网指令
    ok = False
    for i in range(3):
        log(f"发送组网指令（第 {i + 1}/3 次）: ap {args.ssid} ***")
        if cmd_setup(args.ssid, args.password) == 0:
            ok = True
            break
        time.sleep(3)
    if not ok:
        log("组网指令失败：检查飞机电量（低电量会拒绝 SDK 模式）后重跑本脚本")
        return 1

    # 3. 切回路由器
    time.sleep(3)
    log(f"把 Mac Wi-Fi 切回 [{args.ssid}] ...")
    switch_wifi(args.ssid, args.password)
    deadline = time.time() + 60
    while time.time() < deadline:
        ip = wifi_ip()
        if ip and not ip.startswith("192.168.10.") and not ip.startswith("169.254."):
            log(f"已回到路由器网络（本机 {ip}）")
            break
        time.sleep(2)
    else:
        log(f"没能自动切回 [{args.ssid}]，请手动在 Wi-Fi 菜单选择它，脚本继续扫描 ...")

    # 4. 等飞机重启入网并扫描
    log("等待飞机重启加入路由器（约 10-30 秒）...")
    lan, drone = try_find(retries=12, interval=10)
    if not drone:
        log("扫描不到飞机。确认飞机指示灯为组网状态后重跑本脚本（会直接跳到扫描步骤）")
        return 1

    log(f"找到飞机: {drone}（本机 {lan}）")
    # 5. 拉起控制界面
    launch_gui(drone, lan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
