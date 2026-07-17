#!/usr/bin/env python3
"""本地 Wi-Fi 配置模块（仅标准库）。

配置优先级：命令行参数 > wifi_config.json > 首次运行交互向导。

真实配置保存在仓库根目录 wifi_config.json（已加入 .gitignore，不会被提交）；
仓库中只提交模板 wifi_config.example.json。想重新配置，删掉 wifi_config.json
再运行任意脚本即可重新进入向导。
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(REPO, "wifi_config.json")

DEFAULTS = {
    "router_ssid": "",
    "router_password": "",
    "drone_ssid": "",
    "wifi_interface": "",
}

_iface_cache = ""


def detect_wifi_interface() -> str:
    """通过 networksetup 找到 Wi-Fi 网卡（en0/en1...），失败时退回 en0。"""
    global _iface_cache
    if _iface_cache:
        return _iface_cache
    try:
        out = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "en0"
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "Wi-Fi" in line or "AirPort" in line:
            for follow in lines[i + 1:i + 3]:
                if follow.startswith("Device:"):
                    _iface_cache = follow.split(":", 1)[1].strip()
                    return _iface_cache
    return "en0"


def current_ssid(interface: str) -> str:
    """探测当前连接的 Wi-Fi 名，探测不到返回空串。"""
    try:
        out = subprocess.run(
            ["networksetup", "-getairportnetwork", interface],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        # 形如 "Current Wi-Fi Network: DS"
        if ":" in out and "not associated" not in out.lower():
            ssid = out.split(":", 1)[1].strip()
            if ssid:
                return ssid
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    # 新版 macOS 上 networksetup 可能拿不到，改用 ipconfig getsummary
    try:
        out = subprocess.run(
            ["ipconfig", "getsummary", interface],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("SSID") and ":" in line:
                return line.split(":", 1)[1].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ""


def load() -> dict | None:
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return {**DEFAULTS, **json.load(f)}


def save(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.chmod(CONFIG_PATH, 0o600)  # 含明文密码，仅本人可读


def run_wizard() -> dict:
    if not sys.stdin.isatty():
        print(
            f"未找到 {CONFIG_PATH} 且当前无法交互输入；"
            "请复制 wifi_config.example.json 为 wifi_config.json 并填写",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("未找到 wifi_config.json，开始首次配置（只需一次）：")
    interface = detect_wifi_interface()
    ssid_now = current_ssid(interface)
    if ssid_now:
        ssid = input(f"路由器 Wi-Fi 名（检测到当前连接 [{ssid_now}]，回车直接使用）: ").strip() or ssid_now
    else:
        ssid = ""
        while not ssid:
            ssid = input("路由器 Wi-Fi 名: ").strip()
    password = ""
    while not password:
        password = getpass.getpass(f"[{ssid}] 的 Wi-Fi 密码（输入不回显）: ")
    drone = input("飞机热点名（如 RMTT-A1B2C3，不知道可留空，届时手动连热点）: ").strip()
    cfg = {
        "router_ssid": ssid,
        "router_password": password,
        "drone_ssid": drone,
        "wifi_interface": interface,
    }
    save(cfg)
    print(f"✅ 已保存到 {CONFIG_PATH}（已在 .gitignore 中，不会被提交）")
    return cfg


def get_config(ssid: str | None = None,
               password: str | None = None,
               drone_ssid: str | None = None) -> dict:
    """合并 命令行参数 > 配置文件 > 交互向导，返回完整配置。"""
    cfg = load()
    if cfg is None and not (ssid and password):
        cfg = run_wizard()
    cfg = {**DEFAULTS, **(cfg or {})}
    if ssid:
        cfg["router_ssid"] = ssid
    if password:
        cfg["router_password"] = password
    if drone_ssid:
        cfg["drone_ssid"] = drone_ssid
    if not cfg["wifi_interface"]:
        cfg["wifi_interface"] = detect_wifi_interface()
    return cfg
