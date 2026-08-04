#!/usr/bin/env python3
r"""Tello 连接自检(离线用)。

切到 TELLO WiFi 后运行:
    .\.venv\Scripts\python.exe diag_tello.py

逐层验证并把结果写入 diag.log。跑完切回公司 WiFi,把 diag.log 发回分析。
"""
from __future__ import annotations

import platform
import socket
import subprocess
import sys
import time

TELLO_IP = "192.168.10.1"
CMD_PORT = 8889
STATE_PORT = 8890
VIDEO_PORT = 11111

_lines: list[str] = []


def log(msg: str) -> None:
    print(msg)
    _lines.append(msg)


def flush() -> None:
    with open("diag.log", "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((TELLO_IP, CMD_PORT))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


def ping() -> bool:
    if platform.system().lower().startswith("win"):
        cmd = ["ping", "-n", "1", "-w", "1000", TELLO_IP]
    else:
        cmd = ["ping", "-c", "1", "-W", "1", TELLO_IP]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False


def main() -> int:
    log("==================== Tello 自检 ====================")

    # 1) 等待拿到 192.168.10.x 地址
    ip = ""
    for _ in range(15):
        ip = local_ip()
        if ip.startswith("192.168.10."):
            break
        time.sleep(1)
    log(f"[1] 本机IP: {ip!r}")
    if not ip.startswith("192.168.10."):
        log("    ✗ 没连到 TELLO 热点(未拿到 192.168.10.x)。")
        log("    → 请在 WiFi 菜单手动选择 TELLO-E9F014,确认已连上(即使显示'无 Internet')后重跑。")
        flush()
        return 1
    log("    ✓ 已在 Tello 网段")

    # 2) ping 飞机
    log(f"[2] ping {TELLO_IP}: {'✓ 通' if ping() else '✗ 不通(飞机可能未就绪)'}")

    # 3) command 握手
    cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd.settimeout(5.0)
    try:
        cmd.bind((ip, CMD_PORT))
    except OSError as e:
        log(f"[3] ✗ 绑定 {ip}:{CMD_PORT} 失败: {e}")
        flush()
        return 2

    def send(c: str, timeout: float = 5.0):
        cmd.settimeout(timeout)
        log(f"    >>> {c}")
        try:
            cmd.sendto(c.encode(), (TELLO_IP, CMD_PORT))
            data, _ = cmd.recvfrom(2048)
            r = data.decode(errors="ignore").strip()
            log(f"    <<< {r}")
            return r
        except socket.timeout:
            log(f"    !! 超时,无应答: {c}")
            return None

    log("[3] SDK 握手:")
    if send("command", 5.0) != "ok":
        log("    ✗ command 未返回 ok。飞机可能忙/需重启,或有其它程序占用了 8889。")
        flush()
        cmd.close()
        return 3
    log("    ✓ 进入 SDK 模式")
    log(f"[4] 电量 battery?: {send('battery?', 5.0)}")

    # 5) streamon + 图传收包
    log("[5] 图传测试:")
    if send("streamon", 5.0) == "ok":
        v = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        v.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
        v.settimeout(5.0)
        try:
            v.bind((ip, VIDEO_PORT))
            n, t0 = 0, time.time()
            while time.time() - t0 < 5.0:
                try:
                    v.recvfrom(65535)
                    n += 1
                except socket.timeout:
                    break
            if n > 0:
                log(f"    ✓ 5秒内收到 {n} 个视频UDP包 → 图传正常")
            else:
                log("    ✗ 收不到视频包。防火墙已确认关闭,若仍收不到需查杀软/网卡入站策略。")
        except OSError as e:
            log(f"    ✗ 绑定视频端口失败: {e}")
        finally:
            v.close()
            send("streamoff", 3.0)
    else:
        log("    ✗ streamon 未返回 ok")

    # 6) 状态包(8890)
    log("[6] 状态回传测试:")
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    st.settimeout(3.0)
    try:
        st.bind((ip, STATE_PORT))
        data, _ = st.recvfrom(2048)
        log(f"    ✓ 收到状态: {data.decode(errors='ignore').strip()[:90]}")
    except socket.timeout:
        log("    ✗ 3秒内收不到状态包(8890)")
    except OSError as e:
        log(f"    ✗ 绑定状态端口失败: {e}")
    finally:
        st.close()

    cmd.close()
    log("==================== 自检结束 ====================")
    log("把上面内容 / diag.log 发回即可分析。")
    flush()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        log(f"[异常] {e!r}")
        flush()
        sys.exit(99)
