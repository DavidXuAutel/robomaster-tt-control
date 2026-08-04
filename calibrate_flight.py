#!/usr/bin/env python3
"""脚本化校正飞行:用 rc 杆量做受控小动作,记录(动作 + 真机位姿 + 时间)→ frames.csv,
供 sim_match_report 标定 SimDrone 运动学常数(VMAX/VZMAX/YAWRATE)。

为什么用 rc 而非 SDK 位移指令:要标定的正是"rc 杆量→速度"这套模型(避障与 SimDrone
用的就是它);SDK 的 forward/back 走飞控自身里程闭环,是另一回事,不能用来标定 rc 模型。

安全:仅用于**有人在旁看护**的受控校正。内置:电量门 / 丢垫即停 / 任何异常即降 /
硬超时 / Ctrl-C 即降。动作幅度小、单张 Mission Pad 上方完成。

用法:
    python calibrate_flight.py --sim --axis fwd          # 离线自检
    python calibrate_flight.py --axis fwd --stick 25 --dur 1.2   # 真机(前后, 标定 VMAX)
    python calibrate_flight.py --axis up                  # 升降, 标定 VZMAX
    python calibrate_flight.py --axis yaw                 # 原地偏航, 标定 YAWRATE
产物:logs/calib/calib_<axis>_<stamp>.csv(可直接喂 sim_match_report.py)
"""

from __future__ import annotations

import argparse
import csv
import time
import pathlib

from tt_control.config import detect_local_ip

# 单位方向 (roll, pitch, throttle, yaw)
AXES = {"fwd": (0, 1, 0, 0), "up": (0, 0, 1, 0), "yaw": (0, 0, 0, 1)}
FIELDS = ["t_mono_ms", "act_roll", "act_pitch", "act_throttle", "act_yaw",
          "pad_id", "pos_x_cm", "pos_y_cm", "pos_z_cm", "yaw_deg", "height_cm",
          "vgx", "vgy", "vgz", "bat_pct"]


def build_client(sim: bool):
    if sim:
        from tt_control.sim_drone import SimDrone
        return SimDrone()
    from tt_control.tello_client import TelloClient
    ip = detect_local_ip()
    if not ip.startswith("192.168.10."):
        raise SystemExit(f"未连到飞机网段(detect_local_ip={ip!r});请先在控制台连 TELLO WiFi")
    print(f"WiFi 网卡地址: {ip}")
    return TelloClient(local_ip=ip, tello_ip="192.168.10.1")


def main() -> int:
    ap = argparse.ArgumentParser(description="脚本化校正飞行")
    ap.add_argument("--sim", action="store_true", help="用 SimDrone 离线自检")
    ap.add_argument("--axis", choices=list(AXES), default="fwd")
    ap.add_argument("--stick", type=int, default=25, help="杆量幅度(小,保证留在卡上)")
    ap.add_argument("--dur", type=float, default=2.0, help="每段(去/回)秒数;越长稳态越足")
    ap.add_argument("--hz", type=float, default=20.0, help="记录/下发频率")
    ap.add_argument("--no-pad", action="store_true",
                    help="不开 Mission Pad(诊断:靠 vgx/高度),用于测 VPS 速度是否恢复")
    ap.add_argument("--min-bat", type=int, default=30)
    ap.add_argument("--hard-timeout", type=float, default=30.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    c = build_client(args.sim)
    if not c.connect():
        raise SystemExit("connect 失败")
    c.start_state_listener()
    if not args.no_pad:
        c.mission_pad_on()
    time.sleep(1.0)
    try:
        bat = int(float(c.state.get("bat", "0") or 0))
    except ValueError:
        bat = 0
    print(f"起飞前:电量={bat}% mid(pad)={c.state.get('mid')}")
    if bat < args.min_bat:
        raise SystemExit(f"电量 {bat}% < {args.min_bat}%,中止")

    out = pathlib.Path(args.out) if args.out else (
        pathlib.Path.cwd() / "logs" / "calib" /
        f"calib_{args.axis}_{time.strftime('%H%M%S')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader()

    t0 = None
    dt = 1.0 / args.hz
    deadline = time.monotonic() + args.hard_timeout

    def rec(rc):
        nonlocal t0
        s = c.state
        now = time.monotonic()
        if t0 is None:
            t0 = now

        def num(k):
            try:
                return float(s.get(k))
            except (TypeError, ValueError):
                return ""
        writer.writerow({
            "t_mono_ms": int((now - t0) * 1000),
            "act_roll": rc[0], "act_pitch": rc[1], "act_throttle": rc[2], "act_yaw": rc[3],
            "pad_id": num("mid"), "pos_x_cm": num("x"), "pos_y_cm": num("y"),
            "pos_z_cm": num("z"), "yaw_deg": num("yaw"), "height_cm": num("h"),
            "vgx": num("vgx"), "vgy": num("vgy"), "vgz": num("vgz"),
            "bat_pct": num("bat"),
        })
        fh.flush()

    def _height():
        try:
            return int(float(c.state.get("h", "0") or 0))
        except (TypeError, ValueError):
            return 0

    def _send_land():
        try:
            c.rc(0, 0, 0, 0)
        except Exception:
            pass
        try:
            c.land() if args.sim else c.send("land", wait_response=False)
        except Exception:
            pass

    def safe_land(reason: str):
        # 降落用"发指令 + 高度确认"(land 'ok' 回执可能同样丢,不死等)
        print(f"LAND: {reason}")
        for _ in range(3):
            _send_land()
            t = time.monotonic()
            while time.monotonic() - t < 5.0:
                if _height() <= 15:
                    print("  已落地(高度确认)")
                    return
                time.sleep(0.2)

    def hover(sec, tag=""):
        t = time.monotonic()
        while time.monotonic() - t < sec:
            c.rc(0, 0, 0, 0)
            rec((0, 0, 0, 0))
            time.sleep(dt)

    def drive(rc, sec, stop_on_pad_loss):
        t = time.monotonic()
        while time.monotonic() - t < sec:
            if time.monotonic() > deadline:
                safe_land("hard timeout")
                return False
            if stop_on_pad_loss:
                try:
                    if float(c.state.get("mid", -1) or -1) < 0:
                        print("  丢垫 → 停止本段(避免飞出卡外)")
                        break
                except ValueError:
                    pass
            c.rc(*rc)
            rec(rc)
            time.sleep(dt)
        return True

    ux, up_, ut, uy = AXES[args.axis]
    s = args.stick
    ok = True
    try:
        # 起飞:该机型 takeoff 的 'ok' 回执可能丢(实测飞机已离地却无应答),
        # 故"发指令 + 用高度确认离地",不死等 'ok'。
        try:
            c.rc(0, 0, 0, 0)
        except Exception:
            pass
        if args.sim:
            c.takeoff()
        else:
            c.send("takeoff", wait_response=False)
        t = time.monotonic()
        took_off = False
        while time.monotonic() - t < 10.0:
            if _height() > 30:
                took_off = True
                break
            time.sleep(0.2)
        if not took_off:
            safe_land("起飞后 10s 未检测到离地高度(检查桨是否卡阻/机身水平/电量)")
            return 1
        print(f"已离地(高度 {_height()}cm),稳定 1.0s(不记录)")
        t = time.monotonic()
        while time.monotonic() - t < 1.0:
            c.rc(0, 0, 0, 0)
            time.sleep(dt)
        if args.no_pad:
            print("no-pad 诊断模式:跳过垫子,靠 vgx/高度;悬停 2s(开始记录)")
        else:
            # Mission Pad 锁定门:锁不到就没有位姿真值,校正数据无意义 → 直接降落
            t = time.monotonic()
            locked = False
            while time.monotonic() - t < 2.5:
                c.rc(0, 0, 0, 0)
                try:
                    if float(c.state.get("mid", -1) or -1) >= 0:
                        locked = True
                        break
                except ValueError:
                    pass
                time.sleep(dt)
            if not locked:
                safe_land("起飞后未锁定 Mission Pad(降低高度/把飞机摆正对准卡,重摆再飞)")
                return 1
            print(f"pad 已锁定(mid={c.state.get('mid')}),悬停 2s(开始记录)")
        hover(2.0)
        print(f"+ 段:{args.axis} 杆量 {s} × {args.dur}s")
        # 不因丢垫中断:速度(vgx)测量不依赖 pad,需飞满时长以取到稳态速度(留足净空)
        if not drive((ux * s, up_ * s, ut * s, uy * s), args.dur, False):
            return 1
        hover(1.5)
        print(f"- 段:反向回中")
        if not drive((-ux * s, -up_ * s, -ut * s, -uy * s), args.dur, False):
            return 1
        hover(1.0)
        safe_land("正常降落")
    except KeyboardInterrupt:
        safe_land("Ctrl-C"); ok = False
    except Exception as e:
        safe_land(f"exception {e}"); ok = False
    finally:
        fh.close()
        try:
            c.close()
        except Exception:
            pass
    print(f"{'完成' if ok else '异常降落'};数据 → {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
