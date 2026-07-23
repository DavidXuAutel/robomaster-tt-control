#!/usr/bin/env python3
"""避障预览 / 接管飞行。

默认(--engage 不给):**只悬停观测不规避** —— 起飞悬停,打印左/中/右近度 + 控制律
此刻的决策(不下发避障杆量),存热力图叠图。真飞前的安全检查。

--engage:悬停 2s 后**由避障控制律接管**下发杆量,飞 --engage-secs 秒再降落。
内置:电量门 / 感知失联看门狗(depth 陈旧>1.5s 悬停停) / 硬超时 / 异常或 Ctrl-C 即降。
避障参数可 CLI 调保守(近人时 --approach-pitch 调小 = 原地转、几乎不前冲)。

用法:
  python avoid_preview.py                         # 只观测
  python avoid_preview.py --engage --engage-secs 6 --cruise 14 --approach-pitch 5 --yaw 30
"""
from __future__ import annotations

import argparse
import time
import pathlib

import cv2

from tt_control.config import detect_local_ip
from tt_control.tello_client import TelloClient
from tt_control.video_stream import VideoStream
from tt_control.depth_backend import DepthAnythingBackend
from tt_control.avoidance import AvoidanceController, AvoidParams
from tt_control.auto_safety import AutoWatchdog
from tt_control.episode_recorder import EpisodeRecorder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=3.0, help="悬停观测秒数")
    ap.add_argument("--service", default="http://127.0.0.1:8899/depth")
    ap.add_argument("--engage", action="store_true", help="悬停后由避障接管下发杆量")
    ap.add_argument("--engage-secs", type=float, default=6.0, help="避障接管时长")
    ap.add_argument("--record", action="store_true", help="接管期间录制 episode(视频+动作+状态)")
    ap.add_argument("--cruise", type=int, default=14, help="通畅前进杆量(保守)")
    ap.add_argument("--approach-pitch", type=int, default=5, help="接近区前进量(小=原地转)")
    ap.add_argument("--yaw", type=int, default=30, help="转向杆量")
    args = ap.parse_args()

    ip = detect_local_ip()
    if not ip.startswith("192.168.10."):
        raise SystemExit(f"未连飞机网段(detect_local_ip={ip!r})")
    print("WiFi:", ip)
    c = TelloClient(local_ip=ip, tello_ip="192.168.10.1")
    if not c.connect():
        raise SystemExit("connect 失败")
    c.start_state_listener()
    c.stream_on()
    vid = VideoStream(ip, 11111)
    vid.start()
    depth = DepthAnythingBackend(service_url=args.service, overlay=True)
    ctrl = AvoidanceController(AvoidParams(
        cruise_speed=args.cruise, approach_pitch=args.approach_pitch, yaw_speed=args.yaw))
    wd = AutoWatchdog()
    outdir = pathlib.Path.home() / "Projects/robomaster-tt-control/logs/preview"
    outdir.mkdir(parents=True, exist_ok=True)

    time.sleep(1.0)
    try:
        bat = int(float(c.state.get("bat", "0") or 0))
    except ValueError:
        bat = 0
    print("电量:", bat, "%")
    if bat < 30:
        raise SystemExit(f"电量 {bat}% 太低")

    def height():
        try:
            return int(float(c.state.get("h", "0") or 0))
        except (TypeError, ValueError):
            return 0

    def safe_land(reason=""):
        print("LAND", reason)
        for _ in range(3):
            try:
                c.rc(0, 0, 0, 0)
                c.send("land", wait_response=False)
            except Exception:
                pass
            t = time.time()
            while time.time() - t < 5:
                if height() <= 15:
                    return
                time.sleep(0.2)

    deadline = time.time() + 40.0  # 全程硬超时兜底
    try:
        t = time.time()
        while time.time() - t < 6:
            if vid.read() is not None:
                break
            time.sleep(0.2)
        if vid.read() is None:
            raise SystemExit("图传无画面")

        # 起飞(非阻塞 + 高度确认)
        c.rc(0, 0, 0, 0)
        c.send("takeoff", wait_response=False)
        t = time.time()
        while time.time() - t < 10 and height() <= 30:
            time.sleep(0.2)
        if height() <= 30:
            safe_land("未离地")
            return 1
        print("已离地", height(), "cm;稳定 1.5s")
        time.sleep(1.5)

        # 悬停观测(不发避障杆量)
        print("--- 悬停观测 %.0fs(仅记录决策)---" % args.secs)
        saved = 0
        t = time.time()
        while time.time() - t < args.secs:
            c.rc(0, 0, 0, 0)
            frame = vid.read()
            if frame is not None:
                overlaid = depth.infer(frame.copy())
                d = depth.latest_depth()
                if d is not None:
                    l, m, r = ctrl.zone_nearness(d.nearness)
                    dec = ctrl.decide(d.nearness)
                    print("  obs L%.2f M%.2f R%.2f -> %s rc%s" % (l, m, r, dec.state, dec.axes.as_tuple()))
                    if saved < 2:
                        cv2.imwrite(str(outdir / f"overlay_{saved}.jpg"), overlaid)
                        saved += 1
            time.sleep(0.25)

        # 避障接管
        rec = None
        if args.engage:
            print("--- 避障接管 %.0fs(下发杆量;背离右侧转左)---" % args.engage_secs)
            if args.record:
                rec = EpisodeRecorder(
                    pathlib.Path.home() / "Projects/robomaster-tt-control/logs/episodes",
                    meta_base={"env": "office avoidance test", "action_source": "avoidance",
                               "sim": False, "scale_anchor": "mission_pad",
                               "depth": {"model": "DepthAnythingV2-Small", "semantic": "nearness(relative)"}},
                    record_hz=10.0)
                print("  录制 ->", rec.dir)
            ctrl.reset()
            engaged_since = time.time()
            clear_since = None      # 前方持续通畅起始时刻
            avoided = False         # 是否发生过避障(TURN/BLOCKED);只有避过障再通畅才降落
            MIN_FLY = 3.0           # 至少飞 3s 再允许"通畅降落"
            CLEAR_HOLD = 1.0        # 越障后前方通畅持续多久判定"已越过"
            t = time.time()
            while time.time() - t < args.engage_secs:
                if time.time() > deadline:
                    print("硬超时"); break
                frame = vid.read()
                raw = frame.copy() if frame is not None else None
                if frame is not None:
                    depth.infer(frame)  # 叠图并刷新 latest_depth
                d = depth.latest_depth()
                now = time.time()
                reason = wd.check(now, engaged_since, d.ts if d is not None else None)
                if reason:
                    print("  看门狗:", reason, "-> 悬停解除")
                    c.rc(0, 0, 0, 0)
                    break
                if d is not None:
                    dec = ctrl.decide(d.nearness)
                    a, b, cc, dd = dec.axes.as_tuple()
                    c.rc(a, b, cc, dd)
                    l, mm, rr = dec.zones
                    danger = max(l, mm, rr)
                    print("  ENGAGE %s rc%s  L%.2f M%.2f R%.2f  danger%.2f" % (dec.state, dec.axes.as_tuple(), l, mm, rr, danger))
                    if rec is not None and raw is not None:
                        rec.capture(t_mono=now, rgb=raw, depth=d, depth_rtt_ms=depth.infer_ms,
                                    state=c.state, act=dec.axes, ctrl_state=dec.state, zones=dec.zones)
                    # 只有"避过障(TURN/BLOCKED)之后再通畅"才判定越过 → 降落
                    if dec.state in ("TURN_L", "TURN_R", "BLOCKED"):
                        avoided = True
                        clear_since = None      # 正在避障,重置通畅计时
                    elif avoided and danger <= ctrl.p.clear_thresh:
                        if clear_since is None:
                            clear_since = now
                        elif (now - engaged_since) > MIN_FLY and (now - clear_since) >= CLEAR_HOLD:
                            print("  越障后前方持续通畅 %.1fs -> 降落" % CLEAR_HOLD)
                            c.rc(0, 0, 0, 0)
                            break
                    else:
                        clear_since = None
                time.sleep(1.0 / 12.0)
            c.rc(0, 0, 0, 0)
            if rec is not None:
                rec.set_outcome("completed")
                print("  episode ->", rec.close())

        safe_land("结束")
    except SystemExit:
        raise
    except KeyboardInterrupt:
        safe_land("Ctrl-C")
    except Exception as e:
        safe_land(f"异常 {e}")
    finally:
        vid.stop()
        time.sleep(1)
        try:
            c.close()
        except Exception:
            pass
    print("完成,叠图存于", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
