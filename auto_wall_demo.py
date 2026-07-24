#!/usr/bin/env python3
"""auto_wall_demo.py - conservative automated forward-until-wall demo for RoboMaster TT.

Behaviour: take off, descend to ~TARGET_H cm, then creep forward slowly while
watching the depth/nearness map. As soon as the front becomes too close it STOPS
(hovers), holds briefly, then lands. Forward + stop only (no yaw/roll into unseen
space) to keep the first real auto flight predictable.

Safeguards:
  * battery gate (>= --min-bat)
  * 5s countdown before takeoff (Ctrl+C to abort)
  * hard airborne time limit -> auto land no matter what (--max-airborne)
  * lost video / stale perception -> hover then land
  * any exception or SIGINT/SIGTERM -> hover + land in finally
  * --dry-run: run the full connect+stream+perception+decision loop WITHOUT
    taking off or sending any rc (safe pre-flight test with the real camera)

Usage:
  python auto_wall_demo.py --dry-run            # no motion, validate perception
  python auto_wall_demo.py                       # real flight
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Optional

import cv2

from tt_control.avoidance import AvoidanceController, AvoidParams
from tt_control.depth_backend import DepthAnythingBackend
from tt_control.tello_client import TelloClient
from tt_control.video_stream import VideoStream

try:
    from tt_control.config import detect_local_ip
except Exception:  # pragma: no cover
    detect_local_ip = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("auto_wall")

_ABORT = False


def _handle_sig(signum, _frame):
    global _ABORT
    _ABORT = True
    log.warning("signal %s received -> aborting, will hover+land", signum)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="TT auto forward-until-wall demo")
    p.add_argument("--local-ip", default="", help="local wifi ip (default autodetect / 192.168.10.2)")
    p.add_argument("--service", default="http://127.0.0.1:8899/depth", help="depth service url")
    p.add_argument("--cruise", type=int, default=12, help="forward stick (slow)")
    p.add_argument("--target-h", type=int, default=50, help="target height cm after takeoff")
    p.add_argument("--stop-thresh", type=float, default=0.55, help="danger >= this -> STOP")
    p.add_argument("--max-airborne", type=float, default=15.0, help="hard airborne limit sec")
    p.add_argument("--min-bat", type=int, default=25, help="minimum battery percent")
    p.add_argument("--dry-run", action="store_true", help="no takeoff / no rc, just perceive+decide")
    p.add_argument("--save-prefix", default="/tmp/awd", help="prefix for saved annotated frames")
    return p.parse_args(argv)


def resolve_local_ip(cli_ip: str) -> str:
    if cli_ip:
        return cli_ip
    if detect_local_ip is not None:
        try:
            ip = detect_local_ip()
            if ip:
                return ip
        except Exception:
            pass
    return "192.168.10.2"


def wait_first_frame(video: VideoStream, timeout: float = 8.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if video.read() is not None:
            return True
        time.sleep(0.2)
    return False


def descend_to(client: TelloClient, target_h: int) -> None:
    time.sleep(4.0)  # let takeoff stabilise + state arrive
    h = client.height_cm()
    log.info("post-takeoff height=%s cm (target %d)", h, target_h)
    if h is None:
        log.warning("height unknown, skip descent (accepting default takeoff height)")
        return
    delta = h - target_h
    if delta >= 20:
        delta = min(500, delta)
        log.info("descending down %d cm -> ~%d cm", delta, target_h)
        client.send(f"down {delta}", timeout=15.0)
        time.sleep(2.0)
    else:
        log.info("already near target (delta %d < 20cm min move), keeping height", delta)


def main(argv=None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    local_ip = resolve_local_ip(args.local_ip)
    log.info("mode=%s local_ip=%s service=%s cruise=%d stop=%.2f target_h=%d max_air=%.0fs",
             "DRY-RUN" if args.dry_run else "FLIGHT", local_ip, args.service,
             args.cruise, args.stop_thresh, args.target_h, args.max_airborne)

    client = TelloClient(local_ip=local_ip)
    video = VideoStream(local_ip=local_ip)
    controller = AvoidanceController(AvoidParams(cruise_speed=args.cruise))
    backend = DepthAnythingBackend(service_url=args.service, controller=controller, overlay=True)

    airborne = False
    try:
        client.start_state_listener()
        if not client.connect():
            log.error("cannot connect to drone (no 'ok' to 'command'). Abort.")
            return 2
        log.info("drone connected")
        if not client.stream_on():
            log.error("streamon failed. Abort.")
            return 2
        video.start()
        if not wait_first_frame(video):
            log.error("no video frame received. Abort.")
            return 2
        log.info("video up, fps~%.1f", video.fps)

        bat = client.battery()
        log.info("battery=%s%%", bat)
        if bat is not None and bat < args.min_bat:
            log.error("battery %d%% < min %d%%. Abort.", bat, args.min_bat)
            return 2

        # perception warmup + decision preview on ground
        for i in range(10):
            frame = video.read()
            if frame is None:
                time.sleep(0.1)
                continue
            annotated = backend.infer(frame)
            df = backend.latest_depth()
            if df is not None:
                l, m, r = controller.zone_nearness(df.nearness)
                danger = max(l, m, r)
                act = "STOP" if danger >= args.stop_thresh else "FWD"
                log.info("[preview %d] L%.2f M%.2f R%.2f danger=%.2f -> %s", i, l, m, r, danger, act)
                if i == 5:
                    path = f"{args.save_prefix}_preview.png"
                    cv2.imwrite(path, annotated)
                    log.info("saved preview overlay -> %s", path)
            time.sleep(0.2)

        if args.dry_run:
            log.info("DRY-RUN: no takeoff. Running perception/decision loop for %.0fs...", args.max_airborne)
            t0 = time.time()
            n = 0
            while not _ABORT and (time.time() - t0) < args.max_airborne:
                frame = video.read()
                if frame is None:
                    time.sleep(0.05)
                    continue
                annotated = backend.infer(frame)
                df = backend.latest_depth()
                if df is None:
                    continue
                l, m, r = controller.zone_nearness(df.nearness)
                danger = max(l, m, r)
                act = "STOP" if danger >= args.stop_thresh else "FWD (would send rc 0 %d 0 0)" % args.cruise
                if n % 10 == 0:
                    log.info("L%.2f M%.2f R%.2f danger=%.2f -> %s", l, m, r, danger, act)
                    cv2.imwrite(f"{args.save_prefix}_dry.png", annotated)
                n += 1
                time.sleep(0.05)
            log.info("DRY-RUN done.")
            return 0

        # ---------------- REAL FLIGHT ----------------
        log.info("Battery ok. Takeoff in 5s. Ctrl+C to abort NOW.")
        for c in range(5, 0, -1):
            if _ABORT:
                log.warning("aborted during countdown, no takeoff.")
                return 0
            log.info("  %d ...", c)
            time.sleep(1.0)

        r = client.takeoff()
        log.info("takeoff -> %s", r)
        if r != "ok":
            log.error("takeoff NOT acknowledged (%s) -> abort, no forward flight", r)
            airborne = True  # assume possibly airborne; finally will land it
            return 2
        airborne = True
        descend_to(client, args.target_h)

        log.info("entering forward-until-wall loop (max %.0fs)", args.max_airborne)
        t0 = time.time()
        no_frame = 0
        stale = 0
        stop_hold = 0
        fi = 0
        while not _ABORT and (time.time() - t0) < args.max_airborne:
            frame = video.read()
            if frame is None:
                no_frame += 1
                client.rc(0, 0, 0, 0)
                if no_frame > 20:
                    log.error("video lost -> land")
                    break
                time.sleep(0.05)
                continue
            no_frame = 0
            try:
                annotated = backend.infer(frame)
                if fi % 4 == 0:
                    cv2.imwrite("%s_live%02d.png" % (args.save_prefix, (fi // 4) % 20), annotated)
                fi += 1
                df = backend.latest_depth()
                if df is None or (time.time() - df.ts) > 1.5:
                    raise RuntimeError("stale depth")
            except Exception as e:
                stale += 1
                client.rc(0, 0, 0, 0)
                log.warning("perception issue (%s) -> hover", e)
                if stale > 15:
                    log.error("perception stale -> land")
                    break
                time.sleep(0.05)
                continue
            stale = 0
            l, m, r = controller.zone_nearness(df.nearness)
            danger = max(l, m, r)
            h = client.height_cm()
            if danger >= args.stop_thresh:
                client.rc(0, 0, 0, 0)
                stop_hold += 1
                log.info("WALL L%.2f M%.2f R%.2f danger=%.2f h=%s -> STOP(%d)", l, m, r, danger, h, stop_hold)
                if stop_hold >= 20:
                    log.info("held stop ~1s -> mission complete, land")
                    break
            else:
                client.rc(0, args.cruise, 0, 0)
                stop_hold = 0
                log.info("FWD  L%.2f M%.2f R%.2f danger=%.2f h=%s -> pitch %d", l, m, r, danger, h, args.cruise)
            time.sleep(0.05)
        log.info("loop end (abort=%s, elapsed=%.1fs)", _ABORT, time.time() - t0)
        return 0

    finally:
        try:
            client.rc(0, 0, 0, 0)
        except Exception:
            pass
        if airborne:
            log.info("landing...")
            try:
                client.land()
            except Exception as e:
                log.error("land error: %s -> EMERGENCY", e)
                try:
                    client.emergency()
                except Exception:
                    pass
        try:
            video.stop()
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass
        log.info("cleanup done.")


if __name__ == "__main__":
    sys.exit(main())
