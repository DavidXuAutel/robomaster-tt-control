#!/usr/bin/env python3
"""真机定点航线 + 遥测/轨迹记录(在服务器上运行,WiFi 连 Tello)。

航线(与仿真一致):
    起飞 → 降到 ~50cm → 前进 100cm → 后退 100cm(回原点) → 降落

用真机 SDK **位移指令**(forward/back/up/down)——飞控自带视觉里程闭环,比 rc
开环更稳更准更安全。全程记录:
  - 指令/响应 + 遥测 CSV(高度/姿态/速度/电量/Mission Pad 位姿)
  - MuJoCo 数字孪生轨迹(有垫锁定时为垫局部系),复用 trajectory_plot 出 PNG

安全:起飞前查电量;任何位移指令非 ok 或异常 → 立即降落;TelloClient 保活防中途关机。

用法:
    # 在线仿真自检(不连飞机,用 SimDrone 跑通同一套逻辑):
    .venv/bin/python fly_real_mission.py --sim
    # 真机(先确认 WiFi 已连到 TELLO-xxxx):
    .venv/bin/python fly_real_mission.py
"""

from __future__ import annotations

import argparse
import csv as csvmod
import logging
import pathlib
import threading
import time
from datetime import datetime
from typing import Optional

from tt_control.config import detect_local_ip

logger = logging.getLogger("fly_real")


class MissionAbort(Exception):
    """位移指令未返回 ok:主动中止并降落。"""


class Telemetry:
    """后台以固定频率采样 drone.state,带阶段标签,供事后核对/绘图。"""

    def __init__(self, get_state, hz: float = 10.0) -> None:
        self._get_state = get_state
        self._dt = 1.0 / hz
        self._rows: list[dict] = []
        self._phase = "init"
        self._t0 = time.time()
        self._run = False
        self._th: Optional[threading.Thread] = None

    @property
    def phase(self) -> str:
        return self._phase

    @phase.setter
    def phase(self, v: str) -> None:
        self._phase = v

    def start(self) -> None:
        self._run = True
        self._t0 = time.time()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self) -> None:
        while self._run:
            st = self._get_state() or {}
            row = {"t": round(time.time() - self._t0, 3), "phase": self._phase}
            row.update(st)
            self._rows.append(row)
            time.sleep(self._dt)

    def stop(self) -> None:
        self._run = False
        if self._th and self._th.is_alive():
            self._th.join(timeout=1.0)

    def save(self, path: pathlib.Path) -> Optional[pathlib.Path]:
        if not self._rows:
            return None
        keys = ["t", "phase"]
        seen = set(keys)
        for r in self._rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csvmod.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self._rows:
                w.writerow(r)
        return path

    def build_plot_csv(self, path: pathlib.Path) -> Optional[pathlib.Path]:
        """把遥测转成 trajectory_plot 所需的 schema(丢垫时 xy 沿用上次、z 用 h)。"""
        if len(self._rows) < 2:
            return None
        last_xy = (0.0, 0.0)
        out = []
        for r in self._rows:
            try:
                mid = int(float(r.get("mid", "-1")))
            except (TypeError, ValueError):
                mid = -1
            locked = mid >= 0
            if locked:
                try:
                    last_xy = (float(r.get("x", 0)) / 100.0, float(r.get("y", 0)) / 100.0)
                except (TypeError, ValueError):
                    pass
            try:
                z = float(r.get("h", r.get("z", 0))) / 100.0
            except (TypeError, ValueError):
                z = 0.0
            try:
                yaw = float(r.get("yaw", 0))
            except (TypeError, ValueError):
                yaw = 0.0
            out.append((r["t"], mid, last_xy[0], last_xy[1], max(z, 0.0), yaw, locked))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csvmod.writer(f)
            w.writerow(["t", "mid", "x", "y", "z", "yaw", "pitch", "roll",
                        "vgx", "vgy", "vgz", "h", "bat", "pad_locked"])
            for t, mid, x, y, z, yaw, locked in out:
                w.writerow([t, mid, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{yaw:.0f}",
                            0, 0, 0, 0, 0, f"{z * 100:.0f}", 0, locked])
        return path


def _connect(sim: bool, ip_timeout: float):
    """返回一个鸭子类型兼容的 client(SimDrone 或 TelloClient)。"""
    if sim:
        from tt_control.sim_drone import SimDrone
        c = SimDrone()
        if not c.connect():
            raise RuntimeError("SimDrone.connect 失败")
        logger.info("使用 SimDrone(离线自检)")
        return c

    from tt_control.tello_client import TelloClient
    logger.info("等待本机进入 Tello 网段(192.168.10.x)...")
    ip = ""
    deadline = time.time() + ip_timeout
    while time.time() < deadline:
        ip = detect_local_ip()
        if ip.startswith("192.168.10."):
            break
        time.sleep(1.0)
    if not ip.startswith("192.168.10."):
        raise RuntimeError(f"未连到 Tello 网段(当前 {ip!r})。请确认 WiFi 已连 TELLO 热点")
    logger.info("本机 IP=%s,连接飞机 ...", ip)
    c = TelloClient(ip)
    if not c.connect():
        raise RuntimeError("飞机 SDK 握手失败(command 无 ok)。检查电量/是否被占用后重试")
    logger.info("SDK 握手成功")
    return c


def _height(client) -> int:
    try:
        return int(float(client.state.get("h", 0)))
    except (TypeError, ValueError):
        return 0


def _battery(client) -> int:
    try:
        return int(float(client.state.get("bat", 0)))
    except (TypeError, ValueError):
        return 0


def run(
    sim: bool = False,
    forward_cm: int = 100,
    height_cm: int = 50,
    min_bat: int = 25,
    settle: float = 2.0,
    do_plot: bool = True,
    ip_timeout: float = 60.0,
    shape: str = "line",
    right_cm: int = 0,
    speed_cms: int = 0,
) -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = pathlib.Path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs / f"real_mission_{stamp}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    client = _connect(sim, ip_timeout)
    client.start_state_listener()
    time.sleep(1.5)  # 等状态回传

    bat0 = _battery(client)
    logger.info("起飞前电量: %d%%", bat0)
    if not sim and bat0 < min_bat:
        client.close()
        raise RuntimeError(f"电量过低 {bat0}% < {min_bat}%,拒绝起飞。请换满电电池")

    # 位移速度(cm/s):调慢让每段停得更干净、减少 mission pad 进视野时的截断
    if speed_cms:
        try:
            r = client.send(f"speed {speed_cms}")
            logger.info("speed %d -> %s", speed_cms, r)
        except Exception as e:  # noqa: BLE001
            logger.warning("speed 设置异常(忽略): %s", e)

    # Mission Pad 检测(有垫时孪生可记垫局部系;best-effort)
    try:
        r = client.mission_pad_on(downward=True)
        logger.info("mission_pad_on -> %s", r)
    except Exception as e:  # noqa: BLE001
        logger.warning("mission_pad_on 异常(忽略): %s", e)

    # 数字孪生记录(headless)
    twin = None
    try:
        from tt_control.mujoco_twin import MujocoPadTwin
        # stitch_pads=True:沿途多张卡自动拼接成一条连续轨迹(3 张卡直线支线)
        twin = MujocoPadTwin(
            get_state=lambda: client.state, headless=True, stitch_pads=True)
        if not twin.start():
            logger.warning("孪生启动失败(继续,仅靠遥测): %s", twin.status)
            twin = None
    except Exception as e:  # noqa: BLE001
        logger.warning("孪生不可用(继续): %s", e)
        twin = None

    telem = Telemetry(lambda: client.state)
    telem.start()

    def must(cmd: str, timeout: float = 20.0, retries: int = 1) -> None:
        # 'error Not joystick' 是偶发的模式冲突:指令被拒、飞机未动,暂停后重发是安全的。
        for attempt in range(retries + 1):
            r = client.send(cmd, timeout=timeout)
            logger.info("cmd %-14s -> %s", cmd, r)
            if r == "ok":
                return
            if "Not joystick" in str(r) and attempt < retries:
                logger.warning("'%s' 被拒(%s),悬停 2s 后重试 ...", cmd, r)
                time.sleep(2.0)
                continue
            raise MissionAbort(f"{cmd} 返回 {r!r}")

    ok = True
    err = ""
    reached = {}
    try:
        telem.phase = "takeoff"
        logger.info("=== 起飞 ===")
        r = client.takeoff()
        logger.info("cmd takeoff       -> %s", r)
        if r != "ok":
            raise MissionAbort(f"takeoff 返回 {r!r}")
        time.sleep(settle)
        h_after = _height(client)
        reached["takeoff_h"] = h_after
        logger.info("起飞后高度 ~%dcm", h_after)

        telem.phase = "descend"
        delta = h_after - height_cm
        if delta >= 20:
            logger.info("=== 降到 %dcm(down %d) ===", height_cm, delta)
            must(f"down {delta}")
        elif delta <= -20:
            logger.info("=== 升到 %dcm(up %d) ===", height_cm, -delta)
            must(f"up {-delta}")
        else:
            logger.info("=== 已接近 %dcm(Δ%+dcm 小于最小步进 20,免调) ===", height_cm, -delta)
        time.sleep(settle)
        reached["cruise_h"] = _height(client)

        telem.phase = "forward"
        logger.info("=== 前进 %dcm ===", forward_cm)
        must(f"forward {forward_cm}")
        time.sleep(settle)

        if shape == "L":
            rc = right_cm or forward_cm
            telem.phase = "right"
            logger.info("=== 向右平移 %dcm(机头不转) ===", rc)
            must(f"right {rc}")
            time.sleep(settle)
        else:
            telem.phase = "back"
            logger.info("=== 后退 %dcm(回原点) ===", forward_cm)
            must(f"back {forward_cm}")
            time.sleep(settle)

        telem.phase = "hover"
        time.sleep(1.0)
    except MissionAbort as e:
        ok = False
        err = str(e)
        logger.error("任务中止 → 立即降落: %s", err)
    except Exception as e:  # noqa: BLE001
        ok = False
        err = repr(e)
        logger.exception("异常 → 立即降落")
    finally:
        telem.phase = "land"
        logger.info("=== 降落 ===")
        try:
            r = client.land()
            logger.info("cmd land          -> %s", r)
            if r != "ok" and not sim:
                logger.error("land 未确认,发送 emergency 停桨兜底")
                client.emergency()
        except Exception:  # noqa: BLE001
            logger.exception("降落异常,发送 emergency 停桨兜底")
            try:
                client.emergency()
            except Exception:  # noqa: BLE001
                pass
        time.sleep(3.0 if not sim else 0.3)
        telem.stop()
        if twin is not None:
            twin.stop()
        try:
            client.mission_pad_off()
        except Exception:  # noqa: BLE001
            pass
        bat1 = _battery(client)
        client.close()

    # ---- 落盘 + 出图 ----
    telem_csv = telem.save(logs / f"real_mission_{stamp}.telemetry.csv")
    twin_csv = getattr(twin, "_saved_path", None) if twin else None

    png = None
    if do_plot:
        try:
            from tt_control.trajectory_plot import plot_trajectory
            src = twin_csv
            if not src:
                # 无垫锁定 → 用遥测重建可绘制的 CSV(以高度为主)
                src = telem.build_plot_csv(logs / f"real_mission_{stamp}.plot.csv")
            if src:
                png = plot_trajectory(src)
        except Exception as e:  # noqa: BLE001
            logger.warning("出图失败(忽略): %s", e)

    summary = {
        "ok": ok,
        "error": err,
        "mission": "real" if not sim else "sim-dryrun",
        "bat_start": bat0,
        "bat_end": bat1,
        "reached_cm": reached,
        "telemetry_csv": str(telem_csv) if telem_csv else None,
        "twin_csv": str(twin_csv) if twin_csv else None,
        "png": str(png) if png else None,
        "telem_rows": len(telem._rows),
        "log": str(logs / f"real_mission_{stamp}.log"),
    }
    logger.info("=== 摘要 === %s", summary)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="真机定点航线 + 遥测/轨迹记录")
    p.add_argument("--sim", action="store_true", help="用 SimDrone 离线自检(不连飞机)")
    p.add_argument("--shape", choices=["line", "L"], default="line",
                   help="航线:line 直线出返 / L 前进后向右平移")
    p.add_argument("--forward-cm", type=int, default=100, help="前进距离(cm)")
    p.add_argument("--right-cm", type=int, default=0, help="[L] 向右平移距离(cm,默认=前进距离)")
    p.add_argument("--height-cm", type=int, default=50)
    p.add_argument("--speed-cms", type=int, default=0,
                   help="位移速度 cm/s(10-100,0=不设、用飞机默认)。调慢更稳")
    p.add_argument("--min-bat", type=int, default=25, help="起飞最低电量%%")
    p.add_argument("--settle", type=float, default=2.0, help="每段之间悬停秒数")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    s = run(
        sim=args.sim,
        forward_cm=args.forward_cm,
        height_cm=args.height_cm,
        min_bat=args.min_bat,
        settle=args.settle,
        do_plot=not args.no_plot,
        shape=args.shape,
        right_cm=args.right_cm,
        speed_cms=args.speed_cms,
    )
    print("\n=== 真机航线摘要 ===")
    for k, v in s.items():
        print(f"  {k}: {v}")
    return 0 if s["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
