#!/usr/bin/env python3
"""定点航线仿真 + MuJoCo 数字孪生轨迹记录(交付给大众的"匹配验证"证据)。

不连飞机、不用 GPU、无需显示器。闭环走可复现的定点航线,两种预设:

  straight(默认): 起飞 → 降到 50cm → 前进 100cm → 后退 100cm(回原点) → 降落
  square         : 起飞 → 降到 50cm → 方形巡航(每边 100cm,四角各偏航 90°) → 降落

数据通路(下游零改动):
    SimDrone(运动学仿真) --state 20Hz--> MujocoPadTwin(headless 镜像+记录)
        --> logs/trajectories/traj_<stamp>.csv|json --> trajectory_plot --> .png

为什么闭环:SimDrone 是速度型运动学(rc 杆量≈机体速度),盯着 state 到位即
悬停,能精确命中距离/航向且轨迹平滑连续;比开环按时间走(ScriptedPolicy)可复现。

用法:
    python sim_mission.py                              # 直线出返
    python sim_mission.py --mission square --side-cm 100
    python sim_mission.py --forward-cm 150 --height-cm 60
    python sim_mission.py --no-plot                    # 只出 CSV/JSON,不画图
"""

from __future__ import annotations

import argparse
import csv as csvmod
import logging
import math
import pathlib
import threading
import time
from typing import Callable, Optional

from tt_control.mujoco_twin import MujocoPadTwin
from tt_control.sim_drone import SimDrone

logger = logging.getLogger("sim_mission")

# rc 轴索引:0=roll(左右) 1=pitch(前后) 2=throttle(升降) 3=yaw(偏航)
AX_PITCH = 1
AX_THROTTLE = 2

# 方形边长的安全上限:角点最远处距原点 side*√2 需 < 丢垫阈值(PAD_RANGE=1.5m)
# 才能全程保持垫锁定(绿实线);超过则转为丢垫滑行段(橙线),轨迹仍记录。
PAD_RANGE_CM = 150.0


def _num(state: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(state.get(key, default))
    except (TypeError, ValueError):
        return default


def _wrap180(deg: float) -> float:
    """把角度归一到 (-180, 180]。"""
    return (deg + 180.0) % 360.0 - 180.0


def _hover(drone: SimDrone, seconds: float = 0.6) -> None:
    """悬停一小段,让轨迹分段清晰、孪生多采样几点。"""
    drone.rc(0, 0, 0, 0)
    time.sleep(seconds)


def _fly_axis(
    drone: SimDrone,
    read_key: str,
    stick_index: int,
    target_cm: float,
    *,
    cruise: int = 40,
    kp: float = 2.0,
    tol_cm: float = 1.5,
    timeout: float = 12.0,
    dt: float = 0.03,
    label: str = "",
) -> float:
    """闭环把 state[read_key](cm)驱到 target_cm 附近(绝对坐标轴),到位悬停,返回实测值。

    比例控制 + 杆量封顶;速度型运动学下置零即停,超调仅约一个 dt 的位移。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur = _num(drone.state, read_key)
        err = target_cm - cur
        if abs(err) <= tol_cm:
            break
        mag = min(float(cruise), abs(err) * kp)
        stick = int(mag if err > 0 else -mag)
        rc = [0, 0, 0, 0]
        rc[stick_index] = stick
        drone.rc(*rc)
        time.sleep(dt)
    drone.rc(0, 0, 0, 0)
    cur = _num(drone.state, read_key)
    logger.info("  [%s] %s -> %.0fcm (目标 %.0f, 误差 %+.0f)",
                label, read_key, cur, target_cm, cur - target_cm)
    return cur


def _fly_distance(
    drone: SimDrone,
    dist_cm: float,
    *,
    forward: bool = True,
    cruise: int = 40,
    kp: float = 2.0,
    tol_cm: float = 2.0,
    timeout: float = 15.0,
    dt: float = 0.03,
    label: str = "",
) -> float:
    """沿当前机头方向闭环飞 dist_cm(与航向无关,用离本段起点的欧氏距离判定)。

    forward=True 用 +pitch(前进),False 用 -pitch(后退)。返回实际行进距离(cm)。
    """
    x0, y0 = _num(drone.state, "x"), _num(drone.state, "y")
    t0 = time.time()
    traveled = 0.0
    while time.time() - t0 < timeout:
        x, y = _num(drone.state, "x"), _num(drone.state, "y")
        traveled = math.hypot(x - x0, y - y0)
        rem = dist_cm - traveled
        if rem <= tol_cm:
            break
        mag = min(float(cruise), rem * kp)
        pitch = int(mag if forward else -mag)
        drone.rc(0, pitch, 0, 0)
        time.sleep(dt)
    drone.rc(0, 0, 0, 0)
    x, y = _num(drone.state, "x"), _num(drone.state, "y")
    traveled = math.hypot(x - x0, y - y0)
    logger.info("  [%s] 行进 %.0fcm (目标 %.0f, 误差 %+.0f) -> (%.0f,%.0f)",
                label, traveled, dist_cm, traveled - dist_cm, x, y)
    return traveled


def _turn_to(
    drone: SimDrone,
    target_deg: float,
    *,
    cruise: int = 40,
    kp: float = 1.5,
    tol_deg: float = 2.0,
    timeout: float = 12.0,
    dt: float = 0.03,
    label: str = "",
) -> float:
    """偏航闭环:原地转到 target_deg(就近方向,处理 0/360 环绕),返回实测航向。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur = _num(drone.state, "yaw")
        err = _wrap180(target_deg - cur)
        if abs(err) <= tol_deg:
            break
        mag = min(float(cruise), abs(err) * kp)
        yaw_stick = int(mag if err > 0 else -mag)  # +d 使航向角增大
        drone.rc(0, 0, 0, yaw_stick)
        time.sleep(dt)
    drone.rc(0, 0, 0, 0)
    cur = _num(drone.state, "yaw")
    logger.info("  [%s] 航向 -> %.0f° (目标 %.0f, 误差 %+.0f)",
                label, cur, target_deg, _wrap180(cur - target_deg))
    return cur


def _fly_and_record(
    build_plan: Callable[[SimDrone, int, dict], None],
    mission_name: str,
    *,
    height_cm: float,
    cruise: int,
    traj_dir: Optional[pathlib.Path],
    do_plot: bool,
) -> tuple[dict, dict]:
    """公共骨架:起飞→降到巡航高→执行中段航线→降落→记录/出图。

    build_plan(drone, cruise, marks) 负责中段各航段,并把关键量写进 marks(cm/deg)。
    返回 (base_summary, marks)。
    """
    drone = SimDrone()
    drone.connect()
    drone.start_state_listener()

    twin = MujocoPadTwin(
        get_state=lambda: drone.state,
        traj_dir=traj_dir,
        headless=True,
    )
    if not twin.start():
        drone.close()
        raise RuntimeError(f"MuJoCo 孪生启动失败: {twin.status}")

    # 地面真值采样:SimDrone 内部一直有真实 x/y(不受"丢垫"影响),
    # 用它画完整路径,避免孪生在 >1.5m 处因丢垫冻结 XY(孪生仍作数字孪生记录)。
    truth: list = []
    _t0 = time.time()
    _stop = threading.Event()

    def _sample() -> None:
        while not _stop.is_set():
            st = drone.state or {}
            def g(k: str) -> float:
                try:
                    return float(st.get(k, 0))
                except (TypeError, ValueError):
                    return 0.0
            truth.append((round(time.time() - _t0, 3),
                          g("x") / 100, g("y") / 100, g("h") / 100,
                          g("yaw"), int(g("mid")) >= 0))
            time.sleep(0.05)

    _th = threading.Thread(target=_sample, name="truth-sampler", daemon=True)
    _th.start()
    time.sleep(0.2)  # 让状态线程与孪生线程先产出

    marks: dict[str, float] = {}

    # 0. 起飞(sim 固定升到 ~100cm,贴合真机 Tello)
    logger.info("[%s] 起飞 ...", mission_name)
    drone.takeoff()
    _hover(drone, 0.6)
    marks["takeoff_z"] = _num(drone.state, "z")

    # 1. 降到巡航高度
    logger.info("[%s] 降到 %.0fcm ...", mission_name, height_cm)
    marks["cruise_z"] = _fly_axis(
        drone, "z", AX_THROTTLE, height_cm, cruise=cruise, label="降高")
    _hover(drone, 0.6)

    # 2. 中段航线
    build_plan(drone, cruise, marks)

    # 3. 降落
    logger.info("[%s] 降落 ...", mission_name)
    drone.land()
    _hover(drone, 0.4)

    _stop.set()
    _th.join(timeout=1.0)
    twin.stop()
    csv_path = twin._saved_path
    cmd_count = drone.cmd_count
    traj_count = twin.traj_count
    drone.close()

    # 写地面真值 CSV(trajectory_plot schema),用它出图 → 完整路径不丢
    out_dir = traj_dir or (pathlib.Path.cwd() / "logs" / "trajectories")
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_csv = None
    if len(truth) >= 2:
        stem = csv_path.stem if csv_path else f"truth_{mission_name}"
        truth_csv = out_dir / f"{stem}_truth.csv"
        with truth_csv.open("w", newline="", encoding="utf-8") as f:
            w = csvmod.writer(f)
            w.writerow(["t", "mid", "x", "y", "z", "yaw", "pitch", "roll",
                        "vgx", "vgy", "vgz", "h", "bat", "pad_locked"])
            for t, x, y, z, yaw, locked in truth:
                w.writerow([t, 1 if locked else -1, f"{x:.3f}", f"{y:.3f}",
                            f"{z:.3f}", f"{yaw:.0f}", 0, 0, 0, 0, 0,
                            f"{z * 100:.0f}", 0, locked])

    png_path = None
    if do_plot:
        from tt_control.trajectory_plot import plot_trajectory
        src = truth_csv or csv_path  # 优先地面真值(完整路径),否则孪生 CSV
        if src is not None:
            png_path = plot_trajectory(src)

    base = {
        "mission": mission_name,
        "traj_count": traj_count,
        "csv": str(csv_path) if csv_path else None,
        "truth_csv": str(truth_csv) if truth_csv else None,
        "png": str(png_path) if png_path else None,
        "cmd_count": cmd_count,
        "marks_cm": {k: round(v, 1) for k, v in marks.items()},
        "cruise_z_err_cm": round(marks.get("cruise_z", 0) - height_cm, 1),
    }
    return base, marks


def run_mission(
    forward_cm: float = 100.0,
    height_cm: float = 50.0,
    cruise: int = 40,
    traj_dir: Optional[pathlib.Path] = None,
    do_plot: bool = True,
) -> dict:
    """直线出返:前进 forward_cm 再后退回原点。返回摘要 dict。"""

    def plan(drone: SimDrone, cruise: int, marks: dict) -> None:
        logger.info("前进 %.0fcm ...", forward_cm)
        marks["forward_x"] = _fly_axis(
            drone, "x", AX_PITCH, forward_cm, cruise=cruise, label="前进")
        _hover(drone, 0.6)
        logger.info("后退 %.0fcm 回原点 ...", forward_cm)
        marks["return_x"] = _fly_axis(
            drone, "x", AX_PITCH, 0.0, cruise=cruise, label="后退")
        marks["return_y"] = _num(drone.state, "y")
        _hover(drone, 0.6)

    base, marks = _fly_and_record(
        plan, "straight",
        height_cm=height_cm, cruise=cruise, traj_dir=traj_dir, do_plot=do_plot)
    base["forward_err_cm"] = round(marks.get("forward_x", 0) - forward_cm, 1)
    base["return_err_cm"] = round(
        math.hypot(marks.get("return_x", 0), marks.get("return_y", 0)), 1)
    return base


def run_square_mission(
    side_cm: float = 100.0,
    height_cm: float = 50.0,
    cruise: int = 40,
    traj_dir: Optional[pathlib.Path] = None,
    do_plot: bool = True,
) -> dict:
    """方形巡航:四条边各 side_cm,每角原地偏航 90°,回到起点并复位航向。"""
    if side_cm * math.sqrt(2) >= PAD_RANGE_CM:
        logger.warning(
            "边长 %.0fcm 使角点距原点 %.0fcm ≥ 丢垫阈值 %.0fcm,将出现丢垫滑行段(橙线)",
            side_cm, side_cm * math.sqrt(2), PAD_RANGE_CM)

    def plan(drone: SimDrone, cruise: int, marks: dict) -> None:
        for i in range(4):
            logger.info("第 %d 条边:前进 %.0fcm ...", i + 1, side_cm)
            _fly_distance(drone, side_cm, forward=True, cruise=cruise,
                          label=f"边{i + 1}")
            _hover(drone, 0.6)
            marks[f"corner{i + 1}_x"] = _num(drone.state, "x")
            marks[f"corner{i + 1}_y"] = _num(drone.state, "y")
            heading = ((i + 1) * 90) % 360
            logger.info("第 %d 角:偏航到 %d° ...", i + 1, heading)
            _turn_to(drone, heading, cruise=cruise, label=f"转{i + 1}")
            _hover(drone, 0.6)
        marks["final_yaw"] = _num(drone.state, "yaw")

    base, marks = _fly_and_record(
        plan, "square",
        height_cm=height_cm, cruise=cruise, traj_dir=traj_dir, do_plot=do_plot)
    base["return_err_cm"] = round(
        math.hypot(marks.get("corner4_x", 0), marks.get("corner4_y", 0)), 1)
    base["heading_err_deg"] = round(_wrap180(marks.get("final_yaw", 0)), 1)
    return base


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="定点航线仿真 + MuJoCo 轨迹记录")
    p.add_argument("--mission", choices=["straight", "square"], default="straight",
                   help="航线预设:straight 直线出返 / square 方形巡航")
    p.add_argument("--forward-cm", type=float, default=100.0, help="[straight] 前进/后退距离(cm)")
    p.add_argument("--side-cm", type=float, default=100.0, help="[square] 方形边长(cm)")
    p.add_argument("--height-cm", type=float, default=50.0, help="巡航高度(cm)")
    p.add_argument("--cruise", type=int, default=40, help="巡航杆量上限(0..100)")
    p.add_argument("--out-dir", default="", help="轨迹输出目录(默认 logs/trajectories)")
    p.add_argument("--no-plot", action="store_true", help="不出 PNG,只存 CSV/JSON")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    traj_dir = pathlib.Path(args.out_dir) if args.out_dir else None
    if args.mission == "square":
        s = run_square_mission(
            side_cm=args.side_cm, height_cm=args.height_cm, cruise=args.cruise,
            traj_dir=traj_dir, do_plot=not args.no_plot)
    else:
        s = run_mission(
            forward_cm=args.forward_cm, height_cm=args.height_cm, cruise=args.cruise,
            traj_dir=traj_dir, do_plot=not args.no_plot)

    print(f"\n=== 仿真航线摘要 [{s['mission']}] ===")
    print(f"  轨迹点数     : {s['traj_count']}")
    print(f"  指令数       : {s['cmd_count']}")
    print(f"  关键点(cm)   : {s['marks_cm']}")
    if s["mission"] == "straight":
        print(f"  前进到位误差 : {s['forward_err_cm']:+.1f} cm")
    else:
        print(f"  航向复位误差 : {s['heading_err_deg']:+.1f} °")
    print(f"  回原点误差   : {s['return_err_cm']:.1f} cm")
    print(f"  巡航高误差   : {s['cruise_z_err_cm']:+.1f} cm")
    print(f"  CSV          : {s['csv']}")
    print(f"  JSON         : {s['csv'][:-4] + '.json' if s['csv'] else None}")
    print(f"  轨迹图 PNG   : {s['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
