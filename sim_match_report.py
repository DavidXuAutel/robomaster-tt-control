#!/usr/bin/env python3
"""仿真可信度(Tier 1)匹配报告：用真机记录的动作重放仿真运动学，量化仿真与真机轨迹的偏差。

为什么不是"忠实回放"：把真机位姿直接抄进 MuJoCo 只是镜像，误差恒为 0、毫无意义。
真正的可信度证据是——**给仿真喂真机当时下发的同一串动作(rc 杆量)，看仿真的运动学
能否重现真机走出的轨迹**。偏差越小，说明"仿真是真机的可信替身"，后续仿真增广/想象
rollout 才站得住脚(见 docs/design/2026-07-21 的 D3)。

输入：episode 目录(含 frames.csv)或直接给 frames.csv。frames.csv 由 episode_recorder
产出，含逐帧动作(act_*)与真机位姿(pos_*_cm / yaw_deg)。

用法：
    python sim_match_report.py logs/episodes/ep_20260721_153000
    python sim_match_report.py <episode>/frames.csv --out report.json

产物：<episode>/match_report.json + match_report.png(真机实线 vs 仿真虚线俯视图)。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
from typing import List, Optional

# 复用 SimDrone 的运动学常数，保证"仿真回放"与在线 SimDrone 同一套物理
from tt_control.sim_drone import VMAX, VZMAX, YAWRATE


def _read_frames(frames_csv: pathlib.Path) -> List[dict]:
    with frames_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fnum(row: dict, key: str, default: Optional[float] = None) -> Optional[float]:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _has_real_pose(row: dict) -> bool:
    """真机位姿仅在垫子锁定(pad_id>=0)且 x/y 有效时可比。"""
    pad = _fnum(row, "pad_id")
    return pad is not None and pad >= 0 and _fnum(row, "pos_x_cm") is not None


def replay_sim(rows: List[dict]) -> List[tuple]:
    """用逐帧动作(act_*)按 SimDrone 运动学积分，返回 [(t, x, y, z, yaw)]（米/度）。

    起点对齐到第一帧的真机位姿(若有)，使两条轨迹同起点，只比"之后怎么走"。
    """
    out: List[tuple] = []
    x = y = z = yaw = 0.0
    prev_t: Optional[float] = None
    for r in rows:
        t = (_fnum(r, "t_mono_ms", 0.0) or 0.0) / 1000.0
        a = _fnum(r, "act_roll", 0.0) or 0.0
        b = _fnum(r, "act_pitch", 0.0) or 0.0
        c = _fnum(r, "act_throttle", 0.0) or 0.0
        d = _fnum(r, "act_yaw", 0.0) or 0.0
        if prev_t is None:
            x = (_fnum(r, "pos_x_cm", 0.0) or 0.0) / 100.0
            y = (_fnum(r, "pos_y_cm", 0.0) or 0.0) / 100.0
            z = (_fnum(r, "pos_z_cm") or _fnum(r, "height_cm", 0.0) or 0.0) / 100.0
            yaw = _fnum(r, "yaw_deg", 0.0) or 0.0
            prev_t = t
            out.append((t, x, y, z, yaw))
            continue
        dt = max(0.0, t - prev_t)
        prev_t = t
        vx_b = (b / 100.0) * VMAX   # pitch -> 前后
        vy_b = (a / 100.0) * VMAX   # roll  -> 左右(右为正)
        yr = math.radians(yaw)
        x += (vx_b * math.cos(yr) - vy_b * math.sin(yr)) * dt
        y += (vx_b * math.sin(yr) + vy_b * math.cos(yr)) * dt
        z = max(0.0, z + (c / 100.0) * VZMAX * dt)
        yaw = (yaw + (d / 100.0) * YAWRATE * dt) % 360.0
        out.append((t, x, y, z, yaw))
    return out


def _yaw_err(sim_deg: float, real_deg: float) -> float:
    """就近环绕的航向误差，落在 [-180,180]。"""
    return ((sim_deg - real_deg + 180.0) % 360.0) - 180.0


def _path_len(xs: List[float], ys: List[float]) -> float:
    return sum(
        math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(xs))
    )


def compute_metrics(rows: List[dict], sim: List[tuple]) -> dict:
    rx, ry, rz, sx, sy, sz = [], [], [], [], [], []
    dyaw = []
    for r, s in zip(rows, sim):
        if not _has_real_pose(r):
            continue
        _, sxv, syv, szv, syaw = s
        rx.append((_fnum(r, "pos_x_cm", 0.0) or 0.0) / 100.0)
        ry.append((_fnum(r, "pos_y_cm", 0.0) or 0.0) / 100.0)
        rz.append((_fnum(r, "pos_z_cm") or _fnum(r, "height_cm", 0.0) or 0.0) / 100.0)
        sx.append(sxv)
        sy.append(syv)
        sz.append(szv)
        dyaw.append(_yaw_err(syaw, _fnum(r, "yaw_deg", 0.0) or 0.0))

    n = len(rx)
    if n < 2:
        return {"comparable_frames": n, "note": "真机位姿点不足(需 pad 锁定帧≥2)，无法量化匹配"}

    def _mae(a, b):
        return sum(abs(ai - bi) for ai, bi in zip(a, b)) / len(a)

    def _rmse(a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)) / len(a))

    endpoint = math.sqrt(
        (sx[-1] - rx[-1]) ** 2 + (sy[-1] - ry[-1]) ** 2 + (sz[-1] - rz[-1]) ** 2
    )
    real_len = _path_len(rx, ry)
    sim_len = _path_len(sx, sy)
    return {
        "comparable_frames": n,
        "mae_m": {"x": round(_mae(sx, rx), 4), "y": round(_mae(sy, ry), 4), "z": round(_mae(sz, rz), 4)},
        "rmse_m": {"x": round(_rmse(sx, rx), 4), "y": round(_rmse(sy, ry), 4), "z": round(_rmse(sz, rz), 4)},
        "yaw_mae_deg": round(sum(abs(v) for v in dyaw) / len(dyaw), 3),
        "endpoint_error_m": round(endpoint, 4),
        "path_len_real_m": round(real_len, 4),
        "path_len_sim_m": round(sim_len, 4),
        "path_len_ratio": round(sim_len / real_len, 4) if real_len > 1e-6 else None,
    }


def _plot(rows: List[dict], sim: List[tuple], out_png: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rx, ry, sx, sy = [], [], [], []
    for r, s in zip(rows, sim):
        if not _has_real_pose(r):
            continue
        rx.append((_fnum(r, "pos_x_cm", 0.0) or 0.0) / 100.0)
        ry.append((_fnum(r, "pos_y_cm", 0.0) or 0.0) / 100.0)
        sx.append(s[1])
        sy.append(s[2])
    if len(rx) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(rx, ry, "-", color="#1f9d63", lw=2.4, label="Real (recorded)")
    ax.plot(sx, sy, "--", color="#2f6fed", lw=2.2, label="Sim (action replay)")
    ax.plot(rx[0], ry[0], "o", color="#2f6fed", ms=10, label="Start")
    ax.plot(rx[-1], ry[-1], "s", color="#1f9d63", ms=10, label="Real end")
    ax.plot(sx[-1], sy[-1], "s", color="#d4483b", ms=10, label="Sim end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.12)
    ax.set_xlabel("X (m, pad frame)")
    ax.set_ylabel("Y (m, pad frame)")
    ax.set_title("Sim-vs-Real fidelity (action-driven replay)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def make_report(target: pathlib.Path, out_json: Optional[pathlib.Path] = None) -> dict:
    frames_csv = target / "frames.csv" if target.is_dir() else target
    if not frames_csv.is_file():
        raise FileNotFoundError(f"找不到 frames.csv: {frames_csv}")
    ep_dir = frames_csv.parent
    rows = _read_frames(frames_csv)
    if len(rows) < 2:
        raise ValueError(f"帧太少({len(rows)})，无法生成匹配报告")

    sim = replay_sim(rows)
    metrics = compute_metrics(rows, sim)
    out_json = out_json or (ep_dir / "match_report.json")
    out_png = ep_dir / "match_report.png"
    try:
        _plot(rows, sim, out_png)
        png_note = str(out_png) if out_png.is_file() else None
    except Exception as e:  # 无 matplotlib 等：报告仍出，只是没有图
        png_note = f"plot skipped: {e}"

    report = {
        "episode": ep_dir.name,
        "frames_total": len(rows),
        "method": "action-driven sim replay vs recorded real pose",
        "sim_kinematics": {"VMAX": VMAX, "VZMAX": VZMAX, "YAWRATE": YAWRATE},
        "metrics": metrics,
        "plot": png_note,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="仿真可信度(Tier 1)匹配报告")
    p.add_argument("target", help="episode 目录或 frames.csv 路径")
    p.add_argument("--out", default="", help="报告 JSON 输出路径(默认 <episode>/match_report.json)")
    args = p.parse_args(argv)

    report = make_report(
        pathlib.Path(args.target),
        pathlib.Path(args.out) if args.out else None,
    )
    m = report["metrics"]
    print(f"episode: {report['episode']}  frames: {report['frames_total']}")
    if "endpoint_error_m" in m:
        print(f"  可比帧: {m['comparable_frames']}")
        print(f"  逐轴 MAE(m): {m['mae_m']}   RMSE(m): {m['rmse_m']}")
        print(f"  航向 MAE: {m['yaw_mae_deg']} deg   终点误差: {m['endpoint_error_m']} m")
        print(f"  路径长 真={m['path_len_real_m']}m 仿真={m['path_len_sim_m']}m 比={m['path_len_ratio']}")
    else:
        print(f"  {m.get('note')}")
    print(f"  报告: {report.get('plot')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
