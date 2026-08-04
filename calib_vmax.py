#!/usr/bin/env python3
"""多卡长直线 VMAX 标定分析。

背景:单卡短距 + ~1.5s 延迟 + 粗糙垫子位置 → VMAX 测不准。解法:沿直线铺多张**同向**
Mission Pad(间距 ≤0.5m 保证连续锁定),飞一条长直线。位移够大(1~2m)时信噪比高,
VMAX = 稳态速度 / (杆量/100) 就收敛。

本脚本:
  1. 读 calibrate_flight 产出的 CSV(逐帧 pad_id / pos_x_cm / pos_y_cm / act_pitch / t);
  2. **多卡拼接**成统一全局系(同向,仅平移;换卡时用"位置连续性"反解偏移,
     与 mujoco_twin.stitch 同法);
  3. 取前进段外程的**中段(20%~80% 峰值位移)线性拟合**速度(避开加速/延迟起始与掉头),
     → 估计 VMAX。

用法:
  python calib_vmax.py logs/calib/calib_fwd_XXXX.csv --stick 20
  python calib_vmax.py --selftest        # 合成双卡数据自测(不需飞机)
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Optional, Tuple


def _f(r: dict, k: str) -> float:
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return 0.0


def stitch(rows: List[dict]) -> List[Tuple[float, Optional[float], Optional[float], float]]:
    """把逐帧(pad_id, pos_x_cm, pos_y_cm)拼成全局系(米)。

    返回 [(t_s, gx, gy, act_pitch)];丢垫帧 gx/gy=None。
    假设各卡同向(火箭一致),仅平移:第一张卡定原点;换到新卡时用"飞机位置不瞬移"
    的连续性,以换卡瞬间的上一全局位置反解该卡偏移量,一次标定复用。
    """
    offsets: dict[int, tuple[float, float]] = {}
    last_g: Optional[tuple[float, float]] = None
    out = []
    for r in rows:
        t = _f(r, "t_mono_ms") / 1000.0
        pitch = _f(r, "act_pitch")
        mid = int(_f(r, "pad_id"))
        lx, ly = _f(r, "pos_x_cm") / 100.0, _f(r, "pos_y_cm") / 100.0
        if mid < 0:
            out.append((t, None, None, pitch))
            continue
        if mid not in offsets:
            if not offsets:
                offsets[mid] = (0.0, 0.0)          # 第一张卡 = 全局原点
            else:
                bx, by = last_g or (0.0, 0.0)
                offsets[mid] = (bx - lx, by - ly)   # 新卡:位置连续性反解
        ox, oy = offsets[mid]
        gx, gy = lx + ox, ly + oy
        last_g = (gx, gy)
        out.append((t, gx, gy, pitch))
    return out


def estimate_vmax(rows: List[dict], stick: int) -> dict:
    g = stitch(rows)
    locked = [(t, gx, gy) for (t, gx, gy, _) in g if gx is not None]
    if len(locked) < 5:
        return {"error": f"锁定帧太少({len(locked)}),需要连续多卡锁定的长直线"}
    t0, x0, y0 = locked[0]
    import math
    disp = [(t - t0, math.hypot(gx - x0, gy - y0)) for (t, gx, gy) in locked]
    dmax = max(d for _, d in disp)
    t_at_peak = next(t for t, d in disp if d >= dmax - 1e-9)
    # 外程(到峰值前)中段 20%~80% dmax 线性拟合
    lo, hi = 0.2 * dmax, 0.8 * dmax
    seg = [(t, d) for t, d in disp if t <= t_at_peak and lo <= d <= hi]
    if len(seg) < 3:
        return {"error": f"外程中段样本太少({len(seg)});位移 {dmax*100:.0f}cm 可能太短,多铺几张卡飞更远"}
    n = len(seg)
    st = sum(t for t, _ in seg); sd = sum(d for _, d in seg)
    stt = sum(t * t for t, _ in seg); std = sum(t * d for t, d in seg)
    denom = n * stt - st * st
    v = (n * std - st * sd) / denom if abs(denom) > 1e-9 else 0.0   # m/s
    frac = stick / 100.0
    return {
        "locked_frames": len(locked),
        "peak_disp_m": round(dmax, 3),
        "fit_samples": n,
        "steady_v_mps": round(v, 3),
        "stick_frac": frac,
        "VMAX_mps": round(v / frac, 3) if frac else None,
        "pads_seen": len(set(int(_f(r, "pad_id")) for r in rows if _f(r, "pad_id") >= 0)),
    }


def _selftest() -> int:
    """合成双卡直线:真值速度 0.70 m/s,卡2 在全局 +0.50m。检验拼接与 VMAX 回收。"""
    dt, v_true = 0.05, 0.70
    rows = []
    t = 0.0
    xg = 0.0
    card2_x = 0.50
    while xg <= 1.0:
        # 选最近的卡;卡1原点=全局0,卡2原点=全局0.5
        if xg < 0.30:
            mid, lx = 1, xg
        else:
            mid, lx = 2, xg - card2_x
        rows.append({"t_mono_ms": f"{t*1000:.0f}", "pad_id": str(mid),
                     "pos_x_cm": f"{lx*100:.1f}", "pos_y_cm": "0", "act_pitch": "20"})
        xg += v_true * dt
        t += dt
    r = estimate_vmax(rows, stick=20)
    print("自测(真值 v=0.70, 双卡):", r)
    ok = bool(r.get("steady_v_mps")) and abs(r["steady_v_mps"] - 0.70) < 0.06
    print("拼接+测速:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="多卡长直线 VMAX 标定")
    ap.add_argument("csv", nargs="?", help="calibrate_flight 产出的 CSV")
    ap.add_argument("--stick", type=int, default=20)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.csv:
        ap.error("给出 CSV 路径,或用 --selftest")
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    r = estimate_vmax(rows, args.stick)
    if "error" in r:
        print("无法估计:", r["error"])
        return 1
    print(f"用卡 {r['pads_seen']} 张 | 锁定帧 {r['locked_frames']} | 峰值位移 {r['peak_disp_m']}m")
    print(f"外程中段拟合稳态速度 = {r['steady_v_mps']} m/s (杆量 {r['stick_frac']:.2f})")
    print(f"→ 实测 VMAX ≈ {r['VMAX_mps']} m/s   (当前 SimDrone VMAX=0.6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
