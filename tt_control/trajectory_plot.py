"""把 MujocoPadTwin 导出的轨迹 CSV 画成一条实线(交付给大众的"匹配验证"证据)。

用法:
    python -m tt_control.trajectory_plot <traj.csv> [out.png]
"""

from __future__ import annotations

import csv
import pathlib
import sys
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # 无头
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# 配色(与项目 HTML 报告一致)
C_LOCK = "#1f9d63"   # 有垫子锁定段
C_COAST = "#c07a12"  # 丢垫滑行段
C_START = "#2f6fed"
C_END = "#d4483b"
C_PAD = "#d4483b"


def _read_csv(csv_path: pathlib.Path) -> List[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_trajectory(csv_path, out_png=None) -> pathlib.Path:
    csv_path = pathlib.Path(csv_path)
    rows = _read_csv(csv_path)
    if len(rows) < 2:
        raise ValueError(f"轨迹点太少({len(rows)}),无法绘制: {csv_path}")

    t = np.array([float(r["t"]) for r in rows])
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    z = np.array([float(r["z"]) for r in rows])
    locked = np.array([str(r.get("pad_locked", "True")).lower() == "true" for r in rows])

    out_png = pathlib.Path(out_png) if out_png else csv_path.with_suffix(".png")

    fig, (axxy, axz) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # --- 俯视 XY:整条轨迹一条实线,按锁定/滑行分色 ---
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_locked = locked[:-1] & locked[1:]
    colors = [C_LOCK if s else C_COAST for s in seg_locked]
    lc = LineCollection(segs, colors=colors, linewidths=2.4)
    axxy.add_collection(lc)

    axxy.plot(x[0], y[0], "o", color=C_START, ms=11, label="Start", zorder=5)
    axxy.plot(x[-1], y[-1], "s", color=C_END, ms=11, label="End", zorder=5)
    axxy.plot(0, 0, "*", color=C_PAD, ms=16, label="Mission Pad origin", zorder=4)

    axxy.set_aspect("equal", adjustable="datalim")
    axxy.margins(0.12)
    axxy.set_xlabel("X (m, pad frame)")
    axxy.set_ylabel("Y (m, pad frame)")
    axxy.set_title("Flight trajectory (top-down, solid line)")
    axxy.grid(True, ls=":", alpha=0.5)
    axxy.legend(loc="best", fontsize=9, framealpha=0.9)

    # --- 高度-时间 ---
    axz.plot(t, z, "-", color=C_START, lw=2.2)
    axz.set_xlabel("Time (s)")
    axz.set_ylabel("Altitude Z (m)")
    axz.set_title("Altitude vs Time")
    axz.grid(True, ls=":", alpha=0.5)
    axz.margins(x=0.05)

    lock_pct = 100.0 * float(locked.sum()) / len(locked)
    dur = float(t[-1] - t[0])
    fig.suptitle(
        f"RoboMaster TT trajectory  |  points {len(rows)}  |  {dur:.1f}s  |  pad-lock {lock_pct:.0f}%  |  {csv_path.name}",
        fontsize=12, y=0.99,
    )
    # 中文字体兜底(服务器无中文字体时不至于报错,仅显示方块)
    try:
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    csv_path = argv[0]
    out = argv[1] if len(argv) > 1 else None
    p = plot_trajectory(csv_path, out)
    print(f"saved {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
