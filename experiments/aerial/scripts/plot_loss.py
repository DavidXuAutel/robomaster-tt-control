#!/usr/bin/env python3
"""Plot the B0/B1 training loss curve straight from a trainer log.

The trainer emits one progress block per `log_every` steps. Depending on the
logger (plain vs Rich-console->file), the fields (step / loss / loss_action /
loss_video / lr) may share one physical line OR be wrapped across several, and
the "[train]" tag may be dropped in rendering. So we do NOT parse line-by-line:
we scan key=value tokens in file order and start a new record whenever a fresh
`step=` appears. This matches how eta_watch.sh reads the same logs.

Usage:
  python experiments/aerial/scripts/plot_loss.py <logfile-or-logdir> [--out curve.png] [--smooth 21]

If given a directory, the newest *.log inside it is used.
Falls back to writing a CSV next to the PNG so the data survives even if
matplotlib is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# loss_action / loss_video must precede the bare "loss" alternative so the
# longer token wins at a shared prefix; step swallows its trailing "/max".
_TOKEN = re.compile(
    r"\b(step|loss_action|loss_video|loss|lr)=([-+0-9.eE]+)(?:/[0-9]+)?"
)


def parse_log(text: str) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    cur: dict[str, float] = {}
    for m in _TOKEN.finditer(text):
        name, val = m.group(1), m.group(2)
        if name == "step":
            if "step" in cur:
                records.append(cur)
            cur = {"step": int(val)}
        elif "step" in cur:
            try:
                cur[name] = float(val)
            except ValueError:
                pass
    if "step" in cur:
        records.append(cur)
    # keep only blocks that actually carried a total loss
    return [r for r in records if "loss" in r]


def _pick_log(path: Path) -> Path:
    if path.is_dir():
        logs = sorted(path.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            raise FileNotFoundError(f"no *.log under {path}")
        return logs[0]
    return path


def _rolling_mean(xs: list[float], win: int) -> list[float]:
    if win <= 1:
        return xs
    out: list[float] = []
    acc = 0.0
    from collections import deque

    dq: deque[float] = deque()
    for x in xs:
        dq.append(x)
        acc += x
        if len(dq) > win:
            acc -= dq.popleft()
        out.append(acc / len(dq))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot trainer loss curve from a log")
    ap.add_argument("log", type=Path, help="log file or directory containing *.log")
    ap.add_argument("--out", type=Path, default=None, help="output PNG (default: <logdir>/loss_curve.png)")
    ap.add_argument("--smooth", type=int, default=21, help="rolling-mean window in samples (0/1 = off)")
    args = ap.parse_args()

    logf = _pick_log(args.log)
    records = parse_log(logf.read_text(errors="ignore"))
    if not records:
        raise SystemExit(f"no loss records parsed from {logf}")

    out = args.out or (logf.parent / "loss_curve.png")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "loss", "loss_action", "loss_video", "lr"])
        for r in records:
            w.writerow([r.get("step"), r.get("loss"), r.get("loss_action"), r.get("loss_video"), r.get("lr")])

    steps = [r["step"] for r in records]
    total = [r["loss"] for r in records]
    action = [r.get("loss_action", float("nan")) for r in records]
    video = [r.get("loss_video", float("nan")) for r in records]

    last = records[-1]
    print(f"log={logf}")
    print(f"parsed {len(records)} points, step {steps[0]}..{steps[-1]}")
    print(f"last: step={last['step']} loss={last['loss']:.4f} "
          f"action={last.get('loss_action', float('nan')):.4f} "
          f"video={last.get('loss_video', float('nan')):.4f}")
    print(f"min total loss = {min(total):.4f} at step {steps[total.index(min(total))]}")
    print(f"csv -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; wrote CSV only.")
        return 0

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(steps, total, color="#c0c0c0", lw=0.8, alpha=0.7, label="loss (raw)")
    ax.plot(steps, action, color="#7fbf7f", lw=0.6, alpha=0.5, label="action (raw)")
    ax.plot(steps, video, color="#7f9fbf", lw=0.6, alpha=0.5, label="video (raw)")
    if args.smooth and args.smooth > 1:
        ax.plot(steps, _rolling_mean(total, args.smooth), color="#1f1f1f", lw=1.8, label=f"loss (mean{args.smooth})")
        ax.plot(steps, _rolling_mean(action, args.smooth), color="#2ca02c", lw=1.6, label=f"action (mean{args.smooth})")
        ax.plot(steps, _rolling_mean(video, args.smooth), color="#1f77b4", lw=1.6, label=f"video (mean{args.smooth})")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"B0 v2 loss — {logf.name} (step {steps[0]}..{steps[-1]}, n={len(records)})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"png -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
