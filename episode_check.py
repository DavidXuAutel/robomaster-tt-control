#!/usr/bin/env python3
"""Wander / episode 数据 QA 工具。

核算：帧数一致性、depth_age 分布、动作直方图、wander_event 统计，
以及设计文档 §9.3 验收指标（可对任意 episode 目录运行）。

用法:
  .venv/bin/python episode_check.py logs/episodes/ep_YYYYMMDD_HHMMSS
  .venv/bin/python episode_check.py logs/episodes/ep_... --json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _hist(values: list[float], bins: list[float]) -> dict[str, int]:
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1] or (i == len(bins) - 2 and v == bins[i + 1]):
                counts[i] += 1
                break
    return {f"[{bins[i]},{bins[i+1]})": counts[i] for i in range(len(counts))}


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def check_episode(ep_dir: Path, clear_thresh: float = 0.40) -> dict[str, Any]:
    frames_csv = ep_dir / "frames.csv"
    meta_path = ep_dir / "meta.json"
    video = ep_dir / "video.mp4"

    if not frames_csv.exists():
        return {"ok": False, "error": f"missing {frames_csv}"}

    with frames_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_csv = len(rows)
    n_video = None
    if video.exists():
        try:
            import av  # type: ignore

            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                n_video = stream.frames or sum(1 for _ in container.decode(video=0))
        except Exception as e:
            n_video = f"unreadable:{e}"

    depth_ages: list[float] = []
    pitches: list[float] = []
    yaws: list[float] = []
    throttles: list[float] = []
    events: Counter[str] = Counter()
    states: Counter[str] = Counter()
    yaw_nonzero_low_mid = 0
    yaw_nonzero = 0
    turn_dirs: Counter[str] = Counter()

    for r in rows:
        try:
            if r.get("depth_age_ms") not in ("", None):
                depth_ages.append(float(r["depth_age_ms"]))
        except ValueError:
            pass
        try:
            pitches.append(float(r.get("act_pitch") or 0))
            yaws.append(float(r.get("act_yaw") or 0))
            throttles.append(float(r.get("act_throttle") or 0))
        except ValueError:
            pass
        ws = (r.get("wander_state") or "").strip()
        we = (r.get("wander_event") or "").strip()
        if ws:
            states[ws] += 1
        if we:
            events[we.split("(")[0]] += 1
            if we.startswith("TURN("):
                # TURN(L,50,obstacle) / TURN(R,...)
                try:
                    dir_ch = we.split("(")[1].split(",")[0]
                    turn_dirs[dir_ch] += 1
                except IndexError:
                    pass
        try:
            yaw = float(r.get("act_yaw") or 0)
            mid = float(r["near_mid"]) if r.get("near_mid") not in ("", None) else None
        except ValueError:
            yaw, mid = 0.0, None
        if abs(yaw) > 0:
            yaw_nonzero += 1
            if mid is not None and mid < clear_thresh:
                yaw_nonzero_low_mid += 1

    depth_ages_sorted = sorted(depth_ages)
    pitch_bins = [-100, -20, -5, 0, 5, 12, 20, 30, 100]
    yaw_bins = [-100, -40, -15, -1, 1, 15, 40, 100]
    thr_bins = [-100, -20, -1, 1, 20, 100]

    pitch_hist = _hist(pitches, pitch_bins)
    yaw_hist = _hist(yaws, yaw_bins)
    thr_hist = _hist(throttles, thr_bins)
    pitch_nonzero_bins = sum(1 for k, v in pitch_hist.items() if v > 0 and not k.startswith("[0,"))
    # 非空 bin：计数 > 0
    pitch_active = sum(1 for v in pitch_hist.values() if v > 0)
    yaw_active = sum(1 for v in yaw_hist.values() if v > 0)

    free_turn = sum(v for k, v in events.items() if k == "TURN")  # 粗计；细分看 raw
    # 更精确：扫原始事件
    obstacle_turns = 0
    free_turns = 0
    for r in rows:
        we = (r.get("wander_event") or "").strip()
        if ",obstacle)" in we:
            obstacle_turns += 1
        elif ",free)" in we:
            free_turns += 1

    yaw_low_mid_ratio = (
        (yaw_nonzero_low_mid / yaw_nonzero) if yaw_nonzero else 0.0
    )

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    frame_ok = (n_video is None) or (isinstance(n_video, int) and n_video == n_csv)
    depth_p95 = _pct(depth_ages_sorted, 95) if depth_ages_sorted else float("nan")
    depth_ok = (not depth_ages_sorted) or depth_p95 < 600.0
    diversity_ok = pitch_active >= 4 and yaw_active >= 4
    free_yaw_ok = yaw_low_mid_ratio >= 0.20
    turns_ok = obstacle_turns >= 8 and free_turns >= 3
    dirs_ok = len([k for k, v in turn_dirs.items() if v > 0]) >= 2 or obstacle_turns + free_turns == 0

    report = {
        "episode": str(ep_dir),
        "n_frames_csv": n_csv,
        "n_frames_video": n_video,
        "frame_match": frame_ok,
        "depth_age_ms": {
            "n": len(depth_ages),
            "p50": _pct(depth_ages_sorted, 50) if depth_ages_sorted else None,
            "p95": depth_p95 if depth_ages_sorted else None,
            "ok_p95_lt_600": depth_ok,
        },
        "action_hist": {
            "pitch": pitch_hist,
            "yaw": yaw_hist,
            "throttle": thr_hist,
            "pitch_active_bins": pitch_active,
            "yaw_active_bins": yaw_active,
            "diversity_ok": diversity_ok,
        },
        "wander": {
            "states": dict(states),
            "events": dict(events),
            "obstacle_turns": obstacle_turns,
            "free_turns": free_turns,
            "turn_dirs": dict(turn_dirs),
            "yaw_nonzero": yaw_nonzero,
            "yaw_nonzero_low_mid": yaw_nonzero_low_mid,
            "yaw_low_mid_ratio": round(yaw_low_mid_ratio, 3),
            "free_yaw_ok": free_yaw_ok,
            "turns_ok_acceptance": turns_ok,
            "dirs_not_all_same_side": dirs_ok,
        },
        "meta_notes": meta.get("notes", {}),
        "outcome": meta.get("outcome"),
        "abort_reason": meta.get("abort_reason"),
        "acceptance": {
            "frame_match": frame_ok,
            "depth_p95_ok": depth_ok,
            "diversity_ok": diversity_ok,
            "free_yaw_ok": free_yaw_ok,
            "turns_ok": turns_ok,
            "dirs_ok": dirs_ok,
        },
    }
    report["ok"] = all(report["acceptance"].values()) if n_csv > 0 else False
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Episode QA for wander explore data")
    ap.add_argument("episode", type=Path, help="episode 目录（含 frames.csv）")
    ap.add_argument("--json", action="store_true", help="仅输出 JSON")
    ap.add_argument("--clear-thresh", type=float, default=0.40)
    args = ap.parse_args(argv)

    report = check_episode(args.episode, clear_thresh=args.clear_thresh)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        acc = report.get("acceptance", {})
        print(f"episode: {report.get('episode')}")
        print(f"frames: csv={report.get('n_frames_csv')} video={report.get('n_frames_video')} match={acc.get('frame_match')}")
        da = report.get("depth_age_ms", {})
        print(f"depth_age_ms: p50={da.get('p50')} p95={da.get('p95')} ok={acc.get('depth_p95_ok')}")
        w = report.get("wander", {})
        print(
            f"turns: obstacle={w.get('obstacle_turns')} free={w.get('free_turns')} "
            f"dirs={w.get('turn_dirs')} ok={acc.get('turns_ok')}"
        )
        print(
            f"yaw_low_mid_ratio={w.get('yaw_low_mid_ratio')} "
            f"diversity={acc.get('diversity_ok')} free_yaw={acc.get('free_yaw_ok')}"
        )
        print(f"acceptance: {acc}")
        print(f"OVERALL: {'PASS' if report.get('ok') else 'FAIL'}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
