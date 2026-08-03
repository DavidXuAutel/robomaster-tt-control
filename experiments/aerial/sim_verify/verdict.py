#!/usr/bin/env python3
"""Read sim_capability_report.json and decide the Fork (A / A- / B).

  Fork A   real RGB + IMU + (baro|gps) + collision + depth + physics
           + continuous_frames (L2f) all pass
           -> v2's inertial+dense+optical-flow foundation is achievable;
              proceed to v2 §7 V0.
  Fork A-  real RGB + depth pass, but IMU/physics/L2f fail (often CV mode
           or insufficient frame rate)
           -> switch settings.json to SimMode:Multirotor / fix camera rate,
              re-run, re-verify.
  Fork B   no real RGB (can't connect / only placeholder)
           -> sim renderer unusable; v2 unreachable until fixed.

`pass` on T2 items already includes numerical sanity (see lib/sanity.py).

Exit code: 0=Fork A, 2=Fork A-, 3=Fork B (so run_all.sh can branch).
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import report  # noqa: E402


def _ok(node) -> bool:
    return isinstance(node, dict) and bool(node.get("pass"))


def main() -> int:
    data = report.load()
    if not data:
        print("[verdict] no report found at", report.report_path())
        return 3

    t0 = data.get("t0_connectivity", {})
    t1 = data.get("t1_render", {})
    t2 = data.get("t2_capability", {})

    real_rgb = _ok(t1)
    connected = bool(t0.get("connected"))
    imu = _ok(t2.get("imu"))
    baro = _ok(t2.get("barometer"))
    gps = _ok(t2.get("gps"))
    height = baro or gps
    coll = _ok(t2.get("collision"))
    depth = _ok(t2.get("depth"))
    phys = _ok(t2.get("physics"))
    continuous = _ok(t2.get("continuous_frames"))

    caps = {
        "connected": connected,
        "real_rgb (L1)": real_rgb,
        "imu (L2a)": imu,
        "height baro|gps (L2b)": height,
        "collision (L2c)": coll,
        "depth_sane (L2d)": depth,
        "physics (L2e)": phys,
        "continuous_frames (L2f)": continuous,
    }
    print("[verdict] capability matrix:")
    for k, v in caps.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")

    if not (connected and real_rgb):
        fork, code, why = "B", 3, "no real scene RGB (connect/render failed)"
    elif imu and height and coll and depth and phys and continuous:
        fork, code, why = (
            "A",
            0,
            "full inertial+dense+physics+continuous RGB available (sanity-checked)",
        )
    elif real_rgb and depth:
        missing = [k for k, v in caps.items() if not v and k not in ("connected", "real_rgb (L1)")]
        fork, code, why = (
            "A-",
            2,
            "camera+depth ok but missing/failed: "
            + ", ".join(missing)
            + " — often ComputerVision mode or insufficient frame rate; fix settings and re-verify",
        )
    else:
        fork, code, why = "A-", 2, "partial capability — inspect report, fix settings.json, re-verify"

    print(f"\n[verdict] ==> Fork {fork}: {why}")
    print(f"[verdict] report: {report.report_path()}")
    report.merge("verdict", {"fork": fork, "reason": why, "matrix": caps})
    return code


if __name__ == "__main__":
    sys.exit(main())
