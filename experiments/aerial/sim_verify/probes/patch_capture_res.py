#!/usr/bin/env python3
"""Patch AirSim settings.json CaptureSettings Width/Height for Scene (ImageType 0).

Usage:
  python3 patch_capture_res.py --w 224 --h 224 \\
    --settings /home/yao/aerial_airsim_persistent/AirSim/settings.json \\
               /home/yao/Documents/AirSim/settings.json \\
               /home/yao/aerial_airsim_persistent/scene/env_airsim_16/LinuxNoEditor/AirVLN/Binaries/Linux/settings.json

On AirVLN Shipping layouts, the binary-adjacent settings.json is required —
Documents/AirSim alone may be ignored and Capture stays at 1920x1080.
Creates *.bak_l2f_res alongside each file the first time. Restart renderer after.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil


def patch(path: pathlib.Path, w: int, h: int) -> None:
    bak = path.with_suffix(path.suffix + ".bak_l2f_res")
    if not bak.exists():
        shutil.copy2(path, bak)
    data = json.loads(path.read_text())

    def fix_list(captures: list) -> int:
        n = 0
        for c in captures:
            if int(c.get("ImageType", -1)) == 0:  # Scene
                c["Width"] = w
                c["Height"] = h
                n += 1
        return n

    touched = 0
    cams = data.get("CameraDefaults", {}).get("CaptureSettings")
    if isinstance(cams, list):
        touched += fix_list(cams)

    for _vname, vcfg in (data.get("Vehicles") or {}).items():
        for _cname, ccfg in (vcfg.get("Cameras") or {}).items():
            cs = ccfg.get("CaptureSettings")
            if isinstance(cs, list):
                touched += fix_list(cs)

    if touched == 0:
        # ensure CameraDefaults has a Scene entry
        data.setdefault("CameraDefaults", {}).setdefault("CaptureSettings", []).append(
            {"ImageType": 0, "Width": w, "Height": h, "FOV_Degrees": 90}
        )
        touched = 1

    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"patched {path} scene_entries={touched} -> {w}x{h}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--settings", nargs="+", required=True)
    args = ap.parse_args()
    for s in args.settings:
        p = pathlib.Path(s)
        if not p.exists():
            print(f"skip missing {p}")
            continue
        patch(p, args.w, args.h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
