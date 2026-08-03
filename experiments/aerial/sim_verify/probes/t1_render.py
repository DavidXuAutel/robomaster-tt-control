#!/usr/bin/env python3
"""T1 — Real scene RGB via OpenFly AirsimBridge (L1 capability).

Confirms the EXISTING deploy path works: instantiate the OpenFly
`AirsimBridge(env_name)`, teleport to a pose, grab a color frame, and verify
it's a REAL rendered scene — not the MockBridge constant-fill placeholder
(which has pixel std == 0).

Needs the OpenFly clone at $OPENFLY_ROOT (uses <root>/scripts/sim/airsim_bridge.py)
and a downloaded scene matching $ENV_NAME. Merges under "t1_render".
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import report  # noqa: E402

OPENFLY_ROOT = pathlib.Path(os.environ.get("OPENFLY_ROOT", "")).expanduser()
ENV_NAME = os.environ.get("ENV_NAME", "env_airsim_16")

res: dict = {"openfly_root": str(OPENFLY_ROOT), "env_name": ENV_NAME}

sim_dir = OPENFLY_ROOT / "scripts" / "sim"
if not sim_dir.is_dir():
    res["pass"] = False
    res["error"] = f"OpenFly scripts/sim not found under {OPENFLY_ROOT}"
    report.merge("t1_render", res)
    print("[T1]", res)
    sys.exit(1)

sys.path.insert(0, str(sim_dir))
try:
    from airsim_bridge import AirsimBridge  # type: ignore

    b = AirsimBridge(ENV_NAME)
    b.set_drone_pos(0.0, 0.0, -10.0, 0.0, 0.0, 0.0)  # NED: z<0 is up
    bgr = b.get_camera_data("color")
    img = np.asarray(bgr)
    ok = img.ndim == 3 and img.shape[2] == 3 and float(img.std()) > 3.0
    res.update({
        "pass": bool(ok),
        "shape": list(img.shape),
        "std": float(img.std()),
        "mean": float(img.mean()),
        "note": "std>3 => real scene, not MockBridge constant fill",
    })
except Exception as e:  # noqa: BLE001
    res["pass"] = False
    res["error"] = repr(e)

report.merge("t1_render", res)
print("[T1] render:", res)
sys.exit(0 if res.get("pass") else 1)
