#!/usr/bin/env python3
"""T0 — AirSim RPC connectivity + SimMode guess.

Cheapest real check. Confirms:
  1. the `airsim` python client imports,
  2. the RPC port on the 4090 renderer answers,
  3. a rough guess of SimMode (Multirotor vs ComputerVision).

Runs from the H100 client against $AIRSIM_HOST:$AIRSIM_PORT. Does NOT need the
OpenFly clone. Merges result under key "t0_connectivity".
"""
from __future__ import annotations

import os
import pathlib
import socket
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import report  # noqa: E402

HOST = os.environ.get("AIRSIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("AIRSIM_PORT", "41451"))

res: dict = {"host": HOST, "port": PORT}

# 1) raw TCP reach (like `nc -vz`)
try:
    with socket.create_connection((HOST, PORT), timeout=5):
        res["tcp_reachable"] = True
except OSError as e:
    res["tcp_reachable"] = False
    res["tcp_error"] = repr(e)

# 2) import airsim
try:
    import airsim  # type: ignore
    res["import_airsim"] = True
except Exception as e:  # noqa: BLE001
    res["import_airsim"] = False
    res["import_error"] = repr(e)

# 3) connect + mode guess
if res.get("tcp_reachable") and res.get("import_airsim"):
    import airsim  # type: ignore
    mode = None
    connected = False
    err = None
    try:
        c = airsim.MultirotorClient(ip=HOST, port=PORT)
        c.confirmConnection()
        connected = True
        # getMultirotorState only meaningful in Multirotor mode.
        try:
            c.getMultirotorState()
            mode = "Multirotor(or compatible)"
        except Exception:  # noqa: BLE001
            mode = "unknown (MultirotorClient connected but no drone state)"
    except Exception as e1:  # noqa: BLE001
        err = repr(e1)
        try:
            c = airsim.VehicleClient(ip=HOST, port=PORT)
            c.confirmConnection()
            connected = True
            mode = "ComputerVision(likely)"
        except Exception as e2:  # noqa: BLE001
            err = f"multirotor={err}; vehicle={e2!r}"
    res["connected"] = connected
    res["mode_guess"] = mode
    if err and not connected:
        res["connect_error"] = err

res["pass"] = bool(res.get("connected"))
report.merge("t0_connectivity", res)
print("[T0] connectivity:", res)
sys.exit(0 if res["pass"] else 1)
