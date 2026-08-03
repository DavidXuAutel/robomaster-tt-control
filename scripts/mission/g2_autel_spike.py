#!/usr/bin/env python3
"""Autel spike 机读摘要。无 SDK 时 dry_run → simulated；--require-hardware 遇 NOT_RUN/SKIP 非零退出。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adapters.drone_autel import AutelScoutAdapter  # noqa: E402
from mission_brain.map_model import SharedMap  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-hardware", action="store_true")
    ap.add_argument("--map", default="configs/mission/shared_map.example.json")
    args = ap.parse_args()
    out: list = []
    adapter = AutelScoutAdapter(out.append, SharedMap.load(args.map), dry_run=True)
    adapter.connect()
    adapter.takeoff()
    adapter.ingest_telemetry({"alt_m": 1.0})
    adapter.land()
    adapter.abort("spike_script")
    summary = adapter.spike.summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    code = adapter.spike.exit_code(require_hardware=args.require_hardware)
    print(f"exit_code={code}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
