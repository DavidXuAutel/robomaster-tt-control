"""Tiny JSON report accumulator shared by all probes.

Each probe loads the report (if present), merges its own top-level key, and
writes it back. No external deps. The report path comes from $OUT (default
./artifacts/sim_capability_report.json).
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any


def report_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("OUT", "./artifacts/sim_capability_report.json"))


def load() -> dict[str, Any]:
    p = report_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def merge(key: str, value: dict[str, Any]) -> dict[str, Any]:
    """Merge one probe's result under `key` and persist. Returns full report."""
    p = report_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load()
    data[key] = value
    p.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data
