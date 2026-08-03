from __future__ import annotations

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any


class Phase(str, Enum):
    WAIT_B0_COMPLETE = "WAIT_B0_COMPLETE"
    EVAL_B0_CHECKPOINTS = "EVAL_B0_CHECKPOINTS"
    LOCK_BASELINE = "LOCK_BASELINE"
    B1_GATES = "B1_GATES"
    RUN_B1_TRAIN = "RUN_B1_TRAIN"
    EVAL_B1_CHECKPOINTS = "EVAL_B1_CHECKPOINTS"
    S1_REPORT = "S1_REPORT"
    DONE = "DONE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


def read_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".status.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
