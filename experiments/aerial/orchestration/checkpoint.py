from __future__ import annotations

import time
from pathlib import Path


def is_complete_checkpoint(
    pt_path: Path,
    *,
    settle_s: float = 5.0,
    min_bytes: int = 1_000_000_000,
) -> bool:
    sha = Path(str(pt_path) + ".sha256")
    if not pt_path.is_file() or not sha.is_file():
        return False
    try:
        size1 = pt_path.stat().st_size
    except FileNotFoundError:
        return False
    if size1 < min_bytes:
        return False
    if settle_s > 0:
        time.sleep(settle_s)
        if not pt_path.is_file() or not sha.is_file():
            return False
        try:
            size2 = pt_path.stat().st_size
        except FileNotFoundError:
            return False
        if size1 != size2:
            return False
    return True
