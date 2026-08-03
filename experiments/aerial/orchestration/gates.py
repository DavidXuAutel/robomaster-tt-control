from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.aerial.orchestration.state import Phase


def blocked_payload(
    *,
    stamp: str,
    failed_gate: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "phase": Phase.BLOCKED.value,
        "stamp": stamp,
        "failed_gate": failed_gate,
        "reason": reason,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def queue_is_idle(queue_dir: Path) -> bool:
    for subdir in ("pending", "running"):
        path = queue_dir / subdir
        if path.is_dir() and any(path.glob("*.json")):
            return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_baseline_lock(lock_path: Path) -> tuple[bool, str]:
    if not lock_path.is_file():
        return False, f"missing baseline lock: {lock_path}"
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid baseline lock JSON: {exc}"
    checkpoint = Path(str(data.get("checkpoint", "")))
    expected = str(data.get("sha256", ""))
    if not checkpoint.is_file():
        return False, f"lock checkpoint missing: {checkpoint}"
    if len(expected) != 64:
        return False, "lock sha256 missing or malformed"
    actual = _sha256_file(checkpoint)
    if actual != expected:
        return False, f"lock sha256 mismatch for {checkpoint}"
    return True, "ok"


def check_collection_source(
    *,
    collection_source: Path,
    heldout_ann: Path,
) -> tuple[bool, str]:
    if not collection_source.is_file():
        return (
            False,
            f"missing collection source: {collection_source}; refuse fabrication from held-out",
        )
    if not heldout_ann.is_file():
        return False, f"missing held-out annotation: {heldout_ann}"
    from experiments.aerial.route_ids import assert_disjoint, route_id

    collection = json.loads(collection_source.read_text(encoding="utf-8"))
    heldout = json.loads(heldout_ann.read_text(encoding="utf-8"))
    try:
        assert_disjoint(
            {route_id(e) for e in collection},
            {route_id(e) for e in heldout},
        )
    except ValueError as exc:
        return False, str(exc)
    return True, "ok"


def check_oracle_gate(oracle_json: Path) -> tuple[bool, str]:
    if not oracle_json.is_file():
        return False, f"missing oracle gate report: {oracle_json}"
    data = json.loads(oracle_json.read_text(encoding="utf-8"))
    if not data.get("passed", False):
        return False, f"oracle gate failed: {oracle_json}"
    return True, "ok"


def check_correction_dataset(correction_root: Path) -> tuple[bool, str]:
    meta = correction_root / "meta" / "info.json"
    if not meta.is_file():
        return False, f"missing correction dataset meta: {meta}"
    return True, "ok"


def check_ft_cache_manifest(manifest: Path) -> tuple[bool, str]:
    if not manifest.is_file():
        return False, f"missing FT SHA256 manifest: {manifest}"
    return True, "ok"


def check_smoke_status(smoke_status: Path) -> tuple[bool, str]:
    if not smoke_status.is_file():
        return False, f"missing FT smoke status: {smoke_status}"
    text = smoke_status.read_text(encoding="utf-8").strip().upper()
    if text != "PASSED":
        return False, f"FT smoke not passed: {smoke_status}={text!r}"
    return True, "ok"


def evaluate_b1_gates(
    *,
    stamp: str,
    lock_path: Path,
    collection_source: Path,
    heldout_ann: Path,
    oracle_json: Path,
    correction_root: Path,
    ft_manifest: Path,
    smoke_status: Path,
) -> dict[str, Any]:
    checks = [
        ("baseline_lock", lambda: check_baseline_lock(lock_path)),
        (
            "collection_source",
            lambda: check_collection_source(
                collection_source=collection_source,
                heldout_ann=heldout_ann,
            ),
        ),
        ("oracle_gate", lambda: check_oracle_gate(oracle_json)),
        ("correction_dataset", lambda: check_correction_dataset(correction_root)),
        ("ft_cache_manifest", lambda: check_ft_cache_manifest(ft_manifest)),
        ("ft_smoke", lambda: check_smoke_status(smoke_status)),
    ]
    for name, fn in checks:
        ok, reason = fn()
        if not ok:
            return blocked_payload(stamp=stamp, failed_gate=name, reason=reason)
    return {
        "phase": Phase.RUN_B1_TRAIN.value,
        "stamp": stamp,
        "gates_passed": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
