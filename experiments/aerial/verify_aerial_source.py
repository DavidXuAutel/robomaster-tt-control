"""Verify OpenFly LeRobot datasets against aerial_openfly registry criteria."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

SOURCE_ID = "openfly_lerobot"
PROFILE = "aerial_openfly"
DEFAULT_CRITERIA: Dict[str, Any] = {
    "sample_size": 100,
    "action_dim": 4,
    "max_step_translation_m": 15.0,
    "max_step_yaw_rad": math.pi,
    "require_gripper": False,
}


@dataclass
class AerialVerificationResult:
    source_id: str
    passed: bool
    samples_checked: int
    action_dim_ok_pct: float
    translation_bounds_ok_pct: float
    yaw_bounds_ok_pct: float
    gripper_absent: bool
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_aerial_criteria(registry_path: Path) -> Dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    criteria = dict(DEFAULT_CRITERIA)
    verification = registry.get("verification_criteria", {})
    if isinstance(verification.get(PROFILE), dict):
        criteria.update(verification[PROFILE])
    return criteria


def _action_vector(sample: Dict[str, Any]) -> np.ndarray:
    action = sample.get("action")
    if action is None:
        raise ValueError("sample missing action")
    return np.asarray(action, dtype=np.float64).reshape(-1)


def check_action_sample(action: np.ndarray, criteria: Dict[str, Any]) -> tuple[bool, List[str]]:
    notes: List[str] = []
    expected_dim = int(criteria["action_dim"])
    if action.shape[-1] != expected_dim:
        notes.append(f"action dim {action.shape[-1]} != {expected_dim}")
        return False, notes

    max_trans = float(criteria["max_step_translation_m"])
    max_yaw = float(criteria["max_step_yaw_rad"])
    dx, dy, dz, dyaw = action[:4]
    if max(abs(dx), abs(dy), abs(dz)) > max_trans:
        notes.append(
            f"translation out of bounds: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f} (max {max_trans})"
        )
        return False, notes
    if abs(dyaw) > max_yaw:
        notes.append(f"yaw out of bounds: dyaw={dyaw:.3f} (max {max_yaw})")
        return False, notes
    return True, notes


def verify_aerial_lerobot(
    *,
    lerobot_root: Path,
    criteria: Optional[Dict[str, Any]] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> AerialVerificationResult:
    from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset

    crit = dict(DEFAULT_CRITERIA)
    if criteria:
        crit.update(criteria)
    n_samples = int(sample_size if sample_size is not None else crit["sample_size"])

    root = lerobot_root.expanduser().resolve()
    if not root.exists():
        return AerialVerificationResult(
            source_id=SOURCE_ID,
            passed=False,
            samples_checked=0,
            action_dim_ok_pct=0.0,
            translation_bounds_ok_pct=0.0,
            yaw_bounds_ok_pct=0.0,
            gripper_absent=False,
            notes=[f"dataset root missing: {root}"],
        )

    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        features = info.get("features", {})
        if bool(crit.get("require_gripper")) is False and "gripper" in features:
            return AerialVerificationResult(
                source_id=SOURCE_ID,
                passed=False,
                samples_checked=0,
                action_dim_ok_pct=0.0,
                translation_bounds_ok_pct=0.0,
                yaw_bounds_ok_pct=0.0,
                gripper_absent=False,
                notes=["gripper feature present but aerial profile forbids it"],
            )

    dataset = LeRobotDataset(repo_id=str(root), root=root, download_videos=False)
    total = len(dataset)
    if total == 0:
        return AerialVerificationResult(
            source_id=SOURCE_ID,
            passed=False,
            samples_checked=0,
            action_dim_ok_pct=0.0,
            translation_bounds_ok_pct=0.0,
            yaw_bounds_ok_pct=0.0,
            gripper_absent=True,
            notes=["dataset empty"],
        )

    rng = random.Random(seed)
    indices = [rng.randrange(total) for _ in range(min(n_samples, total))]
    dim_ok = trans_ok = yaw_ok = 0
    notes: List[str] = []
    for idx in indices:
        action = _action_vector(dataset[idx])
        if action.shape[-1] == int(crit["action_dim"]):
            dim_ok += 1
        max_trans = float(crit["max_step_translation_m"])
        max_yaw = float(crit["max_step_yaw_rad"])
        dx, dy, dz, dyaw = action[:4]
        if max(abs(dx), abs(dy), abs(dz)) <= max_trans:
            trans_ok += 1
        if abs(dyaw) <= max_yaw:
            yaw_ok += 1

    checked = len(indices)
    dim_pct = 100.0 * dim_ok / checked
    trans_pct = 100.0 * trans_ok / checked
    yaw_pct = 100.0 * yaw_ok / checked
    passed = dim_ok == checked and trans_ok == checked and yaw_ok == checked
    if not passed:
        notes.append("one or more samples failed aerial bounds checks")
    return AerialVerificationResult(
        source_id=SOURCE_ID,
        passed=passed,
        samples_checked=checked,
        action_dim_ok_pct=dim_pct,
        translation_bounds_ok_pct=trans_pct,
        yaw_bounds_ok_pct=yaw_pct,
        gripper_absent=True,
        notes=notes,
    )


def write_verification_artifact(result: AerialVerificationResult, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out = artifact_dir / f"{result.source_id}.json"
    out.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path("./data/openfly_lerobot/train_subset"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("./configs/data_compatibility.yaml"),
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()

    criteria = load_aerial_criteria(args.registry.resolve())
    result = verify_aerial_lerobot(
        lerobot_root=args.lerobot_root,
        criteria=criteria,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(result.to_dict(), indent=2))
    artifact_dir = args.artifact_dir
    if artifact_dir is None:
        artifact_dir = Path(criteria.get("artifact_dir", "reports/registry_verification/aerial"))
    write_verification_artifact(result, artifact_dir)
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
