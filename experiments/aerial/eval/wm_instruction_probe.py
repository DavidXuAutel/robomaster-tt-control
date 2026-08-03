from __future__ import annotations

"""Controlled instruction-sensitivity probe for the FastWAM world model.

Closed-loop eval cannot separate "goal-conditioning is broken" from "the world
model is undertrained": every episode sees a different observation, so a
degenerate rollout could be blamed on either. This probe removes that
confound. It holds the observation image, the proprio state, and the sampler
seed FIXED, then sweeps a list of instructions. With the initial noise latent
identical across runs, any difference in the imagined video (or the chosen
primitive) is attributable solely to the instruction text.

Read the verdict off ``summary.json`` + the dumped clips:
  * clips/primitives VARY across instructions  -> text conditioning is live;
    a degenerate closed loop is an undertrained-WM / IL-divergence problem.
  * clips/primitives are ~identical (incl. the empty-instruction baseline)
    -> goal-conditioning is not reaching the model; fix the text-encoder /
    instruction-injection path before spending more compute on training.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from experiments.aerial.eval.run_closed_loop import _save_wm_clip, build_policy


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image  # type: ignore[import-not-found]

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_instructions(path: Path) -> list[str]:
    """Accept either a JSON list of strings or a plain one-per-line text file."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        return [str(item) for item in data]
    return [line.strip() for line in text.splitlines() if line.strip()]


def probe_instructions(
    policy,
    obs_rgb: np.ndarray,
    state: np.ndarray,
    instructions: Sequence[str],
    out_dir: Path,
) -> list[dict]:
    """Run one forward pass per instruction on a fixed obs+state; dump each clip."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for index, instruction in enumerate(instructions):
        policy.dump_video = True
        primitive = int(policy.predict_primitive(obs_rgb, state, instruction))
        frames = getattr(policy, "last_generated_frames", None)
        prefix = f"probe_{index:02d}_wm"
        clip = _save_wm_clip(out_dir, prefix, frames) if frames else None
        records.append(
            {
                "index": index,
                "instruction": instruction,
                "primitive": primitive,
                "n_frames": len(frames) if frames else 0,
                "clip": str(clip) if clip else None,
            }
        )
        preview = instruction[:70].replace("\n", " ")
        print(
            f"[{index:02d}] primitive={primitive} "
            f"frames={len(frames) if frames else 0} :: {preview}"
        )
    return records


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FastWAM instruction-sensitivity probe (fixed obs+state, vary instruction)"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--obs", type=Path, required=True, help="fixed observation RGB (png)")
    parser.add_argument(
        "--instructions",
        type=Path,
        required=True,
        help="JSON list of instruction strings, or a one-per-line text file",
    )
    parser.add_argument("--out", type=Path, required=True, help="output dir for clips + summary.json")
    parser.add_argument("--task", default="aerial_joint_1cam_1e-4")
    parser.add_argument(
        "--state",
        type=float,
        nargs=4,
        metavar=("X", "Y", "Z", "YAW"),
        default=[0.0, 0.0, 0.0, 0.0],
        help="fixed proprio [x y z yaw]; held constant so only the instruction varies",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    instructions = load_instructions(args.instructions)
    if not instructions:
        raise SystemExit(f"no instructions found in {args.instructions}")

    obs_rgb = _load_rgb(args.obs)
    state = np.asarray(args.state, dtype=np.float32)

    # fastwam ignores the episode arg (only the replay policy reads it).
    policy = build_policy(
        "fastwam",
        {},
        checkpoint=args.checkpoint,
        task=str(args.task),
        seed=int(args.seed),
    )

    records = probe_instructions(policy, obs_rgb, state, instructions, args.out)

    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    primitives = {r["primitive"] for r in records}
    print(
        f"\n[probe] {len(records)} instructions -> "
        f"{len(primitives)} distinct primitive(s): {sorted(primitives)}"
    )
    print(f"[probe] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
