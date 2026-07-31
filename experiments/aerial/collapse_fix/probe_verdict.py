"""Stage-0 instruction-probe verdict (v3.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

Verdict = Literal["insensitive", "partial", "sensitive"]


def probe_sensitivity_verdict(primitives: Sequence[int]) -> Verdict:
    """Classify instruction sensitivity from predicted primitive list.

    Spec v3.2:
    - insensitive: |set(p)| == 1
    - sensitive: |set(p)| >= 3, OR (non-empty differs from empty AND >=2 distinct non-empty)
      Here we only see the primitive sequence; empty instruction is assumed index 0
      when the probe file puts "" first.
    - else partial
    """
    p = [int(x) for x in primitives]
    if not p:
        raise ValueError("empty primitive list")
    distinct = set(p)
    if len(distinct) == 1:
        return "insensitive"
    if len(distinct) >= 3:
        return "sensitive"
    # len(distinct) == 2
    empty_prim = p[0]
    non_empty = p[1:]
    if non_empty and any(x != empty_prim for x in non_empty) and len(set(non_empty)) >= 2:
        return "sensitive"
    return "partial"


def stage3_recipe(verdict: Verdict) -> dict[str, Any]:
    if verdict == "insensitive":
        return {
            "conditioning_dropout": True,
            "cfg": True,
            "weaken_first_frame_pin": False,
            "note": "full Stage 3 minus pin weaken (pin A/B only if still fails)",
        }
    if verdict == "partial":
        return {
            "conditioning_dropout": False,
            "cfg": True,
            "cfg_light": True,
            "weaken_first_frame_pin": False,
            "note": "light CFG only",
        }
    return {
        "conditioning_dropout": False,
        "cfg": False,
        "weaken_first_frame_pin": False,
        "note": "goal path live; rely on cls head + stop relabel; light CFG only if heading still bad",
    }


def verdict_from_summary_json(path: Path) -> dict[str, Any]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(records, dict) and "records" in records:
        records = records["records"]
    primitives = [int(r["primitive"]) for r in records]
    instructions = [str(r.get("instruction", "")) for r in records]
    verdict = probe_sensitivity_verdict(primitives)
    return {
        "verdict": verdict,
        "n_instructions": len(primitives),
        "n_distinct_primitives": len(set(primitives)),
        "primitives": primitives,
        "instructions": instructions,
        "stage3": stage3_recipe(verdict),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Classify wm_instruction_probe summary.json")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="write verdict.json")
    args = parser.parse_args(argv)
    result = verdict_from_summary_json(args.summary)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
