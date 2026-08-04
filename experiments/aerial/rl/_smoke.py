"""Smoke driver for the RL skeleton — one collection pass with hard assertions.

Not a pytest module (leading underscore → not collected). Invoked by
``run_rl_smoke.sh``; also runnable directly:

    python -m experiments.aerial.rl._smoke --backend mock  --max-steps 100
    python -m experiments.aerial.rl._smoke --backend airsim --max-steps 100 \
        --min-hz 24 --host 10.229.20.125 --port 41451

Goes through ``build_from_config`` (no Hydra dependency) with
``corrector.smoke=true``, runs exactly one collection episode, prints a compact
summary, and exits non-zero if any invariant fails:

  * at least one step was collected and pushed to the buffer;
  * the V1/V4 gates stayed OFF (``wm``/``rl`` report ``skipped`` — a smoke run
    must not fabricate training progress);
  * (airsim only, when ``--min-hz`` > 0) the achieved rate clears the threshold,
    validating the Plan-A ~30 Hz serial-real-env assumption on real hardware.
"""
from __future__ import annotations

import argparse
import logging
import sys

from experiments.aerial.rl.train_rl import build_from_config


def _build_cfg(args: argparse.Namespace) -> dict:
    env: dict = {"backend": args.backend, "step_hz": args.step_hz}
    if args.backend == "airsim":
        env.update(
            host=args.host,
            port=args.port,
            camera=args.camera,
            vehicle=args.vehicle,
            health_check=not args.no_health_check,
        )
    return {
        "env": env,
        "dynamics": {"kind": "stub", "latent_dim": 8},
        "corrector": {"smoke": True, "max_steps": args.max_steps},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["mock", "airsim"], default="mock")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--step-hz", type=float, default=30.0)
    p.add_argument(
        "--min-hz", type=float, default=0.0,
        help="fail if achieved Hz below this (0 = don't assert; use ~24 for airsim)",
    )
    p.add_argument("--host", default="10.229.20.125")
    p.add_argument("--port", type=int, default=41451)
    p.add_argument("--camera", default="front_custom")
    p.add_argument("--vehicle", default="drone_1")
    p.add_argument("--no-health-check", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[smoke] %(message)s")

    loop = build_from_config(_build_cfg(args))
    reports = loop.run()
    rep = reports[0]
    steps = rep.collect.steps
    hz = rep.collect.achieved_hz
    wm_status = rep.wm.get("status", "skipped" if rep.wm.get("skipped") else "?")
    rl_status = rep.rl.get("status", "skipped" if rep.rl.get("skipped") else "?")

    print("=" * 56)
    print(f" backend      : {args.backend}")
    print(f" steps        : {steps}")
    print(f" achieved_hz  : {hz:.1f}" + (f"  (target ≥ {args.min_hz:.0f})" if args.min_hz > 0 else ""))
    print(f" buffer eps   : {len(loop.buffer)}")
    print(f" wm gate      : {wm_status}")
    print(f" rl gate      : {rl_status}")
    print("=" * 56)

    failures: list[str] = []
    if steps <= 0:
        failures.append("no steps collected")
    if len(loop.buffer) <= 0:
        failures.append("buffer empty after collection")
    # A smoke run exercises V0 collection only; the learning gates must stay OFF.
    if wm_status not in ("skipped", "noop"):
        failures.append(f"wm gate reported {wm_status!r} (expected skipped/noop)")
    if rl_status not in ("skipped", "noop"):
        failures.append(f"rl gate reported {rl_status!r} (expected skipped/noop)")
    if args.min_hz > 0 and hz < args.min_hz:
        failures.append(
            f"achieved {hz:.1f} Hz < {args.min_hz:.0f} Hz — Plan-A ~30 Hz "
            "serial-real-env assumption NOT met on this renderer"
        )

    if failures:
        for f in failures:
            print(f"[smoke] FAIL: {f}", file=sys.stderr)
        return 1
    print("[smoke] OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
