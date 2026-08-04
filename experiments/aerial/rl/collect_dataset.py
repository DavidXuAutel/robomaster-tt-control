"""Collect a V0 replay dataset to disk + emit a data-quality report.

The V1 world model trains on real rollouts, which the in-memory collector throws
away. This entrypoint runs the serial collector for ``--episodes`` episodes,
writes each to ``out_dir/episode_XXXXX.npz``, and aggregates a quality report so
a silent-but-fatal collection (frozen renderer, dead API control, black frames)
fails loudly instead of producing unusable training data.

    # offline dry-run (mock env, no GPU / renderer):
    python -m experiments.aerial.rl.collect_dataset --backend mock --episodes 3

    # on a 4090-reachable host (RGB-only, matches configs/aerial_rl.yaml V0):
    python -m experiments.aerial.rl.collect_dataset --backend airsim \
        --episodes 20 --max-steps 200 --step-hz 8 \
        --out experiments/aerial/rl/artifacts/dataset_v0

Hydra-free (builds via ``build_from_config``). Exits non-zero if any episode
trips ``assert_nontrivial`` — a dataset you shouldn't hand to WM training.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl.train_rl import build_from_config

logger = logging.getLogger(__name__)


def _build_cfg(args: argparse.Namespace) -> dict:
    env: dict = {"backend": args.backend, "step_hz": args.step_hz}
    if args.backend == "airsim":
        env.update(
            host=args.host, port=args.port, camera=args.camera,
            vehicle=args.vehicle, grab_depth=bool(args.grab_depth),
        )
    cfg: dict = {
        "env": env,
        "dynamics": {"kind": "stub", "latent_dim": 8},
        "corrector": {
            "iterations": args.episodes,
            "episodes_per_iter": 1,
            "max_steps": args.max_steps,
        },
    }
    if args.annotation:
        cfg["annotation"] = args.annotation
        cfg["max_episodes"] = args.episodes
    return cfg


def _mock_goal_episode() -> dict:
    """A start→goal episode so the heuristic actually moves the mock drone.

    Without a goal the heuristic idles → identical frames + zero path, which the
    quality gate (correctly) rejects. Real collection supplies goals via
    ``--annotation``; this keeps the offline dry-run a meaningful non-trivial run.
    """
    return {"pos": [[0.0, 0.0, 0.0], [30.0, 0.0, 5.0]], "yaw": [0.0, 0.0]}


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["mock", "airsim"], default="mock")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--step-hz", type=float, default=8.0,  # measured closed-loop floor (7.1–8.3 Hz)
                   help="serial-loop rate; keep <= measured closed-loop floor so dt matches")
    p.add_argument("--out", default="experiments/aerial/rl/artifacts/dataset_v0")
    p.add_argument("--host", default="10.229.20.125")
    p.add_argument("--port", type=int, default=41451)
    p.add_argument("--camera", default="front_custom")
    p.add_argument("--vehicle", default="drone_1")
    p.add_argument("--grab-depth", action="store_true")
    p.add_argument("--annotation", default=None,
                   help="OpenFly annotation JSON of start/goal episodes (real collection)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[collect] %(message)s")
    out_dir = Path(args.out)

    manifest: list[dict] = []
    reports: list[dict] = []
    failures: list[str] = []
    quarantined: list[str] = []

    def _sink(transitions, stats) -> None:
        idx = len(manifest)
        path = ds.write_episode(out_dir, idx, transitions)
        rep = ds.quality_report(transitions)
        rep["achieved_hz"] = round(stats.achieved_hz, 2)
        bad = ds.assert_nontrivial(rep)           # silent-but-fatal -> hard fail
        quar = ds.quarantine_reasons(rep)         # instant crash -> soft exclude
        status = "BAD" if bad else ("QUARANTINE" if quar else "OK")
        logger.info(
            "ep %d: %d steps @ %.1f Hz | path %.2f m | rew_sum %.2f | %s | %s",
            idx, rep["steps"], stats.achieved_hz, rep["path_length_m"],
            rep["reward_sum"], status, path.name,
        )
        for f in bad:
            failures.append(f"ep{idx}: {f}")
        for q in quar:
            quarantined.append(f"ep{idx}: {q}")
        manifest.append({"file": path.name, "steps": rep["steps"],
                         "return": rep["reward_sum"], "achieved_hz": rep["achieved_hz"],
                         "nontrivial": not bad, "quarantined": bool(quar),
                         "usable": not bad and not quar})
        reports.append(rep)

    loop = build_from_config(_build_cfg(args))
    loop.collector.on_episode = _sink
    # Mock dry-run with no annotation: inject a goal so the heuristic moves and
    # the run is non-trivial (real collection gets goals from --annotation).
    if args.backend == "mock" and loop.episodes is None:
        loop.episodes = [_mock_goal_episode()]
        logger.info("mock backend: injected a synthetic start→goal episode")

    # Collect N episodes in ONE collector.collect call so episode indexing
    # advances (i % len(episodes)). Routing through SerialCorrectorLoop with
    # iterations=N / episodes_per_iter=1 would restart i at 0 every iter and
    # silently re-collect annotation[0] N times.
    try:
        stats = loop.collector.collect(args.episodes, episodes=loop.episodes)
    finally:
        close = getattr(loop.collector.env, "close", None)
        if callable(close):
            close()

    n = len(manifest)
    quar_frac = (len(quarantined) / n) if n else 0.0
    ds.write_manifest(out_dir, manifest, meta={
        "backend": args.backend, "step_hz": args.step_hz,
        "max_steps": args.max_steps, "grab_depth": bool(args.grab_depth),
        "skipped_reset_collision": stats.skipped,
        "quarantined": len(quarantined),
        "quarantine_fraction": round(quar_frac, 3),
    })
    summary_path = ds.write_quality_summary(out_dir, reports)
    logger.info("wrote %d episodes + %s (skipped %d spawn-collision at reset)",
                n, summary_path.name, stats.skipped)

    # Silent-but-fatal episodes (frozen / black / never-moved) always fail.
    if failures:
        for f in failures:
            print(f"[collect] FAIL: {f}", file=sys.stderr)
        print(f"[collect] {len(failures)} trivial/dead episode(s) — dataset unusable", file=sys.stderr)
        return 1

    # Instant crashes are excluded per-episode, but a flood means the start-pose
    # distribution is broken — fail the run only past MAX_QUARANTINE_FRACTION.
    if quarantined:
        for q in quarantined:
            print(f"[collect] QUARANTINE: {q}", file=sys.stderr)
        if quar_frac > ds.MAX_QUARANTINE_FRACTION:
            print(f"[collect] {len(quarantined)}/{n} episodes quarantined "
                  f"({quar_frac:.0%} > {ds.MAX_QUARANTINE_FRACTION:.0%}) — "
                  "start poses likely broken", file=sys.stderr)
            return 1
        print(f"[collect] {len(quarantined)}/{n} quarantined "
              f"({quar_frac:.0%}, within tolerance) — excluded from usable set",
              file=sys.stderr)

    usable = n - len(quarantined)
    print(f"[collect] OK: {usable}/{n} usable episodes in {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
