# dataset_v0 — live AirSim collection (2026-08-04)

## Status: **V0 smoke set — do NOT feed V1 WM training**

Commanded `--step-hz 12` but the closed loop only achieved **7.1–8.3 Hz**
(mean ≈7.9). Every action label therefore encodes an 83 ms `dt` the loop never
hit (~17% per-step dynamics desync). Treat this corpus as a **pipeline / quality
bring-up smoke**, not as a training set.

- **Next V1 training collection:** re-run with `--step-hz 8` (config default as of
  `493d846`), under the measured floor, so commanded `dt` matches wall-clock.
- Instant-crash quarantine (ep 8/9/19) is now formalized in code; this README's
  health JSON still reflects the pre-quarantine gate.

## Collection facts

- Host path (H100): `/home/a25689/rl_collect_run/experiments/aerial/rl/artifacts/dataset_v0`
- Client: H100 `:31126` → renderer `10.229.20.125:41451`
- Annotation: `aerial_cache_shared/orchestration/heldout/seen_airsim16_m1a20.json` (20 eps)
- Config used: `--step-hz 12`, RGB-only (`grab_depth=false`), `--max-steps 200`
- Heavy `episode_*.npz` stay on the collect host (~238 MB); this dir tracks health JSON only.

## Outcome (raw run)

- Exit 0 under the then-current gate (all 20 marked nontrivial; `steps<=1` skipped path check).
- Achieved Hz ≈ **7.1–8.3** (mean ~7.9).
- 3 episodes are 1-step immediate collisions (ep 8/9/19, return ≈ −10) → quarantine.
- Remaining ~17 fly ~46–55 m / 200 steps — usable for **load/buffer/stub-WM bring-up only**.
