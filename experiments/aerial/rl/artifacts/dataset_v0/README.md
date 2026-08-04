# dataset_v0 — live AirSim collection (2026-08-04)

- Host path (H100): `/home/a25689/rl_collect_run/experiments/aerial/rl/artifacts/dataset_v0`
- Client: H100 `:31126` → renderer `10.229.20.125:41451`
- Annotation: `aerial_cache_shared/orchestration/heldout/seen_airsim16_m1a20.json` (20 eps)
- Config: `--step-hz 12`, RGB-only (`grab_depth=false`), `--max-steps 200`
- Heavy `episode_*.npz` stay on the collect host (~238 MB); this dir tracks health JSON only.

## Outcome

- Exit 0; gate marked all 20 nontrivial under current rules (`steps<=1` skips path-length check).
- Achieved Hz ≈ **7.1–8.3** (mean ~7.9), under the 12 Hz command rate (closed-loop + RPC).
- 3 episodes are 1-step immediate collisions (ep 8/9/19, return ≈ −10); remaining ~17 fly ~46–55 m / 200 steps.
