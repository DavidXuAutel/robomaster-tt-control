# Aerial closed-loop eval (M1a)

Phase-1 gate status: see [main checklist](../README.md#gate-checklist).

Closed-loop evaluation runner for OpenFly aerial VLN: FastWAM policy (or replay
baseline) drives discrete OpenFly primitives inside a sim bridge; metrics SR / NE /
SPL are written to JSON.

## Prerequisites

| Host | Purpose |
|------|---------|
| **Linux + NVIDIA GPU + CUDA** | Real M1a gate: AirSim env, OpenFly bridge, FastWAM checkpoint |
| **macOS / no GPU** | Offline wiring only: `--bridge mock --policy replay` |

### Linux: clone OpenFly-Platform

On the eval host (example path — adjust to your layout):

```bash
git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform.git /data/OpenFly-Platform
cd /data/OpenFly-Platform
# Follow upstream README: conda env, AirSim env download, deps
conda activate openfly
```

Download at least one AirSim scene used by `seen.json` (e.g. from
[Hugging Face OpenFly_DataGen/airsim](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen/tree/main/airsim))
into `/data/OpenFly-Platform/envs/airsim/env_airsim_xx/`.

Launch the bridge in a separate terminal before real eval:

```bash
cd /data/OpenFly-Platform
conda activate openfly
python scripts/sim/env_bridge.py --env env_airsim_16   # match your scene
# wait ~20s for "ready to be connected"
```

FastWAM imports `AirsimBridge` from
`/data/OpenFly-Platform/scripts/sim/airsim_bridge.py` via `sys.path` when
`--bridge openfly` is set.

### FastWAM checkpoint (Task 5)

M1a expects a Task 5 M0/M1 checkpoint, e.g.:

```text
runs/aerial_joint_1cam_1e-4/<RUN_ID>/checkpoints/weights/step_000050.pt
```

Training is blocked on macOS (no CUDA). Use the Linux train runbook in
`experiments/aerial/README.md`.

## CLI

```bash
cd /path/to/FastWAM
export PYTHONPATH=.

python -m experiments.aerial.eval.run_closed_loop \
  --openfly-root /data/OpenFly-Platform \
  --ann /data/OpenFly-Platform/Annotation/seen.json \
  --checkpoint runs/aerial_joint_1cam_1e-4/<RUN_ID>/checkpoints/weights/step_000050.pt \
  --bridge openfly \
  --policy fastwam \
  --max-episodes 20 \
  --max-steps 100 \
  --seed 42 \
  --out results/aerial_m1a/metrics.json
```

### Offline mock smoke (no AirSim)

Uses kinematic `MockBridge` + expert `ReplayPolicy` on tiny fixture episodes:

```bash
cd /path/to/FastWAM
PYTHONPATH=. python -m experiments.aerial.eval.run_closed_loop \
  --bridge mock \
  --policy replay \
  --ann experiments/aerial/tests/fixtures/mini_openfly/seen_mini.json \
  --max-episodes 3 \
  --max-steps 50 \
  --seed 42 \
  --out experiments/aerial/results/aerial_m1a/metrics_mock.json
```

Expected output shape:

```json
{"SR": 1.0, "NE": 0.0, "SPL": 1.0, "n": 3.0}
```

(Zeros are acceptable for an untrained policy on real sim; values must be finite
and seed-reproducible.)

## Bridges and policies

| Flag | Options | Notes |
|------|---------|-------|
| `--bridge` | `mock`, `openfly` | `mock` applies `primitive_to_delta` kinematically |
| `--policy` | `replay`, `fastwam` | `replay` replays annotation actions; `fastwam` loads checkpoint |

Success criterion: final distance to goal `< 20 m` (`OPENFLY_SUCCESS_DIST_M`).

## M1a exit criteria

- **Gate:** `results/aerial_m1a/metrics.json` from **≥ 20** closed-loop episodes
  on real OpenFly/AirSim with `--bridge openfly --policy fastwam`.
- **This macOS dev host:** mock smoke only (`metrics_mock.json`). True M1a gate
  remains **pending Linux + OpenFly + checkpoint**.

## Evidence (mock smoke)

```bash
PYTHONPATH=. python3 -m experiments.aerial.eval.run_closed_loop \
  --bridge mock --policy replay \
  --ann experiments/aerial/tests/fixtures/mini_openfly/seen_mini.json \
  --max-episodes 3 --max-steps 50 --seed 42 \
  --out experiments/aerial/results/aerial_m1a/metrics_mock.json
```

Artifact: `experiments/aerial/results/aerial_m1a/metrics_mock.json`

```json
{"SR": 1.0, "NE": 0.0, "SPL": 0.6666666666666666, "n": 3.0}
```

## Tests

```bash
PYTHONPATH=. pytest experiments/aerial/tests/test_run_closed_loop_mock.py -v
PYTHONPATH=. pytest experiments/aerial/tests/ -q
```
