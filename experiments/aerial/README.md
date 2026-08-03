# Aerial OpenFly Phase-1

Phase-1 scope (Tasks 1–7): OpenFly→LeRobot bridge, `aerial_openfly` profile
(4D action, 1 camera), M0 train gate, and closed-loop SR/NE/SPL eval (M1a).

## Gate checklist

| Gate | Status | Evidence |
|------|--------|----------|
| M0 train (50 steps) | **BLOCKED** — pending Linux CUDA | Runbook below: [Prepare dataset](#prepare-the-m0-dataset) → [Precompute text embeddings](#precompute-text-embeddings) → [Run 50-step gate](#run-the-50-step-m0-gate) → [Evidence to retain](#evidence-to-retain). Checkpoint path: `runs/aerial_joint_1cam_1e-4/<RUN_ID>/checkpoints/weights/step_000050.pt`. Not executed on macOS dev host. |
| M1a mock | **DONE** | [eval/README.md — offline mock smoke](eval/README.md#evidence-mock-smoke): `PYTHONPATH=. python -m experiments.aerial.eval.run_closed_loop --bridge mock --policy replay --ann experiments/aerial/tests/fixtures/mini_openfly/seen_mini.json --max-episodes 3 --max-steps 50 --seed 42 --out experiments/aerial/results/aerial_m1a/metrics_mock.json` |
| M1a real | **PENDING** — Linux OpenFly + checkpoint | [eval/README.md — Linux setup & CLI](eval/README.md#linux-clone-openfly-platform): ≥20 episodes with `--bridge openfly --policy fastwam`; requires M0 checkpoint. Output: `results/aerial_m1a/metrics.json`. |

## M0 training runbook

This runbook is for the Phase-1 M0 gate: run the `aerial_joint_1cam_1e-4`
profile for 50 optimizer steps without a crash, with finite `loss_video` and
`loss_action`, and retain the resulting checkpoint. It must be run on a Linux
host with NVIDIA CUDA; the Wan2.2-5B model and the project CUDA dependencies
are not supported by this macOS development host.

## Prerequisites

Set up the project environment and model checkpoints as described in the
repository root `README.md`. Confirm CUDA is available before proceeding:

```bash
cd /path/to/FastWAM
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is required for the M0 train gate"
print("CUDA devices:", torch.cuda.device_count())
PY
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
```

Use a host with enough GPU memory for Wan2.2-5B training. The commands below
use eight local GPUs and DeepSpeed ZeRO-2; set `NPROC_PER_NODE` to the number
of GPUs available on the Linux host.

## Prepare the M0 dataset

M0 requires at least 200 converted trajectories. The small local `smoke`
dataset is only for structural checks and is not eligible for this gate.

```bash
cd /path/to/FastWAM

python -m experiments.aerial.download_openfly_subset \
  --config experiments/aerial/subset_manifest.example.yaml \
  --max-trajs 200

python -m experiments.aerial.convert_openfly_to_lerobot \
  --ann data/openfly_raw/Annotation/subset_train.json \
  --image-root data/openfly_raw \
  --out data/openfly_lerobot/train_subset

python experiments/aerial/verify_aerial_source.py \
  --lerobot-root data/openfly_lerobot/train_subset \
  --sample-size 200
```

The verification command must report `passed: true`. Confirm that
`data/openfly_lerobot/train_subset/meta/info.json` records at least 200
episodes before starting the gate.

## Precompute text embeddings

The dataset loader requires cached text contexts and deliberately fails if a
cache entry is absent. Generate them after the final dataset is in place:

```bash
cd /path/to/FastWAM
NPROC_PER_NODE=8
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  scripts/precompute_text_embeds.py task=aerial_joint_1cam_1e-4
```

This writes cache files under `data/text_embeds_cache/openfly/`. For a
single-GPU host, use `NPROC_PER_NODE=1`; the same command remains valid.

## Run the 50-step M0 gate

`scripts/train_zero2.sh` takes the GPU process count as its first positional
argument. The aerial task config fixes `max_steps=50`, `action_dim=4`,
`num_epochs=1`, `save_every=50`, and `eval_every=50`.

```bash
cd /path/to/FastWAM
NPROC_PER_NODE=8
RUN_ID="m0-$(date +%Y%m%d-%H%M%S)"
export RUN_ID

bash scripts/train_zero2.sh "$NPROC_PER_NODE" \
  task=aerial_joint_1cam_1e-4 \
  mixed_precision=bf16
```

For a multi-node launch, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`,
`MASTER_PORT`, and the same `RUN_ID` on every node, then run the same command
on each node.

## Evidence to retain

The expected run directory is:

```text
runs/aerial_joint_1cam_1e-4/$RUN_ID/
```

Record the GPU model/count, converted trajectory count, final log line, and:

```text
runs/aerial_joint_1cam_1e-4/$RUN_ID/checkpoints/weights/step_000050.pt
runs/aerial_joint_1cam_1e-4/$RUN_ID/checkpoints/state/step_000050/
```

Pass M0 only when logs show `[done] max_steps reached step=50`, the logged
`loss_video` and `loss_action` values are finite, and both checkpoint paths
exist. A CPU/MPS dataloader smoke is useful for wiring checks but is not
evidence for the M0 train gate.

## macOS / no-GPU preflight (optional)

Use the parent-repo virtualenv (`../.venv` or project `.venv`) and run wiring
checks only. This host cannot satisfy M0 because DeepSpeed training requires
NVIDIA CUDA.

```bash
cd /path/to/FastWAM
source .venv/bin/activate   # or /path/to/FastWAM/.venv/bin/python

python -m pytest experiments/aerial/tests/ -q
python experiments/aerial/m0_preflight.py --lerobot-root data/openfly_lerobot/smoke
```

Expected on smoke data without text cache: Hydra resolves `action_dim=4`,
`max_steps=50`, verification passes, dataset builds, then
`sample_ok=False` with `Missing text embedding cache` until
`precompute_text_embeds.py` runs on a GPU host.
