# B0→B1 Orchestration Deploy Checklist

Stamp: `20260727-072347-5k-2gpu-b0-to-joint-video`

## Hosts

| Role | SSH | Notes |
|------|-----|-------|
| Train / supervisor (B0 + gates) | `a25689@10.239.121.22:31660` | Legacy B0 host; may be offline — prefer `:31103` for gates+FT |
| B1 fine-tune (+ gates after cutover) | `a25689@10.239.121.22:31103` | 2×H100; FT sync + smoke + B1 train; orch status after `:31660` down |
| Eval worker | `a25689@10.239.121.22:30682` | 1×H100; serial AirSim consumer |
| AirSim renderer | `yao@10.229.20.125` RPC `41451` | Single consumer only |

Do **not** touch Franka / `10.229.66.70` robot network.

## Shared paths (train pods and eval `:30682` do **not** share `/tmp`)

Use Ceph shared orchestration roots so enqueue/worker/lock agree:

| Artifact | Path |
|----------|------|
| Eval queue | `/home/a25689/aerial_cache_shared/orchestration/eval_queue` |
| Metrics/results | `/home/a25689/aerial_cache_shared/orchestration/results` |
| Supervisor status (train-local ok) | `/tmp/aerial_cache/orchestration/status.json` |
| Eval env (eval-local) | `/tmp/aerial_eval_cache/env.sh` |
| Held-out ann / OpenFly (eval-local paths embedded in jobs) | `/tmp/aerial_eval_cache/...` |

Export before starting workers:

```bash
export EVAL_QUEUE_DIR=/home/a25689/aerial_cache_shared/orchestration/eval_queue
export RESULTS_ROOT=/home/a25689/aerial_cache_shared/orchestration/results
```

## Code sync

1. From the orchestration worktree (`feat/aerial-b0-b1-orchestration`):
   - rsync/archive `experiments/aerial/orchestration/`, `experiments/aerial/eval/`, `experiments/aerial/scripts/orch_*.sh`, and related helpers into each host’s FastWAM runtime tree.
2. Confirm `PYTHONPATH=.` resolves `experiments.aerial.orchestration.supervisor`.
3. Confirm scripts are executable: `orch_eval_worker.sh`, `orch_supervisor.sh`, `orch_b0_wait_and_enqueue.sh`, `orch_b1_gates.sh`, `orch_b1_train.sh`, `orch_ckpt_watch_enqueue.sh`, `orch_s1_report.sh`.

## Eval host (`:30682`)

1. Ensure `/tmp/aerial_eval_cache/env.sh` exports the FastWAM/OpenFly runtime.
2. Confirm OpenFly bridge honors `AIRSIM_HOST=10.229.20.125` / `AIRSIM_PORT=41451`.
3. Confirm held-out ann exists: `/tmp/aerial_eval_cache/Annotation/seen_airsim16_m1a20.json`.
4. Confirm text embeds symlink under the eval cache.
5. Create dirs:
   - `/tmp/aerial_eval_cache/orchestration/eval_queue/{pending,running,done,failed}`
   - `/tmp/aerial_eval_cache/results/`
6. Start single-flight worker:

```bash
nohup bash experiments/aerial/scripts/orch_eval_worker.sh \
  > /tmp/aerial_eval_cache/orchestration/eval_worker.log 2>&1 &
```

7. Verify lock dir `/tmp/aerial_eval_cache/orchestration/eval_worker.lock` exists and only one worker is alive.

## Train host (`:31660` — B0 / gates / supervisor)

1. Mirror orchestration status root:
   - `/tmp/aerial_cache/orchestration/`
   - shared mirror under `/home/a25689/aerial_cache_shared/orchestration/` (optional but preferred)
2. Confirm B0 weights path:
   - `/home/a25689/aerial_cache_shared/runs/aerial_joint_b0_to_joint_video/m1b-<STAMP>/checkpoints/weights/`
3. Initialize / start supervisor (blocks until DONE/BLOCKED/FAILED):

```bash
export STAMP=20260727-072347-5k-2gpu-b0-to-joint-video
nohup bash experiments/aerial/scripts/orch_supervisor.sh \
  > /tmp/aerial_cache/orchestration/supervisor.log 2>&1 &
```

4. Watch `/tmp/aerial_cache/orchestration/status.json` for phase advances.
5. Confirm step_001000 seen-20 metrics are reused (not re-run) when already present.
6. Confirm no second AirSim client appears on the renderer while the eval worker is active.

## B1 fine-tune host (`:31103`)

1. `sync_b0_ft_to_h100.sh` defaults to `REMOTE_PORT=31103` → `/tmp/aerial_ft_cache`.
2. Mirror gate artifacts needed by `orch_b1_train.sh` onto `:31103`:
   - `/tmp/aerial_cache/orchestration/status.json` (must show `gates_passed` / `RUN_B1_TRAIN`)
   - `/tmp/aerial_cache/orchestration/baseline_lock.manifest.json`
3. Run 1-step / 10-step smoke on `:31103`, then `orch_b1_train.sh` there (not on `:31660`).
4. Checkpoint watch / B1 eval enqueue can remain driven from shared results + eval `:30682`.

## Arming gates before B1

Supervisor will enter `B1_GATES` after baseline lock. Gates remain `BLOCKED` until:

1. Collection source deployed and disjoint from held-out
2. Oracle gate JSON passed
3. Correction dataset accepted
4. FT cache SHA256 sync OK (`sync_b0_ft_to_h100.sh`)
5. 1-step / 10-step smoke marked `PASSED` in `ft_smoke.status`

Do **not** fabricate collection routes from held-out.

## Verification snippets

```bash
# status
python3 -c 'import json; print(json.load(open("/tmp/aerial_cache/orchestration/status.json")))'

# queue depth
find /tmp/aerial_eval_cache/orchestration/eval_queue -name '*.json' | sort

# worker lock
ls -ld /tmp/aerial_eval_cache/orchestration/eval_worker.lock
```
