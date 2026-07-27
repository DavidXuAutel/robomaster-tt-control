# Aerial B0→B1 Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the current B0→joint-video run finishes, automatically evaluate available checkpoints, lock the lowest mean-NE baseline, launch B1 failure-replay fine-tune, and serially evaluate every B1 checkpoint as soon as it is persisted.

**Architecture:** A small idempotent Python state machine writes status/manifests under `/tmp/aerial_*_cache/orchestration/` and mirrors critical artifacts to shared Ceph. Training (`:31660`) and eval (`:30682`) are separate workers sharing a FIFO eval queue; AirSim stays single-consumer. Reuse existing aerial eval/DAgger/FT helpers where present; add missing baseline lock + orchestration wrappers.

**Tech Stack:** Python 3.10, pytest, bash, FastWAM `experiments/aerial`, AirSim RPC `10.229.20.125:41451`, dual-H100 ZeRO-2, JSON manifests.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-aerial-b0-to-b1-orchestration-design.md`
- Dependent r2 FT design: `docs/superpowers/specs/2026-07-24-aerial-b0-failure-replay-finetune-design.md`
- Code lives primarily in FastWAM worktree: `/Users/xudazhong/Projects/FastWAM/.worktrees/aerial-wam-phase1`
- Docs/plans live in: `/Users/xudazhong/Projects/robomaster-tt-control`
- Train host: `a25689@10.239.121.22:31660`; eval host: `a25689@10.239.121.22:30682`; renderer: `10.229.20.125:41451`
- Never touch robot network / `10.229.66.70`
- Never fabricate collection routes from held-out
- Never start B1 while any B1 gate is unmet — enter `BLOCKED` with reason
- Baseline selection: lowest finite mean NE among complete B0 `step_001000`–`step_005000`; ties → later step
- B1 recipe: resume weights only, `lambda_action=1`, `lambda_video=0`, 75/25 mix, LR `1e-5`, max_steps 1000, save 250/500/1000
- Eval protocol: held-out seen-20, seed 42, max_steps 100, task matching the checkpoint family
- Current B0 stamp: `20260727-072347-5k-2gpu-b0-to-joint-video`

## File Map

| File | Responsibility |
|------|----------------|
| `experiments/aerial/orchestration/state.py` | Atomic JSON status read/write, phase enum |
| `experiments/aerial/orchestration/checkpoint.py` | Complete-ckpt detection (size stable + sha256) |
| `experiments/aerial/orchestration/eval_queue.py` | FIFO queue enqueue/dequeue/mark done |
| `experiments/aerial/eval/lock_baseline.py` | Select best B0 metrics → lock manifest + S1_NE |
| `experiments/aerial/scripts/orch_b0_wait_and_enqueue.sh` | Wait B0 complete; enqueue missing B0 evals |
| `experiments/aerial/scripts/orch_eval_worker.sh` | Serial eval worker consuming queue |
| `experiments/aerial/scripts/orch_b1_gates.sh` | Run B1 gates; BLOCKED on failure |
| `experiments/aerial/scripts/orch_b1_train.sh` | Launch B1 FT on `:31660` |
| `experiments/aerial/scripts/orch_ckpt_watch_enqueue.sh` | Watch B1 ckpts and enqueue evals |
| `experiments/aerial/scripts/orch_s1_report.sh` | Compare FT metrics vs lock manifest |
| `experiments/aerial/scripts/orch_supervisor.sh` | Top-level phase driver |
| tests under `experiments/aerial/tests/test_orch_*.py` / `test_lock_baseline.py` | Unit coverage |

Reuse existing (do not rewrite unless needed for dynamic S1):

- `experiments/aerial/eval/run_closed_loop.py`
- `experiments/aerial/eval/compare_finetune.py` (update to read `s1_ne` from lock manifest)
- `experiments/aerial/eval/collect_dagger.py`, `run_oracle_gate.py`, `ft_mix_dataset.py`, `path_expert.py`

---

### Task 1: Orchestration state + checkpoint helpers

**Files:**
- Create: `experiments/aerial/orchestration/__init__.py`
- Create: `experiments/aerial/orchestration/state.py`
- Create: `experiments/aerial/orchestration/checkpoint.py`
- Test: `experiments/aerial/tests/test_orch_state.py`
- Test: `experiments/aerial/tests/test_orch_checkpoint.py`

**Interfaces:**
- Produces: `Phase` enum; `read_status(path) -> dict`; `write_status(path, payload)` atomic; `is_complete_checkpoint(pt_path: Path, *, settle_s: float = 5.0) -> bool`

- [ ] **Step 1: Write failing tests**

```python
# experiments/aerial/tests/test_orch_state.py
from experiments.aerial.orchestration.state import Phase, read_status, write_status

def test_write_status_atomic_roundtrip(tmp_path):
    path = tmp_path / "status.json"
    write_status(path, {"phase": Phase.WAIT_B0_COMPLETE.value, "stamp": "x"})
    assert read_status(path)["phase"] == "WAIT_B0_COMPLETE"

def test_read_missing_returns_empty(tmp_path):
    assert read_status(tmp_path / "missing.json") == {}
```

```python
# experiments/aerial/tests/test_orch_checkpoint.py
from experiments.aerial.orchestration.checkpoint import is_complete_checkpoint

def test_complete_requires_sha_and_stable_size(tmp_path):
    pt = tmp_path / "step_001000.pt"
    pt.write_bytes(b"abc")
    assert is_complete_checkpoint(pt, settle_s=0.0) is False
    (tmp_path / "step_001000.pt.sha256").write_text("deadbeef  step_001000.pt\n")
    assert is_complete_checkpoint(pt, settle_s=0.0) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/xudazhong/Projects/FastWAM/.worktrees/aerial-wam-phase1 && PYTHONPATH=. pytest experiments/aerial/tests/test_orch_state.py experiments/aerial/tests/test_orch_checkpoint.py -v`
Expected: FAIL import / not found

- [ ] **Step 3: Implement minimal helpers**

```python
# experiments/aerial/orchestration/state.py
from __future__ import annotations
import json, os, tempfile
from enum import Enum
from pathlib import Path
from typing import Any

class Phase(str, Enum):
    WAIT_B0_COMPLETE = "WAIT_B0_COMPLETE"
    EVAL_B0_CHECKPOINTS = "EVAL_B0_CHECKPOINTS"
    LOCK_BASELINE = "LOCK_BASELINE"
    B1_GATES = "B1_GATES"
    RUN_B1_TRAIN = "RUN_B1_TRAIN"
    EVAL_B1_CHECKPOINTS = "EVAL_B1_CHECKPOINTS"
    S1_REPORT = "S1_REPORT"
    DONE = "DONE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"

def read_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".status.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
```

```python
# experiments/aerial/orchestration/checkpoint.py
from __future__ import annotations
import time
from pathlib import Path

def is_complete_checkpoint(pt_path: Path, *, settle_s: float = 5.0) -> bool:
    sha = Path(str(pt_path) + ".sha256")
    if not pt_path.is_file() or not sha.is_file():
        return False
    size1 = pt_path.stat().st_size
    if size1 < 1_000_000_000:  # refuse tiny/partial aerial ckpts
        return False
    if settle_s > 0:
        time.sleep(settle_s)
        size2 = pt_path.stat().st_size
        if size1 != size2:
            return False
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest experiments/aerial/tests/test_orch_state.py experiments/aerial/tests/test_orch_checkpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/xudazhong/Projects/FastWAM/.worktrees/aerial-wam-phase1 add experiments/aerial/orchestration experiments/aerial/tests/test_orch_state.py experiments/aerial/tests/test_orch_checkpoint.py
git -C /Users/xudazhong/Projects/FastWAM/.worktrees/aerial-wam-phase1 commit -m "feat(aerial): add orchestration state and checkpoint helpers"
```

---

### Task 2: Eval queue

**Files:**
- Create: `experiments/aerial/orchestration/eval_queue.py`
- Test: `experiments/aerial/tests/test_orch_eval_queue.py`

**Interfaces:**
- Consumes: `write_status` atomic pattern
- Produces: `enqueue(queue_dir, job: dict) -> str`; `claim_next(queue_dir) -> dict | None`; `mark_done(queue_dir, job_id, result: dict)`

Job schema:

```python
{
  "id": "b0-step_001000",
  "kind": "b0" | "b1",
  "checkpoint": "/abs/path/step_XXXXXX.pt",
  "out_metrics": "/abs/path/metrics.json",
  "task": "aerial_joint_b1_joint",
  "ann": "/abs/seen_airsim16_m1a20.json",
  "openfly_root": "/abs/OpenFly-Platform",
  "seed": 42,
  "max_steps": 100,
  "max_episodes": 20,
}
```

- [ ] **Step 1: Write failing test**

```python
from experiments.aerial.orchestration.eval_queue import enqueue, claim_next, mark_done

def test_fifo_claim_and_skip_existing_metrics(tmp_path):
    q = tmp_path / "queue"
    metrics = tmp_path / "m.json"
    metrics.write_text('{"NE": 1.0, "SR": 0.0, "n": 20}\n')
    jid = enqueue(q, {
        "id": "b0-step_001000",
        "kind": "b0",
        "checkpoint": "/c.pt",
        "out_metrics": str(metrics),
        "task": "aerial_joint_b1_joint",
        "ann": "/a.json",
        "openfly_root": "/of",
        "seed": 42,
        "max_steps": 100,
        "max_episodes": 20,
    })
    assert claim_next(q) is None  # already has valid metrics
    metrics.unlink()
    job = claim_next(q)
    assert job is not None and job["id"] == jid
    mark_done(q, jid, {"NE": 12.3, "SR": 0.0, "n": 20.0})
    assert claim_next(q) is None
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `PYTHONPATH=. pytest experiments/aerial/tests/test_orch_eval_queue.py -v`

- [ ] **Step 3: Implement queue**

Store jobs as `queue/pending/<id>.json`, claimed as `queue/running/<id>.json`, done as `queue/done/<id>.json`. `claim_next` skips jobs whose `out_metrics` already contains finite `NE` and `n >= 1`. Valid metrics check:

```python
def metrics_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    data = json.loads(path.read_text())
    return math.isfinite(float(data.get("NE", "nan"))) and float(data.get("n", 0)) >= 1
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(aerial): add idempotent FIFO eval queue"
```

---

### Task 3: Baseline lock

**Files:**
- Create: `experiments/aerial/eval/lock_baseline.py`
- Test: `experiments/aerial/tests/test_lock_baseline.py`

**Interfaces:**
- Produces: `select_baseline(candidates: list[dict]) -> dict`; CLI writing `baseline_lock.manifest.json`
- Candidate dict: `{"step": int, "checkpoint": str, "metrics_path": str, "mean_ne": float, "sha256": str}`

- [ ] **Step 1: Write failing tests**

```python
from experiments.aerial.eval.lock_baseline import select_baseline, build_lock_manifest

def test_select_lowest_ne_tie_breaks_later_step():
    chosen = select_baseline([
        {"step": 1000, "mean_ne": 150.0, "checkpoint": "a", "metrics_path": "a.json", "sha256": "1"},
        {"step": 4000, "mean_ne": 120.0, "checkpoint": "b", "metrics_path": "b.json", "sha256": "2"},
        {"step": 5000, "mean_ne": 120.0, "checkpoint": "c", "metrics_path": "c.json", "sha256": "3"},
    ])
    assert chosen["step"] == 5000
    man = build_lock_manifest(chosen, candidates=[...], stamp="20260727-072347-5k-2gpu-b0-to-joint-video")
    assert man["s1_ne"] == 96.0
    assert man["baseline_mean_ne"] == 120.0
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement**

```python
def select_baseline(candidates: list[dict]) -> dict:
    finite = [c for c in candidates if math.isfinite(float(c["mean_ne"]))]
    if not finite:
        raise ValueError("no finite mean_ne candidates")
    return sorted(finite, key=lambda c: (float(c["mean_ne"]), -int(c["step"])))[0]

def build_lock_manifest(chosen: dict, *, candidates: list[dict], stamp: str) -> dict:
    baseline = float(chosen["mean_ne"])
    return {
        "stamp": stamp,
        "checkpoint": chosen["checkpoint"],
        "sha256": chosen["sha256"],
        "metrics_path": chosen["metrics_path"],
        "baseline_mean_ne": baseline,
        "s1_ne": 0.8 * baseline,
        "candidates": candidates,
        "selection_rule": "min_mean_ne_tie_later_step",
    }
```

CLI: `--stamp`, `--candidate step=ckpt=metrics=sha256` (repeatable), `--out`.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(aerial): add dynamic baseline lock with S1_NE"
```

---

### Task 4: Update compare_finetune for dynamic S1

**Files:**
- Modify: `experiments/aerial/eval/compare_finetune.py`
- Modify: `experiments/aerial/tests/test_compare_finetune.py`

**Interfaces:**
- Consumes: `baseline_lock.manifest.json` with `baseline_mean_ne` / `s1_ne`
- Produces: report using manifest thresholds (no hard-coded 135.94 / 108.75)

- [ ] **Step 1: Write/adjust failing test** asserting `summarize(..., s1_ne=120.0)` uses provided threshold, not module constants

- [ ] **Step 2: Run test — expect FAIL against old constants**

- [ ] **Step 3: Change `compare_finetune.py` to require `--lock-manifest` (or accept `--s1-ne` override) and remove hard pass dependency on `S1_NE_THRESHOLD` / `LOCKED_BASELINE_NE` for gate decisions. Keep old constants only as deprecated diagnostics if needed, but do not use them for exit code.

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "fix(aerial): compare_finetune uses locked dynamic S1_NE"
```

---

### Task 5: Eval worker script

**Files:**
- Create: `experiments/aerial/scripts/orch_eval_worker.sh`
- Test: `experiments/aerial/tests/test_orch_eval_worker_script.py` (dry-run / arg parsing style like existing script tests)

**Interfaces:**
- Consumes: eval queue jobs
- Produces: metrics JSON via `python -m experiments.aerial.eval.run_closed_loop`

- [ ] **Step 1: Write failing script test** that dry-run worker prints `run_closed_loop` with `--seed 42 --max-episodes 20`

- [ ] **Step 2: Implement worker**

```bash
#!/usr/bin/env bash
set -euo pipefail
# loop:
#   claim_next via python -m experiments.aerial.orchestration.eval_queue --claim
#   if empty: sleep 30; continue
#   source /tmp/aerial_eval_cache/env.sh
#   run_closed_loop ...
#   mark_done
# single-flight lockfile /tmp/aerial_eval_cache/orchestration/eval_worker.lock
```

Require `AIRSIM_HOST=10.229.20.125`, `AIRSIM_PORT=41451`, `AIRSIM_ALLOW_LOCAL_LAUNCH=0`.

- [ ] **Step 3: Dry-run PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(aerial): add serial orchestration eval worker"
```

---

### Task 6: B0 wait + enqueue

**Files:**
- Create: `experiments/aerial/scripts/orch_b0_wait_and_enqueue.sh`
- Create: `experiments/aerial/orchestration/b0_discover.py`
- Test: `experiments/aerial/tests/test_b0_discover.py`

**Interfaces:**
- Produces: enqueued jobs for each complete `step_00{1000,2000,3000,4000,5000}.pt` under shared weights dir

- [ ] **Step 1: Failing test for discover**

```python
def test_discover_only_complete_steps(tmp_path):
    weights = tmp_path / "checkpoints" / "weights"
    weights.mkdir(parents=True)
    # create fake large files + sha for 1000 only
    ...
    found = discover_b0_checkpoints(weights)
    assert [c["step"] for c in found] == [1000]
```

- [ ] **Step 2: Implement discover + wait script**

Wait until `step_005000.pt` complete on:

`/home/a25689/aerial_cache_shared/runs/aerial_joint_b0_to_joint_video/m1b-20260727-072347-5k-2gpu-b0-to-joint-video/checkpoints/weights/`

Then enqueue missing evals to eval-host queue dir (or shared queue mirror). Reuse existing step1000 metrics path if valid:

`/tmp/aerial_eval_cache/results/step_001000_seen20/metrics.json`

- [ ] **Step 3: Tests PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(aerial): discover and enqueue B0 checkpoint evals"
```

---

### Task 7: Supervisor phases LOCK → B1_GATES

**Files:**
- Create: `experiments/aerial/scripts/orch_b1_gates.sh`
- Create: `experiments/aerial/scripts/orch_supervisor.sh`
- Modify/wrap existing: `wait_videos_then_collect.sh`, `run_oracle_gate.py`, `collect_dagger.py`

**Interfaces:**
- Supervisor advances phases using `state.write_status`
- Gates script exits 0 only if all gates pass; else writes `BLOCKED` reason

Gate checklist (fail → BLOCKED, do not continue):

1. `baseline_lock.manifest.json` exists + sha matches file
2. collection source present + disjoint from held-out (`route_ids.assert_disjoint`)
3. oracle gate JSON passed
4. correction dataset accepted
5. FT cache SHA256 sync OK
6. 1-step and 10-step smoke OK

- [ ] **Step 1: Write failing test for gate reason writer** (pure function packaging blocked payload)

- [ ] **Step 2: Implement gates script calling existing tools; no fabrication**

- [ ] **Step 3: Supervisor: after all B0 queue jobs done → lock_baseline → b1_gates**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(aerial): add supervisor lock and B1 gate phase"
```

---

### Task 8: B1 train + ckpt watch enqueue

**Files:**
- Create: `experiments/aerial/scripts/orch_b1_train.sh` (adapt from `run_b0_ft_4090.sh` → H100 `:31660`, task `aerial_joint_b0_ft_dagger`, λ_v=0)
- Create: `experiments/aerial/scripts/orch_ckpt_watch_enqueue.sh`
- Create: `experiments/aerial/scripts/sync_b0_ft_to_h100.sh` (adapt from `sync_b0_ft_to_4090.sh`)

**Interfaces:**
- Train writes under `/tmp/aerial_ft_cache/runs/...` and persist mirrors to shared
- Watcher enqueues B1 jobs for 250/500/1000 when complete

- [ ] **Step 1: Failing dry-run tests asserting train command contains `lambda_video=0.0`, `max_steps=1000`, `save_every=250`, `learning_rate=1e-5`**

- [ ] **Step 2: Implement sync/smoke/train/watch scripts**

Train must:

- resume **weights only** from lock manifest checkpoint
- use nanfix ZeRO-2 2-proc accelerate config
- refuse start if gates status ≠ passed

Watcher:

- poll shared weights every 60s
- enqueue only complete ckpts
- do not block training

- [ ] **Step 3: Dry-run PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(aerial): add B1 H100 train and checkpoint eval enqueue"
```

---

### Task 9: S1 report + end-to-end supervisor wiring

**Files:**
- Create: `experiments/aerial/scripts/orch_s1_report.sh`
- Modify: `experiments/aerial/scripts/orch_supervisor.sh`
- Modify: `experiments/aerial/scripts/eval_ft_ckpts_seen20.sh` to accept lock manifest instead of hard-coded `step_004000.json`

**Interfaces:**
- After B1 250/500/1000 metrics exist, call `compare_finetune --lock-manifest ...`
- Write `ft_selection_report.json`; exit 0 iff S1 pass; on fail write diagnosis scaffold and set phase `DONE` with `s1_pass=false` (not auto-retry)

- [ ] **Step 1: Failing test that supervisor transitions `EVAL_B1_CHECKPOINTS` → `S1_REPORT` → `DONE` on fixture tree**

- [ ] **Step 2: Implement report script + wire supervisor**

Supervisor loop pseudocode:

```text
while True:
  status = read_status(...)
  phase = status.phase
  if phase == WAIT_B0_COMPLETE: run orch_b0_wait_and_enqueue; set EVAL_B0
  if phase == EVAL_B0_CHECKPOINTS: wait until pending+running empty; set LOCK
  if phase == LOCK_BASELINE: lock_baseline; set B1_GATES
  if phase == B1_GATES: orch_b1_gates || set BLOCKED; break
  if phase == RUN_B1_TRAIN: start train+watch if not running; set EVAL_B1
  if phase == EVAL_B1_CHECKPOINTS: wait for 250/500/1000 metrics; set S1_REPORT
  if phase == S1_REPORT: orch_s1_report; set DONE
  if phase in {DONE, BLOCKED, FAILED}: exit
  sleep 30
```

Eval worker runs as separate long-lived process on `:30682`.

- [ ] **Step 3: Local pytest for phase transition helper PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(aerial): wire B0→B1 supervisor through S1 report"
```

---

### Task 10: Deploy and arm on live hosts

**Files:**
- Create: `experiments/aerial/scripts/deploy_orchestration.md` (ops checklist only)
- Ops only on remotes (no secrets in repo)

- [ ] **Step 1: Sync FastWAM aerial orchestration code to both pods' runtime trees**

- [ ] **Step 2: On `:30682`, ensure `env.sh`, OpenFly patched bridge, held-out ann, text embeds symlink**

- [ ] **Step 3: Start `orch_eval_worker.sh` under nohup with lockfile**

- [ ] **Step 4: On `:31660`, start `orch_supervisor.sh` with stamp `20260727-072347-5k-2gpu-b0-to-joint-video`**

- [ ] **Step 5: Verify status JSON advances or correctly waits; confirm step1000 metrics reused; confirm no second AirSim client**

- [ ] **Step 6: Commit ops checklist**

```bash
git commit -m "docs(aerial): add B0→B1 orchestration deploy checklist"
```

Also commit the plan/spec references in robomaster-tt-control if not already:

```bash
git -C /Users/xudazhong/Projects/robomaster-tt-control add docs/superpowers/plans/2026-07-27-aerial-b0-to-b1-orchestration.md
git -C /Users/xudazhong/Projects/robomaster-tt-control commit -m "Add B0→B1 orchestration implementation plan."
```

---

## Self-Review

| Spec requirement | Task |
|------------------|------|
| Wait B0 complete + durable ckpts | Task 6, 10 |
| Eval all B0 1k–5k, reuse existing | Task 2, 5, 6 |
| Lock min mean NE + S1_NE | Task 3 |
| B1 gates / BLOCKED no fabrication | Task 7 |
| B1 train λ_v=0, 75/25, 1000/250 | Task 8 |
| Sync eval each B1 ckpt, serial AirSim | Task 2, 5, 8 |
| S1 report dynamic threshold | Task 4, 9 |
| Idempotent recovery | Task 1, 2, 9 |
| Host split / no robot net | Global + Task 10 |

No TBD placeholders remain. Hard-coded historical S1 constants are explicitly retired in Task 4.
