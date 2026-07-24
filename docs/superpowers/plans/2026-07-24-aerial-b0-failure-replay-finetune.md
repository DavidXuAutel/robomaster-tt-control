# Aerial B0 Failure-Replay Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect 40-episode AirSim threshold-takeover DAgger corrections for B0, fine-tune on dual RTX 4090 with 75/25 original/correction sampling, and pass S1 (held-out seen-20 mean NE ≤ 108.75649833236835).

**Architecture:** Reuse FastWAM aerial eval bridges/policies. Add a path expert + takeover controller that labels every state with continuous expert actions while B0 flies under threshold takeover. Export a separate LeRobot correction dataset, mix it with the existing OpenFly train_subset via a probability sampler, and resume B0 `step_004000` on `10.239.121.14:30879` with ZeRO-2 optimizer CPU offload.

**Tech Stack:** FastWAM (Hydra/Accelerate/DeepSpeed), OpenFly AirSim bridge, LeRobot v2.1 datasets, pytest, expect/SSH remotes.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-24-aerial-b0-failure-replay-finetune-design.md` (r1 confirmed).
- Code worktree: `/Users/xudazhong/Projects/FastWAM/.worktrees/aerial-wam-phase1` (branch `feat/aerial-wam-phase1`); do not train on the B1 H100 (`:31893`).
- Eval/collection host: `a25689@10.239.121.25:31126`; renderer: `yao@10.229.20.125` RPC `10.229.20.125:41451`; fine-tune host: `a25689@10.239.121.14:30879`.
- Baseline lock: B0 `step_004000.pt`, baseline NE `135.94562291546043`, S1 NE ≤ `108.75649833236835`.
- Held-out only: `/tmp/aerial_eval_cache/OpenFly-Platform/Annotation/seen_airsim16_m1a20.json`. Never collect from it.
- Collection source must exist at `/tmp/aerial_eval_cache/OpenFly-Platform/Annotation/seen_airsim16_collection_source.json` before Task 2 proceeds past dry-run fixtures.
- Do not touch robot network / `10.229.66.70`.
- Wait for current B0 video queue to finish before starting AirSim collection.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `experiments/aerial/route_ids.py` | Stable route ID + split/manifest helpers |
| `experiments/aerial/path_expert.py` | Monotone projection + lookahead expert actions |
| `experiments/aerial/takeover.py` | Threshold takeover state machine |
| `experiments/aerial/eval/collect_dagger.py` | AirSim DAgger collector CLI |
| `experiments/aerial/write_correction_lerobot.py` | Correction episode → LeRobot writer |
| `experiments/aerial/eval/compare_finetune.py` | Held-out baseline vs FT report |
| `src/fastwam/datasets/lerobot/weighted_source_dataset.py` | 75/25 source sampler wrapper |
| `configs/data/aerial_openfly_b0_ft_mix.yaml` | Mixed data config |
| `configs/task/aerial_joint_b0_ft_dagger.yaml` | FT task config |
| `experiments/aerial/scripts/` | Remote sync/smoke/FT shell helpers |
| `experiments/aerial/tests/test_*.py` | Unit tests for each module |

---

### Task 1: Route IDs and leak-free collection split

**Files:**
- Create: `experiments/aerial/route_ids.py`
- Create: `experiments/aerial/tests/test_route_ids.py`
- Create: `experiments/aerial/tests/fixtures/split/heldout_mini.json`
- Create: `experiments/aerial/tests/fixtures/split/collection_source_mini.json`

**Interfaces:**
- Produces: `route_id(episode: dict) -> str`, `assert_disjoint(a: set[str], b: set[str]) -> None`, `build_collection40(source_path, heldout_path, out_ann_path, out_manifest_path, *, seed=42, n=40) -> dict`

- [ ] **Step 1: Write failing tests**

```python
# experiments/aerial/tests/test_route_ids.py
import json
from pathlib import Path
import pytest
from experiments.aerial.route_ids import route_id, assert_disjoint, build_collection40

FIX = Path(__file__).parent / "fixtures" / "split"


def test_route_id_prefers_stable_fields():
    ep = {"trajectory_id": "t1", "scene_id": "env_airsim_16", "gpt_instruction": "go", "image_path": "env_airsim_16/a"}
    assert route_id(ep) == "t1"


def test_assert_disjoint_raises():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint({"a", "b"}, {"b"})


def test_build_collection40_no_heldout_leak(tmp_path):
    out_ann = tmp_path / "c40.json"
    out_man = tmp_path / "c40.manifest.json"
    man = build_collection40(
        FIX / "collection_source_mini.json",
        FIX / "heldout_mini.json",
        out_ann,
        out_man,
        seed=42,
        n=3,
    )
    held = {route_id(e) for e in json.loads((FIX / "heldout_mini.json").read_text())}
    got = {route_id(e) for e in json.loads(out_ann.read_text())}
    assert held.isdisjoint(got)
    assert man["n"] == 3
    assert man["seed"] == 42
```

Create fixtures: heldout 2 episodes with ids `h1,h2`; source 5 episodes with ids `h1,c1,c2,c3,c4` so selection must drop `h1`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/xudazhong/Projects/FastWAM/.worktrees/aerial-wam-phase1 && PYTHONPATH=. pytest experiments/aerial/tests/test_route_ids.py -v`
Expected: FAIL with `ModuleNotFoundError: experiments.aerial.route_ids`

- [ ] **Step 3: Implement `route_ids.py`**

```python
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any

def route_id(episode: dict[str, Any]) -> str:
    for key in ("trajectory_id", "traj_id", "id", "uuid"):
        if episode.get(key):
            return str(episode[key])
    blob = json.dumps(
        {
            "scene": episode.get("scene_id", episode.get("scene")),
            "image_path": episode.get("image_path"),
            "instruction": episode.get("gpt_instruction"),
            "start": (episode.get("pos") or [None])[0],
            "goal": (episode.get("pos") or [None])[-1],
        },
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]

def assert_disjoint(a: set[str], b: set[str]) -> None:
    overlap = sorted(a & b)
    if overlap:
        raise ValueError(f"heldout/collection overlap: {overlap[:10]}")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def build_collection40(
    source_path: Path,
    heldout_path: Path,
    out_ann_path: Path,
    out_manifest_path: Path,
    *,
    seed: int = 42,
    n: int = 40,
) -> dict[str, Any]:
    source = json.loads(Path(source_path).read_text())
    heldout = json.loads(Path(heldout_path).read_text())
    held_ids = {route_id(e) for e in heldout}
    candidates = [e for e in source if route_id(e) not in held_ids]
    if len(candidates) < n:
        raise ValueError(f"need {n} collection routes, have {len(candidates)}")
    rng = random.Random(seed)
    chosen = rng.sample(candidates, n)
    assert_disjoint(held_ids, {route_id(e) for e in chosen})
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
    out_ann_path.write_text(json.dumps(chosen, indent=2))
    manifest = {
        "seed": seed,
        "n": n,
        "heldout_path": str(heldout_path),
        "source_path": str(source_path),
        "out_ann_path": str(out_ann_path),
        "heldout_sha256": sha256_file(Path(heldout_path)),
        "source_sha256": sha256_file(Path(source_path)),
        "heldout_route_ids": sorted(held_ids),
        "collection_route_ids": [route_id(e) for e in chosen],
    }
    out_manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
```

- [ ] **Step 4: Re-run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add experiments/aerial/route_ids.py experiments/aerial/tests/test_route_ids.py experiments/aerial/tests/fixtures/split
git commit -m "feat(aerial): add leak-free collection40 route split"
```

---

### Task 2: Path expert (projection + lookahead)

**Files:**
- Create: `experiments/aerial/path_expert.py`
- Create: `experiments/aerial/tests/test_path_expert.py`
- Modify: `experiments/aerial/openfly_actions.py` only if a shared helper is missing (prefer not)

**Interfaces:**
- Consumes: `pos_yaw_to_body_delta`, `clip_body_delta` from `openfly_actions`
- Produces: `class PathExpert` with `reset(episode)`, `label(pos, yaw) -> ExpertLabel` where `ExpertLabel` has `action: np.ndarray`, `progress_m: float`, `cross_track_m: float`, `lookahead_pos: np.ndarray`

- [ ] **Step 1: Write failing tests** covering:
  - projection onto a straight polyline
  - monotone cursor never decreases after forward motion
  - lookahead shortens near goal
  - returned action is clipped into training ranges

```python
def test_monotone_progress_on_line():
    expert = PathExpert()
    expert.reset(_line_episode())  # pos [[0,0,0],[10,0,0],[20,0,0]], yaw zeros
    a = expert.label(np.array([1.0, 0.1, 0.0]), 0.0)
    b = expert.label(np.array([5.0, -0.1, 0.0]), 0.0)
    assert b.progress_m >= a.progress_m
    assert a.action.shape == (4,)
```

- [ ] **Step 2: Run tests — expect FAIL (import)**

- [ ] **Step 3: Implement `PathExpert`**
  - Build cumulative arc-length polyline from normalized episode poses
  - Project current xyz to nearest point on segments **at or after** cursor
  - Advance cursor to projection arc length
  - Lookahead target = point at `min(progress + 6.0, total_length)`
  - Action = `clip_body_delta(pos_yaw_to_body_delta(pos, yaw, lookahead_pos, lookahead_yaw))`

- [ ] **Step 4: Tests PASS**

- [ ] **Step 5: Commit** `feat(aerial): add monotone path expert for DAgger labels`

---

### Task 3: Takeover state machine

**Files:**
- Create: `experiments/aerial/takeover.py`
- Create: `experiments/aerial/tests/test_takeover.py`

**Interfaces:**
- Produces: `TakeoverConfig(takeover_m, release_m, abort_m, worsen_steps=3, stall_steps=8, release_stable_steps=3, no_progress_abort_steps=20)`, `freeze_thresholds(oracle_cross_track_p95: float) -> TakeoverConfig`, `class TakeoverController` with `step(cross_track_m, progress_m) -> TakeoverDecision` (`mode in {"policy","expert","abort"}`, `intervene: bool`)

- [ ] **Step 1: Failing tests**
  - `freeze_thresholds(1.0)` → takeover `max(9,3)=9`, release `6`, abort `max(30,27)=30`
  - `freeze_thresholds(5.0)` → takeover `15`, release `10`, abort `45`
  - controller enters expert when cross-track > takeover_m
  - NE is **not** an input to `step` (signature has no NE)
  - abort after 20 steps with no progress gain

- [ ] **Step 2: Implement minimal state machine**

```python
@dataclass
class TakeoverDecision:
    mode: str  # policy | expert | abort
    intervene: bool
    reason: str
```

- [ ] **Step 3: Tests PASS + commit** `feat(aerial): add threshold takeover controller`

---

### Task 4: DAgger collector + correction LeRobot writer

**Files:**
- Create: `experiments/aerial/eval/collect_dagger.py`
- Create: `experiments/aerial/write_correction_lerobot.py`
- Create: `experiments/aerial/tests/test_collect_dagger_mock.py`
- Create: `experiments/aerial/tests/test_write_correction_lerobot.py`

**Interfaces:**
- Consumes: `PathExpert`, `TakeoverController`, `build_bridge`, `build_policy`, `normalize_episode_poses`, `apply_body_delta`, `delta_to_nearest_primitive`, `clip_body_delta`
- Produces: per-episode JSONL of frames under `results/dagger/.../epXXX.jsonl`; LeRobot dataset dir `data/openfly_lerobot/b0_dagger_correction`

- [ ] **Step 1: Mock collector test**
  - MockBridge + ReplayPolicy + PathExpert on a short polyline
  - Force takeover by starting offset > takeover_m
  - Assert every saved frame has finite expert action and `intervention` bool
  - Assert abort path truncates distorted tail when configured

- [ ] **Step 2: Implement collector loop**

For each step:
1. `rgb, state = bridge.render(), bridge.state()`
2. `label = expert.label(pos, yaw)`
3. `policy_delta = policy.predict_delta(...)` (or primitive→delta)
4. `decision = controller.step(label.cross_track_m, label.progress_m)`
5. If abort: stop episode, keep frames so far
6. `executed = label.action if decision.mode=="expert" else policy_delta`
7. Save record with continuous **expert** action as training label
8. Execute via `bridge.step_delta(executed)` (or primitive map only for bridge if required — prefer continuous `step_delta`)

Atomic manifest update after each episode: `{"completed": [...], "failed": [...], "thresholds": {...}}`.

- [ ] **Step 3: Writer**
  - Convert collected frames into LeRobot v2.1 matching `convert_openfly_to_lerobot.py` schema (`observation.images.ego`, `observation.state`, `action` 4D, `task`)
  - Reuse existing FPS=10 / action_source=`pos_delta_v1`
  - Validate length equality and finite actions before write

- [ ] **Step 4: Tests PASS + commit** `feat(aerial): add DAgger collector and correction LeRobot writer`

---

### Task 5: Oracle / pilot gates and remote collection scripts

**Files:**
- Create: `experiments/aerial/eval/run_oracle_gate.py`
- Create: `experiments/aerial/scripts/wait_videos_then_collect.sh`
- Create: `experiments/aerial/scripts/deploy_collection_source.md` (ops checklist only; no secrets)

**Interfaces:**
- Produces: oracle JSON report with `SR`, `median_NE`, `projection_failures`, `cross_track_p95`; frozen `TakeoverConfig` written into collection manifest

- [ ] **Step 1: Implement oracle gate CLI** using PathExpert-only control on collection-40
  - Pass if `SR >= 0.80` and `median_NE < 20` and `projection_failures == 0`
  - Write `oracle_gate.json`

- [ ] **Step 2: Pilot on first 10 collection routes**
  - Run oracle + B0 shadow (label-only) to record cross-track P95
  - Freeze thresholds via `freeze_thresholds` into manifest

- [ ] **Step 3: Shell helper**
  - Poll `/tmp/aerial_eval_cache/logs/eval/b0_seen_videos.status` until `COMPLETED` or `FAILED`
  - Refuse to start if collection source missing
  - Launch `collect_dagger.py` with B0 `step_004000.pt`, collection-40 ann, frozen thresholds

- [ ] **Step 4: Commit** `feat(aerial): add oracle gate and collection launch scripts`

**Ops note (blocking):** Before remote run, deploy `seen_airsim16_collection_source.json` to eval H100 Annotation path and record SHA256 in the collection manifest. Do not fabricate from held-out.

---

### Task 6: 75/25 weighted source dataset

**Files:**
- Create: `src/fastwam/datasets/lerobot/weighted_source_dataset.py`
- Create: `tests/datasets/test_weighted_source_dataset.py` (or under `experiments/aerial/tests/` if repo prefers)
- Create: `configs/data/aerial_openfly_b0_ft_mix.yaml`
- Create: `configs/task/aerial_joint_b0_ft_dagger.yaml`

**Interfaces:**
- Produces: `class WeightedSourceDataset(Dataset)` wrapping two datasets with `source_probs=(0.75, 0.25)` and `pop_source_counts() -> dict[str,int]`

- [ ] **Step 1: Failing statistical test**
  - Two tiny fake datasets lengths 100/100
  - Draw 2000 indices with fixed torch Generator seed
  - Correction hit rate in 0.20–0.30

- [ ] **Step 2: Implementation**

```python
class WeightedSourceDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: list[torch.utils.data.Dataset], probs: list[float], names: list[str], generator=None):
        assert abs(sum(probs) - 1.0) < 1e-6
        self.datasets = datasets
        self.probs = probs
        self.names = names
        self.generator = generator or torch.Generator().manual_seed(0)
        self.counts = {n: 0 for n in names}
    def __len__(self):
        return sum(len(d) for d in self.datasets)
    def __getitem__(self, _idx):
        src = torch.multinomial(torch.tensor(self.probs), 1, generator=self.generator).item()
        self.counts[self.names[src]] += 1
        j = torch.randint(0, len(self.datasets[src]), (1,), generator=self.generator).item()
        sample = self.datasets[src][j]
        sample["data_source"] = self.names[src]
        return sample
```

Wire into data config by a thin `RobotVideoDataset` subclass or Hydra `_target_` that builds two `RobotVideoDataset`s (original + correction) then wraps them. Prefer a dedicated `_target_` under `experiments/aerial/ft_mix_dataset.py` to avoid breaking MultiLeRobot concat semantics (concat ≠ 75/25).

- [ ] **Step 3: Task config**

```yaml
# configs/task/aerial_joint_b0_ft_dagger.yaml
defaults:
  - override /data: aerial_openfly_b0_ft_mix
  - override /model: fastwam_joint
  - _self_
batch_size: 1
model:
  mot_checkpoint_mixed_attn: true
  loss:
    lambda_video: 0.0
    lambda_action: 1.0
learning_rate: 1.0e-5
lr_scheduler_type: cosine
max_steps: 1000
log_every: 50
save_every: 250
eval_every: 0
warmup_steps: 50
gradient_accumulation_steps: 1
weight_decay: 1.0e-2
seed: 42
```

Trainer must log `data_source` counts every 50 steps; fail run if any complete 200-step window correction rate ∉ [0.20, 0.30]. If trainer hooks are hard, add a small callback in `experiments/aerial/ft_source_monitor.py` invoked from a thin train wrapper script.

- [ ] **Step 4: Tests PASS + commit** `feat(aerial): add 75/25 weighted FT data mix`

---

### Task 7: Dual-4090 sync, smoke, and fine-tune

**Files:**
- Create: `experiments/aerial/scripts/sync_b0_ft_to_4090.sh`
- Create: `experiments/aerial/scripts/smoke_b0_ft_4090.sh`
- Create: `experiments/aerial/scripts/run_b0_ft_4090.sh`
- Create: `experiments/aerial/scripts/accelerate_zero2_opt_offload_2proc.yaml` (copy/adapt from train H100)

**Interfaces:**
- Sync SHA256 manifest listing: `step_004000.pt`, `dataset_stats.json`, train_subset, correction set, text embeds, code tarball/commit, Hydra configs, DeepSpeed/Accelerate yaml, collection manifest

- [ ] **Step 1: Sync script**
  - `rsync`/scp from eval/train caches to `/tmp/aerial_ft_cache/` on `:30879`
  - Verify SHA256 list; refuse smoke if mismatch

- [ ] **Step 2: Smoke**
  - `max_steps=1` then `max_steps=10`
  - Accelerate config: DeepSpeed ZeRO-2, `offload_optimizer_device: cpu`, `num_processes: 2`, bf16
  - Pass if peak mem < 23GiB/GPU and losses finite
  - On OOM: one retry with smaller reduce/allgather buckets; second failure → STOP and escalate to ZeRO-3 review (do not keep retrying)

- [ ] **Step 3: Full FT**
  - Resume **model weights only** from `step_004000.pt` (no optimizer state)
  - Save `step_000250.pt`, `step_000500.pt`, `step_001000.pt`
  - Write `ft.status` with RUNNING/COMPLETED/FAILED

- [ ] **Step 4: Commit scripts** `feat(aerial): add dual-4090 B0 FT sync and smoke runners`

---

### Task 8: Held-out compare report and S1 gate

**Files:**
- Create: `experiments/aerial/eval/compare_finetune.py`
- Create: `experiments/aerial/tests/test_compare_finetune.py`
- Create: `experiments/aerial/scripts/eval_ft_ckpts_seen20.sh`

**Interfaces:**
- Consumes baseline `step_004000.json` and FT result JSONs
- Produces `ft_selection_report.json` with mean/median NE, SR/SPL, per-episode deltas, improve/flat/regress counts, quantization gap stats if present
- Exit code 0 only if best mean NE ≤ `108.75649833236835`

- [ ] **Step 1: Unit test** on fake metrics dicts

```python
def test_s1_pass_and_fail():
    report = summarize(baseline_ne=135.94562291546043, cand={"250": 120.0, "500": 100.0})
    assert report["best_step"] == "500"
    assert report["s1_pass"] is True
```

- [ ] **Step 2: Eval script**
  - Run `run_closed_loop` for FT ckpts 250/500/1000 on held-out seen-20, seed 42, max_steps 100, task `aerial_joint_b0_novideo` (or FT task if identical action-only)
  - Record continuous vs primitive L2 gap counters during eval if policy exposes both
  - Call `compare_finetune.py` against locked baseline

- [ ] **Step 3: If S1 fails**, write diagnosis scaffold listing failure bins; do **not** auto-expand data or start unseen

- [ ] **Step 4: Commit** `feat(aerial): add FT held-out compare and S1 gate`

---

## Self-Review (spec coverage)

| Spec section | Task |
|--------------|------|
| §1 S1 NE lock / step_4000 | Task 7–8 |
| §2 machine split | Global + Task 5/7 |
| §3 leak-free collection files | Task 1, Task 5 ops note |
| §4 path expert + oracle gates | Task 2, Task 5 |
| §5 threshold freeze + takeover (no NE trigger) | Task 3, Task 5 |
| §6 correction dataset + 20–30% hit rate | Task 4, Task 6 |
| §7 4090 ZeRO-2 opt-offload + SHA sync + smoke | Task 7 |
| §8 compare + quantization gap | Task 8 |
| §9 schedule / video wait | Task 5 script |
| §10 no auto-expand on fail | Task 8 Step 3 |

No TBD/placeholder steps remain. Weighted mix intentionally avoids `MultiLeRobotDataset` concat (would not enforce 75/25).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-24-aerial-b0-failure-replay-finetune.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute in this session with checkpoints  

Which approach?
