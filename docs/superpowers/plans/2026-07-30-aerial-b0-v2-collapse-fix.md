# Aerial B0 v2 Collapse-Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute v3.2 collapse-fix: Stage 0 oracle-stop + probe runs, Stage 1 data hygiene, scheme-B `head_cls(t)` + stop relabel, then one merged retrain.

**Architecture:** Keep ActionDiT flow-matching (`head_fm`) and MoT denoise loop; add timestep-conditioned `head_cls` (reuse `ActionHead`) for 10-class CE; closed-loop executes `argmax(cls)`. Eval diagnostics already in `run_closed_loop` (`stage0-oracle-v1`).

**Tech Stack:** FastWAM / ActionDiT, Hydra configs, OpenFly+AirSim eval on `:30905`, train on `:31126`.

**Spec:** `docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design-v3.2.md`

## Global Constraints

- Scope: aerial-wam only; do not touch TT orbit / true-aircraft avoidance.
- Scheme B only: never remove action denoise loop (no scheme A).
- `head_cls` must be timestep-conditioned; `t_final` = last `_predict_joint_noise` `timestep_action`.
- Default pool: `action_tokens[:, 0, :]` (first horizon step).
- `λ_ce` primary; `λ_fm` small default (e.g. 0.1); `λ_video > 0`.
- Inference: `argmax(cls)` only — no nearest-primitive for control.
- Stage 0 probe mandatory before Stage 3 full CFG/dropout.
- No same-distribution bare data scale-up before Stages 1–3.
- Runtime code lives in aerial-wam worktree / `aerial-wam` branch checkout on hosts.

## File map

| Path | Responsibility |
|------|----------------|
| `experiments/aerial/collapse_fix/labels.py` | nearest-prim labels, `d_max`, stop relabel, minority exemption |
| `experiments/aerial/collapse_fix/probe_verdict.py` | probe sensitivity → Stage 3 branch |
| `experiments/aerial/scripts/stage0_oracle_eval.sh` | enqueue/run oracle-stop eval on 3 ckpts |
| `experiments/aerial/scripts/stage0_instruction_probe.sh` | run probe on 2 ckpts |
| `experiments/aerial/scripts/run_collapse_fix_smoke.sh` | local unit + optional 1-step train smoke |
| `configs/data/aerial_openfly_collapse_fix.yaml` | `num_frames=9`, `action_video_freq_ratio=2`, `skip_padding_as_possible=true` |
| `configs/task/aerial_joint_collapse_fix.yaml` | λ_ce / λ_fm / λ_video recipe |
| `src/fastwam/models/wan22/action_dit.py` | `head_cls` + `classify_from_tokens` |
| `src/fastwam/models/wan22/fastwam.py` | CE in `training_loss` |
| `src/fastwam/models/wan22/fastwam_joint.py` | expose last-step tokens/`t` for classify |
| `experiments/aerial/eval/policy_fastwam.py` | prefer `argmax(cls)` when available |

---

### Task 1: Label + probe utilities (CPU, no GPU)

**Files:**
- Create: `experiments/aerial/collapse_fix/labels.py`
- Create: `experiments/aerial/collapse_fix/probe_verdict.py`
- Create: `experiments/aerial/collapse_fix/__init__.py`
- Test: `experiments/aerial/tests/test_collapse_fix_labels.py`

**Interfaces:**
- Produces: `delta_nearest_with_dist`, `build_ce_mask`, `relabel_stop_on_trajectory`, `probe_sensitivity_verdict`

- [x] **Step 1:** Implement labels + verdict helpers per v3.2 §0.4.1 / Stage 0.
- [x] **Step 2:** Unit tests for stop relabel, minority `d_max` exemption, probe three-way verdict.
- [x] **Step 3:** `pytest experiments/aerial/tests/test_collapse_fix_labels.py -q` (7 passed 2026-07-30)

---

### Task 2: Stage 0 runnable scripts

**Files:**
- Create: `experiments/aerial/scripts/stage0_oracle_eval.sh`
- Create: `experiments/aerial/scripts/stage0_instruction_probe.sh`
- Create: `experiments/aerial/collapse_fix/probe_instructions.txt`
- Create: `artifacts/b0_v2_20260729-b0v2-10k-2gpu/run_stage0_from_mac.sh` (SSH wrapper)

**Run (eval host `:30905`):**
```bash
ORACLE_STOP=1 bash experiments/aerial/scripts/stage0_oracle_eval.sh
bash experiments/aerial/scripts/stage0_instruction_probe.sh
```

- [ ] **Step 1:** Scripts default to ckpts `step_{000500,001500,003500}`, ann `seen_airsim16_m1a20.json`, write under `.../results/b0_v2_stage0_oracle/`.
- [ ] **Step 2:** Probe script writes `summary.json` + runs `probe_verdict.py` → `verdict.json`.
- [ ] **Step 3:** Mac wrapper SSHes with existing askpass pattern.

---

### Task 3: Data hygiene config (Stage 1)

**Files:**
- Create: `configs/data/aerial_openfly_collapse_fix.yaml` (`num_frames: 9`, `action_video_freq_ratio: 2`, `skip_padding_as_possible: true`)
- Create: `configs/task/aerial_joint_collapse_fix.yaml`
- Create: `experiments/aerial/collapse_fix/compute_dmax.py` (stats over train_subset)

- [x] **Step 1:** New Hydra data/task configs; do not mutate production `aerial_openfly.yaml` in place until recipe locks.
- [x] **Step 2:** `compute_dmax.py` prints p90 + per-class dists; writes `d_max.json`.

> **⚠️ `num_frames=9` forces `action_video_freq_ratio=2`, not 4.** `RobotVideoDataset`
> asserts BOTH `(num_frames-1) % ratio == 0` AND `((num_frames-1)//ratio) % 4 == 0`
> (`robot_video_dataset.py:59-62`). With `ratio=4`: `(9-1)//4 = 2`, `2%4 ≠ 0` → hard
> crash at dataset init (`ratio=4` only admits `num_frames ∈ {17,33,…}`). `ratio=2`
> gives `8%2=0`, `8//2=4`, `4%4=0`, and 5 video frames (indices 0,2,4,6,8) with
> `5%4==1` for the VAE. b0_v2 trains from scratch, so the ratio change costs no
> pretrained action weights. **Fixed 2026-07-30** in `aerial_openfly_collapse_fix.yaml`
> (the initial draft left `ratio=4` and would have crashed on the host).
>
> **`d_max` wiring:** `compute_dmax.py` writes `artifacts/collapse_fix_dmax.json`
> (`d_max_p90_forward`). That value must be fed to training via
> `model.action_cls_d_max=<val>` (read by `fastwam.py:593` as
> `getattr(self, "action_cls_d_max", 1e9)`). **Host-gated:** requires the parquet
> data. Until it lands, `d_max` defaults to 1e9 (no forward filtering; minority
> classes are exempt regardless).

---

### Task 4: ActionDiT `head_cls` (scheme B)

**Files:**
- Modify: `src/fastwam/models/wan22/action_dit.py`
- Test: `experiments/aerial/tests/test_action_cls_head.py`

**Interfaces:**
- `ActionDiT(..., enable_action_cls: bool = False)`
- `classify_from_tokens(tokens, pre_state) -> logits [B,10]` using `tokens[:,0,:]` + `pre_state["t"]`
- `post_dit` still returns FM velocity via `self.head`

- [x] **Step 1:** Failing test: classify shape + requires `t`.
- [x] **Step 2:** Implement `head_cls = ActionHead(H, 10, eps)` when enabled; skip in backbone load prefixes (`action_dit.py:33,104-108,314-330`).
- [x] **Step 3:** Tests green (host-verified only — torch not on Mac; label/probe tests green locally).

---

### Task 5: Training CE + infer classify path

**Files:**
- Modify: `src/fastwam/models/wan22/fastwam.py` (`loss_lambda_ce`, CE branch)
- Modify: `src/fastwam/models/wan22/fastwam_joint.py` (cache last action tokens/`t`)
- Modify: `experiments/aerial/eval/policy_fastwam.py`

- [x] **Step 1:** When `enable_action_cls` and `loss_lambda_ce > 0`, add CE on first-step prim id with `build_ce_mask` (`fastwam.py:569-600`).
- [x] **Step 2:** Infer: last denoise step → `classify_from_tokens` → policy `predict_primitive` uses argmax if logits present (`fastwam_joint.py:238-242`, `policy_fastwam.py:147-148,166-168`).
- [x] **Step 3:** Default `loss_lambda_ce=0` keeps old B0 recipe bit-compatible (`fastwam.py:42,90`).

> **✅ RESOLVED (2026-07-30) — stop relabel wired at conversion time (root cause #1 addressed).**
> Rather than thread raw positions/goal through `build_inputs` into `training_loss`
> (the training `sample` carries only normalized `proprio`/`action` — no explicit
> goal), stop supervision is now injected **at dataset-conversion time** in
> `experiments/aerial/convert_openfly_to_lerobot.py::convert_trajectory`
> (`stop_relabel_radius` param, `--stop-relabel-radius` CLI flag). When set, any
> emitted frame whose start position is within `R` metres of the episode goal
> (= end position of the last non-padding transition) — plus the terminal frame
> unconditionally — has its action zeroed to `[0,0,0,0]`.
>
> **Why this needs zero torch-side changes:** a zero body-delta is the OpenFly
> `stop` primitive (id 0), and the existing CE branch labels each sample via
> `delta_nearest_with_dist(action[:, 0, :])`, which maps `[0,0,0,0] → (0, 0.0)`.
> So the relabeled zeros flow through the *unchanged* CE pipeline as `stop`
> labels. Fully locally testable: `test_convert_openfly_fixture.py` covers
> near-goal zeroing, forced terminal stop, `None`-is-noop, negative-radius
> rejection, and the delta→primitive-0 round-trip (17 pure-Python tests green
> 2026-07-30; the one host-only failure needs the `datasets` pkg).
>
> **Host action required before retrain:** the existing `train_subset` lerobot
> dataset was converted WITHOUT relabel, so it must be **re-converted** with the
> flag before the merged retrain:
> ```
> python experiments/aerial/convert_openfly_to_lerobot.py \
>   --ann <ann.json> --image-root <images> --out ./data/openfly_lerobot/train_subset \
>   --stop-relabel-radius 20.0   # = OPENFLY_SUCCESS_DIST_M (eval/metrics.py)
> ```
> Recommended `R = 20.0` (matches `OPENFLY_SUCCESS_DIST_M`, the eval success
> threshold). Successful OpenFly demos end at the goal, so their final frames
> zero out naturally; `R` widens the stop band and the terminal-force guarantees
> at least one stop label per episode. This lifts natural `stop` prevalence far
> above the ~0.1% (2-sample) floor that caused the b0_v2 never-terminate failure.
>
> Note: `relabel_stop_on_trajectory` in `labels.py` remains as an alternative
> in-pipeline implementation but stays unwired — the conversion-time path
> supersedes it and is the one exercised by tests + retrain.

---

### Task 6: Smoke + Stage 0 execution checklist

**Files:**
- Create: `experiments/aerial/scripts/run_collapse_fix_smoke.sh`

- [ ] **Step 1:** Local pytest for collapse_fix + action_cls + existing closed_loop oracle tests.
- [ ] **Step 2:** On `:30905`, run Stage 0 scripts; sync metrics via existing `sync_eval_results.sh`.
- [ ] **Step 3:** Record probe verdict → decide Stage 3 intensity before any long train.

---

### Task 7: Merged retrain (after Stage 0 pass)

- Train on `:31126` with `task=aerial_joint_collapse_fix`, `max_steps` TBD from smoke throughput (target ≤5 epoch-equivalents on current subset, then expand data only in Stage 4).
- Enqueue ckpts with `ORACLE_STOP=0` for learned-stop eval + closest_approach diagnostics.
- Gate on v3.2 §1 hard success.

---

## Done when

1. Stage 0 metrics + probe verdicts on disk for 500/1500/3500.
2. Unit tests green for labels, probe verdict, `head_cls`.
3. Collapse-fix Hydra configs exist; smoke train 1–10 steps finite losses when cls enabled.
4. Plan checklist above completed or explicitly deferred with reason.
