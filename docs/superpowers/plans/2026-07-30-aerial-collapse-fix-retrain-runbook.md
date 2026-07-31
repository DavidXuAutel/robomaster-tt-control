# Aerial B0 v2 Collapse-Fix — Retrain Runbook (Implementation Plan)

> **For agentic workers:** This is the executable retrain plan that follows
> `plans/2026-07-30-aerial-b0-v2-collapse-fix.md` (the design blueprint). The
> collapse-fix *code* (scheme-B `head_cls`, CE loss, conversion-time stop
> relabel, Stage-1 data config) is already implemented and unit-tested; this
> doc is the **host-side execution sequence** to turn that code into a trained,
> terminating policy. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain b0_v2 with scheme-B classification + stop supervision so the
policy actually terminates (root cause #1) and stops being goal-blind (root
cause #2), then eval with learned stop.

**Blueprint:** `docs/superpowers/plans/2026-07-30-aerial-b0-v2-collapse-fix.md`
**Spec:** `docs/superpowers/specs/2026-07-30-aerial-b0-v2-collapse-fix-design-v3.2.md`
**Launcher:** `experiments/aerial/scripts/run_collapse_fix_retrain.sh`
**Hosts:** train `:31126` (2×H100), eval `:30905` (OpenFly+AirSim).

---

## Preconditions & Invariants (the three points — READ FIRST)

These are load-bearing. Getting any wrong reproduces the b0_v2 SR=0 failure.

1. **Opt-in / default no-op.** The collapse-fix changes do not alter current
   training behavior. `configs/data/aerial_openfly.yaml`,
   `configs/task/aerial_joint_1cam_1e-4.yaml`, and the raw
   `./data/openfly_lerobot/train_subset` are untouched.
   `convert_openfly_to_lerobot.py` defaults `stop_relabel_radius=None` (raw
   deltas, bit-identical output — locked by `test_stop_relabel_none_is_noop`),
   and `loss_lambda_ce` defaults to 0. Nothing below fires unless you run the
   launcher.

2. **Training distribution/window changes come from TWO independent switches:**
   - **(a) Data:** re-convert with `--stop-relabel-radius 20` → injects `stop`
     (primitive 0) labels as zero body-deltas.
   - **(b) Recipe/window:** `task=aerial_joint_collapse_fix` → `num_frames=9`,
     `action_video_freq_ratio=2`, `skip_padding_as_possible=true`, and
     `λ_video=1.0 / λ_fm=0.1 / λ_ce=1.0` with `enable_action_cls=true`.

   They are orthogonal. **Flipping only (b)** changes the window but leaves stop
   labels at the ~2/2709 (~0.07%) floor → the CE head never learns to stop →
   policy still never terminates. **You need BOTH.**

3. **Retrain prerequisite:** `reconvert` MUST precede `train`. Without the
   relabeled dataset, `labels.relabel_stop_on_trajectory`'s in-pipeline path is
   still unwired (lacks per-frame pose/goal in the training `sample`), so stop
   supervision stays ≈2/2709 and the policy reproduces the b0_v2
   never-terminate failure.

**Normalizer note (consequence of #2a):** re-converting recomputes the
`FastWAMProcessor` min/max normalization over the new zero-heavy action
distribution (many `[0,0,0,0]` rows compress the action min/max span). The
relabeled subset's normalizer is therefore **not interchangeable** with old
b0_v2 checkpoints. This retrain is from scratch, so that is expected and
harmless — but never point an old ckpt at the relabeled data for closed-loop
eval; the denormalized actions would be wrong.

---

## File map

| Path | Role in retrain |
|------|-----------------|
| `experiments/aerial/scripts/run_collapse_fix_retrain.sh` | reconvert → preflight → smoke → train launcher (this plan's driver) |
| `experiments/aerial/convert_openfly_to_lerobot.py` | `--stop-relabel-radius` stop relabel (invariant #2a / #3) |
| `configs/task/aerial_joint_collapse_fix.yaml` | λ recipe + `override /data: aerial_openfly_collapse_fix` |
| `configs/data/aerial_openfly_collapse_fix.yaml` | `num_frames=9`, `ratio=2`, `skip_padding` (invariant #2b) |
| `src/fastwam/models/wan22/action_dit.py` | `head_cls` + `classify_from_tokens` (enabled via override) |
| `src/fastwam/models/wan22/fastwam.py` | CE branch (`loss_lambda_ce>0` + `head_cls`) |
| `experiments/aerial/eval/policy_fastwam.py` | closed-loop `argmax(cls)` when `primitive` present |
| `experiments/aerial/collapse_fix/compute_dmax.py` | informational d_max (see §d_max) |

---

### Task 1: Regenerate the relabeled dataset (host `:31126`)

Non-destructive: writes to `train_subset_stop20`, leaving the raw subset intact.

```bash
OPENFLY_ANN=<path/to/annotation.json> \
OPENFLY_IMAGE_ROOT=<path/to/images> \
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh reconvert
```

- [ ] **Step 1:** Run `reconvert`; confirm `./data/openfly_lerobot/train_subset_stop20` is created.
- [ ] **Step 2:** Sanity: the raw `train_subset` is byte-unchanged (invariant #1).
- [ ] **Step 3:** (optional) eyeball a few episodes — final + near-goal frames should carry `action==[0,0,0,0]`.

> `R=20.0` matches `OPENFLY_SUCCESS_DIST_M` (`experiments/aerial/eval/metrics.py`).
> Override with `STOP_RADIUS=<m>` if the eval threshold changes. Re-run with
> `FORCE=1` to overwrite an existing relabeled dir.

---

### Task 2: Preflight + smoke (host `:31126`)

```bash
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh preflight
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh smoke
```

- [ ] **Step 1:** Preflight passes: relabeled data present, 2 GPUs, `resume=null`, `head_cls` wiring detected, `RESUME`/`AERIAL_ALLOW_LEGACY_RESUME` empty.
- [ ] **Step 2:** Smoke (`SMOKE_STEPS=10`) prints **finite** `loss_video`, `loss_action` (fm), and **`loss_ce`** — the presence of a non-trivial `loss_ce` is the signal the CE head is actually wired and receiving stop labels.
- [ ] **Step 3:** No NaN; peak mem < 90%. Record steps/s → set `MAX_STEPS`.

> The launcher passes the recipe explicitly (`enable_action_cls=true`,
> `λ_video/λ_fm/λ_ce`, and `data.train.dataset_dirs=[…train_subset_stop20]`) on
> top of the task config, so the override chain is self-documenting in the log.

---

### Task 3: Full retrain (host `:31126`)

```bash
MAX_STEPS=<from smoke> SAVE_EVERY=500 \
bash experiments/aerial/scripts/run_collapse_fix_retrain.sh train
```

- [ ] **Step 1:** Launch full run; checkpoints land under `checkpoints/weights/step_*.pt`.
- [ ] **Step 2:** Watch `loss_ce` decrease and the predicted-primitive histogram broaden away from the fwd-only collapse (log/plot via `plot_loss.py`).
- [ ] **Step 3:** Target ≤5 epoch-equivalents on the current subset; expand data only in a later stage.

---

### Task 4: Eval with learned stop (host `:30905`)

- [ ] **Step 1:** Enqueue ckpts with `ORACLE_STOP=0` (learned stop via `argmax(cls)`) + `closest_approach` diagnostics.
- [ ] **Step 2:** Compare SR / NE / SPL against b0_v2 (SR=0) and the Stage-0 oracle-stop ceiling (~10%).
- [ ] **Step 3:** Gate on v3.2 §1 hard-success criteria before locking a baseline.

---

## d_max (known limitation — not blocking retrain-1)

`compute_dmax.py` writes `artifacts/collapse_fix_dmax.json` (`d_max_p90_forward`),
and `fastwam.py:593` reads the threshold via
`getattr(self, "action_cls_d_max", 1e9)`. **But `action_cls_d_max` is not
plumbed through `create_fastwam_joint`**, so a `model.action_cls_d_max=<val>`
override has no effect today. For retrain-1 this means forward-class d_max
filtering is **OFF** (threshold 1e9); minority classes (stop + turns + vertical)
are exempt regardless, so stop supervision is unaffected. `preflight` runs
`compute_dmax` for information only.

- [ ] **Optional follow-up (host-verify):** add an `action_cls_d_max` param to
  `create_fastwam_joint` → set it as a `FastWAMJoint` attribute, then pass
  `model.action_cls_d_max=$(jq .d_max_p90_forward artifacts/collapse_fix_dmax.json)`
  in `launch()`. Only worth doing if forward-primitive label noise proves to
  hurt CE; otherwise leave inert.

---

## Done when

1. `train_subset_stop20` regenerated with stop labels; raw subset intact.
2. Smoke shows finite `loss_ce` (CE head receiving stop supervision).
3. Full retrain checkpoints saved; predicted-primitive distribution no longer fwd-collapsed.
4. `:30905` eval with `ORACLE_STOP=0` reports SR > 0 (beats b0_v2) with learned stop.

## Notes / risks

- Sandbox cannot reconvert data, reach GPUs, or SSH the hosts — Tasks 1–4 run
  by the user on `:31126` / `:30905`. This plan + launcher are the SOP.
- All pure-Python conversion/label logic is unit-tested locally
  (`test_convert_openfly_fixture.py`, `test_collapse_fix_labels.py`); torch
  paths (head_cls, CE, denoise) are host-verified.
- Do not mutate the base `aerial_openfly.yaml` / `aerial_joint_1cam_1e-4.yaml`
  or the raw `train_subset` — the collapse-fix path is fully parallel.
