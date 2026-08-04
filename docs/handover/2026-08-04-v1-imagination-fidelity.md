# V1 Imagination-Fidelity Criteria & Runbook

**Date:** 2026-08-04 · **Branch:** `aerial-rl-skeleton` · Follows the V1 WM gate
([validation runbook](2026-08-04-v1-wm-h100-validation-runbook.md), which
validated *learning* + *non-divergence*).

## Why this exists

The §7 pure-vision ladder defines **V1** as *three* things (spec lines 293-295):

| V1 sub-criterion | status |
|---|---|
| ① 多步 rollout **达标** (rollout meets criteria) | **this doc** — non-divergence was only the floor |
| ② 碰撞率相对 V0 **下降** | out of scope here — needs the short-horizon imagination *planner* (§4.5) |
| ③ τ/D̂ **双通道验证** | out of scope here — needs depth + optical-flow data (the open ~0.7 Hz depth issue; `dataset_v1_rgb` is RGB-only) |

`_wm_train_validate` proved the WM *learns* and its H=15 rollout is
*non-divergent* (§9). "达标" is stronger than "not diverging": the imagined
trajectory must **track** the real one. This doc pins that bar and the eval that
measures it. ②/③ are explicitly deferred — do not read a fidelity PASS as full
§7-V1 closure.

## The pass口径 (baseline-relative + bounded-growth)

Our 4-D kinematic SEARCH regime has **no paper number** (design-doc §1.5 — do
NOT import DreamerV3's T=16 / threshold-50). So the bar is relative to trivial
baselines and to error growth, never an absolute tolerance. Thresholds live at
the top of [`wm_eval.py`](../../experiments/aerial/rl/wm_eval.py) as named,
project-tuned constants.

| Head | Metric | PASS condition |
|---|---|---|
| **reward/progress** | per-horizon MAE of the open-loop reward readout vs recorded reward | beats the constant-mean baseline on ≥ `REWARD_BEAT_FRAC` (0.8) of horizons, 1-step error below baseline, AND `growth_bounded` (err(H) ≤ err(1)·(1+slope·(H−1)), slope=1 ⇒ no worse than ~linear) |
| **p_coll** | trajectory-level AUROC: label = did the trajectory collide, score = max predicted p_coll over the horizon | AUROC ≥ `PCOLL_AUROC_MIN` (0.65); NaN (a class absent in the split) is *insufficient signal* → FAIL, not a pass |
| **done/continue** | per-step done accuracy over valid steps | beats the majority-class baseline (`DONE_ACC_MARGIN`=0) |
| **decoder recon** (train-only, §2.3) | multi-step open-loop decode MSE vs the real frame at each depth | `growth_bounded` — the imagined latent still carries the scene H steps out |

Overall PASS = all four. The reward baseline is the **constant-mean** predictor
(reward is a Δ-distance signal, not a level; the persistence baseline degenerates
at t=0, so it is logged for reference but not gated).

**Alignment / ±1 timing:** rollout step `t` calls `step(z_t, a_t)` and its heads
are compared to the recorded consequence at `window[t]`. Exactly when a contact
is *labeled* vs *predicted* has a ±1-step ambiguity; the p_coll metric is
trajectory-level (max over the horizon) precisely so it is robust to that.

## Held-out discipline (important)

The §3 checkpoint `wm_step_5000.pt` was trained on **all** episodes. Running the
eval with `--heldout-frac 0` therefore measures **in-sample** fidelity — a lower
bound only ("if it fails in-sample it definitely fails"). For an **honest** gate:

1. Retrain the WM with a held-out tail excluded (same split the eval will use).
2. Eval here with `--heldout-frac` matching that split.

The script logs loudly which regime it ran in and tags an in-sample PASS as
"lower bound, not the honest gate".

## Run it (H100)

```bash
export PYTHONPATH="$PWD"
python -m experiments.aerial.rl._wm_fidelity_eval \
  --dataset /home/a25689/rl_collect_run/.../artifacts/dataset_v1_rgb \
  --ckpt    experiments/aerial/rl/artifacts/wm_ckpt/wm_step_5000.pt \
  --config  configs/aerial_rl.yaml \
  --heldout-frac 0.25 --horizon 15 --n-starts 2
```

Prints a per-horizon table (`wm_mae | mean-base | persist | recon_mse`) and a
per-head OK/FAIL line, then `PASS`/`FAIL` (exit 0/1).

- **In-sample first look:** `--heldout-frac 0` — quick lower-bound sanity on the
  existing checkpoint before spending a retrain.
- **Honest gate:** retrain with the held-out split, then `--heldout-frac 0.25`.

**If reward FAILs:** likely under-training or window length — raise §3 `--steps`
or `--window`; tune the `world_model:` block, not the recipe math.
**If recon growth FAILs but reward/coll/done pass:** the heads are fine but the
latent loses scene content deep in the rollout — shorten the effective planning
horizon (< 15) until it holds; this directly informs the V4 imagination depth.

## Local (dev host)

The metric math + harness are pure numpy and unit-tested off-GPU:

```bash
python -m pytest experiments/aerial/rl/tests/test_wm_eval.py -q
```

(The torch checkpoint eval itself only runs on the H100.)
