# Proposal: Signal ③ relative margin (vs absolute `scale_rel_err_max=0.25`)

**Status:** proposal for adjudication — **not** a frozen-spec change  
**Date:** 2026-08-05  
**Audience:** owner / Claude (spec editor)  
**Related:** `docs/superpowers/specs/2026-08-04-aerial-wam-v2-frozen-spec.md` § signal ③; P1 AbsRel pipeline diag ([Priority-1 AbsRel pipeline diag](867621e9-b89f-4e3f-80e5-81ce94c2d9c2)); AbsRel-safe train fixes @ `71f09eb`

---

## Hard constraint (read first)

- **`docs/superpowers/specs/2026-08-04-aerial-wam-v2-frozen-spec.md` is Claude-only** (chmod 444). Non-Claude agents must **not** edit it, chmod it, or silently “implement” a threshold change.
- This note is a **dev-note proposal only**. Any change to `scale_rel_err_max`, relative-margin language, or gate wiring requires **Claude adjudication + explicit frozen-spec edit**.
- Do **not** flip `world_model.depth_head.enable` or overwrite the H100 canonical depth ckpt as part of exploring this proposal.

---

## Problem: absolute 0.25 is knife-edge vs GT oracle

Frozen-spec ③b keeps an **absolute** gate:

| knob | value | meaning |
|------|-------|---------|
| `scale_rel_err_max` | **0.25** | median \|ŝ_D − s_VIO\| / max(s_VIO, ε) |

After the 2026-08-05 protocol revision (nav-band [1,40] m, `fwd_cos_min=0.7`, `scale_support_ratio=0.6`, `min_scale_windows≥8`), `--signal3-diagnose` made **GT oracle** reachable on resize/approach corpora (spec note: med ≈ 0.21; measured GT-oracle med ≈ **0.229** on the diagnose path used for D̂ attribution).

That leaves only ~**0.02** headroom to the absolute 0.25 bar:

```
GT_oracle ≈ 0.229
absolute gate     = 0.25
slack             ≈ 0.021   ← knife-edge
```

So a depth head that is only modestly worse than GT on the *same* windows can FAIL ③ even when the failure mode is “estimator noise on an already-tight oracle,” not “broken scale / wrong geometry.” Absolute 0.25 was left intentionally nail-down when the protocol filters were fixed; that was correct then, but it makes D̂ grading brittle once AbsRel/Δ training actually moves the needle.

---

## Proposal (two alternatives; pick one or combine)

### A — Relative margin to GT oracle

Require predicted scale error to stay within a factor of the GT-oracle error on the **same** forward / support windows:

\[
\hat{D}_{\mathrm{med}} \le k \cdot \mathrm{GT\_oracle}_{\mathrm{med}}
\quad\text{with suggested } k \approx 1.3
\]

Example: if GT_oracle = 0.229, then D̂ must be ≤ ~0.298 (still comparable to today’s AbsRel ①d band), rather than racing a fixed 0.25 that GT itself almost fills.

**Pros:** self-calibrates when protocol/corpus shift the oracle; preserves “don’t beat physics” intent.  
**Cons:** needs diagnose/oracle path in the gated run (or a cached oracle med); k itself becomes a policy knob.

### B — Absolute gate only after oracle is healthy

Keep `scale_rel_err_max=0.25`, but apply it **only if** GT_oracle_med ≤ **0.15** (illustrative). If oracle is worse than 0.15, treat absolute ③b as **SKIP / protocol FAIL** (recollect or revisit filters) instead of blaming D̂ — or fall back to relative margin (A).

**Pros:** absolute bar stays meaningful when the corpus actually has slack.  
**Cons:** introduces a second threshold; needs clear FAIL vs SKIP semantics so bring-up doesn’t thrash.

**Recommended default for adjudication:** adopt **A** (k≈1.3) as the D̂ vs oracle grade, and keep absolute 0.25 as a **soft ceiling / report-only** until oracle_med ≤ 0.15 (blend of A+B). Exact numbers are for Claude to nail in the frozen-spec if accepted.

---

## Why wait / remeasure: P1 AbsRel pipeline findings

P1 ([Priority-1 AbsRel pipeline diag](867621e9-b89f-4e3f-80e5-81ce94c2d9c2)) showed the merged-corpus AbsRel “collapse” was **mostly training-pipeline**, not a hard capacity wall:

| finding | evidence (merged local+approach, canonical init) |
|---------|---------------------------------------------------|
| Eval path OK | 0-step canonical gate AbsRel **0.281094** bit-identical to reference |
| OS=4 inflates AbsRel | Same recipe 200-step: OS=1 → **0.2991 PASS**; OS=4 → **0.3130 FAIL** |
| Sampler dominates long drift | 1000-step paired: OS=1 **0.3118** vs OS=4 **0.3619** (~62% of extra drift) |
| Residual ceiling still real | Even OS=1 + lr 5e-6 drifts past 0.30 by ~1k steps (slow, smooth) |

Follow-up `71f09eb` shipped AbsRel-preserving train controls (`--approach-oversample`, effective `grad_clip=5`, `--eval-every` / `holdout_absrel` vs `train_batch_absrel`).

**Implication for this proposal:** do **not** retune `scale_rel_err_max` off of Δ-finetune numbers collected under **OS=4** / no-op clip. After the pipeline fix, **remeasure**:

1. Signal ①d holdout AbsRel under OS=1 AbsRel-preserving FT (verify ~0.29–0.30 @ 200 steps).  
2. Signal ③ diagnose: GT_oracle_med vs D̂_med on the same approach-biased corpus.  
3. Only then decide whether absolute 0.25, relative k≈1.3, or oracle≤0.15 gating is the right frozen rule.

Until that remeasure, treat any “③ FAIL under D̂” as **contaminated by prior sampler inflation** unless the ckpt was trained with OS=1 (AbsRel phase) and logged `holdout_absrel`.

---

## Explicit non-goals (this note)

- No edit to the frozen-spec file.  
- No yaml `enable: true`.  
- No canonical ckpt overwrite.  
- No claim that k=1.3 or oracle≤0.15 are final — they are **starting points for Claude**.

---

## Suggested adjudication checklist (Claude)

- [ ] Confirm current GT_oracle_med on the merged approach corpus under the 2026-08-05 protocol (expect ~0.21–0.229).  
- [ ] After AbsRel-safe FT (`71f09eb` recipe), remeasure D̂_med and slack to 0.25.  
- [ ] Choose A / B / A+B; write exact PASS/FAIL/SKIP into frozen-spec if accepted.  
- [ ] Keep ①d AbsRel ≤ 0.30 independent unless jointly redesigned.
