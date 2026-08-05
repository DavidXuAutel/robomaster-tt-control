# DepthHead capacity lift — base 32→64 (DA-V2 teacher unavailable offline)

**Date:** 2026-08-05  
**Branch:** `aerial-rl-skeleton`  
**Frozen-spec:** untouched (`[1b]` multi-frame DepthHead spirit unchanged; `enable` stays false)

## Why not DA-V2 teacher distillation

Preferred path was offline DA-V2 soft-target distillation (analysis priority 4).
Probe on H100 `:31126` + local Mac (2026-08-05):

| Asset | Present? |
|-------|----------|
| `depth-anything/Depth-Anything-V2-Small-hf` (TT `da_v2_service` MODEL_ID) | **No** — not in `~/.cache/huggingface/hub`, aerial caches, or Mac HF cache |
| `models--depth-anything--Video-Depth-Anything-Large` (2.9G ViT-L) | Yes under FastWAM scoutxwam hub — **video** teacher, not DA-V2-Small; wiring would be a separate third_party integration |
| Auto-download | **Forbidden** (`HF_HUB_OFFLINE=1`; missing → error stop) |

So teacher distill cannot run without a download. Fallback per task brief: **widen student**.

## Capacity change

- Architecture already parameterized: `_DepthHead(..., base=)`.
- Canonical AbsRel-PASS `depth_step_5000.pt` is **base=32** (~holdout AbsRel 0.281 on merge).
- Capacity lift: CLI `--base 64` (yaml default stays **32** so init-ckpt from canonical still matches).
- Wide ckpts save as `depth_step_*_base64.pt` (stem suffix when `base≠32`) — never overwrite base-32 canonical.
- Loading base-32 weights into base-64 refuses with a clear `arch mismatch` FAIL (`strict=True`).
- Train from scratch on `/tmp/dataset_v0_merged_local_approach` with clean recipe:
  OS=1, `grad_clip=5`, `delta_weight=0` Stage A, lr≈1e-4, `--eval-every 100`,
  `--split-seed 0 --holdout-frac 0.2 --window 8`, CUDA≠0, artifacts under `/tmp`.
- Optional Stage B (δ=0.05 OS=1 low lr) only if Stage A AbsRel≤0.30.
- Gate `--signals 1,3`; promote only AbsRel≤0.30 **and** D̂≤0.25; `enable` stays false.

## Promote rule (unchanged)

Overwrite canonical only if AbsRel≤0.30 **and** D̂≤0.25. Else keep candidate under `/tmp` / dated artifact dir; `depth_head.enable` remains false. Keep old base-32 canonical as archive.

## H100 run @ `01111be` (2026-08-05)

| item | value |
|------|-------|
| recipe | `--base 64` from scratch, OS=1, δ=0, lr=1e-4, grad_clip=5, steps=5000, eval-every=500, split-seed=0 |
| CUDA | `CUDA_VISIBLE_DEVICES=1` |
| artifacts | `/tmp/depth_capacity_base64/` → `experiments/aerial/rl/artifacts/depth_ckpt_capacity_base64_20260805/` |
| holdout AbsRel trajectory | 500:0.481 → 1k:0.362 → 2k:0.316 → 3.5k:0.292 → 4.5k:0.273 → **5k:0.242** |
| gate ①d | **0.2418 PASS** (11 holdout eps) |
| gate ③ D̂ | **0.292 FAIL** (n_valid=95; ≤0.25) |
| GT oracle (diagnose) | all-motion / forward-only median_rel **0.229 PASS** (n=167) |
| promote | **No** — AbsRel cleared with headroom vs canonical ~0.28/0.30, but D̂ still above 0.25; canonical `depth_step_5000.pt` (base=32) untouched; `enable` stays false |

Capacity lift worked for AbsRel. Remaining ③ gap is scale-change teaching (δ / temporal), not width alone — diagnose: “GT passes but D̂ fails → retrain with temporal / Δ-depth”.
