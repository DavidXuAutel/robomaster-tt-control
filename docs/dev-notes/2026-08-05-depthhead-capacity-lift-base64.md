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
- Train from scratch on `/tmp/dataset_v0_merged_local_approach` with clean recipe:
  OS=1, `grad_clip=5`, `delta_weight=0` (no δ fiddling), `--eval-every`, CUDA≠0, artifacts under `/tmp`.

## Promote rule (unchanged)

Overwrite canonical only if AbsRel≤0.30 **and** D̂≤0.25. Else keep candidate under `/tmp` / dated artifact dir; `depth_head.enable` remains false.
