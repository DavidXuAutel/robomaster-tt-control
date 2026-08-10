# §3/§6/§9 revision proposal — V0 signal ①d depth backbone: from-scratch → DA3 (optional)

**Date:** 2026-08-10  **Status:** code landed (inert, default unchanged); ready for spec re-freeze then authoritative H100 gate run.
**Red lines UNCHANGED:** ①d `depth_absrel_max = 0.30`, ③ `scale_rel_err_max = 0.25`. This revision adds an *alternate depth-head backbone*; it changes no threshold, no gate math, and no interface semantics.

## Problem (why the from-scratch head is topped out)

The frozen `[1b]` depth head `_DepthHead` is a from-scratch 4-layer conv encoder + mirror deconv (base=32),
trained on ~40–58 episodes with no pretrained backbone. Its ①d AbsRel ceiling:

- Representative holdout (`dataset_v0_local_depth`): **AbsRel 0.281 PASS** (n_holdout=4) — a thin ~6% margin under the 0.30 line.
- Approach-biased probe (`dataset_v0_approach_scale_d18`, the deployment-relevant slice): **AbsRel 0.3105 FAIL** (n_holdout=13).

So the depth pillar's ①d passes only on the representative corpus with no real margin, and *fails* on the
approach imagery that V0 actually cares about. This is not a recipe-tuning gap — the OS/δ sweeps and two-stage
FT were exhausted (see [`aerial_v0_signal3_dhat_status`](../../../.claude/... memory)); it is a **capacity ceiling**
of a small from-scratch net (OS/δ sweeps and two-stage FT exhausted). The frozen spec's own structural
verdict named the fix: adopt a pretrained backbone.

Note: ③ is already SOLVED and re-frozen (reprojection estimator, canonical D̂ 0.122 ≤ 0.25). DA3 does **not**
touch ③ — it is purely an ①d capacity lift. DA3 is therefore *optional* (③ never depended on it), justified
solely by widening the thin/failing ①d margin toward 0.1x.

## Fix — DA3METRIC-LARGE backbone (frozen DINOv2-ViT-L + trainable DPT)

Add a sibling head `DA3DepthHead` (Depth-Anything-3 metric-large: frozen DINOv2-ViT-L encoder + trainable DPT
depth decoder), warm-started from the Apache-2.0 `depth-anything/DA3METRIC-LARGE` weights and fine-tuned on our
GT depth to learn metric scale directly. `_DepthHead` is **left byte-for-byte unchanged** (it is the canonical
`depth_step_5000.pt` loader); dispatch is via a new `backbone` payload key through a `build_depth_head` factory
(`"scratch"` default → `_DepthHead`; `"da3"` → `DA3DepthHead`). All existing checkpoints (incl. canonical) have
no `backbone` key → rebuild as scratch, unchanged.

Design points:

1. **Wrapper-bypass.** `DA3DepthHead` constructs `DinoV2` + `DPT` directly and replicates only the metric-large
   depth forward — `feats,_ = encoder(x[B,N,3,H,W], cam_token=None, export_feat_layers=[])` then
   `decoder(feats, H, W, patch_start_idx=0)["depth"]`. It never instantiates `DepthAnything3Net`, so it drags
   in none of omegaconf / open3d / pycolmap / trimesh / alignment / geometry / ray_utils / gs heads.
2. **Single-frame, N=1.** For metric-large `alt_start=-1`, so the reference-view / cam-token / global-attention
   branch in `_get_intermediate_layers_not_chunked` is skipped entirely — N=1 is safe and `reference_view_selector`
   is inert (vendored only to satisfy a module-level import). `predict_from_window` takes the window's **last
   frame** (matches ①d "last frame" and ③ per-frame reprojection), [0,1] → ImageNet-normalizes
   (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) → shapes to N=1. **224 % 14 == 0 → no resize.**
   `n_frames` / `motion_channels` are inert for this backbone.
3. **Metric scale is learned, not heuristic.** DA3METRIC standalone emits up-to-scale exp-depth
   (upstream uses `metric = focal·net/300`); we do NOT use that heuristic — fine-tuning the DPT head on our GT
   with the existing `depth_head_loss` (AbsRel + SILog) learns metric scale directly. No GT/focal at inference.
4. **No Δ-loss, no NLL.** ③ is solved, so `--backbone da3` forces `delta_weight=0`. DPT emits depth only
   (no conf) → zero `log_sigma`, so `nll_weight=0`. runtime `predict_min` uses depth only; ①d carries AbsRel+SILog.
5. **Freeze semantics reuse.** `_apply_freeze_encoder` freezes `model.encoder` (= DINOv2) and trains
   `model.decoder` (= DPT) + `new_pathway_parameters()` (empty for DA3) — exactly freeze-backbone / train-head.
   Default `freeze_encoder=True` for da3 unless `--no-freeze-encoder`.
6. **Self-contained checkpoints.** The saved ①d ckpt stores full backbone+head state + `backbone:"da3"` +
   `da3_arch`, so the gate/runtime need only the vendored *code*, not the DA3 source weights. DA3 weights are
   fetched once on the training machine (H100, `mot-wam`) via `huggingface_hub.hf_hub_download(model.safetensors)`.

## Delivery — vendored pure-torch subset

`experiments/aerial/rl/third_party/depth_anything_3/` vendors the minimal DINOv2+DPT subset (pinned upstream
commit `3d835ec`, Apache-2.0): `model/dpt.py`, `model/utils/head_utils.py`, `model/dinov2/{dinov2,vision_transformer}.py`,
`model/dinov2/layers/*`, `model/reference_view_selector.py` (import-only), `utils/{logger,constants}.py`. External
deps (torch, addict, einops, numpy) already in `mot-wam`; xformers optional. `model/__init__.py` emptied to drop
the eager `da3` import. See `third_party/.../VENDOR.md` for provenance and update procedure.

## Contract (drop-in, unchanged consumer sites)

`DA3DepthHead` exposes the same surface consumed at the 3 call sites (train / gate ×2 / runtime):
`predict_from_window(rgb[B,L,H,W,3]) → (depth[B,H,W]>0, log_sigma[B,H,W])`, `from_payload`, `state_dict()` /
`load_state_dict(strict=True)`, `.encoder` / `.decoder`, `new_pathway_parameters()`, attrs `.n_frames` / `.image_size`
/ `.arch`. All three sites route through `build_depth_head`.

## Gate protocol (unchanged thresholds)

Run authoritative `_v0_gate --signals 1,3` with the DA3 ①d ckpt on BOTH the representative (`local_depth`) and
approach corpora. Expect ①d ≤ 0.30 with real margin (target 0.1x) on both, and ③ still ≤ 0.25 (reprojection,
band [1,40]). Canonical `depth_step_5000.pt` is not overwritten (DA3 ckpt stem `depth_step_N_da3`); yaml
`world_model.depth_head.enable` / `safety.kind` stay OFF until all four V0 signals PASS; no 4090 ②/④ until the
full V0 verdict is green.

## Acceptance

1. DA3 net imports + single-frame forward in `mot-wam` yields correct-shape >0 depth.
2. `--backbone da3` training records holdout AbsRel ≤ 0.30 with margin on representative AND approach corpora.
3. Authoritative `_v0_gate --signals 1,3` on the DA3 ckpt: ①d PASS (margin) + ③ PASS.
4. `_DepthHead` canonical rebuild still works (`backbone="scratch"` default) — gate self-check + `depth_step_5000.pt` load regression.
