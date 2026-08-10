# Vendored subset of Depth-Anything-3 (pure-torch depth path)

**Upstream:** https://github.com/ByteDance-Seed/Depth-Anything-3 (Apache-2.0)
**Pinned commit:** `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` ("Update backend dashboard rendering")
**Vendored:** 2026-08-10, for V0 signal ①d (DA3METRIC-LARGE depth backbone).

## Why vendored (not pip-installed)

The aerial gate/train env (`mot-wam`, numpy 2.x) must not gain heavy/conflicting
deps (open3d, pycolmap, trimesh, omegaconf). Those live behind `api.py` /
`utils/export.py` / `model/da3.py`, none of which are needed to run the
DINOv2-ViT-L encoder + DPT depth head. This is the minimal pure-torch subset
that `DinoV2` + `DPT` transitively require. External deps: `torch`, `addict`,
`einops`, `numpy` (all already in `mot-wam`); `xformers` is optional (try/except
torch fallback in `model/dinov2/layers/swiglu_ffn.py`, and vitl uses `ffn_layer="mlp"`
so `SwiGLUFFNFused` is never instantiated).

## What is NOT vendored (deliberately)

`model/da3.py` (`DepthAnything3Net` wrapper) is **bypassed** — it imports
`cfg.create_object` (omegaconf), `utils/alignment`, `utils/geometry`,
`utils/ray_utils`, `model/utils/transform`, and the gaussian-splatting heads.
`DA3DepthHead` (in `../../dynamics_torch.py`) constructs `DinoV2` + `DPT`
directly and replicates only the metric-large depth forward:

```
feats, _ = encoder(x, cam_token=None, export_feat_layers=[])   # x: (B, N, 3, H, W)
depth = decoder(feats, H, W, patch_start_idx=0)["depth"]        # (B, N, H, W)
```

For metric-large `alt_start=-1`, so the reference-view / cam-token / global-attention
branch in `_get_intermediate_layers_not_chunked` is skipped entirely — N=1 is safe
and `reference_view_selector` is never invoked at runtime (it is vendored only to
satisfy the module-level import in `vision_transformer.py`).

## Vendored files

- `model/dpt.py`, `model/utils/head_utils.py` — DPT depth decoder.
- `model/dinov2/{dinov2,vision_transformer}.py`, `model/dinov2/layers/*` — DINOv2-ViT-L encoder.
- `model/reference_view_selector.py` — import-time dep of `vision_transformer.py` (unused at N=1/alt_start=-1).
- `utils/{logger,constants}.py` — torch-free helpers (`logger`, `THRESH_FOR_REF_SELECTION`).

`__init__.py` files are empty except `model/dinov2/layers/__init__.py` (upstream,
required) — in particular `model/__init__.py` is emptied to drop its `da3` eager import.

## Weights

Vendored code carries **no weights**. DA3METRIC-LARGE is fetched once on the
training machine via `from_pretrained("depth-anything/DA3METRIC-LARGE")` (HF), the
`model.backbone.*` / `model.head.*` state loaded (prefix-stripped) to warm-start.
Fine-tuned ①d checkpoints are self-contained (full backbone+head state), so the
gate/runtime only need this code, not the DA3 weights.

## Updating

Re-copy the same file list from the pinned upstream path
`src/depth_anything_3/`, re-run the in-memory syntax check, and bump the commit
above. Do not add files that pull omegaconf/open3d/pycolmap/trimesh.
