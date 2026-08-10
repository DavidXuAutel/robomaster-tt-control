# §4.1 revision proposal — V0 signal ③ estimator: band-median → reprojection

**Date:** 2026-08-10  **Status:** validated on Mac (local probe corpora); awaits spec re-freeze before authoritative wiring.
**Red lines UNCHANGED:** ③ pass threshold `scale_rel_err_max = 0.25`, ①d `depth_absrel_max = 0.30`. This revision changes *how ŝ is estimated*, not any threshold.

## Problem (why the current ③ is mis-specified)

The frozen ③ estimator (`vio.scale_from_depth_change` + `scale_relative_error`) computes

```
ŝ_D = | nanmedian(depth_last, band[1,40]) − nanmedian(depth_first, band[1,40]) |
rel  = | ŝ_D − ‖Δp‖ | / ‖Δp‖        (‖Δp‖ = GT metric displacement)
```

This equates *change in band-median depth* with *forward displacement* — valid only for fronto-parallel
axial translation into a surface that fills the band for the whole window. Band pixel composition churns as
the camera moves, so `|Δ median|` carries a per-window geometric bias independent of the depth model.

**Proof it's the estimator, not the model:** feeding **GT (perfect) depth** into this estimator on approach
corpora gives a forward-only median of **~0.19–0.36** (≈ or above the 0.25 line). No depth model can pass a
gate whose own oracle bias eats the budget. A valid scale estimator must give GT-oracle ≈ 0.

## Fix — reprojection / correspondence estimator

Replace aggregate medians with per-pixel geometric correspondence (`vio.reproject_scale_error`,
`vio.reproject_pair_rel_err`, added 2026-08-10, torch-free numpy):

1. Backproject frame-0 navigational-band pixels to 3D via intrinsics `K` (224², hfov 90° → fx=fy=112, cx=cy=111.5).
2. Transform camera-0 → world → camera-N using the **GT metric pose** (Δposition, Δyaw from proprio).
   Conventions: AirSim NED, body-forward camera; body(fwd,right,down) ↔ cam(Z,X,Y); DepthPlanar = Zc.
3. Reproject to frame-N pixels; the transformed point's forward coord is the **predicted** planar depth.
4. Compare predicted vs **observed** depth at the matched pixel; robust-median `|Zpred−Zobs|/Zobs`.

With GT depth + GT pose this is a geometric identity → GT-oracle → 0 (residual only from NN pixel rounding /
occlusion). Feeding predicted D̂ tests whether D̂'s metric scale is consistent with metric camera motion —
exactly signal ③.

## Validation (Mac, numpy 2.0.2, local `/tmp` approach probes)

Forward windows (|cos∠(Δp,heading)|≥0.7, window=8, band [1,40] m); GT-oracle leg:

| corpus | OLD band-median (n) | NEW reprojection (n) |
|---|---|---|
| d12 | 0.356 (4) | **0.0080** (52) |
| d18 | 0.219 (11) | **0.0055** (109) |
| d25 | 0.186 (15) | **0.0055** (153) |
| ALL | 0.241 (225) | **0.0054** (1281) |

Acceptance (GT-oracle < 0.05) **met with ~10× margin**, and the reprojection estimator yields ~6–10× more
valid windows (no support-gate discards). Repro: `/tmp/claude-501/reproj_ab.py` (standalone) and
`experiments/aerial/rl/vio.py::reproject_scale_error` (module; reproduces 0.005436 on the same windows).

## Wiring plan (apply after re-freeze; needs H100/torch for the D̂ leg)

1. **Read-only first (no verdict change):** add a reprojection GT + D̂ leg to `_v0_gate._run_signal3_diagnose`
   alongside the band-median legs. Pass `positions` (prefer GT `obs.position`, not vel-integration) and per-frame
   `yaw` (already assembled in the diagnose) + intrinsics. This is A/B only.
2. **Authoritative swap (dated §4.1 revision):** point `check_scale_consistency` / `_rel_over` at
   `reproject_scale_error`. Keep band-median helpers for regression/A-B. `scale_support_ratio` becomes moot
   (reprojection has no dead-proxy failure mode); `fwd_cos_min` may relax.

## Caveats / open items

- Estimator assumes a **static scene** and accurate GT pose; dynamic obstacles or pose noise raise the floor
  above 0 (still far below 0.25).
- Intrinsics are hard-derived from 224²/hfov 90°. If capture res/FOV changes, pass `intrinsics_from_hfov(...)`.
- This fixes ③ only. ①d (from-scratch `_DepthHead` AbsRel ~0.28 ceiling) is a **separate** defect →
  DA3 pretrained backbone. DA3 does NOT fix ③; ③ does not fix ①d. Order: **③ (done) → DA3 → (opt) Wan2.1 offline**.
