# §4.1 revision proposal — V0 signal ③ estimator: band-median → reprojection

**Date:** 2026-08-10  **Status:** validated on Mac (GT-oracle) **and H100 (D̂, 5 corpora + band sweep)**; ready for spec re-freeze + authoritative wiring.
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

## Validation (H100, `mot-wam` torch 2.6.0+cu126, canonical `depth_ckpt/depth_step_5000.pt`)

Same forward-window protocol; the D̂ leg the Mac could not run. `--signal3-diagnose` on
`dataset_v0_approach_scale_d18` (window 8, max-windows 256), OLD vs NEW side by side on one machine/corpus/ckpt:

| leg | OLD band-median (n) | NEW reprojection (n) |
|---|---|---|
| GT-oracle fwd | 0.263 **FAIL** (151) | **0.002 PASS** (154) |
| D̂ fwd | 0.619 **FAIL** (84) | **0.122 PASS** (154) |

The old proxy fails its *own* perfect-depth oracle (0.263 > 0.25) — reconfirming it never measured scale. Under
reprojection, canonical D̂ passes ③ with ~2× margin. **Corroborated across 5 corpora** (canonical ckpt, D̂ reproj-fwd):
d18 0.122 (154) / local_depth 0.112 (46) / approach_A 0.147 (62) / approach_B 0.148 (63) / approach_C 0.144 (65) —
all PASS, GT reproj-fwd 0.002–0.003 everywhere.

## Band-sensitivity (H100, canonical ckpt, dataset_v0_approach_scale_d18, n=154)

Read-only `--reproj-band-max` sweep (commit d266d60 added `--reproj-band-min/max`, overriding ONLY the reprojection
leg; verdict/yaml untouched):

| band (m) | D̂ reproj-fwd | GT reproj-fwd |
|---|---|---|
| [1, 40] | **0.122** ✅ | 0.002 |
| [1, 80] | 0.097 ✅ | 0.003 |
| [1, 200] | 0.045 ✅ | 0.004 |
| [1, ∞) | 0.043 ✅ | 0.001 |

Two findings: (1) **the band is not a bias knob** in the reprojection estimator — GT-oracle stays ≈0 at every band
(the old band-median proxy flew to 0.26 from band-composition churn). (2) D̂ error is **monotone-decreasing** as the
band widens (near-field surfaces carry larger relative scale error; far pixels stabilize the robust median), so the
frozen **[1,40] is the worst case for D̂ and still passes at 0.122**.

**DECISION: §4.1 freezes the band at [1,40] m** — most conservative, matches ①d depth masking / navigation
domain-of-interest, and GT-oracle≈0 proves zero estimator bias regardless. The band is a legitimate
domain-of-interest mask, not a bias source.

## Wiring plan (apply after re-freeze; needs H100/torch for the D̂ leg)

1. **Read-only first (no verdict change) — DONE.** Reprojection GT + D̂ legs live in
   `_v0_gate._run_signal3_diagnose` alongside the band-median legs (uses GT `obs.position` + per-frame `yaw` +
   `intrinsics_from_hfov`). Read-only band override `--reproj-band-min/max` added (commit d266d60). Validated above.
2. **Authoritative swap (dated §4.1 revision) — PENDING re-freeze.** Point `check_scale_consistency` / `_rel_over` at
   `reproject_scale_error` with band frozen [1,40] m. Keep band-median helpers for regression/A-B. `scale_support_ratio`
   becomes moot (reprojection has no dead-proxy failure mode); keep `fwd_cos_min=0.7` (unchanged for now — relaxing is a
   separate future revision, not part of this one).

## Re-freeze checklist (execute in order; red lines & yaml flags untouched)

1. **Spec:** add a dated §4.1 entry to the frozen V0 spec recording: estimator = `reproject_scale_error`, band
   [1,40] m, GT-oracle acceptance <0.05 (achieved 0.002), thresholds `scale_rel_err_max=0.25` **unchanged**. Cite this
   doc + the 5-corpus / band-sweep evidence. Re-freeze.
2. **Code:** in `v0_metrics.check_scale_consistency` / `_rel_over` (the authoritative `--signals 1,3` path), compute ŝ
   via `reproject_scale_error` instead of band-median `scale_from_depth_change`. Band-median fns stay in `vio.py`
   (deprecated-in-place, retained for A/B). No threshold edits.
3. **Verify:** run authoritative `--signals 1,3` on H100 (canonical ckpt) → expect ①d AbsRel 0.281 PASS + ③ ≈0.12 PASS.
   Record the formal verdict. Only then are ①③ authoritatively green.
4. **Guardrails held:** canonical ckpt not overwritten; `depth_head.enable`/`safety.kind` stay off until all four V0
   signals pass; do NOT start 4090 ②/④ until this authoritative ③ verdict is green.

## Caveats / open items

- Estimator assumes a **static scene** and accurate GT pose; dynamic obstacles or pose noise raise the floor
  above 0 (still far below 0.25).
- Intrinsics are hard-derived from 224²/hfov 90°. If capture res/FOV changes, pass `intrinsics_from_hfov(...)`.
- This fixes ③ only. ①d (from-scratch `_DepthHead` AbsRel ~0.28 ceiling) is a **separate** defect →
  DA3 pretrained backbone. DA3 does NOT fix ③; ③ does not fix ①d. Order: **③ (done) → DA3 → (opt) Wan2.1 offline**.
