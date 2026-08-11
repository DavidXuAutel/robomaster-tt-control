"""V0 four-signal gate entrypoint (frozen §4 / §4.1).

Mac-testable scorers live in :mod:`v0_metrics` / :mod:`vio`; the paired rollout
runners in :mod:`v0_rollout_eval`. This module is the H100-facing CLI: it refuses
desynced / RGB-only corpora, and assembles the four signals.

**Depth pillar is enforced** (the failure mode that invalidated the old single-
pillar checkpoint): signal ① = ①a–c *and* ①d (holdout depth AbsRel), and ③
consumes the depth head's **predicted** D̂ — not GT depth. Without ``--depth-ckpt``
both ①d and ③ FAIL, so the whole gate FAILs. GT depth alone can never pass V0.

    # metric self-check (no GPU):
    python -m experiments.aerial.rl._v0_gate --self-check

    # on H100 with a depth-capable corpus + a trained depth head:
    python -m experiments.aerial.rl._v0_gate \\
        --dataset .../artifacts/dataset_v0_local_depth \\
        --learning-log .../wm_ckpt/wm_train.jsonl \\
        --depth-ckpt .../depth_ckpt/depth_step_2000.pt \\
        --rollout-eval --device cuda

**Split evaluation (§6 Step 6 plan B — the renderer lives on the 4090, not the
H100).** ②/④ need a live obstacle env, so score the offline signals where the
data + weights are and the online signals where the renderer is, then merge:

    # H100 — offline ①(a–d) + ③ (no renderer needed):
    python -m experiments.aerial.rl._v0_gate --signals 1,3 \\
        --dataset .../dataset_v0_local_depth \\
        --learning-log .../wm_ckpt/wm_train.jsonl \\
        --depth-ckpt .../depth_ckpt/depth_step_2000.pt \\
        --device cuda --emit part_13.json

    # 4090 — online ②/④ against the airsim renderer (env.backend=airsim):
    python -m experiments.aerial.rl._v0_gate --signals 2,4 \\
        --config configs/aerial_rl.yaml \\
        --depth-ckpt .../depth_ckpt/depth_step_2000.pt \\
        --device cuda --emit part_24.json

    # anywhere — authoritative four-signal verdict:
    python -m experiments.aerial.rl._v0_gate --merge part_13.json part_24.json

A subset run emits a ``partial`` verdict and exits 0 iff every *requested* signal
passed — it is NOT the gate. Only ``--merge`` of all four (or a full single-host
run) produces the authoritative pass that a human may act on.

Does **not** flip ``configs/aerial_rl.yaml`` gates — that happens only after the
authoritative verdict exits 0 (human / follow-up commit).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.aerial.rl import dataset as ds
from experiments.aerial.rl import v0_metrics as metrics


def _refuse_rgb_only_desync(root: Path, allow: bool) -> None:
    """Refuse over-commanded / incomplete corpora (aligned with ``_refuse_v0``)."""
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    meta = manifest.get("meta") or {}
    step_hz = float(meta.get("step_hz", 0) or 0)
    grab_depth = bool(meta.get("grab_depth", False))
    if not allow:
        if grab_depth and step_hz > 6.5:
            print(
                f"[v0-gate] REFUSE: step_hz={step_hz} with grab_depth exceeds the "
                "measured 4090-local depth closed-loop ceiling (~6.2 Hz).",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if step_hz > 8.5:
            print(
                f"[v0-gate] REFUSE: step_hz={step_hz} looks like the dt-desynced "
                "cross-net RGB smoke corpus — not a V0 training set.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if not grab_depth:
            print(
                "[v0-gate] REFUSE: manifest grab_depth=false — V0 four-signal gate "
                "needs schema-v2 depth/IMU corpus from 4090-local collect. "
                "Pass --allow-incomplete only to exercise scorers on synthetic inputs.",
                file=sys.stderr,
            )
            raise SystemExit(2)


# --------------------------------------------------------------------------- #
# Signal assembly (pure — Mac-unit-testable by injecting the per-signal dicts) #
# --------------------------------------------------------------------------- #
def assemble_verdict(
    *,
    s1abc: Dict[str, Any],
    s1d: Dict[str, Any],
    s2: Dict[str, Any],
    s3: Dict[str, Any],
    s4: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine the four signals with ① = ①a–c ∧ ①d (depth pillar folded in)."""
    sig1 = {
        "ok": bool(s1abc.get("ok") and s1d.get("ok")),
        "abc": s1abc,
        "d": s1d,
    }
    return metrics.aggregate_v0_verdict({"1": sig1, "2": s2, "3": s3, "4": s4})


_ALL_SIGNALS = ("1", "2", "3", "4")


def _parse_signals(spec: Optional[str]) -> set:
    """Parse ``--signals 1,3`` → {"1","3"}; default (None) = all four."""
    if not spec:
        return set(_ALL_SIGNALS)
    req = {s.strip() for s in spec.split(",") if s.strip()}
    bad = req - set(_ALL_SIGNALS)
    if bad:
        raise SystemExit(f"[v0-gate] --signals: unknown {sorted(bad)}; pick from 1,2,3,4")
    if not req:
        raise SystemExit("[v0-gate] --signals: empty selection")
    return req


def _merge_partials(paths: List[Path]) -> Dict[str, Any]:
    """Union the per-signal dicts from partial (or full) verdict JSON files.

    Accepts either a partial ``{"signals": {...}}`` or a full verdict
    ``{"details": {...}}`` blob. Later files win on a duplicate signal id (with a
    warning) — normally the two hosts cover disjoint ids (1,3 vs 2,4).
    """
    signals: Dict[str, Any] = {}
    for pth in paths:
        blob = json.loads(pth.read_text())
        part = blob.get("signals") or blob.get("details") or {}
        if not part:
            print(f"[v0-gate] WARN: {pth} has no signals/details block", file=sys.stderr)
        for k, v in part.items():
            if k in signals:
                print(f"[v0-gate] WARN: signal {k} in multiple partials; {pth} wins",
                      file=sys.stderr)
            signals[k] = v
    return signals


def _emit(obj: Dict[str, Any], path: Optional[str]) -> None:
    text = json.dumps(obj, indent=2, default=str)
    print(text)
    if path:
        Path(path).write_text(text + "\n")
        print(f"[v0-gate] wrote {path}", file=sys.stderr)


def _signal1abc_from_log(
    path: Optional[Path], thr: metrics.V0GateThresholds
) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"ok": False, "reason": "missing --learning-log (①a–c needs train curves)"}
    losses: List[float] = []
    recons: List[float] = []
    ents: List[float] = []
    recon_missing = ent_missing = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        losses.append(float(row["loss"]))
        if "recon_err" in row or "recon" in row:
            recons.append(float(row.get("recon_err", row.get("recon"))))
        else:
            recon_missing = True
        if "post_entropy_frac" in row or "ent" in row:
            ents.append(float(row.get("post_entropy_frac", row.get("ent"))))
        else:
            ent_missing = True
    # ①a–c is recon-monotonicity ∧ no-posterior-collapse ∧ loss-drop. A log that
    # lacks recon/entropy cannot evidence the first two — fail rather than let
    # pass-safe defaults (recon 0, ent 1) green-light on the loss drop alone.
    if recon_missing or ent_missing:
        miss = [k for k, m in (("recon_err/recon", recon_missing),
                               ("post_entropy_frac/ent", ent_missing)) if m]
        return {
            "ok": False,
            "reason": f"learning-log missing {miss}; ①a–c cannot evidence "
                      "recon-monotonicity / posterior-collapse without them",
        }
    return metrics.check_learning_curves(losses, recons, ents, thr=thr)


def _signal1d(
    pred_depth: Optional[np.ndarray],
    gt_depth: Optional[np.ndarray],
    thr: metrics.V0GateThresholds,
) -> Dict[str, Any]:
    """①d — holdout depth AbsRel. FAIL (not SKIP) when no depth head / GT.

    Frozen §4.1: "否则 ①d=SKIP（整门 FAIL——V0 需要深度柱）". We surface that as an
    explicit FAIL so the aggregate can never pass without the depth pillar.
    """
    if pred_depth is None:
        return {"ok": False, "reason": "no depth head predictions (pass --depth-ckpt); V0 needs the depth pillar"}
    if gt_depth is None:
        return {"ok": False, "reason": "dataset carries no GT depth to score ①d against"}
    return metrics.check_depth_absrel(pred_depth, gt_depth, thr=thr)


def _signal3(
    pred_depth: Optional[np.ndarray],
    positions: Optional[np.ndarray],
    yaw: Optional[np.ndarray],
    thr: metrics.V0GateThresholds,
) -> Dict[str, Any]:
    """③ — D̂-scale vs GT metric motion (reprojection, §4.1 rev 2026-08-10).
    Must use PREDICTED depth; GT-only cannot pass V0."""
    if pred_depth is None:
        return {"ok": False, "reason": "③ needs predicted D̂ (pass --depth-ckpt); GT depth is plumbing-only"}
    if positions is None or yaw is None:
        return {"ok": False, "reason": "dataset missing proprio pose (position/yaw) for ③ reprojection"}
    from experiments.aerial.rl import vio as vio_lib
    hh, ww = int(pred_depth.shape[-2]), int(pred_depth.shape[-1])
    fx, fy, cx, cy = vio_lib.intrinsics_from_hfov(ww, hh, 90.0)
    return metrics.check_scale_consistency_reproj(
        pred_depth, positions, yaw, fx=fx, fy=fy, cx=cx, cy=cy, thr=thr,
    )


# --------------------------------------------------------------------------- #
# Depth-head inference over dataset windows (lazy torch — H100 only)          #
# --------------------------------------------------------------------------- #
def _forwardness(dvec: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """|cos| between net horizontal displacement and the window's mean heading.

    ``dvec [B,3]`` is Δp = pos_last − pos_first; ``yaw [B,L]`` is per-frame heading
    (state[6]). Camera is forward-facing so the optical axis projects to
    ``[cos ȳ, sin ȳ]`` (mean heading, robust to a small in-window turn). Returns
    ``[B]`` in [0,1]: ~1 = axis-aligned translation (forward OR backward — both
    change |Δ median depth|, which is what ③'s proxy needs); ~0 = lateral / yaw /
    climb-dominated, where the median-depth proxy is physically meaningless (frozen
    §4.1 ③ applicability note). Vertical motion enters the ‖Δp‖ denominator, so a
    climb-dominated window scores low.
    """
    dvec = np.asarray(dvec, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    disp = np.linalg.norm(dvec, axis=-1)                     # [B] full 3-D
    mc = np.cos(yaw).mean(axis=-1)
    ms = np.sin(yaw).mean(axis=-1)
    hn = np.hypot(mc, ms)
    with np.errstate(invalid="ignore", divide="ignore"):
        fwd_hat = np.stack([mc / hn, ms / hn], axis=-1)      # [B,2]
        fdot = dvec[..., 0] * fwd_hat[..., 0] + dvec[..., 1] * fwd_hat[..., 1]
        out = np.where((disp > 1e-9) & (hn > 1e-9), np.abs(fdot) / disp, 0.0)
    return np.nan_to_num(out, nan=0.0).astype(np.float64)


def _sample_windows(root: Path, *, window: int, max_windows: int) -> List[Any]:
    """Legacy prefix sampler (first ``window`` frames per episode). Prefer the
    holdout / scale samplers below for gate scoring."""
    episodes = ds.load_dataset(root, skip_quarantined=True)
    windows: List[Any] = []
    for ep in episodes:
        if len(ep) < window:
            continue
        windows.append(ep[:window])
        if len(windows) >= max_windows:
            break
    return windows


def _window_forward_frac(window: Any) -> float:
    """Heading-aligned forwardness in [0, 1] for a scored window.

    Dated §4.1 revision 2026-08-05: replaces the world-+x proxy (which selected
    open-horizon cruises when yaw ̸≈ 0) with ``|cos∠(Δp, mean heading)|`` via
    :func:`_forwardness`. Camera is body-forward, so optical-axis translation
    is what makes ``|Δ median depth|`` share scale with ``‖Δp‖``.
    """
    pos = np.asarray(
        [np.asarray(t.obs.position, dtype=np.float64).reshape(3) for t in window],
        dtype=np.float64,
    )
    yaw = np.asarray([float(t.obs.yaw) for t in window], dtype=np.float64)
    dvec = pos[-1] - pos[0]
    return float(_forwardness(dvec[None, :], yaw[None, :])[0])


def _nav_band_frac(depth: np.ndarray, *, lo: float, hi: float) -> float:
    """Fraction of finite positive (≤200 m) pixels inside the navigational band."""
    d = np.asarray(depth, dtype=np.float64).reshape(-1)
    valid = np.isfinite(d) & (d > 0) & (d <= 200.0)
    if not np.any(valid):
        return 0.0
    return float(np.count_nonzero((d >= lo) & (d <= hi) & valid) / np.count_nonzero(valid))


def _sample_scale_windows(
    root: Path,
    *,
    window: int,
    max_windows: int,
    n_context: int,
    min_forward_frac: float = 0.7,
    min_motion_m: float = 0.5,
    scale_depth_min_m: float = 1.0,
    scale_depth_max_m: float = 40.0,
    min_band_frac: float = 0.05,
    support_ratio: float = 0.6,
) -> List[Any]:
    """Non-overlapping windows with RGB context prefix for full-context D̂ at t0.

    Each returned chunk has length ``n_context + window``; the *scored* tail is
    ``chunk[n_context:]``. Context lets ``_DepthHead`` see ``n_frames`` history at
    the first scored frame (no left-pad), so ``ŝ_D = |d_last − d_first|`` is not
    inflated by a single-frame warmup prediction.

    Dated §4.1 revision 2026-08-05 filters (GT used only to select physical
    situations — never as the scored ŝ_D for the gate):
      - heading-aligned forwardness ≥ ``min_forward_frac``
      - GT navigational-band content ≥ ``min_band_frac`` on first & last scored
      - GT approach support: |Δ band-median| ≥ ``support_ratio · ‖Δp‖``
        (drops wall-parallel / open-horizon dead-proxy windows)
    """
    from experiments.aerial.rl import vio as vio_lib

    episodes = ds.load_dataset(root, skip_quarantined=True)
    need = int(n_context) + int(window)
    windows: List[Any] = []
    lo, hi = float(scale_depth_min_m), float(scale_depth_max_m)
    # Prefer denser stride when episodes are long so approach windows are not
    # starved by early wall-parallel segments (OpenFly outdoor corpus).
    stride = max(1, int(window) // 2)
    for ep in episodes:
        if len(ep) < need:
            continue
        for start in range(0, len(ep) - need + 1, stride):
            chunk = ep[start : start + need]
            scored = chunk[int(n_context) :]
            pos = np.asarray(
                [np.asarray(t.obs.position, dtype=np.float64).reshape(3) for t in scored],
                dtype=np.float64,
            )
            dvec = pos[-1] - pos[0]
            motion = float(np.linalg.norm(dvec))
            if motion < float(min_motion_m):
                continue
            if _window_forward_frac(scored) < float(min_forward_frac):
                continue
            d0 = getattr(scored[0].obs, "depth", None)
            d1 = getattr(scored[-1].obs, "depth", None)
            if d0 is None or d1 is None:
                continue
            if (
                _nav_band_frac(d0, lo=lo, hi=hi) < float(min_band_frac)
                or _nav_band_frac(d1, lo=lo, hi=hi) < float(min_band_frac)
            ):
                continue
            # GT approach-support gate for window *selection* only.
            stack = np.stack(
                [np.asarray(d0, dtype=np.float32), np.asarray(d1, dtype=np.float32)],
                axis=0,
            )[None, ...]
            d_med = vio_lib.depth_median(stack, max_depth_m=hi, min_depth_m=lo)
            s_d = float(vio_lib.scale_from_depth_change(d_med)[0])
            if not np.isfinite(s_d) or s_d < float(support_ratio) * motion:
                continue
            windows.append(chunk)
            if len(windows) >= max_windows:
                return windows
    return windows


def _score_1d_holdout(
    root: Path,
    ckpt_path: Path,
    *,
    device: str,
    window: int,
    thr: metrics.V0GateThresholds,
    config_path: Path,
    holdout_frac: float = 0.2,
    split_seed: int = 0,
    wm_batch: int = 8,
) -> Dict[str, Any]:
    """①d — same episode-holdout median AbsRel protocol as ``train_depth_head``.

    Frozen §4.1: "holdout median AbsRel ≤ 0.30". The previous gate path scored the
    first ``window`` frames of the first N episodes (train+holdout mixed), which
    systematically disagreed with the trainer's true-holdout number.
    """
    import torch  # lazy: H100 only

    from experiments.aerial.rl.dynamics_torch import build_depth_head
    from experiments.aerial.rl.train_depth_head import (
        _holdout_absrel,
        _load_depth_cfg,
        _split_train_holdout,
        _usable_episodes,
    )

    dh_cfg = _load_depth_cfg(config_path)
    all_eps = _usable_episodes(root, int(window))
    _train_eps, holdout_eps = _split_train_holdout(
        all_eps, holdout_frac=float(holdout_frac), seed=int(split_seed)
    )
    if not holdout_eps:
        return {
            "ok": False,
            "absrel": float("nan"),
            "reason": "no held-out episodes for ①d (need ≥2 usable depth episodes)",
        }
    payload = torch.load(str(ckpt_path), map_location="cpu")
    # Payload wins, yaml is the fallback for checkpoints predating a given key.
    # Must go through from_payload: ①d and ③ load the SAME checkpoint, so a
    # loader that forgets an architecture flag crashes ①d on any net the other
    # loader can read.
    model = build_depth_head(
        {
            "backbone": payload.get("backbone", dh_cfg.get("backbone", "scratch")),
            "da3_arch": payload.get("da3_arch", None),
            **{
                key: payload.get(key, dh_cfg.get(key, default))
                for key, default in (
                    ("image_size", 224),
                    ("n_frames", 4),
                    ("base", 32),
                    ("motion_channels", False),
                    ("scale_factorized", False),
                )
            },
        }
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval().to(device)
    absrel = _holdout_absrel(
        model,
        holdout_eps,
        wm_batch=int(wm_batch),
        window=int(window),
        device=torch.device(device),
        max_depth_m=float(dh_cfg.get("max_depth_m", 200.0)),
    )
    if not np.isfinite(absrel):
        return {"ok": False, "absrel": absrel, "reason": "no valid depth pixels on holdout"}
    return {
        "ok": bool(absrel <= thr.depth_absrel_max),
        "absrel": float(absrel),
        "n_holdout_eps": len(holdout_eps),
        "protocol": "episode_holdout",
    }


def _predict_depth_over_windows(
    ckpt_path: Path,
    windows: List[Any],
    *,
    device: str = "cpu",
    score_context: int = 0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Run the trained ``_DepthHead`` on each frame of each window (lazy torch).

    Returns ``(perception_arrays, pred_depth[B, L_score, H, W])``. When
    ``score_context > 0``, ``windows`` are assumed to carry a leading context
    prefix of that length; only the trailing scored frames are returned in both
    ``arr`` and ``pred`` (so ③'s first frame has full ``n_frames`` history).
    """
    import torch  # lazy: H100 only

    from experiments.aerial.rl.dynamics_torch import build_depth_head
    from experiments.aerial.rl.perception_data import windows_to_perception_arrays

    payload = torch.load(str(ckpt_path), map_location="cpu")
    model = build_depth_head(payload)  # payload["backbone"] selects scratch/da3
    model.load_state_dict(payload["model"], strict=True)
    model.eval().to(device)

    arr_full = windows_to_perception_arrays(windows)
    rgb = torch.from_numpy(np.ascontiguousarray(arr_full["rgb"]))  # [B, L, H, W, 3]
    B, L = int(rgb.shape[0]), int(rgb.shape[1])
    n = model.n_frames
    ctx = max(0, int(score_context))
    if ctx >= L:
        raise ValueError(f"score_context={ctx} must be < window length {L}")
    preds = np.empty((B, L, model.image_size, model.image_size), dtype=np.float32)
    with torch.no_grad():
        for t in range(L):
            lo = max(0, t - n + 1)
            sub = rgb[:, lo : t + 1]  # pack_rgb_nhwc left-pads if shorter than n
            d, _ = model.predict_from_window(sub.to(device))
            preds[:, t] = d.cpu().numpy()
    if ctx == 0:
        return arr_full, preds
    arr: Dict[str, np.ndarray] = {}
    for k, v in arr_full.items():
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[0] == B and v.shape[1] == L:
            arr[k] = v[:, ctx:]
        else:
            arr[k] = v
    return arr, preds[:, ctx:]


# --------------------------------------------------------------------------- #
# ② / ④ via paired rollouts (mock env by default; airsim for a real ④ pass)   #
# --------------------------------------------------------------------------- #
def _obstacle_candidate_positions(
    dataset_dir: Path, *, stride: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """+up world positions (and recorded yaws) sampled from a collection.

    ④'s obstacle-facing generator teleports to these and keeps the yaws with a
    forward obstacle. Using collection trajectory positions (the drone flew
    *there*) guarantees the renderer's geometry is nearby AND side-steps every
    world-frame ambiguity — we scan the exact coordinates the data was logged
    at, never a guessed obstacle location.

    Positions are returned **nearest-geometry-first**: when the collection stored
    per-frame depth, we rank each sampled pose by its full-field min depth (the
    nearest geometry in *any* direction at that pose) ascending, so the scan
    (with ``preserve_order=True``) walks the near-building poses first instead of
    burning ``max_scans`` on open cruise corridors. The 2026-08-11 diag showed
    392/400 scanned poses were open-ahead (>15 m) — cruise flight is mostly open,
    so ordering the few near-obstacle poses to the front is what makes ④ scannable.
    If the corpus is RGB-only (no stored depth) every pose ranks ``inf`` and the
    order is left unchanged — a safe no-op fallback.

    Also returns the per-pose **recorded yaw** (radians, row-aligned with the
    positions) so the obstacle scan can try the heading the drone actually flew —
    on a head-on approach corpus that yaw points straight at the obstacle, which
    the 8-yaw grid (≤22.5° off) misses (2026-08-11: proxy_ok=19/probe_no_hit=19).
    """
    episodes = ds.load_dataset(Path(dataset_dir), skip_quarantined=True)
    pts: List[np.ndarray] = []
    yaws: List[float] = []
    prox: List[float] = []
    for ep in episodes:
        for i in range(0, len(ep), max(1, int(stride))):
            pts.append(np.asarray(ep[i].obs.position, dtype=np.float64))
            yaws.append(float(ep[i].obs.yaw))
            d = getattr(ep[i].obs, "depth", None)
            if d is None:
                prox.append(float("inf"))
                continue
            d = np.asarray(d, dtype=np.float64)
            finite = d[np.isfinite(d) & (d > 0)]
            prox.append(float(np.min(finite)) if finite.size else float("inf"))
    if not pts:
        raise ValueError(f"no positions in dataset {dataset_dir}")
    out = np.stack(pts)
    yaw_arr = np.asarray(yaws, dtype=np.float64)
    prox_arr = np.asarray(prox, dtype=np.float64)
    if np.isfinite(prox_arr).any():
        # stable ascending sort → near-building poses first; ties keep dataset order.
        order = np.argsort(prox_arr, kind="stable")
        out = out[order]
        yaw_arr = yaw_arr[order]
    return out, yaw_arr


def _signals_2_4_from_rollouts(
    config_path: Path,
    thr: metrics.V0GateThresholds,
    *,
    depth_ckpt: Optional[Path],
    device: str,
    n_episodes: int,
    max_steps: int,
    seed: int,
    rollout_dataset: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import yaml

    from experiments.aerial.rl import v0_rollout_eval as rollout
    from experiments.aerial.rl.reward import RewardConfig
    from experiments.aerial.rl.train_rl import HeuristicPolicy, _build_env

    cfg = yaml.safe_load(config_path.read_text()) or {}
    env = _build_env(cfg.get("env", {}) or {})
    reward_cfg = RewardConfig(**(cfg.get("reward", {}) or {})) if cfg.get("reward") else None

    # ④ is only meaningful when the start/goal geometry points at real obstacles.
    # With a collection dataset (--rollout-dataset) we scan its trajectory
    # positions live and keep obstacle-facing (start, heading) pairs; the SAME
    # set drives ② and ④ (frozen §4.1). Without a dataset we fall back to level
    # over-origin starts — fine for ② and the --allow-mock dev smoke, but ④ will
    # honestly degenerate in open airspace (near_coll_rate_off == 0).
    # ② progress-vs-random (goal-seeking HeuristicPolicy is the V0 baseline
    # policy). Built before the scan so it can PROBE-VERIFY obstacle starts:
    # the same straight-line goal-seeker drives ②, ④ and the scan probe, so a
    # start where the probe reaches the near-zone reproduces near_coll_off>0 on
    # the ④ shield-off arm by construction (fixes the 2026-08-11 near_coll_off==0
    # dead end where the proxy accepted wide-cone hits the policy threaded past).
    policy = HeuristicPolicy(goal_getter=lambda: getattr(env, "goal", None))
    if rollout_dataset is not None:
        cand, cand_yaw = _obstacle_candidate_positions(Path(rollout_dataset))
        # obstacle_max_m 25 (was 15): the probe is the real filter (rams the wall
        # head-on), so accept mid-range frontal obstacles up to just under the 30 m
        # goal — sky/max-range artifacts get rejected by the probe anyway.
        # probe_steps 40 (was 12): with the step_m double-clip fixed the policy now
        # steps ~1.0 m at 5 Hz, so 40 steps reaches a ≤25 m frontal obstacle head-on
        # (12 steps only travelled ~2 m and could not).
        # preserve_order: candidates are pre-ranked nearest-geometry-first, so walk
        # them in order; max_scans 1000 (was 400) still covers far more near-obstacle
        # poses because the open corridors sort to the back.
        starts, scan_diag = rollout.make_obstacle_facing_episodes(
            env, int(n_episodes), cand, seed=int(seed),
            candidate_yaws=cand_yaw,
            obstacle_max_m=25.0, center_frac=0.3,
            probe_policy=policy, probe_near_m=float(thr.near_collision_depth_m),
            probe_steps=40, reward_cfg=reward_cfg,
            preserve_order=True, max_scans=1000, log_every=20,
        )
        print(f"[v0-gate] obstacle-facing scan: {json.dumps(scan_diag)}")
        if not starts:
            reason = ("④ found 0 near-collision starts (probe-verified) scanning "
                      f"{scan_diag['scanned']} (pos,yaw) pairs from {rollout_dataset} "
                      "— straight-line goal-seeker never entered the 1.5 m near-zone "
                      "at these coords; ④ cannot be scored (fails closed)")
            s2 = {"ok": False, "reason": reason, "scan": scan_diag}
            s4 = {"ok": False, "reason": reason, "scan": scan_diag}
            return s2, s4
    else:
        starts = rollout.make_start_episodes(int(n_episodes), seed=int(seed))

    rnd = rollout.RandomActionPolicy(seed=int(seed))
    prog = rollout.run_progress_eval(
        env, policy, rnd, starts, max_steps=int(max_steps), reward_cfg=reward_cfg
    )
    s2 = metrics.check_progress_vs_random(
        prog["policy_progress_sums"], prog["random_progress_sums"],
        prog["policy_final_dists"], prog["random_final_dists"], thr=thr,
    )

    # ④ shield on/off — requires the real depth head to fill depth_min_pred.
    if depth_ckpt is None:
        s4 = {"ok": False, "reason": "④ needs --depth-ckpt (shield reads predicted D̂); depth pillar enforced"}
        return s2, s4
    from experiments.aerial.rl.depth_predictor import DepthMinPredictor

    predictor = DepthMinPredictor.from_checkpoint(depth_ckpt, device=device)
    masks = rollout.run_shield_eval(
        env, policy, predictor, starts,
        near_collision_depth_m=thr.near_collision_depth_m,
        # Reaction standoff > the frozen 1.5 m metric so the shield reacts before
        # the band (frozen-spec ④a re-freeze 2026-08-11). NOT a §4.1 gate threshold.
        shield_trigger_depth_m=3.0,
        max_steps=int(max_steps), reward_cfg=reward_cfg,
    )
    # ④'s near-collision rate is GT-depth-driven. If no episode carried a usable
    # depth field (grab_depth=false), every near mask is all-False and the ratio
    # would be nan/degenerate — fail closed with a clear reason instead of
    # leaning on rate_off<=0 inside the scorer (which reads as an opaque nan).
    if int(masks.get("depth_steps", 0)) == 0:
        s4 = {"ok": False,
              "reason": "④ has no GT depth in any rollout (grab_depth=false?); "
                        "cannot score near-collision rate — depth pillar enforced"}
        return s2, s4
    s4 = metrics.check_shield_effectiveness(
        interventions_on=masks["interventions_on"],
        collided_on=masks["collided_on"],
        near_coll_on=masks["near_coll_on"],
        near_coll_off=masks["near_coll_off"],
        thr=thr,
    )
    return s2, s4


# --------------------------------------------------------------------------- #
# ③ read-only diagnostic (forward-window + GT oracle) — NEVER touches verdict  #
# --------------------------------------------------------------------------- #
def _rel_over(
    depth: np.ndarray, motion: np.ndarray, thr: metrics.V0GateThresholds
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-window ③ rel-err + base-valid mask (mirrors the gate's ③ math)."""
    from experiments.aerial.rl import vio as vio_lib

    d_med = vio_lib.depth_median(
        depth,
        max_depth_m=thr.scale_depth_max_m,
        min_depth_m=thr.scale_depth_min_m,
    )
    s_d = vio_lib.scale_from_depth_change(d_med)
    err = vio_lib.scale_relative_error(
        s_d,
        motion,
        eps=thr.scale_eps,
        min_motion_m=thr.min_motion_m,
        motion_m=motion,
        support_ratio=thr.scale_support_ratio,
    )
    return np.asarray(err["rel_err"], dtype=np.float64), np.asarray(err["valid"], dtype=bool)


def _median_over(rel: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    m = mask & np.isfinite(rel)
    n = int(np.count_nonzero(m))
    return (float(np.median(rel[m])) if n else float("nan")), n


def _run_signal3_diagnose(
    dataset: Optional[str],
    depth_ckpt: Optional[str],
    thr: metrics.V0GateThresholds,
    *,
    device: str,
    window: int,
    max_windows: int,
    cos_min: float,
    allow_incomplete: bool,
    reproj_band_min: Optional[float] = None,
    reproj_band_max: Optional[float] = None,
) -> int:
    """Print D̂-vs-GT ③ rel under the §4.1 (2026-08-05) protocol. Read-only.

    ``reproj_band_min`` / ``reproj_band_max`` (read-only A/B): override the
    reprojection leg's depth band [1, 40] m to probe band-sensitivity of the
    candidate §4.1 estimator (e.g. drop the ceiling to see far-field D̂ error).
    ``None`` keeps the frozen thresholds. Never touches the authoritative verdict.
    """
    if not dataset or not depth_ckpt:
        print("[v0-gate] ③-diagnose needs --dataset and --depth-ckpt", file=sys.stderr)
        return 2
    from experiments.aerial.rl import vio as vio_lib

    root = Path(dataset)
    _refuse_rgb_only_desync(root, allow_incomplete)
    # Match the gate's sampler (context + heading-forward + nav-band content).
    import torch  # lazy

    payload = torch.load(str(depth_ckpt), map_location="cpu")
    n_frames = int(payload.get("n_frames", 4))
    n_context = max(0, n_frames - 1)
    windows = _sample_scale_windows(
        root,
        window=int(window),
        max_windows=max(int(max_windows), int(thr.min_scale_windows) * 4),
        n_context=n_context,
        min_forward_frac=float(cos_min),
        min_motion_m=thr.min_motion_m,
        scale_depth_min_m=float(thr.scale_depth_min_m),
        scale_depth_max_m=float(thr.scale_depth_max_m),
        support_ratio=float(thr.scale_support_ratio),
    )
    if not windows:
        print(
            "[v0-gate] ③-diagnose: no approach-support windows — "
            "corpus lacks approach geometry; recollect before concluding.",
            file=sys.stderr,
        )
        return 1
    arr, pred = _predict_depth_over_windows(
        Path(depth_ckpt), windows, device=device, score_context=n_context,
    )
    vel, ts = arr.get("vel"), arr.get("timestamps")
    gt = arr.get("depth")
    if vel is None or ts is None:
        print("[v0-gate] ③-diagnose: corpus missing vel/timestamps", file=sys.stderr)
        return 1

    pos, _dt = vio_lib.integrate_velocity(vel, ts)           # [B,L,3]
    motion = vio_lib.window_motion_m(pos)                    # [B]
    dvec = pos[:, -1, :] - pos[:, 0, :]                      # [B,3]
    L = int(pos.shape[1])
    yaw = np.asarray(
        [
            [float(windows[b][n_context + t].obs.yaw) for t in range(L)]
            for b in range(len(windows))
        ],
        dtype=np.float64,
    )
    fwd = _forwardness(dvec, yaw) >= float(cos_min)          # [B] bool

    rel_p, valid_p = _rel_over(pred, motion, thr)
    lines = [
        "[v0-gate] ③ DIAGNOSE (read-only; does NOT affect the verdict / yaml)",
        f"  sampled {len(windows)} scale-windows (score len {L}, ctx {n_context}); "
        f"forward = |cos∠(Δp,heading)| ≥ {cos_min}",
        f"  band = [{thr.scale_depth_min_m:g}, {thr.scale_depth_max_m:g}] m; "
        f"support = ŝ_D ≥ {thr.scale_support_ratio:g}·‖Δp‖; "
        f"pass = median rel ≤ {thr.scale_rel_err_max} with n≥{thr.min_scale_windows}",
        "  leg                 n_valid   median_rel   verdict",
    ]

    def _row(tag: str, rel: np.ndarray, base: np.ndarray, mask: np.ndarray) -> str:
        med, n = _median_over(rel, base & mask)
        if not np.isfinite(med) or n < int(thr.min_scale_windows):
            v = "n/a" if not np.isfinite(med) else "FAIL(n)"
        else:
            v = "PASS" if med <= thr.scale_rel_err_max else "FAIL"
        return f"  {tag:<18} {n:>7}   {med:>10.3f}   {v}"

    all_mask = np.ones_like(fwd, dtype=bool)
    lines.append(_row("D̂  all-motion", rel_p, valid_p, all_mask))
    lines.append(_row("D̂  forward-only", rel_p, valid_p, fwd))

    gt_fwd_med = float("nan")
    gt_fwd_n = 0
    if gt is not None:
        rel_g, valid_g = _rel_over(gt, motion, thr)
        lines.append(_row("GT  all-motion", rel_g, valid_g, all_mask))
        lines.append(_row("GT  forward-only", rel_g, valid_g, fwd))
        gt_fwd_med, gt_fwd_n = _median_over(rel_g, valid_g & fwd)
    else:
        lines.append("  GT  (absent)         GT depth not in corpus — oracle leg skipped")

    dhat_fwd_med, dhat_fwd_n = _median_over(rel_p, valid_p & fwd)

    # -- read-only reprojection leg (candidate §4.1 revision 2026-08-10) -------
    # Correspondence-based ③ (backproject frame-0 depth → GT pose → reproject).
    # GT-oracle ~0.005 vs band-median ~0.24 on local approach probes → returns the
    # 0.25 budget to real D̂ error. A/B ONLY; does NOT feed the verdict/rc/yaml.
    # See docs/handover/2026-08-10-signal3-reprojection-estimator.md.
    try:
        hh, ww = int(pred.shape[-2]), int(pred.shape[-1])
        fx, fy, cx, cy = vio_lib.intrinsics_from_hfov(ww, hh, 90.0)
        pos_gt = np.asarray(
            [
                [np.asarray(windows[b][n_context + t].obs.position,
                            dtype=np.float64).reshape(3) for t in range(L)]
                for b in range(len(windows))
            ],
            dtype=np.float64,
        )

        band_lo = float(thr.scale_depth_min_m if reproj_band_min is None else reproj_band_min)
        band_hi = float(thr.scale_depth_max_m if reproj_band_max is None else reproj_band_max)

        def _reproj_row(tag: str, depth_arr: np.ndarray, mask: np.ndarray) -> str:
            rr = vio_lib.reproject_scale_error(
                depth_arr, pos_gt, yaw, fx=fx, fy=fy, cx=cx, cy=cy,
                min_depth_m=band_lo, max_depth_m=band_hi,
                min_motion_m=thr.min_motion_m,
            )
            return _row(tag, np.asarray(rr["rel_err"], dtype=np.float64),
                        np.asarray(rr["valid"], dtype=bool), mask)

        _band_txt = f"[{band_lo:g}, {band_hi:g}] m" if band_hi < 1e8 else f"[{band_lo:g}, inf) m"
        lines.append(f"  ─ reprojection (read-only A/B; §4.1 candidate; band {_band_txt}) ─")
        lines.append(_reproj_row("D̂  reproj-fwd", pred, fwd))
        if gt is not None:
            lines.append(_reproj_row("GT  reproj-fwd", gt, fwd))
    except Exception as exc:  # never break the authoritative diagnose
        lines.append(f"  reprojection leg skipped ({type(exc).__name__}: {exc})")

    lines.append("  ─ interpretation ─")
    # rc mirrors the D̂ verdict so the diagnostic is scriptable: 0 only when D̂
    # passes on approach-support windows, 1 for any not-green outcome (matching
    # the run-failure exits above). It still never touches the authoritative
    # verdict or yaml — it just lets a wrapper branch on the conclusion.
    rc = 1
    if dhat_fwd_n < int(thr.min_scale_windows) and (
        gt is None or gt_fwd_n < int(thr.min_scale_windows)
    ):
        lines.append(
            f"  approach-support window count too small "
            f"(D̂ n={dhat_fwd_n}, GT n={gt_fwd_n}; need ≥{thr.min_scale_windows}) — "
            "recollect approach-biased trajectories (fly toward surfaces) before concluding."
        )
    elif gt is not None and (
        gt_fwd_n < int(thr.min_scale_windows)
        or (np.isfinite(gt_fwd_med) and gt_fwd_med > thr.scale_rel_err_max)
    ):
        lines.append(
            "  GT oracle still fails under the 2026-08-05 protocol (nav-band + approach "
            "support) → corpus geometry lacks enough approach windows, or the proxy needs "
            "another dated §4.1 revision. Do NOT retrain the depth head yet."
        )
    elif (not np.isfinite(dhat_fwd_med)) or dhat_fwd_n < int(thr.min_scale_windows):
        lines.append(
            f"  GT passes but D̂ has no approach-support windows (n={dhat_fwd_n}) — "
            "predicted ŝ_D is dead on GT-selected approach geometry. Retrain with a "
            "stronger temporal / Δ-depth term (and/or approach-biased sampling)."
        )
    elif dhat_fwd_med > thr.scale_rel_err_max:
        lines.append(
            "  GT passes but D̂ fails on approach-support windows → model under-standardizes "
            "scale change. Retrain the depth head with a temporal / Δ-depth consistency term "
            "(single-frame AbsRel loss does not teach metric scale-change), then re-run ③."
        )
    else:
        lines.append(
            "  D̂ passes on approach-support windows under the 2026-08-05 protocol — "
            "re-run the authoritative `_v0_gate --signals 1,3` (and later ②/④) before "
            "flipping yaml flags."
        )
        rc = 0
    lines.append(f"  (exit {rc}: 0=D̂ passes diagnostic, 1=not green; read-only — "
                 "verdict/yaml unaffected)")
    print("\n".join(lines))
    return rc


def _self_check(thr: metrics.V0GateThresholds) -> int:
    """Synthetic fixtures — locks §4.1 numbers + the depth-pillar aggregation."""
    s1abc = metrics.check_learning_curves(
        [10.0] * 50 + [5.0] * 50, [0.2] * 50 + [0.1] * 50, [0.5] * 100, thr=thr
    )
    assert s1abc["ok"], s1abc
    # ①d with predicted depth close to GT.
    gt = np.ones((4, 4, 4), dtype=np.float32) * 10.0
    pred = gt * 1.1
    s1d = _signal1d(pred, gt, thr)
    assert s1d["ok"], s1d
    s2 = metrics.check_progress_vs_random(
        [20.0] * 16, [10.0] * 16, [30.0] * 16, [30.0] * 16, thr=thr
    )
    assert s2["ok"], s2
    # ③ predicted depth-change matches metric motion (reprojection).
    # Straight-in approach: 1 m forward (North), depth 5 m → 4 m in step → rel≈0.
    B, L, H, W = 8, 8, 16, 16
    positions = np.zeros((B, L, 3), dtype=np.float64)
    positions[..., 0] = np.linspace(0.0, 1.0, L)
    yaw = np.zeros((B, L), dtype=np.float64)
    depth = np.ones((B, L, H, W), dtype=np.float32) * 5.0
    for t in range(L):
        depth[:, t] = 5.0 - positions[0, t, 0]
    s3 = _signal3(depth, positions, yaw, thr)
    assert s3["ok"], s3
    s4 = metrics.check_shield_effectiveness(
        interventions_on=[[False, True, False, False]],
        collided_on=[[False, False, False, True]],
        near_coll_on=[[False, False, False, False]],
        near_coll_off=[[True, True, False, False]],
        thr=thr,
    )
    assert s4["ok"], s4

    verdict = assemble_verdict(s1abc=s1abc, s1d=s1d, s2=s2, s3=s3, s4=s4)
    assert verdict["ok"], verdict
    # Depth pillar removed → whole gate must fail even with ①a–c/②/③/④ green.
    no_depth = assemble_verdict(
        s1abc=s1abc, s1d=_signal1d(None, None, thr), s2=s2, s3=s3, s4=s4
    )
    assert not no_depth["ok"], no_depth

    # Split-eval (plan B): merging disjoint partials {1,3}+{2,4} must reproduce
    # the single-host verdict, and a lone partial must NOT read as a pass.
    sig1 = {"ok": bool(s1abc.get("ok") and s1d.get("ok")), "abc": s1abc, "d": s1d}
    part_13 = {"1": sig1, "3": s3}
    part_24 = {"2": s2, "4": s4}
    merged = metrics.aggregate_v0_verdict({**part_13, **part_24})
    assert merged["ok"] == verdict["ok"], (merged, verdict)
    lone = metrics.aggregate_v0_verdict(part_13)  # missing 2,4
    assert not lone["ok"], lone
    print("[v0-gate] self-check PASS (incl. depth-pillar enforcement + split merge)")
    print(json.dumps({"thresholds": asdict(thr), "verdict": verdict["passed"]}, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None, help="schema-v2 episode dir")
    p.add_argument("--learning-log", default=None, help="jsonl with loss/recon/ent")
    p.add_argument("--depth-ckpt", default=None, help="trained _DepthHead .pt (①d/③/④ depth pillar)")
    p.add_argument("--config", default="configs/aerial_rl.yaml", help="for ②/④ rollout env")
    p.add_argument("--rollout-eval", action="store_true", help="run ②/④ paired rollouts")
    p.add_argument("--rollout-dataset", default=None,
                   help="collection dir (e.g. dataset_v1_rgb) whose trajectory positions "
                        "seed ④'s live obstacle-facing scan; omit → level over-origin starts "
                        "(② only; ④ degenerates in open airspace)")
    p.add_argument("--signals", default=None,
                   help="subset to score, e.g. '1,3' (H100 offline) or '2,4' (4090 renderer); "
                        "default = all four. A subset emits a PARTIAL verdict, not the gate.")
    p.add_argument("--emit", default=None, help="write the (partial or full) verdict JSON here")
    p.add_argument("--merge", nargs="+", default=None,
                   help="combine partial verdict JSONs into the authoritative four-signal gate")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-eval-episodes", type=int, default=None, help="override N (default §4.1 = 16)")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=16)
    p.add_argument("--holdout-frac", type=float, default=0.2,
                   help="①d episode holdout fraction (must match train_depth_head)")
    p.add_argument("--split-seed", type=int, default=0,
                   help="①d holdout split seed (must match train_depth_head)")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--allow-mock-rollout", action="store_true",
                   help="permit ②/④ on a non-airsim (mock/analytic) env — dev only; "
                        "the resulting ②/④ are NON-authoritative and must not gate flag flips")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--signal3-diagnose", action="store_true",
                   help="read-only: D̂-vs-GT ③ rel on all-motion vs forward-only windows (never affects verdict)")
    p.add_argument("--fwd-cos-min", type=float, default=0.7,
                   help="forward-window threshold |cos∠(Δp,heading)| (③-diagnose only)")
    p.add_argument("--reproj-band-min", type=float, default=None,
                   help="read-only A/B: override reprojection-leg depth band floor (m); "
                        "None keeps the frozen 1.0 m (③-diagnose only, never gates)")
    p.add_argument("--reproj-band-max", type=float, default=None,
                   help="read-only A/B: override reprojection-leg depth band ceiling (m); "
                        "use e.g. 80/200/inf to probe far-field D̂; None keeps the frozen 40 m "
                        "(③-diagnose only, never gates)")
    args = p.parse_args(argv)
    thr = metrics.DEFAULT_THRESHOLDS

    if args.self_check:
        return _self_check(thr)

    if args.signal3_diagnose:
        return _run_signal3_diagnose(
            args.dataset, args.depth_ckpt, thr,
            device=args.device, window=int(args.window),
            max_windows=int(args.max_windows), cos_min=float(args.fwd_cos_min),
            allow_incomplete=args.allow_incomplete,
            reproj_band_min=args.reproj_band_min,
            reproj_band_max=args.reproj_band_max,
        )

    if args.merge:
        signals = _merge_partials([Path(m) for m in args.merge])
        verdict = metrics.aggregate_v0_verdict(signals)
        _emit(verdict, args.emit)
        print(f"[v0-gate] MERGED {'PASS' if verdict['ok'] else 'FAIL'}")
        return 0 if verdict["ok"] else 1

    req = _parse_signals(args.signals)
    need_dataset_depth = bool({"1", "3"} & req)   # ①d + ③ read depth over the corpus
    need_rollout = bool({"2", "4"} & req) or args.rollout_eval

    thr_eff = thr
    if args.n_eval_episodes is not None:
        from dataclasses import replace

        thr_eff = replace(thr, n_eval_episodes=int(args.n_eval_episodes))

    # --- depth-head predictions / holdout AbsRel (①d + ③; H100/4090 GPU) --- #
    pred_depth: Optional[np.ndarray] = None
    gt_depth: Optional[np.ndarray] = None
    vel: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    pos_s3: Optional[np.ndarray] = None
    yaw_s3: Optional[np.ndarray] = None
    s1d_holdout: Optional[Dict[str, Any]] = None
    if need_dataset_depth:
        if not args.dataset:
            print("[v0-gate] FAIL: --dataset required for signals ①/③", file=sys.stderr)
            return 2
        root = Path(args.dataset)
        _refuse_rgb_only_desync(root, args.allow_incomplete)
        if args.depth_ckpt and "1" in req:
            # ①d: trainer-identical episode-holdout median AbsRel (frozen §4.1).
            s1d_holdout = _score_1d_holdout(
                root, Path(args.depth_ckpt),
                device=args.device, window=int(args.window), thr=thr,
                config_path=Path(args.config),
                holdout_frac=float(args.holdout_frac),
                split_seed=int(args.split_seed),
            )
        if args.depth_ckpt and "3" in req:
            # Peek n_frames from ckpt so scored frames have full temporal context.
            import torch  # lazy

            payload = torch.load(str(args.depth_ckpt), map_location="cpu")
            n_frames = int(payload.get("n_frames", 4))
            n_context = max(0, n_frames - 1)
            windows = _sample_scale_windows(
                root,
                window=int(args.window),
                max_windows=max(int(args.max_windows), int(thr.min_scale_windows) * 4),
                n_context=n_context,
                min_forward_frac=float(thr.fwd_cos_min),
                min_motion_m=thr.min_motion_m,
                scale_depth_min_m=float(thr.scale_depth_min_m),
                scale_depth_max_m=float(thr.scale_depth_max_m),
                support_ratio=float(thr.scale_support_ratio),
            )
            if not windows:
                print(
                    "[v0-gate] WARN: no approach-support windows for ③; "
                    "falling back to prefix sampler (proxy may be invalid).",
                    file=sys.stderr,
                )
                windows = _sample_windows(
                    root, window=int(args.window), max_windows=int(args.max_windows)
                )
                n_context = 0
            if not windows:
                print(f"[v0-gate] FAIL: no episode >= {args.window} steps for depth eval",
                      file=sys.stderr)
                return 1
            arr, pred_full = _predict_depth_over_windows(
                Path(args.depth_ckpt), windows, device=args.device, score_context=n_context,
            )
            pred_depth = pred_full
            gt_depth = arr.get("depth")
            vel = arr.get("vel")
            timestamps = arr.get("timestamps")
            # ③ (reprojection, §4.1 rev 2026-08-10): GT metric pose over the
            # scored frames (skip the n_context prefix stripped off pred_full).
            L_score = int(pred_full.shape[1])
            pos_s3 = np.asarray(
                [[np.asarray(windows[b][n_context + t].obs.position, dtype=np.float64).reshape(3)
                  for t in range(L_score)] for b in range(len(windows))], dtype=np.float64)
            yaw_s3 = np.asarray(
                [[float(windows[b][n_context + t].obs.yaw) for t in range(L_score)]
                 for b in range(len(windows))], dtype=np.float64)

    # --- score only the requested signals ------------------------------------ #
    signals: Dict[str, Dict[str, Any]] = {}
    if "1" in req:
        s1abc = _signal1abc_from_log(
            Path(args.learning_log) if args.learning_log else None, thr
        )
        if s1d_holdout is not None:
            s1d = s1d_holdout
        else:
            # No depth ckpt → explicit FAIL (depth pillar enforced).
            s1d = _signal1d(None, None, thr)
        signals["1"] = {"ok": bool(s1abc.get("ok") and s1d.get("ok")), "abc": s1abc, "d": s1d}
    if "3" in req:
        signals["3"] = _signal3(pred_depth, pos_s3, yaw_s3, thr)

    if need_rollout and ({"2", "4"} & req):
        import yaml as _yaml

        env_cfg = (_yaml.safe_load(Path(args.config).read_text()) or {}).get("env", {}) or {}
        backend = str(env_cfg.get("backend", "mock")).lower()
        if backend != "airsim" and not args.allow_mock_rollout:
            # Fail-CLOSED: a mock/analytic env cannot authoritatively pass ②/④.
            # The goal-seeking baseline trivially out-progresses random on an
            # analytic env, and there are no real obstacles to exercise the
            # shield — scoring ②/④ here reproduces the class of false pass that
            # invalidated the single-pillar checkpoint. The real pass runs on
            # the 4090 renderer (env.backend=airsim). --allow-mock-rollout
            # yields NON-authoritative numbers for a dev smoke only.
            reason = (
                f"②/④ require env.backend='airsim' (got '{backend}'); a mock/analytic "
                "env cannot authoritatively pass them. Run on the 4090 renderer, or "
                "pass --allow-mock-rollout for a non-authoritative dev check."
            )
            s2 = {"ok": False, "reason": reason, "backend": backend}
            s4 = {"ok": False, "reason": reason, "backend": backend}
        else:
            s2, s4 = _signals_2_4_from_rollouts(
                Path(args.config), thr_eff,
                depth_ckpt=Path(args.depth_ckpt) if args.depth_ckpt else None,
                device=args.device,
                n_episodes=thr_eff.n_eval_episodes,
                max_steps=int(args.max_steps),
                seed=int(args.seed),
                rollout_dataset=Path(args.rollout_dataset) if args.rollout_dataset else None,
            )
        if "2" in req:
            signals["2"] = s2
        if "4" in req:
            signals["4"] = s4

    # --- assemble: full run = authoritative gate; subset = partial ----------- #
    if req == set(_ALL_SIGNALS):
        verdict = metrics.aggregate_v0_verdict(signals)
        # aggregate_v0_verdict records DEFAULT_THRESHOLDS; overwrite with the
        # EFFECTIVE thresholds so a --n-eval-episodes override is faithfully
        # recorded (②/④ actually ran with thr_eff.n_eval_episodes).
        verdict["thresholds"] = asdict(thr_eff)
        _emit(verdict, args.emit)
        print(f"[v0-gate] {'PASS' if verdict['ok'] else 'FAIL'}")
        return 0 if verdict["ok"] else 1

    # Strict ``is True`` (never bool(v.get("ok"))): a partial deserialized with
    # ok="false" must not coerce to a pass. Mirrors aggregate_v0_verdict.
    all_ok = bool(signals) and all(v.get("ok") is True for v in signals.values())
    partial = {
        "partial": True,
        "requested": sorted(req),
        "all_requested_ok": all_ok,
        "signals": signals,
        "thresholds": asdict(thr_eff),  # effective (reflects --n-eval-episodes)
    }
    _emit(partial, args.emit)
    print(f"[v0-gate] PARTIAL {sorted(req)}: {'PASS' if all_ok else 'FAIL'} "
          "(NOT the gate — merge all four with --merge for the authoritative verdict)")
    return 0 if all_ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
