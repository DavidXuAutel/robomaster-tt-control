#!/usr/bin/env python3
"""Clean exclusive L2f (+ stationary IMU) retest.

Intended to run ON the renderer host after:
  1) no other AirSim clients
  2) AirVLN / OpenFly scene freshly started (not a connect-only stub launch)

Measures:
  - batch Scene grabs via one simGetImages([req]*N) RPC  -> effective fps
  - sequential grabs (legacy / primary gate signal)
  - stationary IMU |lin_acc| vs ~9.8 m/s^2 (no motion nudge)

Optional capture-resolution override (WAM working res):
  L2F_W / L2F_H  e.g. 224 224 (aerial OpenFly train/eval default)
  When set, expects the *running* AirSim camera CaptureSettings to already
  match (restart renderer after patching settings.json). This probe verifies
  response.width/height and refuses to claim a res-gated pass on mismatch.

Does NOT change L2F_MIN_FPS gate (spec target remains 5.0). Writes JSON under
$OUT or ./artifacts/l2f_clean_retest.json.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time

HOST = os.environ.get("AIRSIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("AIRSIM_PORT", "41451"))
VEHICLE = os.environ.get("AIRSIM_VEHICLE", "drone_1").strip()
CAMERA = os.environ.get("AIRSIM_CAMERA", "front_custom")
N = int(os.environ.get("L2F_N", "20"))
MIN_FPS = float(os.environ.get("L2F_MIN_FPS", "5.0"))  # target gate, unchanged
# Optional WAM working resolution (aerial train/eval is 224x224).
L2F_W = os.environ.get("L2F_W")
L2F_H = os.environ.get("L2F_H")
WANT_W = int(L2F_W) if L2F_W else None
WANT_H = int(L2F_H) if L2F_H else None
OUT = pathlib.Path(
    os.environ.get("OUT", "./artifacts/l2f_clean_retest.json")
)

vk = {"vehicle_name": VEHICLE} if VEHICLE else {}


def _mag(xs) -> float:
    return math.sqrt(sum(float(x) ** 2 for x in xs))


def _as_list(vec) -> list:
    if hasattr(vec, "x_val"):
        return [float(vec.x_val), float(vec.y_val), float(vec.z_val)]
    return [float(x) for x in vars(vec).values()]


def _check_res(r, label: str) -> dict:
    got = {"width": int(r.width), "height": int(r.height)}
    if WANT_W is None or WANT_H is None:
        got["res_match"] = None
        got["res_note"] = "L2F_W/H unset — measuring whatever CaptureSettings is"
        return got
    match = got["width"] == WANT_W and got["height"] == WANT_H
    got["wanted"] = [WANT_W, WANT_H]
    got["res_match"] = match
    if not match:
        got["res_note"] = (
            f"{label}: got {got['width']}x{got['height']}, "
            f"wanted {WANT_W}x{WANT_H} — restart renderer after patching settings"
        )
    return got


def main() -> int:
    import airsim  # type: ignore
    import numpy as np  # type: ignore

    report: dict = {
        "host": HOST,
        "port": PORT,
        "vehicle": VEHICLE,
        "camera": CAMERA,
        "n": N,
        "l2f_min_fps_target": MIN_FPS,
        "l2f_wanted_wh": [WANT_W, WANT_H] if WANT_W and WANT_H else None,
        "mode": "clean_exclusive_batch",
    }

    c = airsim.MultirotorClient(ip=HOST, port=PORT)
    c.confirmConnection()
    report["client"] = "MultirotorClient"
    try:
        c.enableApiControl(True, **vk)
    except Exception as e:  # noqa: BLE001
        report["enableApiControl_error"] = repr(e)

    # --- stationary IMU (no motion) ---
    samples = []
    for _ in range(10):
        d = c.getImuData(**vk)
        lin = _as_list(d.linear_acceleration)
        ang = _as_list(d.angular_velocity)
        samples.append({"lin_acc": lin, "ang_vel": ang, "mag": _mag(lin)})
        time.sleep(0.05)
    mags = [s["mag"] for s in samples]
    mag_mean = sum(mags) / len(mags)
    near_ms2 = abs(mag_mean - 9.80665) < 2.5
    near_g = abs(mag_mean - 1.0) < 0.35
    report["imu_stationary"] = {
        "samples": samples,
        "mag_mean": mag_mean,
        "mag_std": float(np.std(mags)),
        "near_9p8_ms2": bool(near_ms2),
        "near_1g": bool(near_g),
        "pass": bool(near_ms2 or near_g),
        "note": "prefer ~9.8 m/s^2; ~1.0 accepted only as 'reported in g' annotation",
    }

    cams = [CAMERA, "front_custom", "0", "front_center"]

    # --- batch capture ---
    batch: dict = {"pass": False}
    last = None
    for cam in cams:
        try:
            reqs = [
                airsim.ImageRequest(cam, airsim.ImageType.Scene, False, True)
                for _ in range(N)
            ]
            t0 = time.perf_counter()
            responses = c.simGetImages(reqs, **vk)
            t1 = time.perf_counter()
            nbytes = [len(r.image_data_uint8) for r in responses]
            if not all(n > 0 for n in nbytes):
                raise RuntimeError(f"empty buffers: {nbytes[:3]}...")
            elapsed = t1 - t0
            fps = (N - 1) / elapsed if elapsed > 0 and N > 1 else 0.0
            res_info = _check_res(responses[0], "batch")
            import cv2  # type: ignore

            def dec(r):
                arr = np.frombuffer(bytes(r.image_data_uint8), dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)

            a, b = dec(responses[0]), dec(responses[-1])
            mean_abs = None
            if a is not None and b is not None and a.shape == b.shape:
                mean_abs = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
            res_ok = res_info.get("res_match") in (True, None)
            batch = {
                "pass": bool(fps >= MIN_FPS and res_ok),
                "camera": cam,
                "n_frames": N,
                "elapsed_s": elapsed,
                "fps": fps,
                "bytes0": nbytes[0],
                "mean_abs_diff_first_last": mean_abs,
                "meets_target_5fps": fps >= 5.0,
                "method": "batch_simGetImages",
                **res_info,
            }
            break
        except Exception as e:  # noqa: BLE001
            last = e
            batch = {"pass": False, "error": repr(e), "camera_tried": cam}
    if not batch.get("fps"):
        batch["error"] = batch.get("error") or repr(last)
    report["l2f_batch"] = batch

    # --- sequential capture (primary) ---
    seq: dict = {"pass": False}
    for cam in cams:
        try:
            timestamps = []
            sizes = []
            for _ in range(N):
                t_cap = time.perf_counter()
                r = c.simGetImages(
                    [airsim.ImageRequest(cam, airsim.ImageType.Scene, False, True)],
                    **vk,
                )[0]
                if len(r.image_data_uint8) <= 0:
                    raise RuntimeError("empty scene")
                timestamps.append(t_cap)
                sizes.append((int(r.width), int(r.height)))
            elapsed = timestamps[-1] - timestamps[0]
            fps = (N - 1) / elapsed if elapsed > 0 else 0.0
            dts = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
            # fake response-like object for res check
            class _R:
                width, height = sizes[0]

            res_info = _check_res(_R, "sequential")
            res_ok = res_info.get("res_match") in (True, None)
            seq = {
                "pass": bool(fps >= MIN_FPS and res_ok),
                "camera": cam,
                "n_frames": N,
                "elapsed_s": elapsed,
                "fps": fps,
                "dt_mean": float(sum(dts) / len(dts)) if dts else None,
                "monotonic": all(dt > 0 for dt in dts),
                "meets_target_5fps": fps >= 5.0,
                "method": "sequential_simGetImages",
                "observed_sizes": sizes[:3] + (["..."] if len(sizes) > 3 else []),
                **res_info,
            }
            break
        except Exception as e:  # noqa: BLE001
            seq = {"pass": False, "error": repr(e)}
    report["l2f_sequential"] = seq

    report["verdict"] = {
        "imu_stationary_ok": report["imu_stationary"]["pass"],
        "imu_mag_mean": report["imu_stationary"]["mag_mean"],
        "wanted_wh": [WANT_W, WANT_H] if WANT_W and WANT_H else None,
        "got_wh": [seq.get("width"), seq.get("height")],
        "res_match": seq.get("res_match"),
        "batch_fps": batch.get("fps"),
        "sequential_fps": seq.get("fps"),
        "batch_meets_L2F_MIN_FPS_5": bool(batch.get("meets_target_5fps")),
        "sequential_meets_L2F_MIN_FPS_5": bool(seq.get("meets_target_5fps")),
        "fork_a_at_working_res": bool(
            seq.get("meets_target_5fps")
            and report["imu_stationary"]["pass"]
            and (seq.get("res_match") in (True, None))
        ),
        "note": (
            "If sequential_meets_L2F_MIN_FPS_5 at WAM WH → L2f OK at working res. "
            "Else ~3fps is a real constraint for v2 rollout design."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["verdict"], indent=2))
    print(f"[l2f_clean] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
