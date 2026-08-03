#!/usr/bin/env python3
"""T2 — Direct AirSim capability probe (L2, the decisive test).

Bypasses the thin OpenFly wrapper and talks straight to AirSim to see which of
the signals v2 needs actually exist on this renderer:

  L2a IMU              getImuData (+ numerical sanity)
  L2b baro/gps         getBarometerData / getGpsData (+ finite altitude)
  L2c collision        simGetCollisionInfo (field readable)
  L2d depth GT         simGetImages(DepthPlanar) (+ dense + dynamic range)
  L2e physics          moveByVelocityAsync -> altitude actually changes
  L2f continuous RGB   N frames at fixed interval: fps / monotonic / inter-frame diff

Every probe is wrapped in try/except and REPORTS actual availability — it does
not assume a particular AirSim fork's API. API-readable alone is not enough:
sanity checks fold into `pass`. Merges under "t2_capability".

Run from the H100 client against $AIRSIM_HOST:$AIRSIM_PORT. Needs `airsim`.
Optional: `numpy` for depth stats / frame diffs (falls back to stdlib if absent).
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import report  # noqa: E402
from lib import sanity  # noqa: E402

HOST = os.environ.get("AIRSIM_HOST", "127.0.0.1")
PORT = int(os.environ.get("AIRSIM_PORT", "41451"))
CAMERAS = [os.environ.get("AIRSIM_CAMERA", "0"), "front_center"]
L2F_N = int(os.environ.get("L2F_N", "10"))
L2F_INTERVAL_S = float(os.environ.get("L2F_INTERVAL_S", "0.05"))  # target ~20 Hz
L2F_MIN_FPS = float(os.environ.get("L2F_MIN_FPS", "5.0"))

res: dict = {
    "host": HOST,
    "port": PORT,
    "l2f_n": L2F_N,
    "l2f_interval_s": L2F_INTERVAL_S,
    "l2f_min_fps": L2F_MIN_FPS,
}


def _probe(name: str, fn, sanity_fn=None) -> None:
    try:
        detail = fn()
        entry: dict = {"pass": True, "detail": detail}
        if sanity_fn is not None:
            ok, note = sanity_fn(detail)
            entry["sanity"] = note
            entry["pass"] = bool(ok)
            if not ok:
                entry["error"] = f"sanity failed: {note}"
        res[name] = entry
    except Exception as e:  # noqa: BLE001
        res[name] = {"pass": False, "error": repr(e)}


try:
    import airsim  # type: ignore
except Exception as e:  # noqa: BLE001
    res["import_airsim"] = {"pass": False, "error": repr(e)}
    report.merge("t2_capability", res)
    print("[T2]", res)
    sys.exit(1)

try:
    import numpy as np  # type: ignore
except Exception:  # noqa: BLE001
    np = None  # type: ignore

# Connect (prefer MultirotorClient; fall back to VehicleClient for CV mode).
try:
    client = airsim.MultirotorClient(ip=HOST, port=PORT)
    client.confirmConnection()
    res["client"] = "MultirotorClient"
except Exception:  # noqa: BLE001
    client = airsim.VehicleClient(ip=HOST, port=PORT)
    client.confirmConnection()
    res["client"] = "VehicleClient"

c = client


def _as_list(vec) -> list:
    if hasattr(vec, "x_val"):
        return [float(vec.x_val), float(vec.y_val), float(vec.z_val)]
    return list(vars(vec).values())


# --- L2a IMU ---
def _imu():
    d = c.getImuData()
    return {
        "ang_vel": _as_list(d.angular_velocity),
        "lin_acc": _as_list(d.linear_acceleration),
    }


_probe("imu", _imu, sanity.imu_ok)

# --- L2b barometer / gps ---
_probe("barometer", lambda: {"altitude": float(c.getBarometerData().altitude)}, sanity.altitude_ok)
_probe(
    "gps",
    lambda: {"alt": float(c.getGpsData().gnss.geo_point.altitude)},
    lambda d: sanity.altitude_ok(d, "alt"),
)

# --- L2c collision (readable field is enough; has_collided may be False) ---
_probe("collision", lambda: {"has_collided": bool(c.simGetCollisionInfo().has_collided)})

# --- L2d depth GT ---
def _depth():
    last = None
    for cam in CAMERAS:
        try:
            rq = [airsim.ImageRequest(cam, airsim.ImageType.DepthPlanar, True, False)]
            r = c.simGetImages(rq)[0]
            if not (r.width and r.height):
                continue
            floats = list(r.image_data_float)
            n = len(floats)
            dense = n == int(r.width) * int(r.height)
            detail = {
                "camera": cam,
                "w": int(r.width),
                "h": int(r.height),
                "n_floats": n,
                "dense": dense,
            }
            if np is not None and n:
                arr = np.asarray(floats, dtype=np.float64)
                fin = arr[np.isfinite(arr)]
                detail["n_finite"] = int(fin.size)
                if fin.size:
                    detail["finite_min"] = float(fin.min())
                    detail["finite_max"] = float(fin.max())
                    detail["finite_std"] = float(fin.std())
                else:
                    detail["finite_min"] = detail["finite_max"] = detail["finite_std"] = None
            else:
                # stdlib fallback
                fin = [float(x) for x in floats if x == x and abs(x) != float("inf")]
                detail["n_finite"] = len(fin)
                if fin:
                    detail["finite_min"] = min(fin)
                    detail["finite_max"] = max(fin)
                    mean = sum(fin) / len(fin)
                    detail["finite_std"] = (sum((x - mean) ** 2 for x in fin) / len(fin)) ** 0.5
                else:
                    detail["finite_min"] = detail["finite_max"] = detail["finite_std"] = None
            return detail
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"no camera returned depth (tried {CAMERAS}): {last!r}")


_probe("depth", _depth, sanity.depth_ok)


# --- L2f continuous frames (fps + monotonic + inter-frame change) ---
def _decode_scene(r):
    raw = bytes(r.image_data_uint8)
    if np is None:
        # coarse fingerprint without numpy/cv2
        return None, len(raw), sum(raw[:: max(1, len(raw) // 256)]) if raw else 0
    try:
        import cv2  # type: ignore

        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, len(raw), int(arr.mean()) if arr.size else 0
        return img, len(raw), float(img.mean())
    except Exception:  # noqa: BLE001
        arr = np.frombuffer(raw, dtype=np.uint8)
        return None, len(raw), float(arr.mean()) if arr.size else 0.0


def _grab_scene(cam: str):
    rq = [airsim.ImageRequest(cam, airsim.ImageType.Scene, False, True)]
    return c.simGetImages(rq)[0]


def _maybe_nudge():
    """Introduce a tiny viewpoint change so consecutive frames can differ."""
    try:
        if res["client"] == "MultirotorClient":
            c.enableApiControl(True)
            c.armDisarm(True)
            c.moveByVelocityAsync(0.5, 0.0, 0.0, 0.3).join()
            return "velocity_nudge"
        # CV / VehicleClient: small teleport if available
        pose = c.simGetVehiclePose()
        pose.position.x_val += 0.3
        c.simSetVehiclePose(pose, True)
        return "teleport_nudge"
    except Exception as e:  # noqa: BLE001
        return f"nudge_failed:{e!r}"


def _continuous():
    last = None
    for cam in CAMERAS:
        try:
            # warm-up + camera select
            _grab_scene(cam)
            nudge = _maybe_nudge()
            timestamps: list[float] = []
            means: list[float] = []
            imgs = []
            t_start = time.perf_counter()
            for i in range(L2F_N):
                t_target = t_start + i * L2F_INTERVAL_S
                now = time.perf_counter()
                if t_target > now:
                    time.sleep(t_target - now)
                t_cap = time.perf_counter()
                r = _grab_scene(cam)
                img, nbytes, mean = _decode_scene(r)
                if nbytes <= 0:
                    raise RuntimeError("empty scene buffer")
                timestamps.append(t_cap)
                means.append(float(mean))
                if img is not None:
                    imgs.append(img)

            dts = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
            monotonic = all(dt > 0 for dt in dts)
            elapsed = timestamps[-1] - timestamps[0]
            fps = (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0

            mean_abs_diff = 0.0
            frames_differ = False
            if len(imgs) >= 2 and np is not None:
                diffs = []
                for a, b in zip(imgs[:-1], imgs[1:]):
                    if a.shape != b.shape:
                        continue
                    diffs.append(float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))))
                if diffs:
                    mean_abs_diff = sum(diffs) / len(diffs)
                    frames_differ = mean_abs_diff > 0.5  # >0.5 gray-level mean abs diff
            else:
                # fingerprint fallback
                mean_abs_diff = float(
                    sum(abs(means[i + 1] - means[i]) for i in range(len(means) - 1))
                    / max(1, len(means) - 1)
                )
                frames_differ = mean_abs_diff > 1e-3

            return {
                "camera": cam,
                "n_frames": L2F_N,
                "interval_s_target": L2F_INTERVAL_S,
                "min_fps_required": L2F_MIN_FPS,
                "fps": fps,
                "monotonic": monotonic,
                "dt_mean": float(sum(dts) / len(dts)) if dts else None,
                "dt_min": float(min(dts)) if dts else None,
                "dt_max": float(max(dts)) if dts else None,
                "mean_abs_diff": mean_abs_diff,
                "frames_differ": frames_differ,
                "nudge": nudge,
                "fps_ok": fps >= L2F_MIN_FPS,
            }
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"continuous capture failed (tried {CAMERAS}): {last!r}")


_probe("continuous_frames", _continuous, sanity.continuous_ok)

# Keep a one-shot scene key for backward-compatible report readers.
if res.get("continuous_frames", {}).get("pass"):
    d = res["continuous_frames"]["detail"]
    res["scene"] = {
        "pass": True,
        "detail": {"camera": d.get("camera"), "bytes": "see continuous_frames", "via": "L2f"},
    }
else:
    # still try a single grab so report shows whether Scene API exists at all
    def _scene_once():
        last = None
        for cam in CAMERAS:
            try:
                r = _grab_scene(cam)
                if len(r.image_data_uint8) > 0:
                    return {"camera": cam, "bytes": len(r.image_data_uint8)}
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"no camera returned scene (tried {CAMERAS}): {last!r}")

    _probe("scene", _scene_once)


# --- L2e physics (only meaningful with MultirotorClient) ---
def _physics():
    if res["client"] != "MultirotorClient":
        raise RuntimeError("VehicleClient (ComputerVision) — no flight dynamics")
    c.enableApiControl(True)
    c.armDisarm(True)
    z0 = c.getMultirotorState().kinematics_estimated.position.z_val
    c.moveByVelocityAsync(0, 0, -2, 1).join()  # climb 1s (NED: -z up)
    z1 = c.getMultirotorState().kinematics_estimated.position.z_val
    try:
        c.armDisarm(False)
        c.enableApiControl(False)
    except Exception:  # noqa: BLE001
        pass
    dz = float(z1 - z0)
    return {"dz": dz, "moved": abs(dz) > 0.2}


def _physics_ok(detail: dict):
    if not detail.get("moved"):
        return False, f"dz={detail.get('dz')} — no significant altitude change (teleport-only?)"
    return True, f"dz={detail.get('dz')}"


_probe("physics", _physics, _physics_ok)

report.merge("t2_capability", res)
print("[T2] capability:")
for k, v in res.items():
    print(f"  {k}: {v}")
sys.exit(0)
