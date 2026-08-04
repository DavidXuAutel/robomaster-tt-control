"""Offline unit tests for T2 numerical sanity helpers (no AirSim required)."""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import sanity  # noqa: E402


class TestImu(unittest.TestCase):
    def test_rejects_zeros(self):
        ok, note = sanity.imu_ok({"ang_vel": [0, 0, 0], "lin_acc": [0, 0, 0]})
        self.assertFalse(ok)
        self.assertIn("near zero", note)

    def test_accepts_gravity_like(self):
        ok, note = sanity.imu_ok({"ang_vel": [0.01, 0, 0], "lin_acc": [0.0, 0.0, -9.8]})
        self.assertTrue(ok)
        self.assertIn("lin_acc_mag", note)

    def test_rejects_nan(self):
        ok, _ = sanity.imu_ok({"ang_vel": [0, 0, 0], "lin_acc": [float("nan"), 0, 0]})
        self.assertFalse(ok)


class TestDepth(unittest.TestCase):
    def test_rejects_not_dense(self):
        ok, _ = sanity.depth_ok({"dense": False, "n_floats": 10, "n_finite": 10})
        self.assertFalse(ok)

    def test_rejects_constant(self):
        ok, note = sanity.depth_ok(
            {
                "dense": True,
                "n_floats": 100,
                "n_finite": 100,
                "finite_min": 5.0,
                "finite_max": 5.0,
                "finite_std": 0.0,
            }
        )
        self.assertFalse(ok)
        self.assertIn("constant", note)

    def test_accepts_varied(self):
        ok, _ = sanity.depth_ok(
            {
                "dense": True,
                "n_floats": 100,
                "n_finite": 90,
                "finite_min": 1.0,
                "finite_max": 40.0,
                "finite_std": 8.0,
            }
        )
        self.assertTrue(ok)


class TestContinuous(unittest.TestCase):
    def test_rejects_low_fps(self):
        ok, note = sanity.continuous_ok(
            {
                "monotonic": True,
                "fps": 2.0,
                "min_fps_required": 5.0,
                "frames_differ": True,
                "mean_abs_diff": 3.0,
            }
        )
        self.assertFalse(ok)
        self.assertIn("fps", note)

    def test_accepts_ok(self):
        ok, _ = sanity.continuous_ok(
            {
                "monotonic": True,
                "fps": 12.0,
                "min_fps_required": 5.0,
                "frames_differ": True,
                "mean_abs_diff": 4.0,
            }
        )
        self.assertTrue(ok)


class TestDepthRate(unittest.TestCase):
    def test_rejects_cross_net_slow(self):
        # ~0.7 Hz cross-net DepthPlanar path — readable but too slow for V0 collection.
        ok, note = sanity.depth_rate_ok(
            {"monotonic": True, "fps": 0.7, "min_fps_required": 5.0}
        )
        self.assertFalse(ok)
        self.assertIn("127.0.0.1", note)

    def test_rejects_non_monotonic(self):
        ok, note = sanity.depth_rate_ok(
            {"monotonic": False, "fps": 30.0, "min_fps_required": 5.0}
        )
        self.assertFalse(ok)
        self.assertIn("monotonic", note)

    def test_accepts_loopback_fast(self):
        # 4090 loopback clears tens of Hz.
        ok, note = sanity.depth_rate_ok(
            {"monotonic": True, "fps": 24.0, "min_fps_required": 8.0}
        )
        self.assertTrue(ok)
        self.assertIn("24", note)

    def test_default_floor_when_missing(self):
        # min_fps_required absent -> default floor 5.0 applies.
        ok, _ = sanity.depth_rate_ok({"monotonic": True, "fps": 3.0})
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
