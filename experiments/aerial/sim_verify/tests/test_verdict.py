"""Offline unit tests for the Fork decision matrix (no AirSim / no file IO).

Exercises the pure ``verdict.decide`` so the depth-rate gate is provably part of
the Fork A conjunction: a report that is otherwise complete but whose depth is
too slow to collect the V0 perception dataset must NOT earn Fork A.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import verdict  # noqa: E402


def _pass():
    return {"pass": True}


def _fail():
    return {"pass": False}


def _full_report(**overrides):
    """A report where every capability passes; override individual keys."""
    t2 = {
        "imu": _pass(),
        "barometer": _pass(),
        "gps": _pass(),
        "collision": _pass(),
        "depth": _pass(),
        "depth_rate": _pass(),
        "physics": _pass(),
        "continuous_frames": _pass(),
    }
    t2.update(overrides)
    return {
        "t0_connectivity": {"connected": True},
        "t1_render": _pass(),
        "t2_capability": t2,
    }


class TestVerdictMatrix(unittest.TestCase):
    def test_full_stack_is_fork_a(self):
        fork, code, _why, caps = verdict.decide(_full_report())
        self.assertEqual(fork, "A")
        self.assertEqual(code, 0)
        self.assertTrue(caps["depth_rate (L2d-rate)"])

    def test_slow_depth_blocks_fork_a(self):
        # Everything passes EXCEPT depth_rate -> must fall to A-, not A.
        fork, code, why, caps = verdict.decide(_full_report(depth_rate=_fail()))
        self.assertEqual(fork, "A-")
        self.assertEqual(code, 2)
        self.assertFalse(caps["depth_rate (L2d-rate)"])
        self.assertIn("depth_rate (L2d-rate)", why)

    def test_missing_depth_rate_key_blocks_fork_a(self):
        # Legacy report with no depth_rate probe at all -> not Fork A.
        rep = _full_report()
        del rep["t2_capability"]["depth_rate"]
        fork, _code, _why, caps = verdict.decide(rep)
        self.assertEqual(fork, "A-")
        self.assertFalse(caps["depth_rate (L2d-rate)"])

    def test_no_rgb_is_fork_b(self):
        rep = _full_report()
        rep["t1_render"] = _fail()
        fork, code, _why, _caps = verdict.decide(rep)
        self.assertEqual(fork, "B")
        self.assertEqual(code, 3)

    def test_height_needs_baro_or_gps(self):
        rep = _full_report(barometer=_fail(), gps=_fail())
        fork, _code, _why, caps = verdict.decide(rep)
        self.assertEqual(fork, "A-")
        self.assertFalse(caps["height baro|gps (L2b)"])

    def test_gps_alone_satisfies_height(self):
        fork, _code, _why, caps = verdict.decide(_full_report(barometer=_fail()))
        self.assertEqual(fork, "A")
        self.assertTrue(caps["height baro|gps (L2b)"])


if __name__ == "__main__":
    unittest.main()
