from __future__ import annotations

import unittest

from tt_control.gesture_control import GestureSequenceDetector


def open_hand(y_shift: float = 0.0):
    """构造尺度稳定的简化 21 点张开手掌。"""
    pts = [(0.50, 0.72 + y_shift, 0.0) for _ in range(21)]
    values = {
        0: (0.50, 0.78),
        1: (0.43, 0.68), 2: (0.38, 0.61), 3: (0.33, 0.55), 4: (0.28, 0.50),
        5: (0.39, 0.62), 6: (0.38, 0.49), 7: (0.37, 0.37), 8: (0.36, 0.25),
        9: (0.50, 0.59), 10: (0.50, 0.44), 11: (0.50, 0.31), 12: (0.50, 0.18),
        13: (0.60, 0.62), 14: (0.62, 0.48), 15: (0.63, 0.37), 16: (0.64, 0.28),
        17: (0.70, 0.67), 18: (0.73, 0.56), 19: (0.74, 0.47), 20: (0.75, 0.39),
    }
    for i, (x, y) in values.items():
        pts[i] = (x, y + y_shift, 0.0)
    return pts


def snap_hand(released: bool):
    pts = open_hand()
    # 拇指与中指先捏合；释放后中指落到掌心，拇指移开。
    if not released:
        pts[4] = (0.47, 0.36, 0.0)
        pts[12] = (0.48, 0.35, 0.0)
    else:
        pts[4] = (0.31, 0.52, 0.0)
        pts[12] = (0.53, 0.66, 0.0)
    return pts


class GestureSequenceDetectorTests(unittest.TestCase):
    def test_open_palm_rise_emits_takeoff(self):
        detector = GestureSequenceDetector()
        events = []
        for i in range(9):
            events.extend(detector.update(i * 0.08, open_hand(-i * 0.018), 0.95))
        self.assertEqual([event.kind for event in events], ["takeoff"])

    def test_static_open_palm_does_not_takeoff(self):
        detector = GestureSequenceDetector()
        events = []
        for i in range(15):
            events.extend(detector.update(i * 0.08, open_hand(), 0.95))
        self.assertEqual(events, [])

    def test_visual_snap_emits_land(self):
        detector = GestureSequenceDetector()
        self.assertEqual(detector.update(1.0, snap_hand(False), 0.0), [])
        self.assertEqual(detector.update(1.05, snap_hand(False), 0.0), [])
        self.assertEqual(detector.update(1.10, snap_hand(False), 0.0), [])
        events = detector.update(1.18, snap_hand(True), 0.0)
        self.assertEqual([event.kind for event in events], ["land"])

    def test_whole_hand_translation_is_not_snap(self):
        detector = GestureSequenceDetector()
        detector.update(1.0, snap_hand(False), 0.0)
        detector.update(1.05, snap_hand(False), 0.0)
        detector.update(1.10, snap_hand(False), 0.0)
        moved = [(x, y - 0.2, z) for x, y, z in snap_hand(False)]
        self.assertEqual(detector.update(1.2, moved, 0.0), [])

    def test_single_frame_pinch_does_not_arm_snap(self):
        detector = GestureSequenceDetector()
        detector.update(1.0, snap_hand(False), 0.0)
        self.assertEqual(detector.update(1.05, snap_hand(True), 0.0), [])


if __name__ == "__main__":
    unittest.main()
