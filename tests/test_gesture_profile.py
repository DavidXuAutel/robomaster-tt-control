from __future__ import annotations

import tempfile
import unittest
from collections import deque
from pathlib import Path

from tt_control.gesture_control import GuidedTrainingSession, MediaPipeGestureBackend
from tt_control.gesture_profile import GestureProfile, dtw_distance, sequence_features

from tests.test_gesture_control import open_hand, snap_hand


def takeoff_sequence(frames: int = 14, distance: float = 0.20):
    return [open_hand(-distance * i / (frames - 1)) for i in range(frames)]


def land_sequence(frames: int = 12, variation: float = 0.0):
    sequence = []
    pinched = snap_hand(False)
    released = snap_hand(True)
    split = frames // 2
    for i in range(frames):
        if i < split:
            sequence.append([(x + variation, y, z) for x, y, z in pinched])
        else:
            alpha = (i - split + 1) / max(frames - split, 1)
            sequence.append([
                (
                    a[0] + (b[0] - a[0]) * alpha + variation,
                    a[1] + (b[1] - a[1]) * alpha,
                    0.0,
                )
                for a, b in zip(pinched, released)
            ])
    return sequence


def none_sequence(frames: int = 13, drift: float = 0.0):
    return [open_hand(drift * i / max(frames - 1, 1)) for i in range(frames)]


def make_samples():
    return {
        "takeoff": [
            takeoff_sequence(12 + i, 0.18 + i * 0.008) for i in range(5)
        ],
        "land": [land_sequence(11 + i, (i - 2) * 0.002) for i in range(5)],
        "none": [none_sequence(11 + i, (i - 2) * 0.001) for i in range(5)],
    }


class GestureProfileTests(unittest.TestCase):
    def test_profile_classifies_each_dynamic_gesture(self):
        profile = GestureProfile.build(make_samples())
        for expected, sequence in (
            ("takeoff", takeoff_sequence(15, 0.205)),
            ("land", land_sequence(14, 0.001)),
            ("none", none_sequence(15, 0.001)),
        ):
            best = profile.nearest(sequence)[0]
            self.assertEqual(best.label, expected)
            self.assertLessEqual(best.distance, best.threshold)

    def test_dtw_handles_different_action_speed(self):
        fast = sequence_features(takeoff_sequence(10, 0.20))
        slow = sequence_features(takeoff_sequence(20, 0.20))
        self.assertLess(dtw_distance(fast, slow), 0.12)

    def test_class_order_uses_raw_distance_not_broad_none_threshold(self):
        live = takeoff_sequence(12, 0.20)
        features = sequence_features(live)
        shifted = lambda amount: [  # noqa: E731 - compact synthetic template helper
            [value + amount for value in frame] for frame in features
        ]
        profile = GestureProfile(
            templates={"takeoff": [shifted(0.05)], "land": [], "none": [shifted(0.10)]},
            thresholds={"takeoff": 1.0, "land": 1.0, "none": 100.0},
            created_at="test",
        )
        self.assertEqual(profile.nearest(live)[0].label, "takeoff")

    def test_profile_save_never_overwrites(self):
        profile = GestureProfile.build(make_samples())
        with tempfile.TemporaryDirectory() as directory:
            first = profile.save_new(directory)
            second = profile.save_new(directory)
            self.assertNotEqual(first, second)
            self.assertEqual(GestureProfile.load(first).created_at, profile.created_at)
            self.assertEqual(len(list(Path(directory).glob("profile_*.json"))), 2)

    def test_online_matcher_emits_profile_event_after_confirmation(self):
        backend = MediaPipeGestureBackend.__new__(MediaPipeGestureBackend)
        backend._profile = GestureProfile.build(make_samples())
        backend._live_landmarks = deque()
        backend._last_match_at = -1e9
        backend._match_candidate = ""
        backend._match_hits = 0
        backend._profile_armed = True
        backend._hand_absent_since = None
        backend._profile_last_event = {"takeoff": -1e9, "land": -1e9}
        backend._events = deque()
        backend._status = ""
        for index, landmarks in enumerate(takeoff_sequence(24, 0.21)):
            backend._update_profile_match(index * 0.08, landmarks)
        # 手不离开画面，即使冷却时间已过也不能用同一段长动作再次触发。
        for index, landmarks in enumerate(takeoff_sequence(24, 0.21), start=50):
            backend._update_profile_match(index * 0.08, landmarks)
        self.assertEqual([event.kind for event in backend._events], ["takeoff"])
        backend._mark_hand_absent(10.0)
        backend._mark_hand_absent(10.2)
        self.assertFalse(backend._profile_armed)
        backend._mark_hand_absent(10.41)
        self.assertTrue(backend._profile_armed)


class GuidedTrainingSessionTests(unittest.TestCase):
    def test_guided_capture_saves_visible_hand_sequence(self):
        session = GuidedTrainingSession("takeoff", 0.0)
        session.TARGET_SAMPLES = 1
        session.update(1.20, open_hand())
        for i in range(8):
            session.update(1.25 + i * 0.20, open_hand(-i * 0.02))
        self.assertTrue(session.completed)
        self.assertEqual(session.count, 1)


if __name__ == "__main__":
    unittest.main()
