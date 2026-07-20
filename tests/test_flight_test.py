from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tt_control.flight_test import FlightTestRecorder


class FlightTestRecorderTests(unittest.TestCase):
    def test_recorder_creates_append_only_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FlightTestRecorder(directory)
            path = recorder.path
            recorder.record("armed", battery=90)
            recorder.record("passed", height=0)
            recorder.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["event"] for row in rows], ["armed", "passed"])
            self.assertEqual(rows[0]["battery"], 90)
            self.assertEqual(len(list(Path(directory).glob("gesture_flight_*.jsonl"))), 1)


if __name__ == "__main__":
    unittest.main()
