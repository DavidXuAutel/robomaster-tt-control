from __future__ import annotations

import unittest

from tt_control.app import App, ConnState
from tt_control.config import AppConfig
from tt_control.inference import InferenceEvent, PassthroughBackend


class _FakeClient:
    def __init__(self, battery: int = 80, height: int = 0) -> None:
        self.state = {"bat": str(battery), "h": str(height)}


class _FakeFlightClient(_FakeClient):
    def __init__(self) -> None:
        super().__init__(battery=90, height=0)
        self.calls: list[object] = []

    def rc(self, *axes) -> None:
        self.calls.append(("rc", axes))

    def takeoff(self):
        self.calls.append("takeoff")
        self.state["h"] = "40"
        return "ok"

    def land(self):
        self.calls.append("land")
        self.state["h"] = "0"
        return "ok"

    def height_cm(self):
        return int(self.state["h"])


class AppGestureEventTests(unittest.TestCase):
    def make_app(self, battery: int = 80, height: int = 0):
        app = App(AppConfig(), PassthroughBackend())
        app._conn_state = ConnState.CONNECTED
        app.client = _FakeClient(battery, height)  # type: ignore[assignment]
        commands: list[str] = []
        app._async_flight_cmd = commands.append  # type: ignore[method-assign]
        return app, commands

    def test_takeoff_gesture_passes_when_grounded_and_battery_is_safe(self):
        app, commands = self.make_app(battery=80, height=0)
        app._handle_inference_event(InferenceEvent("takeoff", 0.9, "test"))
        self.assertEqual(commands, ["takeoff"])

    def test_takeoff_gesture_is_blocked_on_low_battery(self):
        app, commands = self.make_app(battery=20, height=0)
        app._handle_inference_event(InferenceEvent("takeoff", 0.9, "test"))
        self.assertEqual(commands, [])
        self.assertIn("battery 20%", app._hint)

    def test_land_gesture_is_ignored_on_ground(self):
        app, commands = self.make_app(height=0)
        app._handle_inference_event(InferenceEvent("land", 0.9, "test"))
        self.assertEqual(commands, [])

    def test_land_gesture_passes_when_height_reports_airborne(self):
        app, commands = self.make_app(height=60)
        app._handle_inference_event(InferenceEvent("land", 0.9, "test"))
        self.assertEqual(commands, ["land"])

    def test_dry_run_never_sends_flight_command(self):
        app, commands = self.make_app(height=60)
        app.config.gesture_commands_enabled = False
        app._handle_inference_event(InferenceEvent("land", 0.9, "test"))
        self.assertEqual(commands, [])
        self.assertIn("test PASS", app._hint)

    def test_dry_run_stops_after_takeoff_and_land_pass(self):
        app, commands = self.make_app(height=60)
        app.config.gesture_commands_enabled = False
        app._handle_inference_event(InferenceEvent("takeoff", 0.9, "test"))
        self.assertEqual(app._gesture_test_results, {"takeoff"})
        self.assertFalse(app._gesture_test_complete)
        app._last_inference_event = 0.0
        app._handle_inference_event(InferenceEvent("land", 0.9, "test"))
        self.assertEqual(commands, [])
        self.assertTrue(app._gesture_test_complete)
        self.assertEqual(app._gesture_banner, "GESTURE TEST PASSED")
        self.assertIn("inference stopped", app._hint)

    def test_real_flight_protocol_takeoff_hover_then_land_without_extra_up(self):
        app = App(AppConfig(gesture_flight_test=True), PassthroughBackend())
        app._conn_state = ConnState.CONNECTED
        client = _FakeFlightClient()
        app.client = client  # type: ignore[assignment]
        app._flight_test_state = "TAKING_OFF"
        app._run_flight_cmd("takeoff")
        self.assertEqual(app._flight_test_state, "HOVERING_WAIT_LAND")
        self.assertFalse(any(isinstance(call, tuple) and call[0] == "up" for call in client.calls))
        self.assertTrue(app._flying)
        app._flight_test_state = "LANDING"
        app._run_flight_cmd("land")
        self.assertEqual(app._flight_test_state, "PASSED")
        self.assertTrue(app._gesture_test_complete)
        self.assertFalse(app._flying)

    def test_real_flight_takeoff_gesture_requires_arm(self):
        app, commands = self.make_app(battery=90, height=0)
        app.config.gesture_flight_test = True
        app._handle_inference_event(InferenceEvent("takeoff", 0.9, "test"))
        self.assertEqual(commands, [])
        app._flight_test_state = "ARMED"
        app._last_inference_event = 0.0
        app._handle_inference_event(InferenceEvent("takeoff", 0.9, "test"))
        self.assertEqual(commands, ["takeoff"])
        self.assertEqual(app._flight_test_state, "TAKING_OFF")

    def test_fail_button_on_ground_freezes_test_and_is_idempotent(self):
        app, commands = self.make_app(battery=90, height=0)
        app.config.gesture_flight_test = True
        app._flight_test_state = "ARMED"
        app._fail_flight_test("test failure")
        self.assertEqual(app._flight_test_state, "FAILED")
        self.assertTrue(app._gesture_test_complete)
        self.assertEqual(commands, [])
        app._fail_flight_test("clicked twice")
        self.assertIn("already recorded", app._hint)


if __name__ == "__main__":
    unittest.main()
