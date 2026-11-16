"""
tests/test_router.py - Unit tests for output/router.py (Stage 4).

All tests use mocked controllers — no real hardware access.

Run with:
    python -m pytest tests/test_router.py -v
"""

import sys
import pathlib
import threading
import unittest
from unittest.mock import MagicMock, call

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from output.router import Router
from output.controller import ActionResult


def _make_mock_controller(stopped: bool = False) -> MagicMock:
    ctrl = MagicMock()
    ctrl.is_stopped.return_value = stopped
    ctrl.execute.return_value = ActionResult(success=True, command="click", message="OK")
    ctrl.type_dictation.return_value = ActionResult(success=True, command="dictation", message="OK")
    return ctrl


# ---------------------------------------------------------------------------
# Face event handling
# ---------------------------------------------------------------------------
class TestRouterFaceEvents(unittest.TestCase):

    def test_face_left_click_forwarded(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_face_events({
            "face_detected": True,
            "face_confidence": 0.9,
            "cursor_delta_x": 0.0,
            "cursor_delta_y": 0.0,
            "left_click": True,
            "right_click": False,
        })
        ctrl.execute.assert_called_once_with("click", confidence=0.9, source="face")

    def test_face_right_click_forwarded(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_face_events({
            "face_detected": True,
            "face_confidence": 0.8,
            "cursor_delta_x": 0.0,
            "cursor_delta_y": 0.0,
            "left_click": False,
            "right_click": True,
        })
        ctrl.execute.assert_called_once_with("right_click", confidence=0.8, source="face")

    def test_face_cursor_move_forwarded(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_face_events({
            "face_detected": True,
            "face_confidence": 0.7,
            "cursor_delta_x": 5.0,
            "cursor_delta_y": -3.0,
            "left_click": False,
            "right_click": False,
        })
        ctrl.move_cursor.assert_called_once_with(5.0, -3.0)

    def test_face_no_move_when_delta_zero(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_face_events({
            "face_detected": True,
            "face_confidence": 0.7,
            "cursor_delta_x": 0.0,
            "cursor_delta_y": 0.0,
            "left_click": False,
            "right_click": False,
        })
        ctrl.move_cursor.assert_not_called()

    def test_face_no_move_when_not_detected(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_face_events({
            "face_detected": False,
            "face_confidence": 0.0,
            "cursor_delta_x": 10.0,
            "cursor_delta_y": 10.0,
            "left_click": False,
            "right_click": False,
        })
        ctrl.move_cursor.assert_not_called()

    def test_face_no_click_when_flags_false(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_face_events({
            "face_detected": True,
            "face_confidence": 0.9,
            "cursor_delta_x": 2.0,
            "cursor_delta_y": 1.0,
            "left_click": False,
            "right_click": False,
        })
        ctrl.execute.assert_not_called()

    def test_face_both_clicks_forwarded(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_face_events({
            "face_detected": True,
            "face_confidence": 0.95,
            "cursor_delta_x": 0.0,
            "cursor_delta_y": 0.0,
            "left_click": True,
            "right_click": True,
        })
        self.assertEqual(ctrl.execute.call_count, 2)
        calls = [c.args[0] for c in ctrl.execute.call_args_list]
        self.assertIn("click", calls)
        self.assertIn("right_click", calls)


# ---------------------------------------------------------------------------
# Speech event handling
# ---------------------------------------------------------------------------
class TestRouterSpeechEvents(unittest.TestCase):

    def test_speech_command_forwarded(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_speech_events({
            "command": "scroll_up",
            "command_phrase": "scroll up",
            "speech_confidence": 0.85,
            "dictation_text": None,
        })
        ctrl.execute.assert_called_once_with("scroll_up", confidence=0.85, source="speech")

    def test_speech_dictation_forwarded(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_speech_events({
            "command": None,
            "speech_confidence": 0.9,
            "dictation_text": "hello world",
        })
        ctrl.type_dictation.assert_called_once_with("hello world")

    def test_speech_stop_control_triggers_emergency(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_speech_events({
            "command": "stop_control",
            "speech_confidence": 0.95,
            "dictation_text": None,
        })
        ctrl.trigger_emergency_stop.assert_called_once()
        # execute should NOT be called for stop_control
        ctrl.execute.assert_not_called()

    def test_speech_no_action_when_no_command_no_dictation(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_speech_events({
            "command": None,
            "speech_confidence": 0.5,
            "dictation_text": None,
        })
        ctrl.execute.assert_not_called()
        ctrl.type_dictation.assert_not_called()

    def test_speech_confidence_passed_through(self):
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        router.handle_speech_events({
            "command": "click",
            "speech_confidence": 0.42,
            "dictation_text": None,
        })
        ctrl.execute.assert_called_once_with("click", confidence=0.42, source="speech")


# ---------------------------------------------------------------------------
# CSV logger integration (optional logger)
# ---------------------------------------------------------------------------
class TestRouterCsvLogger(unittest.TestCase):

    def test_csv_logged_on_face_click(self):
        ctrl = _make_mock_controller()
        csv = MagicMock()
        router = Router(ctrl, csv_logger=csv)
        router.handle_face_events({
            "face_detected": True,
            "face_confidence": 0.9,
            "cursor_delta_x": 0.0,
            "cursor_delta_y": 0.0,
            "left_click": True,
            "right_click": False,
        })
        csv.log.assert_called()

    def test_csv_logged_on_speech_command(self):
        ctrl = _make_mock_controller()
        csv = MagicMock()
        router = Router(ctrl, csv_logger=csv)
        router.handle_speech_events({
            "command": "click",
            "speech_confidence": 0.8,
            "dictation_text": None,
        })
        csv.log.assert_called()

    def test_no_csv_no_crash(self):
        """Router without csv_logger must not raise."""
        ctrl = _make_mock_controller()
        router = Router(ctrl, csv_logger=None)
        # Should complete without exception
        router.handle_face_events({
            "face_detected": True, "face_confidence": 0.9,
            "cursor_delta_x": 1.0, "cursor_delta_y": 1.0,
            "left_click": True, "right_click": False,
        })
        router.handle_speech_events({
            "command": "click", "speech_confidence": 0.8, "dictation_text": None,
        })


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
class TestRouterThreadSafety(unittest.TestCase):

    def test_concurrent_face_speech(self):
        """Router must handle concurrent face and speech calls without deadlock."""
        ctrl = _make_mock_controller()
        router = Router(ctrl)
        errors = []

        def face_calls():
            for _ in range(50):
                try:
                    router.handle_face_events({
                        "face_detected": True, "face_confidence": 0.8,
                        "cursor_delta_x": 1.0, "cursor_delta_y": 0.5,
                        "left_click": False, "right_click": False,
                    })
                except Exception as e:
                    errors.append(e)

        def speech_calls():
            for _ in range(50):
                try:
                    router.handle_speech_events({
                        "command": "scroll_up", "speech_confidence": 0.8,
                        "dictation_text": None,
                    })
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=face_calls)
        t2 = threading.Thread(target=speech_calls)
        t1.start(); t2.start()
        t1.join(timeout=5); t2.join(timeout=5)
        self.assertEqual(errors, [], f"Thread errors: {errors}")


if __name__ == "__main__":
    unittest.main()

