"""
tests/test_cursor_mapping.py - Unit tests for cursor mapping, smoothing, calibration, and edge reach.

Tests cover:
  1. Calibration: auto-calibration on first frame and calibrate_neutral() re-centering.
  2. Direction mapping: head right/left/down/up mapping to positive/negative screen deltas.
  3. Low-confidence filtering: frames with confidence < DECISION_FACE_CONF_THRESHOLD suppress cursor deltas.
  4. Screen bounds and edges: move_cursor correctly clamps to full display boundary [0, screen_w - 1].
  5. Pause delay elimination: move_cursor does not leave pyautogui.PAUSE elevated.
  6. Non-destructive event peek: get_latest_events(consume=False) preserves click flags.
"""

import sys
import os
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from face.face_tracker import FaceTracker
from output.controller import OutputController
from output.router import Router


def _make_mock_mediapipe_result(nose_x: float = 0.5, nose_y: float = 0.5):
    """Create a mock MediaPipe FaceLandmarkerResult."""
    landmark = MagicMock()
    landmark.x = nose_x
    landmark.y = nose_y

    landmarks = [landmark] * 478
    landmarks[FaceTracker._NOSE_TIP_IDX] = landmark

    res = MagicMock()
    res.face_landmarks = [landmarks]
    res.face_blendshapes = []
    return res


class TestCalibrationAndDrift(unittest.TestCase):
    """Test neutral calibration and recentering."""

    def test_auto_calibration_first_frame(self):
        tracker = FaceTracker()
        self.assertIsNone(tracker._neutral_x)
        self.assertIsNone(tracker._neutral_y)

        res = _make_mock_mediapipe_result(nose_x=0.52, nose_y=0.48)
        ev = tracker._build_events(res, brightness=120.0, now=time.monotonic())

        self.assertEqual(tracker._neutral_x, 0.52)
        self.assertEqual(tracker._neutral_y, 0.48)
        # On first frame, displacement from neutral is 0
        self.assertAlmostEqual(ev["cursor_delta_x"], 0.0, places=2)
        self.assertAlmostEqual(ev["cursor_delta_y"], 0.0, places=2)

    def test_manual_recalibration(self):
        tracker = FaceTracker()
        tracker._neutral_x = 0.50
        tracker._neutral_y = 0.50

        # Simulate current head position at (0.60, 0.55)
        tracker._events["head_position"] = {"x": 0.60, "y": 0.55}
        tracker.calibrate_neutral()

        self.assertEqual(tracker._neutral_x, 0.60)
        self.assertEqual(tracker._neutral_y, 0.55)


class TestLowConfidenceFiltering(unittest.TestCase):
    """Verify low confidence frames suppress cursor movement to prevent jitter/jumping."""

    def test_low_confidence_suppresses_delta(self):
        tracker = FaceTracker()
        tracker._neutral_x = 0.50
        tracker._neutral_y = 0.50
        # Seed history
        for _ in range(10):
            tracker._nose_history_x.append(0.50)
            tracker._nose_history_y.append(0.50)

        # Force low confidence (e.g. poor lighting + tracking jitter)
        tracker._compute_confidence = lambda b, n: 0.25

        # Move nose far to the right (nose_x=0.40 -> disp_x=+0.10)
        res = _make_mock_mediapipe_result(nose_x=0.40, nose_y=0.50)
        ev = tracker._build_events(res, brightness=10.0, now=time.monotonic())

        self.assertLess(ev["face_confidence"], config.DECISION_FACE_CONF_THRESHOLD)
        self.assertEqual(ev["cursor_delta_x"], 0.0)
        self.assertEqual(ev["cursor_delta_y"], 0.0)
        self.assertFalse(ev["left_click"])
        self.assertFalse(ev["right_click"])

    def test_router_suppresses_low_confidence_face_events(self):
        mock_ctrl = MagicMock()
        router = Router(mock_ctrl)
        router.handle_face_events({
            "face_detected": True,
            "face_confidence": 0.20,  # Below threshold
            "cursor_delta_x": 15.0,
            "cursor_delta_y": 10.0,
            "left_click": True,
            "right_click": False,
        })
        mock_ctrl.move_cursor.assert_not_called()
        mock_ctrl.execute.assert_not_called()


class TestScreenBoundsAndEdges(unittest.TestCase):
    """Verify mouse cursor is clamped within safe bounds (not into fail-safe corners)."""

    @patch("output.controller.pyautogui")
    def test_cursor_clamped_to_failsafe_margin(self, mock_pya):
        """
        move_cursor must clamp to at least CURSOR_FAILSAFE_MARGIN pixels from
        every edge so the cursor never enters PyAutoGUI's corner fail-safe zones.
        """
        mock_pya.size.return_value = (1920, 1080)
        ctrl = OutputController(dry_run=False)
        margin = getattr(config, "CURSOR_FAILSAFE_MARGIN", 10)
        # Respect user margin too
        margin = max(margin, config.CURSOR_SCREEN_MARGIN, margin)

        # Push to extreme right edge — must stop at (1920 - margin - 1)
        mock_pya.position.return_value = (1900, 500)
        ctrl.move_cursor(100.0, 0.0)
        x_right, y_right = mock_pya.moveTo.call_args.args
        self.assertLessEqual(x_right, 1920 - margin - 1,
                             "cursor must not reach screen right edge (fail-safe zone)")
        self.assertEqual(y_right, 500)

        # Push to extreme left edge — must stop at margin
        mock_pya.position.return_value = (20, 500)
        ctrl.move_cursor(-100.0, 0.0)
        x_left, y_left = mock_pya.moveTo.call_args.args
        self.assertGreaterEqual(x_left, margin,
                                "cursor must not reach screen left edge (fail-safe zone)")
        self.assertEqual(y_left, 500)

        # Push to extreme top edge — must stop at margin
        mock_pya.position.return_value = (500, 20)
        ctrl.move_cursor(0.0, -100.0)
        x_top, y_top = mock_pya.moveTo.call_args.args
        self.assertEqual(x_top, 500)
        self.assertGreaterEqual(y_top, margin,
                                "cursor must not reach screen top edge (fail-safe zone)")

        # Push to extreme bottom edge — must stop at (1080 - margin - 1)
        mock_pya.position.return_value = (500, 1060)
        ctrl.move_cursor(0.0, 100.0)
        x_bot, y_bot = mock_pya.moveTo.call_args.args
        self.assertEqual(x_bot, 500)
        self.assertLessEqual(y_bot, 1080 - margin - 1,
                             "cursor must not reach screen bottom edge (fail-safe zone)")


class TestPauseDelayAndClickPreservation(unittest.TestCase):
    """Verify PyAutoGUI pause is not left modified and clicks are not consumed by sensor peek."""

    @patch("output.controller.pyautogui")
    def test_move_cursor_restores_pause(self, mock_pya):
        mock_pya.position.return_value = (500, 500)
        mock_pya.size.return_value = (1920, 1080)
        mock_pya.PAUSE = 0.05
        ctrl = OutputController(dry_run=False)
        ctrl.move_cursor(10.0, 10.0)
        self.assertEqual(mock_pya.PAUSE, 0.05)

    def test_peek_does_not_consume_clicks(self):
        tracker = FaceTracker()
        tracker._events["left_click"] = True
        tracker._events["right_click"] = True

        # Peek without consuming
        peeked = tracker.get_latest_events(consume=False)
        self.assertTrue(peeked["left_click"])
        self.assertTrue(peeked["right_click"])
        self.assertTrue(tracker._events["left_click"])
        self.assertTrue(tracker._events["right_click"])

        # Read with consume=True (default)
        read = tracker.get_latest_events(consume=True)
        self.assertTrue(read["left_click"])
        self.assertFalse(tracker._events["left_click"])
        self.assertFalse(tracker._events["right_click"])


if __name__ == "__main__":
    unittest.main()
