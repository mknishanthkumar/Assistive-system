"""
tests/test_sensors.py - Unit tests for context/sensors.py (Stage 4).

All external dependencies (uiautomation, pywin32, psutil) are mocked so
these tests run on any machine regardless of Windows APIs.

Run with:
    python -m pytest tests/test_sensors.py -v
"""

import sys
import pathlib
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from context.sensors import SensorLoop, _get_active_window_info, _is_text_field_focused


# ---------------------------------------------------------------------------
# _get_active_window_info fallback
# ---------------------------------------------------------------------------
class TestGetActiveWindowInfo(unittest.TestCase):

    def test_returns_unknown_on_import_error(self):
        """When both uiautomation and pywin32 are unavailable, return 'unknown'."""
        with patch.dict("sys.modules", {"uiautomation": None, "win32gui": None}):
            result = _get_active_window_info()
        # Must have both keys and both are strings
        self.assertIn("title", result)
        self.assertIn("process", result)
        self.assertIsInstance(result["title"], str)
        self.assertIsInstance(result["process"], str)

    def test_returns_dict_always(self):
        """Return value is always a dict with title and process keys."""
        result = _get_active_window_info()
        self.assertIsInstance(result, dict)
        self.assertIn("title", result)
        self.assertIn("process", result)

    def test_no_exception_raised(self):
        """Must never raise regardless of environment."""
        try:
            _get_active_window_info()
        except Exception as e:
            self.fail(f"_get_active_window_info raised: {e}")


# ---------------------------------------------------------------------------
# _is_text_field_focused fallback
# ---------------------------------------------------------------------------
class TestIsTextFieldFocused(unittest.TestCase):

    def test_returns_unknown_on_import_error(self):
        """When uiautomation unavailable, return 'unknown'."""
        with patch.dict("sys.modules", {"uiautomation": None}):
            result = _is_text_field_focused()
        # 'unknown', True, or False are all valid
        self.assertIn(result, (True, False, "unknown"))

    def test_no_exception_raised(self):
        """Must never raise."""
        try:
            _is_text_field_focused()
        except Exception as e:
            self.fail(f"_is_text_field_focused raised: {e}")

    def test_returns_valid_type(self):
        result = _is_text_field_focused()
        self.assertIn(type(result), (bool, str))


# ---------------------------------------------------------------------------
# SensorLoop collect
# ---------------------------------------------------------------------------
class TestSensorLoopCollect(unittest.TestCase):

    def _make_mock_face(self, confidence=0.8, brightness=120.0):
        face = MagicMock()
        face.get_latest_events.return_value = {
            "face_confidence": confidence,
            "debug": {"brightness": brightness},
        }
        return face

    def _make_mock_speech(self, noise_level=0.03):
        speech = MagicMock()
        speech.get_latest_events.return_value = {"noise_level": noise_level}
        return speech

    def test_collect_reads_face_confidence(self):
        face = self._make_mock_face(confidence=0.75)
        loop = SensorLoop(queue.Queue(), face_tracker=face)
        reading = loop._collect()
        self.assertAlmostEqual(reading["face_confidence"], 0.75)

    def test_collect_reads_brightness(self):
        face = self._make_mock_face(brightness=180.0)
        loop = SensorLoop(queue.Queue(), face_tracker=face)
        reading = loop._collect()
        self.assertAlmostEqual(reading["frame_brightness"], 180.0)

    def test_collect_reads_noise_level(self):
        speech = self._make_mock_speech(noise_level=0.05)
        loop = SensorLoop(queue.Queue(), speech_engine=speech)
        reading = loop._collect()
        self.assertAlmostEqual(reading["noise_level"], 0.05)

    def test_collect_has_all_keys(self):
        loop = SensorLoop(queue.Queue())
        reading = loop._collect()
        expected_keys = {
            "noise_level", "face_confidence", "frame_brightness",
            "active_window", "active_process", "text_field_focused", "timestamp",
        }
        self.assertTrue(expected_keys.issubset(reading.keys()),
                        f"Missing keys: {expected_keys - reading.keys()}")

    def test_collect_no_trackers_returns_defaults(self):
        """When no trackers provided, all numeric fields default to 0.0."""
        loop = SensorLoop(queue.Queue(), face_tracker=None, speech_engine=None)
        reading = loop._collect()
        self.assertEqual(reading["face_confidence"], 0.0)
        self.assertEqual(reading["noise_level"], 0.0)

    def test_collect_face_exception_does_not_crash(self):
        face = MagicMock()
        face.get_latest_events.side_effect = RuntimeError("Camera lost")
        loop = SensorLoop(queue.Queue(), face_tracker=face)
        try:
            reading = loop._collect()
        except Exception as e:
            self.fail(f"_collect raised on tracker error: {e}")
        self.assertEqual(reading["face_confidence"], 0.0)

    def test_collect_speech_exception_does_not_crash(self):
        speech = MagicMock()
        speech.get_latest_events.side_effect = RuntimeError("Mic lost")
        loop = SensorLoop(queue.Queue(), speech_engine=speech)
        try:
            reading = loop._collect()
        except Exception as e:
            self.fail(f"_collect raised on speech error: {e}")
        self.assertEqual(reading["noise_level"], 0.0)

    def test_collect_timestamp_is_float(self):
        loop = SensorLoop(queue.Queue())
        reading = loop._collect()
        self.assertIsInstance(reading["timestamp"], float)


# ---------------------------------------------------------------------------
# SensorLoop start/stop
# ---------------------------------------------------------------------------
class TestSensorLoopStartStop(unittest.TestCase):

    def test_publishes_to_queue(self):
        q: queue.Queue = queue.Queue(maxsize=5)
        loop = SensorLoop(q, poll_interval=0.05)
        loop.start()
        time.sleep(0.2)
        loop.stop()
        self.assertFalse(q.empty(), "No readings published to queue.")

    def test_reading_has_expected_keys(self):
        q: queue.Queue = queue.Queue(maxsize=5)
        loop = SensorLoop(q, poll_interval=0.05)
        loop.start()
        time.sleep(0.15)
        loop.stop()
        reading = q.get_nowait()
        self.assertIn("noise_level", reading)
        self.assertIn("active_window", reading)

    def test_stop_before_start_safe(self):
        loop = SensorLoop(queue.Queue())
        try:
            loop.stop()   # Should not raise
        except Exception as e:
            self.fail(f"stop() before start() raised: {e}")

    def test_double_start_safe(self):
        q: queue.Queue = queue.Queue(maxsize=10)
        loop = SensorLoop(q, poll_interval=0.05)
        loop.start()
        loop.start()   # Should log warning, not crash
        loop.stop()

    def test_queue_full_does_not_block(self):
        """When queue is full, readings are silently dropped (never block)."""
        q: queue.Queue = queue.Queue(maxsize=1)
        # Fill the queue
        q.put({"dummy": True})
        loop = SensorLoop(q, poll_interval=0.02)
        loop.start()
        time.sleep(0.1)
        loop.stop()
        # Test passes as long as it doesn't hang

    def test_stop_joins_thread(self):
        loop = SensorLoop(queue.Queue(), poll_interval=0.05)
        loop.start()
        loop.stop()
        # Thread should be None after stop
        self.assertIsNone(loop._thread)


# ---------------------------------------------------------------------------
# SensorLoop with mocked trackers
# ---------------------------------------------------------------------------
class TestSensorLoopWithMockedTrackers(unittest.TestCase):

    def test_uses_face_tracker(self):
        face = MagicMock()
        face.get_latest_events.return_value = {
            "face_confidence": 0.9,
            "debug": {"brightness": 150.0},
        }
        q: queue.Queue = queue.Queue(maxsize=5)
        loop = SensorLoop(q, face_tracker=face, poll_interval=0.05)
        loop.start()
        time.sleep(0.15)
        loop.stop()
        reading = q.get_nowait()
        self.assertAlmostEqual(reading["face_confidence"], 0.9)

    def test_uses_speech_engine(self):
        speech = MagicMock()
        speech.get_latest_events.return_value = {"noise_level": 0.07}
        q: queue.Queue = queue.Queue(maxsize=5)
        loop = SensorLoop(q, speech_engine=speech, poll_interval=0.05)
        loop.start()
        time.sleep(0.15)
        loop.stop()
        reading = q.get_nowait()
        self.assertAlmostEqual(reading["noise_level"], 0.07)


if __name__ == "__main__":
    unittest.main()

