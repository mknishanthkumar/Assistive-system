"""
tests/test_speech_loop_features.py - Automated tests for new speech loop features:
1. "Both" mode manual override via voice command ("switch to both", "switch to face and speech")
2. Voice re-center command ("arrange center", "recenter", "re-center", "center")
3. Hold-to-scroll mode ("hold scroll down", "hold scroll up", "stop scroll", "stop", "release")
"""

import os
import queue
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from context.decision_engine import DecisionEngine, Mode
from main import _speech_loop
from output.controller import OutputController
from output.router import Router


class TestSpeechLoopFeatures(unittest.TestCase):
    """Verify speech loop intercepts and processes the new voice commands."""

    def setUp(self):
        self.stop_event = threading.Event()
        self.speech_queue = queue.Queue()
        self.engine = MagicMock()
        self.router = MagicMock()
        self.csv_logger = MagicMock()
        self.decision_engine = DecisionEngine()
        self.tracker = MagicMock()
        self.controller = OutputController(dry_run=True)
        self.event_counter = 0

    def tearDown(self):
        self.stop_event.set()
        if self.controller.hold_scroll.is_active():
            self.controller.hold_scroll.stop()

    def _feed_transcription(self, text: str, command: str = None):
        """Helper to simulate an event returned by SpeechEngine."""
        self.event_counter += 1
        return {
            "event_id": self.event_counter,
            "transcription_text": text,
            "command": command,
            "command_phrase": text,
            "speech_confidence": 0.95,
            "speech_detected": False,
            "dictation_text": None,
            "noise_level": 0.01,
            "vad_status": "silence",
        }

    def _run_single_loop_cycle(self, event_dict):
        """Feed one event through _speech_loop and exit cleanly."""
        self.engine.get_latest_events.return_value = event_dict
        stop_event = threading.Event()

        # Run loop in thread and stop it immediately after processing
        t = threading.Thread(
            target=_speech_loop,
            args=(
                self.engine,
                self.router,
                self.csv_logger,
                stop_event,
                self.speech_queue,
                self.decision_engine,
                self.tracker,
                self.controller,
            ),
        )
        t.start()
        time.sleep(0.08)
        stop_event.set()
        t.join(timeout=1.0)

    # -----------------------------------------------------------------------
    # 1. "Both" mode manual override
    # -----------------------------------------------------------------------
    def test_switch_to_both_voice_command(self):
        # Start in FACE mode
        self.decision_engine.set_manual_override(Mode.FACE)
        self.assertEqual(self.decision_engine.current_mode, Mode.FACE)

        event = self._feed_transcription("switch to both")
        self._run_single_loop_cycle(event)

        self.assertEqual(self.decision_engine.current_mode, Mode.BOTH)
        self.assertTrue(self.decision_engine.is_override)

    def test_switch_to_face_and_speech_voice_command(self):
        # Start in SPEECH mode
        self.decision_engine.set_manual_override(Mode.SPEECH)
        self.assertEqual(self.decision_engine.current_mode, Mode.SPEECH)

        event = self._feed_transcription("switch to face and speech")
        self._run_single_loop_cycle(event)

        self.assertEqual(self.decision_engine.current_mode, Mode.BOTH)
        self.assertTrue(self.decision_engine.is_override)

    # -----------------------------------------------------------------------
    # 2. Voice re-center command
    # -----------------------------------------------------------------------
    def test_arrange_center_triggers_calibrate_neutral(self):
        event = self._feed_transcription("arrange center", command="recenter")
        self._run_single_loop_cycle(event)
        self.tracker.calibrate_neutral.assert_called_once()

    def test_recenter_variations_trigger_calibrate(self):
        for phrase in ("recenter", "re-center", "center"):
            self.tracker.reset_mock()
            event = self._feed_transcription(phrase, command="recenter")
            self._run_single_loop_cycle(event)
            self.tracker.calibrate_neutral.assert_called_once()

    # -----------------------------------------------------------------------
    # 3. Hold-to-scroll mode
    # -----------------------------------------------------------------------
    def test_hold_scroll_down_starts_and_stop_scroll_stops(self):
        self.assertFalse(self.controller.hold_scroll.is_active())

        # Start hold scroll down
        event_down = self._feed_transcription("hold scroll down", command="hold_scroll_down")
        self._run_single_loop_cycle(event_down)
        self.assertTrue(self.controller.hold_scroll.is_active())
        self.assertEqual(self.controller.hold_scroll._direction, -1)

        # Stop via "stop scroll"
        event_stop = self._feed_transcription("stop scroll", command="stop_scroll")
        self._run_single_loop_cycle(event_stop)
        self.assertFalse(self.controller.hold_scroll.is_active())

    def test_hold_scroll_up_starts_and_release_stops(self):
        self.assertFalse(self.controller.hold_scroll.is_active())

        # Start hold scroll up
        event_up = self._feed_transcription("hold scroll up", command="hold_scroll_up")
        self._run_single_loop_cycle(event_up)
        self.assertTrue(self.controller.hold_scroll.is_active())
        self.assertEqual(self.controller.hold_scroll._direction, 1)

        # Stop via "release"
        event_release = self._feed_transcription("release", command="stop_scroll")
        self._run_single_loop_cycle(event_release)
        self.assertFalse(self.controller.hold_scroll.is_active())

    def test_hold_scroll_stops_via_raw_stop_word(self):
        self.controller.hold_scroll.start(direction=-1)
        self.assertTrue(self.controller.hold_scroll.is_active())

        # Say "stop" without parsed command
        event_stop = self._feed_transcription("stop", command=None)
        self._run_single_loop_cycle(event_stop)
        self.assertFalse(self.controller.hold_scroll.is_active())


if __name__ == "__main__":
    unittest.main()

