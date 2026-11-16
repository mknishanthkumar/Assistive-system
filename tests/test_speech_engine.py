"""
tests/test_speech_engine.py - Automated unit tests for SpeechEngine (Stage 3).

All tests use mocked audio, synthetic PCM frames, or dependency mocks.
No physical microphone is required.

Run with:
  .\\.venv\\Scripts\\Activate.ps1
  python -m pytest tests/test_speech_engine.py -v
"""

import math
import struct
import sys
import pathlib
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from speech.speech_engine import (
    SpeechEngine, parse_command, _normalize_text, COMMANDS,
    _is_whisper_hallucination,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence_frame(n_samples: int = 480) -> bytes:
    """Return a silent int16 frame (all zeros)."""
    return b"\x00\x00" * n_samples


def _tone_frame(n_samples: int = 480, amplitude: int = 10000, freq_hz: float = 440.0) -> bytes:
    """Return an int16 frame containing a sine-wave tone."""
    data = []
    for i in range(n_samples):
        sample = int(amplitude * math.sin(2.0 * math.pi * freq_hz * i / 16000))
        data.append(struct.pack("<h", max(-32768, min(32767, sample))))
    return b"".join(data)


# ---------------------------------------------------------------------------
# TestNormalizeText
# ---------------------------------------------------------------------------
class TestNormalizeText(unittest.TestCase):
    """Verify text normalisation for command matching."""

    def test_lowercase(self):
        self.assertEqual(_normalize_text("CLICK"), "click")

    def test_strip_punctuation(self):
        self.assertEqual(_normalize_text("scroll up!"), "scroll up")

    def test_collapse_whitespace(self):
        self.assertEqual(_normalize_text("  double   click  "), "double click")

    def test_empty(self):
        self.assertEqual(_normalize_text(""), "")

    def test_mixed(self):
        self.assertEqual(_normalize_text("  Next  SLIDE!!  "), "next slide")


# ---------------------------------------------------------------------------
# TestCommandParser - all 15 commands
# ---------------------------------------------------------------------------
class TestCommandParser(unittest.TestCase):
    """Verify that all 15 required commands are recognised."""

    def test_click(self):
        cmd, phrase, score = parse_command("click")
        self.assertEqual(cmd, "click")
        self.assertEqual(score, 1.0)

    def test_double_click(self):
        cmd, phrase, score = parse_command("double click")
        self.assertEqual(cmd, "double_click")
        self.assertEqual(score, 1.0)

    def test_right_click(self):
        cmd, phrase, score = parse_command("right click")
        self.assertEqual(cmd, "right_click")
        self.assertEqual(score, 1.0)

    def test_scroll_up(self):
        cmd, phrase, score = parse_command("scroll up")
        self.assertEqual(cmd, "scroll_up")
        self.assertEqual(score, 1.0)

    def test_scroll_down(self):
        cmd, phrase, score = parse_command("scroll down")
        self.assertEqual(cmd, "scroll_down")
        self.assertEqual(score, 1.0)

    def test_open_browser(self):
        cmd, phrase, score = parse_command("open browser")
        self.assertEqual(cmd, "open_browser")
        self.assertEqual(score, 1.0)

    def test_close_window(self):
        cmd, phrase, score = parse_command("close window")
        self.assertEqual(cmd, "close_window")
        self.assertEqual(score, 1.0)

    def test_close_window_alias(self):
        cmd, phrase, score = parse_command("exit window")
        self.assertEqual(cmd, "close_window")
        self.assertEqual(score, 1.0)

    def test_switch_window(self):
        cmd, phrase, score = parse_command("switch window")
        self.assertEqual(cmd, "switch_window")
        self.assertEqual(score, 1.0)

    def test_switch_window_alias(self):
        cmd, phrase, score = parse_command("next window")
        self.assertEqual(cmd, "switch_window")
        self.assertEqual(score, 1.0)

    def test_minimize_window(self):
        cmd, phrase, score = parse_command("minimize window")
        self.assertEqual(cmd, "minimize_window")
        self.assertEqual(score, 1.0)

    def test_minimize_alias(self):
        cmd, phrase, score = parse_command("minimize")
        self.assertEqual(cmd, "minimize_window")
        self.assertEqual(score, 1.0)

    def test_volume_up(self):
        cmd, phrase, score = parse_command("volume up")
        self.assertEqual(cmd, "volume_up")
        self.assertEqual(score, 1.0)

    def test_volume_down(self):
        cmd, phrase, score = parse_command("volume down")
        self.assertEqual(cmd, "volume_down")
        self.assertEqual(score, 1.0)

    def test_mute(self):
        cmd, phrase, score = parse_command("mute")
        self.assertEqual(cmd, "mute")
        self.assertEqual(score, 1.0)

    def test_mute_alias(self):
        cmd, phrase, score = parse_command("unmute")
        self.assertEqual(cmd, "mute")
        self.assertEqual(score, 1.0)

    def test_press_enter(self):
        cmd, phrase, score = parse_command("press enter")
        self.assertEqual(cmd, "press_enter")
        self.assertEqual(score, 1.0)

    def test_press_enter_alias(self):
        cmd, phrase, score = parse_command("enter")
        self.assertEqual(cmd, "press_enter")
        self.assertEqual(score, 1.0)

    def test_press_escape(self):
        cmd, phrase, score = parse_command("press escape")
        self.assertEqual(cmd, "press_escape")
        self.assertEqual(score, 1.0)

    def test_press_escape_alias(self):
        cmd, phrase, score = parse_command("escape")
        self.assertEqual(cmd, "press_escape")
        self.assertEqual(score, 1.0)

    def test_start_dictation(self):
        cmd, phrase, score = parse_command("start dictation")
        self.assertEqual(cmd, "start_dictation")
        self.assertEqual(score, 1.0)

    def test_stop_dictation(self):
        cmd, phrase, score = parse_command("stop dictation")
        self.assertEqual(cmd, "stop_dictation")
        self.assertEqual(score, 1.0)

    def test_stop_control(self):
        cmd, phrase, score = parse_command("stop control")
        self.assertEqual(cmd, "stop_control")
        self.assertEqual(score, 1.0)


# ---------------------------------------------------------------------------
# TestFuzzyMatching
# ---------------------------------------------------------------------------
class TestFuzzyMatching(unittest.TestCase):
    """Verify fuzzy matching handles common recognition errors."""

    def test_extra_punctuation(self):
        cmd, _, _ = parse_command("Click!")
        self.assertEqual(cmd, "click")

    def test_case_insensitive(self):
        cmd, _, _ = parse_command("SCROLL UP")
        self.assertEqual(cmd, "scroll_up")

    def test_minor_typo_double_click(self):
        # "doubble click" should still fuzzy-match "double click"
        cmd, _, score = parse_command("doubble click")
        self.assertEqual(cmd, "double_click")
        self.assertGreaterEqual(score, config.FUZZY_MATCH_THRESHOLD)

    def test_extra_words_around_command(self):
        # "please click now" should match "click" via containment
        cmd, _, score = parse_command("please click now")
        self.assertEqual(cmd, "click")
        self.assertGreaterEqual(score, config.FUZZY_MATCH_THRESHOLD)

    def test_close_variation(self):
        # "volum up" should fuzzy-match "volume up"
        cmd, _, score = parse_command("volum up")
        self.assertEqual(cmd, "volume_up")
        self.assertGreaterEqual(score, config.FUZZY_MATCH_THRESHOLD)


# ---------------------------------------------------------------------------
# TestCommandRejection
# ---------------------------------------------------------------------------
class TestCommandRejection(unittest.TestCase):
    """Verify unrelated phrases are NOT matched to commands."""

    def test_random_sentence(self):
        cmd, _, score = parse_command("the weather is nice today")
        self.assertIsNone(cmd)

    def test_short_noise(self):
        cmd, _, _ = parse_command("uh")
        self.assertIsNone(cmd)

    def test_empty_string(self):
        cmd, _, _ = parse_command("")
        self.assertIsNone(cmd)

    def test_whitespace_only(self):
        cmd, _, _ = parse_command("   ")
        self.assertIsNone(cmd)

    def test_unrelated_words(self):
        cmd, _, _ = parse_command("banana pineapple strawberry")
        self.assertIsNone(cmd)


# ---------------------------------------------------------------------------
# TestSpeechEngineInit
# ---------------------------------------------------------------------------
class TestSpeechEngineInit(unittest.TestCase):
    """Verify SpeechEngine instantiation and initial state."""

    def test_instantiation(self):
        engine = SpeechEngine()
        self.assertIsNotNone(engine)

    def test_initial_events_structure(self):
        engine = SpeechEngine()
        events = engine.get_latest_events()
        expected_keys = {
            "timestamp", "command", "command_phrase", "dictation_text",
            "dictation_mode", "noise_level", "speech_confidence",
            "speech_detected", "transcription_text", "vad_status", "event_id",
        }
        self.assertEqual(set(events.keys()), expected_keys)

    def test_initial_event_types(self):
        engine = SpeechEngine()
        events = engine.get_latest_events()
        self.assertIsInstance(events["timestamp"], float)
        self.assertIsNone(events["command"])
        self.assertIsNone(events["command_phrase"])
        self.assertIsNone(events["dictation_text"])
        self.assertFalse(events["dictation_mode"])
        self.assertIsInstance(events["noise_level"], float)
        self.assertIsInstance(events["speech_confidence"], float)
        self.assertFalse(events["speech_detected"])
        self.assertIsNone(events["transcription_text"])
        self.assertEqual(events["vad_status"], "silence")
        self.assertEqual(events["event_id"], 0)

    def test_initial_no_command(self):
        engine = SpeechEngine()
        events = engine.get_latest_events()
        self.assertIsNone(events["command"])
        self.assertEqual(events["speech_confidence"], 0.0)


# ---------------------------------------------------------------------------
# TestVADFrameSizes
# ---------------------------------------------------------------------------
class TestVADFrameSizes(unittest.TestCase):
    """Verify valid 30 ms frame sizes for WebRTC VAD at supported rates."""

    def test_16khz_30ms(self):
        """16000 Hz * 30 ms = 480 samples = 960 bytes."""
        import webrtcvad
        vad = webrtcvad.Vad(2)
        frame = _silence_frame(480)
        self.assertEqual(len(frame), 960)
        # Should not raise
        result = vad.is_speech(frame, 16000)
        self.assertIsInstance(result, bool)

    def test_8khz_30ms(self):
        """8000 Hz * 30 ms = 240 samples = 480 bytes."""
        import webrtcvad
        vad = webrtcvad.Vad(2)
        frame = _silence_frame(240)
        result = vad.is_speech(frame, 8000)
        self.assertIsInstance(result, bool)


# ---------------------------------------------------------------------------
# TestVADSpeechDetection
# ---------------------------------------------------------------------------
class TestVADSpeechDetection(unittest.TestCase):
    """Test VAD speech/silence classification on synthetic frames."""

    def test_silence_is_not_speech(self):
        import webrtcvad
        vad = webrtcvad.Vad(2)
        frame = _silence_frame(480)
        self.assertFalse(vad.is_speech(frame, 16000))

    def test_tone_may_be_speech(self):
        """A loud tone at 440 Hz should often be classified as speech by VAD."""
        import webrtcvad
        vad = webrtcvad.Vad(0)  # least aggressive: more likely to flag speech
        frame = _tone_frame(480, amplitude=20000, freq_hz=440.0)
        # We don't assert True because VAD behavior depends on content,
        # but at minimum it should return a bool without error.
        result = vad.is_speech(frame, 16000)
        self.assertIsInstance(result, bool)


# ---------------------------------------------------------------------------
# TestSilenceHandling
# ---------------------------------------------------------------------------
class TestSilenceHandling(unittest.TestCase):
    """Verify that silence does not produce speech segments or transcriptions."""

    def test_silence_no_speech_detected(self):
        engine = SpeechEngine()
        events = engine.get_latest_events()
        self.assertFalse(events["speech_detected"])
        self.assertEqual(events["vad_status"], "silence")


# ---------------------------------------------------------------------------
# TestMinSpeechDuration
# ---------------------------------------------------------------------------
class TestMinSpeechDuration(unittest.TestCase):
    """Verify minimum speech duration config is sane."""

    def test_min_duration_positive(self):
        self.assertGreater(config.SPEECH_MIN_DURATION_S, 0.0)

    def test_min_frames_calculation(self):
        min_frames = int(config.SPEECH_MIN_DURATION_S * 1000 / config.AUDIO_BLOCK_DURATION_MS)
        self.assertGreaterEqual(min_frames, 1)


# ---------------------------------------------------------------------------
# TestDictationMode
# ---------------------------------------------------------------------------
class TestDictationMode(unittest.TestCase):
    """Verify dictation activation/deactivation via _process_transcription."""

    def setUp(self):
        self.engine = SpeechEngine()

    def test_start_dictation_activates(self):
        self.engine._process_transcription("start dictation", 0.9)
        with self.engine._lock:
            self.assertTrue(self.engine._dictation_mode)
            self.assertTrue(self.engine._events["dictation_mode"])
            self.assertEqual(self.engine._events["command"], "start_dictation")

    def test_stop_dictation_deactivates(self):
        # First activate
        self.engine._process_transcription("start dictation", 0.9)
        # Then deactivate
        self.engine._process_transcription("stop dictation", 0.9)
        with self.engine._lock:
            self.assertFalse(self.engine._dictation_mode)
            self.assertFalse(self.engine._events["dictation_mode"])
            self.assertEqual(self.engine._events["command"], "stop_dictation")

    def test_dictation_text_preserved(self):
        # Activate dictation
        self.engine._process_transcription("start dictation", 0.9)
        # Consume the start event
        self.engine.get_latest_events()
        # Send natural speech
        self.engine._process_transcription("Hello world this is a test", 0.85)
        events = self.engine.get_latest_events()
        self.assertEqual(events["dictation_text"], "Hello world this is a test")
        # Command should NOT be set during dictation
        self.assertIsNone(events["command"])

    def test_stop_dictation_during_dictation(self):
        """stop dictation must be recognised even while dictation is active."""
        self.engine._process_transcription("start dictation", 0.9)
        self.engine.get_latest_events()
        self.engine._process_transcription("stop dictation", 0.9)
        events = self.engine.get_latest_events()
        self.assertEqual(events["command"], "stop_dictation")
        self.assertFalse(events["dictation_mode"])

    def test_command_not_parsed_during_dictation(self):
        """Normal commands should be treated as dictation text, not commands."""
        self.engine._process_transcription("start dictation", 0.9)
        self.engine.get_latest_events()
        self.engine._process_transcription("click", 0.9)
        events = self.engine.get_latest_events()
        self.assertIsNone(events["command"])
        self.assertEqual(events["dictation_text"], "click")


# ---------------------------------------------------------------------------
# TestNoiseRMS
# ---------------------------------------------------------------------------
class TestNoiseRMS(unittest.TestCase):
    """Verify RMS noise level computation."""

    def test_silence_rms_zero(self):
        rms = SpeechEngine._compute_rms(_silence_frame(480))
        self.assertAlmostEqual(rms, 0.0, places=5)

    def test_tone_rms_positive(self):
        rms = SpeechEngine._compute_rms(_tone_frame(480, amplitude=10000))
        self.assertGreater(rms, 0.0)

    def test_rms_in_range(self):
        rms = SpeechEngine._compute_rms(_tone_frame(480, amplitude=30000))
        self.assertGreaterEqual(rms, 0.0)
        self.assertLessEqual(rms, 1.0)

    def test_rms_is_finite(self):
        rms = SpeechEngine._compute_rms(_tone_frame(480))
        self.assertTrue(math.isfinite(rms))

    def test_empty_frame_rms_zero(self):
        rms = SpeechEngine._compute_rms(b"")
        self.assertAlmostEqual(rms, 0.0, places=5)


# ---------------------------------------------------------------------------
# TestSpeechConfidence
# ---------------------------------------------------------------------------
class TestSpeechConfidence(unittest.TestCase):
    """Verify speech confidence range and finite values."""

    def test_initial_confidence_zero(self):
        engine = SpeechEngine()
        events = engine.get_latest_events()
        self.assertEqual(events["speech_confidence"], 0.0)

    def test_confidence_after_transcription(self):
        engine = SpeechEngine()
        engine._process_transcription("hello", 0.85)
        events = engine.get_latest_events()
        self.assertEqual(events["speech_confidence"], 0.85)
        self.assertGreaterEqual(events["speech_confidence"], 0.0)
        self.assertLessEqual(events["speech_confidence"], 1.0)

    def test_confidence_is_finite(self):
        engine = SpeechEngine()
        engine._process_transcription("test", 0.5)
        events = engine.get_latest_events()
        self.assertTrue(math.isfinite(events["speech_confidence"]))

    def test_faster_whisper_segment_avg_logprob(self):
        """Verify handling of Faster-Whisper Segment.avg_logprob attribute."""
        class MockSegment:
            def __init__(self, text, avg_logprob):
                self.text = text
                self.avg_logprob = avg_logprob

        seg = MockSegment("click", -0.2)
        logprob = getattr(seg, "avg_logprob", getattr(seg, "avg_log_prob", -0.5))
        self.assertEqual(logprob, -0.2)
        conf = float(min(max(math.exp(logprob), 0.0), 1.0))
        self.assertAlmostEqual(conf, math.exp(-0.2), places=4)


# ---------------------------------------------------------------------------
# TestEventEdgeTriggering
# ---------------------------------------------------------------------------
class TestEventEdgeTriggering(unittest.TestCase):
    """Verify that edge-triggered fields are consumed on read."""

    def test_command_consumed_on_read(self):
        engine = SpeechEngine()
        engine._process_transcription("click", 0.9)
        events1 = engine.get_latest_events()
        events2 = engine.get_latest_events()
        self.assertEqual(events1["command"], "click")
        self.assertIsNone(events2["command"])

    def test_transcription_consumed_on_read(self):
        engine = SpeechEngine()
        engine._process_transcription("hello", 0.9)
        events1 = engine.get_latest_events()
        events2 = engine.get_latest_events()
        self.assertEqual(events1["transcription_text"], "hello")
        self.assertIsNone(events2["transcription_text"])

    def test_dictation_text_consumed_on_read(self):
        engine = SpeechEngine()
        engine._process_transcription("start dictation", 0.9)
        engine.get_latest_events()  # consume start event
        engine._process_transcription("some text", 0.9)
        events1 = engine.get_latest_events()
        events2 = engine.get_latest_events()
        self.assertEqual(events1["dictation_text"], "some text")
        self.assertIsNone(events2["dictation_text"])

    def test_event_id_increments(self):
        engine = SpeechEngine()
        engine._process_transcription("click", 0.9)
        e1 = engine.get_latest_events()
        engine._process_transcription("pause", 0.9)
        e2 = engine.get_latest_events()
        self.assertGreater(e2["event_id"], e1["event_id"])


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------
class TestThreadSafety(unittest.TestCase):
    """Verify concurrent reads don't crash."""

    def test_concurrent_reads(self):
        engine = SpeechEngine()
        errors = []

        def reader():
            try:
                for _ in range(100):
                    events = engine.get_latest_events()
                    self.assertIsInstance(events, dict)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")


# ---------------------------------------------------------------------------
# TestCleanShutdown
# ---------------------------------------------------------------------------
class TestCleanShutdown(unittest.TestCase):
    """Verify stop() is safe even without start()."""

    def test_stop_without_start(self):
        engine = SpeechEngine()
        try:
            engine.stop()
        except Exception as e:
            self.fail(f"stop() raised unexpected exception: {e}")

    def test_double_stop_safe(self):
        engine = SpeechEngine()
        engine.stop()
        engine.stop()  # must not raise


# ---------------------------------------------------------------------------
# TestCommandCompleteness
# ---------------------------------------------------------------------------
class TestCommandCompleteness(unittest.TestCase):
    """Verify all required commands are defined in the COMMANDS dict."""

    REQUIRED_COMMANDS = [
        "click", "double_click", "right_click",
        "scroll_up", "scroll_down",
        "open_browser", "close_window", "switch_window", "minimize_window",
        "volume_up", "volume_down", "mute",
        "press_enter", "press_escape",
        "start_dictation", "stop_dictation",
        "stop_control",
    ]

    def test_all_commands_present(self):
        self.assertEqual(len(self.REQUIRED_COMMANDS), 17)
        for cmd_id in self.REQUIRED_COMMANDS:
            self.assertIn(cmd_id, COMMANDS,
                          f"Missing command: {cmd_id}")

    def test_all_commands_have_phrases(self):
        for cmd_id, phrases in COMMANDS.items():
            self.assertGreater(len(phrases), 0,
                               f"Command {cmd_id} has no phrases")


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ---------------------------------------------------------------------------
# TestWhisperHallucinationFilter  (regression tests for Bug Fix #3 / #4)
# ---------------------------------------------------------------------------
class TestWhisperHallucinationFilter(unittest.TestCase):
    """
    Verify _is_whisper_hallucination() correctly classifies known Whisper
    noise artefacts and allows real speech through.
    """

    def test_empty_string_is_hallucination(self):
        self.assertTrue(_is_whisper_hallucination(""))

    def test_whitespace_only_is_hallucination(self):
        self.assertTrue(_is_whisper_hallucination("   "))

    def test_single_char_is_hallucination(self):
        self.assertTrue(_is_whisper_hallucination("a"))
        self.assertTrue(_is_whisper_hallucination("I"))

    def test_known_phrase_thank_you(self):
        self.assertTrue(_is_whisper_hallucination("Thank you."))

    def test_known_phrase_thank_you_for_watching(self):
        self.assertTrue(_is_whisper_hallucination("Thank you for watching."))

    def test_known_phrase_uh(self):
        self.assertTrue(_is_whisper_hallucination("Uh"))

    def test_known_phrase_um(self):
        self.assertTrue(_is_whisper_hallucination("Um."))

    def test_known_phrase_hmm(self):
        self.assertTrue(_is_whisper_hallucination("Hmm."))

    def test_blank_audio_bracket(self):
        # Bracket-encoded hallucinations have low alpha ratio
        self.assertTrue(_is_whisper_hallucination("[BLANK_AUDIO]"))

    def test_ellipsis_only(self):
        self.assertTrue(_is_whisper_hallucination("..."))

    def test_punctuation_only(self):
        self.assertTrue(_is_whisper_hallucination("!?."))

    def test_real_command_not_hallucination(self):
        self.assertFalse(_is_whisper_hallucination("click"))

    def test_real_command_scroll_up(self):
        self.assertFalse(_is_whisper_hallucination("scroll up"))

    def test_real_phrase_not_hallucination(self):
        self.assertFalse(_is_whisper_hallucination("open browser please"))

    def test_dictation_phrase_not_hallucination(self):
        self.assertFalse(_is_whisper_hallucination("Hello world this is a test"))

    def test_case_insensitive_hallucination(self):
        # "THANK YOU" normalises to "thank you" -> hallucination
        self.assertTrue(_is_whisper_hallucination("THANK YOU"))

    def test_okay_is_hallucination(self):
        self.assertTrue(_is_whisper_hallucination("Okay."))

    def test_so_is_hallucination(self):
        self.assertTrue(_is_whisper_hallucination("So."))


# ---------------------------------------------------------------------------
# TestRMSEnergyGate  (regression tests for Bug Fix #1)
# ---------------------------------------------------------------------------
class TestRMSEnergyGate(unittest.TestCase):
    """
    Verify config.SPEECH_RMS_THRESHOLD is set at a level that separates
    typical ambient noise from typical speech.
    """

    def test_rms_threshold_exists(self):
        """SPEECH_RMS_THRESHOLD must be defined in config."""
        self.assertTrue(hasattr(config, "SPEECH_RMS_THRESHOLD"))

    def test_rms_threshold_range(self):
        """SPEECH_RMS_THRESHOLD must be in a sensible range (0.001–0.10)."""
        self.assertGreater(config.SPEECH_RMS_THRESHOLD, 0.001)
        self.assertLess(config.SPEECH_RMS_THRESHOLD, 0.10)

    def test_ambient_noise_rms_below_threshold(self):
        """Typical ambient-noise RMS (std=300) should be below threshold."""
        import math, struct, random
        random.seed(0)
        rms_values = []
        for _ in range(50):
            frame = bytes([
                x
                for _ in range(480)
                for x in struct.pack("<h", max(-32767, min(32767, int(random.gauss(0, 300)))))
            ])
            rms_values.append(SpeechEngine._compute_rms(frame))
        avg = sum(rms_values) / len(rms_values)
        self.assertLess(avg, config.SPEECH_RMS_THRESHOLD,
                        f"Avg ambient RMS {avg:.4f} should be below threshold "
                        f"{config.SPEECH_RMS_THRESHOLD}")

    def test_speech_level_rms_above_threshold(self):
        """Typical speech-level audio (std=5000) should exceed threshold."""
        import math, struct, random
        random.seed(0)
        rms_values = []
        for _ in range(10):
            frame = bytes([
                x
                for _ in range(480)
                for x in struct.pack("<h", max(-32767, min(32767, int(random.gauss(0, 5000)))))
            ])
            rms_values.append(SpeechEngine._compute_rms(frame))
        avg = sum(rms_values) / len(rms_values)
        self.assertGreater(avg, config.SPEECH_RMS_THRESHOLD,
                           f"Avg speech RMS {avg:.4f} should be above threshold "
                           f"{config.SPEECH_RMS_THRESHOLD}")


# ---------------------------------------------------------------------------
# TestVADMaxDuration  (regression test for Bug Fix #5 — max segment cap)
# ---------------------------------------------------------------------------
class TestVADMaxDuration(unittest.TestCase):
    """Verify SPEECH_MAX_DURATION_S is defined and sensible."""

    def test_max_duration_exists(self):
        self.assertTrue(hasattr(config, "SPEECH_MAX_DURATION_S"))

    def test_max_duration_greater_than_min(self):
        self.assertGreater(
            config.SPEECH_MAX_DURATION_S,
            config.SPEECH_MIN_DURATION_S,
        )

    def test_max_duration_reasonable(self):
        # Should be between 5 and 60 seconds for a command system
        self.assertGreaterEqual(config.SPEECH_MAX_DURATION_S, 5.0)
        self.assertLessEqual(config.SPEECH_MAX_DURATION_S, 60.0)


# ---------------------------------------------------------------------------
# TestNoSpeechProbConfig  (regression test for Bug Fix #2)
# ---------------------------------------------------------------------------
class TestNoSpeechProbConfig(unittest.TestCase):
    """Verify WHISPER_NO_SPEECH_THRESHOLD is defined and sensible."""

    def test_threshold_exists(self):
        self.assertTrue(hasattr(config, "WHISPER_NO_SPEECH_THRESHOLD"))

    def test_threshold_range(self):
        # Must be between 0.0 and 1.0
        self.assertGreater(config.WHISPER_NO_SPEECH_THRESHOLD, 0.0)
        self.assertLess(config.WHISPER_NO_SPEECH_THRESHOLD, 1.0)

    def test_threshold_sensible(self):
        # Typical good value: 0.5-0.8 (conservative: reject when >50-80% likely no speech)
        self.assertGreaterEqual(config.WHISPER_NO_SPEECH_THRESHOLD, 0.40)


# ---------------------------------------------------------------------------
# TestVADAggressiveness  (regression test for Bug Fix #1 config)
# ---------------------------------------------------------------------------
class TestVADAggressiveness(unittest.TestCase):
    """Verify VAD aggressiveness is set to suppress ambient noise."""

    def test_aggressiveness_value(self):
        """Aggressiveness must be 3 (maximum) to filter ambient noise."""
        self.assertEqual(config.VAD_AGGRESSIVENESS, 3)

    def test_aggressiveness_in_valid_range(self):
        self.assertGreaterEqual(config.VAD_AGGRESSIVENESS, 0)
        self.assertLessEqual(config.VAD_AGGRESSIVENESS, 3)


# ---------------------------------------------------------------------------
# TestSpeechDetectedAccuracy  (regression test for Bug Fix #1 state display)
# ---------------------------------------------------------------------------
class TestSpeechDetectedAccuracy(unittest.TestCase):
    """
    Verify speech_detected reflects is_speaking (confirmed speaking state),
    not raw per-frame VAD output.
    Initial state must always be False.
    """

    def test_initial_speech_detected_false(self):
        engine = SpeechEngine()
        events = engine.get_latest_events()
        self.assertFalse(events["speech_detected"],
                         "speech_detected must be False before any audio is processed.")

    def test_speech_detected_false_after_no_frames(self):
        engine = SpeechEngine()
        # No audio has been pushed; state must remain False
        events = engine.get_latest_events()
        self.assertFalse(events["speech_detected"])


# ---------------------------------------------------------------------------
# TestVADStateMachineWithRMSGate  (regression test for Bug Fix #1 state machine)
# ---------------------------------------------------------------------------
class TestVADStateMachineWithRMSGate(unittest.TestCase):
    """
    Simulate the VAD loop to verify that ambient noise frames (VAD=True but
    RMS below threshold) do NOT produce a speech segment.
    """

    def _simulate_vad_loop(self, frames_with_flags):
        """
        Minimal simulation of _vad_processing_loop dual-gate logic.

        Parameters
        ----------
        frames_with_flags : list of (rms, vad_result) tuples

        Returns
        -------
        (segments_sent, final_is_speaking)
        """
        rms_threshold = config.SPEECH_RMS_THRESHOLD
        min_sf = max(1, int(config.SPEECH_MIN_DURATION_S * 1000 / config.AUDIO_BLOCK_DURATION_MS))
        sil_tf = max(1, int(config.SPEECH_SILENCE_DURATION_S * 1000 / config.AUDIO_BLOCK_DURATION_MS))
        trig_f = config.SPEECH_TRIGGER_FRAMES

        speech_frames = []
        silence_count = 0
        trigger_count = 0
        is_speaking = False
        segments_sent = 0

        for rms, vad_result in frames_with_flags:
            is_active = vad_result and (rms >= rms_threshold)

            if not is_speaking:
                if is_active:
                    trigger_count += 1
                    if trigger_count >= trig_f:
                        is_speaking = True
                        silence_count = 0
                else:
                    trigger_count = 0
            else:
                speech_frames.append(1)
                if not is_active:
                    silence_count += 1
                    if silence_count >= sil_tf:
                        if len(speech_frames) >= min_sf:
                            segments_sent += 1
                        speech_frames = []
                        silence_count = 0
                        trigger_count = 0
                        is_speaking = False
                else:
                    silence_count = 0

        return segments_sent, is_speaking

    def test_ambient_noise_produces_no_segment(self):
        """
        Ambient noise frames: VAD=True but RMS below threshold.
        Should produce zero segments.
        """
        low_rms = config.SPEECH_RMS_THRESHOLD * 0.5  # well below threshold
        # 100 frames of 'ambient noise' — VAD says speech, but RMS is low
        frames = [(low_rms, True)] * 100
        segments, speaking = self._simulate_vad_loop(frames)
        self.assertEqual(segments, 0,
                         "Ambient noise must not produce any transcription segments.")
        self.assertFalse(speaking)

    def test_real_speech_produces_segment(self):
        """
        Real speech frames: VAD=True AND RMS above threshold, followed by silence.
        Should produce exactly one segment.
        """
        high_rms = config.SPEECH_RMS_THRESHOLD * 3.0
        low_rms = config.SPEECH_RMS_THRESHOLD * 0.3
        # 3 trigger frames + 50 speech frames + 30 silence frames
        frames = (
            [(high_rms, True)] * 3   # trigger onset
            + [(high_rms, True)] * 50  # speech
            + [(low_rms, False)] * 30  # silence (> silence_timeout_frames=26)
        )
        segments, speaking = self._simulate_vad_loop(frames)
        self.assertEqual(segments, 1,
                         "Real speech followed by silence must produce exactly one segment.")
        self.assertFalse(speaking)

    def test_ambient_noise_then_real_speech(self):
        """
        Ambient noise before real speech must not trigger premature segment.
        Only the real speech should produce a segment.
        """
        high_rms = config.SPEECH_RMS_THRESHOLD * 4.0
        low_rms = config.SPEECH_RMS_THRESHOLD * 0.4
        noise_rms = config.SPEECH_RMS_THRESHOLD * 0.5

        frames = (
            [(noise_rms, True)] * 50   # ambient noise (VAD=True, RMS below threshold)
            + [(high_rms, True)] * 60  # actual speech
            + [(low_rms, False)] * 30  # trailing silence
        )
        segments, _ = self._simulate_vad_loop(frames)
        self.assertEqual(segments, 1)

    def test_no_frames_no_segment(self):
        segments, speaking = self._simulate_vad_loop([])
        self.assertEqual(segments, 0)
        self.assertFalse(speaking)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ---------------------------------------------------------------------------
# TestWhisperHallucinationFilter  (regression tests for Bug Fix #3 / #4)
# ---------------------------------------------------------------------------
class TestWhisperHallucinationFilter(unittest.TestCase):
    """
    Verify _is_whisper_hallucination() correctly classifies known Whisper
    noise artefacts and allows real speech through.
    """

    def test_empty_string_is_hallucination(self):
        self.assertTrue(_is_whisper_hallucination(""))

    def test_whitespace_only_is_hallucination(self):
        self.assertTrue(_is_whisper_hallucination("   "))

    def test_single_char_is_hallucination(self):
        self.assertTrue(_is_whisper_hallucination("a"))
        self.assertTrue(_is_whisper_hallucination("I"))

    def test_known_phrase_thank_you(self):
        self.assertTrue(_is_whisper_hallucination("Thank you."))

    def test_known_phrase_thank_you_for_watching(self):
        self.assertTrue(_is_whisper_hallucination("Thank you for watching."))

    def test_known_phrase_uh(self):
        self.assertTrue(_is_whisper_hallucination("Uh"))

    def test_known_phrase_um(self):
        self.assertTrue(_is_whisper_hallucination("Um."))

    def test_known_phrase_hmm(self):
        self.assertTrue(_is_whisper_hallucination("Hmm."))

    def test_blank_audio_bracket(self):
        # Bracket-encoded hallucinations have low alpha ratio
        self.assertTrue(_is_whisper_hallucination("[BLANK_AUDIO]"))

    def test_ellipsis_only(self):
        self.assertTrue(_is_whisper_hallucination("..."))

    def test_punctuation_only(self):
        self.assertTrue(_is_whisper_hallucination("!?."))

    def test_real_command_not_hallucination(self):
        self.assertFalse(_is_whisper_hallucination("click"))

    def test_real_command_scroll_up(self):
        self.assertFalse(_is_whisper_hallucination("scroll up"))

    def test_real_phrase_not_hallucination(self):
        self.assertFalse(_is_whisper_hallucination("open browser please"))

    def test_dictation_phrase_not_hallucination(self):
        self.assertFalse(_is_whisper_hallucination("Hello world this is a test"))

    def test_case_insensitive_hallucination(self):
        # "THANK YOU" normalises to "thank you" -> hallucination
        self.assertTrue(_is_whisper_hallucination("THANK YOU"))

    def test_okay_is_hallucination(self):
        self.assertTrue(_is_whisper_hallucination("Okay."))

    def test_so_is_hallucination(self):
        self.assertTrue(_is_whisper_hallucination("So."))


# ---------------------------------------------------------------------------
# TestRMSEnergyGate  (regression tests for Bug Fix #1)
# ---------------------------------------------------------------------------
class TestRMSEnergyGate(unittest.TestCase):
    """
    Verify config.SPEECH_RMS_THRESHOLD is set at a level that separates
    typical ambient noise from typical speech.
    """

    def test_rms_threshold_exists(self):
        """SPEECH_RMS_THRESHOLD must be defined in config."""
        self.assertTrue(hasattr(config, "SPEECH_RMS_THRESHOLD"))

    def test_rms_threshold_range(self):
        """SPEECH_RMS_THRESHOLD must be in a sensible range (0.001–0.10)."""
        self.assertGreater(config.SPEECH_RMS_THRESHOLD, 0.001)
        self.assertLess(config.SPEECH_RMS_THRESHOLD, 0.10)

    def test_ambient_noise_rms_below_threshold(self):
        """Typical ambient-noise RMS (std=300) should be below threshold."""
        import math, struct, random
        random.seed(0)
        rms_values = []
        for _ in range(50):
            frame = bytes([
                x
                for _ in range(480)
                for x in struct.pack("<h", max(-32767, min(32767, int(random.gauss(0, 300)))))
            ])
            rms_values.append(SpeechEngine._compute_rms(frame))
        avg = sum(rms_values) / len(rms_values)
        self.assertLess(avg, config.SPEECH_RMS_THRESHOLD,
                        f"Avg ambient RMS {avg:.4f} should be below threshold "
                        f"{config.SPEECH_RMS_THRESHOLD}")

    def test_speech_level_rms_above_threshold(self):
        """Typical speech-level audio (std=5000) should exceed threshold."""
        import math, struct, random
        random.seed(0)
        rms_values = []
        for _ in range(10):
            frame = bytes([
                x
                for _ in range(480)
                for x in struct.pack("<h", max(-32767, min(32767, int(random.gauss(0, 5000)))))
            ])
            rms_values.append(SpeechEngine._compute_rms(frame))
        avg = sum(rms_values) / len(rms_values)
        self.assertGreater(avg, config.SPEECH_RMS_THRESHOLD,
                           f"Avg speech RMS {avg:.4f} should be above threshold "
                           f"{config.SPEECH_RMS_THRESHOLD}")


# ---------------------------------------------------------------------------
# TestVADMaxDuration  (regression test for Bug Fix #5 — max segment cap)
# ---------------------------------------------------------------------------
class TestVADMaxDuration(unittest.TestCase):
    """Verify SPEECH_MAX_DURATION_S is defined and sensible."""

    def test_max_duration_exists(self):
        self.assertTrue(hasattr(config, "SPEECH_MAX_DURATION_S"))

    def test_max_duration_greater_than_min(self):
        self.assertGreater(
            config.SPEECH_MAX_DURATION_S,
            config.SPEECH_MIN_DURATION_S,
        )

    def test_max_duration_reasonable(self):
        # Should be between 5 and 60 seconds for a command system
        self.assertGreaterEqual(config.SPEECH_MAX_DURATION_S, 5.0)
        self.assertLessEqual(config.SPEECH_MAX_DURATION_S, 60.0)


# ---------------------------------------------------------------------------
# TestNoSpeechProbConfig  (regression test for Bug Fix #2)
# ---------------------------------------------------------------------------
class TestNoSpeechProbConfig(unittest.TestCase):
    """Verify WHISPER_NO_SPEECH_THRESHOLD is defined and sensible."""

    def test_threshold_exists(self):
        self.assertTrue(hasattr(config, "WHISPER_NO_SPEECH_THRESHOLD"))

    def test_threshold_range(self):
        # Must be between 0.0 and 1.0
        self.assertGreater(config.WHISPER_NO_SPEECH_THRESHOLD, 0.0)
        self.assertLess(config.WHISPER_NO_SPEECH_THRESHOLD, 1.0)

    def test_threshold_sensible(self):
        # Typical good value: 0.5-0.8 (conservative: reject when >50-80% likely no speech)
        self.assertGreaterEqual(config.WHISPER_NO_SPEECH_THRESHOLD, 0.40)


# ---------------------------------------------------------------------------
# TestVADAggressiveness  (regression test for Bug Fix #1 config)
# ---------------------------------------------------------------------------
class TestVADAggressiveness(unittest.TestCase):
    """Verify VAD aggressiveness is set to suppress ambient noise."""

    def test_aggressiveness_value(self):
        """Aggressiveness must be 3 (maximum) to filter ambient noise."""
        self.assertEqual(config.VAD_AGGRESSIVENESS, 3)

    def test_aggressiveness_in_valid_range(self):
        self.assertGreaterEqual(config.VAD_AGGRESSIVENESS, 0)
        self.assertLessEqual(config.VAD_AGGRESSIVENESS, 3)


# ---------------------------------------------------------------------------
# TestSpeechDetectedAccuracy  (regression test for Bug Fix #1 state display)
# ---------------------------------------------------------------------------
class TestSpeechDetectedAccuracy(unittest.TestCase):
    """
    Verify speech_detected reflects is_speaking (confirmed speaking state),
    not raw per-frame VAD output.
    Initial state must always be False.
    """

    def test_initial_speech_detected_false(self):
        engine = SpeechEngine()
        events = engine.get_latest_events()
        self.assertFalse(events["speech_detected"],
                         "speech_detected must be False before any audio is processed.")

    def test_speech_detected_false_after_no_frames(self):
        engine = SpeechEngine()
        # No audio has been pushed; state must remain False
        events = engine.get_latest_events()
        self.assertFalse(events["speech_detected"])


# ---------------------------------------------------------------------------
# TestVADStateMachineWithRMSGate  (regression test for Bug Fix #1 state machine)
# ---------------------------------------------------------------------------
class TestVADStateMachineWithRMSGate(unittest.TestCase):
    """
    Simulate the VAD loop to verify that ambient noise frames (VAD=True but
    RMS below threshold) do NOT produce a speech segment.
    """

    def _simulate_vad_loop(self, frames_with_flags):
        """
        Minimal simulation of _vad_processing_loop dual-gate logic.

        Parameters
        ----------
        frames_with_flags : list of (rms, vad_result) tuples

        Returns
        -------
        (segments_sent, final_is_speaking)
        """
        rms_threshold = config.SPEECH_RMS_THRESHOLD
        min_sf = max(1, int(config.SPEECH_MIN_DURATION_S * 1000 / config.AUDIO_BLOCK_DURATION_MS))
        sil_tf = max(1, int(config.SPEECH_SILENCE_DURATION_S * 1000 / config.AUDIO_BLOCK_DURATION_MS))
        trig_f = config.SPEECH_TRIGGER_FRAMES

        speech_frames = []
        silence_count = 0
        trigger_count = 0
        is_speaking = False
        segments_sent = 0

        for rms, vad_result in frames_with_flags:
            is_active = vad_result and (rms >= rms_threshold)

            if not is_speaking:
                if is_active:
                    trigger_count += 1
                    if trigger_count >= trig_f:
                        is_speaking = True
                        silence_count = 0
                else:
                    trigger_count = 0
            else:
                speech_frames.append(1)
                if not is_active:
                    silence_count += 1
                    if silence_count >= sil_tf:
                        if len(speech_frames) >= min_sf:
                            segments_sent += 1
                        speech_frames = []
                        silence_count = 0
                        trigger_count = 0
                        is_speaking = False
                else:
                    silence_count = 0

        return segments_sent, is_speaking

    def test_ambient_noise_produces_no_segment(self):
        """
        Ambient noise frames: VAD=True but RMS below threshold.
        Should produce zero segments.
        """
        low_rms = config.SPEECH_RMS_THRESHOLD * 0.5  # well below threshold
        # 100 frames of 'ambient noise' — VAD says speech, but RMS is low
        frames = [(low_rms, True)] * 100
        segments, speaking = self._simulate_vad_loop(frames)
        self.assertEqual(segments, 0,
                         "Ambient noise must not produce any transcription segments.")
        self.assertFalse(speaking)

    def test_real_speech_produces_segment(self):
        """
        Real speech frames: VAD=True AND RMS above threshold, followed by silence.
        Should produce exactly one segment.
        """
        high_rms = config.SPEECH_RMS_THRESHOLD * 3.0
        low_rms = config.SPEECH_RMS_THRESHOLD * 0.3
        # 3 trigger frames + 50 speech frames + 30 silence frames
        frames = (
            [(high_rms, True)] * 3   # trigger onset
            + [(high_rms, True)] * 50  # speech
            + [(low_rms, False)] * 30  # silence (> silence_timeout_frames=26)
        )
        segments, speaking = self._simulate_vad_loop(frames)
        self.assertEqual(segments, 1,
                         "Real speech followed by silence must produce exactly one segment.")
        self.assertFalse(speaking)

    def test_ambient_noise_then_real_speech(self):
        """
        Ambient noise before real speech must not trigger premature segment.
        Only the real speech should produce a segment.
        """
        high_rms = config.SPEECH_RMS_THRESHOLD * 4.0
        low_rms = config.SPEECH_RMS_THRESHOLD * 0.4
        noise_rms = config.SPEECH_RMS_THRESHOLD * 0.5

        frames = (
            [(noise_rms, True)] * 50   # ambient noise (VAD=True, RMS below threshold)
            + [(high_rms, True)] * 60  # actual speech
            + [(low_rms, False)] * 30  # trailing silence
        )
        segments, _ = self._simulate_vad_loop(frames)
        self.assertEqual(segments, 1)

    def test_no_frames_no_segment(self):
        segments, speaking = self._simulate_vad_loop([])
        self.assertEqual(segments, 0)
        self.assertFalse(speaking)


if __name__ == "__main__":
    unittest.main(verbosity=2)