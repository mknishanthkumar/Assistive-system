"""
tests/test_decision_engine.py – Automated tests for context/decision_engine.py

Covers:
  - Mode enum values
  - DecisionResult dataclass
  - All six rules (PAUSED, FACE/noisy, SPEECH/text-field, SPEECH/face-lost,
    BOTH, FACE/fallback)
  - Hysteresis: hold duration guard
  - Hysteresis: minimum switch interval guard
  - Manual override (FACE, SPEECH, BOTH, PAUSED)
  - Return to automatic mode (clear_manual_override)
  - Thread safety
  - CSV logger integration
  - Logging: every decision includes mode, reason, is_override
  - Decision log completeness
  - Rule priority ordering
"""

import sys
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup – allow running from project root or tests/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from context.decision_engine import DecisionEngine, DecisionResult, Mode


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _inputs(
    noise_level: float = 0.0,
    face_confidence: float = 0.8,
    brightness: float = 128.0,
    text_field_focused: "bool | str" = False,
    speech_confidence: float = 0.8,
    active_app: str = "unknown",
) -> dict:
    """Build a sensor-inputs dict with sane defaults for most fields."""
    return {
        "noise_level":        noise_level,
        "face_confidence":    face_confidence,
        "brightness":         brightness,
        "text_field_focused": text_field_focused,
        "speech_confidence":  speech_confidence,
        "active_app":         active_app,
    }


def _stable_inputs(**kwargs) -> dict:
    """Inputs that always pass both reliability checks (good face + good speech)."""
    base = _inputs(face_confidence=0.9, speech_confidence=0.9,
                   brightness=128.0, noise_level=0.0)
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Base test class
# ---------------------------------------------------------------------------

class DecisionEngineTestBase(unittest.TestCase):
    """
    Creates a fresh engine with hysteresis frozen at zero for each test.

    With hold=0.0 and interval=0.0:
      - The engine registers a candidate on the FIRST call with a new target.
      - On the SECOND consecutive call with the same target the switch commits
        (0 seconds elapsed >= 0.0 required, and interval is also 0).

    _eval(inputs) calls evaluate() twice so the switch always commits.
    For tests that only need to verify the returned mode (not a live switch),
    use _eval(); for tests that examine the initial-registration phase, call
    self.engine.evaluate() directly.
    """

    def setUp(self) -> None:
        self.engine = DecisionEngine()
        self._orig_hold     = config.DECISION_HOLD_DURATION_S
        self._orig_interval = config.DECISION_MIN_SWITCH_INTERVAL_S
        config.DECISION_HOLD_DURATION_S       = 0.0
        config.DECISION_MIN_SWITCH_INTERVAL_S = 0.0

    def tearDown(self) -> None:
        config.DECISION_HOLD_DURATION_S       = self._orig_hold
        config.DECISION_MIN_SWITCH_INTERVAL_S = self._orig_interval

    def _eval(self, inputs: dict) -> DecisionResult:
        """
        Evaluate inputs twice so the hysteresis hold timer is satisfied.
        Returns the second (committed) result.
        """
        self.engine.evaluate(inputs)        # registers candidate
        return self.engine.evaluate(inputs) # commits switch


# ===========================================================================
# 1. Mode enum
# ===========================================================================

class TestModeEnum(unittest.TestCase):
    def test_mode_values_exist(self) -> None:
        self.assertEqual(Mode.FACE.value,   "FACE")
        self.assertEqual(Mode.SPEECH.value, "SPEECH")
        self.assertEqual(Mode.BOTH.value,   "BOTH")
        self.assertEqual(Mode.PAUSED.value, "PAUSED")

    def test_all_four_modes(self) -> None:
        self.assertEqual(len(Mode), 4)


# ===========================================================================
# 2. DecisionResult dataclass
# ===========================================================================

class TestDecisionResult(unittest.TestCase):
    def test_fields_accessible(self) -> None:
        r = DecisionResult(
            mode=Mode.BOTH,
            reason="test",
            is_override=False,
            timestamp=1.0,
            inputs={"noise_level": 0.01},
        )
        self.assertEqual(r.mode, Mode.BOTH)
        self.assertEqual(r.reason, "test")
        self.assertFalse(r.is_override)
        self.assertEqual(r.timestamp, 1.0)
        self.assertEqual(r.inputs["noise_level"], 0.01)

    def test_inputs_defaults_to_empty_dict(self) -> None:
        r = DecisionResult(mode=Mode.FACE, reason="x", is_override=False, timestamp=0.0)
        self.assertIsInstance(r.inputs, dict)


# ===========================================================================
# 3. Instantiation
# ===========================================================================

class TestDecisionEngineInit(DecisionEngineTestBase):
    def test_instantiation(self) -> None:
        engine = DecisionEngine()
        self.assertIsNotNone(engine)

    def test_initial_mode_is_both(self) -> None:
        # Engine starts permissive (BOTH) until first evaluate().
        self.assertEqual(self.engine.current_mode, Mode.BOTH)

    def test_initial_is_not_override(self) -> None:
        self.assertFalse(self.engine.is_override)

    def test_csv_logger_optional(self) -> None:
        engine = DecisionEngine(csv_logger=None)
        result = engine.evaluate(_stable_inputs())
        self.assertIsInstance(result, DecisionResult)


# ===========================================================================
# 4. Rule: Both unreliable → PAUSED
# ===========================================================================

class TestRuleBothUnreliable(DecisionEngineTestBase):
    """Rule 1 (highest priority): noisy AND face lost AND speech unreliable → PAUSED."""

    def _paused_inputs(self) -> dict:
        return _inputs(noise_level=1.0, face_confidence=0.0,
                       brightness=5.0, speech_confidence=0.0)

    def test_both_unreliable_gives_paused(self) -> None:
        result = self._eval(self._paused_inputs())
        self.assertEqual(result.mode, Mode.PAUSED)

    def test_paused_reason_contains_both_unreliable(self) -> None:
        result = self._eval(self._paused_inputs())
        self.assertIn("both unreliable", result.reason.lower())

    def test_paused_not_override(self) -> None:
        result = self._eval(self._paused_inputs())
        self.assertFalse(result.is_override)

    def test_paused_has_inputs_snapshot(self) -> None:
        inp = self._paused_inputs()
        result = self._eval(inp)
        self.assertIn("noise_level", result.inputs)
        self.assertEqual(result.inputs["noise_level"], inp["noise_level"])


# ===========================================================================
# 5. Rule: Noisy → FACE only
# ===========================================================================

class TestRuleNoisy(DecisionEngineTestBase):
    """Rule 2: noisy environment (noise >= threshold) → FACE, speech disabled."""

    def test_noisy_gives_face(self) -> None:
        result = self._eval(_inputs(
            noise_level=1.0, face_confidence=0.9, speech_confidence=0.9,
        ))
        self.assertEqual(result.mode, Mode.FACE)

    def test_noisy_at_exact_threshold_gives_face(self) -> None:
        result = self._eval(_inputs(
            noise_level=config.DECISION_NOISE_THRESHOLD,
            face_confidence=0.9, speech_confidence=0.9,
        ))
        self.assertEqual(result.mode, Mode.FACE)

    def test_noisy_reason_mentions_noise(self) -> None:
        result = self._eval(_inputs(noise_level=1.0, face_confidence=0.9))
        self.assertIn("noisy", result.reason.lower())

    def test_just_below_noise_threshold_does_not_trigger_face_from_noise(self) -> None:
        """Below threshold: noise rule must NOT fire."""
        result = self._eval(_inputs(
            noise_level=config.DECISION_NOISE_THRESHOLD - 0.001,
            face_confidence=0.9, speech_confidence=0.9,
        ))
        if result.mode == Mode.FACE:
            self.assertNotIn("noisy", result.reason.lower())


# ===========================================================================
# 6. Rule: Text field focused + quiet → SPEECH
# ===========================================================================

class TestRuleTextFieldFocused(DecisionEngineTestBase):
    """Rule 3: text field focused AND quiet → SPEECH (dictation context)."""

    def test_text_focused_quiet_gives_speech(self) -> None:
        result = self._eval(_inputs(
            noise_level=0.0, text_field_focused=True,
            face_confidence=0.9, speech_confidence=0.9,
        ))
        self.assertEqual(result.mode, Mode.SPEECH)

    def test_text_focused_noisy_does_not_give_speech(self) -> None:
        """Noise rule (priority 2) overrides text-field rule (priority 3)."""
        result = self._eval(_inputs(
            noise_level=1.0, text_field_focused=True,
            face_confidence=0.9, speech_confidence=0.9,
        ))
        self.assertEqual(result.mode, Mode.FACE)

    def test_text_focused_unknown_not_speech(self) -> None:
        """'unknown' (not True) must NOT trigger the text-field rule."""
        result = self._eval(_inputs(
            noise_level=0.0, text_field_focused="unknown",
            face_confidence=0.9, speech_confidence=0.9,
        ))
        self.assertNotEqual(result.mode, Mode.SPEECH)

    def test_text_focused_reason_mentions_text_field(self) -> None:
        result = self._eval(_inputs(
            noise_level=0.0, text_field_focused=True,
        ))
        self.assertEqual(result.mode, Mode.SPEECH)
        self.assertIn("text field", result.reason.lower())


# ===========================================================================
# 7. Rule: Face lost / poor lighting → SPEECH
# ===========================================================================

class TestRuleFaceLost(DecisionEngineTestBase):
    """Rule 4: face_confidence < threshold OR brightness < threshold → SPEECH."""

    def test_low_face_confidence_gives_speech(self) -> None:
        result = self._eval(_inputs(
            noise_level=0.0, face_confidence=0.1,
            brightness=128.0, speech_confidence=0.9,
        ))
        self.assertEqual(result.mode, Mode.SPEECH)

    def test_dark_room_gives_speech(self) -> None:
        result = self._eval(_inputs(
            noise_level=0.0, face_confidence=0.9,
            brightness=5.0, speech_confidence=0.9,
        ))
        self.assertEqual(result.mode, Mode.SPEECH)

    def test_face_lost_reason_mentions_face(self) -> None:
        result = self._eval(_inputs(
            noise_level=0.0, face_confidence=0.1, brightness=128.0,
        ))
        self.assertEqual(result.mode, Mode.SPEECH)
        self.assertIn("face", result.reason.lower())

    def test_at_brightness_threshold_not_face_lost(self) -> None:
        """Exactly at threshold → NOT considered dark → face-lost rule does not fire."""
        result = self._eval(_inputs(
            noise_level=0.0, face_confidence=0.9,
            brightness=config.DECISION_BRIGHTNESS_THRESHOLD,
            speech_confidence=0.9,
        ))
        self.assertNotEqual(result.mode, Mode.SPEECH,
                            "Brightness at threshold should not trigger face-lost rule")

    def test_dark_and_noisy_and_poor_speech_gives_paused(self) -> None:
        """Rule 1 (PAUSED) beats Rule 4 (face lost)."""
        result = self._eval(_inputs(
            noise_level=1.0, face_confidence=0.0,
            brightness=0.0, speech_confidence=0.0,
        ))
        self.assertEqual(result.mode, Mode.PAUSED)


# ===========================================================================
# 8. Rule: Both reliable → BOTH
# ===========================================================================

class TestRuleBothReliable(DecisionEngineTestBase):
    """Rule 5: face good AND speech good AND quiet → BOTH."""

    def test_both_reliable_gives_both(self) -> None:
        result = self._eval(_stable_inputs())
        self.assertEqual(result.mode, Mode.BOTH)

    def test_both_reliable_reason_mentions_both(self) -> None:
        # Seed in a different mode so a switch actually happens
        self._eval(_inputs(noise_level=1.0, face_confidence=0.9))  # → FACE
        result = self._eval(_stable_inputs())                        # → BOTH
        self.assertIn("both reliable", result.reason.lower())

    def test_both_reliable_at_exact_thresholds(self) -> None:
        """Values exactly at their thresholds count as 'reliable'."""
        result = self._eval(_inputs(
            face_confidence=config.DECISION_FACE_CONF_THRESHOLD,
            speech_confidence=config.DECISION_SPEECH_CONF_THRESHOLD,
            noise_level=0.0,
            brightness=config.DECISION_BRIGHTNESS_THRESHOLD,
        ))
        self.assertEqual(result.mode, Mode.BOTH)


# ===========================================================================
# 9. Rule: Fallback → FACE
# ===========================================================================

class TestRuleFallback(DecisionEngineTestBase):
    """Rule 6 (lowest): good face, unreliable speech, quiet → FACE."""

    def test_fallback_gives_face(self) -> None:
        result = self._eval(_inputs(
            noise_level=0.0, face_confidence=0.6,
            brightness=128.0, speech_confidence=0.1,
        ))
        self.assertEqual(result.mode, Mode.FACE)

    def test_fallback_reason_mentions_fallback(self) -> None:
        result = self._eval(_inputs(
            noise_level=0.0, face_confidence=0.6,
            brightness=128.0, speech_confidence=0.1,
        ))
        self.assertEqual(result.mode, Mode.FACE)
        self.assertIn("fallback", result.reason.lower())


# ===========================================================================
# 10. Missing / default inputs
# ===========================================================================

class TestDefaultInputs(DecisionEngineTestBase):
    def test_empty_dict_does_not_raise(self) -> None:
        result = self.engine.evaluate({})
        self.assertIsInstance(result, DecisionResult)
        self.assertIn(result.mode, list(Mode))

    def test_partial_inputs_no_crash(self) -> None:
        result = self.engine.evaluate({"noise_level": 0.5})
        self.assertIsInstance(result, DecisionResult)


# ===========================================================================
# 11. Hysteresis: hold duration guard
# ===========================================================================

class TestHysteresisHoldDuration(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_hold     = config.DECISION_HOLD_DURATION_S
        self._orig_interval = config.DECISION_MIN_SWITCH_INTERVAL_S
        config.DECISION_HOLD_DURATION_S       = 0.1   # 100 ms hold
        config.DECISION_MIN_SWITCH_INTERVAL_S = 0.0   # no switch-interval guard

    def tearDown(self) -> None:
        config.DECISION_HOLD_DURATION_S       = self._orig_hold
        config.DECISION_MIN_SWITCH_INTERVAL_S = self._orig_interval

    def test_mode_does_not_switch_before_hold_expires(self) -> None:
        """Engine stays on current mode while candidate hold timer is running."""
        engine = DecisionEngine()
        # Engine starts at BOTH.  Push face-only scenario.
        face_only = _inputs(noise_level=0.0, face_confidence=0.6,
                            brightness=128.0, speech_confidence=0.1)
        result = engine.evaluate(face_only)
        # First call: candidate registered; hold timer started; mode unchanged.
        self.assertEqual(result.mode, Mode.BOTH)

    def test_mode_switches_after_hold_expires(self) -> None:
        """After holding the candidate for HOLD_DURATION_S, switch commits."""
        engine = DecisionEngine()
        face_only = _inputs(noise_level=0.0, face_confidence=0.6,
                            brightness=128.0, speech_confidence=0.1)
        engine.evaluate(face_only)   # start hold timer
        time.sleep(0.15)             # exceed 100 ms hold
        result = engine.evaluate(face_only)   # hold elapsed → switch
        self.assertEqual(result.mode, Mode.FACE)

    def test_hold_timer_resets_on_different_candidate(self) -> None:
        """If the candidate changes before hold expires, the timer resets."""
        engine = DecisionEngine()
        face_only = _inputs(noise_level=0.0, face_confidence=0.6,
                            brightness=128.0, speech_confidence=0.1)
        engine.evaluate(face_only)         # candidate A: FACE
        time.sleep(0.05)                   # wait less than hold
        engine.evaluate(_stable_inputs())  # candidate B: BOTH → resets timer
        # Now push face_only again: timer should have restarted
        result = engine.evaluate(face_only)
        # Mode must still be BOTH (hold for A was broken, B never switched,
        # C (face again) hasn't held long enough yet)
        self.assertEqual(result.mode, Mode.BOTH)


# ===========================================================================
# 12. Hysteresis: minimum switch interval guard
# ===========================================================================

class TestHysteresisMinSwitchInterval(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_hold     = config.DECISION_HOLD_DURATION_S
        self._orig_interval = config.DECISION_MIN_SWITCH_INTERVAL_S
        config.DECISION_HOLD_DURATION_S       = 0.0   # no hold needed
        config.DECISION_MIN_SWITCH_INTERVAL_S = 0.2   # 200 ms cooldown

    def tearDown(self) -> None:
        config.DECISION_HOLD_DURATION_S       = self._orig_hold
        config.DECISION_MIN_SWITCH_INTERVAL_S = self._orig_interval

    def test_rapid_switch_suppressed_within_interval(self) -> None:
        """A second switch attempted before the cooldown expires is suppressed."""
        engine = DecisionEngine()
        face_only = _inputs(noise_level=0.0, face_confidence=0.6,
                            brightness=128.0, speech_confidence=0.1)
        # Switch 1: BOTH → FACE
        engine.evaluate(face_only)
        engine.evaluate(face_only)  # commits FACE

        # Immediately try to switch back to BOTH (within 200 ms cooldown)
        engine.evaluate(_stable_inputs())
        result = engine.evaluate(_stable_inputs())
        # Must still be FACE because the cooldown hasn't elapsed
        self.assertEqual(result.mode, Mode.FACE)

    def test_switch_allowed_after_interval(self) -> None:
        """After the cooldown elapses, the next switch commits."""
        engine = DecisionEngine()
        face_only = _inputs(noise_level=0.0, face_confidence=0.6,
                            brightness=128.0, speech_confidence=0.1)
        # Switch 1: BOTH → FACE
        engine.evaluate(face_only)
        engine.evaluate(face_only)   # commits FACE

        time.sleep(0.25)             # exceed 200 ms cooldown

        # Switch 2: FACE → BOTH (now allowed)
        engine.evaluate(_stable_inputs())
        result = engine.evaluate(_stable_inputs())
        self.assertEqual(result.mode, Mode.BOTH)


# ===========================================================================
# 13. Manual override
# ===========================================================================

class TestManualOverride(DecisionEngineTestBase):
    def test_set_override_wins_over_all_rules(self) -> None:
        """Manual override ignores sensor readings — even a noisy room."""
        self.engine.set_manual_override(Mode.SPEECH)
        result = self._eval(_inputs(noise_level=1.0, face_confidence=0.9))
        self.assertEqual(result.mode, Mode.SPEECH)

    def test_override_is_override_flag_true(self) -> None:
        self.engine.set_manual_override(Mode.FACE)
        result = self._eval(_stable_inputs())
        self.assertTrue(result.is_override)

    def test_override_face_mode(self) -> None:
        self.engine.set_manual_override(Mode.FACE)
        result = self._eval(_stable_inputs())
        self.assertEqual(result.mode, Mode.FACE)

    def test_override_speech_mode(self) -> None:
        self.engine.set_manual_override(Mode.SPEECH)
        result = self._eval(_stable_inputs())
        self.assertEqual(result.mode, Mode.SPEECH)

    def test_override_both_mode(self) -> None:
        self.engine.set_manual_override(Mode.BOTH)
        result = self._eval(_stable_inputs())
        self.assertEqual(result.mode, Mode.BOTH)

    def test_override_paused_mode(self) -> None:
        self.engine.set_manual_override(Mode.PAUSED)
        result = self._eval(_stable_inputs())
        self.assertEqual(result.mode, Mode.PAUSED)

    def test_is_override_property_true_when_set(self) -> None:
        self.engine.set_manual_override(Mode.SPEECH)
        self.assertTrue(self.engine.is_override)

    def test_is_override_property_false_before_set(self) -> None:
        self.assertFalse(self.engine.is_override)

    def test_override_survives_multiple_evaluates(self) -> None:
        """Override must persist across many evaluate() calls."""
        self.engine.set_manual_override(Mode.PAUSED)
        for _ in range(10):
            result = self.engine.evaluate(_stable_inputs())
            self.assertEqual(result.mode, Mode.PAUSED)
            self.assertTrue(result.is_override)


# ===========================================================================
# 14. Return to automatic mode (clear_manual_override)
# ===========================================================================

class TestClearOverride(DecisionEngineTestBase):
    def test_clear_override_returns_to_auto_rules(self) -> None:
        """After clearing override, the engine obeys rules again."""
        self.engine.set_manual_override(Mode.BOTH)
        self.engine.clear_manual_override()
        # In a noisy room with good face, automatic mode gives FACE
        result = self._eval(_inputs(noise_level=1.0, face_confidence=0.9))
        self.assertEqual(result.mode, Mode.FACE)

    def test_clear_override_flag_false(self) -> None:
        self.engine.set_manual_override(Mode.BOTH)
        self.engine.clear_manual_override()
        self.assertFalse(self.engine.is_override)

    def test_clear_before_set_no_crash(self) -> None:
        """Clearing when no override is set must not raise."""
        self.engine.clear_manual_override()
        self.assertFalse(self.engine.is_override)

    def test_clear_then_evaluate_not_override(self) -> None:
        self.engine.set_manual_override(Mode.PAUSED)
        self.engine.clear_manual_override()
        result = self._eval(_stable_inputs())
        self.assertFalse(result.is_override)

    def test_clear_resets_candidate_timer(self) -> None:
        """After clearing, the hold timer starts fresh (no stale candidate)."""
        self.engine.set_manual_override(Mode.SPEECH)
        self.engine.clear_manual_override()
        # With hold=0 and interval=0 the first post-clear call registers,
        # and the second commits.  Should NOT crash or use a stale timer.
        result = self._eval(_inputs(noise_level=1.0, face_confidence=0.9))
        self.assertEqual(result.mode, Mode.FACE)


# ===========================================================================
# 15. get_decision() — non-evaluating snapshot
# ===========================================================================

class TestGetDecision(DecisionEngineTestBase):
    def test_get_decision_returns_decision_result(self) -> None:
        result = self.engine.get_decision()
        self.assertIsInstance(result, DecisionResult)

    def test_get_decision_matches_current_mode(self) -> None:
        self._eval(_stable_inputs())
        snap = self.engine.get_decision()
        self.assertEqual(snap.mode, self.engine.current_mode)

    def test_get_decision_has_timestamp(self) -> None:
        result = self.engine.get_decision()
        self.assertIsInstance(result.timestamp, float)
        self.assertGreater(result.timestamp, 0.0)

    def test_get_decision_does_not_change_mode(self) -> None:
        """get_decision() must be a pure read — no side effects."""
        before = self.engine.current_mode
        self.engine.get_decision()
        after = self.engine.current_mode
        self.assertEqual(before, after)


# ===========================================================================
# 16. DecisionResult has inputs snapshot
# ===========================================================================

class TestDecisionInputsSnapshot(DecisionEngineTestBase):
    def test_result_includes_inputs_on_switch(self) -> None:
        inp = _inputs(noise_level=1.0, face_confidence=0.9)
        result = self._eval(inp)
        self.assertIn("noise_level", result.inputs)
        self.assertAlmostEqual(
            result.inputs["noise_level"], inp["noise_level"], places=5
        )

    def test_result_inputs_is_a_copy(self) -> None:
        """Mutating the returned inputs dict must not affect future evaluations."""
        inp = _stable_inputs()
        result = self._eval(inp)
        result.inputs["noise_level"] = 999.0
        result2 = self._eval(inp)
        self.assertNotAlmostEqual(result2.inputs.get("noise_level"), 999.0)


# ===========================================================================
# 17. Thread safety
# ===========================================================================

class TestThreadSafety(DecisionEngineTestBase):
    def test_concurrent_evaluate_no_exception(self) -> None:
        """Multiple threads calling evaluate() simultaneously must not crash."""
        errors: list = []

        def _worker() -> None:
            try:
                for _ in range(20):
                    self.engine.evaluate(_stable_inputs())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        self.assertEqual(errors, [], f"Thread errors: {errors}")

    def test_concurrent_override_and_evaluate(self) -> None:
        """override + evaluate from different threads must not corrupt state."""
        errors: list = []

        def _evaluator() -> None:
            try:
                for _ in range(30):
                    self.engine.evaluate(_stable_inputs())
            except Exception as exc:
                errors.append(exc)

        def _overrider() -> None:
            try:
                for mode in [Mode.FACE, Mode.SPEECH, Mode.BOTH, Mode.PAUSED]:
                    self.engine.set_manual_override(mode)
                    time.sleep(0.001)
                self.engine.clear_manual_override()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_evaluator)
        t2 = threading.Thread(target=_overrider)
        t1.start(); t2.start()
        t1.join(timeout=5.0); t2.join(timeout=5.0)
        self.assertEqual(errors, [], f"Thread errors: {errors}")

    def test_concurrent_get_decision_safe(self) -> None:
        """get_decision() can safely be called from multiple threads."""
        errors: list = []

        def _reader() -> None:
            try:
                for _ in range(50):
                    r = self.engine.get_decision()
                    self.assertIn(r.mode, list(Mode))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        self.assertEqual(errors, [])


# ===========================================================================
# 18. CSV logger integration
# ===========================================================================

class TestCsvLoggerIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_hold     = config.DECISION_HOLD_DURATION_S
        self._orig_interval = config.DECISION_MIN_SWITCH_INTERVAL_S
        config.DECISION_HOLD_DURATION_S       = 0.0
        config.DECISION_MIN_SWITCH_INTERVAL_S = 0.0

    def tearDown(self) -> None:
        config.DECISION_HOLD_DURATION_S       = self._orig_hold
        config.DECISION_MIN_SWITCH_INTERVAL_S = self._orig_interval

    def test_csv_logged_on_switch(self) -> None:
        """When mode switches, csv_logger.log() is called."""
        mock_csv = MagicMock()
        engine = DecisionEngine(csv_logger=mock_csv)
        # Force BOTH → FACE by calling twice with noisy inputs
        noisy = _inputs(noise_level=1.0, face_confidence=0.9)
        engine.evaluate(noisy)   # candidate registered
        engine.evaluate(noisy)   # switch commits → CSV called
        mock_csv.log.assert_called()

    def test_csv_called_with_decision_event_type(self) -> None:
        """The event type logged must be 'decision'."""
        mock_csv = MagicMock()
        engine = DecisionEngine(csv_logger=mock_csv)
        noisy = _inputs(noise_level=1.0, face_confidence=0.9)
        engine.evaluate(noisy)
        engine.evaluate(noisy)
        calls = [c for c in mock_csv.log.call_args_list if c[0][0] == "decision"]
        self.assertGreaterEqual(len(calls), 1,
                                "Expected at least one 'decision' log event")

    def test_csv_message_contains_mode_names(self) -> None:
        """The logged message must name both old and new modes."""
        mock_csv = MagicMock()
        engine = DecisionEngine(csv_logger=mock_csv)
        noisy = _inputs(noise_level=1.0, face_confidence=0.9)
        engine.evaluate(noisy)
        engine.evaluate(noisy)
        # Find the decision log call
        for c in mock_csv.log.call_args_list:
            if c[0][0] == "decision":
                msg = str(c[1].get("message", "") or c[0][1].get("message", ""))
                self.assertIn("FACE", msg)
                return
        self.fail("No 'decision' CSV log entry found")

    def test_no_csv_no_crash(self) -> None:
        """Engine with csv_logger=None must never raise, even on a switch."""
        engine = DecisionEngine(csv_logger=None)
        noisy = _inputs(noise_level=1.0, face_confidence=0.9)
        engine.evaluate(noisy)
        result = engine.evaluate(noisy)
        self.assertEqual(result.mode, Mode.FACE)


# ===========================================================================
# 19. Logging
# ===========================================================================

class TestLogging(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_hold     = config.DECISION_HOLD_DURATION_S
        self._orig_interval = config.DECISION_MIN_SWITCH_INTERVAL_S
        config.DECISION_HOLD_DURATION_S       = 0.0
        config.DECISION_MIN_SWITCH_INTERVAL_S = 0.0

    def tearDown(self) -> None:
        config.DECISION_HOLD_DURATION_S       = self._orig_hold
        config.DECISION_MIN_SWITCH_INTERVAL_S = self._orig_interval

    def test_logger_info_called_on_switch(self) -> None:
        """logger.info is called with 'MODE SWITCH' when a switch commits."""
        with patch("context.decision_engine.logger") as mock_log:
            engine = DecisionEngine()
            noisy = _inputs(noise_level=1.0, face_confidence=0.9)
            engine.evaluate(noisy)
            engine.evaluate(noisy)
            info_msgs = " ".join(str(c) for c in mock_log.info.call_args_list)
            self.assertIn("MODE SWITCH", info_msgs)

    def test_paused_triggers_logger_warning(self) -> None:
        """PAUSED mode must emit a logger.warning call."""
        with patch("context.decision_engine.logger") as mock_log:
            engine = DecisionEngine()
            paused = _inputs(noise_level=1.0, face_confidence=0.0,
                             brightness=5.0, speech_confidence=0.0)
            engine.evaluate(paused)
            engine.evaluate(paused)
            warn_msgs = " ".join(str(c) for c in mock_log.warning.call_args_list)
            self.assertTrue(
                "paused" in warn_msgs.lower() or "unreliable" in warn_msgs.lower(),
                f"Expected PAUSED warning, got: {warn_msgs}",
            )

    def test_every_switch_logs_reason(self) -> None:
        """The info log must include the reason string."""
        with patch("context.decision_engine.logger") as mock_log:
            engine = DecisionEngine()
            noisy = _inputs(noise_level=1.0, face_confidence=0.9)
            engine.evaluate(noisy)
            engine.evaluate(noisy)
            info_msgs = " ".join(str(c) for c in mock_log.info.call_args_list)
            self.assertIn("noisy", info_msgs.lower())


# ===========================================================================
# 20. Config constants exist and are sane
# ===========================================================================

class TestConfigConstants(unittest.TestCase):
    def test_decision_hold_duration_exists(self) -> None:
        self.assertTrue(hasattr(config, "DECISION_HOLD_DURATION_S"))
        self.assertIsInstance(config.DECISION_HOLD_DURATION_S, float)
        self.assertGreaterEqual(config.DECISION_HOLD_DURATION_S, 0.0)

    def test_decision_min_switch_interval_exists(self) -> None:
        self.assertTrue(hasattr(config, "DECISION_MIN_SWITCH_INTERVAL_S"))
        self.assertIsInstance(config.DECISION_MIN_SWITCH_INTERVAL_S, float)
        self.assertGreaterEqual(config.DECISION_MIN_SWITCH_INTERVAL_S, 0.0)

    def test_decision_face_conf_threshold_exists_and_sane(self) -> None:
        self.assertTrue(hasattr(config, "DECISION_FACE_CONF_THRESHOLD"))
        self.assertGreater(config.DECISION_FACE_CONF_THRESHOLD, 0.0)
        self.assertLess(config.DECISION_FACE_CONF_THRESHOLD, 1.0)

    def test_decision_noise_threshold_exists_and_sane(self) -> None:
        self.assertTrue(hasattr(config, "DECISION_NOISE_THRESHOLD"))
        self.assertGreater(config.DECISION_NOISE_THRESHOLD, 0.0)
        self.assertLess(config.DECISION_NOISE_THRESHOLD, 1.0)

    def test_decision_brightness_threshold_exists_and_sane(self) -> None:
        self.assertTrue(hasattr(config, "DECISION_BRIGHTNESS_THRESHOLD"))
        self.assertGreater(config.DECISION_BRIGHTNESS_THRESHOLD, 0.0)
        self.assertLess(config.DECISION_BRIGHTNESS_THRESHOLD, 255.0)

    def test_decision_speech_conf_threshold_exists_and_sane(self) -> None:
        self.assertTrue(hasattr(config, "DECISION_SPEECH_CONF_THRESHOLD"))
        self.assertGreater(config.DECISION_SPEECH_CONF_THRESHOLD, 0.0)
        self.assertLess(config.DECISION_SPEECH_CONF_THRESHOLD, 1.0)


# ===========================================================================
# 21. Rule priority: higher-priority rule wins
# ===========================================================================

class TestRulePriority(DecisionEngineTestBase):
    def test_rule1_beats_rule2_paused_beats_noisy(self) -> None:
        """PAUSED (rule 1) must override FACE/noisy (rule 2)."""
        result = self._eval(_inputs(
            noise_level=1.0, face_confidence=0.0,
            brightness=5.0, speech_confidence=0.0,
        ))
        self.assertEqual(result.mode, Mode.PAUSED)

    def test_rule2_beats_rule3_noisy_beats_text_field(self) -> None:
        """FACE/noisy (rule 2) must override SPEECH/text-field (rule 3)."""
        result = self._eval(_inputs(
            noise_level=1.0, text_field_focused=True,
            face_confidence=0.9, speech_confidence=0.9,
        ))
        self.assertEqual(result.mode, Mode.FACE)

    def test_rule2_beats_rule5_noisy_beats_both_reliable(self) -> None:
        """In a noisy room, even reliable speech → FACE not BOTH."""
        result = self._eval(_inputs(
            noise_level=1.0, face_confidence=0.9, speech_confidence=0.9,
        ))
        self.assertEqual(result.mode, Mode.FACE)

    def test_rule3_beats_rule5_text_field_beats_both_reliable(self) -> None:
        """Text field focused + quiet overrides BOTH (rule 5)."""
        result = self._eval(_inputs(
            noise_level=0.0, text_field_focused=True,
            face_confidence=0.9, speech_confidence=0.9,
        ))
        self.assertEqual(result.mode, Mode.SPEECH)

    def test_rule4_beats_rule5_face_lost_beats_both_reliable(self) -> None:
        """Face lost (rule 4) → SPEECH even if speech would be reliable."""
        result = self._eval(_inputs(
            noise_level=0.0, face_confidence=0.1,
            brightness=128.0, speech_confidence=0.9,
        ))
        self.assertEqual(result.mode, Mode.SPEECH)


# ===========================================================================
# 22. No switch needed: same target as current
# ===========================================================================

class TestNoSwitchNeeded(DecisionEngineTestBase):
    def test_same_mode_no_switch_stable(self) -> None:
        """When target matches current mode, result is stable."""
        self._eval(_stable_inputs())                    # → BOTH
        result = self._eval(_stable_inputs())           # still BOTH
        self.assertEqual(result.mode, Mode.BOTH)

    def test_candidate_cleared_when_same_mode(self) -> None:
        """Engine resets candidate when target equals current mode."""
        # Switch to FACE
        self._eval(_inputs(noise_level=1.0, face_confidence=0.9))
        # Confirm FACE
        self.assertEqual(self.engine.current_mode, Mode.FACE)
        # Evaluate same FACE scenario — candidate should clear
        result = self._eval(_inputs(noise_level=1.0, face_confidence=0.9))
        self.assertEqual(result.mode, Mode.FACE)


if __name__ == "__main__":
    unittest.main()
