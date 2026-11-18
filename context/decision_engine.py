"""
context/decision_engine.py - Context-aware decision engine for the Assistive Control System.

Decides which input mode is active (FACE, SPEECH, BOTH, or PAUSED) based on
real-time sensor readings.  Rules are evaluated in priority order; the highest-
priority rule that matches wins.

Modes
-----
  FACE   – face/head tracking active; speech commands ignored by Router.
  SPEECH – voice commands active; face cursor/click ignored by Router.
  BOTH   – both sources active simultaneously (face for pointing, voice for
            typing and confirmation).
  PAUSED – both sources disabled; a warning is issued.

Rule priority (highest first)
------------------------------
  1. BOTH_UNRELIABLE → PAUSED       : both face and speech unreliable
  2. NOISY → FACE                   : noise above threshold, speech disabled
  3. TEXT_FIELD_FOCUSED + QUIET → SPEECH : dictation context
  4. FACE_LOST → SPEECH             : face not detected / poor lighting
  5. BOTH_RELIABLE → BOTH           : face and speech both reliable
  6. FALLBACK → FACE                : default when only face is marginal

Hysteresis
----------
Two config knobs prevent flapping:
  DECISION_MIN_SWITCH_INTERVAL_S : minimum wall-clock seconds between any
      two automatic mode switches.
  DECISION_HOLD_DURATION_S       : a new rule must hold continuously for at
      least this many seconds before it triggers a switch.

Manual override
---------------
  set_manual_override(mode)   : forces a specific mode; persists until cleared.
  clear_manual_override()     : returns to automatic rule evaluation.
  "switch to voice" phrase    : handled upstream; calls set_manual_override(SPEECH).
  "switch to face" phrase     : handled upstream; calls set_manual_override(FACE).
  Jaw-gesture shortcut        : caller can invoke set_manual_override() too.

Thread safety
-------------
  All public methods acquire self._lock before touching shared state.
  get_decision() is safe to call from any thread at any time.

Logging
-------
  Every decision transition is logged with:
    timestamp, all inputs, chosen mode, reason, is_override flag.
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import config

logger = logging.getLogger("assistive.decision_engine")


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class Mode(Enum):
    """Active input mode."""
    FACE   = "FACE"    # head tracking only
    SPEECH = "SPEECH"  # voice commands only
    BOTH   = "BOTH"    # face + speech simultaneously
    PAUSED = "PAUSED"  # all input disabled (both unreliable)


# ---------------------------------------------------------------------------
# Decision result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    """
    Returned by DecisionEngine.get_decision() and recorded in every log entry.

    Fields
    ------
    mode        : chosen Mode
    reason      : human-readable explanation of why this mode was chosen
    is_override : True when a manual override is in effect
    timestamp   : monotonic clock seconds when the decision was made
    inputs      : snapshot of the sensor readings that drove this decision
    """
    mode:        Mode
    reason:      str
    is_override: bool
    timestamp:   float
    inputs:      Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Context-aware rule engine that maps sensor readings to an active Mode.

    Parameters
    ----------
    csv_logger : CsvLogger | None
        Optional CSV logger for decision audit trail.
    """

    def __init__(self, csv_logger: Optional[Any] = None) -> None:
        self._csv = csv_logger
        self._lock = threading.Lock()

        # Current state
        self._current_mode:  Mode    = Mode.BOTH   # start permissive
        self._current_reason: str    = "initial"
        self._is_override:   bool    = False
        self._override_mode: Optional[Mode] = None

        # Hysteresis state
        # _candidate: (mode, reason, time_first_seen)
        self._candidate: Optional[Tuple[Mode, str, float]] = None
        self._last_switch_time: float = 0.0   # monotonic; starts far in past

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, inputs: Dict[str, Any]) -> DecisionResult:
        """
        Evaluate sensor inputs and (if appropriate) switch the active mode.

        This should be called regularly, e.g. once per sensor poll cycle.

        Parameters
        ----------
        inputs : dict
            Expected keys (all optional; missing → safe defaults):
              noise_level        float [0-1]
              face_confidence    float [0-1]
              brightness         float [0-255]
              active_app         str
              text_field_focused bool | "unknown"
              speech_confidence  float [0-1]

        Returns
        -------
        DecisionResult
            The current (possibly just-switched) decision.
        """
        with self._lock:
            now = time.monotonic()

            # Manual override always wins — skip rule evaluation entirely.
            if self._is_override and self._override_mode is not None:
                result = DecisionResult(
                    mode=self._override_mode,
                    reason="manual override",
                    is_override=True,
                    timestamp=now,
                    inputs=dict(inputs),
                )
                # Still log each override evaluation so the audit trail is
                # continuous; only log if mode changed.
                if self._override_mode != self._current_mode:
                    self._apply_switch(result, now)
                return result

            # Derive booleans from raw inputs.
            noise_level       = float(inputs.get("noise_level", 0.0))
            face_confidence   = float(inputs.get("face_confidence", 0.0))
            brightness        = float(inputs.get("brightness", 128.0))
            text_field_focused = inputs.get("text_field_focused", False)
            speech_confidence = float(inputs.get("speech_confidence", 0.0))

            # Derived booleans — all thresholds live in config.py
            noisy           = noise_level       >= config.DECISION_NOISE_THRESHOLD
            face_good       = face_confidence   >= config.DECISION_FACE_CONF_THRESHOLD
            face_poor       = face_confidence   <  config.DECISION_FACE_CONF_THRESHOLD
            dark            = brightness        <  config.DECISION_BRIGHTNESS_THRESHOLD
            face_lost       = face_poor or dark
            speech_good     = speech_confidence >= config.DECISION_SPEECH_CONF_THRESHOLD
            text_focused    = text_field_focused is True  # "unknown" → False safely
            quiet           = not noisy
            pointing_active = bool(inputs.get("pointing_active", False))

            # --- Rule evaluation (priority order: highest first) ----------

            # Rule 1: Both unreliable → PAUSED (safety catch-all)
            if face_lost and not speech_good and noisy:
                target_mode   = Mode.PAUSED
                target_reason = (
                    f"both unreliable: face_conf={face_confidence:.2f} "
                    f"brightness={brightness:.1f} noise={noise_level:.3f}"
                )

            # Rule 2: Noisy environment → FACE only (voice unreliable)
            elif noisy:
                target_mode   = Mode.FACE
                target_reason = (
                    f"noisy environment: noise={noise_level:.3f} >= "
                    f"threshold={config.DECISION_NOISE_THRESHOLD:.3f}"
                )

            # Rule 2b: Active pointing / scrolling / clicking context → FACE
            elif pointing_active and face_good:
                target_mode   = Mode.FACE
                target_reason = (
                    f"pointing/scrolling/clicking context: active face pointing"
                )

            # Rule 3: Text field focused + quiet → SPEECH (dictation)
            elif text_focused and quiet:
                target_mode   = Mode.SPEECH
                target_reason = (
                    f"text field focused + quiet environment: "
                    f"noise={noise_level:.3f}"
                )

            # Rule 4: Face lost or dark → SPEECH with navigation commands
            elif face_lost:
                target_mode   = Mode.SPEECH
                target_reason = (
                    f"face lost/poor lighting: face_conf={face_confidence:.2f} "
                    f"brightness={brightness:.1f}"
                )

            # Rule 5: Both reliable → BOTH
            elif face_good and speech_good:
                target_mode   = Mode.BOTH
                target_reason = (
                    f"both reliable: face_conf={face_confidence:.2f} "
                    f"speech_conf={speech_confidence:.2f}"
                )

            # Rule 6: Fallback → FACE (marginal face, quiet)
            else:
                target_mode   = Mode.FACE
                target_reason = (
                    f"fallback to face: face_conf={face_confidence:.2f} "
                    f"speech_conf={speech_confidence:.2f}"
                )

            # --- Hysteresis -------------------------------------------
            # The candidate must hold continuously for DECISION_HOLD_DURATION_S
            # AND the last switch must be at least DECISION_MIN_SWITCH_INTERVAL_S ago.
            result = self._apply_hysteresis(
                target_mode, target_reason, inputs, now
            )
            return result

    def get_decision(self) -> DecisionResult:
        """
        Return the current decision without re-evaluating rules.

        Thread-safe; never blocks for long.
        """
        with self._lock:
            return DecisionResult(
                mode=self._current_mode,
                reason=self._current_reason,
                is_override=self._is_override,
                timestamp=time.monotonic(),
                inputs={},
            )

    def set_manual_override(self, mode: Mode) -> None:
        """
        Force a specific mode regardless of sensor state.

        Call clear_manual_override() to return to automatic mode.

        Parameters
        ----------
        mode : Mode
            One of Mode.FACE, Mode.SPEECH, Mode.BOTH, Mode.PAUSED.
        """
        with self._lock:
            self._is_override   = True
            self._override_mode = mode
            # Reset candidate so hysteresis doesn't fire immediately after clearing
            self._candidate     = None
            logger.info("Manual override set: mode=%s", mode.value)

    def clear_manual_override(self) -> None:
        """
        Return to automatic rule-based mode selection.

        The engine immediately re-evaluates inputs on the next evaluate() call.
        """
        with self._lock:
            self._is_override   = False
            self._override_mode = None
            # Reset candidate so the first valid reading after clearing doesn't
            # get an artificial head-start on the hold timer.
            self._candidate     = None
            logger.info("Manual override cleared — returning to automatic mode.")

    @property
    def is_override(self) -> bool:
        """True if a manual override is currently active."""
        with self._lock:
            return self._is_override

    @property
    def current_mode(self) -> Mode:
        """The currently active mode (thread-safe read)."""
        with self._lock:
            return self._current_mode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_hysteresis(
        self,
        target_mode:   Mode,
        target_reason: str,
        inputs:        Dict[str, Any],
        now:           float,
    ) -> DecisionResult:
        """
        Apply two-stage hysteresis before accepting a mode switch.

          1. The target mode must hold for DECISION_HOLD_DURATION_S seconds.
          2. DECISION_MIN_SWITCH_INTERVAL_S must have elapsed since the last switch.

        If neither guard fires, the target mode is accepted and a log entry written.
        """
        # Already in the target mode — no switch needed; reset candidate.
        if target_mode == self._current_mode:
            self._candidate = None
            return DecisionResult(
                mode=self._current_mode,
                reason=self._current_reason,
                is_override=False,
                timestamp=now,
                inputs=dict(inputs),
            )

        # Track how long this candidate has been consistent.
        if self._candidate is None or self._candidate[0] != target_mode:
            # New candidate — start the hold timer.
            self._candidate = (target_mode, target_reason, now)
            logger.debug(
                "New candidate mode=%s (hold timer started)", target_mode.value
            )
            return DecisionResult(
                mode=self._current_mode,
                reason=self._current_reason,
                is_override=False,
                timestamp=now,
                inputs=dict(inputs),
            )

        # Candidate is consistent — check hold duration.
        hold_elapsed = now - self._candidate[2]
        if hold_elapsed < config.DECISION_HOLD_DURATION_S:
            # Not long enough yet.
            logger.debug(
                "Candidate mode=%s holding (%.2fs / %.2fs)",
                target_mode.value, hold_elapsed, config.DECISION_HOLD_DURATION_S,
            )
            return DecisionResult(
                mode=self._current_mode,
                reason=self._current_reason,
                is_override=False,
                timestamp=now,
                inputs=dict(inputs),
            )

        # Hold duration satisfied — check minimum switch interval.
        since_last = now - self._last_switch_time
        if since_last < config.DECISION_MIN_SWITCH_INTERVAL_S:
            logger.debug(
                "Switch suppressed (cooldown %.2fs / %.2fs)",
                since_last, config.DECISION_MIN_SWITCH_INTERVAL_S,
            )
            return DecisionResult(
                mode=self._current_mode,
                reason=self._current_reason,
                is_override=False,
                timestamp=now,
                inputs=dict(inputs),
            )

        # Both guards cleared — commit the switch.
        result = DecisionResult(
            mode=target_mode,
            reason=target_reason,
            is_override=False,
            timestamp=now,
            inputs=dict(inputs),
        )
        self._apply_switch(result, now)
        return result

    def _apply_switch(self, result: DecisionResult, now: float) -> None:
        """
        Persist a mode switch, update timers, and emit a structured log entry.

        Called while self._lock is already held.
        """
        old_mode = self._current_mode
        self._current_mode   = result.mode
        self._current_reason = result.reason
        self._last_switch_time = now
        self._candidate = None   # reset hold timer after switch

        # Structured decision log — matches the spec: timestamp, all inputs,
        # chosen mode, reason, is_override.
        log_entry = {
            "timestamp":   now,
            "mode":        result.mode.value,
            "prev_mode":   old_mode.value,
            "reason":      result.reason,
            "is_override": result.is_override,
            "inputs":      result.inputs,
        }
        logger.info(
            "MODE SWITCH %s → %s | reason=%s | override=%s | "
            "inputs=%s",
            old_mode.value,
            result.mode.value,
            result.reason,
            result.is_override,
            result.inputs,
        )

        if result.mode == Mode.PAUSED:
            logger.warning(
                "PAUSED: both face and speech are unreliable. "
                "Improve lighting or move to a quieter environment."
            )

        # CSV audit trail
        if self._csv is not None:
            try:
                self._csv.log("decision", {
                    "message":     (
                        f"MODE {old_mode.value}->{result.mode.value} "
                        f"reason={result.reason} override={result.is_override}"
                    ),
                    "active_window": str(result.inputs.get("active_app", "")),
                })
            except Exception as exc:
                logger.debug("CSV log failed: %s", exc)

