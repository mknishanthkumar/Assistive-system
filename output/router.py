"""
output/router.py - Mode-aware event router for the Assistive Control System.

Stage 4: simple pass-through (face + speech always active).
Stage 5: upgraded to filter events based on the DecisionEngine's active Mode,
         so the inactive module's actions are silently dropped.

The Router receives events from FaceTracker and SpeechEngine and conditionally
forwards them to the OutputController depending on the current mode:

  FACE   – only face events forwarded; speech events dropped.
  SPEECH – only speech events forwarded; face cursor/click events dropped.
  BOTH   – all events forwarded (face for pointing, voice for confirmation).
  PAUSED – all events dropped (both modules inactive).

Design rationale
----------------
Stage 4 Router is a minimal pass-through class.  Stage 5 adds a DecisionEngine
reference.  The two are backward-compatible: if no engine is supplied the Router
behaves identically to Stage 4 (always BOTH).

Usage
-----
    # Stage 4 (no engine — pass-through)
    router = Router(controller)

    # Stage 5 (mode-filtered)
    engine = DecisionEngine()
    router = Router(controller, decision_engine=engine)

    router.handle_face_events(face_events)      # called by face thread
    router.handle_speech_events(speech_events)  # called by speech thread

Both methods are thread-safe and may be called from different threads
concurrently.
"""

import logging
import threading
from typing import Any, Dict, Optional

import config
from context.decision_engine import Mode

logger = logging.getLogger("assistive.router")


class Router:
    """
    Mode-aware event router.

    Parameters
    ----------
    controller : OutputController
        The action executor to forward events to.
    csv_logger : CsvLogger | None
        Optional logger for routing decisions.
    decision_engine : DecisionEngine | None
        Stage 5 context engine.  When supplied, face/speech events are filtered
        by the engine's current mode.  When None, all events are forwarded
        (Stage 4 backward-compatible pass-through behaviour).
    """

    def __init__(
        self,
        controller: Any,
        csv_logger: Optional[Any] = None,
        decision_engine: Optional[Any] = None,
    ) -> None:
        self._controller = controller
        self._csv = csv_logger
        self._engine = decision_engine
        self._lock = threading.Lock()   # for future subclasses that need state

    # ------------------------------------------------------------------
    # Face events
    # ------------------------------------------------------------------

    def handle_face_events(self, events: Dict[str, Any]) -> None:
        """
        Forward face events to the controller, filtered by the active mode.

        When a DecisionEngine is attached:
          - FACE or BOTH → forward cursor movement and click gestures.
          - SPEECH or PAUSED → drop all face events silently.

        Without a DecisionEngine (Stage 4 pass-through):
          - All face events are always forwarded.

        Face confidence is passed as the confidence score so the controller
        can apply its threshold gate.
        """
        # --- Mode filtering -------------------------------------------
        # Determine whether face events are allowed in the current mode.
        if self._engine is not None:
            mode = self._engine.current_mode
            if mode not in (Mode.FACE, Mode.BOTH):
                logger.debug(
                    "Face events suppressed (mode=%s)", mode.value
                )
                return
        # --------------------------------------------------------------

        confidence = float(events.get("face_confidence", 1.0))

        # Filter out low-confidence frames to prevent erratic cursor jumps
        min_conf = getattr(config, "DECISION_FACE_CONF_THRESHOLD", 0.4)
        if confidence < min_conf:
            logger.debug(
                "Face events suppressed: low confidence (%.2f < %.2f)",
                confidence, min_conf,
            )
            return

        # Cursor movement (always attempted when face detected and confidence is good)
        if events.get("face_detected", False):
            dx = float(events.get("cursor_delta_x", 0.0))
            dy = float(events.get("cursor_delta_y", 0.0))
            if dx != 0.0 or dy != 0.0:
                self._controller.move_cursor(dx, dy)

        logger.debug(
            "Router face event forwarded: mode=%s, conf=%.2f, delta=(%.1f, %.1f)",
            self._engine.current_mode.value if self._engine else "PASS",
            confidence, dx, dy,
        )

        # Edge-triggered click gestures
        if events.get("left_click"):
            result = self._controller.execute("click", confidence=confidence, source="face")
            if self._csv:
                self._csv.log("action", {
                    "action_command": "click",
                    "action_success": result.success,
                    "action_message": result.message,
                    "message": f"[face] left_click",
                })
            logger.debug("Face left_click → %s", result.message)

        if events.get("right_click"):
            result = self._controller.execute("right_click", confidence=confidence, source="face")
            if self._csv:
                self._csv.log("action", {
                    "action_command": "right_click",
                    "action_success": result.success,
                    "action_message": result.message,
                    "message": f"[face] right_click",
                })
            logger.debug("Face right_click → %s", result.message)

    # ------------------------------------------------------------------
    # Speech events
    # ------------------------------------------------------------------

    def handle_speech_events(self, events: Dict[str, Any]) -> None:
        """
        Forward speech events to the controller, filtered by the active mode.

        When a DecisionEngine is attached:
          - SPEECH or BOTH → forward commands and dictation.
          - stop_control is ALWAYS forwarded regardless of mode (safety command).
          - FACE or PAUSED → drop all speech events silently (except stop_control).

        Without a DecisionEngine (Stage 4 pass-through):
          - All speech events are always forwarded.

        Actions taken:
        - command set → controller.execute(command)
        - dictation_text set → controller.type_dictation(text)
        - stop_control command → controller.trigger_emergency_stop() (always)
        """
        confidence = float(events.get("speech_confidence", 0.0))
        cmd = events.get("command")
        dictation = events.get("dictation_text")

        # --- stop_control bypasses mode filter (safety) ---------------
        if cmd == "stop_control":
            logger.warning("Voice command 'stop control' received — triggering emergency stop.")
            self._controller.trigger_emergency_stop()
            if self._csv:
                self._csv.log("action", {
                    "action_command": "stop_control",
                    "action_success": True,
                    "message": "Emergency stop via voice command.",
                })
            return

        # --- Mode filtering -------------------------------------------
        if self._engine is not None:
            mode = self._engine.current_mode
            if mode not in (Mode.SPEECH, Mode.BOTH):
                logger.debug(
                    "Speech events suppressed (mode=%s)", mode.value
                )
                return
        # --------------------------------------------------------------

        if cmd:
            result = self._controller.execute(cmd, confidence=confidence, source="speech")
            if self._csv:
                self._csv.log("action", {
                    "action_command": cmd,
                    "action_success": result.success,
                    "action_message": result.message,
                    "message": f"[speech] {cmd}",
                })
            logger.debug("Speech command '%s' → %s", cmd, result.message)

        if dictation:
            result = self._controller.type_dictation(dictation)
            if self._csv:
                self._csv.log("action", {
                    "action_command": "dictation",
                    "action_success": result.success,
                    "action_message": result.message,
                    "message": f"[speech] dictation",
                })
            logger.debug("Speech dictation → %s", result.message)

