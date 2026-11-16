"""
output/router.py - Simple event router for Stage 4 integration.

The Router receives events from FaceTracker and SpeechEngine and
forwards them directly to the OutputController.  There is NO mode-switching
logic here: face and speech events can act simultaneously.

Design rationale
----------------
Keeping the Router as a small, replaceable class means the Stage 5 context
engine can subclass or replace it with a smarter routing strategy without
touching the controller or the input modules.

Usage
-----
    router = Router(controller)
    router.handle_face_events(face_events)   # called by face thread
    router.handle_speech_events(speech_events)  # called by speech thread

Both methods are thread-safe and may be called from different threads
concurrently.
"""

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("assistive.router")


class Router:
    """
    Minimal pass-through router.

    Parameters
    ----------
    controller : OutputController
        The action executor to forward events to.
    csv_logger : CsvLogger | None
        Optional logger for routing decisions.
    """

    def __init__(
        self,
        controller: Any,
        csv_logger: Optional[Any] = None,
    ) -> None:
        self._controller = controller
        self._csv = csv_logger
        self._lock = threading.Lock()   # for future subclasses that need state

    # ------------------------------------------------------------------
    # Face events
    # ------------------------------------------------------------------

    def handle_face_events(self, events: Dict[str, Any]) -> None:
        """
        Forward face events to the controller.

        Actions taken:
        - left_click=True  → controller.execute("click")
        - right_click=True → controller.execute("right_click")
        - cursor_delta_x/y → controller.move_cursor(dx, dy)

        Face confidence is passed as the confidence score so the controller
        can apply its threshold gate.
        """
        confidence = float(events.get("face_confidence", 1.0))

        # Cursor movement (always attempted when face detected)
        if events.get("face_detected", False):
            dx = float(events.get("cursor_delta_x", 0.0))
            dy = float(events.get("cursor_delta_y", 0.0))
            if dx != 0.0 or dy != 0.0:
                self._controller.move_cursor(dx, dy)

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
        Forward speech events to the controller.

        Actions taken:
        - command set → controller.execute(command)
        - dictation_text set → controller.type_dictation(text)
        - stop_control command → controller.trigger_emergency_stop()
        """
        confidence = float(events.get("speech_confidence", 0.0))
        cmd = events.get("command")
        dictation = events.get("dictation_text")

        if cmd:
            # "stop_control" triggers emergency stop immediately
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

