"""
output/undo.py - Undo / "go back" system for the Assistive Control System.

Architecture
------------
UndoManager maintains a bounded action-history stack (max 20 entries).  Every
undoable action is pushed onto the stack by the caller (router, hotkey handler,
main loop) immediately after it has been executed.  Calling undo() pops the
most recent entry and executes its reverse callable, if any.

Supported action types
-----------------------
  click        – reverse: press Escape (closes menus / tooltips)
  right_click  – reverse: press Escape
  open_app     – reverse: Alt+F4 (close focused window)
  open_browser – reverse: Alt+F4
  dictation    – reverse: Ctrl+A then Delete (clear the typed text)
  mode_switch  – reverse: set_manual_override(previous_mode) on DecisionEngine
  scroll       – reverse: scroll the opposite direction the same amount
  selection    – reverse: best-effort focus restore via pyautogui
  irreversible – reverse: None → "cannot undo: <description>"

Voice trigger phrases (checked by is_undo_phrase)
--------------------------------------------------
  "go back", "undo", "cancel that"
  Fuzzy-matched at ratio >= config.FUZZY_MATCH_THRESHOLD (default 0.70).
  The speech_loop in main.py should call:
      if undo_manager.is_undo_phrase(text):
          msg = undo_manager.undo(overlay_manager, csv_logger)

Hotkey
------
  Ctrl+Alt+Z is registered by HotkeyManager (output/hotkeys.py), which calls
  undo_manager.undo() directly.

Face gesture
------------
  A triple-blink gesture can optionally trigger undo.  Detection is done in
  face/face_tracker.py; on detection it should call undo_manager.undo().

Irreversible commands
---------------------
  ["delete", "send", "submit", "empty_trash", "format"]
  Push these with push_irreversible(); undo() returns "cannot undo: <desc>".

Thread safety
-------------
  A single threading.Lock guards all stack reads and writes.
"""

import logging
import time
import threading
from collections import deque
from difflib import SequenceMatcher
from typing import Any, Callable, Deque, Dict, Optional

import pyautogui

import config

logger = logging.getLogger("assistive.undo")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HISTORY: int = 20

# Phrases that trigger undo (fuzzy-matched)
_UNDO_PHRASES: tuple = ("go back", "undo", "cancel that")

# Commands that are irreversible
IRREVERSIBLE_COMMANDS: tuple = ("delete", "send", "submit", "empty_trash", "format")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _send_escape() -> None:
    """Press Escape once — closes most menus / context menus."""
    pyautogui.press("escape")


def _close_window() -> None:
    """Send Alt+F4 to close the currently focused window."""
    pyautogui.hotkey("alt", "f4")


def _undo_dictation(text: str) -> None:
    """Select-all then delete to erase dictated text.

    Uses Ctrl+A to select the field content, then Delete to clear it.
    This is best-effort: it works for most single-field scenarios.
    """
    pyautogui.hotkey("ctrl", "a")
    pyautogui.press("delete")


def _undo_scroll(direction: str, amount: int) -> None:
    """Scroll in the opposite direction by the same amount."""
    if direction == "up":
        pyautogui.scroll(-amount)
    elif direction == "down":
        pyautogui.scroll(amount)
    else:
        logger.warning("undo_scroll: unknown direction %r", direction)


def _fuzzy_ratio(a: str, b: str) -> float:
    """Return SequenceMatcher ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------------------------
# UndoManager
# ---------------------------------------------------------------------------

class UndoManager:
    """Manages a bounded undo history stack for the Assistive Control System.

    Parameters
    ----------
    decision_engine:
        Optional DecisionEngine instance used to reverse mode_switch actions.
    controller:
        Optional OutputController instance (reserved for future use).
    """

    def __init__(
        self,
        decision_engine=None,
        controller=None,
    ) -> None:
        self._decision_engine = decision_engine
        self._controller = controller
        self._stack: Deque[Dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal stack helper
    # ------------------------------------------------------------------

    def _push(
        self,
        action_type: str,
        target: Any,
        reverse: Optional[Callable],
        description: str,
    ) -> None:
        """Append an action entry to the stack (thread-safe)."""
        entry: Dict[str, Any] = {
            "type": action_type,
            "target": target,
            "reverse": reverse,
            "description": description,
            "timestamp": time.time(),
        }
        with self._lock:
            self._stack.append(entry)
        logger.debug("undo push: %s", description)

    # ------------------------------------------------------------------
    # Push methods
    # ------------------------------------------------------------------

    def push_click(self, x: int, y: int) -> None:
        """Record a left-click action at screen coordinates (x, y)."""
        self._push(
            action_type="click",
            target=(x, y),
            reverse=_send_escape,
            description=f"click at ({x}, {y})",
        )

    def push_right_click(self, x: int, y: int) -> None:
        """Record a right-click action at screen coordinates (x, y)."""
        self._push(
            action_type="right_click",
            target=(x, y),
            reverse=_send_escape,
            description=f"right-click at ({x}, {y})",
        )

    def push_dictation(self, text: str) -> None:
        """Record dictated text so it can be cleared on undo."""
        captured = text  # capture in closure

        def _reverse() -> None:
            _undo_dictation(captured)

        self._push(
            action_type="dictation",
            target=text,
            reverse=_reverse,
            description=f"dictation: \"{text[:40]}{'...' if len(text) > 40 else ''}\"",
        )

    def push_mode_switch(self, old_mode: Any, new_mode: Any) -> None:
        """Record a mode switch from old_mode → new_mode.

        On undo, calls decision_engine.set_manual_override(old_mode) if the
        decision engine was supplied at construction time.
        """
        engine = self._decision_engine
        captured_old = old_mode

        def _reverse() -> None:
            if engine is not None:
                engine.set_manual_override(captured_old)
                logger.info("undo mode_switch: restored %s", captured_old)
            else:
                logger.warning("undo mode_switch: no decision_engine available")

        old_name = old_mode.name if hasattr(old_mode, "name") else str(old_mode)
        new_name = new_mode.name if hasattr(new_mode, "name") else str(new_mode)
        self._push(
            action_type="mode_switch",
            target=old_mode,
            reverse=_reverse,
            description=f"mode switch {old_name} → {new_name}",
        )

    def push_scroll(self, direction: str, amount: int) -> None:
        """Record a scroll action.

        Parameters
        ----------
        direction: "up" or "down"
        amount:    scroll clicks / units
        """
        captured_dir = direction
        captured_amt = amount

        def _reverse() -> None:
            _undo_scroll(captured_dir, captured_amt)

        self._push(
            action_type="scroll",
            target=(direction, amount),
            reverse=_reverse,
            description=f"scroll {direction} {amount}",
        )

    def push_open_app(self, app_name: str) -> None:
        """Record an application-open action; undo closes it with Alt+F4."""
        self._push(
            action_type="open_app",
            target=app_name,
            reverse=_close_window,
            description=f"open app \"{app_name}\"",
        )

    def push_open_browser(self, url: str) -> None:
        """Record a browser-open action; undo closes it with Alt+F4."""
        self._push(
            action_type="open_browser",
            target=url,
            reverse=_close_window,
            description=f"open browser \"{url}\"",
        )

    def push_irreversible(self, description: str) -> None:
        """Record an irreversible action (delete / send / submit / …).

        undo() will return "cannot undo: <description>" for this entry.
        """
        self._push(
            action_type="irreversible",
            target=description,
            reverse=None,
            description=description,
        )

    # ------------------------------------------------------------------
    # Undo execution
    # ------------------------------------------------------------------

    def undo(
        self,
        overlay_manager=None,
        csv_logger=None,
    ) -> str:
        """Pop the most recent action and execute its reverse.

        Each successive call pops one more entry, stepping further back through
        history ("repeated go back steps further back").

        Parameters
        ----------
        overlay_manager:
            If provided, its update() method is called with a status message so
            the overlay can display what was (or could not be) undone.
        csv_logger:
            If provided, every undo attempt is logged:
            event_type="undo", data={action_command, action_success, message}.

        Returns
        -------
        str
            Human-readable result:
            • "Undid: <description>" on success
            • "cannot undo: <description>" for irreversible actions
            • "cannot undo: nothing to undo" when the stack is empty
        """
        with self._lock:
            if not self._stack:
                result = "cannot undo: nothing to undo"
                self._log(csv_logger, success=False, message=result)
                self._notify_overlay(overlay_manager, result)
                logger.info(result)
                return result

            entry = self._stack.pop()

        # Outside the lock — the reverse callable may itself push new entries
        action_type = entry["type"]
        description = entry["description"]
        reverse = entry["reverse"]

        if reverse is None:
            # Irreversible
            result = f"cannot undo: {description}"
            self._log(csv_logger, success=False, message=result)
            self._notify_overlay(overlay_manager, result)
            logger.info(result)
            return result

        # Execute the reverse action
        try:
            reverse()
            result = f"Undid: {description}"
            success = True
            logger.info(result)
        except Exception as exc:
            result = f"cannot undo: {description} ({exc})"
            success = False
            logger.exception("undo failed for %r: %s", description, exc)

        self._log(csv_logger, success=success, message=result)
        self._notify_overlay(overlay_manager, result)
        return result

    # ------------------------------------------------------------------
    # Voice phrase detection
    # ------------------------------------------------------------------

    def is_undo_phrase(self, text: str) -> bool:
        """Return True if *text* fuzzy-matches any undo trigger phrase.

        Trigger phrases: "go back", "undo", "cancel that".
        Threshold: config.FUZZY_MATCH_THRESHOLD (default 0.70).

        Usage in speech_loop (main.py):
            if undo_manager.is_undo_phrase(recognised_text):
                msg = undo_manager.undo(overlay_manager, csv_logger)
        """
        threshold: float = getattr(config, "FUZZY_MATCH_THRESHOLD", 0.70)
        normalised = text.strip().lower()
        for phrase in _UNDO_PHRASES:
            ratio = _fuzzy_ratio(normalised, phrase)
            if ratio >= threshold:
                logger.debug(
                    "is_undo_phrase: %r matched %r (ratio=%.2f)", text, phrase, ratio
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    def history_length(self) -> int:
        """Return the number of actions currently in the stack."""
        with self._lock:
            return len(self._stack)

    def peek(self) -> Optional[Dict[str, Any]]:
        """Return the most recent action entry without removing it, or None."""
        with self._lock:
            return self._stack[-1] if self._stack else None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log(
        self,
        csv_logger,
        success: bool,
        message: str,
    ) -> None:
        """Write an undo event to the CsvLogger, if one was supplied."""
        if csv_logger is None:
            return
        try:
            csv_logger.log(
                "undo",
                {
                    "action_command": "undo",
                    "action_success": success,
                    "message": message,
                },
            )
        except Exception as exc:
            logger.warning("undo: csv_logger.log failed: %s", exc)

    def _notify_overlay(self, overlay_manager, message: str) -> None:
        """Push a status update to the overlay, if one was supplied."""
        if overlay_manager is None:
            return
        try:
            overlay_manager.update({"last_cmd": message, "last_ok": not message.startswith("cannot")})
        except Exception as exc:
            logger.warning("undo: overlay_manager.update failed: %s", exc)
