"""
output/controller.py - Execute real Windows actions from speech and face events.

Architecture
------------
OutputController receives command strings (from SpeechEngine / Router) and
face cursor/click events (from FaceTracker / Router), and maps them to mouse,
keyboard, browser, and system actions via injectable backend functions.

Backends are injected through the constructor so that tests can mock them
without touching real hardware.

Safety
------
- PyAutoGUI fail-safe stays enabled (move mouse to corner to abort).
- Confidence threshold: commands below SPEECH_CONFIDENCE_THRESHOLD are rejected.
- Per-command cooldown prevents duplicate actions from a single utterance.
- Emergency stop flag disables all execution instantly.
- DRY_RUN mode: log actions without executing (config.DRY_RUN or constructor arg).
- Cursor coordinates are clamped to the screen (never outside the display).
- No destructive actions (shutdown, restart, delete, auto-submit).

Stage 4 additions
-----------------
- move_cursor(dx, dy) : applies face cursor deltas to the real mouse position.
- execute() accepts an optional source="face"|"speech" parameter for logging.
- DRY_RUN default read from config.DRY_RUN.
"""

import logging
import time
import threading
import webbrowser
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import pyautogui

import config

logger = logging.getLogger("assistive.output_controller")

# ---------------------------------------------------------------------------
# Action result
# ---------------------------------------------------------------------------
@dataclass
class ActionResult:
    """Structured result from a command execution attempt."""
    success: bool
    command: str
    message: str


# ---------------------------------------------------------------------------
# Default backend functions (real Windows actions)
# ---------------------------------------------------------------------------

def _default_mouse_click() -> None:
    """Left-click at the current cursor position."""
    pyautogui.click()


def _default_mouse_double_click() -> None:
    """Double-click at the current cursor position."""
    pyautogui.doubleClick()


def _default_mouse_right_click() -> None:
    """Right-click at the current cursor position."""
    pyautogui.rightClick()


def _default_scroll_up() -> None:
    """Scroll up by SCROLL_AMOUNT notches."""
    pyautogui.scroll(config.SCROLL_AMOUNT)


def _default_scroll_down() -> None:
    """Scroll down by SCROLL_AMOUNT notches."""
    pyautogui.scroll(-config.SCROLL_AMOUNT)


def _default_open_browser() -> None:
    """Open the system default web browser."""
    webbrowser.open("https://www.google.com")


def _default_close_window() -> None:
    """Close the active window via Alt+F4."""
    pyautogui.hotkey("alt", "F4")


def _default_switch_window() -> None:
    """Switch window via Alt+Tab."""
    pyautogui.hotkey("alt", "tab")


def _default_minimize_window() -> None:
    """Minimize active window via Win+Down."""
    pyautogui.hotkey("win", "down")


def _default_volume_up() -> None:
    """Increase system volume via media key."""
    pyautogui.press("volumeup")


def _default_volume_down() -> None:
    """Decrease system volume via media key."""
    pyautogui.press("volumedown")


def _default_mute() -> None:
    """Toggle system mute via media key."""
    pyautogui.press("volumemute")


def _default_press_enter() -> None:
    """Send the Enter key."""
    pyautogui.press("enter")


def _default_press_escape() -> None:
    """Send the Escape key."""
    pyautogui.press("escape")


def _default_type_text(text: str) -> None:
    """Type dictation text into the focused field."""
    pyautogui.typewrite(text, interval=config.DICTATION_TYPE_DELAY_S)


# ---------------------------------------------------------------------------
# OutputController
# ---------------------------------------------------------------------------
class OutputController:
    """
    Maps speech command IDs to real Windows actions.

    Parameters
    ----------
    dry_run : bool
        If True, log actions but do not execute them.
    mouse_click, mouse_double_click, mouse_right_click, scroll_up, scroll_down,
    open_browser, close_window, switch_window, minimize_window,
    volume_up, volume_down, mute, press_enter, press_escape, type_text :
        Injectable backend callables for testing.
    """

    def __init__(
        self,
        dry_run: bool = config.DRY_RUN,
        *,
        mouse_click: Optional[Callable] = None,
        mouse_double_click: Optional[Callable] = None,
        mouse_right_click: Optional[Callable] = None,
        scroll_up: Optional[Callable] = None,
        scroll_down: Optional[Callable] = None,
        open_browser: Optional[Callable] = None,
        close_window: Optional[Callable] = None,
        switch_window: Optional[Callable] = None,
        minimize_window: Optional[Callable] = None,
        volume_up: Optional[Callable] = None,
        volume_down: Optional[Callable] = None,
        mute: Optional[Callable] = None,
        press_enter: Optional[Callable] = None,
        press_escape: Optional[Callable] = None,
        type_text: Optional[Callable] = None,
    ) -> None:
        self._dry_run = dry_run
        self._emergency_stop = False
        self._lock = threading.Lock()

        # Per-command cooldown tracking: command_id -> last_execution_time
        self._last_exec_time: Dict[str, float] = {}

        # Face-click rate-limit (separate from speech command cooldown)
        self._last_face_click_time: float = 0.0

        # Build the command -> backend mapping
        self._actions: Dict[str, Callable] = {
            "click":            mouse_click or _default_mouse_click,
            "double_click":     mouse_double_click or _default_mouse_double_click,
            "right_click":      mouse_right_click or _default_mouse_right_click,
            "scroll_up":        scroll_up or _default_scroll_up,
            "scroll_down":      scroll_down or _default_scroll_down,
            "open_browser":     open_browser or _default_open_browser,
            "close_window":     close_window or _default_close_window,
            "switch_window":    switch_window or _default_switch_window,
            "minimize_window":  minimize_window or _default_minimize_window,
            "volume_up":        volume_up or _default_volume_up,
            "volume_down":      volume_down or _default_volume_down,
            "mute":             mute or _default_mute,
            "press_enter":      press_enter or _default_press_enter,
            "press_escape":     press_escape or _default_press_escape,
        }

        # Dictation typing backend
        self._type_text = type_text or _default_type_text

        logger.info(
            "OutputController initialised (dry_run=%s, actions=%d)",
            self._dry_run, len(self._actions),
        )

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------

    def trigger_emergency_stop(self) -> None:
        """Immediately disable all action execution."""
        with self._lock:
            self._emergency_stop = True
        logger.warning("EMERGENCY STOP activated — all actions disabled.")

    def is_stopped(self) -> bool:
        """Return True if emergency stop is active."""
        with self._lock:
            return self._emergency_stop

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        confidence: float = 1.0,
        source: str = "speech",
    ) -> ActionResult:
        """
        Execute a single command action.

        Parameters
        ----------
        command    : normalised command ID from SpeechEngine (e.g. "click")
        confidence : Whisper speech confidence [0.0, 1.0]
        source     : "speech" or "face" (used for logging only)

        Returns
        -------
        ActionResult with success flag, command ID, and descriptive message.
        """
        # ── Emergency stop ────────────────────────────────────────────
        if self.is_stopped():
            msg = f"BLOCKED by emergency stop: {command}"
            logger.warning(msg)
            return ActionResult(success=False, command=command, message=msg)

        # ── Skip non-actionable commands (dictation control, mode switches) ──
        if command in ("start_dictation", "stop_dictation", "stop_control"):
            msg = f"Control command '{command}' handled by SpeechEngine, not OutputController."
            logger.debug(msg)
            return ActionResult(success=True, command=command, message=msg)

        # ── Confidence threshold ──────────────────────────────────────
        threshold = config.SPEECH_CONFIDENCE_THRESHOLD
        if confidence < threshold:
            msg = (
                f"Low confidence ({confidence:.2f} < {threshold:.2f}): "
                f"'{command}' NOT executed."
            )
            logger.info(msg)
            return ActionResult(success=False, command=command, message=msg)

        # ── Lookup action ─────────────────────────────────────────────
        action_fn = self._actions.get(command)
        if action_fn is None:
            msg = f"Unsupported command: '{command}'"
            logger.warning(msg)
            return ActionResult(success=False, command=command, message=msg)

        # ── Per-command cooldown ──────────────────────────────────────
        now = time.monotonic()
        cooldown = config.SPEECH_COMMAND_COOLDOWN_S
        with self._lock:
            last_time = self._last_exec_time.get(command, 0.0)
            if now - last_time < cooldown:
                msg = (
                    f"Cooldown active ({now - last_time:.2f}s < {cooldown}s): "
                    f"'{command}' NOT executed."
                )
                logger.debug(msg)
                return ActionResult(success=False, command=command, message=msg)

        # ── Verify cursor is readable (for click commands) ────────────
        if command in ("click", "double_click", "right_click"):
            try:
                pos = pyautogui.position()
                if pos is None:
                    msg = f"Cannot read cursor position: '{command}' NOT executed."
                    logger.warning(msg)
                    return ActionResult(success=False, command=command, message=msg)
            except Exception as exc:
                msg = f"Cursor position check failed ({exc}): '{command}' NOT executed."
                logger.warning(msg)
                return ActionResult(success=False, command=command, message=msg)

        # ── Execute (or dry-run) ──────────────────────────────────────
        if self._dry_run:
            msg = f"[DRY-RUN] Would execute: {command}"
            logger.info(msg)
            with self._lock:
                self._last_exec_time[command] = now
            return ActionResult(success=True, command=command, message=msg)

        try:
            action_fn()
            with self._lock:
                self._last_exec_time[command] = now
            msg = _ACTION_MESSAGES.get(command, f"Executed: {command}")
            logger.info(msg)
            return ActionResult(success=True, command=command, message=msg)
        except Exception as exc:
            msg = f"Action failed for '{command}': {exc}"
            logger.error(msg)
            return ActionResult(success=False, command=command, message=msg)

    # ------------------------------------------------------------------
    # Cursor movement (from face tracker)
    # ------------------------------------------------------------------

    def move_cursor(self, dx: float, dy: float) -> None:
        """
        Move the mouse cursor by (dx, dy) pixels relative to current position.

        Safety
        ------
        - Blocked when emergency stop is active.
        - Deltas are clamped to config.CONTROLLER_MAX_CURSOR_DELTA.
        - Resulting coordinates are clamped to stay within the screen
          (with a CURSOR_SCREEN_MARGIN pixel border).
        - In DRY_RUN mode, the move is logged but not executed.

        Parameters
        ----------
        dx : horizontal delta in pixels (positive = right)
        dy : vertical delta in pixels (positive = down)
        """
        if self.is_stopped():
            return

        # Clamp deltas
        max_d = config.CONTROLLER_MAX_CURSOR_DELTA
        dx = max(-max_d, min(max_d, dx))
        dy = max(-max_d, min(max_d, dy))

        if dx == 0.0 and dy == 0.0:
            return

        if self._dry_run:
            logger.debug("[DRY-RUN] move_cursor(dx=%.1f, dy=%.1f)", dx, dy)
            return

        try:
            current_x, current_y = pyautogui.position()
            screen_w, screen_h = pyautogui.size()
            margin = config.CURSOR_SCREEN_MARGIN
            max_x = screen_w - 1 if margin == 0 else screen_w - margin
            max_y = screen_h - 1 if margin == 0 else screen_h - margin

            new_x = int(round(max(margin, min(max_x, current_x + dx))))
            new_y = int(round(max(margin, min(max_y, current_y + dy))))

            # Temporarily set PAUSE to 0.0 to prevent 50ms pause delay on cursor movements
            old_pause = getattr(pyautogui, "PAUSE", 0.0)
            try:
                pyautogui.PAUSE = 0.0
                pyautogui.moveTo(new_x, new_y)
            finally:
                pyautogui.PAUSE = old_pause

            logger.debug(
                "Cursor move: delta=(%.1f, %.1f) | current=(%d, %d) -> mapped=(%d, %d) [screen: %dx%d]",
                dx, dy, current_x, current_y, new_x, new_y, screen_w, screen_h,
            )
        except Exception as exc:
            if exc.__class__.__name__ == "FailSafeException":
                logger.warning("PyAutoGUI fail-safe triggered in move_cursor.")
                self.trigger_emergency_stop()
            else:
                logger.debug("move_cursor failed: %s", exc)

    # ------------------------------------------------------------------
    # Dictation typing
    # ------------------------------------------------------------------

    def type_dictation(self, text: str) -> ActionResult:
        """
        Type dictation text into the currently focused field.

        Parameters
        ----------
        text : recognised dictation text from SpeechEngine
        """
        if self.is_stopped():
            msg = f"BLOCKED by emergency stop: dictation '{text[:30]}...'"
            logger.warning(msg)
            return ActionResult(success=False, command="dictation", message=msg)

        if not text or not text.strip():
            return ActionResult(
                success=False, command="dictation",
                message="Empty dictation text — nothing typed.",
            )

        if self._dry_run:
            msg = f"[DRY-RUN] Would type: '{text}'"
            logger.info(msg)
            return ActionResult(success=True, command="dictation", message=msg)

        try:
            # Add trailing space so words don't merge
            self._type_text(text.strip() + " ")
            msg = f"Typed dictation: '{text.strip()}'"
            logger.info(msg)
            return ActionResult(success=True, command="dictation", message=msg)
        except Exception as exc:
            msg = f"Dictation typing failed: {exc}"
            logger.error(msg)
            return ActionResult(success=False, command="dictation", message=msg)


# ---------------------------------------------------------------------------
# Human-friendly action messages
# ---------------------------------------------------------------------------
_ACTION_MESSAGES: Dict[str, str] = {
    "click":            "Left click executed.",
    "double_click":     "Double click executed.",
    "right_click":      "Right click executed.",
    "scroll_up":        f"Scrolled up ({config.SCROLL_AMOUNT} notches).",
    "scroll_down":      f"Scrolled down ({config.SCROLL_AMOUNT} notches).",
    "open_browser":     "Default browser opened.",
    "close_window":     "Active window closed (Alt+F4).",
    "switch_window":    "Switched window (Alt+Tab).",
    "minimize_window":  "Window minimized (Win+Down).",
    "volume_up":        "Volume increased.",
    "volume_down":      "Volume decreased.",
    "mute":             "Mute toggled.",
    "press_enter":      "Enter key pressed.",
    "press_escape":     "Escape key pressed.",
}
