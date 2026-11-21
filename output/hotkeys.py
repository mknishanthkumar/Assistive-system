"""
output/hotkeys.py - Global configurable mode-switch hotkeys for the Assistive Control System.

Registers pynput GlobalHotKeys that work system-wide with TOP PRIORITY,
even when face/speech modules are stuck. All hotkey bindings are configurable
via config.py.

Hotkeys
-------
  Ctrl+Alt+1        → Force FACE mode
  Ctrl+Alt+2        → Force SPEECH mode
  Ctrl+Alt+3        → Force MANUAL (BOTH) mode
  Ctrl+Alt+4        → Force AUTO mode (clears override → returns to auto-switching)
  Ctrl+Alt+` (96)   → Cycle next mode (FACE→SPEECH→BOTH→FACE, skips PAUSED)
  Ctrl+Alt+Space    → Pause/Resume panic key (toggles PAUSED; restores previous mode)
  Ctrl+Alt+Z        → Undo last action (delegates to UndoManager)

Public API
----------
  HotkeyManager(decision_engine, csv_logger=None, overlay_manager=None, undo_manager=None)
  .start()   – registers pynput GlobalHotKeys listener
  .stop()    – stops listener, joins thread
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import config
from context.decision_engine import DecisionEngine, Mode

logger = logging.getLogger("assistive.hotkeys")

# ---------------------------------------------------------------------------
# Sound helper
# ---------------------------------------------------------------------------

def _play_sound_async(mode: Mode) -> None:
    """Play a mode-switch sound asynchronously (never blocks the hotkey thread)."""
    def _play() -> None:
        # Try output.overlay.play_mode_sound first (richer tones)
        try:
            from output.overlay import play_mode_sound  # type: ignore
            play_mode_sound(mode)
            return
        except Exception:
            pass
        # Fallback: winsound.Beep
        try:
            import winsound
            freq_map = {
                Mode.FACE:   800,
                Mode.SPEECH: 600,
                Mode.BOTH:   1000,
                Mode.PAUSED: 400,
            }
            freq = freq_map.get(mode, 750)
            winsound.Beep(freq, 150)
        except Exception:
            pass  # Silent fallback — never raise in hotkey context

    t = threading.Thread(target=_play, daemon=True, name="hotkey-sound")
    t.start()


# ---------------------------------------------------------------------------
# Cycle order (skips PAUSED)
# ---------------------------------------------------------------------------

_CYCLE_ORDER = [Mode.FACE, Mode.SPEECH, Mode.BOTH]


def _next_cycle_mode(current: Mode) -> Mode:
    """Return the next mode in the cycle FACE→SPEECH→BOTH→FACE (skips PAUSED)."""
    try:
        idx = _CYCLE_ORDER.index(current)
    except ValueError:
        # current is PAUSED or unknown → start from FACE
        idx = -1
    return _CYCLE_ORDER[(idx + 1) % len(_CYCLE_ORDER)]


# ---------------------------------------------------------------------------
# HotkeyManager
# ---------------------------------------------------------------------------

class HotkeyManager:
    """
    Registers global hotkeys via pynput and routes them to DecisionEngine.

    Parameters
    ----------
    decision_engine : DecisionEngine
        The shared engine whose mode is switched by the hotkeys.
    csv_logger : CsvLogger | None
        Optional CSV logger; each hotkey activation is recorded.
    overlay_manager : OverlayManager | None
        Optional overlay; updated non-blocking after each hotkey action.
    undo_manager : UndoManager | None
        Optional undo stack; mode switches are pushed for later undo.
    """

    def __init__(
        self,
        decision_engine: DecisionEngine,
        csv_logger=None,
        overlay_manager=None,
        undo_manager=None,
    ) -> None:
        self._engine   = decision_engine
        self._csv      = csv_logger
        self._overlay  = overlay_manager
        self._undo     = undo_manager
        self._listener = None          # pynput GlobalHotKeys instance
        self._lock     = threading.Lock()

        # Track "previous non-paused mode" for pause/resume toggle
        self._pre_pause_mode: Mode = Mode.BOTH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register all hotkeys and start the pynput listener thread."""
        try:
            from pynput import keyboard  # type: ignore
        except ImportError:
            logger.error(
                "pynput is not installed; global hotkeys will NOT work. "
                "Install it with: pip install pynput"
            )
            return

        hotkey_map = {
            config.HOTKEY_FORCE_FACE:    self._on_force_face,
            config.HOTKEY_FORCE_SPEECH:  self._on_force_speech,
            config.HOTKEY_FORCE_MANUAL:  self._on_force_manual,
            config.HOTKEY_FORCE_AUTO:    self._on_force_auto,
            config.HOTKEY_CYCLE_MODE:    self._on_cycle_mode,
            config.HOTKEY_PAUSE_RESUME:  self._on_pause_resume,
            config.HOTKEY_UNDO:          self._on_undo,
        }

        self._listener = keyboard.GlobalHotKeys(hotkey_map)
        self._listener.daemon = True
        self._listener.start()
        logger.info("HotkeyManager started — %d hotkeys registered.", len(hotkey_map))

    def stop(self) -> None:
        """Stop the pynput listener (waits for thread to exit, timeout=2 s)."""
        if self._listener is not None:
            try:
                self._listener.stop()
                self._listener.join(timeout=2.0)
            except Exception as exc:
                logger.debug("HotkeyManager stop error: %s", exc)
            self._listener = None
            logger.info("HotkeyManager stopped.")

    # ------------------------------------------------------------------
    # Hotkey handlers
    # ------------------------------------------------------------------

    def _on_force_face(self) -> None:
        self._apply_override(
            new_mode=Mode.FACE,
            action_command="force_face",
            message="Hotkey: forced FACE mode (manual override)",
        )

    def _on_force_speech(self) -> None:
        self._apply_override(
            new_mode=Mode.SPEECH,
            action_command="force_speech",
            message="Hotkey: forced SPEECH mode (manual override)",
        )

    def _on_force_manual(self) -> None:
        self._apply_override(
            new_mode=Mode.BOTH,
            action_command="force_manual",
            message="Hotkey: forced MANUAL (BOTH) mode (manual override)",
        )

    def _on_force_auto(self) -> None:
        """Clear override → engine returns to automatic rule-based switching."""
        with self._lock:
            old_mode = self._engine.current_mode
            self._engine.clear_manual_override()
            new_mode = self._engine.current_mode   # may still be old until evaluate()
            logger.info("Hotkey: cleared manual override → AUTO mode")

        self._post_action(
            old_mode=old_mode,
            new_mode=new_mode,
            action_command="force_auto",
            message="Hotkey: cleared manual override → AUTO mode",
        )

    def _on_cycle_mode(self) -> None:
        """Cycle through FACE→SPEECH→BOTH→FACE (skips PAUSED)."""
        with self._lock:
            old_mode = self._engine.current_mode
            new_mode = _next_cycle_mode(old_mode)
            self._engine.set_manual_override(new_mode)
            logger.info("Hotkey: cycled mode %s → %s", old_mode.value, new_mode.value)

        self._post_action(
            old_mode=old_mode,
            new_mode=new_mode,
            action_command="cycle_mode",
            message=f"Hotkey: cycled mode {old_mode.value} → {new_mode.value}",
        )

    def _on_pause_resume(self) -> None:
        """Toggle PAUSED. If currently PAUSED, restore previous mode."""
        with self._lock:
            old_mode = self._engine.current_mode
            if old_mode == Mode.PAUSED:
                # Resume: restore pre-pause mode
                new_mode = self._pre_pause_mode
                self._engine.set_manual_override(new_mode)
                action = "resume"
                msg = f"Hotkey: resumed → {new_mode.value}"
            else:
                # Pause: remember current mode and set PAUSED
                self._pre_pause_mode = old_mode
                new_mode = Mode.PAUSED
                self._engine.set_manual_override(Mode.PAUSED)
                action = "pause"
                msg = f"Hotkey: paused (was {old_mode.value})"
            logger.info(msg)

        self._post_action(
            old_mode=old_mode,
            new_mode=new_mode,
            action_command=action,
            message=msg,
        )

    def _on_undo(self) -> None:
        """Trigger undo via UndoManager (Ctrl+Alt+Z)."""
        if self._undo is None:
            logger.info("Hotkey: undo triggered but no UndoManager attached.")
            return
        try:
            result = self._undo.undo(
                overlay_manager=self._overlay,
                csv_logger=self._csv,
            )
            logger.info("Hotkey: undo → %s", result)
        except Exception as exc:
            logger.warning("Hotkey: undo failed: %s", exc)

    # ------------------------------------------------------------------
    # Shared override helper
    # ------------------------------------------------------------------

    def _apply_override(
        self,
        new_mode: Mode,
        action_command: str,
        message: str,
    ) -> None:
        """Set a manual override and trigger all side-effects."""
        with self._lock:
            old_mode = self._engine.current_mode
            self._engine.set_manual_override(new_mode)
            logger.info(message)

        self._post_action(
            old_mode=old_mode,
            new_mode=new_mode,
            action_command=action_command,
            message=message,
        )

    def _post_action(
        self,
        old_mode: Mode,
        new_mode: Mode,
        action_command: str,
        message: str,
    ) -> None:
        """
        Shared side-effects after any hotkey action:
          1. CSV log
          2. Overlay update (non-blocking)
          3. Undo stack push (mode switch only)
          4. Async sound
        """
        # 1. CSV log
        if self._csv is not None:
            try:
                self._csv.log(
                    "hotkey",
                    {
                        "action_command": action_command,
                        "action_success": True,
                        "message": message,
                    },
                )
            except Exception as exc:
                logger.debug("CSV log failed in hotkey handler: %s", exc)

        # 2. Overlay update (non-blocking)
        if self._overlay is not None:
            try:
                state = {
                    "mode":    new_mode.value,
                    "paused":  new_mode == Mode.PAUSED,
                    "last_cmd": action_command,
                    "last_ok": True,
                }
                self._overlay.update(state)
            except Exception as exc:
                logger.debug("Overlay update failed in hotkey handler: %s", exc)

        # 3. Undo stack push (only for real mode switches, not for undo itself)
        if (
            self._undo is not None
            and old_mode != new_mode
            and action_command not in ("undo", "force_auto")
        ):
            try:
                self._undo.push_mode_switch(old_mode, new_mode)
            except Exception as exc:
                logger.debug("Undo push failed in hotkey handler: %s", exc)

        # 4. Play mode-switch sound asynchronously
        _play_sound_async(new_mode)
