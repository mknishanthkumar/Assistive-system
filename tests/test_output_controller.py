"""
tests/test_output_controller.py - Unit tests for OutputController (Stage 3).

All tests use mocked backends — no real mouse clicks, key presses, or browser
launches occur.  Run with:

    python -m pytest tests/test_output_controller.py -v
"""

import sys
import math
import pathlib
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from output.controller import OutputController, ActionResult


class _MockController:
    """Helper that creates an OutputController with all backends mocked."""

    @staticmethod
    def create(dry_run: bool = False) -> tuple:
        """Return (controller, mocks_dict)."""
        mocks = {
            "mouse_click": MagicMock(),
            "mouse_double_click": MagicMock(),
            "mouse_right_click": MagicMock(),
            "scroll_up": MagicMock(),
            "scroll_down": MagicMock(),
            "open_browser": MagicMock(),
            "close_window": MagicMock(),
            "switch_window": MagicMock(),
            "minimize_window": MagicMock(),
            "volume_up": MagicMock(),
            "volume_down": MagicMock(),
            "mute": MagicMock(),
            "press_enter": MagicMock(),
            "press_escape": MagicMock(),
            "type_text": MagicMock(),
        }
        ctrl = OutputController(dry_run=dry_run, **mocks)
        return ctrl, mocks


# ---------------------------------------------------------------------------
# Mouse actions
# ---------------------------------------------------------------------------
class TestMouseActions(unittest.TestCase):
    """Verify click, double click, and right click at current cursor."""

    @patch("output.controller.pyautogui")
    def test_click(self, mock_pyautogui):
        mock_pyautogui.position.return_value = (100, 200)
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("click")
        self.assertTrue(result.success)
        mocks["mouse_click"].assert_called_once()
        self.assertIn("Left click", result.message)

    @patch("output.controller.pyautogui")
    def test_double_click(self, mock_pyautogui):
        mock_pyautogui.position.return_value = (100, 200)
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("double_click")
        self.assertTrue(result.success)
        mocks["mouse_double_click"].assert_called_once()
        self.assertIn("Double click", result.message)

    @patch("output.controller.pyautogui")
    def test_right_click(self, mock_pyautogui):
        mock_pyautogui.position.return_value = (100, 200)
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("right_click")
        self.assertTrue(result.success)
        mocks["mouse_right_click"].assert_called_once()
        self.assertIn("Right click", result.message)


# ---------------------------------------------------------------------------
# Scroll actions
# ---------------------------------------------------------------------------
class TestScrollActions(unittest.TestCase):
    """Verify scroll up/down calls the correct backend."""

    def test_scroll_up(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("scroll_up")
        self.assertTrue(result.success)
        mocks["scroll_up"].assert_called_once()
        self.assertIn("Scrolled up", result.message)

    def test_scroll_down(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("scroll_down")
        self.assertTrue(result.success)
        mocks["scroll_down"].assert_called_once()
        self.assertIn("Scrolled down", result.message)


# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------
class TestBrowser(unittest.TestCase):
    """Verify open_browser calls the browser backend."""

    def test_open_browser(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("open_browser")
        self.assertTrue(result.success)
        mocks["open_browser"].assert_called_once()
        self.assertIn("browser", result.message.lower())


# ---------------------------------------------------------------------------
# Window management hotkeys
# ---------------------------------------------------------------------------
class TestWindowHotkeys(unittest.TestCase):
    """Verify close/switch/minimize window commands."""

    def test_close_window(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("close_window")
        self.assertTrue(result.success)
        mocks["close_window"].assert_called_once()
        self.assertIn("close", result.message.lower())

    def test_switch_window(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("switch_window")
        self.assertTrue(result.success)
        mocks["switch_window"].assert_called_once()
        self.assertIn("switch", result.message.lower())

    def test_minimize_window(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("minimize_window")
        self.assertTrue(result.success)
        mocks["minimize_window"].assert_called_once()
        self.assertIn("minimize", result.message.lower())


# ---------------------------------------------------------------------------
# Keyboard keys
# ---------------------------------------------------------------------------
class TestKeyboardKeys(unittest.TestCase):
    """Verify press_enter and press_escape."""

    def test_press_enter(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("press_enter")
        self.assertTrue(result.success)
        mocks["press_enter"].assert_called_once()
        self.assertIn("Enter", result.message)

    def test_press_escape(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("press_escape")
        self.assertTrue(result.success)
        mocks["press_escape"].assert_called_once()
        self.assertIn("Escape", result.message)


# ---------------------------------------------------------------------------
# Volume keys
# ---------------------------------------------------------------------------
class TestVolumeKeys(unittest.TestCase):
    """Verify volume up, down, and mute toggle."""

    def test_volume_up(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("volume_up")
        self.assertTrue(result.success)
        mocks["volume_up"].assert_called_once()
        self.assertIn("Volume increased", result.message)

    def test_volume_down(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("volume_down")
        self.assertTrue(result.success)
        mocks["volume_down"].assert_called_once()
        self.assertIn("Volume decreased", result.message)

    def test_mute(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("mute")
        self.assertTrue(result.success)
        mocks["mute"].assert_called_once()
        self.assertIn("Mute", result.message)


# ---------------------------------------------------------------------------
# Unknown / unsupported command
# ---------------------------------------------------------------------------
class TestUnknownCommand(unittest.TestCase):
    """Verify unknown commands are rejected cleanly."""

    def test_unknown_command(self):
        ctrl, _ = _MockController.create()
        result = ctrl.execute("fly_to_moon")
        self.assertFalse(result.success)
        self.assertIn("Unsupported", result.message)

    def test_empty_command(self):
        ctrl, _ = _MockController.create()
        result = ctrl.execute("")
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# Low confidence rejection
# ---------------------------------------------------------------------------
class TestConfidenceThreshold(unittest.TestCase):
    """Verify commands below confidence threshold are rejected."""

    def test_low_confidence_rejected(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.execute("click", confidence=0.10)
        self.assertFalse(result.success)
        self.assertIn("Low confidence", result.message)
        mocks["mouse_click"].assert_not_called()

    def test_high_confidence_accepted(self):
        ctrl, mocks = _MockController.create()
        with patch("output.controller.pyautogui") as mock_pyautogui:
            mock_pyautogui.position.return_value = (100, 200)
            result = ctrl.execute("click", confidence=0.95)
        self.assertTrue(result.success)
        mocks["mouse_click"].assert_called_once()

    def test_exact_threshold_accepted(self):
        ctrl, mocks = _MockController.create()
        threshold = config.SPEECH_CONFIDENCE_THRESHOLD
        with patch("output.controller.pyautogui") as mock_pyautogui:
            mock_pyautogui.position.return_value = (100, 200)
            result = ctrl.execute("click", confidence=threshold)
        self.assertTrue(result.success)
        mocks["mouse_click"].assert_called_once()


# ---------------------------------------------------------------------------
# Dictation separate from commands
# ---------------------------------------------------------------------------
class TestDictation(unittest.TestCase):
    """Verify dictation typing is separate from command execution."""

    def test_dictation_types_text(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.type_dictation("hello world")
        self.assertTrue(result.success)
        mocks["type_text"].assert_called_once_with("hello world ")

    def test_dictation_empty_rejected(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.type_dictation("")
        self.assertFalse(result.success)
        mocks["type_text"].assert_not_called()

    def test_dictation_whitespace_rejected(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.type_dictation("   ")
        self.assertFalse(result.success)
        mocks["type_text"].assert_not_called()

    def test_dictation_strips_text(self):
        ctrl, mocks = _MockController.create()
        result = ctrl.type_dictation("  hello  ")
        self.assertTrue(result.success)
        mocks["type_text"].assert_called_once_with("hello ")

    def test_dictation_does_not_execute_command(self):
        """Typing 'click' as dictation must NOT trigger a mouse click."""
        ctrl, mocks = _MockController.create()
        result = ctrl.type_dictation("click")
        self.assertTrue(result.success)
        self.assertEqual(result.command, "dictation")
        mocks["mouse_click"].assert_not_called()
        mocks["type_text"].assert_called_once_with("click ")


# ---------------------------------------------------------------------------
# Duplicate events give one action (cooldown)
# ---------------------------------------------------------------------------
class TestCooldown(unittest.TestCase):
    """Verify per-command cooldown prevents duplicate rapid actions."""

    def test_duplicate_within_cooldown_blocked(self):
        ctrl, mocks = _MockController.create()
        r1 = ctrl.execute("volume_up")
        self.assertTrue(r1.success)
        # Second call immediately — should be blocked by cooldown
        r2 = ctrl.execute("volume_up")
        self.assertFalse(r2.success)
        self.assertIn("Cooldown", r2.message)
        mocks["volume_up"].assert_called_once()

    def test_different_commands_not_blocked(self):
        ctrl, mocks = _MockController.create()
        r1 = ctrl.execute("volume_up")
        self.assertTrue(r1.success)
        # Different command should not be affected
        r2 = ctrl.execute("volume_down")
        self.assertTrue(r2.success)
        mocks["volume_up"].assert_called_once()
        mocks["volume_down"].assert_called_once()


# ---------------------------------------------------------------------------
# Emergency stop blocks execution
# ---------------------------------------------------------------------------
class TestEmergencyStop(unittest.TestCase):
    """Verify emergency stop disables all action execution."""

    def test_stop_blocks_commands(self):
        ctrl, mocks = _MockController.create()
        ctrl.trigger_emergency_stop()
        result = ctrl.execute("click")
        self.assertFalse(result.success)
        self.assertIn("emergency stop", result.message.lower())
        mocks["mouse_click"].assert_not_called()

    def test_stop_blocks_dictation(self):
        ctrl, mocks = _MockController.create()
        ctrl.trigger_emergency_stop()
        result = ctrl.type_dictation("hello")
        self.assertFalse(result.success)
        self.assertIn("emergency stop", result.message.lower())
        mocks["type_text"].assert_not_called()

    def test_is_stopped_flag(self):
        ctrl, _ = _MockController.create()
        self.assertFalse(ctrl.is_stopped())
        ctrl.trigger_emergency_stop()
        self.assertTrue(ctrl.is_stopped())


# ---------------------------------------------------------------------------
# Dry run mode
# ---------------------------------------------------------------------------
class TestDryRun(unittest.TestCase):
    """Verify dry-run mode logs but does not execute."""

    def test_dry_run_does_not_call_backend(self):
        ctrl, mocks = _MockController.create(dry_run=True)
        result = ctrl.execute("volume_up")
        self.assertTrue(result.success)
        self.assertIn("DRY-RUN", result.message)
        mocks["volume_up"].assert_not_called()

    def test_dry_run_dictation(self):
        ctrl, mocks = _MockController.create(dry_run=True)
        result = ctrl.type_dictation("hello world")
        self.assertTrue(result.success)
        self.assertIn("DRY-RUN", result.message)
        mocks["type_text"].assert_not_called()


# ---------------------------------------------------------------------------
# Control commands passthrough
# ---------------------------------------------------------------------------
class TestControlCommands(unittest.TestCase):
    """Verify start_dictation/stop_dictation/stop_control are handled."""

    def test_start_dictation_passthrough(self):
        ctrl, _ = _MockController.create()
        result = ctrl.execute("start_dictation")
        self.assertTrue(result.success)
        self.assertIn("SpeechEngine", result.message)

    def test_stop_dictation_passthrough(self):
        ctrl, _ = _MockController.create()
        result = ctrl.execute("stop_dictation")
        self.assertTrue(result.success)

    def test_stop_control_passthrough(self):
        ctrl, _ = _MockController.create()
        result = ctrl.execute("stop_control")
        self.assertTrue(result.success)


# ---------------------------------------------------------------------------
# Action result structure
# ---------------------------------------------------------------------------
class TestActionResult(unittest.TestCase):
    """Verify ActionResult dataclass structure."""

    def test_result_fields(self):
        result = ActionResult(success=True, command="click", message="ok")
        self.assertTrue(result.success)
        self.assertEqual(result.command, "click")
        self.assertEqual(result.message, "ok")

    def test_result_false(self):
        result = ActionResult(success=False, command="x", message="fail")
        self.assertFalse(result.success)


# ---------------------------------------------------------------------------
# Backend exception handling
# ---------------------------------------------------------------------------
class TestBackendExceptions(unittest.TestCase):
    """Verify that backend exceptions are caught and reported."""

    def test_click_backend_raises(self):
        failing_click = MagicMock(side_effect=RuntimeError("hardware fault"))
        ctrl = OutputController(
            mouse_click=failing_click,
            mouse_double_click=MagicMock(),
            mouse_right_click=MagicMock(),
            scroll_up=MagicMock(),
            scroll_down=MagicMock(),
            open_browser=MagicMock(),
            close_window=MagicMock(),
            switch_window=MagicMock(),
            minimize_window=MagicMock(),
            volume_up=MagicMock(),
            volume_down=MagicMock(),
            mute=MagicMock(),
            press_enter=MagicMock(),
            press_escape=MagicMock(),
        )
        with patch("output.controller.pyautogui") as mock_pyautogui:
            mock_pyautogui.position.return_value = (100, 200)
            result = ctrl.execute("click")
        self.assertFalse(result.success)
        self.assertIn("hardware fault", result.message)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------
class TestThreadSafety(unittest.TestCase):
    """Verify concurrent calls don't crash."""

    def test_concurrent_executions(self):
        ctrl, _ = _MockController.create(dry_run=True)
        errors = []

        def worker(cmd):
            try:
                for _ in range(10):
                    ctrl.execute(cmd, confidence=0.99)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("volume_up",)),
            threading.Thread(target=worker, args=("mute",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(errors), 0)


# ---------------------------------------------------------------------------
# Clean shutdown
# ---------------------------------------------------------------------------
class TestCleanShutdown(unittest.TestCase):
    """Verify controller can be created and emergency-stopped cleanly."""

    def test_stop_then_execute(self):
        ctrl, mocks = _MockController.create()
        ctrl.trigger_emergency_stop()
        result = ctrl.execute("click")
        self.assertFalse(result.success)

    def test_multiple_stops_safe(self):
        ctrl, _ = _MockController.create()
        ctrl.trigger_emergency_stop()
        ctrl.trigger_emergency_stop()  # Should not raise
        self.assertTrue(ctrl.is_stopped())


# ---------------------------------------------------------------------------
# Hold-to-scroll tests (Change 3)
# ---------------------------------------------------------------------------
class TestHoldScrollManager(unittest.TestCase):
    """Verify HoldScrollManager behavior, commands, and safety controls."""

    def test_config_settings(self):
        self.assertTrue(hasattr(config, "HOLD_SCROLL_AMOUNT"))
        self.assertTrue(hasattr(config, "HOLD_SCROLL_INTERVAL_S"))
        self.assertTrue(hasattr(config, "HOLD_SCROLL_TIMEOUT_S"))
        self.assertGreater(config.HOLD_SCROLL_AMOUNT, 0)
        self.assertGreater(config.HOLD_SCROLL_INTERVAL_S, 0.0)
        self.assertGreater(config.HOLD_SCROLL_TIMEOUT_S, 0.0)

    def test_start_and_stop_up(self):
        ctrl, _ = _MockController.create(dry_run=True)
        mgr = ctrl.hold_scroll
        self.assertFalse(mgr.is_active())

        mgr.start(direction=1)
        self.assertTrue(mgr.is_active())

        mgr.stop()
        self.assertFalse(mgr.is_active())

    def test_start_and_stop_down(self):
        ctrl, _ = _MockController.create(dry_run=True)
        mgr = ctrl.hold_scroll
        self.assertFalse(mgr.is_active())

        mgr.start(direction=-1)
        self.assertTrue(mgr.is_active())

        mgr.stop()
        self.assertFalse(mgr.is_active())

    def test_direction_switch_while_active(self):
        ctrl, _ = _MockController.create(dry_run=True)
        mgr = ctrl.hold_scroll

        mgr.start(direction=-1)
        self.assertTrue(mgr.is_active())
        self.assertEqual(mgr._direction, -1)

        mgr.start(direction=1)
        self.assertTrue(mgr.is_active())
        self.assertEqual(mgr._direction, 1)

        mgr.stop()
        self.assertFalse(mgr.is_active())

    def test_emergency_stop_halts_hold_scroll(self):
        ctrl, _ = _MockController.create(dry_run=True)
        ctrl.hold_scroll.start(direction=-1)
        self.assertTrue(ctrl.hold_scroll.is_active())

        ctrl.trigger_emergency_stop()
        self.assertFalse(ctrl.hold_scroll.is_active())

    def test_skip_non_actionable_new_commands(self):
        ctrl, mocks = _MockController.create()
        for cmd in ("recenter", "hold_scroll_down", "hold_scroll_up", "stop_scroll"):
            res = ctrl.execute(cmd, confidence=1.0)
            self.assertTrue(res.success)
            self.assertIn("handled by", res.message)
            # Backend scroll functions should NOT be called directly by execute()
            mocks["scroll_up"].assert_not_called()
            mocks["scroll_down"].assert_not_called()

    def test_timeout_auto_stops(self):
        ctrl, _ = _MockController.create(dry_run=True)
        mgr = ctrl.hold_scroll
        orig_timeout = config.HOLD_SCROLL_TIMEOUT_S
        orig_interval = config.HOLD_SCROLL_INTERVAL_S
        try:
            config.HOLD_SCROLL_TIMEOUT_S = 0.05
            config.HOLD_SCROLL_INTERVAL_S = 0.01
            mgr.start(direction=-1)
            self.assertTrue(mgr.is_active())
            time.sleep(0.12)
            self.assertFalse(mgr.is_active())
        finally:
            config.HOLD_SCROLL_TIMEOUT_S = orig_timeout
            config.HOLD_SCROLL_INTERVAL_S = orig_interval


if __name__ == "__main__":
    unittest.main()
