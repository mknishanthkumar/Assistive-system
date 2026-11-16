"""
tests/test_controller_stage4.py - Stage 4 additions to OutputController.

Tests cover:
  - move_cursor() with clamping and dry-run
  - DRY_RUN config default
  - execute() source parameter (logging only, doesn't change behaviour)
  - Emergency stop blocks move_cursor

Run with:
    python -m pytest tests/test_controller_stage4.py -v
"""

import sys
import pathlib
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from output.controller import OutputController, ActionResult


def _make_ctrl(dry_run: bool = False) -> OutputController:
    return OutputController(
        dry_run=dry_run,
        mouse_click=MagicMock(),
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
        type_text=MagicMock(),
    )


# ---------------------------------------------------------------------------
# move_cursor
# ---------------------------------------------------------------------------
class TestMoveCursor(unittest.TestCase):

    @patch("output.controller.pyautogui")
    def test_move_cursor_basic(self, mock_pya):
        mock_pya.position.return_value = (100, 100)
        mock_pya.size.return_value = (1920, 1080)
        ctrl = _make_ctrl()
        ctrl.move_cursor(5.0, 3.0)
        mock_pya.moveTo.assert_called_once_with(105, 103)

    @patch("output.controller.pyautogui")
    def test_move_cursor_clamps_to_screen(self, mock_pya):
        mock_pya.position.return_value = (1919, 1079)
        mock_pya.size.return_value = (1920, 1080)
        ctrl = _make_ctrl()
        margin = config.CURSOR_SCREEN_MARGIN
        ctrl.move_cursor(50.0, 50.0)
        x, y = mock_pya.moveTo.call_args.args
        self.assertLessEqual(x, 1920 - margin)
        self.assertLessEqual(y, 1080 - margin)

    @patch("output.controller.pyautogui")
    def test_move_cursor_clamps_max_delta(self, mock_pya):
        mock_pya.position.return_value = (500, 500)
        mock_pya.size.return_value = (1920, 1080)
        ctrl = _make_ctrl()
        # Pass delta larger than CONTROLLER_MAX_CURSOR_DELTA
        ctrl.move_cursor(9999.0, 9999.0)
        x, y = mock_pya.moveTo.call_args.args
        # Should be clamped: 500 + max_delta (40) = 540
        self.assertLessEqual(x, 500 + config.CONTROLLER_MAX_CURSOR_DELTA + 1)
        self.assertLessEqual(y, 500 + config.CONTROLLER_MAX_CURSOR_DELTA + 1)

    @patch("output.controller.pyautogui")
    def test_move_cursor_zero_delta_no_move(self, mock_pya):
        ctrl = _make_ctrl()
        ctrl.move_cursor(0.0, 0.0)
        mock_pya.moveTo.assert_not_called()

    @patch("output.controller.pyautogui")
    def test_move_cursor_blocked_by_emergency_stop(self, mock_pya):
        ctrl = _make_ctrl()
        ctrl.trigger_emergency_stop()
        ctrl.move_cursor(5.0, 5.0)
        mock_pya.moveTo.assert_not_called()

    @patch("output.controller.pyautogui")
    def test_move_cursor_dry_run_no_moveto(self, mock_pya):
        ctrl = _make_ctrl(dry_run=True)
        ctrl.move_cursor(5.0, 5.0)
        mock_pya.moveTo.assert_not_called()

    @patch("output.controller.pyautogui")
    def test_move_cursor_exception_does_not_raise(self, mock_pya):
        mock_pya.position.side_effect = RuntimeError("pyautogui error")
        ctrl = _make_ctrl()
        try:
            ctrl.move_cursor(5.0, 5.0)   # Must not propagate exception
        except Exception as e:
            self.fail(f"move_cursor raised: {e}")

    @patch("output.controller.pyautogui")
    def test_move_cursor_negative_delta(self, mock_pya):
        mock_pya.position.return_value = (200, 200)
        mock_pya.size.return_value = (1920, 1080)
        ctrl = _make_ctrl()
        ctrl.move_cursor(-10.0, -5.0)
        mock_pya.moveTo.assert_called_once_with(190, 195)

    @patch("output.controller.pyautogui")
    def test_move_cursor_clamps_min_to_margin(self, mock_pya):
        mock_pya.position.return_value = (1, 1)
        mock_pya.size.return_value = (1920, 1080)
        ctrl = _make_ctrl()
        ctrl.move_cursor(-100.0, -100.0)
        x, y = mock_pya.moveTo.call_args.args
        margin = config.CURSOR_SCREEN_MARGIN
        self.assertGreaterEqual(x, margin)
        self.assertGreaterEqual(y, margin)


# ---------------------------------------------------------------------------
# execute() source parameter
# ---------------------------------------------------------------------------
class TestExecuteSourceParam(unittest.TestCase):

    @patch("output.controller.pyautogui")
    def test_execute_accepts_source_face(self, mock_pya):
        mock_pya.position.return_value = (100, 100)
        ctrl = _make_ctrl()
        result = ctrl.execute("click", confidence=0.9, source="face")
        self.assertTrue(result.success)

    @patch("output.controller.pyautogui")
    def test_execute_accepts_source_speech(self, mock_pya):
        mock_pya.position.return_value = (100, 100)
        ctrl = _make_ctrl()
        result = ctrl.execute("click", confidence=0.9, source="speech")
        self.assertTrue(result.success)

    @patch("output.controller.pyautogui")
    def test_execute_default_source(self, mock_pya):
        mock_pya.position.return_value = (100, 100)
        ctrl = _make_ctrl()
        result = ctrl.execute("click", confidence=0.9)
        self.assertTrue(result.success)


# ---------------------------------------------------------------------------
# DRY_RUN default from config
# ---------------------------------------------------------------------------
class TestDryRunDefault(unittest.TestCase):

    @patch("output.controller.pyautogui")
    def test_dry_run_default_false(self, mock_pya):
        """Default controller (config.DRY_RUN=False) executes real actions."""
        mock_pya.position.return_value = (100, 100)
        mock_btn = MagicMock()
        ctrl = OutputController(dry_run=False, mouse_click=mock_btn)
        ctrl.execute("click", confidence=1.0)
        mock_btn.assert_called_once()

    @patch("output.controller.pyautogui")
    def test_dry_run_true_skips_actions(self, mock_pya):
        mock_pya.position.return_value = (100, 100)
        mock_btn = MagicMock()
        ctrl = OutputController(dry_run=True, mouse_click=mock_btn)
        result = ctrl.execute("click", confidence=1.0)
        mock_btn.assert_not_called()
        self.assertIn("DRY-RUN", result.message)


if __name__ == "__main__":
    unittest.main()

