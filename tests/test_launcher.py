"""
tests/test_launcher.py – Unit tests for launcher.py start/stop logic.

Mocks subprocess so no camera, microphone, or real process is involved.
Tests run with: python -m pytest tests/test_launcher.py -v
"""

from __future__ import annotations

import io
import queue
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure the project root is on sys.path so `import launcher` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import launcher  # noqa: E402  (import after sys.path fix)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_popen(lines: list[str], returncode: int = 0) -> MagicMock:
    """
    Build a MagicMock that looks like a subprocess.Popen object.
    Its .stdout iterates over *lines* then stops.
    Its .poll() returns *returncode*.
    Its .pid is 12345.
    """
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll.return_value = returncode
    # stdout must be iterable line-by-line
    mock_proc.stdout = iter(lines)
    return mock_proc


# ---------------------------------------------------------------------------
# OutputReader tests
# ---------------------------------------------------------------------------

class TestOutputReader(unittest.TestCase):
    """Tests for the background reader thread."""

    def test_posts_all_lines_then_sentinel(self) -> None:
        """Reader should put every line then None onto the queue."""
        lines = ["line one\n", "line two\n", "line three\n"]
        proc  = _make_fake_popen(lines)
        q: queue.Queue = queue.Queue()

        reader = launcher.OutputReader(proc, q)
        reader.start()
        reader.join(timeout=3)

        collected = []
        while True:
            item = q.get(timeout=1)
            collected.append(item)
            if item is None:
                break

        self.assertEqual(collected, lines + [None])

    def test_sentinel_posted_even_on_empty_output(self) -> None:
        """When stdout has no lines, only the sentinel should appear."""
        proc = _make_fake_popen([])
        q: queue.Queue = queue.Queue()

        reader = launcher.OutputReader(proc, q)
        reader.start()
        reader.join(timeout=3)

        item = q.get(timeout=1)
        self.assertIsNone(item)

    def test_reader_is_daemon_thread(self) -> None:
        """Reader thread must be a daemon so it doesn't block exit."""
        proc   = _make_fake_popen([])
        reader = launcher.OutputReader(proc, queue.Queue())
        self.assertTrue(reader.daemon)


# ---------------------------------------------------------------------------
# _resolve_python tests
# ---------------------------------------------------------------------------

class TestResolvePython(unittest.TestCase):

    def test_returns_venv_python_when_exists(self) -> None:
        with patch("launcher.VENV_PYTHON") as mock_path:
            mock_path.exists.return_value = True
            mock_path.__str__ = lambda self: r"C:\fake\.venv\Scripts\python.exe"
            result = launcher._resolve_python()
        self.assertIn("python", result.lower())

    def test_falls_back_to_sys_executable(self) -> None:
        with patch("launcher.VENV_PYTHON") as mock_path:
            mock_path.exists.return_value = False
            result = launcher._resolve_python()
        self.assertEqual(result, sys.executable)


# ---------------------------------------------------------------------------
# _kill_process_tree tests
# ---------------------------------------------------------------------------

class TestKillProcessTree(unittest.TestCase):

    @patch("launcher.subprocess.run")
    def test_calls_taskkill_with_correct_flags(self, mock_run: MagicMock) -> None:
        launcher._kill_process_tree(9999)
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]  # first positional arg (list)
        self.assertIn("taskkill", call_args)
        self.assertIn("/T",  call_args)
        self.assertIn("/F",  call_args)
        self.assertIn("/PID", call_args)
        self.assertIn("9999", call_args)


# ---------------------------------------------------------------------------
# _launch_process tests
# ---------------------------------------------------------------------------

class TestLaunchProcess(unittest.TestCase):

    @patch("launcher.subprocess.Popen")
    @patch("launcher._resolve_python", return_value=r"C:\fake\python.exe")
    def test_uses_unbuffered_flag(self, _mock_resolve: MagicMock,
                                  mock_popen: MagicMock) -> None:
        launcher._launch_process()
        args = mock_popen.call_args[0][0]   # argv list
        self.assertIn("-u", args)

    @patch("launcher.subprocess.Popen")
    @patch("launcher._resolve_python", return_value=r"C:\fake\python.exe")
    def test_uses_create_no_window(self, _mock_resolve: MagicMock,
                                   mock_popen: MagicMock) -> None:
        launcher._launch_process()
        kwargs = mock_popen.call_args[1]
        CREATE_NO_WINDOW = 0x08000000
        self.assertEqual(kwargs.get("creationflags"), CREATE_NO_WINDOW)

    @patch("launcher.subprocess.Popen")
    @patch("launcher._resolve_python", return_value=r"C:\fake\python.exe")
    def test_runs_main_py(self, _mock_resolve: MagicMock,
                          mock_popen: MagicMock) -> None:
        launcher._launch_process()
        args = mock_popen.call_args[0][0]
        self.assertTrue(
            any("main.py" in str(a) for a in args),
            msg=f"Expected main.py in args: {args}",
        )


# ---------------------------------------------------------------------------
# Mode parsing tests (_extract_mode via LauncherApp)
# ---------------------------------------------------------------------------

class TestExtractMode(unittest.TestCase):
    """
    Test the log-line mode parser without opening any GUI window.
    We instantiate a minimal object using object.__new__ to avoid Tk.
    """

    def _parser(self):
        # Bypass __init__ (which creates a Tk window)
        obj = object.__new__(launcher.LauncherApp)
        return obj

    def test_detects_mode_switch_line(self) -> None:
        app = self._parser()
        line = "[2026-11-18 20:21:40] INFO     assistive – MODE SWITCH FACE → SPEECH | reason=noisy"
        self.assertEqual(app._extract_mode(line), "SPEECH")

    def test_detects_decision_mode_line(self) -> None:
        app = self._parser()
        line = "Decision: mode=FACE reason=bright override=False"
        self.assertEqual(app._extract_mode(line), "FACE")

    def test_detects_mode_eq_both(self) -> None:
        app = self._parser()
        line = "some log line mode=BOTH things"
        self.assertEqual(app._extract_mode(line), "BOTH")

    def test_detects_paused(self) -> None:
        app = self._parser()
        line = "MODE SWITCH FACE → PAUSED | reason=noisy+dark"
        self.assertEqual(app._extract_mode(line), "PAUSED")

    def test_returns_none_for_unrelated_line(self) -> None:
        app = self._parser()
        line = "FaceTracker started successfully on camera index 0."
        self.assertIsNone(app._extract_mode(line))

    def test_ignores_unknown_mode_values(self) -> None:
        app = self._parser()
        line = "mode=UNKNOWN_VALUE"
        self.assertIsNone(app._extract_mode(line))


# ---------------------------------------------------------------------------
# Log classification tests
# ---------------------------------------------------------------------------

class TestClassifyLine(unittest.TestCase):

    def _app(self):
        return object.__new__(launcher.LauncherApp)

    def test_error_tag_for_error_lines(self) -> None:
        app = self._app()
        self.assertEqual(app._classify_line("ERROR: something broke"), "error")
        self.assertEqual(app._classify_line("Traceback (most recent call last)"), "error")

    def test_warn_tag_for_warning_lines(self) -> None:
        app = self._app()
        self.assertEqual(app._classify_line("WARNING camera not found"), "warn")

    def test_mode_tag_for_mode_switch_lines(self) -> None:
        app = self._app()
        self.assertEqual(app._classify_line("MODE SWITCH FACE → SPEECH"), "mode")

    def test_info_tag_for_started_lines(self) -> None:
        app = self._app()
        self.assertEqual(app._classify_line("FaceTracker started."), "info")

    def test_empty_tag_for_generic_lines(self) -> None:
        app = self._app()
        self.assertEqual(app._classify_line("some random output line"), "")


# ---------------------------------------------------------------------------
# Integration-style test: start → lines → sentinel → stopped
# ---------------------------------------------------------------------------

class TestStartStopFlow(unittest.TestCase):
    """
    Simulate the full start→output→exit flow without a real Tk mainloop.
    We patch Popen and exercise _start_system/_stop_system/_poll_queue manually.
    """

    @patch("launcher.subprocess.Popen")
    @patch("launcher._resolve_python", return_value=r"C:\fake\python.exe")
    def test_start_sets_running_state(self, _mock_resolve: MagicMock,
                                      mock_popen: MagicMock) -> None:
        lines = ["Starting up...\n", "FaceTracker started.\n"]
        mock_popen.return_value = _make_fake_popen(lines, returncode=0)

        # Build app without mainloop
        app = _HeadlessLauncherApp()
        self.assertFalse(app._running)

        app._start_system()

        self.assertTrue(app._running)
        self.assertEqual(app._var_pid.get(), "12345")
        self.assertIsNotNone(app._proc)

    @patch("launcher.subprocess.run")
    @patch("launcher.subprocess.Popen")
    @patch("launcher._resolve_python", return_value=r"C:\fake\python.exe")
    def test_stop_calls_taskkill(self, _mock_resolve: MagicMock,
                                 mock_popen: MagicMock,
                                 mock_run: MagicMock) -> None:
        mock_popen.return_value = _make_fake_popen([], returncode=0)

        app = _HeadlessLauncherApp()
        app._start_system()
        self.assertTrue(app._running)

        app._stop_system()

        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        self.assertIn("taskkill", call_args)

    @patch("launcher.subprocess.Popen")
    @patch("launcher._resolve_python", return_value=r"C:\fake\python.exe")
    def test_sentinel_resets_ui_to_stopped(self, _mock_resolve: MagicMock,
                                           mock_popen: MagicMock) -> None:
        lines = ["All good\n"]
        mock_popen.return_value = _make_fake_popen(lines, returncode=0)

        app = _HeadlessLauncherApp()
        app._start_system()

        # Wait for reader thread to finish and post sentinel
        if app._reader:
            app._reader.join(timeout=3)

        # Drain queue manually (simulates poll_queue)
        app._drain_queue()

        self.assertFalse(app._running)

    @patch("launcher.subprocess.Popen")
    @patch("launcher._resolve_python", return_value=r"C:\fake\python.exe")
    def test_nonzero_exit_still_resets_ui(self, _mock_resolve: MagicMock,
                                          mock_popen: MagicMock) -> None:
        mock_popen.return_value = _make_fake_popen(["crash\n"], returncode=1)

        app = _HeadlessLauncherApp()
        app._start_system()

        if app._reader:
            app._reader.join(timeout=3)

        app._drain_queue()

        self.assertFalse(app._running)


# ---------------------------------------------------------------------------
# Headless helper (bypasses Tk widget creation for unit tests)
# ---------------------------------------------------------------------------

class _StringVar:
    """Minimal StringVar replacement that doesn't need a Tk root."""
    def __init__(self, value: str = "") -> None:
        self._value = value
    def get(self) -> str:
        return self._value
    def set(self, value: str) -> None:
        self._value = value


class _HeadlessLauncherApp(launcher.LauncherApp):
    """
    Subclass that skips all Tk widget creation so tests can run headlessly.
    Overrides _build_ui, _append_log, _set_running_state, etc. with no-ops
    or lightweight replacements.
    """

    def __init__(self) -> None:  # noqa: D107
        # Skip tk.Tk.__init__ entirely
        self._proc: "launcher.subprocess.Popen | None" = None
        self._reader: "launcher.OutputReader | None" = None
        self._log_q: queue.Queue = queue.Queue()
        self._start_time: float | None = None
        self._running: bool = False
        self._current_mode: str = "–"
        self._last_event_line: str = ""
        self._log_lines: list[str] = []

        self._var_pid  = _StringVar("–")
        self._var_last = _StringVar("–")

    # Override Tk-dependent methods with no-ops
    def after(self, _ms: int, _func=None, *_args) -> None:  # type: ignore[override]
        pass

    def _build_ui(self) -> None:
        pass

    def _append_log(self, text: str, tag: str = "") -> None:
        self._log_lines.append(text)

    def _set_running_state(self, running: bool) -> None:
        self._running = running
        if not running:
            self._start_time = None
            self._var_pid.set("–")
            self._update_mode("–")

    def _update_mode(self, mode: str) -> None:
        self._current_mode = mode

    def _update_uptime(self) -> None:
        pass

    def _drain_queue(self) -> None:
        """Synchronously drain the log queue (replaces _poll_queue in tests)."""
        while True:
            try:
                item = self._log_q.get(timeout=2)
                if item is None:
                    self._handle_process_exit()
                    break
            except queue.Empty:
                break


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)

