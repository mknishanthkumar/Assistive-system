"""
launcher.py – Demo Control Panel for the Assistive Control System.

Provides a Tkinter GUI to start/stop main.py without a terminal window.
Streams stdout/stderr live, tracks uptime, and shows the current mode
(FACE / SPEECH / BOTH / PAUSED) parsed from log output.

Rules:
  - Only adds new functionality; never imports from or modifies project modules.
  - Runs main.py as a subprocess; no camera or mic access inside this file.
  - Uses only stdlib (tkinter, subprocess, threading, queue, re, time, sys,
    pathlib, os) – no new dependencies.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import scrolledtext

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON  = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
ENTRY_POINT  = PROJECT_ROOT / "main.py"

# Fonts (large for projector readability)
FONT_TITLE   = ("Segoe UI", 20, "bold")
FONT_LABEL   = ("Segoe UI", 13)
FONT_VALUE   = ("Segoe UI", 13, "bold")
FONT_MODE    = ("Segoe UI", 28, "bold")
FONT_BUTTON  = ("Segoe UI", 14, "bold")
FONT_LOG     = ("Consolas", 11)
FONT_STATUS  = ("Segoe UI", 14, "bold")

# Colours
BG           = "#1e1e2e"      # dark background
BG_PANEL     = "#2a2a3e"      # slightly lighter panel
FG           = "#cdd6f4"      # text
ACCENT_GREEN = "#a6e3a1"
ACCENT_RED   = "#f38ba8"
ACCENT_BLUE  = "#89b4fa"
ACCENT_GOLD  = "#f9e2af"
ACCENT_TEAL  = "#94e2d5"
LOG_BG       = "#11111b"
LOG_FG       = "#cdd6f4"

# Mode colours
MODE_COLOURS = {
    "FACE":    "#89dceb",
    "SPEECH":  "#a6e3a1",
    "BOTH":    "#cba6f7",
    "PAUSED":  "#f38ba8",
    "–":       "#585b70",
}

# Regex patterns to detect current mode from log lines (no changes to main.py needed)
_RE_MODE_SWITCH = re.compile(
    r"MODE\s+SWITCH\s+\S+\s+.*?→\s*(\w+)", re.IGNORECASE
)
_RE_MODE_VALUE = re.compile(
    r"\bmode=(\w+)", re.IGNORECASE
)
# Also capture banner-style: "  Input sources  : Face…"
_RE_DECISION = re.compile(
    r"Decision:\s+mode=(\w+)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Subprocess management helpers
# ---------------------------------------------------------------------------

def _resolve_python() -> str:
    """Return the .venv Python path, falling back to sys.executable."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _launch_process() -> subprocess.Popen:
    """Start main.py with no console window, capturing stdout+stderr."""
    python = _resolve_python()
    CREATE_NO_WINDOW = 0x08000000  # Windows flag
    return subprocess.Popen(
        [python, "-u", str(ENTRY_POINT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge so both streams appear in log
        cwd=str(PROJECT_ROOT),
        creationflags=CREATE_NO_WINDOW,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _kill_process_tree(pid: int) -> None:
    """Kill the entire process tree so camera/mic are released."""
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(pid)],
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Output reader thread
# ---------------------------------------------------------------------------

class OutputReader(threading.Thread):
    """
    Reads lines from a Popen object in a background thread and
    posts them onto a queue so the UI thread never blocks.
    Posts None as a sentinel when the stream is exhausted.
    """

    def __init__(self, proc: subprocess.Popen, q: queue.Queue) -> None:
        super().__init__(daemon=True, name="OutputReader")
        self._proc = proc
        self._q = q

    def run(self) -> None:
        try:
            for line in self._proc.stdout:  # type: ignore[union-attr]
                self._q.put(line)
        except Exception:
            pass
        finally:
            self._q.put(None)   # sentinel → process finished


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class LauncherApp(tk.Tk):
    """Demo Control Panel window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Assistive Control System — Demo Panel")
        self.configure(bg=BG)
        self.minsize(900, 680)
        self.resizable(True, True)

        # State
        self._proc: subprocess.Popen | None = None
        self._reader: OutputReader | None = None
        self._log_q: queue.Queue = queue.Queue()
        self._start_time: float | None = None
        self._running: bool = False
        self._current_mode: str = "–"
        self._last_event_line: str = ""

        self._build_ui()
        self._poll_queue()   # start Tk polling loop
        self._update_uptime()

        # Intercept close button
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Title bar ─────────────────────────────────────────────────
        title_bar = tk.Frame(self, bg=BG_PANEL, pady=10)
        title_bar.pack(fill=tk.X)
        tk.Label(
            title_bar,
            text="🧠  Assistive Control System",
            font=FONT_TITLE,
            bg=BG_PANEL,
            fg=ACCENT_BLUE,
        ).pack(side=tk.LEFT, padx=20)
        tk.Label(
            title_bar,
            text="Demo Control Panel",
            font=("Segoe UI", 13),
            bg=BG_PANEL,
            fg=FG,
        ).pack(side=tk.LEFT, padx=4)

        # ── Top section: controls left, mode right ───────────────────
        top = tk.Frame(self, bg=BG)
        top.pack(fill=tk.X, padx=20, pady=(14, 0))

        self._build_control_panel(top)
        self._build_mode_panel(top)

        # ── Info panel ───────────────────────────────────────────────
        self._build_info_panel()

        # ── Log area ─────────────────────────────────────────────────
        self._build_log_panel()

    def _build_control_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text=" Controls ", bg=BG_PANEL, fg=FG,
            font=FONT_LABEL, bd=2, relief=tk.GROOVE, padx=16, pady=12,
        )
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        # Status indicator row
        status_row = tk.Frame(frame, bg=BG_PANEL)
        status_row.pack(anchor=tk.W, pady=(0, 10))
        self._status_dot = tk.Label(status_row, text="●", font=("Segoe UI", 22),
                                    bg=BG_PANEL, fg=ACCENT_RED)
        self._status_dot.pack(side=tk.LEFT)
        self._status_label = tk.Label(status_row, text=" Stopped",
                                      font=FONT_STATUS, bg=BG_PANEL, fg=ACCENT_RED)
        self._status_label.pack(side=tk.LEFT)

        # Uptime
        uptime_row = tk.Frame(frame, bg=BG_PANEL)
        uptime_row.pack(anchor=tk.W, pady=(0, 14))
        tk.Label(uptime_row, text="Uptime:", font=FONT_LABEL,
                 bg=BG_PANEL, fg=FG).pack(side=tk.LEFT)
        self._uptime_label = tk.Label(uptime_row, text=" 00:00:00",
                                      font=FONT_VALUE, bg=BG_PANEL, fg=ACCENT_GOLD)
        self._uptime_label.pack(side=tk.LEFT)

        # Buttons
        btn_frame = tk.Frame(frame, bg=BG_PANEL)
        btn_frame.pack()

        self._start_btn = tk.Button(
            btn_frame, text="▶  Start",
            font=FONT_BUTTON, bg="#40a02b", fg="white",
            activebackground="#2d7a1f", activeforeground="white",
            relief=tk.FLAT, padx=18, pady=8,
            command=self._start_system,
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self._stop_btn = tk.Button(
            btn_frame, text="■  Stop",
            font=FONT_BUTTON, bg="#d20f39", fg="white",
            activebackground="#9c0b2a", activeforeground="white",
            relief=tk.FLAT, padx=18, pady=8,
            command=self._stop_system,
            state=tk.DISABLED,
        )
        self._stop_btn.pack(side=tk.LEFT)

    def _build_mode_panel(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(
            parent, text=" Active Mode ", bg=BG_PANEL, fg=FG,
            font=FONT_LABEL, bd=2, relief=tk.GROOVE, padx=20, pady=12,
        )
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._mode_label = tk.Label(
            frame,
            text="–",
            font=FONT_MODE,
            bg=BG_PANEL,
            fg=MODE_COLOURS["–"],
        )
        self._mode_label.pack(expand=True)

        self._mode_sub = tk.Label(
            frame,
            text="Waiting for system…",
            font=("Segoe UI", 11),
            bg=BG_PANEL,
            fg="#585b70",
        )
        self._mode_sub.pack()

    def _build_info_panel(self) -> None:
        frame = tk.LabelFrame(
            self, text=" System Info ", bg=BG_PANEL, fg=FG,
            font=FONT_LABEL, bd=2, relief=tk.GROOVE, padx=14, pady=8,
        )
        frame.pack(fill=tk.X, padx=20, pady=(10, 0))

        grid = tk.Frame(frame, bg=BG_PANEL)
        grid.pack(fill=tk.X)

        def _row(label: str, var: tk.StringVar, row: int, col_offset: int = 0) -> None:
            tk.Label(grid, text=label, font=FONT_LABEL, bg=BG_PANEL, fg=FG,
                     anchor=tk.W).grid(row=row, column=col_offset * 2,
                                       sticky=tk.W, padx=(0, 6), pady=2)
            tk.Label(grid, textvariable=var, font=FONT_VALUE, bg=BG_PANEL,
                     fg=ACCENT_TEAL, anchor=tk.W).grid(
                row=row, column=col_offset * 2 + 1, sticky=tk.W, padx=(0, 30))

        self._var_python  = tk.StringVar(value=_resolve_python())
        self._var_entry   = tk.StringVar(value=str(ENTRY_POINT))
        self._var_pid     = tk.StringVar(value="–")
        self._var_last    = tk.StringVar(value="–")

        _row("Python:",      self._var_python,  row=0, col_offset=0)
        _row("Entry point:", self._var_entry,   row=1, col_offset=0)
        _row("PID:",         self._var_pid,     row=0, col_offset=1)
        _row("Last event:",  self._var_last,    row=1, col_offset=1)

        # Allow columns to expand
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

    def _build_log_panel(self) -> None:
        log_frame = tk.LabelFrame(
            self, text=" Live Output ", bg=BG_PANEL, fg=FG,
            font=FONT_LABEL, bd=2, relief=tk.GROOVE, padx=8, pady=6,
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 14))

        # Toolbar
        tb = tk.Frame(log_frame, bg=BG_PANEL)
        tb.pack(fill=tk.X, pady=(0, 4))
        tk.Button(
            tb, text="🗑  Clear Log",
            font=("Segoe UI", 11), bg="#313244", fg=FG,
            activebackground="#45475a", relief=tk.FLAT, padx=10, pady=4,
            command=self._clear_log,
        ).pack(side=tk.RIGHT)

        # Scrolled text widget
        self._log_text = scrolledtext.ScrolledText(
            log_frame,
            font=FONT_LOG,
            bg=LOG_BG,
            fg=LOG_FG,
            insertbackground=FG,
            selectbackground="#45475a",
            relief=tk.FLAT,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self._log_text.pack(fill=tk.BOTH, expand=True)

        # Tag colours for severity
        self._log_text.tag_config("warn",  foreground=ACCENT_GOLD)
        self._log_text.tag_config("error", foreground=ACCENT_RED)
        self._log_text.tag_config("info",  foreground=ACCENT_GREEN)
        self._log_text.tag_config("mode",  foreground="#cba6f7")
        self._log_text.tag_config("dim",   foreground="#585b70")

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def _start_system(self) -> None:
        if self._running:
            return

        self._log_q = queue.Queue()
        try:
            self._proc = _launch_process()
        except Exception as exc:
            self._append_log(f"[LAUNCHER] Failed to start: {exc}\n", "error")
            return

        self._running    = True
        self._start_time = time.monotonic()
        self._current_mode = "–"

        self._reader = OutputReader(self._proc, self._log_q)
        self._reader.start()

        # Update UI
        self._set_running_state(True)
        self._var_pid.set(str(self._proc.pid))
        self._append_log(
            f"[LAUNCHER] Started PID {self._proc.pid} → {ENTRY_POINT.name}\n", "info"
        )

    def _stop_system(self) -> None:
        if not self._running or self._proc is None:
            return
        self._append_log("[LAUNCHER] Stopping — killing process tree…\n", "warn")
        try:
            _kill_process_tree(self._proc.pid)
        except Exception as exc:
            self._append_log(f"[LAUNCHER] Kill error: {exc}\n", "error")
        # UI reset happens when sentinel arrives via poll_queue

    # ------------------------------------------------------------------
    # UI state helpers
    # ------------------------------------------------------------------

    def _set_running_state(self, running: bool) -> None:
        self._running = running
        if running:
            self._status_dot.config(fg=ACCENT_GREEN)
            self._status_label.config(text=" Running", fg=ACCENT_GREEN)
            self._start_btn.config(state=tk.DISABLED)
            self._stop_btn.config(state=tk.NORMAL)
        else:
            self._status_dot.config(fg=ACCENT_RED)
            self._status_label.config(text=" Stopped", fg=ACCENT_RED)
            self._start_btn.config(state=tk.NORMAL)
            self._stop_btn.config(state=tk.DISABLED)
            self._start_time = None
            self._uptime_label.config(text=" 00:00:00")
            self._var_pid.set("–")
            self._update_mode("–")

    def _update_mode(self, mode: str) -> None:
        self._current_mode = mode
        colour = MODE_COLOURS.get(mode.upper(), MODE_COLOURS["–"])
        self._mode_label.config(text=mode, fg=colour)
        subtext = {
            "FACE":   "Head/eye control active",
            "SPEECH": "Voice command active",
            "BOTH":   "Face + voice active",
            "PAUSED": "System paused",
            "–":      "Waiting for system…",
        }.get(mode.upper(), "")
        self._mode_sub.config(text=subtext)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _append_log(self, text: str, tag: str = "") -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, text, tag if tag else ())
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _classify_line(self, line: str) -> str:
        """Return a log tag based on content."""
        low = line.lower()
        if "error" in low or "traceback" in low or "exception" in low:
            return "error"
        if "warning" in low or "warn" in low:
            return "warn"
        if "mode switch" in low or "mode=" in low:
            return "mode"
        if "info" in low or "started" in low or "running" in low:
            return "info"
        return ""

    def _extract_mode(self, line: str) -> str | None:
        """Try to parse current mode from a log line."""
        for pattern in (_RE_MODE_SWITCH, _RE_DECISION, _RE_MODE_VALUE):
            m = pattern.search(line)
            if m:
                candidate = m.group(1).upper()
                if candidate in ("FACE", "SPEECH", "BOTH", "PAUSED"):
                    return candidate
        return None

    # ------------------------------------------------------------------
    # Polling loops (called repeatedly via after())
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        """Drain the output queue and update the log widget — never blocks."""
        try:
            while True:
                item = self._log_q.get_nowait()
                if item is None:
                    # Sentinel: process has exited
                    self._handle_process_exit()
                    break
                # item is a text line
                tag = self._classify_line(item)
                self._append_log(item, tag)

                # Update last event line (strip ANSI/whitespace)
                stripped = item.strip()
                if stripped:
                    self._last_event_line = stripped[:120]
                    self._var_last.set(self._last_event_line)

                # Parse mode
                mode = self._extract_mode(item)
                if mode:
                    self._update_mode(mode)

        except queue.Empty:
            pass

        # Reschedule every 50 ms
        self.after(50, self._poll_queue)

    def _update_uptime(self) -> None:
        """Update the HH:MM:SS uptime counter."""
        if self._running and self._start_time is not None:
            elapsed = int(time.monotonic() - self._start_time)
            h, rem = divmod(elapsed, 3600)
            m, s   = divmod(rem, 60)
            self._uptime_label.config(text=f" {h:02d}:{m:02d}:{s:02d}")
        self.after(1000, self._update_uptime)

    def _handle_process_exit(self) -> None:
        """Called when the reader thread posts a sentinel (process ended)."""
        return_code = None
        if self._proc is not None:
            return_code = self._proc.poll()

        if return_code is None or return_code == 0:
            self._append_log(
                "\n[LAUNCHER] System stopped cleanly.\n", "info"
            )
        else:
            self._append_log(
                f"\n[LAUNCHER] Process exited with code {return_code} — check log above.\n",
                "error",
            )

        self._proc = None
        self._set_running_state(False)

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        """Stop the system cleanly before closing the window."""
        if self._running:
            self._stop_system()
            # Give the process a moment to die, then close
            self.after(1500, self.destroy)
        else:
            self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()

