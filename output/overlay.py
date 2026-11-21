"""
output/overlay.py - Compact always-on-top mode-badge for the Assistive Control System.

Architecture
------------
  OverlayManager   : Public API; thread-safe; non-blocking queue-based.
  OverlayThread    : Daemon thread that owns the tkinter mainloop.
  _OverlayWindow   : Internal class; ALL tkinter calls happen exclusively here.

Design: compact horizontal pill at top-centre of the screen, never in a corner
(which would trigger PyAutoGUI's fail-safe).  Position is configurable via
OVERLAY_POSITION in config.py ("top-center" | "top-left" | "top-right").

Windows-specific features
--------------------------
  WS_EX_TRANSPARENT   -- click-through (mouse events pass to window beneath)
  WS_EX_NOACTIVATE    -- never steals focus or keyboard input
  WS_EX_TOOLWINDOW    -- hidden from Alt-Tab / taskbar
  SetLayeredWindowAttributes -- per-window alpha (OVERLAY_OPACITY)
  SetWindowPos(HWND_TOPMOST) -- re-asserted every 1 s to survive fullscreen apps

Pill layout (single row, ~260 x 32 px):
  [MODE]  ● face  ● mic  [last cmd — fades after 2 s]  [PAUSED]
"""

from __future__ import annotations

import ctypes
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from typing import Optional, Tuple

import config

logger = logging.getLogger("assistive.overlay")

# ---------------------------------------------------------------------------
# Config knobs (read from config.py with safe defaults)
# ---------------------------------------------------------------------------
OVERLAY_ENABLED: bool   = getattr(config, "OVERLAY_ENABLED",          True)
OVERLAY_HOTKEY: str     = getattr(config, "OVERLAY_HOTKEY_PYNPUT",    "<ctrl>+<alt>+h")
OVERLAY_OPACITY: float  = getattr(config, "OVERLAY_OPACITY",          0.92)
OVERLAY_FONT_SIZE: int  = getattr(config, "OVERLAY_FONT_SIZE",        12)
OVERLAY_POSITION: str   = getattr(config, "OVERLAY_POSITION",         "top-center")
# Inset from each edge so the badge never sits in a corner
OVERLAY_MARGIN_X: int   = getattr(config, "OVERLAY_MARGIN_X",         60)
OVERLAY_MARGIN_Y: int   = getattr(config, "OVERLAY_MARGIN_Y",         8)

# Update cadence
QUEUE_DRAIN_MS: int     = 40    # drain queue every 40 ms (25 Hz)
TOPMOST_ASSERT_MS: int  = getattr(config, "OVERLAY_TOPMOST_ASSERT_MS", 500)  # re-assert every 500 ms
PROXIMITY_POLL_MS: int  = 500
PROXIMITY_PX: int       = 120
CMD_FADE_S: float       = 2.5   # seconds before last-command text disappears

SPEECH_FAIL_HINT_N: int = 3
BEEP_DURATION_MS: int   = 150

# ---------------------------------------------------------------------------
# Windows constants
# ---------------------------------------------------------------------------
GWL_EXSTYLE       = -20
WS_EX_LAYERED     = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_NOACTIVATE  = 0x08000000
WS_EX_TOOLWINDOW  = 0x00000080
LWA_ALPHA         = 0x00000002
SWP_NOMOVE        = 0x0002
SWP_NOSIZE        = 0x0001
SWP_NOACTIVATE    = 0x0010
HWND_TOPMOST      = -1

# ---------------------------------------------------------------------------
# Mode metadata  mode → (label, fg-colour, beep-Hz)
# ---------------------------------------------------------------------------
MODE_META = {
    "FACE":   ("FACE",   "#00CFFF", 800),
    "SPEECH": ("SPEECH", "#FFE066", 600),
    "BOTH":   ("BOTH",   "#44FF88", 1000),
    "PAUSED": ("PAUSED", "#FF4444", 400),
    "MANUAL": ("MANUAL", "#FF9900", 700),
    "AUTO":   ("AUTO",   "#44FF88", 1000),
}
DEFAULT_META = ("???", "#FFFFFF", 500)

# Dark pill background
_BG = "#1A1A2E"

# ---------------------------------------------------------------------------
# Async beep helper
# ---------------------------------------------------------------------------

def _beep_async(freq: int, duration_ms: int = BEEP_DURATION_MS) -> None:
    """Fire a beep in a throw-away daemon thread so the caller is never blocked."""
    def _run():
        try:
            import winsound  # type: ignore[import]
            winsound.Beep(max(37, min(32767, freq)), duration_ms)
        except Exception:
            try:
                print("\a", end="", flush=True)
            except Exception:
                pass
    threading.Thread(target=_run, daemon=True).start()


def play_mode_sound(mode: str) -> None:
    """Public helper: play the beep for *mode*.  Safe to call from any thread."""
    _, _, freq = MODE_META.get(mode.upper(), DEFAULT_META)
    _beep_async(freq)


# ---------------------------------------------------------------------------
# Win32 helpers
# ---------------------------------------------------------------------------

def _get_hwnd(winfo_id: int) -> int:
    """Return the top-level HWND for a tkinter window id."""
    try:
        hwnd = ctypes.windll.user32.GetParent(winfo_id)
        return hwnd if hwnd else winfo_id
    except Exception:
        return winfo_id


def _apply_extended_style(hwnd: int, opacity: float) -> None:
    """Set WS_EX_TRANSPARENT|NOACTIVATE|TOOLWINDOW and layered alpha."""
    try:
        u32 = ctypes.windll.user32
        cur = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        u32.SetWindowLongW(
            hwnd, GWL_EXSTYLE,
            cur | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
        )
        u32.SetLayeredWindowAttributes(hwnd, 0, int(opacity * 255), LWA_ALPHA)
    except Exception:
        logger.warning("Win32 extended style failed (non-Windows?)")


def _assert_topmost(hwnd: int) -> None:
    """Re-assert HWND_TOPMOST without moving or resizing."""
    try:
        ctypes.windll.user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        )
    except Exception:
        pass


def _get_mouse_pos() -> Tuple[int, int]:
    try:
        class _POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
        pt = _POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
    except Exception:
        return -9999, -9999


# ---------------------------------------------------------------------------
# _OverlayWindow — all tkinter lives here, on the OverlayThread only
# ---------------------------------------------------------------------------

class _OverlayWindow:
    """
    Compact single-row pill badge.
    Layout: [MODE]  ●face  ●mic   [cmd text — fades]   [⚠PAUSED]
    """

    # Pill dimensions (auto-sized by pack; these are starting hints)
    _PAD_X = 10
    _PAD_Y = 4

    def __init__(self) -> None:
        self._root: Optional[tk.Tk] = None
        self._hwnd: int = 0
        self._visible: bool = True
        self._proximity_hidden: bool = False
        self._state: dict = {}
        self._last_mode: Optional[str] = None
        self._cmd_text: str = ""
        self._cmd_set_time: float = 0.0   # monotonic; for fade-out
        self._bounds: Optional[Tuple[int, int, int, int]] = None
        self._flash_phase: bool = False

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(self) -> bool:
        try:
            root = tk.Tk()
            self._root = root

            root.overrideredirect(True)
            root.wm_attributes("-topmost", True)
            root.wm_attributes("-alpha", OVERLAY_OPACITY)
            root.configure(bg=_BG)
            root.resizable(False, False)

            self._build_pill()
            self._position_window()

            # Grab HWND once the window is mapped
            root.update_idletasks()
            self._hwnd = _get_hwnd(root.winfo_id())
            _apply_extended_style(self._hwnd, OVERLAY_OPACITY)

            # Schedule periodic hooks
            root.after(QUEUE_DRAIN_MS,     self._tick_drain)   # set by OverlayThread
            root.after(TOPMOST_ASSERT_MS,  self._tick_topmost)
            root.after(PROXIMITY_POLL_MS,  self._tick_proximity)
            root.after(QUEUE_DRAIN_MS,     self._tick_cmd_fade)

            return True
        except Exception:
            logger.exception("Failed to create overlay window")
            self._root = None
            return False

    # ------------------------------------------------------------------
    # Pill widget layout (single horizontal row)
    # ------------------------------------------------------------------

    def _build_pill(self) -> None:
        root = self._root
        px, py = self._PAD_X, self._PAD_Y

        bold = tkfont.Font(family="Consolas", size=OVERLAY_FONT_SIZE, weight="bold")
        reg  = tkfont.Font(family="Consolas", size=OVERLAY_FONT_SIZE - 1)
        tiny = tkfont.Font(family="Consolas", size=OVERLAY_FONT_SIZE - 2)

        # Outer frame = the pill shape
        outer = tk.Frame(root, bg=_BG, relief="flat", bd=0)
        outer.pack(fill="x", padx=0, pady=0)

        # Mode badge
        self._mode_lbl = tk.Label(
            outer, text="FACE", font=bold, bg="#003355", fg="#00CFFF",
            padx=8, pady=py, relief="flat",
        )
        self._mode_lbl.pack(side="left", padx=(px, 4))

        # Face dot
        self._face_dot = tk.Label(outer, text="●", font=reg, bg=_BG, fg="#FF4444")
        self._face_dot.pack(side="left", padx=1)
        self._face_lbl = tk.Label(outer, text="face", font=tiny, bg=_BG, fg="#888888")
        self._face_lbl.pack(side="left", padx=(0, 6))

        # Mic dot
        self._mic_dot = tk.Label(outer, text="●", font=reg, bg=_BG, fg="#FF4444")
        self._mic_dot.pack(side="left", padx=1)
        self._mic_lbl = tk.Label(outer, text="mic", font=tiny, bg=_BG, fg="#888888")
        self._mic_lbl.pack(side="left", padx=(0, 8))

        # Last command (fades after CMD_FADE_S seconds)
        self._cmd_lbl = tk.Label(
            outer, text="", font=tiny, bg=_BG, fg="#AAAAAA",
            width=16, anchor="w",
        )
        self._cmd_lbl.pack(side="left", padx=(0, 6))

        # PAUSED tag (hidden by default)
        self._pause_lbl = tk.Label(
            outer, text="⚠ PAUSED", font=bold, bg="#FF4444", fg="#FFFFFF",
            padx=6, pady=py,
        )
        # Not packed yet

        # Speech-failure hint (hidden by default)
        self._hint_lbl = tk.Label(
            outer, text="→ Ctrl+Alt+2", font=tiny, bg=_BG, fg="#FFAA00",
        )
        # Not packed yet

    def _position_window(self) -> None:
        root = self._root
        root.update_idletasks()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        # Force the window to render so we get real dimensions
        root.geometry(f"+{sw // 2}+{OVERLAY_MARGIN_Y}")
        root.update_idletasks()
        w = max(root.winfo_reqwidth(), 260)
        h = max(root.winfo_reqheight(), 32)

        pos = OVERLAY_POSITION.lower()
        mx, my = OVERLAY_MARGIN_X, OVERLAY_MARGIN_Y

        if "right" in pos:
            x = sw - w - mx
        elif "left" in pos:
            x = mx
        else:                        # top-center (default)
            x = (sw - w) // 2

        if "bottom" in pos:
            y = sh - h - mx         # inset from bottom too
        else:
            y = my                  # near-top, never at 0

        # Clamp away from true corners (failsafe zones are 1-pixel corners)
        x = max(mx, min(sw - w - mx, x))
        y = max(my, min(sh - h - mx, y))

        root.geometry(f"{w}x{h}+{x}+{y}")
        self._bounds = (x, y, w, h)

    # ------------------------------------------------------------------
    # Periodic: re-assert HWND_TOPMOST
    # ------------------------------------------------------------------

    def _tick_topmost(self) -> None:
        if self._root is None:
            return
        try:
            if self._hwnd:
                _assert_topmost(self._hwnd)
            else:
                # Retry HWND acquisition
                self._root.update_idletasks()
                self._hwnd = _get_hwnd(self._root.winfo_id())
                if self._hwnd:
                    _apply_extended_style(self._hwnd, OVERLAY_OPACITY)
        except Exception:
            pass
        finally:
            if self._root:
                self._root.after(TOPMOST_ASSERT_MS, self._tick_topmost)

    # ------------------------------------------------------------------
    # Periodic: proximity hide
    # ------------------------------------------------------------------

    def _tick_proximity(self) -> None:
        if self._root is None:
            return
        try:
            if self._visible and self._bounds:
                mx, my = _get_mouse_pos()
                bx, by, bw, bh = self._bounds
                near = (
                    (bx - PROXIMITY_PX) <= mx <= (bx + bw + PROXIMITY_PX)
                    and (by - PROXIMITY_PX) <= my <= (by + bh + PROXIMITY_PX)
                )
                if near and not self._proximity_hidden:
                    self._proximity_hidden = True
                    self._root.wm_attributes("-alpha", 0.08)
                elif not near and self._proximity_hidden:
                    self._proximity_hidden = False
                    self._root.wm_attributes("-alpha", OVERLAY_OPACITY)
        except Exception:
            pass
        finally:
            if self._root:
                self._root.after(PROXIMITY_POLL_MS, self._tick_proximity)

    # ------------------------------------------------------------------
    # Periodic: fade last-command text
    # ------------------------------------------------------------------

    def _tick_cmd_fade(self) -> None:
        if self._root is None:
            return
        try:
            if self._cmd_text and (time.monotonic() - self._cmd_set_time) > CMD_FADE_S:
                self._cmd_text = ""
                self._cmd_lbl.config(text="")
        except Exception:
            pass
        finally:
            if self._root:
                self._root.after(500, self._tick_cmd_fade)

    # ------------------------------------------------------------------
    # Queue drain (called by OverlayThread, re-scheduled here)
    # ------------------------------------------------------------------

    def _tick_drain(self) -> None:
        # This is a placeholder; actual drain is in OverlayThread._drain_queue
        # which re-schedules itself via after().
        pass

    # ------------------------------------------------------------------
    # Apply state → update widgets
    # ------------------------------------------------------------------

    def apply_state(self, state: dict) -> None:
        """
        Called by the queue-drain loop (already on the tk thread via after()).
        Updates all pill widgets immediately.
        """
        new_mode = str(state.get("mode", "FACE")).upper()
        face_ok  = bool(state.get("face_ok", False))
        mic_ok   = bool(state.get("mic_ok", False))
        last_cmd = str(state.get("last_cmd", ""))
        last_ok  = bool(state.get("last_ok", True))
        paused   = bool(state.get("paused", False))
        fail_n   = int(state.get("speech_fail_count", 0))

        # Beep on mode switch
        if new_mode != self._last_mode and self._last_mode is not None:
            _, _, freq = MODE_META.get(new_mode, DEFAULT_META)
            _beep_async(freq)
        self._last_mode = new_mode

        # --- Mode badge ---
        label, fg, _ = MODE_META.get(new_mode, DEFAULT_META)
        # Pick a subtly tinted background per mode
        bg_map = {
            "FACE":   "#002233",
            "SPEECH": "#2A2000",
            "BOTH":   "#002200",
            "PAUSED": "#330000",
            "MANUAL": "#2A1500",
            "AUTO":   "#002200",
        }
        badge_bg = bg_map.get(new_mode, "#1A1A2E")
        self._mode_lbl.config(text=label, fg=fg, bg=badge_bg)

        # --- Status dots ---
        face_color = "#44FF88" if face_ok else "#FF4444"
        mic_color  = "#44FF88" if mic_ok  else "#FF6600"
        self._face_dot.config(fg=face_color)
        self._mic_dot.config(fg=mic_color)

        # --- Last command (brief display) ---
        if last_cmd and last_cmd != self._cmd_text:
            self._cmd_text     = last_cmd
            self._cmd_set_time = time.monotonic()
            icon  = "✓ " if last_ok else "✗ "
            color = "#44FF88" if last_ok else "#FF4444"
            self._cmd_lbl.config(
                text=(icon + last_cmd[:14]).ljust(16),
                fg=color,
            )

        # --- PAUSED banner ---
        self._flash_phase = not self._flash_phase
        if paused:
            flash_bg = "#FF4444" if self._flash_phase else "#CC2200"
            self._pause_lbl.config(bg=flash_bg)
            if not self._pause_lbl.winfo_ismapped():
                self._pause_lbl.pack(side="left", padx=(4, 8))
        else:
            if self._pause_lbl.winfo_ismapped():
                self._pause_lbl.pack_forget()

        # --- Speech-failure hint ---
        if fail_n >= SPEECH_FAIL_HINT_N:
            if not self._hint_lbl.winfo_ismapped():
                self._hint_lbl.pack(side="left", padx=(4, 8))
        else:
            if self._hint_lbl.winfo_ismapped():
                self._hint_lbl.pack_forget()

        # Update cached bounds
        try:
            self._root.update_idletasks()
            self._bounds = (
                self._root.winfo_x(),
                self._root.winfo_y(),
                self._root.winfo_width(),
                self._root.winfo_height(),
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Visibility toggle (must be called on the tk thread)
    # ------------------------------------------------------------------

    def toggle_visibility(self) -> None:
        if self._root is None:
            return
        self._visible = not self._visible
        if self._visible:
            self._root.deiconify()
            if self._hwnd:
                _assert_topmost(self._hwnd)
        else:
            self._root.withdraw()

    # ------------------------------------------------------------------
    # Destroy
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except Exception:
                pass
            self._root = None

    # ------------------------------------------------------------------

    @property
    def bounds(self) -> Optional[Tuple[int, int, int, int]]:
        return self._bounds


# ---------------------------------------------------------------------------
# OverlayThread — owns the tkinter mainloop
# ---------------------------------------------------------------------------

class OverlayThread(threading.Thread):
    """
    Daemon thread for the overlay.
    Receives state dicts from the shared queue and applies them to _OverlayWindow.
    """

    _SENTINEL = object()

    def __init__(self, q: "queue.Queue[object]") -> None:
        super().__init__(name="OverlayThread", daemon=True)
        self._q             = q
        self._window: Optional[_OverlayWindow] = None
        self._bounds_lock   = threading.Lock()
        self._bounds: Optional[Tuple[int, int, int, int]] = None
        self._hotkey        = None

    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("Overlay started (position=%s opacity=%.2f)", OVERLAY_POSITION, OVERLAY_OPACITY)
        win = _OverlayWindow()
        if not win.create():
            logger.error("OverlayWindow creation failed; overlay disabled")
            return

        self._window = win
        logger.info("Overlay window created and visible.")
        self._start_hotkey(win)

        # Schedule the queue-drain loop inside the tk event loop
        win._root.after(QUEUE_DRAIN_MS, self._drain_queue)

        try:
            win._root.mainloop()
        except Exception:
            logger.exception("Overlay mainloop error")
        finally:
            self._stop_hotkey()

    # ------------------------------------------------------------------
    # Queue drainer (runs on tk thread via after())
    # ------------------------------------------------------------------

    def _drain_queue(self) -> None:
        win = self._window
        if win is None or win._root is None:
            return
        try:
            # Drain ALL pending items; only the last state matters for display
            last_state: Optional[dict] = None
            while True:
                item = self._q.get_nowait()
                if item is self._SENTINEL:
                    win.destroy()
                    return
                if isinstance(item, dict):
                    last_state = item
            # unreachable
        except queue.Empty:
            pass
        except Exception:
            logger.exception("Overlay drain error")

        if last_state is not None:
            try:
                win.apply_state(last_state)
            except Exception:
                logger.exception("Overlay apply_state error")
            with self._bounds_lock:
                self._bounds = win.bounds

        # Re-schedule
        if win._root:
            win._root.after(QUEUE_DRAIN_MS, self._drain_queue)

    # ------------------------------------------------------------------
    # pynput hotkey for Ctrl+Alt+H
    # ------------------------------------------------------------------

    def _start_hotkey(self, win: _OverlayWindow) -> None:
        try:
            from pynput import keyboard  # type: ignore[import]

            def _toggle():
                if win._root:
                    win._root.after(0, win.toggle_visibility)

            self._hotkey = keyboard.GlobalHotKeys({OVERLAY_HOTKEY: _toggle})
            self._hotkey.daemon = True
            self._hotkey.start()
        except ImportError:
            logger.info("pynput not available; overlay hotkey Ctrl+Alt+H disabled")
        except Exception:
            logger.warning("Overlay hotkey registration failed", exc_info=True)

    def _stop_hotkey(self) -> None:
        if self._hotkey is not None:
            try:
                self._hotkey.stop()
            except Exception:
                pass
            self._hotkey = None

    # ------------------------------------------------------------------

    def get_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        with self._bounds_lock:
            return self._bounds

    def request_shutdown(self) -> None:
        try:
            self._q.put_nowait(self._SENTINEL)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OverlayManager:
    """
    Thread-safe facade for the overlay subsystem.

    Usage
    -----
        mgr = OverlayManager()
        mgr.start()
        mgr.update({"mode": "FACE", "face_ok": True, "mic_ok": True,
                     "last_cmd": "click", "last_ok": True,
                     "paused": False, "speech_fail_count": 0})
        mgr.stop()
    """

    _QUEUE_MAXSIZE = 8  # large enough that rapid updates are coalesced, not lost

    def __init__(self) -> None:
        self._enabled = OVERLAY_ENABLED
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._thread: Optional[OverlayThread] = None

    def start(self) -> None:
        """Start the overlay daemon thread.  No-op if OVERLAY_ENABLED=False."""
        if not self._enabled:
            logger.info("Overlay disabled via config; skipping start")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = OverlayThread(self._q)
        self._thread.start()
        logger.info("OverlayThread started (position=%s)", OVERLAY_POSITION)

    def update(self, state: dict) -> None:
        """
        Non-blocking state push.  If the queue is full, the oldest item is
        dropped to make room (never stalls the main loop).
        """
        if not self._enabled or self._thread is None:
            return
        try:
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put_nowait(state)
        except Exception:
            pass

    def stop(self) -> None:
        """Signal the overlay thread to exit, then join with a short timeout."""
        if self._thread is None:
            return
        try:
            self._thread.request_shutdown()
            self._thread.join(timeout=2.0)
        except Exception:
            logger.exception("Error stopping OverlayThread")
        finally:
            self._thread = None

    def overlay_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Return (x, y, width, height) of the overlay in screen coords,
        or None when the overlay is not running.
        Use in Router/Controller to exclude this rectangle from face-click targeting.
        """
        if self._thread is None or not self._thread.is_alive():
            return None
        return self._thread.get_bounds()


# ---------------------------------------------------------------------------
# __main__ — self-test mode:  python -m output.overlay --test
# ---------------------------------------------------------------------------

def _run_test() -> None:
    """
    Cycle through all modes and PAUSED every second for 10 cycles then exit.
    Run with:   python -m output.overlay --test
    """
    import time as _time

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
    _log = logging.getLogger("assistive.overlay.test")
    _log.info("Overlay self-test starting — cycling all modes (1 s each)")

    mgr = OverlayManager()
    mgr.start()

    # Give the tk thread a moment to open the window
    _time.sleep(0.5)

    test_sequence = [
        {"mode": "BOTH",   "face_ok": True,  "mic_ok": True,  "paused": False, "last_cmd": "scroll up",  "last_ok": True,  "speech_fail_count": 0},
        {"mode": "SPEECH", "face_ok": False, "mic_ok": True,  "paused": False, "last_cmd": "click",       "last_ok": True,  "speech_fail_count": 0},
        {"mode": "FACE",   "face_ok": True,  "mic_ok": False, "paused": False, "last_cmd": "right_click", "last_ok": False, "speech_fail_count": 0},
        {"mode": "BOTH",   "face_ok": True,  "mic_ok": True,  "paused": False, "last_cmd": "dictate",     "last_ok": True,  "speech_fail_count": 0},
        {"mode": "PAUSED", "face_ok": False, "mic_ok": False, "paused": True,  "last_cmd": "",            "last_ok": True,  "speech_fail_count": 0},
        {"mode": "FACE",   "face_ok": True,  "mic_ok": False, "paused": False, "last_cmd": "scroll down", "last_ok": True,  "speech_fail_count": 1},
        {"mode": "SPEECH", "face_ok": False, "mic_ok": True,  "paused": False, "last_cmd": "open browser","last_ok": True,  "speech_fail_count": 2},
        {"mode": "BOTH",   "face_ok": True,  "mic_ok": True,  "paused": False, "last_cmd": "zoom in",     "last_ok": True,  "speech_fail_count": 0},
        {"mode": "PAUSED", "face_ok": False, "mic_ok": False, "paused": True,  "last_cmd": "",            "last_ok": True,  "speech_fail_count": 0},
        {"mode": "BOTH",   "face_ok": True,  "mic_ok": True,  "paused": False, "last_cmd": "auto",        "last_ok": True,  "speech_fail_count": 0},
    ]

    for i, state in enumerate(test_sequence):
        _log.info(
            "[test %d/%d] mode=%-6s paused=%s last_cmd=%s",
            i + 1, len(test_sequence),
            state["mode"], state["paused"], state.get("last_cmd", ""),
        )
        mgr.update(state)
        _time.sleep(1.0)

    _log.info("Test complete — badge should have cycled BOTH→SPEECH→FACE→BOTH→PAUSED→… Press Ctrl+C to exit.")
    try:
        while True:
            _time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        mgr.stop()
        _log.info("Overlay stopped.")


if __name__ == "__main__":
    import sys as _sys
    if "--test" in _sys.argv:
        _run_test()
    else:
        print("Usage: python -m output.overlay --test")
        print("  Cycles all badge modes every second so you can verify the on-screen pill.")
        _sys.exit(0)
