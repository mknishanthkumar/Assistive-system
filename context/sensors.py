"""
context/sensors.py - Context sensor loop for the Assistive Control System.

Publishes environment readings to a thread-safe queue every
config.SENSOR_POLL_INTERVAL_S seconds.  Each reading is a dict with:

    noise_level       float [0.0, 1.0]  – rolling RMS from SpeechEngine
    face_confidence   float [0.0, 1.0]  – from FaceTracker
    frame_brightness  float [0.0, 255]  – mean pixel brightness from FaceTracker
    active_window     str               – foreground window title
    active_process    str               – process name of foreground window
    text_field_focused bool | "unknown" – whether a text input is focused

Design constraints
------------------
- Never crashes or blocks the caller.  All failure modes return "unknown".
- Sensor readings are PUBLISHED (put into the queue) and LOGGED only.
  No decisions are made here; the context engine will read and decide.
- Uses uiautomation / pywin32 for window info.  If neither is available,
  returns "unknown" for those fields.

Usage
-----
    from context.sensors import SensorLoop
    import queue

    q = queue.Queue(maxsize=10)
    loop = SensorLoop(sensor_queue=q, face_tracker=tracker, speech_engine=engine)
    loop.start()
    ...
    reading = q.get()   # {"noise_level": 0.02, "active_window": "Notepad", ...}
    ...
    loop.stop()
"""

import logging
import queue
import threading
import time
from typing import Any, Dict, Optional, Union

import config

logger = logging.getLogger("assistive.sensors")


# ---------------------------------------------------------------------------
# Active-window helpers (uiautomation / pywin32 / fallback)
# ---------------------------------------------------------------------------

def _get_active_window_info() -> Dict[str, str]:
    """
    Return {"title": str, "process": str} for the foreground window.

    Tries uiautomation first (richer info), then pywin32, then returns
    "unknown" if neither package is importable or the call fails.
    """
    result = {"title": "unknown", "process": "unknown"}
    try:
        import uiautomation as auto  # type: ignore
        # GetForegroundControl can raise if no window is focused
        ctrl = auto.GetForegroundControl()
        if ctrl:
            result["title"] = ctrl.Name or "unknown"
            # uiautomation exposes ProcessId; map to process name via psutil
            try:
                import psutil  # type: ignore
                proc = psutil.Process(ctrl.ProcessId)
                result["process"] = proc.name()
            except Exception:
                result["process"] = str(getattr(ctrl, "ProcessId", "unknown"))
        return result
    except Exception:
        pass

    try:
        import win32gui   # type: ignore
        import win32process  # type: ignore
        import psutil  # type: ignore
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd) or "unknown"
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            proc_name = psutil.Process(pid).name()
        except Exception:
            proc_name = str(pid)
        result["title"] = title
        result["process"] = proc_name
        return result
    except Exception:
        pass

    return result


def _is_text_field_focused() -> Union[bool, str]:
    """
    Return True if the currently focused UI element is a text-entry field,
    False if it is not, or "unknown" if detection fails.

    Uses uiautomation ControlType / ClassName heuristics.
    """
    try:
        import uiautomation as auto  # type: ignore
        ctrl = auto.GetFocusedControl()
        if ctrl is None:
            return "unknown"
        # uiautomation control type IDs
        TEXT_INPUT_TYPES = {
            auto.ControlType.EditControl,
            auto.ControlType.DocumentControl,
        }
        if hasattr(ctrl, "ControlType") and ctrl.ControlType in TEXT_INPUT_TYPES:
            return True
        # Fallback: class name heuristics used by many Win32 / WPF apps
        class_name = getattr(ctrl, "ClassName", "") or ""
        text_classes = {"Edit", "RichEdit", "RichEdit20W", "TextBox", "RICHEDIT50W"}
        if any(tc in class_name for tc in text_classes):
            return True
        return False
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# SensorLoop
# ---------------------------------------------------------------------------

class SensorLoop:
    """
    Periodically collects sensor readings and publishes them to a queue.

    Parameters
    ----------
    sensor_queue : queue.Queue
        Destination queue.  Readings are put() non-blocking (dropped if full).
    face_tracker : FaceTracker | None
        Live FaceTracker instance for face_confidence and brightness.
    speech_engine : SpeechEngine | None
        Live SpeechEngine instance for noise_level.
    poll_interval : float
        Seconds between readings (default: config.SENSOR_POLL_INTERVAL_S).
    """

    def __init__(
        self,
        sensor_queue: "queue.Queue[Dict[str, Any]]",
        face_tracker: Optional[Any] = None,
        speech_engine: Optional[Any] = None,
        poll_interval: float = config.SENSOR_POLL_INTERVAL_S,
    ) -> None:
        self._queue = sensor_queue
        self._face = face_tracker
        self._speech = speech_engine
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sensor polling thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("SensorLoop already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="SensorLoop",
            daemon=True,
        )
        self._thread.start()
        logger.info("SensorLoop started (interval=%.1fs).", self._poll_interval)

    def stop(self) -> None:
        """Signal the loop to stop and wait for it to exit (up to 3 s)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("SensorLoop stopped.")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main polling loop — runs in background thread."""
        while not self._stop_event.is_set():
            reading = self._collect()
            try:
                self._queue.put_nowait(reading)
            except queue.Full:
                pass   # drop if consumer is slow; never block
            logger.debug(
                "Sensor: noise=%.4f face_conf=%.2f brightness=%.1f "
                "window=%r process=%r text_field=%s",
                reading["noise_level"],
                reading["face_confidence"],
                reading["frame_brightness"],
                reading["active_window"],
                reading["active_process"],
                reading["text_field_focused"],
            )
            self._stop_event.wait(self._poll_interval)

    def _collect(self) -> Dict[str, Any]:
        """
        Gather one sensor snapshot.

        Each field degrades gracefully to a safe default on failure.
        """
        # ── Noise level from SpeechEngine ─────────────────────────────
        noise_level: float = 0.0
        try:
            if self._speech is not None:
                evts = self._speech.get_latest_events()
                noise_level = float(evts.get("noise_level", 0.0))
        except Exception:
            noise_level = 0.0

        # ── Face confidence + brightness from FaceTracker ─────────────
        face_confidence: float = 0.0
        frame_brightness: float = 0.0
        try:
            if self._face is not None:
                evts = self._face.get_latest_events()
                face_confidence = float(evts.get("face_confidence", 0.0))
                dbg = evts.get("debug", {})
                frame_brightness = float(dbg.get("brightness", 0.0))
        except Exception:
            pass

        # ── Active window (best-effort, never raises) ──────────────────
        try:
            win_info = _get_active_window_info()
        except Exception:
            win_info = {"title": "unknown", "process": "unknown"}

        # ── Text-field focus (best-effort) ─────────────────────────────
        try:
            text_focused = _is_text_field_focused()
        except Exception:
            text_focused = "unknown"

        return {
            "noise_level": noise_level,
            "face_confidence": face_confidence,
            "frame_brightness": frame_brightness,
            "active_window": win_info.get("title", "unknown"),
            "active_process": win_info.get("process", "unknown"),
            "text_field_focused": text_focused,
            "timestamp": time.monotonic(),
        }

