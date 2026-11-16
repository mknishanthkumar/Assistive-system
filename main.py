"""
main.py – Stage 4 integration entry point for the Assistive Control System.

Runs FaceTracker and SpeechEngine together in a fully threaded architecture:

  Thread 1 (Face loop)   : reads FaceTracker events → Router → OutputController
  Thread 2 (Speech loop) : reads SpeechEngine events → Router → OutputController
  Thread 3 (Sensor loop) : SensorLoop polls context (window, noise, brightness)
  Main thread            : runs pynput GlobalHotKeys + waits for shutdown signal

Shutdown sequence (Ctrl+C, emergency stop, "stop control" voice command):
  1. stop_event is set
  2. queue sentinels sent to face/speech loops
  3. FaceTracker.stop() → releases camera
  4. SpeechEngine.stop() → releases microphone
  5. SensorLoop.stop()
  6. Worker threads joined (5 s timeout each)
  7. CSV logger flushed and closed
  8. All logs flushed
"""

import logging
import queue
import signal
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pyautogui

import config
from face.face_tracker import FaceTracker
from speech.speech_engine import SpeechEngine
from output.controller import OutputController
from output.router import Router
from context.sensors import SensorLoop
from logs.csv_logger import CsvLogger


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    """Configure rotating file + console logging."""
    root = logging.getLogger("assistive")
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    log_file = config.LOG_DIR / "assistive.log"
    file_h = RotatingFileHandler(
        log_file,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_h.setFormatter(fmt)
    root.addHandler(file_h)

    return root


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner(dry_run: bool) -> None:
    width = 70
    print()
    print("=" * width)
    print("  Assistive Control System  —  Stage 4 Integration")
    print("=" * width)
    if dry_run:
        print("  MODE : DRY-RUN  (actions logged, NOT executed)")
    else:
        print("  MODE : REAL ACTIONS  (mouse, keyboard, browser active)")
    print(f"  Emergency stop : {config.EMERGENCY_STOP_HOTKEY.upper()}")
    print(f"  PyAutoGUI fail-safe : move mouse to any screen corner")
    print(f"  Confidence threshold : {config.SPEECH_CONFIDENCE_THRESHOLD}")
    print()
    print("  Input sources  : Face (MediaPipe) + Speech (Faster-Whisper)")
    print("  Context sensor : active window, noise level, brightness")
    print("  CSV log        : logs/runs/run_<timestamp>.csv")
    print()
    print("  Press Ctrl+C or say 'stop control' to exit.")
    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Worker: face event loop
# ---------------------------------------------------------------------------

def _face_loop(
    tracker: FaceTracker,
    router: Router,
    csv_logger: CsvLogger,
    stop_event: threading.Event,
    face_queue: "queue.Queue",
) -> None:
    """
    Drain face events at ~30 Hz, forward to Router.
    Uses a queue sentinel (None) for clean shutdown.
    """
    logger = logging.getLogger("assistive.face_loop")
    logger.info("Face loop started.")

    while not stop_event.is_set():
        # Non-blocking queue drain (sentinel = None → exit)
        try:
            item = face_queue.get(timeout=0.05)
            if item is None:
                break
        except queue.Empty:
            pass

        # Pull latest events directly from tracker at ~30 Hz
        try:
            events = tracker.get_latest_events()
        except Exception as exc:
            logger.warning("Face event error: %s", exc)
            csv_logger.log("error", {"message": f"Face event error: {exc}"})
            continue

        # Log face reading
        csv_logger.log("face", {
            "face_detected": events.get("face_detected", False),
            "face_confidence": events.get("face_confidence", 0.0),
            "cursor_delta_x": events.get("cursor_delta_x", 0.0),
            "cursor_delta_y": events.get("cursor_delta_y", 0.0),
            "left_click": events.get("left_click", False),
            "right_click": events.get("right_click", False),
        })

        # Forward to router
        try:
            router.handle_face_events(events)
        except Exception as exc:
            logger.warning("Router face error: %s", exc)
            csv_logger.log("error", {"message": f"Router face error: {exc}"})

        time.sleep(0.033)   # ~30 Hz

    logger.info("Face loop exited.")


# ---------------------------------------------------------------------------
# Worker: speech event loop
# ---------------------------------------------------------------------------

def _speech_loop(
    engine: SpeechEngine,
    router: Router,
    csv_logger: CsvLogger,
    stop_event: threading.Event,
    speech_queue: "queue.Queue",
) -> None:
    """
    Poll speech events at 20 Hz, forward to Router.
    Uses a queue sentinel (None) for clean shutdown.
    """
    logger = logging.getLogger("assistive.speech_loop")
    logger.info("Speech loop started.")

    last_event_id: int = 0

    while not stop_event.is_set():
        try:
            item = speech_queue.get(timeout=0.05)
            if item is None:
                break
        except queue.Empty:
            pass

        try:
            events = engine.get_latest_events()
        except Exception as exc:
            logger.warning("Speech event error: %s", exc)
            csv_logger.log("error", {"message": f"Speech event error: {exc}"})
            continue

        eid = events.get("event_id", 0)
        if eid > last_event_id:
            last_event_id = eid

            # Log speech reading
            csv_logger.log("speech", {
                "command": events.get("command"),
                "command_phrase": events.get("command_phrase"),
                "speech_confidence": events.get("speech_confidence", 0.0),
                "speech_detected": events.get("speech_detected", False),
                "dictation_text": events.get("dictation_text"),
                "noise_level": events.get("noise_level", 0.0),
                "vad_status": events.get("vad_status", "silence"),
            })

            # Forward to router
            try:
                router.handle_speech_events(events)
            except Exception as exc:
                logger.warning("Router speech error: %s", exc)
                csv_logger.log("error", {"message": f"Router speech error: {exc}"})

        time.sleep(0.05)    # ~20 Hz

    logger.info("Speech loop exited.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = config.DRY_RUN) -> None:
    """Application entry point."""
    logger = _setup_logging()
    logger.info("Assistive Control System starting — Stage 4 Integration")
    logger.info("Python %s on %s", sys.version.split()[0], sys.platform)

    _print_banner(dry_run)

    # ── PyAutoGUI safety ──────────────────────────────────────────────
    pyautogui.FAILSAFE = config.PYAUTOGUI_FAILSAFE
    pyautogui.PAUSE    = config.PYAUTOGUI_PAUSE

    # ── CSV logger ────────────────────────────────────────────────────
    csv_logger = CsvLogger()
    csv_logger.log("start", {"message": f"Stage 4 start (dry_run={dry_run})"})
    logger.info("CSV log: %s", csv_logger.path)

    # ── OutputController ──────────────────────────────────────────────
    controller = OutputController(dry_run=dry_run)

    # ── Router ────────────────────────────────────────────────────────
    router = Router(controller, csv_logger=csv_logger)

    # ── FaceTracker ───────────────────────────────────────────────────
    tracker = FaceTracker()
    face_started = False
    try:
        tracker.start()
        face_started = True
        logger.info("FaceTracker started.")
    except Exception as exc:
        logger.warning("FaceTracker could not start (continuing without face input): %s", exc)
        csv_logger.log("error", {"message": f"FaceTracker start failed: {exc}"})

    # ── SpeechEngine ──────────────────────────────────────────────────
    engine = SpeechEngine()
    speech_started = False
    try:
        engine.start()
        speech_started = True
        logger.info("SpeechEngine started.")
    except Exception as exc:
        logger.warning("SpeechEngine could not start (continuing without speech input): %s", exc)
        csv_logger.log("error", {"message": f"SpeechEngine start failed: {exc}"})

    # ── Shared stop event + sentinel queues ───────────────────────────
    stop_event = threading.Event()
    face_q:   "queue.Queue" = queue.Queue(maxsize=1)
    speech_q: "queue.Queue" = queue.Queue(maxsize=1)

    # ── Sensor loop ───────────────────────────────────────────────────
    sensor_q: "queue.Queue" = queue.Queue(maxsize=10)
    sensor_loop = SensorLoop(
        sensor_queue=sensor_q,
        face_tracker=tracker if face_started else None,
        speech_engine=engine if speech_started else None,
    )
    sensor_loop.start()

    # ── Worker threads ────────────────────────────────────────────────
    face_thread = threading.Thread(
        target=_face_loop,
        args=(tracker, router, csv_logger, stop_event, face_q),
        name="FaceEventLoop",
        daemon=True,
    )
    speech_thread = threading.Thread(
        target=_speech_loop,
        args=(engine, router, csv_logger, stop_event, speech_q),
        name="SpeechEventLoop",
        daemon=True,
    )

    face_thread.start()
    speech_thread.start()

    # ── Emergency stop hotkey (pynput GlobalHotKeys) ─────────────────
    hotkey_listener = None
    try:
        from pynput.keyboard import GlobalHotKeys

        def _on_emergency() -> None:
            controller.trigger_emergency_stop()
            csv_logger.log("action", {
                "action_command": "emergency_stop",
                "action_success": True,
                "message": "Emergency stop via hotkey.",
            })
            print()
            print("  !!! EMERGENCY STOP ACTIVATED !!!")
            print("  Press Ctrl+C to exit cleanly.")
            print()
            # Also trigger full shutdown
            stop_event.set()

        hotkey_listener = GlobalHotKeys({config.EMERGENCY_STOP_HOTKEY_PYNPUT: _on_emergency})
        hotkey_listener.daemon = True
        hotkey_listener.start()
        logger.info("Emergency hotkey registered: %s", config.EMERGENCY_STOP_HOTKEY)
    except Exception as exc:
        logger.warning("Could not register emergency hotkey: %s", exc)

    # ── Sensor drain thread ───────────────────────────────────────────
    def _sensor_drain() -> None:
        """Log sensor readings from the queue to CSV."""
        log = logging.getLogger("assistive.sensor_drain")
        while not stop_event.is_set():
            try:
                reading = sensor_q.get(timeout=0.5)
                csv_logger.log("sensor", {
                    "sensor_noise_level": reading.get("noise_level", 0.0),
                    "sensor_face_confidence": reading.get("face_confidence", 0.0),
                    "sensor_brightness": reading.get("frame_brightness", 0.0),
                    "active_window": reading.get("active_window", "unknown"),
                    "active_process": reading.get("active_process", "unknown"),
                    "text_field_focused": reading.get("text_field_focused", "unknown"),
                    "message": "sensor_reading",
                })
            except queue.Empty:
                continue
            except Exception as exc:
                log.debug("Sensor drain error: %s", exc)

    sensor_drain_thread = threading.Thread(
        target=_sensor_drain,
        name="SensorDrain",
        daemon=True,
    )
    sensor_drain_thread.start()

    logger.info(
        "All subsystems running. Mode: %s. Say 'stop control' or press %s to stop.",
        "DRY-RUN" if dry_run else "REAL ACTIONS",
        config.EMERGENCY_STOP_HOTKEY.upper(),
    )

    # ── Main loop: wait for shutdown ──────────────────────────────────
    try:
        while not stop_event.is_set():
            # Check if "stop control" was received (controller stopped by router)
            if controller.is_stopped() and not stop_event.is_set():
                logger.info("Controller stopped — initiating shutdown.")
                stop_event.set()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print()
        logger.info("Ctrl+C received — shutting down.")

    finally:
        _shutdown(
            stop_event=stop_event,
            face_q=face_q,
            speech_q=speech_q,
            tracker=tracker if face_started else None,
            engine=engine if speech_started else None,
            sensor_loop=sensor_loop,
            face_thread=face_thread,
            speech_thread=speech_thread,
            hotkey_listener=hotkey_listener,
            csv_logger=csv_logger,
        )


def _shutdown(
    stop_event: threading.Event,
    face_q: "queue.Queue",
    speech_q: "queue.Queue",
    tracker,
    engine,
    sensor_loop: SensorLoop,
    face_thread: threading.Thread,
    speech_thread: threading.Thread,
    hotkey_listener,
    csv_logger: CsvLogger,
) -> None:
    """Orderly shutdown: signal threads, release hardware, flush logs."""
    logger = logging.getLogger("assistive.shutdown")
    logger.info("Shutdown sequence starting.")

    stop_event.set()

    # Send sentinels to unblock worker loops
    for q in (face_q, speech_q):
        try:
            q.put_nowait(None)
        except queue.Full:
            pass

    # Stop hardware
    if tracker is not None:
        try:
            tracker.stop()
            logger.info("FaceTracker stopped (camera released).")
        except Exception as exc:
            logger.warning("FaceTracker stop error: %s", exc)

    if engine is not None:
        try:
            engine.stop()
            logger.info("SpeechEngine stopped (microphone released).")
        except Exception as exc:
            logger.warning("SpeechEngine stop error: %s", exc)

    # Stop sensor loop
    try:
        sensor_loop.stop()
    except Exception:
        pass

    # Join worker threads
    for t in (face_thread, speech_thread):
        t.join(timeout=5.0)
        if t.is_alive():
            logger.warning("Thread %s did not exit within 5 s.", t.name)

    # Stop hotkey listener
    if hotkey_listener is not None:
        try:
            hotkey_listener.stop()
        except Exception:
            pass

    csv_logger.log("stop", {"message": "Assistive system shut down cleanly."})
    csv_logger.close()

    # Flush all logging handlers
    for h in logging.getLogger("assistive").handlers:
        try:
            h.flush()
        except Exception:
            pass

    logger.info("Shutdown complete.")
    print("\n  Assistive Control System stopped.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Assistive Control System — Stage 4")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=config.DRY_RUN,
        help="Log actions but do not execute them (overrides config.DRY_RUN).",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
