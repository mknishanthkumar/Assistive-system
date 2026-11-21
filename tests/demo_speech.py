"""
tests/demo_speech.py - Standalone speech-control demo (Stage 3).

This demo listens for voice commands via SpeechEngine AND executes them as
real Windows actions (mouse clicks, key presses, browser launch, volume
control, dictation typing) via OutputController.

HOW TO RUN
----------
  .\.venv\Scripts\Activate.ps1
  python tests\demo_speech.py              # REAL actions
  python tests\demo_speech.py --dry-run    # log-only, no real actions

CONTROLS
--------
  Say voice commands     : Recognised commands execute real Windows actions.
  Say "start dictation"  : Activates dictation mode (text typed into focused field).
  Say "stop dictation"   : Deactivates dictation mode.
  Ctrl+Alt+Q             : Emergency stop — disables all actions immediately.
  Ctrl+C                 : Stop the demo and release microphone + hotkey listener.

SAFETY
------
- PyAutoGUI fail-safe is ON (move mouse to any screen corner to abort).
- Emergency hotkey Ctrl+Alt+Q disables action execution instantly.
- Confidence threshold: commands below SPEECH_CONFIDENCE_THRESHOLD are rejected.
- Per-command cooldown prevents duplicate actions from a single utterance.
- No destructive actions (shutdown, restart, delete, auto-submit).

NOTES
-----
- The Faster-Whisper model is downloaded on first run (~150 MB).
- Subsequent runs load from cache (~2-5 s).
- Use --dry-run to test recognition without executing real actions.
"""

import argparse
import sys
import time
import logging
import pathlib
import threading

# Ensure project root is on path so local imports work
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pyautogui
import config
from speech.speech_engine import SpeechEngine
from output.controller import OutputController

# ── PyAutoGUI safety ──────────────────────────────────────────────────────
pyautogui.FAILSAFE = config.PYAUTOGUI_FAILSAFE
pyautogui.PAUSE = config.PYAUTOGUI_PAUSE

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("assistive.demo_speech")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speech-control demo with real Windows actions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions but do not execute them.",
    )
    return parser.parse_args()


def _print_banner(dry_run: bool) -> None:
    print()
    print("=" * 68)
    print("  Assistive System - Speech Control Demo (Stage 3)")
    print("=" * 68)
    print()
    if dry_run:
        print("  MODE: DRY-RUN (actions are logged but NOT executed)")
    else:
        print("  *** REAL ACTIONS ENABLED ***")
        print("  Commands will move the mouse, press keys, and open apps.")
    print()
    print("  Safety:")
    print(f"    - Emergency stop: {config.EMERGENCY_STOP_HOTKEY.upper()}")
    print("    - PyAutoGUI fail-safe: move mouse to any screen corner")
    print(f"    - Confidence threshold: {config.SPEECH_CONFIDENCE_THRESHOLD}")
    print(f"    - Command cooldown: {config.SPEECH_COMMAND_COOLDOWN_S}s")
    print()
    print("  Controls:")
    print("    - Say any supported command (e.g. 'click', 'scroll up')")
    print("    - Say 'hold scroll down' / 'hold scroll up' (say 'stop' or 'release' to halt)")
    print("    - Say 'arrange center' to re-center head tracking")
    print("    - Say 'start dictation' to type spoken text")
    print("    - Say 'stop dictation' to return to command mode")
    print("    - Press Ctrl+C to stop the demo")
    print()
    print("  Loading Whisper model (first run downloads ~150 MB) ...")
    print("=" * 68)
    print()


def _setup_emergency_hotkey(controller: OutputController) -> threading.Thread:
    """
    Register the global emergency-stop hotkey (Ctrl+Alt+Q) using pynput.
    Returns the listener thread (daemon).
    """
    from pynput.keyboard import GlobalHotKeys

    hotkey_str = "<ctrl>+<alt>+q"

    def on_emergency():
        controller.trigger_emergency_stop()
        print()
        print("  !!! EMERGENCY STOP ACTIVATED !!!")
        print("  All actions are now disabled. Press Ctrl+C to exit.")
        print()

    listener = GlobalHotKeys({hotkey_str: on_emergency})
    listener.daemon = True
    listener.start()
    logger.info("Emergency hotkey registered: %s", config.EMERGENCY_STOP_HOTKEY)
    return listener


def main() -> None:
    args = _parse_args()
    _print_banner(args.dry_run)

    # ── Start SpeechEngine ────────────────────────────────────────────
    engine = SpeechEngine()
    try:
        engine.start()
    except Exception as e:
        logger.error("Failed to start SpeechEngine: %s", e)
        sys.exit(1)

    # ── Start OutputController ────────────────────────────────────────
    controller = OutputController(dry_run=args.dry_run)

    # ── Emergency hotkey listener ─────────────────────────────────────
    hotkey_listener = _setup_emergency_hotkey(controller)

    logger.info(
        "Speech demo running (%s).  Speak into the microphone.  Ctrl+C to stop.",
        "DRY-RUN" if args.dry_run else "REAL ACTIONS",
    )

    last_event_id: int = 0
    last_status_time: float = time.monotonic()
    STATUS_INTERVAL: float = 2.0  # seconds between status-line prints

    try:
        while True:
            events = engine.get_latest_events()
            eid = events["event_id"]

            # ── Process new events (once only per event_id) ───────────
            if eid > last_event_id:
                last_event_id = eid
                cmd = events.get("command")
                conf = events.get("speech_confidence", 0.0)
                dictation = events.get("dictation_text")

                # ── Command event ─────────────────────────────────────
                if cmd:
                    phrase = events.get("command_phrase", "")
                    print(
                        f"  >> COMMAND: {cmd}  "
                        f"(phrase='{phrase}', confidence={conf:.2f})"
                    )
                    # Execute the action
                    if cmd == "hold_scroll_down":
                        controller.hold_scroll.start(direction=-1)
                    elif cmd == "hold_scroll_up":
                        controller.hold_scroll.start(direction=1)
                    elif cmd == "stop_scroll":
                        controller.hold_scroll.stop()

                    result = controller.execute(cmd, confidence=conf)
                    if result.success:
                        print(f"  [ACTION] {result.message}")
                    else:
                        print(f"  [REJECTED] {result.message}")

                # ── Dictation event ───────────────────────────────────
                elif dictation:
                    print(f"  >> DICTATION: {dictation}")
                    result = controller.type_dictation(dictation)
                    if result.success:
                        print(f"  [TYPED] {result.message}")
                    else:
                        print(f"  [TYPE-ERROR] {result.message}")

                # ── Unmatched transcription ───────────────────────────
                else:
                    raw = events.get("transcription_text")
                    if raw:
                        print(f"  >> (unmatched): {raw}")

            # ── Periodic status line ──────────────────────────────────
            now = time.monotonic()
            if now - last_status_time >= STATUS_INTERVAL:
                last_status_time = now
                noise = events.get("noise_level", 0.0)
                conf = events.get("speech_confidence", 0.0)
                speech = events.get("speech_detected", False)
                vad = events.get("vad_status", "silence")
                dictation_on = events.get("dictation_mode", False)
                stopped = controller.is_stopped()
                print(
                    f"  [STATUS] noise={noise:.4f}  "
                    f"conf={conf:.2f}  "
                    f"speech={'YES' if speech else 'no '}  "
                    f"vad={vad:<11s}  "
                    f"dictation={'ON' if dictation_on else 'off'}  "
                    f"actions={'STOPPED' if stopped else 'active'}"
                )

            time.sleep(0.05)  # ~20 Hz poll

    except KeyboardInterrupt:
        print()
        logger.info("Ctrl+C received - stopping demo.")
    finally:
        # Clean shutdown: stop engine, then hotkey listener
        engine.stop()
        try:
            hotkey_listener.stop()
        except Exception:
            pass
        logger.info("Speech demo exited cleanly.")


if __name__ == "__main__":
    main()
