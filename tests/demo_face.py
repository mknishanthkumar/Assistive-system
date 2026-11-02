"""
tests/demo_face.py - Standalone face-tracking + cursor control demo (Stage 2).

SAFETY NOTICE
-------------
This script moves the REAL Windows mouse cursor and executes real mouse clicks.
* Keep your hand near the mouse to regain control at any time.
* PyAutoGUI fail-safe: quickly move the mouse to any corner of the screen
  to raise an exception and stop the demo.
* Press ESC in the preview window to exit gracefully.
* Emergency stop: Ctrl+Alt+Q (global hotkey).

HOW TO RUN
----------
  .\.venv\Scripts\Activate.ps1
  python tests\demo_face.py

CONTROLS
--------
  LOOK AT CAMERA  : Move head to move cursor (centre head to calibrate)
  C               : Re-calibrate neutral head position
  ESC             : Exit demo
  Deliberate long blink (hold ~350 ms)  : Left-click
  Jaw open (hold ~300 ms)               : Right-click
"""

import sys
import time
import logging
import pathlib

# Ensure project root is on path so local imports work
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pyautogui

import config
from face.face_tracker import FaceTracker

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("assistive.demo_face")

# ── PyAutoGUI safety ───────────────────────────────────────────────────────
pyautogui.FAILSAFE = config.PYAUTOGUI_FAILSAFE
pyautogui.PAUSE = config.PYAUTOGUI_PAUSE


def _print_banner() -> None:
    print()
    print("=" * 60)
    print("  Assistive System - Face Tracking Demo (Stage 2)")
    print("=" * 60)
    print()
    print("  SAFETY: This demo controls the REAL mouse cursor.")
    print("  * Move mouse to any screen corner to abort (fail-safe).")
    print("  * Press ESC in the preview window to exit.")
    print("  * Press C to re-calibrate neutral head position.")
    print()
    print("  Starting in 3 seconds ...")
    print("=" * 60)
    print()
    time.sleep(3)


def _draw_overlay(
    frame: np.ndarray,
    events: dict,
    screen_w: int,
    screen_h: int,
    cursor_x: int,
    cursor_y: int,
) -> np.ndarray:
    """Draw HUD overlay on the preview frame (does not modify original)."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Semi-transparent dark bar at top
    bar = overlay[0:80, :].copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay[0:80, :], 0.6, bar, 0.4, 0, overlay[0:80, :])

    def _txt(text: str, y: int, colour: tuple = (255, 255, 255)) -> None:
        cv2.putText(overlay, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 1, cv2.LINE_AA)

    conf = events["face_confidence"]
    conf_colour = (
        (0, 200, 80)   if conf >= 0.7 else
        (0, 180, 255)  if conf >= 0.4 else
        (0, 60, 255)
    )

    face_str = "FACE: YES" if events["face_detected"] else "FACE: NO "
    _txt(f"{face_str}   Confidence: {conf:.2f}", 20, conf_colour)
    _txt(
        f"dX:{events['cursor_delta_x']:+6.1f}  dY:{events['cursor_delta_y']:+6.1f}  "
        f"Cursor:({cursor_x},{cursor_y})",
        42,
    )

    dbg = events.get("debug", {})
    lc = "LEFT-CLICK!" if events["left_click"] else ""
    rc = "RIGHT-CLICK!" if events["right_click"] else ""
    click_str = lc or rc or ""
    click_colour = (0, 255, 100) if click_str else (200, 200, 200)

    _txt(
        f"Blink L:{dbg.get('blink_left_score', 0):.2f}  "
        f"R:{dbg.get('blink_right_score', 0):.2f}  "
        f"Jaw:{dbg.get('jaw_open_score', 0):.2f}   {click_str}",
        65, click_colour,
    )

    # Confidence bar
    bar_w = int(conf * w)
    cv2.rectangle(overlay, (0, 78), (bar_w, 80), conf_colour, -1)

    # Nose-tip crosshair on frame
    hp = events.get("head_position", {})
    if hp:
        nx = int(hp["x"] * w)
        ny = int(hp["y"] * h)
        cv2.drawMarker(overlay, (nx, ny), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

    return overlay


def main() -> None:
    _print_banner()

    tracker = FaceTracker()
    try:
        tracker.start()
    except FileNotFoundError as e:
        logger.error("Model not found: %s", e)
        sys.exit(1)

    # Get screen dimensions for cursor clamping
    screen_w, screen_h = pyautogui.size()
    logger.info("Screen size: %dx%d", screen_w, screen_h)

    # Get current cursor position as starting point
    cur_x, cur_y = pyautogui.position()

    # Open a display window for the webcam preview
    win_name = "Face Tracking Demo  [ESC=exit | C=calibrate]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, config.CAMERA_WIDTH, config.CAMERA_HEIGHT + 10)

    logger.info("Demo running. Press ESC to exit, C to calibrate.")

    try:
        while True:
            # Read latest display frame from FaceTracker
            frame = tracker.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Get latest tracking events
            events = tracker.get_latest_events()

            # ── Apply cursor movement ──────────────────────────────────
            dx = events["cursor_delta_x"]
            dy = events["cursor_delta_y"]

            if events["face_detected"] and (dx != 0.0 or dy != 0.0):
                cur_x = int(np.clip(cur_x + dx, 0, screen_w - 1))
                cur_y = int(np.clip(cur_y + dy, 0, screen_h - 1))
                try:
                    pyautogui.moveTo(cur_x, cur_y, _pause=False)
                except pyautogui.FailSafeException:
                    logger.warning("PyAutoGUI fail-safe triggered! Exiting.")
                    break
            else:
                # Sync display position with actual cursor
                try:
                    pos = pyautogui.position()
                    cur_x, cur_y = pos.x, pos.y
                except Exception:
                    pass

            # ── Handle click events ────────────────────────────────────
            if events["left_click"] and events["face_detected"]:
                logger.info("Executing LEFT-CLICK at (%d, %d)", cur_x, cur_y)
                try:
                    pyautogui.click(button="left", _pause=False)
                except pyautogui.FailSafeException:
                    break

            if events["right_click"] and events["face_detected"]:
                logger.info("Executing RIGHT-CLICK at (%d, %d)", cur_x, cur_y)
                try:
                    pyautogui.click(button="right", _pause=False)
                except pyautogui.FailSafeException:
                    break

            # ── Draw overlay and display ───────────────────────────────
            preview = _draw_overlay(frame, events, screen_w, screen_h,
                                    cur_x, cur_y)
            cv2.imshow(win_name, preview)

            # ── Keyboard handling ──────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                logger.info("ESC pressed - exiting demo.")
                break
            elif key == ord("c") or key == ord("C"):
                logger.info("Re-calibrating neutral position...")
                tracker.calibrate_neutral()

            time.sleep(0.01)  # ~100 Hz poll loop; tracker runs at camera FPS

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        tracker.stop()
        logger.info("Demo exited cleanly.")


if __name__ == "__main__":
    main()
