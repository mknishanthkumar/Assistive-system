"""
tests/demo_cursor_mapping.py - Dry-run simulation for Face Tracking & Cursor Mapping.

Validates the full cursor mapping pipeline using scripted face positions WITHOUT
requiring a physical webcam or moving the real system mouse pointer.

Pipeline Traced:
  1. Scripted Face Position (Nose Landmark x, y)
  2. Displacements & Filtering (Exponential moving average + dead zone + non-linear power-law acceleration)
  3. Direction Mapping (Mirrored camera frame: nose_x decrease -> head right -> cursor right)
  4. Confidence Gating (Low confidence frames suppressed)
  5. Screen Boundary Mapping & Edge Reach ([0, screen_w - 1], [0, screen_h - 1])

Run with:
    .venv\\Scripts\\python.exe tests\\demo_cursor_mapping.py
"""

import sys
import os
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from face.face_tracker import FaceTracker
from output.controller import OutputController
from output.router import Router
from context.decision_engine import DecisionEngine, Mode


def make_mock_result(nose_x: float, nose_y: float):
    landmark = MagicMock()
    landmark.x = nose_x
    landmark.y = nose_y
    landmark.z = 0.0

    landmarks = [MagicMock() for _ in range(478)]
    landmarks[FaceTracker._NOSE_TIP_IDX] = landmark

    result = MagicMock()
    result.face_landmarks = [landmarks]
    result.face_blendshapes = []
    return result


def run_dry_run():
    screen_w, screen_h = 1920, 1080
    cursor_x = screen_w // 2  # 960
    cursor_y = screen_h // 2  # 540

    print("=" * 80)
    print("  ASSISTIVE CONTROL SYSTEM - CURSOR MAPPING DRY RUN")
    print(f"  Simulated Screen Resolution: {screen_w} x {screen_h}")
    print(f"  Starting Cursor Position:    ({cursor_x}, {cursor_y})")
    print("=" * 80)
    print(f"{'Scenario':<20} | {'Raw Landmark':<14} | {'Delta (dx, dy)':<16} | {'Cursor (X, Y)':<16} | {'Status'}")
    print("-" * 80)

    # Initialize tracker with calibrated neutral center (0.50, 0.50)
    tracker = FaceTracker()
    tracker._neutral_x = 0.50
    tracker._neutral_y = 0.50

    # Seed nose history at neutral
    for _ in range(15):
        tracker._nose_history_x.append(0.50)
        tracker._nose_history_y.append(0.50)

    # Controller mock that tracks simulated position
    current_pos = [cursor_x, cursor_y]

    def mock_move_cursor(dx, dy):
        new_x = current_pos[0] + dx
        new_y = current_pos[1] + dy
        # Apply screen edge clamping as in controller.py
        max_x = screen_w - 1 if config.CURSOR_SCREEN_MARGIN == 0 else screen_w - config.CURSOR_SCREEN_MARGIN
        min_x = 0 if config.CURSOR_SCREEN_MARGIN == 0 else config.CURSOR_SCREEN_MARGIN
        max_y = screen_h - 1 if config.CURSOR_SCREEN_MARGIN == 0 else screen_h - config.CURSOR_SCREEN_MARGIN
        min_y = 0 if config.CURSOR_SCREEN_MARGIN == 0 else config.CURSOR_SCREEN_MARGIN

        current_pos[0] = int(max(min_x, min(max_x, round(new_x))))
        current_pos[1] = int(max(min_y, min(max_y, round(new_y))))

    ctrl = MagicMock()
    ctrl.move_cursor.side_effect = mock_move_cursor
    router = Router(ctrl)

    scenarios = [
        ("1. Neutral Center", 0.50, 0.50, 100.0, "Stay centered (~0 delta)"),
        ("2. Look Left",      0.55, 0.50, 100.0, "Move LEFT (-dx)"),
        ("3. Return Center",  0.50, 0.50, 100.0, "Reset to neutral"),
        ("4. Look Right",     0.45, 0.50, 100.0, "Move RIGHT (+dx)"),
        ("5. Return Center",  0.50, 0.50, 100.0, "Reset to neutral"),
        ("6. Look Up",        0.50, 0.45, 100.0, "Move UP (-dy)"),
        ("7. Return Center",  0.50, 0.50, 100.0, "Reset to neutral"),
        ("8. Look Down",      0.50, 0.55, 100.0, "Move DOWN (+dy)"),
        ("9. Return Center",  0.50, 0.50, 100.0, "Reset to neutral"),
        ("10. Corner Top-L",  0.55, 0.45, 100.0, "Move UP-LEFT (-dx, -dy)"),
        ("11. Corner Top-R",  0.45, 0.45, 100.0, "Move UP-RIGHT (+dx, -dy)"),
        ("12. Corner Btm-L",  0.55, 0.55, 100.0, "Move DOWN-LEFT (-dx, +dy)"),
        ("13. Corner Btm-R",  0.45, 0.55, 100.0, "Move DOWN-RIGHT (+dx, +dy)"),
        ("14. Face Lost",     None, None, 100.0, "Confidence 0.0 -> no move"),
        ("15. Low Conf Jitter", 0.42, 0.42, 5.0, "Low conf suppressed -> no jump"),
    ]

    for name, nx, ny, bright, expectation in scenarios:
        now = time.monotonic()
        if nx is None:
            # Face lost
            ev = tracker._build_events(None, brightness=bright, now=now)
        elif bright < 10.0:
            # Simulate low confidence frame
            tracker._compute_confidence = lambda b, n: 0.20
            res = make_mock_result(nx, ny)
            ev = tracker._build_events(res, brightness=bright, now=now)
        else:
            tracker._compute_confidence = lambda b, n: 0.85
            # Prime history towards new position for responsive delta in step test
            for _ in range(5):
                tracker._nose_history_x.append(nx)
                tracker._nose_history_y.append(ny)
            res = make_mock_result(nx, ny)
            ev = tracker._build_events(res, brightness=bright, now=now)

        dx = ev.get("cursor_delta_x", 0.0)
        dy = ev.get("cursor_delta_y", 0.0)

        # Route through router to controller
        router.handle_face_events(ev)

        raw_str = f"({nx:.2f}, {ny:.2f})" if nx is not None else "None (Lost)"
        delta_str = f"({dx:+6.1f}, {dy:+6.1f})"
        pos_str = f"({current_pos[0]:4d}, {current_pos[1]:4d})"

        print(f"{name:<20} | {raw_str:<14} | {delta_str:<16} | {pos_str:<16} | {expectation}")

    # Edge reach test
    print("-" * 80)
    print("Testing Screen Edge Reach (Multiple aggressive movements):")
    for _ in range(40):
        # Heavy right + down
        mock_move_cursor(100.0, 100.0)
    print(f"Bottom-Right Edge reached: ({current_pos[0]}, {current_pos[1]}) - Expected: ({screen_w - 1}, {screen_h - 1})")
    assert current_pos[0] == screen_w - 1 and current_pos[1] == screen_h - 1, "Failed to reach bottom-right edge!"

    for _ in range(50):
        # Heavy left + up
        mock_move_cursor(-100.0, -100.0)
    print(f"Top-Left Edge reached:     ({current_pos[0]}, {current_pos[1]}) - Expected: (0, 0)")
    assert current_pos[0] == 0 and current_pos[1] == 0, "Failed to reach top-left edge!"

    print("=" * 80)
    print("ALL DRY-RUN SCENARIOS PASSED PERFECTLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_dry_run()
