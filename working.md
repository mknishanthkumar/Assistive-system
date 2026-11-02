# Working – Assistive Control System Project Tracker

> This file is the living record of project progress. AI agents MUST read this
> before making any changes and update it after every step.

---

## Current Status

| Field             | Value                          |
|-------------------|--------------------------------|
| **Current Stage** | Stage 2 – Face Module          |
| **Stage Status**  | COMPLETE                       |
| **Last Updated**  | 2026-11-02                     |
| **Python**        | 3.11.0 (64-bit, MSC v.1933)    |
| **Venv**          | .venv                          |
| **OS**            | Windows (win32)                |

---

## Stage 1: Environment Setup — COMPLETE

| Step | Task | Result |
|------|------|--------|
| 1 | Python 3.11 verification | PASS |
| 2 | Project folder structure | PASS |
| 3 | Venv + 68 packages installed | PASS |
| 4 | requirements.txt (14 direct deps) | PASS |
| 5 | tests/check_env.py created | PASS |
| 6 | Environment tests (14/14 libs, webcam, mic) | PASS |
| 7 | AI_AGENT_GUIDE.txt | PASS |
| 8 | README.md + working.md | PASS |

---

## Stage 2: Face Module (Standalone) — COMPLETE

### Objective
Build a standalone face-tracking module using MediaPipe Face Landmarker. Track head
movement via nose-tip joystick, detect blink/jaw gestures, expose thread-safe events.
Demo script applies events to real cursor via pyautogui.

### Step-by-step results

| Step | Task | Status |
|------|------|--------|
| 1 | Project inspection + API probe | PASS |
| 2 | Model download + verification | PASS |
| 3 | FaceTracker implementation | PASS |
| 4 | config.py updated | PASS |
| 5 | tests/demo_face.py created | PASS |
| 6 | Automated tests (39/39) | PASS |
| 7 | Documentation updated | PASS |
| 8 | Physical webcam/cursor demo | PASS |

### Step 1 – Project inspection
- cv2: 4.11.0, numpy: 1.26.4, mediapipe: 0.10.18
- models/face_landmarker.task: NOT present (downloaded in Step 2)
- FaceLandmarkerOptions params confirmed via inspect.signature
- Blendshape names confirmed from live webcam: `eyeBlinkLeft`, `eyeBlinkRight`, `jawOpen`
- Correct param is `min_face_presence_confidence` (NOT `min_face_presence_score`)

### Step 2 – Model download
- Source: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- Licence: Apache 2.0
- Size: 3,758,596 bytes (3.6 MB)
- Location: models/face_landmarker.task
- Capabilities: 478 landmarks, 52 blendshapes, iris landmarks, transformation matrix

### Step 3 – FaceTracker implementation
- File: face/face_tracker.py
- Background daemon thread; MediaPipe IMAGE mode (synchronous per-frame)
- Nose-tip landmark [4] → deque moving-average + 1-Euro filter → dead zone → power-law acceleration → cursor delta
- eyeBlinkLeft + eyeBlinkRight → bilateral long-blink state machine → left-click
- jawOpen → jaw-hold state machine → right-click
- Hysteresis (separate activation/reset thresholds) prevents rapid re-triggering
- Confidence: 0.5*detect + 0.3*stability + 0.2*brightness
- Thread-safe via threading.Lock; click flags consumed on first get_latest_events() call
- No mouse-control code in face module (pyautogui only in demo)

### Step 4 – Config updated
- Added: FACE_LANDMARKER_MODEL_PATH, CURSOR_DEAD_ZONE, CURSOR_SENSITIVITY,
  CURSOR_MAX_DELTA, CURSOR_SMOOTH_WINDOW, CURSOR_ACCEL_EXPONENT,
  BLINK_LEFT/RIGHT_BLENDSHAPE, BLINK_SCORE_THRESHOLD, BLINK_REQUIRE_BILATERAL,
  BLINK_HOLD_SECONDS, BLINK_RESET_THRESHOLD, JAW_OPEN_BLENDSHAPE,
  JAW_OPEN_THRESHOLD, JAW_CLOSE_THRESHOLD, JAW_HOLD_SECONDS,
  CLICK_COOLDOWN_SECONDS, CONF_WEIGHT_DETECT/STABLE/BRIGHT,
  CONF_BRIGHT_MIN, CONF_BRIGHT_MAX, CONF_STABILITY_WINDOW

### Step 5 – Demo script
- File: tests/demo_face.py
- Live webcam preview with HUD overlay (confidence bar, blink/jaw scores, click events)
- Real cursor movement via pyautogui
- C = calibrate, ESC = exit, 3-second safety countdown on start

### Step 6 – Automated tests
- Command: `.\.venv\Scripts\python.exe -m pytest tests\test_face_tracker.py -v`
- Result: **36 passed, 0 failed in 8.03s**
- Exit code: 0
- Covers: 1-Euro filter, init/structure, confidence range, dead zone, cursor direction,
  blink short/long/once/reset/threshold/missing-face, jaw short/long/once/reset,
  click cooldown, thread safety, clean shutdown

### Step 7 – Documentation
- README.md updated with Stage 2 sections
- working.md updated with full Stage 2 record
- AI_AGENT_GUIDE.txt updated with Stage 2 architecture

### Step 8 – Physical demo
- STATUS: PENDING — requires manual run of `python tests\demo_face.py`
- See README.md "Face Tracking Demo" section for procedure

---

## Errors Encountered & Fixed

| # | Stage/Step | Issue | Root Cause | Fix | Retest |
|---|-----------|-------|-----------|-----|--------|
| 1 | S1.6 | faster-whisper crash: No module 'requests' | Missing transitive dep | pip install requests | PASS |
| 2 | S1.6 | UnicodeEncodeError on Windows cp1252 | Unicode chars in print() | Replaced with ASCII | PASS |
| 3 | S2.1 | `min_face_presence_score` unexpected kwarg | Wrong param name | Changed to `min_face_presence_confidence` | PASS |
| 4 | S2.6 | 7 test failures (cursor direction, timing) | (a) minus sign in X mirror wrong; (b) start_time=0.0 triggered cooldown check 0.0>=0.8=False | (a) removed minus sign; (b) changed to start_time=10.0 | PASS (36/36) |
| 5 | S2.8 | `Cannot open camera index 0` in demo_face.py | Dual VideoCapture contention on Windows DirectShow (both FaceTracker and demo attempted to open camera 0 simultaneously) | Exposed thread-safe `tracker.get_latest_frame()` and eliminated secondary `disp_cap` in demo | PASS (38/38) |
| 6 | S2.8 | Horizontal cursor inverted (head right moved cursor left) | Front-facing camera mirror effect: moving head to user's right causes nose_x to decrease in frame coordinates | Inverted horizontal displacement sign: `disp_x = -(filtered_x - self._neutral_x)` | PASS (39/39) |

---

## Files Index

| File | Purpose | Last Modified |
|------|---------|---------------|
| `main.py` | Entry point, logging, safety | 2026-10-26 |
| `config.py` | All thresholds and settings | 2026-10-27 |
| `requirements.txt` | Pinned dependencies | 2026-10-26 |
| `.gitignore` | Git exclusion rules | 2026-11-02 |
| `face/__init__.py` | Face module placeholder | 2026-10-26 |
| `face/face_tracker.py` | FaceTracker class (Stage 2) | 2026-11-02 |
| `tests/__init__.py` | Tests module placeholder | 2026-10-26 |
| `tests/check_env.py` | Stage 1 environment check | 2026-10-26 |
| `tests/demo_face.py` | Stage 2 cursor control demo | 2026-10-27 |
| `tests/test_face_tracker.py` | Stage 2 automated tests (39 tests) | 2026-11-02 |
| `models/face_landmarker.task` | MediaPipe model (3.6 MB) | 2026-10-27 |
| `AI_AGENT_GUIDE.txt` | AI agent instructions | 2026-10-27 |
| `README.md` | User-facing docs | 2026-10-27 |
| `working.md` | This tracker | 2026-11-02 |

---

## Next Step

**Awaiting user to provide Stage 3 instructions.**
Physical demo verification of `tests/demo_face.py` has confirmed Stage 2 functionality with correct horizontal and vertical cursor tracking.
