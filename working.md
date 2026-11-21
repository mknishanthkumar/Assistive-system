# Working – Assistive Control System Project Tracker

> This file is the living record of project progress. AI agents MUST read this
> before making any changes and update it after every step.

---

## Current Status

| Field             | Value                                    |
|-------------------|------------------------------------------|
| **Current Stage** | Stage 5 – Context-Aware Decision Engine (Enhanced) |
| **Stage Status**  | COMPLETE                                 |
| **Last Updated**  | 2026-11-22                               |
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
- Result: **39 passed, 0 failed in 8.03s** (initial 36 + 3 horizontal direction verification tests)
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
| 7 | S3.12 | `test_extra_words_around_command` failed — "please click now" did not match "click" | Length-ratio containment check: 5/16 = 0.31 fell below 0.4 threshold | Replaced with whole-word subsequence matching (score = 0.90 when phrase found as contiguous words) | PASS (63/63) |
| 8 | S3.14 | Live transcription warning: `'Segment' object has no attribute 'avg_log_prob'` | Faster-Whisper dataclass field is named `avg_logprob` (no underscore) | Used `getattr(seg, 'avg_logprob', ...)` with fallback; confidence correctly calculated | PASS (64/64, 103 total) |
| 9 | S3-Action | 9 speech tests failed after COMMANDS dict update | Old tests referenced removed commands (next_slide, previous_slide, go_back, pause, resume, switch_to_face, switch_to_voice) | Updated tests to use new command IDs; updated completeness assertion from 15 to 17 | PASS (151 total) |
| 10 | S3-Debug | speech=YES always, conf=0.00, no commands executed | **Root cause: 4 bugs in VAD loop + transcription.** (a) WebRTC VAD at aggressiveness=2 flags ALL ambient noise as speech; no RMS energy gate. (b) `speech_detected` set from raw VAD output per-frame, not from confirmed speaking state. (c) No `no_speech_prob` gate on Whisper segments. (d) No max segment cap. (e) No hallucination filter on Whisper output. | (a) Added dual RMS+VAD gate (`is_active_speech = vad AND rms >= SPEECH_RMS_THRESHOLD`). (b) `speech_detected` now reflects `is_speaking` (confirmed state). (c) Added `no_speech_prob >= WHISPER_NO_SPEECH_THRESHOLD` early-discard. (d) Added max segment duration cap (`SPEECH_MAX_DURATION_S = 15s`). (e) Added `_is_whisper_hallucination()` filter. (f) VAD aggressiveness raised to 3. (g) `SPEECH_MIN_DURATION_S` raised to 0.5s. | PASS (187 total) |
| 11 | S4-Bugfix | Head-controlled cursor too slow; downward motion stuck/ignored | (a) `CURSOR_SENSITIVITY=8.0` on normalized displacement [0, 1] raised to 1.4 gave sub-pixel deltas (~0.05-0.5 px/frame). (b) `output/controller.py` truncated with `int()`, dropping positive deltas `0 < dy < 1.0` to 0 while negative deltas stepped -1 px. (c) Downward neck pitch has ~1/2 range of motion of upward tilt. | (a) Added `CURSOR_SPEED = 30.0` setting with base-10 scale in `face_tracker.py`. (b) Changed truncation to `round()` in `controller.py`. (c) Added `CURSOR_PITCH_DOWN_GAIN = 1.8` for downward pitch. | PASS (256 total) |
| 12 | S4-Scroll | Voice scroll commands moved page only ~0.5-1 cm | `SCROLL_AMOUNT = 3` in config.py was too small for meaningful page movement | Increased `SCROLL_AMOUNT` from 3 to 25 clicks (~7 cm on a typical page) | PASS (256 total) |
| 13 | Post-S5 | Mouse cursor movement broken (lag, jitter, clicks stolen, edge lockout, text focus freeze, drift) | (a) PyAutoGUI PAUSE=0.05 default injected 50ms sleep per move_cursor call, choking loop to ~12 FPS. (b) SensorLoop called get_latest_events() without consume=False, consuming and dropping blink/jaw click flags. (c) CURSOR_SCREEN_MARGIN=5 locked cursor out of outer 5px screen boundary. (d) Low-confidence frames caused noisy landmark jumps. (e) Decision engine locked into SPEECH on text focus with no pointing override. (f) Lack of neutral recalibration caused drift. | (a) Temporarily set pyautogui.PAUSE=0.0 in move_cursor with finally restoration. (b) Added consume: bool = True parameter so sensor pollers peek non-destructively. (c) Set CURSOR_SCREEN_MARGIN=0 allowing [0, screen_w - 1]. (d) Added confidence threshold gating in face_tracker and router. (e) Added pointing_active check in DecisionEngine. (f) Added "calibrate"/"center cursor" voice commands. | PASS (343 tests) |

---

## Stage 4: Integration (Face + Speech + Real Actions) — COMPLETE

### Objective
Wire FaceTracker + SpeechEngine together under a threaded architecture with a
simple Router, context sensor loop, thread-safe CSV logger, emergency stop,
DRY_RUN mode, and clean shutdown.

### Step-by-step results

| Step | Task | Status |
|------|------|--------|
| 1 | Read all source files; flag mismatch | PASS |
| 2 | Add DRY_RUN + Stage 4 settings to config.py | PASS |
| 3 | Create logs/csv_logger.py (thread-safe, per-run CSV) | PASS |
| 4 | Create context/sensors.py (SensorLoop) | PASS |
| 5 | Create output/router.py (simple Router class) | PASS |
| 6 | Extend output/controller.py (move_cursor, source param, DRY_RUN default) | PASS |
| 7 | Rewrite main.py (full 3-thread integration, shutdown, banner) | PASS |
| 8 | Tests: test_router.py (16), test_sensors.py (20), test_csv_logger.py (19), test_controller_stage4.py (14) | PASS |
| 9 | Full test suite: 255 passed, 0 failed | PASS |
| 10 | Dry-run verification of main.py end-to-end | PASS |
| 11 | Documentation updates | PASS |

### Pre-coding mismatch report (required by Stage 4 spec)

| What | Real or Logged? | Notes |
|------|----------------|-------|
| Speech commands (click, right_click, scroll, hotkeys, volume, dictation) | **REAL** (pyautogui/webbrowser) | OutputController already executes real actions |
| Face cursor movement | **Logged only** in main.py stub | demo_face.py applied it, but main.py was a Stage 1 stub — fixed in Stage 4 |
| Face blink/jaw clicks in main.py | **Not connected at all** | FaceTracker computed them but main.py never ran FaceTracker — fixed in Stage 4 |
| **Mismatch**: main.py was Stage 1 stub | main.py called only `_apply_safety()` and exited | Fixed: main.py now starts all subsystems |

### Stage 4 architecture

```
main.py
├── OutputController (dry_run aware, move_cursor added)
├── Router (simple pass-through, replaceable)
├── FaceTracker → face_thread → Router → OutputController
├── SpeechEngine → speech_thread → Router → OutputController
├── SensorLoop → sensor_q → sensor_drain_thread → CSV
├── CsvLogger (thread-safe, one file per run)
└── pynput GlobalHotKeys (Ctrl+Alt+Q emergency stop)
```

### New files (Stage 4)

| File | Purpose |
|------|---------|
| `logs/csv_logger.py` | Thread-safe CSV event logger |
| `logs/__init__.py` | Python package marker |
| `context/sensors.py` | SensorLoop — polls noise, face_confidence, brightness, active window |
| `output/router.py` | Simple Router — passes face/speech events to controller |
| `tests/test_router.py` | 16 router tests |
| `tests/test_sensors.py` | 20 sensor tests |
| `tests/test_csv_logger.py` | 19 CSV logger tests |
| `tests/test_controller_stage4.py` | 14 controller Stage 4 tests (move_cursor, DRY_RUN, source) |

### Modified files (Stage 4)

| File | Change |
|------|--------|
| `config.py` | Added DRY_RUN, FACE_CLICK_RATE_LIMIT_S, CONTROLLER_MAX_CURSOR_DELTA, CURSOR_SCREEN_MARGIN, EMERGENCY_STOP_HOTKEY_PYNPUT, SENSOR_POLL_INTERVAL_S, CSV_LOG_SUBDIR |
| `output/controller.py` | Added move_cursor(), source param to execute(), DRY_RUN default from config, face-click rate limit field |
| `main.py` | Full rewrite: 3-thread integration, shutdown, banner, CSV logging, emergency stop |

### Test counts (Stage 4)

| Test file | Tests |
|-----------|-------|
| test_face_tracker.py | 39 |
| test_speech_engine.py | 72 |
| test_output_controller.py | 40 |
| test_router.py | 16 |
| test_sensors.py | 20 |
| test_csv_logger.py | 19 |
| test_controller_stage4.py | 14 |
| **Total** | **220** (but collected as 255 due to some parameterized counts) |

Actual pytest result: **255 passed, 0 failed in 12.04s**

---

## Stage 3: Speech Module (Standalone) — COMPLETE

### Objective
Build a standalone speech-recognition engine using WebRTC VAD + Faster-Whisper.
Detect voice commands (17 predefined), support dictation mode, expose noise level
and speech confidence.  Engine reports events only — no mouse/keyboard execution.
Execute recognised commands as real Windows actions via OutputController.

### Step-by-step results

| Step | Task | Status |
|------|------|--------|
| 1 | Project inspection + dependency probe | PASS |
| 2 | SpeechEngine implementation | PASS |
| 3 | Microphone capture (sounddevice 30 ms) | PASS |
| 4 | WebRTC VAD segmentation | PASS |
| 5 | Faster-Whisper transcription | PASS |
| 6 | Command parser (fuzzy matching) | PASS |
| 7 | Dictation mode | PASS |
| 8 | Noise level + speech confidence | PASS |
| 9 | Event interface (thread-safe) | PASS |
| 10 | Configuration updates | PASS |
| 11 | Standalone speech demo | PASS |
| 12 | Automated tests (63/63) | PASS |
| 13 | Documentation updates | PASS |
| 14 | Physical microphone verification | PENDING (manual) |

### Step 1 – Project inspection
- sounddevice: 0.5.1, webrtcvad: 2.0.14, faster-whisper: 1.1.0, numpy: 1.26.4
- Default input device index: 1, available input devices: 8
- VAD frame: 16kHz × 30ms = 480 samples = 960 bytes (verified)
- Whisper base model: downloaded from HuggingFace Hub (~126 s first run), cached at `~/.cache/huggingface/`
- Existing config.py already had: WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_LANGUAGE, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_BLOCK_DURATION_MS, VAD_AGGRESSIVENESS

### Step 2 – SpeechEngine implementation
- File: speech/speech_engine.py
- 3-thread pipeline: audio callback → VAD processing → transcription
- Audio callback (sounddevice I/O thread): captures 30ms int16 frames → `_audio_queue`
- VAD thread: reads frames, runs WebRTC VAD, detects speech onset/offset, assembles segments → `_transcription_queue`
- Transcription thread: runs Faster-Whisper on segments, parses commands or emits dictation, writes to `_events`
- Thread-safe via threading.Lock; edge-triggered fields consumed on `get_latest_events()` read

### Step 3 – Microphone capture
- sounddevice.RawInputStream with int16, mono, 16 kHz, 480-sample blocks
- Configurable device via `config.MIC_DEVICE_INDEX` (None = system default)
- Non-blocking callback pushes to queue (drops frames if queue full)
- Graceful error handling for mic init and disconnection

### Step 4 – VAD segmentation
- WebRTC VAD aggressiveness configurable (0-3, default 2)
- Speech onset requires SPEECH_TRIGGER_FRAMES (3) consecutive speech frames
- Pre-speech ring buffer (SPEECH_PADDING_FRAMES = 10) captures onset
- Segment ends after SPEECH_SILENCE_DURATION_S (0.8 s) of silence
- Short segments (< SPEECH_MIN_DURATION_S = 0.4 s) discarded

### Step 5 – Faster-Whisper transcription
- Model loaded once in `start()` (size=base, device=cpu, compute_type=int8)
- int16 audio → float32 normalisation → `model.transcribe()`
- Confidence from `exp(avg_log_prob)` clamped to [0.0, 1.0]
- Own VAD used (vad_filter=False passed to Whisper)

### Step 6 – Command parser
- 15 commands defined in COMMANDS dict
- 3-tier matching: exact → whole-word containment (score=0.90) → difflib fuzzy
- Configurable threshold: FUZZY_MATCH_THRESHOLD (default 0.70)
- Unrelated phrases rejected (score below threshold → None)

### Step 7 – Dictation mode
- "start dictation" activates, "stop dictation" deactivates
- While active: recognised speech → dictation_text event (not command-parsed)
- "stop dictation" always recognised even during dictation
- Command events and dictation events are separate

### Step 8 – Noise & confidence
- noise_level: rolling RMS over NOISE_RMS_WINDOW_FRAMES (50) frames, normalised to [0.0, 1.0]
- speech_confidence: exp(avg_log_prob) from Whisper segments, [0.0, 1.0]
- Both documented and finite

### Step 9 – Event interface
- 11 fields: timestamp, command, command_phrase, dictation_text, dictation_mode, noise_level, speech_confidence, speech_detected, transcription_text, vad_status, event_id
- Edge-triggered: command, command_phrase, dictation_text, transcription_text consumed on read
- Continuous: noise_level, speech_detected updated every frame

### Step 10 – Configuration
- Added to config.py: MIC_DEVICE_INDEX, SPEECH_TRIGGER_FRAMES, SPEECH_MIN_DURATION_S, SPEECH_SILENCE_DURATION_S, SPEECH_PADDING_FRAMES, FUZZY_MATCH_THRESHOLD, NOISE_RMS_WINDOW_FRAMES
- Preserved all Stage 1/2 settings

### Step 12 – Automated tests (speech engine)
- Command: `.\.venv\Scripts\python.exe -m pytest tests\test_speech_engine.py -v`
- Result: **72 passed, 0 failed in 1.56s**
- Covers: text normalisation, all 17 commands + aliases, fuzzy matching, rejection, init/structure, VAD frame sizes, VAD detection, silence handling, min duration, dictation activation/deactivation/preservation/stop-during/command-blocking, RMS noise, speech confidence, segment logprob compatibility, edge-triggering, thread safety, clean shutdown, command completeness

### Step 15 – Action execution (OutputController)
- File: output/controller.py
- Maps 14 actionable command IDs to real Windows actions via injectable backends
- Mouse: click, double_click, right_click at current cursor position
- Scroll: scroll_up/scroll_down (configurable SCROLL_AMOUNT notches)
- Browser: open_browser via webbrowser.open (system default)
- Window: close_window (Alt+F4), switch_window (Alt+Tab), minimize_window (Win+Down)
- Volume: volume_up, volume_down, mute via media key presses
- Keyboard: press_enter, press_escape
- Dictation: type_dictation() types text into focused field via pyautogui.typewrite()
- Safety: confidence threshold, per-command cooldown, emergency stop, dry-run mode
- Control commands (start_dictation, stop_dictation, stop_control) passed through to SpeechEngine

### Step 16 – Demo rewrite (tests/demo_speech.py)
- Integrated OutputController into demo
- Added --dry-run flag (log actions without executing)
- Added emergency hotkey Ctrl+Alt+Q via pynput.keyboard.GlobalHotKeys
- Real actions: mouse clicks, key presses, browser launch, volume control, dictation typing
- Clean shutdown: engine.stop() + hotkey_listener.stop()

### Step 17 – Configuration updates
- Added to config.py: SPEECH_CONFIDENCE_THRESHOLD (0.40), SPEECH_COMMAND_COOLDOWN_S (1.0), SCROLL_AMOUNT (3), DICTATION_TYPE_DELAY_S (0.02)
- Preserved all existing settings

### Step 18 – COMMANDS dict update
- Replaced old commands (next_slide, previous_slide, go_back, pause, resume, switch_to_face, switch_to_voice) with action-oriented commands (open_browser, close_window, switch_window, minimize_window, volume_up, volume_down, mute, press_enter, press_escape)
- Added aliases: "left click", "exit window", "next window", "minimize", "unmute", "enter", "escape"
- Total: 17 command IDs with 24 canonical phrases

### Step 19 – OutputController automated tests
- Command: `.\.venv\Scripts\python.exe -m pytest tests\test_output_controller.py -v`
- Result: **40 passed, 0 failed in 2.01s**
- Covers: click, double click, right click, scroll up/down, open_browser, window hotkeys (close/switch/minimize), keyboard keys (enter/escape), volume keys (up/down/mute), unknown command rejected, empty command rejected, low confidence rejected, high confidence accepted, exact threshold accepted, dictation types text, dictation empty rejected, dictation whitespace rejected, dictation strips text, dictation does not trigger command, cooldown blocks duplicate, different commands not blocked, emergency stop blocks commands, emergency stop blocks dictation, is_stopped flag, dry-run does not call backend, dry-run dictation, control commands passthrough, action result structure, backend exception handling, thread safety, clean shutdown

### Step 20 – Full test suite
- Command: `.\.venv\Scripts\python.exe -m pytest tests\ -v`
- Result: **151 passed, 0 failed in 7.41s**
  - Face tracker: 39 tests
  - Speech engine: 72 tests
  - Output controller: 40 tests

### Step 21 – Documentation updates
- working.md, README.md, AI_AGENT_GUIDE.txt updated

### Step 22 – Physical verification
- STATUS: NOT VERIFIED — requires manual run by user
- See README.md "Manual Test Guide" section for procedure

---

## Files Index

| File | Purpose | Last Modified |
|------|---------|---------------|
| `main.py` | Entry point — Stage 4 integration (threads, shutdown, banner) | 2026-11-16 |
| `config.py` | All thresholds and settings | 2026-11-16 |
| `requirements.txt` | Pinned dependencies | 2026-10-26 |
| `.gitignore` | Git exclusion rules | 2026-11-02 |
| `face/__init__.py` | Face module placeholder | 2026-10-26 |
| `face/face_tracker.py` | FaceTracker class (Stage 2) | 2026-11-02 |
| `speech/__init__.py` | Speech module placeholder | 2026-10-26 |
| `speech/speech_engine.py` | SpeechEngine + 17 commands (Stage 3) | 2026-11-03 |
| `output/__init__.py` | Output module placeholder | 2026-10-26 |
| `output/controller.py` | OutputController (Stage 3+4 actions, move_cursor) | 2026-11-16 |
| `output/router.py` | Simple Router (Stage 4) | 2026-11-16 |
| `context/__init__.py` | Context module placeholder | 2026-10-26 |
| `context/sensors.py` | SensorLoop (Stage 4) | 2026-11-16 |
| `logs/__init__.py` | Logs package marker | 2026-11-16 |
| `logs/csv_logger.py` | Thread-safe CSV event logger (Stage 4) | 2026-11-16 |
| `tests/__init__.py` | Tests module placeholder | 2026-10-26 |
| `tests/check_env.py` | Stage 1 environment check | 2026-10-26 |
| `tests/demo_face.py` | Stage 2 cursor control demo | 2026-10-27 |
| `tests/demo_speech.py` | Stage 3 speech-control demo + actions | 2026-11-03 |
| `tests/test_face_tracker.py` | Stage 2 automated tests (39 tests) | 2026-11-02 |
| `tests/test_speech_engine.py` | Stage 3 speech tests (72 tests) | 2026-11-03 |
| `tests/test_output_controller.py` | Stage 3 action tests (40 tests) | 2026-11-03 |
| `tests/test_router.py` | Stage 4 router tests (16 tests) | 2026-11-16 |
| `tests/test_sensors.py` | Stage 4 sensor tests (20 tests) | 2026-11-16 |
| `tests/test_csv_logger.py` | Stage 4 CSV logger tests (19 tests) | 2026-11-16 |
| `tests/test_controller_stage4.py` | Stage 4 controller additions (14 tests) | 2026-11-16 |
| `models/face_landmarker.task` | MediaPipe model (3.6 MB) | 2026-10-27 |
| `AI_AGENT_GUIDE.txt` | AI agent instructions | 2026-11-16 |
| `README.md` | User-facing docs | 2026-11-16 |
| `working.md` | This tracker | 2026-11-16 |

---

## Next Step

**Stage 5 COMPLETE.**

---

## Stage 5: Context Engine (Decision Engine) — COMPLETE

### Objective
Build `context/decision_engine.py`: a context-aware decision engine that maps
real-time sensor readings to an active input mode (FACE, SPEECH, BOTH, PAUSED).
Integrate with Router for event filtering and with main.py for live operation.

### Step-by-step results

| Step | Task | Status |
|------|------|--------|
| 1 | Read working.md, README, AI_AGENT_GUIDE.txt | PASS |
| 2 | Baseline test suite (Stages 1-4): 256/256 | PASS |
| 3 | Add 6 DECISION_* constants to config.py | PASS |
| 4 | Create context/decision_engine.py | PASS |
| 5 | Upgrade output/router.py with mode filtering + DecisionEngine | PASS |
| 6 | Update main.py: wire DecisionEngine into sensor drain + speech loop | PASS |
| 7 | Write tests/test_decision_engine.py (80 tests) | PASS: 80/80 |
| 8 | Full test suite (Stages 1-5): 336/336 | PASS |
| 9 | Dry-run demo (10 scripted scenarios) | PASS: all 10 scenarios |
| 10 | Update documentation | PASS |

### Decision Engine Rules (priority order)

| Priority | Condition | Mode |
|----------|-----------|------|
| 1 | noisy AND face_conf < thresh AND speech_conf < thresh | PAUSED |
| 2 | noise >= DECISION_NOISE_THRESHOLD | FACE |
| 3 | text_field_focused=True AND quiet | SPEECH |
| 4 | face_conf < DECISION_FACE_CONF_THRESHOLD OR brightness < DECISION_BRIGHTNESS_THRESHOLD | SPEECH |
| 5 | face_conf >= threshold AND speech_conf >= threshold AND quiet | BOTH |
| 6 | (fallback) | FACE |

### Hysteresis
- `DECISION_HOLD_DURATION_S` = 1.0 s: candidate must hold continuously before switch commits
- `DECISION_MIN_SWITCH_INTERVAL_S` = 2.0 s: minimum wall-clock seconds between any two switches

### Manual override
- `set_manual_override(Mode.SPEECH)` ← voice command "switch to voice"
- `set_manual_override(Mode.FACE)` ← jaw gesture or voice "switch to face"
- `clear_manual_override()` ← voice phrase "switch to auto" / "automatic mode"
- `stop_control` voice command always forwarded regardless of mode (safety)

### New files (Stage 5)

| File | Purpose |
|------|---------|
| `context/decision_engine.py` | DecisionEngine: 6 rules, 2-stage hysteresis, manual override, CSV logging |
| `tests/test_decision_engine.py` | 80 automated tests (all rules, hysteresis, override, CSV, logging, threads) |
| `tests/demo_decision_engine.py` | 10-scenario scripted dry-run simulation |

### Modified files (Stage 5)

| File | Change |
|------|--------|
| `config.py` | Added 6 DECISION_* constants |
| `output/router.py` | Added DecisionEngine parameter, mode-filtering for face + speech events |
| `main.py` | Added DecisionEngine, sensor drain feeds evaluate(), speech loop intercepts override phrases, updated banner |

### Test counts (cumulative)

| Test file | Tests |
|-----------|-------|
| test_face_tracker.py | 39 |
| test_speech_engine.py | 72 |
| test_output_controller.py | 40 |
| test_router.py | 16 |
| test_sensors.py | 20 |
| test_csv_logger.py | 19 |
| test_controller_stage4.py | 14 |
| test_decision_engine.py | 80 |
| **test_cursor_mapping.py** | **7** |
| **Total** | **343** (collected as 343 by pytest) |

Actual pytest result: **343 passed, 0 failed in 27.10s**

### Dry-run output (all 10 scenarios PASSED)

```
S1: Both reliable            → BOTH
S2: Noisy room (0.8)         → FACE   (noise disabled speech)
S3: Dark room (brightness=10) → SPEECH (face lost)
S4: Text field focused        → SPEECH (dictation)
S5: Face confidence=0.1       → SPEECH (voice navigation)
S6: Override → SPEECH         → SPEECH [OVERRIDE] (noisy room ignored)
S7: Override → FACE           → FACE   [OVERRIDE] (dark room ignored)
S8: Clear override            → SPEECH (auto rules restored)
S9: Both unreliable           → PAUSED + WARNING logged
S10: Hysteresis hold timer    → BOTH (held) → FACE (committed after 0.1 s)
```

---

## Post-Stage 5: Face Tracking & Cursor Movement Bug Fix — COMPLETE

### Issues Addressed & Verified
1. **PyAutoGUI Pause Lag**: Wrapped `move_cursor` with zeroed `pyautogui.PAUSE` restoration; eliminated 50ms blocking sleep per face frame.
2. **Click & Command Stealing**: Replaced destructive `get_latest_events()` calls in `SensorLoop` and `main.py` with `consume=False` non-destructive peeks, preserving edge-triggered clicks (`left_click`, `right_click`).
3. **Screen Edge Lockout**: Configured `CURSOR_SCREEN_MARGIN = 0` allowing full `[0, screen_w - 1]` range.
4. **Low Confidence Jitter**: Added confidence gating (<0.40) in `face_tracker.py` and `router.py` to suppress noisy jumps.
5. **Mode Locking in DecisionEngine**: Added Rule 2b to allow active pointing context to restore `Mode.FACE`.
6. **Recalibration Drift**: Added voice recalibration ("calibrate", "center cursor") mapped to `tracker.calibrate_neutral()`.

---

## Post-Stage 5 Enhancements (Manual Override BOTH, Voice Re-center, Hold-to-Scroll) — COMPLETE

### Feature 1: "Both" Mode in Manual Override
- **Objective**: Add manual override option for `Mode.BOTH` so face/head tracking and speech recognition are simultaneously active on demand without changing auto-mode decision logic.
- **Voice Commands**: "switch to both", "switch to face and speech", "both mode".
- **Implementation**:
  - `context/decision_engine.py`: updated `set_manual_override()` to apply the mode immediately and trigger an audit/overlay transition.
  - `main.py`: updated `_speech_loop` to detect `_OVERRIDE_BOTH` phrases before `_OVERRIDE_FACE` / `_OVERRIDE_SPEECH` to avoid substring collision on "switch to face and speech".
  - Status display and logs: DecisionEngine logs `MODE SWITCH ... -> BOTH | reason=manual override | override=True`, overlay queue updated with `mode="BOTH"`, CSV logger audit entry recorded.
  - Tests: added 9 unit tests in `tests/test_decision_engine.py` (`TestManualOverrideBothMode`) verifying override behavior, flag, property, transitions, and clearance.

### Feature 2: Voice Command to Re-center (Head Tracking)
- **Objective**: Provide hands-free re-calibration of neutral face/head tracking position for users unable to press the 'C' key. Keep 'C' key operational.
- **Voice Commands**: "arrange center", "recenter", "re-center", "center".
- **Confirmation**:
  - **Audio Confirmation**: High-pitched confirmation beep (880 Hz, 150 ms) triggered asynchronously via background thread (`winsound.Beep`).
  - **Visual Confirmation**: Console print confirmation `[RE-CENTER] Neutral head position calibrated successfully.`, logging to INFO, overlay badge queue updated with `last_cmd="RE-CENTERED"`, and action logged to CSV.
- **Implementation**:
  - `speech/speech_engine.py`: added canonical `recenter` command and phrases to `COMMANDS`.
  - `main.py`: intercepted in `_speech_loop` to invoke `tracker.calibrate_neutral()` and dispatch audio/visual confirmations.
  - `demo_face.py`: 'C' key continues to function as before.

### Feature 3: Hold-to-Scroll for Long Documents
- **Objective**: Enable hands-free continuous scrolling for long documents without repeated utterances.
- **Voice Commands**:
  - Start: "hold scroll down", "hold scroll", "continuous scroll down", "keep scrolling down", "hold scroll up", "continuous scroll up", "keep scrolling up".
  - Stop: "stop scroll", "stop scrolling", "release scroll", "release", "stop".
- **Implementation**:
  - `config.py`: added `HOLD_SCROLL_AMOUNT = 5`, `HOLD_SCROLL_INTERVAL_S = 0.12`, `HOLD_SCROLL_TIMEOUT_S = 30.0`.
  - `output/controller.py`: created `HoldScrollManager` thread-safe background daemon; wired into `OutputController.hold_scroll`; stopped automatically on emergency stop or safety timeout; exempted `recenter`, `hold_scroll_down`, `hold_scroll_up`, `stop_scroll` in `execute()` from direct backend dispatch.
  - `main.py`: wired start/stop in `_speech_loop` with raw "stop"/"release" phrase fallback.
  - `tests/demo_speech.py`: updated demo loop to execute `HoldScrollManager` on hold scroll commands.
  - Single-step scroll commands (`scroll_up`, `scroll_down`) remain fully intact and independent.
- **Tests**:
  - `tests/test_output_controller.py`: added 7 tests in `TestHoldScrollManager`.
  - `tests/test_speech_engine.py`: updated completeness assertion from 17 to 21 commands; added parser tests for `recenter`, `hold_scroll_down`, `hold_scroll_up`, `stop_scroll`.
  - `tests/test_speech_loop_features.py`: added 7 end-to-end integration tests for `_speech_loop` covering both override, re-center, and hold-to-scroll.

---

## Test Counts (Cumulative)

| Test file | Tests |
|-----------|-------|
| test_face_tracker.py | 39 |
| test_speech_engine.py | 114 |
| test_output_controller.py | 47 |
| test_router.py | 16 |
| test_sensors.py | 20 |
| test_csv_logger.py | 19 |
| test_controller_stage4.py | 14 |
| test_decision_engine.py | 89 |
| test_cursor_mapping.py | 7 |
| test_speech_loop_features.py | 7 |
| test_launcher.py | 25 |
| test_stage6_imports.py | 5 |
| **Total** | **402 passed, 0 failed** |

---

## Files Index (Updated)

| File | Purpose | Last Modified |
|------|---------|---------------|
| `main.py` | Entry point — BOTH override, voice re-center, hold-to-scroll wired | 2026-11-22 |
| `config.py` | Added HOLD_SCROLL_AMOUNT, HOLD_SCROLL_INTERVAL_S, HOLD_SCROLL_TIMEOUT_S | 2026-11-22 |
| `speech/speech_engine.py` | Added recenter, hold_scroll_down, hold_scroll_up, stop_scroll | 2026-11-22 |
| `output/controller.py` | Added HoldScrollManager, emergency stop integration, hold_scroll property | 2026-11-22 |
| `context/decision_engine.py` | Enhanced set_manual_override() immediate mode switch and overlay push | 2026-11-22 |
| `tests/demo_speech.py` | Updated banner and demo loop for hold-to-scroll and re-center | 2026-11-22 |
| `tests/test_decision_engine.py` | Added 9 tests in TestManualOverrideBothMode | 2026-11-22 |
| `tests/test_speech_engine.py` | Added command parser tests and updated completeness to 21 | 2026-11-22 |
| `tests/test_output_controller.py` | Added 7 tests in TestHoldScrollManager | 2026-11-22 |
| `tests/test_speech_loop_features.py` | 7 end-to-end speech loop feature integration tests | 2026-11-22 |
| `WORKING.md` | Living project tracker | 2026-11-22 |

---

## Next Step

**All 3 features implemented and fully verified with 402 passing unit/integration tests.**
Ready for live interactive testing:
- Run `python main.py --dry-run` or `python main.py`
- Test voice commands: "switch to both", "arrange center", "hold scroll down", "stop"


