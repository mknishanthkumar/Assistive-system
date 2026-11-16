# Working – Assistive Control System Project Tracker

> This file is the living record of project progress. AI agents MUST read this
> before making any changes and update it after every step.

---

## Current Status

| Field             | Value                                    |
|-------------------|------------------------------------------|
| **Current Stage** | Stage 3 – Speech Module + Action Execution |
| **Stage Status**  | DEBUGGED & COMPLETE                      |
| **Last Updated**  | 2026-11-13                               |
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
| `main.py` | Entry point, logging, safety | 2026-10-26 |
| `config.py` | All thresholds and settings | 2026-11-03 |
| `requirements.txt` | Pinned dependencies | 2026-10-26 |
| `.gitignore` | Git exclusion rules | 2026-11-02 |
| `face/__init__.py` | Face module placeholder | 2026-10-26 |
| `face/face_tracker.py` | FaceTracker class (Stage 2) | 2026-11-02 |
| `speech/__init__.py` | Speech module placeholder | 2026-10-26 |
| `speech/speech_engine.py` | SpeechEngine + 17 commands (Stage 3) | 2026-11-03 |
| `output/__init__.py` | Output module placeholder | 2026-10-26 |
| `output/controller.py` | OutputController (Stage 3 actions) | 2026-11-03 |
| `tests/__init__.py` | Tests module placeholder | 2026-10-26 |
| `tests/check_env.py` | Stage 1 environment check | 2026-10-26 |
| `tests/demo_face.py` | Stage 2 cursor control demo | 2026-10-27 |
| `tests/demo_speech.py` | Stage 3 speech-control demo + actions | 2026-11-03 |
| `tests/test_face_tracker.py` | Stage 2 automated tests (39 tests) | 2026-11-02 |
| `tests/test_speech_engine.py` | Stage 3 speech tests (72 tests) | 2026-11-03 |
| `tests/test_output_controller.py` | Stage 3 action tests (40 tests) | 2026-11-03 |
| `models/face_landmarker.task` | MediaPipe model (3.6 MB) | 2026-10-27 |
| `AI_AGENT_GUIDE.txt` | AI agent instructions | 2026-11-03 |
| `README.md` | User-facing docs | 2026-11-03 |
| `working.md` | This tracker | 2026-11-03 |

---

## Next Step

**Awaiting user approval of Stage 3 Action Execution.**
Physical demo verification of `tests/demo_speech.py` is pending (requires manual microphone + action test).


