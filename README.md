# Assistive Control System

A Windows desktop application enabling students with upper-limb amputation to control a
computer using **face/head movements** and **voice commands**. A context-aware decision
engine automatically switches between face control and speech control based on the task
and environment.

## Features

- **Face/Head Tracking** – Cursor control via nose-tip head movement (dead zone, smoothing, acceleration), blink-to-left-click, jaw-open-to-right-click (MediaPipe Face Landmarker)
- **Voice Commands** – Hands-free speech recognition (faster-whisper + WebRTC VAD) with 15 predefined commands and free-form dictation mode (Stage 3)
- **Context-Aware Switching** – Auto-selects best input mode — Stage 4+
- **Safety First** – PyAutoGUI fail-safe ON, emergency stop via `Ctrl+Alt+Q`
- **100% Local** – All processing on-device. No frames or audio saved.

## Requirements

| Item | Requirement |
|------|------------|
| OS | Windows 10 or Windows 11 |
| Python | 3.11 (64-bit) |
| Hardware | Webcam + Microphone |
| Model | `models/face_landmarker.task` (downloaded automatically) |

---

## Quick Start

### 1. Clone / download the project

```powershell
cd C:\Users\<you>\Desktop
git clone <repo-url> assistive-system
cd assistive-system
```

### 2. Create and activate the virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
pip install pytest   # for running tests
```

### 4. Download the Face Landmarker model

```powershell
python -c "
import urllib.request, pathlib
url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
urllib.request.urlretrieve(url, 'models/face_landmarker.task')
print('Downloaded:', pathlib.Path('models/face_landmarker.task').stat().st_size, 'bytes')
"
```

Expected output: `Downloaded: 3758596 bytes` (3.6 MB).

Licence: Apache 2.0 — https://www.apache.org/licenses/LICENSE-2.0

### 5. Verify the environment

```powershell
python tests\check_env.py
```

Expected: all 14 libraries PASS, webcam PASS, microphone PASS.

### 6. Run automated tests

```powershell
python -m pytest tests\ -v
```

Expected: **151 passed, 0 failed** (39 face tests + 72 speech tests + 40 output controller tests).
Or run individual test suites:
```powershell
python -m pytest tests\test_face_tracker.py -v      # 39 passed
python -m pytest tests\test_speech_engine.py -v     # 72 passed
python -m pytest tests\test_output_controller.py -v  # 40 passed
```

### 7. Run the main application

```powershell
python main.py
```

---

## Face Tracking Demo (Stage 2)

```powershell
python tests\demo_face.py
```

A 3-second safety countdown is shown before real cursor control begins.

### Controls

| Action | Effect |
|--------|--------|
| Move your head | Moves the mouse cursor |
| Press **C** | Re-calibrate neutral head position |
| Press **ESC** | Exit the demo |
| Deliberate bilateral blink, hold ~350 ms | **Left-click** |
| Open jaw wide, hold ~300 ms | **Right-click** |

### How to calibrate

1. Sit comfortably in front of the camera with your head in its natural position.
2. When the demo starts, it auto-calibrates to your nose position.
3. Press **C** at any time to re-calibrate if the cursor drifts.

### Cursor control

The nose-tip is used as a joystick. Moving your head away from the neutral position moves the cursor in that direction. The further you move, the faster the cursor accelerates (power-law acceleration). A dead zone prevents small involuntary tremors from moving the cursor.

All cursor settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CURSOR_DEAD_ZONE` | 0.012 | Min displacement to activate cursor (fraction of frame) |
| `CURSOR_SENSITIVITY` | 8.0 | Pixel multiplier per unit displacement |
| `CURSOR_MAX_DELTA` | 40.0 px | Maximum per-frame cursor movement |
| `CURSOR_SMOOTH_WINDOW` | 5 frames | Moving-average smoothing window |
| `CURSOR_ACCEL_EXPONENT` | 1.4 | Power-law acceleration exponent |

### Facial gestures

| Gesture | Action | Config keys |
|---------|--------|-------------|
| Bilateral blink, hold ≥ 350 ms | Left-click | `BLINK_HOLD_SECONDS`, `BLINK_SCORE_THRESHOLD` |
| Jaw open wide, hold ≥ 300 ms | Right-click | `JAW_HOLD_SECONDS`, `JAW_OPEN_THRESHOLD` |

Short normal blinks (~150 ms) are **not** triggered — only deliberate long blinks.

### Face confidence score

The HUD shows a confidence score [0.0–1.0] calculated from:
- **50%** — face detected (binary)
- **30%** — nose-tip position stability (less jitter = higher score)
- **20%** — frame brightness (optimal 40–220 out of 255)

Green = good (≥0.7), orange = marginal (≥0.4), red = poor (<0.4).

---

## Speech Control Demo (Stage 3)

```powershell
# Run with real Windows action execution:
python tests\demo_speech.py

# Or run in dry-run mode (recognition only, no actions executed):
python tests\demo_speech.py --dry-run
```

Runs the standalone speech control module with live microphone input and executes recognized commands as real Windows actions (mouse clicks, hotkeys, window management, scrolling, browser launch, volume control, and dictation typing) via `OutputController`.

> **Note on first run**: The Faster-Whisper `base` model (~150 MB) will be downloaded from HuggingFace Hub on initial execution and cached in `~/.cache/huggingface/`. Subsequent startups load from local cache in 2–4 seconds.

### Controls & Hotkeys

| Action / Hotkey | Effect |
|-----------------|--------|
| Speak any supported command | Recognizes command, prints details, and executes the Windows action |
| Say **"start dictation"** | Activates free-form dictation mode (recognized speech is typed into the focused window) |
| Say **"stop dictation"** | Deactivates dictation mode and returns to command mode |
| Say **"stop control"** | Emergency voice command to halt action execution |
| **Ctrl+Alt+Q** | Global emergency hotkey — instantly stops all action execution |
| **Ctrl+C** | Gracefully stops the engine, releases microphone stream and keyboard hooks |

### Supported Voice Commands (17 Commands, 24 Phrases)

The engine supports 17 core commands across mouse actions, window management, system keys, and dictation:

| Category | Command ID | Spoken Phrase / Aliases | Windows Action Executed |
|----------|------------|-------------------------|-------------------------|
| **Mouse Actions** | `click` | "click", "left click", "please click" | Left click at current cursor position |
| | `double_click` | "double click" | Double left click |
| | `right_click` | "right click" | Right click / context menu |
| | `scroll_up` | "scroll up" | Scroll up (`SCROLL_AMOUNT = 3`) |
| | `scroll_down` | "scroll down" | Scroll down (`SCROLL_AMOUNT = 3`) |
| **Window Management** | `open_browser` | "open browser", "browser" | Opens default browser via `webbrowser.open` |
| | `close_window` | "close window", "exit window" | Closes active window (`Alt+F4`) |
| | `switch_window` | "switch window", "next window" | Switches window (`Alt+Tab`) |
| | `minimize_window` | "minimize window", "minimize" | Minimizes active window (`Win+Down`) |
| **System & Keystrokes** | `volume_up` | "volume up" | Increases master volume (`VK_VOLUME_UP`) |
| | `volume_down` | "volume down" | Decreases master volume (`VK_VOLUME_DOWN`) |
| | `mute` | "mute", "unmute" | Toggles audio mute (`VK_VOLUME_MUTE`) |
| | `press_enter` | "press enter", "enter" | Sends Enter key (`pyautogui.press('enter')`) |
| | `press_escape` | "press escape", "escape" | Sends Escape key (`pyautogui.press('escape')`) |
| **Dictation & Safety** | `start_dictation` | "start dictation", "dictation mode" | Begins dictation (types into focused field) |
| | `stop_dictation` | "stop dictation", "end dictation" | Stops dictation mode |
| | `stop_control` | "stop control" | Software stop of action execution |

### Command Matching Pipeline

Spoken audio is segmented using WebRTC VAD and transcribed with Faster-Whisper. Transcriptions are evaluated through a three-tier matching pipeline:
1. **Exact match**: Direct equality with canonical command phrases.
2. **Whole-word containment**: Detects command phrases embedded within natural conversational speech (e.g., "please click now" matches `click` with 0.90 confidence).
3. **Fuzzy sequence matching**: Uses Gestalt pattern matching (`difflib`) against predefined aliases with a configurable threshold (`FUZZY_MATCH_THRESHOLD = 0.70`). Unrelated phrases scoring below threshold are rejected.

### Dictation Mode

- When activated via "start dictation", continuous speech is emitted as `dictation_text` events rather than command matches.
- The `OutputController` types this text directly into whichever window or input field currently has keyboard focus.
- The command "stop dictation" is always intercepted and processed immediately to return the engine to command mode.

### Audio, Speech & Action Settings in `config.py`

| Setting | Default | Description |
|---------|---------|-------------|
| `MIC_DEVICE_INDEX` | `None` | Microphone index (`None` = OS default device) |
| `AUDIO_SAMPLE_RATE` | `16000` | Sample rate (Hz) required for WebRTC VAD and Whisper |
| `AUDIO_BLOCK_DURATION_MS` | `30` | Frame size (ms) fed to VAD (10, 20, or 30 ms) |
| `VAD_AGGRESSIVENESS` | `2` | WebRTC VAD aggressiveness filter level (0 to 3) |
| `WHISPER_MODEL_SIZE` | `"base"` | Faster-Whisper model size (`tiny`, `base`, `small`) |
| `WHISPER_DEVICE` | `"cpu"` | Inference hardware device (`cpu` or `cuda`) |
| `WHISPER_COMPUTE_TYPE` | `"int8"` | Model quantization type (`int8` or `float32`) |
| `SPEECH_TRIGGER_FRAMES` | `3` | Consecutive speech frames needed to trigger speech onset |
| `SPEECH_SILENCE_DURATION_S` | `0.8` | Silence duration before marking utterance complete |
| `SPEECH_MIN_DURATION_S` | `0.4` | Minimum audio length to send to Whisper (filters coughs/clicks) |
| `FUZZY_MATCH_THRESHOLD` | `0.70` | Minimum similarity score required to accept command |
| `NOISE_RMS_WINDOW_FRAMES` | `50` | Rolling frame window for calculating background noise RMS |
| `SPEECH_CONFIDENCE_THRESHOLD` | `0.40` | Min Whisper confidence required to execute actions |
| `SPEECH_COMMAND_COOLDOWN_S` | `1.0` | Cooldown period between duplicate command executions |
| `SCROLL_AMOUNT` | `3` | Number of clicks for scroll up/down actions |
| `DICTATION_TYPE_DELAY_S` | `0.02` | Typing delay per character in dictation mode |

### Manual Verification Guide for Speech Actions

To verify real Windows action execution safely:

1. **Dry-Run Check (Zero Risk)**:
   ```powershell
   python tests\demo_speech.py --dry-run
   ```
   Speak voice commands (e.g., `"click"`, `"open browser"`, `"volume up"`). The console prints `[DRY-RUN] Would execute: ...` without altering Windows state.

2. **Real Action Testing**:
   ```powershell
   python tests\demo_speech.py
   ```
   - **Clicking**: Move cursor over a desktop icon or empty area, say `"click"` or `"right click"`.
   - **Scrolling**: Focus any scrollable page, say `"scroll down"` and `"scroll up"`.
   - **Browser launch**: Say `"open browser"` — default browser launches.
   - **Volume controls**: Say `"volume up"`, `"volume down"`, or `"mute"` — Windows volume overlay appears.
   - **Dictation**: Focus Notepad or a text box, say `"start dictation"`, speak a sentence, observe text typed into the field, then say `"stop dictation"`.
   - **Emergency Stop**: Press **Ctrl+Alt+Q** anytime — console reports `EMERGENCY STOP ACTIVATED` and blocks all further actions.
   - **Exit**: Press **Ctrl+C** to stop the demo cleanly.

---

## Safety

| Feature | Detail |
|---------|--------|
| PyAutoGUI fail-safe | Move mouse to any screen corner to abort immediately |
| Emergency stop | Press `Ctrl+Alt+Q` at any time |
| Deliberate gesture only | Long blink and jaw-hold required to prevent accidental clicks |
| Click cooldown | 0.8 s minimum between consecutive clicks |
| Privacy | No camera frames or audio saved to disk |

---

## Troubleshooting

### Webcam not detected
- Check that the webcam is connected and not used by another application.
- Change `CAMERA_INDEX` in `config.py` (try `1`, `2`, etc.).

### Model not found error
- Run the download command in Step 4 above.
- Verify: `python -c "import pathlib; print(pathlib.Path('models/face_landmarker.task').stat().st_size)"`
- Expected: `3758596` bytes.

### Poor tracking / cursor jitter
- Improve room lighting (avoid backlight or very dark conditions).
- Re-calibrate by pressing **C** in the demo.
- Reduce `CURSOR_SENSITIVITY` in `config.py`.
- Increase `CURSOR_SMOOTH_WINDOW` for smoother but slower response.
- Increase `CURSOR_DEAD_ZONE` to ignore more tremor.

### Accidental clicks
- Increase `BLINK_HOLD_SECONDS` (default 0.35 s) to require a longer blink.
- Increase `BLINK_SCORE_THRESHOLD` to require eyes to close more.
- Increase `JAW_OPEN_THRESHOLD` to require wider jaw opening.
- Increase `CLICK_COOLDOWN_SECONDS` to add more time between clicks.

### Cursor moves in wrong direction
- Press **C** to re-calibrate while sitting in your natural head position.
- Ensure your camera is directly in front of you (not to the side).

### MediaPipe warnings on startup
- Warnings like `inference_feedback_manager` are normal and harmless.
- They come from TensorFlow Lite internals and do not affect tracking.

### Microphone not detected / wrong input device
- Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list all available audio devices and indices.
- Set `MIC_DEVICE_INDEX` in `config.py` to the desired device index (default `None` uses system default).

### Faster-Whisper download on first run
- An active internet connection is required during the first launch to download the Whisper model (~150 MB).
- HuggingFace symlink warnings on Windows are harmless; weights are properly cached locally in `~/.cache/huggingface/`.

### False speech triggers or speech cutting off
- Adjust `VAD_AGGRESSIVENESS` in `config.py` (0 is least aggressive, 3 is most aggressive). Use `3` in noisy rooms.
- Increase `SPEECH_SILENCE_DURATION_S` (e.g. from 0.8 to 1.0 or 1.2 s) if pauses between words cause commands to split prematurely.
- Increase `SPEECH_MIN_DURATION_S` if short ambient noises (coughs, key taps) trigger false transcriptions.

### Command recognition accuracy
- Speak naturally at normal conversational volume.
- Adjust `FUZZY_MATCH_THRESHOLD` in `config.py` (lower to 0.65 for more lenient matching, raise to 0.75 for stricter matching).

---

## Project Structure

```
assistive-system/
├── main.py                    # Application entry point
├── config.py                  # All thresholds and settings
├── requirements.txt           # Pinned dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # User-facing documentation
├── AI_AGENT_GUIDE.txt         # AI agent instructions
├── working.md                 # Project progress tracker
├── face/
│   ├── __init__.py
│   └── face_tracker.py        # FaceTracker class (Stage 2)
├── speech/
│   ├── __init__.py
│   └── speech_engine.py       # SpeechEngine class (Stage 3)
├── output/                    # Output execution drivers
│   ├── __init__.py
│   └── controller.py          # OutputController (Stage 3 actions)
├── context/                   # Context engine (Stage 4+)
│   └── __init__.py
├── tests/
│   ├── check_env.py           # Stage 1 environment check
│   ├── demo_face.py           # Stage 2 face + cursor demo
│   ├── demo_speech.py         # Stage 3 speech control demo + actions
│   ├── test_face_tracker.py   # Stage 2 automated tests (39 tests)
│   ├── test_speech_engine.py  # Stage 3 speech tests (72 tests)
│   └── test_output_controller.py # Stage 3 action tests (40 tests)
├── logs/                      # Runtime logs (not committed)
└── models/
    └── face_landmarker.task   # MediaPipe model (3.6 MB, not committed)
```

## License

This project is for academic / assistive-technology research purposes.
