# Assistive Control System

A Windows desktop application enabling students with upper-limb amputation to control a
computer using **face/head movements** and **voice commands**. A context-aware decision
engine automatically switches between face control and speech control based on the task
and environment.

## Features

- **Face/Head Tracking** – Cursor control via nose-tip head movement (dead zone, smoothing, acceleration), blink-to-left-click, jaw-open-to-right-click (MediaPipe Face Landmarker)
- **Voice Commands** – Hands-free speech recognition (faster-whisper + WebRTC VAD) — Stage 3
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
python -m pytest tests\test_face_tracker.py -v
```

Expected: **36 passed**.

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

---

## Project Structure

```
assistive-system/
├── main.py                    # Application entry point
├── config.py                  # All thresholds and settings
├── requirements.txt           # Pinned dependencies
├── README.md                  # This file
├── AI_AGENT_GUIDE.txt         # AI agent instructions
├── working.md                 # Project progress tracker
├── face/
│   ├── __init__.py
│   └── face_tracker.py        # FaceTracker class (Stage 2)
├── speech/                    # Voice recognition module (Stage 3)
├── context/                   # Context engine (Stage 4+)
├── output/                    # Output drivers
├── tests/
│   ├── check_env.py           # Stage 1 environment check
│   ├── demo_face.py           # Stage 2 face + cursor demo
│   └── test_face_tracker.py   # Stage 2 automated tests (36 tests)
├── logs/                      # Runtime logs (not committed)
└── models/
    └── face_landmarker.task   # MediaPipe model (3.6 MB, not committed)
```

## License

This project is for academic / assistive-technology research purposes.
