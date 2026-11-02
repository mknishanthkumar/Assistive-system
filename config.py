"""
config.py - Central configuration for the Assistive Control System.

All thresholds, paths, hotkeys, and tuneable parameters live here.
Nothing is hardcoded elsewhere in the project.
"""

from pathlib import Path

# -----------------------------------------------
# Project paths
# -----------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
LOG_DIR: Path = PROJECT_ROOT / "logs"
MODEL_DIR: Path = PROJECT_ROOT / "models"

# Ensure directories exist at import time
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------------------------------
# Safety
# -----------------------------------------------
EMERGENCY_STOP_HOTKEY: str = "ctrl+alt+q"      # Global kill-switch
PYAUTOGUI_FAILSAFE: bool = True                 # Move mouse to corner to abort
PYAUTOGUI_PAUSE: float = 0.05                   # Seconds between PyAutoGUI calls

# -----------------------------------------------
# Camera
# -----------------------------------------------
CAMERA_INDEX: int = 0
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
CAMERA_FPS: int = 30

# -----------------------------------------------
# Face detection (MediaPipe Face Landmarker)
# -----------------------------------------------
FACE_MAX_NUM_FACES: int = 1
FACE_REFINE_LANDMARKS: bool = True
FACE_MIN_DETECTION_CONFIDENCE: float = 0.5
FACE_MIN_TRACKING_CONFIDENCE: float = 0.5

# Head-pose thresholds in degrees (reserved for future pose estimation)
HEAD_YAW_THRESHOLD: float = 15.0
HEAD_PITCH_THRESHOLD: float = 10.0
HEAD_ROLL_THRESHOLD: float = 12.0

# Legacy EAR-based thresholds (kept for reference; Stage 2 uses blendshapes)
EAR_BLINK_THRESHOLD: float = 0.22
BLINK_CONSEC_FRAMES: int = 2
MOUTH_OPEN_THRESHOLD: float = 0.6

# Legacy cursor settings (kept for reference)
CURSOR_SMOOTHING_FACTOR: float = 0.3
CURSOR_SPEED_MULTIPLIER: float = 2.5
DWELL_CLICK_SECONDS: float = 1.5

# -----------------------------------------------
# Stage 2 - Face Landmarker model path
# -----------------------------------------------
# Source : https://storage.googleapis.com/mediapipe-models/face_landmarker/
#          face_landmarker/float16/1/face_landmarker.task
# Licence: Apache 2.0  (https://www.apache.org/licenses/LICENSE-2.0)
# Capabilities: 478 landmarks, 52 blendshape coefficients, iris landmarks,
#               facial transformation matrix.
FACE_LANDMARKER_MODEL_PATH: Path = MODEL_DIR / "face_landmarker.task"

# -----------------------------------------------
# Stage 2 - Cursor movement (nose-tip joystick)
# -----------------------------------------------
# Dead zone: minimum nose displacement (fraction of frame dimension) that is
# ignored to absorb small involuntary head tremors.
# At 640x480: 0.012 * 640 = ~7.7 px horizontal, 0.012 * 480 = ~5.8 px vertical.
CURSOR_DEAD_ZONE: float = 0.012

# Sensitivity multiplier applied to displacement after dead-zone subtraction.
# Higher value = faster cursor for the same head movement amplitude.
CURSOR_SENSITIVITY: float = 8.0

# Maximum allowed per-frame cursor delta in pixels.
# Prevents sudden jumps caused by detection noise or rapid head snaps.
CURSOR_MAX_DELTA: float = 40.0

# Moving-average window length for smoothing nose-tip position history.
# Larger = smoother motion but increased latency. Practical range: 2-10.
CURSOR_SMOOTH_WINDOW: int = 5

# Acceleration exponent applied to the normalised displacement before
# multiplying by CURSOR_SENSITIVITY.
# >1.0 keeps small displacements slow while large displacements accelerate.
# Recommended range: 1.0 (linear) - 2.0 (strong acceleration).
CURSOR_ACCEL_EXPONENT: float = 1.4

# -----------------------------------------------
# Stage 2 - Blink gesture (left-click)
# -----------------------------------------------
# MediaPipe Face Landmarker blendshape category names for blink.
# Full list: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
BLINK_LEFT_BLENDSHAPE: str  = "eyeBlinkLeft"
BLINK_RIGHT_BLENDSHAPE: str = "eyeBlinkRight"

# Blendshape score (0.0-1.0) above which an eye is considered closed.
BLINK_SCORE_THRESHOLD: float = 0.55

# When True, both eyes must exceed threshold simultaneously (bilateral blink)
# to reduce false positives from single-eye squinting.
BLINK_REQUIRE_BILATERAL: bool = True

# Minimum duration (seconds) both eyes must stay closed to register a
# deliberate blink-click. Natural involuntary blinks last ~100-150 ms;
# a threshold > 250 ms reliably filters them out.
BLINK_HOLD_SECONDS: float = 0.35

# Score below which an eye is considered fully open again.
# Lower than BLINK_SCORE_THRESHOLD to create hysteresis and prevent chatter.
BLINK_RESET_THRESHOLD: float = 0.25

# -----------------------------------------------
# Stage 2 - Jaw-open gesture (right-click)
# -----------------------------------------------
JAW_OPEN_BLENDSHAPE: str = "jawOpen"

# Blendshape score above which jaw is considered deliberately open.
JAW_OPEN_THRESHOLD: float = 0.55

# Score below which jaw is considered closed again (hysteresis reset).
# Must be lower than JAW_OPEN_THRESHOLD to prevent rapid re-triggering.
JAW_CLOSE_THRESHOLD: float = 0.30

# Minimum seconds the jaw must remain open before right-click fires.
JAW_HOLD_SECONDS: float = 0.30

# -----------------------------------------------
# Stage 2 - Click cooldown
# -----------------------------------------------
# Minimum seconds that must elapse between any two click events (left or
# right). Prevents duplicate clicks when a gesture lingers near the threshold.
CLICK_COOLDOWN_SECONDS: float = 0.8

# -----------------------------------------------
# Stage 2 - Face confidence score
# -----------------------------------------------
# Confidence is a weighted sum (must sum to 1.0):
#   face_confidence = CONF_WEIGHT_DETECT * detect_score      (0 or 1)
#                   + CONF_WEIGHT_STABLE * stability_score   (0-1)
#                   + CONF_WEIGHT_BRIGHT * brightness_score  (0-1)
#
# detect_score  : 1.0 if face detected this frame, else 0.0
# stability_score: 1 - normalised std-dev of recent nose positions
# brightness_score: 1.0 when mean pixel intensity is in [BRIGHT_MIN, BRIGHT_MAX]
CONF_WEIGHT_DETECT: float = 0.5
CONF_WEIGHT_STABLE: float = 0.3
CONF_WEIGHT_BRIGHT: float = 0.2

# Pixel intensity range (0-255) considered "good" for the brightness component.
CONF_BRIGHT_MIN: float = 40.0   # Below this -> dim room, score penalised
CONF_BRIGHT_MAX: float = 220.0  # Above this -> overexposed, score penalised

# Number of recent nose-tip positions used to compute stability_score.
CONF_STABILITY_WINDOW: int = 6

# -----------------------------------------------
# Speech recognition
# -----------------------------------------------
WHISPER_MODEL_SIZE: str = "base"           # tiny | base | small | medium
WHISPER_DEVICE: str = "cpu"               # cpu | cuda
WHISPER_COMPUTE_TYPE: str = "int8"        # int8 | float16 | float32
WHISPER_LANGUAGE: str = "en"

AUDIO_SAMPLE_RATE: int = 16000            # Hz
AUDIO_CHANNELS: int = 1
AUDIO_BLOCK_DURATION_MS: int = 30         # VAD frame size (10/20/30 ms)
VAD_AGGRESSIVENESS: int = 2               # 0-3, higher = more aggressive

# -----------------------------------------------
# Context engine
# -----------------------------------------------
CONTEXT_SWITCH_COOLDOWN_S: float = 2.0    # Min seconds between mode switches
NOISE_LEVEL_THRESHOLD: float = 0.05       # RMS threshold for "noisy env"
ACTIVE_WINDOW_POLL_INTERVAL_S: float = 0.5

# -----------------------------------------------
# UI / Overlay
# -----------------------------------------------
OVERLAY_OPACITY: float = 0.85
OVERLAY_FONT_SIZE: int = 14
STATUS_UPDATE_INTERVAL_MS: int = 200

# -----------------------------------------------
# Logging
# -----------------------------------------------
LOG_LEVEL: str = "INFO"                   # DEBUG | INFO | WARNING | ERROR
LOG_MAX_BYTES: int = 5 * 1024 * 1024      # 5 MB per log file
LOG_BACKUP_COUNT: int = 3
