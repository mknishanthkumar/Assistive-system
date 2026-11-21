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

# Master cursor speed setting. Higher value = faster cursor, lower = slower.
# Recommended range: 15.0 (slow) to 50.0 (fast). Default: 30.0 (medium, comfortable).
CURSOR_SPEED: float = 30.0
CURSOR_SENSITIVITY: float = CURSOR_SPEED  # alias for backwards compatibility

# Downward pitch gain multiplier. Downward head tilt has naturally ~half
# the range of motion as upward tilt while viewing a screen. This gain ensures
# downward movement feels as responsive as upward movement.
CURSOR_PITCH_DOWN_GAIN: float = 1.8

# Maximum allowed per-frame cursor delta in pixels.
# Prevents sudden jumps caused by detection noise or rapid head snaps.
CURSOR_MAX_DELTA: float = 40.0

# Moving-average window length for smoothing nose-tip position history.
# Larger = smoother motion but increased latency. Practical range: 2-10.
CURSOR_SMOOTH_WINDOW: int = 5

# Acceleration exponent applied to the normalised displacement before
# multiplying by CURSOR_SPEED.
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

# -----------------------------------------------
# Stage 3 - Microphone capture
# -----------------------------------------------
# Device index for sounddevice. None = use system default input device.
MIC_DEVICE_INDEX: int | None = None

AUDIO_SAMPLE_RATE: int = 16000            # Hz  (must be 8000, 16000, 32000, or 48000 for VAD)
AUDIO_CHANNELS: int = 1                   # Mono only for WebRTC VAD
AUDIO_BLOCK_DURATION_MS: int = 30         # VAD frame size (10, 20, or 30 ms)

# -----------------------------------------------
# Stage 3 - Voice Activity Detection (WebRTC VAD)
# -----------------------------------------------
VAD_AGGRESSIVENESS: int = 3               # 0-3, higher = more aggressive filtering
                                          # Set to 3 to maximally suppress ambient noise frames.

# Minimum number of consecutive speech frames needed to begin a speech segment.
# Prevents single noisy frames from starting a segment.  3 frames × 30 ms = 90 ms.
SPEECH_TRIGGER_FRAMES: int = 3

# Minimum total duration (seconds) of a speech segment to be sent for transcription.
# Segments shorter than this are discarded as noise or throat-clearing.
SPEECH_MIN_DURATION_S: float = 0.5

# Duration (seconds) of continuous silence needed AFTER speech to finalise a segment.
# Too short → sentence gets split.  Too long → user waits before result appears.
SPEECH_SILENCE_DURATION_S: float = 0.8

# Number of pre-speech frames kept in a ring buffer and prepended to the segment.
# Captures the onset of speech that VAD may initially miss.
SPEECH_PADDING_FRAMES: int = 10

# Minimum RMS energy (0.0–1.0) a frame must have for the VAD result to be treated as
# real speech.  This dual-gates the WebRTC VAD so that ambient noise — which VAD often
# mis-classifies as speech — is rejected before it enters the speech-segment state machine.
# Typical ambient noise RMS: 0.001–0.015.  Typical quiet speech RMS: 0.03–0.10.
SPEECH_RMS_THRESHOLD: float = 0.02

# Maximum duration (seconds) of a single speech segment before it is forcibly finalised
# and sent for transcription.  Prevents an ever-growing buffer when background noise
# prevents the silence timeout from being reached.
SPEECH_MAX_DURATION_S: float = 15.0

# Whisper no_speech_prob above this value → segment is considered non-speech and
# discarded without command matching or dictation routing.
WHISPER_NO_SPEECH_THRESHOLD: float = 0.60

# -----------------------------------------------
# Stage 3 - Command parser (fuzzy matching)
# -----------------------------------------------
# Minimum difflib.SequenceMatcher ratio to accept a recognised phrase as a command.
# Lower → more tolerant of misrecognition.  Higher → fewer false positives.
FUZZY_MATCH_THRESHOLD: float = 0.70

# -----------------------------------------------
# Stage 3 - Noise-level measurement
# -----------------------------------------------
# Number of recent 30 ms frames used to compute the rolling average RMS noise level.
# 50 frames × 30 ms = 1.5 s window.
NOISE_RMS_WINDOW_FRAMES: int = 50

# -----------------------------------------------
# Stage 3 - Action execution (OutputController)
# -----------------------------------------------
# Minimum speech confidence (0.0-1.0) required to execute a command action.
# Commands below this threshold are logged but NOT executed.
SPEECH_CONFIDENCE_THRESHOLD: float = 0.40

# Per-command cooldown in seconds.  Prevents duplicate actions when the same
# command fires multiple times from a single utterance.
SPEECH_COMMAND_COOLDOWN_S: float = 1.0

# Number of scroll "clicks" for scroll_up / scroll_down commands.
# Positive = up, negative = down.  Tuned to ~25 clicks for ~7 cm of page movement.
SCROLL_AMOUNT: int = 40

# -----------------------------------------------
# Hold-to-scroll (continuous scrolling mode)
# -----------------------------------------------
# Number of scroll "clicks" fired per hold-scroll tick.
# Higher = faster scrolling per tick.  Default: 5 (comfortable speed).
HOLD_SCROLL_AMOUNT: int = 5

# Seconds between each hold-scroll tick (controls scroll speed).
# Lower = faster scrolling.  Recommended range: 0.05 (fast) - 0.3 (slow).
HOLD_SCROLL_INTERVAL_S: float = 0.12

# Safety timeout (seconds): hold-scroll stops automatically after this duration
# even if no 'stop' command is received.  Prevents runaway scrolling.
HOLD_SCROLL_TIMEOUT_S: float = 30.0

# Delay in seconds between individual keystrokes when typing dictation text.
# 0.0 = fastest; increase if the target app drops characters.
DICTATION_TYPE_DELAY_S: float = 0.02

# -----------------------------------------------
# Context engine
# -----------------------------------------------
CONTEXT_SWITCH_COOLDOWN_S: float = 2.0    # Min seconds between mode switches
NOISE_LEVEL_THRESHOLD: float = 0.05       # RMS threshold for "noisy env"
ACTIVE_WINDOW_POLL_INTERVAL_S: float = 0.5

# -----------------------------------------------
# Stage 5 – Decision engine
# -----------------------------------------------
# Minimum wall-clock seconds that must have elapsed between any two
# automatic mode switches.  Prevents flapping when sensor values hover
# near a threshold boundary.
DECISION_MIN_SWITCH_INTERVAL_S: float = 2.0

# A new rule must hold continuously for this many seconds before the engine
# commits to a mode switch.  Guards against transient sensor spikes.
DECISION_HOLD_DURATION_S: float = 1.0

# Face-confidence score below which we consider the face "lost" for the
# purposes of mode selection (independent of FACE_MIN_DETECTION_CONFIDENCE).
DECISION_FACE_CONF_THRESHOLD: float = 0.4

# Noise RMS level at or above which the environment is considered "noisy"
# and voice commands are disabled.  Shares the domain with NOISE_LEVEL_THRESHOLD
# but kept separate so each subsystem can be tuned independently.
DECISION_NOISE_THRESHOLD: float = 0.05

# Frame brightness (mean pixel intensity 0-255) below which lighting is
# considered too poor for reliable face tracking.
DECISION_BRIGHTNESS_THRESHOLD: float = 40.0

# Speech confidence below which the speech engine is considered unreliable
# for the purposes of mode selection.
DECISION_SPEECH_CONF_THRESHOLD: float = 0.3

# -----------------------------------------------
# UI / Overlay
# -----------------------------------------------
OVERLAY_OPACITY: float = 0.85
OVERLAY_FONT_SIZE: int = 14
STATUS_UPDATE_INTERVAL_MS: int = 200

# -----------------------------------------------
# Stage 4 – Integration
# -----------------------------------------------
# DRY_RUN: when True, all OutputController actions are logged but NOT executed.
# Default False (real execution). Override via --dry-run CLI flag.
DRY_RUN: bool = False

# Rate-limit for repeated identical face-clicks (seconds between clicks from face).
# Prevents the face tracker from firing rapid duplicate clicks.
FACE_CLICK_RATE_LIMIT_S: float = 0.8   # matches CLICK_COOLDOWN_SECONDS

# Maximum pixels the cursor may be moved per controller tick.
# Mirrors CURSOR_MAX_DELTA but applied at the integration layer.
CONTROLLER_MAX_CURSOR_DELTA: float = 40.0

# Screen-coordinate clamping margin in pixels (cursor is kept this far from edge).
# Set to 0 to allow reaching full screen edges (e.g. window close buttons, taskbar).
CURSOR_SCREEN_MARGIN: int = 0

# Emergency stop hotkey for pynput GlobalHotKeys format.
EMERGENCY_STOP_HOTKEY_PYNPUT: str = "<ctrl>+<alt>+q"

# -----------------------------------------------
# Stage 4 – Context sensor loop
# -----------------------------------------------
# How often (seconds) the sensor loop publishes a reading to the sensor queue.
SENSOR_POLL_INTERVAL_S: float = 1.0

# Active-window title / process sampling interval (seconds).
# Kept separate so UI-automation calls don't block the sensor thread.
ACTIVE_WINDOW_POLL_INTERVAL_S: float = 0.5

# Noise level RMS threshold below which the environment is considered "quiet".
NOISE_LEVEL_THRESHOLD: float = 0.05

# -----------------------------------------------
# Stage 4 – CSV event logger
# -----------------------------------------------
# Subdirectory under LOG_DIR for per-run CSV files.
CSV_LOG_SUBDIR: str = "runs"

# -----------------------------------------------
# Logging
# -----------------------------------------------
LOG_LEVEL: str = "INFO"                   # DEBUG | INFO | WARNING | ERROR
LOG_MAX_BYTES: int = 5 * 1024 * 1024      # 5 MB per log file
LOG_BACKUP_COUNT: int = 3


# -----------------------------------------------
# Stage 6 – Study conditions
# -----------------------------------------------
# Baseline mode flag for study conditions.
# Values: "face_only" | "speech_only" | "manual_switching" | "auto_switching"
STUDY_CONDITION: str = "auto_switching"

# -----------------------------------------------
# Stage 6 – Mode-switch hotkeys (pynput format)
# -----------------------------------------------
HOTKEY_FORCE_FACE: str = "<ctrl>+<alt>+1"
HOTKEY_FORCE_SPEECH: str = "<ctrl>+<alt>+2"
HOTKEY_FORCE_MANUAL: str = "<ctrl>+<alt>+3"
HOTKEY_FORCE_AUTO: str = "<ctrl>+<alt>+4"
HOTKEY_CYCLE_MODE: str = "<ctrl>+<alt>+<96>"   # backtick
HOTKEY_PAUSE_RESUME: str = "<ctrl>+<alt>+<space>"
HOTKEY_UNDO: str = "<ctrl>+<alt>+z"
OVERLAY_HOTKEY_PYNPUT: str = "<ctrl>+<alt>+h"
OVERLAY_ENABLED: bool = True

# -----------------------------------------------
# Stage 6 – Undo
# -----------------------------------------------
UNDO_HISTORY_SIZE: int = 20
UNDO_IRREVERSIBLE_COMMANDS: list = [
    "delete", "send", "submit", "empty_trash", "format"
]

# -----------------------------------------------
# Stage 6 – Calibration
# -----------------------------------------------
CALIBRATION_SETTINGS_PATH: Path = PROJECT_ROOT / "calibration" / "user_settings.json"
CALIBRATION_HOTKEY_PYNPUT: str = "<ctrl>+<alt>+c"

# -----------------------------------------------
# Stage 6 – Overlay position (added by overlay fix)
# -----------------------------------------------
# Badge position: "top-center" (default) | "top-left" | "top-right"
# NEVER use a corner-hugging position: PyAutoGUI's fail-safe fires when the
# mouse cursor reaches any screen corner.
OVERLAY_POSITION: str = "top-center"
OVERLAY_MARGIN_X: int = 60   # horizontal inset from screen edge (px)
OVERLAY_MARGIN_Y: int = 8    # vertical inset from screen top (px)

# How often (ms) the overlay re-asserts HWND_TOPMOST so other apps
# (fullscreen, Alt+Tab switches) cannot push the pill behind.
OVERLAY_TOPMOST_ASSERT_MS: int = 500

# -----------------------------------------------
# Stage 6 – Cursor fail-safe guard (move_cursor)
# -----------------------------------------------
# Minimum pixels from every screen edge the cursor is clamped to.
# Must be > 0 to prevent PyAutoGUI's corner fail-safe from firing.
# Set >= 10; PyAutoGUI treats the literal pixel (0,0) corner as the trigger.
CURSOR_FAILSAFE_MARGIN: int = 10
