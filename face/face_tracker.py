"""
face/face_tracker.py - Standalone face-tracking module (Stage 2).

Architecture
------------
FaceTracker runs a background thread that:
  1. Captures frames from the webcam via OpenCV.
  2. Feeds each frame into the MediaPipe Face Landmarker (Tasks API).
  3. Extracts:
       - Nose-tip normalised coordinates  --> cursor deltas
       - eyeBlinkLeft / eyeBlinkRight blendshapes --> left-click gesture
       - jawOpen blendshape               --> right-click gesture
       - Frame brightness + landmark stability --> face_confidence
  4. Writes results into a thread-safe dict readable via get_latest_events().

The module NEVER imports pyautogui/pynput and NEVER moves the real mouse.
Cursor movement and click execution are the responsibility of the caller
(demo_face.py or the context engine in later stages).

Coordinate convention
---------------------
  cursor_delta_x > 0  : move right
  cursor_delta_x < 0  : move left
  cursor_delta_y > 0  : move down
  cursor_delta_y < 0  : move up

MediaPipe landmark coordinates are normalised to [0,1] with origin at the
top-left of the frame, so moving the nose right increases x and moving it
down increases y. We mirror x so that the user's left-head-tilt moves the
cursor left from the user's perspective.

Privacy
-------
Frames are processed in memory and immediately discarded. No frame or audio
is written to disk. See config.py: LOG_LEVEL can be set to DEBUG to log
numeric values only (no image data).
"""

import threading
import time
import logging
import collections
from typing import Any, Deque, Dict, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config

logger = logging.getLogger("assistive.face_tracker")


# ---------------------------------------------------------------------------
# Helper: One-Euro Filter (low-pass adaptive filter for cursor smoothing)
# Reference: Casiez et al. 2012 "1€ Filter: A Simple Speed-based Low-Pass
#            Filter for Noisy Input in Interactive Systems"
# ---------------------------------------------------------------------------
class _OneEuroFilter:
    """Scalar 1-Euro filter for real-time noisy signal smoothing."""

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.007, d_cutoff: float = 1.0) -> None:
        self._freq = freq
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._last_time: Optional[float] = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self._freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is not None and self._last_time is not None:
            elapsed = timestamp - self._last_time
            if elapsed > 0:
                self._freq = 1.0 / elapsed
        if timestamp is not None:
            self._last_time = timestamp

        if self._x_prev is None:
            self._x_prev = x
            return x

        # Derivative
        dx = (x - self._x_prev) * self._freq
        a_d = self._alpha(self._d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Signal
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._last_time = None


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------
class FaceTracker:
    """
    Background face-tracking engine.

    Usage
    -----
    tracker = FaceTracker()
    tracker.start()
    ...
    events = tracker.get_latest_events()
    ...
    tracker.stop()

    Events dict keys (always present)
    -----------------------------------
    timestamp       : float   - time.monotonic() of last processed frame
    face_detected   : bool    - whether a face was found this frame
    face_confidence : float   - quality score in [0.0, 1.0]
    cursor_delta_x  : float   - horizontal movement in pixels (right = +)
    cursor_delta_y  : float   - vertical movement in pixels (down = +)
    left_click      : bool    - True for exactly one frame when blink-click fires
    right_click     : bool    - True for exactly one frame when jaw-click fires
    head_position   : dict    - {'x': float, 'y': float} normalised nose pos
    debug           : dict    - raw blendshape scores and other diagnostics
    """

    # MediaPipe nose-tip landmark index (consistent across 478-point model)
    _NOSE_TIP_IDX: int = 4

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Latest event payload written by worker, read by caller
        self._events: Dict[str, Any] = self._empty_events()

        # 1-Euro filters for each cursor axis
        self._filter_x = _OneEuroFilter(freq=config.CAMERA_FPS,
                                        min_cutoff=1.0, beta=0.01)
        self._filter_y = _OneEuroFilter(freq=config.CAMERA_FPS,
                                        min_cutoff=1.0, beta=0.01)

        # Calibrated neutral nose position (normalised coords)
        self._neutral_x: Optional[float] = None
        self._neutral_y: Optional[float] = None

        # Rolling buffer of recent nose positions (for smoothing + stability)
        self._nose_history_x: Deque[float] = collections.deque(
            maxlen=max(config.CURSOR_SMOOTH_WINDOW, config.CONF_STABILITY_WINDOW))
        self._nose_history_y: Deque[float] = collections.deque(
            maxlen=max(config.CURSOR_SMOOTH_WINDOW, config.CONF_STABILITY_WINDOW))

        # Blink gesture state machine
        self._blink_start_time: Optional[float] = None
        self._blink_fired: bool = False      # True once click has been sent
        self._blink_reset_ok: bool = True    # False while waiting for eyes to open

        # Jaw gesture state machine
        self._jaw_start_time: Optional[float] = None
        self._jaw_fired: bool = False
        self._jaw_reset_ok: bool = True

        # Global click cooldown
        self._last_click_time: float = 0.0

        # MediaPipe landmarker (created in start())
        self._landmarker: Optional[mp_vision.FaceLandmarker] = None

        # Thread-safe cache of latest raw BGR video frame
        self._latest_frame: Optional[np.ndarray] = None

        logger.debug("FaceTracker initialised with config: "
                     "dead_zone=%.3f speed=%.1f smooth_window=%d",
                     config.CURSOR_DEAD_ZONE,
                     config.CURSOR_SPEED,
                     config.CURSOR_SMOOTH_WINDOW)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise the landmarker and start the webcam processing thread."""
        if self._running:
            logger.warning("FaceTracker.start() called while already running.")
            return

        model_path = config.FACE_LANDMARKER_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found at: {model_path}\n"
                "Download it with:\n"
                "  python -c \"import urllib.request; urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/face_landmarker"
                "/face_landmarker/float16/1/face_landmarker.task', "
                "'models/face_landmarker.task')\""
            )

        logger.info("Loading Face Landmarker model: %s", model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp_vision.RunningMode.IMAGE,  # synchronous per-frame
            num_faces=config.FACE_MAX_NUM_FACES,
            min_face_detection_confidence=config.FACE_MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.FACE_MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=config.FACE_MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        logger.info("Face Landmarker loaded successfully.")

        self._running = True
        self._thread = threading.Thread(
            target=self._processing_loop,
            name="FaceTracker-Worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("FaceTracker worker thread started.")

    def stop(self) -> None:
        """Signal the worker thread to stop and wait for it to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None

        logger.info("FaceTracker stopped and resources released.")

    def get_latest_events(self) -> Dict[str, Any]:
        """
        Return a shallow copy of the latest event dict.

        Thread-safe: acquires the internal lock briefly then returns a copy
        so the caller can inspect values without holding the lock.
        """
        with self._lock:
            snapshot = dict(self._events)
            # Click flags are edge-triggered: consumed on first read
            self._events["left_click"] = False
            self._events["right_click"] = False
        return snapshot

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Return a copy of the latest raw BGR frame from the webcam.

        Thread-safe. Returns None if no frame has been captured yet.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def calibrate_neutral(self) -> None:
        """
        Set the current nose-tip position as the neutral (rest) position.

        Call this after the user has centred their head in front of the camera.
        The method is safe to call from any thread.
        """
        with self._lock:
            hp = self._events.get("head_position", {})
        if hp:
            self._neutral_x = hp.get("x")
            self._neutral_y = hp.get("y")
            self._filter_x.reset()
            self._filter_y.reset()
            logger.info("Neutral position calibrated: x=%.4f y=%.4f",
                        self._neutral_x, self._neutral_y)
        else:
            logger.warning("calibrate_neutral() called but no face detected yet.")

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _processing_loop(self) -> None:
        """Main loop: open webcam, process frames, write events."""
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.error("Cannot open camera index %d. "
                         "Check that the webcam is connected and not in use.",
                         config.CAMERA_INDEX)
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        logger.info("Camera opened: %dx%d @ %d FPS",
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FPS)))

        consecutive_failures = 0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    logger.warning("Frame read failed (%d consecutive).",
                                   consecutive_failures)
                    if consecutive_failures >= 30:
                        logger.error("Camera appears disconnected. Stopping.")
                        break
                    time.sleep(0.033)
                    continue

                consecutive_failures = 0
                now = time.monotonic()

                # ---- Run MediaPipe detection ---------------------------------
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:
                    result = self._landmarker.detect(mp_img)
                except Exception as exc:
                    logger.warning("MediaPipe detection error: %s", exc)
                    result = None

                # ---- Extract brightness (for confidence) --------------------
                brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

                # ---- Process result -----------------------------------------
                events = self._build_events(result, brightness, now)

                with self._lock:
                    # Preserve edge-triggered click flags if they haven't
                    # been consumed yet (get_latest_events() resets them).
                    if self._events.get("left_click"):
                        events["left_click"] = True
                    if self._events.get("right_click"):
                        events["right_click"] = True
                    self._events = events
                    self._latest_frame = frame.copy()

        finally:
            cap.release()
            logger.info("Camera released.")

    # ------------------------------------------------------------------
    # Event construction
    # ------------------------------------------------------------------

    def _build_events(
        self,
        result: Optional[Any],
        brightness: float,
        now: float,
    ) -> Dict[str, Any]:
        """
        Convert a MediaPipe FaceLandmarkerResult into the events dict.

        Parameters
        ----------
        result     : MediaPipe detection result (may be None on error)
        brightness : Mean pixel intensity of the current frame [0, 255]
        now        : time.monotonic() timestamp

        Returns
        -------
        events dict (see FaceTracker docstring for keys)
        """
        # Default empty events
        events = self._empty_events()
        events["timestamp"] = now

        face_detected = (
            result is not None
            and bool(result.face_landmarks)
        )
        events["face_detected"] = face_detected

        # Raw blendshape scores for diagnostics
        debug: Dict[str, Any] = {
            "brightness": brightness,
            "blink_left_score": 0.0,
            "blink_right_score": 0.0,
            "jaw_open_score": 0.0,
        }

        if not face_detected:
            # No face: reset gesture state machines to prevent ghost clicks
            self._blink_start_time = None
            self._jaw_start_time = None
            events["face_confidence"] = 0.0
            events["debug"] = debug
            return events

        # ----------------------------------------------------------------
        # Nose-tip position → cursor delta
        # ----------------------------------------------------------------
        nose = result.face_landmarks[0][self._NOSE_TIP_IDX]
        nose_x: float = nose.x
        nose_y: float = nose.y

        self._nose_history_x.append(nose_x)
        self._nose_history_y.append(nose_y)

        # Auto-calibrate neutral on first detection
        if self._neutral_x is None:
            self._neutral_x = nose_x
            self._neutral_y = nose_y
            logger.info("Auto-calibrated neutral position: x=%.4f y=%.4f",
                        self._neutral_x, self._neutral_y)

        # Smoothed nose position using moving average
        smooth_x = float(np.mean(list(self._nose_history_x)[-config.CURSOR_SMOOTH_WINDOW:]))
        smooth_y = float(np.mean(list(self._nose_history_y)[-config.CURSOR_SMOOTH_WINDOW:]))

        # Apply 1-Euro filter on top of moving average
        filtered_x = self._filter_x(smooth_x, now)
        filtered_y = self._filter_y(smooth_y, now)

        # Displacement from neutral (normalised).
        # Front-facing camera mirror: user moving head right causes nose_x in frame to decrease,
        # so we invert disp_x to make rightward head motion produce positive cursor delta (+dx).
        # Y: nose_y increases (head tilts down) -> cursor moves DOWN (+dy).
        disp_x = -(filtered_x - self._neutral_x)   # head right = positive dx (cursor right)
        disp_y =  (filtered_y - self._neutral_y)   # head down  = positive dy (cursor down)

        # Dead zone: suppress tiny involuntary movements
        dead = config.CURSOR_DEAD_ZONE
        if abs(disp_x) < dead:
            disp_x = 0.0
        else:
            # Remove dead-zone offset so motion starts from 0 at edge
            disp_x = np.sign(disp_x) * (abs(disp_x) - dead)

        if abs(disp_y) < dead:
            disp_y = 0.0
        else:
            disp_y = np.sign(disp_y) * (abs(disp_y) - dead)
            if disp_y > 0:
                disp_y *= getattr(config, "CURSOR_PITCH_DOWN_GAIN", 1.8)

        # Acceleration: scale displacement by power law then speed
        def _accel(d: float) -> float:
            return float(np.sign(d) * ((abs(d) * 10.0) ** config.CURSOR_ACCEL_EXPONENT)
                         * config.CURSOR_SPEED)

        delta_x = _accel(disp_x)
        delta_y = _accel(disp_y)

        # Clamp to max delta
        max_d = config.CURSOR_MAX_DELTA
        delta_x = float(np.clip(delta_x, -max_d, max_d))
        delta_y = float(np.clip(delta_y, -max_d, max_d))

        events["cursor_delta_x"] = delta_x
        events["cursor_delta_y"] = delta_y
        events["head_position"] = {"x": nose_x, "y": nose_y}

        # ----------------------------------------------------------------
        # Blendshape extraction
        # ----------------------------------------------------------------
        blink_left_score = 0.0
        blink_right_score = 0.0
        jaw_open_score = 0.0

        if result.face_blendshapes:
            bs_map: Dict[str, float] = {
                c.category_name: c.score
                for c in result.face_blendshapes[0]
            }
            # Confirmed blendshape names from live webcam probe (mediapipe 0.10.18).
            blink_left_score  = bs_map.get(config.BLINK_LEFT_BLENDSHAPE, 0.0)
            blink_right_score = bs_map.get(config.BLINK_RIGHT_BLENDSHAPE, 0.0)
            jaw_open_score    = bs_map.get(config.JAW_OPEN_BLENDSHAPE, 0.0)

        debug["blink_left_score"] = blink_left_score
        debug["blink_right_score"] = blink_right_score
        debug["jaw_open_score"] = jaw_open_score

        # ----------------------------------------------------------------
        # Gesture: Long bilateral blink → left-click
        # ----------------------------------------------------------------
        left_click = self._update_blink_gesture(
            blink_left_score, blink_right_score, now
        )
        events["left_click"] = left_click

        # ----------------------------------------------------------------
        # Gesture: Jaw open → right-click
        # ----------------------------------------------------------------
        right_click = self._update_jaw_gesture(jaw_open_score, now)
        events["right_click"] = right_click

        # ----------------------------------------------------------------
        # Face confidence score
        # ----------------------------------------------------------------
        events["face_confidence"] = self._compute_confidence(brightness, now)
        events["debug"] = debug

        return events

    # ------------------------------------------------------------------
    # Gesture state machines
    # ------------------------------------------------------------------

    def _update_blink_gesture(
        self, left_score: float, right_score: float, now: float
    ) -> bool:
        """
        Detect a deliberate bilateral long blink and return True once.

        State machine:
          IDLE          -> BLINKING  when both eyes close (above threshold)
                                     and reset_ok is True
          BLINKING      -> FIRED     after BLINK_HOLD_SECONDS
          FIRED                      returns True exactly once
          FIRED/BLINKING-> IDLE      when both eyes reopen (below reset threshold)
          IDLE          (reset_ok = True again)
        """
        # Click cooldown check (shared between left and right)
        cooldown_ok = (now - self._last_click_time) >= config.CLICK_COOLDOWN_SECONDS

        # Is blink active this frame?
        if config.BLINK_REQUIRE_BILATERAL:
            blink_active = (left_score >= config.BLINK_SCORE_THRESHOLD
                            and right_score >= config.BLINK_SCORE_THRESHOLD)
        else:
            blink_active = (left_score >= config.BLINK_SCORE_THRESHOLD
                            or right_score >= config.BLINK_SCORE_THRESHOLD)

        # Are both eyes open again?
        eyes_open = (left_score < config.BLINK_RESET_THRESHOLD
                     and right_score < config.BLINK_RESET_THRESHOLD)

        fired_this_frame = False

        if blink_active and self._blink_reset_ok:
            # Start / continue timing the blink
            if self._blink_start_time is None:
                self._blink_start_time = now
                self._blink_fired = False

            elapsed = now - self._blink_start_time
            if elapsed >= config.BLINK_HOLD_SECONDS and not self._blink_fired:
                if cooldown_ok:
                    self._blink_fired = True
                    self._blink_reset_ok = False   # wait for eyes to reopen
                    self._last_click_time = now
                    fired_this_frame = True
                    logger.debug("LEFT-CLICK via blink (held %.2fs)", elapsed)
        else:
            if not blink_active:
                # Reset start timer if blink dropped below threshold
                self._blink_start_time = None
                self._blink_fired = False

        # Allow next blink only after both eyes have fully reopened
        if eyes_open:
            self._blink_reset_ok = True

        return fired_this_frame

    def _update_jaw_gesture(self, jaw_score: float, now: float) -> bool:
        """
        Detect a deliberate jaw-open gesture and return True once.

        Uses separate activation / reset thresholds (hysteresis) to
        avoid rapid triggering near the boundary.
        """
        cooldown_ok = (now - self._last_click_time) >= config.CLICK_COOLDOWN_SECONDS

        jaw_open = jaw_score >= config.JAW_OPEN_THRESHOLD
        jaw_closed = jaw_score < config.JAW_CLOSE_THRESHOLD

        fired_this_frame = False

        if jaw_open and self._jaw_reset_ok:
            if self._jaw_start_time is None:
                self._jaw_start_time = now
                self._jaw_fired = False

            elapsed = now - self._jaw_start_time
            if elapsed >= config.JAW_HOLD_SECONDS and not self._jaw_fired:
                if cooldown_ok:
                    self._jaw_fired = True
                    self._jaw_reset_ok = False
                    self._last_click_time = now
                    fired_this_frame = True
                    logger.debug("RIGHT-CLICK via jaw-open (held %.2fs)", elapsed)
        else:
            if not jaw_open:
                self._jaw_start_time = None
                self._jaw_fired = False

        if jaw_closed:
            self._jaw_reset_ok = True

        return fired_this_frame

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _compute_confidence(self, brightness: float, now: float) -> float:
        """
        Compute face_confidence in [0.0, 1.0].

        Components
        ----------
        detect_score (weight 0.5):
            1.0 if a face was detected, 0.0 otherwise.

        stability_score (weight 0.3):
            Measures how stable the nose-tip position is over recent frames.
            Computed as 1 - normalised_std_dev. A jittery or rapidly moving
            nose (poor detection) reduces this score.

        brightness_score (weight 0.2):
            1.0 when mean frame brightness is in [CONF_BRIGHT_MIN, CONF_BRIGHT_MAX].
            Linearly penalised outside that range to signal poor lighting.
        """
        # detect_score: face was detected (caller only calls this when True)
        detect_score = 1.0

        # stability_score
        if len(self._nose_history_x) >= 2:
            std_x = float(np.std(list(self._nose_history_x)[-config.CONF_STABILITY_WINDOW:]))
            std_y = float(np.std(list(self._nose_history_y)[-config.CONF_STABILITY_WINDOW:]))
            # Normalise by a "poor stability" reference (~1 % of frame width)
            std_norm = (std_x + std_y) / 0.02
            stability_score = float(np.clip(1.0 - std_norm, 0.0, 1.0))
        else:
            stability_score = 0.5  # insufficient history; neutral estimate

        # brightness_score
        lo, hi = config.CONF_BRIGHT_MIN, config.CONF_BRIGHT_MAX
        mid = (lo + hi) / 2.0
        if lo <= brightness <= hi:
            brightness_score = 1.0
        elif brightness < lo:
            brightness_score = float(np.clip(brightness / lo, 0.0, 1.0))
        else:
            brightness_score = float(np.clip((255.0 - brightness) / (255.0 - hi), 0.0, 1.0))

        confidence = (
            config.CONF_WEIGHT_DETECT * detect_score
            + config.CONF_WEIGHT_STABLE * stability_score
            + config.CONF_WEIGHT_BRIGHT * brightness_score
        )
        return float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_events() -> Dict[str, Any]:
        """Return a zeroed-out events dict (used when no face is detected)."""
        return {
            "timestamp": 0.0,
            "face_detected": False,
            "face_confidence": 0.0,
            "cursor_delta_x": 0.0,
            "cursor_delta_y": 0.0,
            "left_click": False,
            "right_click": False,
            "head_position": {},
            "debug": {
                "brightness": 0.0,
                "blink_left_score": 0.0,
                "blink_right_score": 0.0,
                "jaw_open_score": 0.0,
            },
        }
