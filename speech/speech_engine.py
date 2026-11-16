"""
speech/speech_engine.py - Standalone speech recognition engine (Stage 3).

Architecture
------------
Three-thread pipeline that keeps microphone capture non-blocking:

  1. **Audio callback** (sounddevice thread)
     Captures 30 ms int16 frames and pushes them into _audio_queue.

  2. **VAD processing thread** (_vad_processing_loop)
     Reads frames from _audio_queue, runs WebRTC VAD, accumulates speech
     segments, computes rolling RMS noise level.  When a segment ends
     (silence timeout), the concatenated audio bytes are placed into
     _transcription_queue.

  3. **Transcription thread** (_transcription_loop)
     Waits for audio segments, runs Faster-Whisper, normalises text,
     parses commands or emits dictation events, and writes the result
     into the shared _events dict.

Thread safety
  All reads/writes of _events go through self._lock.
  Edge-triggered fields (command, command_phrase, dictation_text,
  transcription_text) are consumed on get_latest_events() so the same
  event is never returned twice.

Public API
----------
  engine = SpeechEngine()
  engine.start()          # loads model, opens mic, starts threads
  events = engine.get_latest_events()   # thread-safe snapshot
  engine.stop()           # releases everything
"""

import collections
import difflib
import logging
import math
import queue
import re
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import webrtcvad

import config

logger = logging.getLogger("assistive.speech_engine")


# ---------------------------------------------------------------------------
# Command definitions
# ---------------------------------------------------------------------------
# Mapping of normalised command ID -> list of canonical phrases.
# Phrases MUST be lower-case, no punctuation.
COMMANDS: Dict[str, List[str]] = {
    # Mouse navigation
    "click":             ["click", "left click"],
    "double_click":      ["double click"],
    "right_click":       ["right click"],
    "scroll_up":         ["scroll up"],
    "scroll_down":       ["scroll down"],
    # Window management
    "open_browser":      ["open browser"],
    "close_window":      ["close window", "exit window"],
    "switch_window":     ["switch window", "next window"],
    "minimize_window":   ["minimize window", "minimize"],
    # System & keystrokes
    "volume_up":         ["volume up"],
    "volume_down":       ["volume down"],
    "mute":              ["mute", "unmute"],
    "press_enter":       ["press enter", "enter"],
    "press_escape":      ["press escape", "escape"],
    # Dictation & control
    "start_dictation":   ["start dictation"],
    "stop_dictation":    ["stop dictation"],
    "stop_control":      ["stop control"],
}


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
def _normalize_text(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text


# ---------------------------------------------------------------------------
# Whisper hallucination filter
# ---------------------------------------------------------------------------
# Known phrases Whisper produces when fed silence, noise, or very short audio.
# These are common false outputs documented in openai-whisper and faster-whisper
# issue trackers.  Match is performed on the normalised (lower-case, no-punct) text.
_WHISPER_HALLUCINATION_PHRASES: frozenset = frozenset({
    "",
    "thank you",
    "thank you for watching",
    "thanks for watching",
    "you",
    "hmm",
    "uh",
    "um",
    "okay",
    "so",
    "you know",
    "please subscribe",
    "subtitles by",
    "blank audio",
    "music",
    "applause",
    "laughter",
    "silence",
    "noise",
})


def _is_whisper_hallucination(text: str) -> bool:
    """
    Return True if *text* is a known Whisper hallucination / noise artefact.

    Checks:
    1. Normalised text is in the known hallucination set.
    2. Normalised text is 1 character or fewer (single-letter noise).
    3. Text is a Whisper bracket-encoded token (e.g. "[BLANK_AUDIO]", "[Music]").
    4. Text consists entirely of punctuation / filler characters (alpha ratio < 0.2).
    """
    norm = _normalize_text(text)
    if not norm or len(norm) <= 1:
        return True
    if norm in _WHISPER_HALLUCINATION_PHRASES:
        return True
    # Reject Whisper bracket-encoded tokens: text surrounded by [ ] or ( )
    stripped = text.strip()
    if (stripped.startswith("[") and stripped.endswith("]")) or \
       (stripped.startswith("(") and stripped.endswith(")")):
        return True
    # Reject if more than 80 % of the original text is non-alphanumeric
    # (e.g. "... ... ...")
    alpha_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.2:
        return True
    return False


# ---------------------------------------------------------------------------
# Command parser (fuzzy matching via difflib)
# ---------------------------------------------------------------------------
def parse_command(
    text: str,
    threshold: Optional[float] = None,
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Match *text* against the 15 known voice commands.

    Parameters
    ----------
    text      : raw recognised phrase from Whisper
    threshold : minimum SequenceMatcher ratio to accept (default from config)

    Returns
    -------
    (command_id, matched_phrase, match_score) or (None, None, 0.0)
    """
    if threshold is None:
        threshold = config.FUZZY_MATCH_THRESHOLD

    normalized = _normalize_text(text)
    if not normalized:
        return None, None, 0.0

    best_cmd: Optional[str] = None
    best_phrase: Optional[str] = None
    best_score: float = 0.0

    for cmd_id, phrases in COMMANDS.items():
        for phrase in phrases:
            norm_phrase = _normalize_text(phrase)

            # --- exact match ---
            if normalized == norm_phrase:
                return cmd_id, phrase, 1.0

            # --- whole-word containment (handles "please click now" → "click") ---
            # Check if command phrase appears as contiguous whole words in text.
            text_words = normalized.split()
            phrase_words = norm_phrase.split()
            for i in range(len(text_words) - len(phrase_words) + 1):
                if text_words[i : i + len(phrase_words)] == phrase_words:
                    # Whole-word match found; score high but below exact (1.0).
                    contain_score = 0.90
                    if contain_score > best_score:
                        best_score = contain_score
                        best_cmd = cmd_id
                        best_phrase = phrase
                    break

            # --- fuzzy match ---
            ratio = difflib.SequenceMatcher(
                None, normalized, norm_phrase,
            ).ratio()
            if ratio > best_score:
                best_score = ratio
                best_cmd = cmd_id
                best_phrase = phrase

    if best_score >= threshold:
        return best_cmd, best_phrase, best_score

    return None, None, best_score


# ---------------------------------------------------------------------------
# SpeechEngine
# ---------------------------------------------------------------------------
class SpeechEngine:
    """
    Background speech-recognition engine.

    Usage
    -----
    engine = SpeechEngine()
    engine.start()
    ...
    events = engine.get_latest_events()
    ...
    engine.stop()

    Events dict keys (always present)
    ----------------------------------
    timestamp         : float        - time.monotonic() of last update
    command           : str | None   - normalised command id (edge-triggered)
    command_phrase    : str | None   - canonical phrase that matched
    dictation_text    : str | None   - recognised text in dictation mode
    dictation_mode    : bool         - whether dictation is currently active
    noise_level       : float        - rolling RMS in [0.0, 1.0]
    speech_confidence : float        - [0.0, 1.0] from Whisper avg_log_prob
    speech_detected   : bool         - True while VAD detects speech frames
    transcription_text: str | None   - raw Whisper output (edge-triggered)
    vad_status        : str          - "silence" | "speech" | "processing"
    event_id          : int          - sequence counter (monotonically increases)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: bool = False

        # Audio stream (sounddevice)
        self._stream: Optional[sd.RawInputStream] = None

        # Inter-thread queues
        self._audio_queue: queue.Queue = queue.Queue(maxsize=300)
        self._transcription_queue: queue.Queue = queue.Queue(maxsize=10)

        # Worker threads
        self._vad_thread: Optional[threading.Thread] = None
        self._transcription_thread: Optional[threading.Thread] = None

        # Faster-Whisper model (loaded once in start())
        self._whisper_model: Optional[Any] = None
        self._model_loaded = threading.Event()

        # WebRTC VAD instance
        self._vad: Optional[webrtcvad.Vad] = None

        # Frame geometry (set in start())
        self._frame_samples: int = 0
        self._frame_bytes: int = 0

        # Dictation mode
        self._dictation_mode: bool = False

        # Event counter
        self._event_id: int = 0

        # Noise RMS rolling window
        self._noise_frames: Deque[float] = collections.deque(
            maxlen=config.NOISE_RMS_WINDOW_FRAMES,
        )

        # Latest event snapshot (written by workers, read by caller)
        self._events: Dict[str, Any] = self._empty_events()

        logger.debug(
            "SpeechEngine initialised: sample_rate=%d vad_aggr=%d model=%s",
            config.AUDIO_SAMPLE_RATE,
            config.VAD_AGGRESSIVENESS,
            config.WHISPER_MODEL_SIZE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load Whisper model, open the microphone, and start worker threads."""
        if self._running:
            logger.warning("SpeechEngine.start() called while already running.")
            return

        # ── VAD ───────────────────────────────────────────────────────
        self._vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

        # ── Whisper model ─────────────────────────────────────────────
        logger.info(
            "Loading Whisper model: size=%s device=%s compute_type=%s "
            "(first run downloads from HuggingFace Hub; please wait) ...",
            config.WHISPER_MODEL_SIZE,
            config.WHISPER_DEVICE,
            config.WHISPER_COMPUTE_TYPE,
        )
        try:
            from faster_whisper import WhisperModel  # type: ignore

            t0 = time.monotonic()
            self._whisper_model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )
            elapsed = time.monotonic() - t0
            self._model_loaded.set()
            logger.info("Whisper model loaded in %.1f s.", elapsed)
        except Exception as exc:
            logger.error("Failed to load Whisper model: %s", exc)
            raise

        # ── Frame geometry ────────────────────────────────────────────
        self._frame_samples = int(
            config.AUDIO_SAMPLE_RATE * config.AUDIO_BLOCK_DURATION_MS / 1000
        )
        self._frame_bytes = self._frame_samples * 2  # int16 = 2 bytes/sample

        # ── Audio stream ──────────────────────────────────────────────
        self._running = True

        try:
            self._stream = sd.RawInputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                blocksize=self._frame_samples,
                device=config.MIC_DEVICE_INDEX,
                dtype="int16",
                channels=config.AUDIO_CHANNELS,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info(
                "Audio stream started: %d Hz, %d ms frames, device=%s",
                config.AUDIO_SAMPLE_RATE,
                config.AUDIO_BLOCK_DURATION_MS,
                config.MIC_DEVICE_INDEX if config.MIC_DEVICE_INDEX is not None
                else "default",
            )
        except Exception as exc:
            logger.error("Failed to start audio stream: %s", exc)
            self._running = False
            raise

        # ── Worker threads ────────────────────────────────────────────
        self._vad_thread = threading.Thread(
            target=self._vad_processing_loop,
            name="SpeechEngine-VAD",
            daemon=True,
        )
        self._vad_thread.start()

        self._transcription_thread = threading.Thread(
            target=self._transcription_loop,
            name="SpeechEngine-Transcription",
            daemon=True,
        )
        self._transcription_thread.start()

        logger.info("SpeechEngine started (all threads running).")

    def stop(self) -> None:
        """Stop all threads, close the microphone, and release resources."""
        self._running = False

        # Stop audio stream first (prevents new frames)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        # Unblock worker threads via sentinel values
        for q in (self._audio_queue, self._transcription_queue):
            try:
                q.put(None, timeout=0.1)
            except queue.Full:
                pass

        if self._vad_thread is not None:
            self._vad_thread.join(timeout=5.0)
            self._vad_thread = None

        if self._transcription_thread is not None:
            self._transcription_thread.join(timeout=5.0)
            self._transcription_thread = None

        self._whisper_model = None
        self._vad = None
        self._model_loaded.clear()

        logger.info("SpeechEngine stopped and resources released.")

    def get_latest_events(self) -> Dict[str, Any]:
        """
        Return a snapshot of the latest events.

        Thread-safe.  Edge-triggered fields (command, command_phrase,
        dictation_text, transcription_text) are consumed on first read
        so the same event is never returned twice.
        """
        with self._lock:
            snapshot = dict(self._events)
            # Consume edge-triggered fields
            self._events["command"] = None
            self._events["command_phrase"] = None
            self._events["dictation_text"] = None
            self._events["transcription_text"] = None
        return snapshot

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_events() -> Dict[str, Any]:
        """Return a default events dict with all keys present."""
        return {
            "timestamp": 0.0,
            "command": None,
            "command_phrase": None,
            "dictation_text": None,
            "dictation_mode": False,
            "noise_level": 0.0,
            "speech_confidence": 0.0,
            "speech_detected": False,
            "transcription_text": None,
            "vad_status": "silence",
            "event_id": 0,
        }

    @staticmethod
    def _compute_rms(frame_data: bytes) -> float:
        """
        Compute RMS of an int16 audio frame, normalised to [0.0, 1.0].

        Scale: 0.0 = digital silence, 1.0 = maximum int16 amplitude (32 767).
        Typical quiet room ≈ 0.01–0.03, normal speech ≈ 0.05–0.20.
        """
        samples = np.frombuffer(frame_data, dtype=np.int16).astype(np.float64)
        if len(samples) == 0:
            return 0.0
        rms = math.sqrt(float(np.mean(samples ** 2)))
        return min(rms / 32767.0, 1.0)

    # ------------------------------------------------------------------
    # Audio callback (runs on the sounddevice I/O thread)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: bytes,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Push raw audio frames into _audio_queue (non-blocking)."""
        if status:
            logger.warning("Audio stream status: %s", status)
        if self._running:
            try:
                self._audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                pass  # drop frame rather than blocking the I/O thread

    # ------------------------------------------------------------------
    # VAD processing thread
    # ------------------------------------------------------------------

    def _vad_processing_loop(self) -> None:
        """Read frames, run VAD, assemble speech segments."""
        speech_frames: List[bytes] = []
        silence_count: int = 0
        speech_trigger_count: int = 0
        is_speaking: bool = False

        # Pre-compute frame counts from config durations
        min_speech_frames = max(
            1,
            int(config.SPEECH_MIN_DURATION_S * 1000 / config.AUDIO_BLOCK_DURATION_MS),
        )
        silence_timeout_frames = max(
            1,
            int(config.SPEECH_SILENCE_DURATION_S * 1000 / config.AUDIO_BLOCK_DURATION_MS),
        )
        max_speech_frames = max(
            min_speech_frames + 1,
            int(config.SPEECH_MAX_DURATION_S * 1000 / config.AUDIO_BLOCK_DURATION_MS),
        )
        trigger_frames = config.SPEECH_TRIGGER_FRAMES
        padding = config.SPEECH_PADDING_FRAMES
        rms_threshold = config.SPEECH_RMS_THRESHOLD

        # Ring buffer for pre-speech onset padding
        ring_buffer: Deque[bytes] = collections.deque(maxlen=padding)

        logger.debug(
            "VAD loop started: min_speech=%d frames, silence_timeout=%d frames, "
            "max_speech=%d frames, trigger=%d frames, padding=%d frames, "
            "rms_threshold=%.4f",
            min_speech_frames, silence_timeout_frames, max_speech_frames,
            trigger_frames, padding, rms_threshold,
        )

        while self._running:
            try:
                frame_data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame_data is None:
                break

            # ── Noise RMS ─────────────────────────────────────────────
            rms = self._compute_rms(frame_data)
            self._noise_frames.append(rms)
            avg_noise = (
                sum(self._noise_frames) / len(self._noise_frames)
                if self._noise_frames else 0.0
            )

            # ── WebRTC VAD + RMS dual gate ─────────────────────────────
            # WebRTC VAD alone classifies ANY non-silent audio (including ambient
            # microphone noise with RMS ~0.005-0.015) as speech at aggressiveness<3.
            # Combining VAD with an RMS energy gate suppresses those false positives:
            # only frames that are both VAD=True AND loud enough count as speech.
            try:
                vad_result = self._vad.is_speech(frame_data, config.AUDIO_SAMPLE_RATE)
            except Exception:
                vad_result = False

            is_active_speech = vad_result and (rms >= rms_threshold)

            # speech_detected reflects confirmed speaking state (is_speaking), not
            # raw per-frame VAD output, so the status display is accurate.
            with self._lock:
                self._events["noise_level"] = avg_noise
                self._events["speech_detected"] = is_speaking
                self._events["timestamp"] = time.monotonic()

            # ── State machine ─────────────────────────────────────────
            if not is_speaking:
                # Waiting for speech onset
                if is_active_speech:
                    speech_trigger_count += 1
                    ring_buffer.append(frame_data)
                    if speech_trigger_count >= trigger_frames:
                        # Speech onset confirmed
                        is_speaking = True
                        speech_frames = list(ring_buffer)
                        ring_buffer.clear()
                        silence_count = 0
                        with self._lock:
                            self._events["vad_status"] = "speech"
                            self._events["speech_detected"] = True
                        logger.debug("Speech onset detected.")
                else:
                    speech_trigger_count = 0
                    ring_buffer.append(frame_data)
            else:
                # Currently in a speech segment
                speech_frames.append(frame_data)

                # Force-finalise if segment exceeds maximum allowed duration
                if len(speech_frames) >= max_speech_frames:
                    logger.debug(
                        "Max segment duration reached (%d frames = %.1f s) — "
                        "force-finalising.",
                        len(speech_frames),
                        len(speech_frames) * config.AUDIO_BLOCK_DURATION_MS / 1000,
                    )
                    segment_audio = b"".join(speech_frames)
                    try:
                        self._transcription_queue.put_nowait(segment_audio)
                    except queue.Full:
                        logger.warning("Transcription queue full – dropping max-duration segment.")
                    with self._lock:
                        self._events["vad_status"] = "processing"
                    speech_frames = []
                    silence_count = 0
                    speech_trigger_count = 0
                    is_speaking = False
                    continue

                if not is_active_speech:
                    silence_count += 1
                    if silence_count >= silence_timeout_frames:
                        # Segment complete
                        if len(speech_frames) >= min_speech_frames:
                            segment_audio = b"".join(speech_frames)
                            try:
                                self._transcription_queue.put_nowait(segment_audio)
                            except queue.Full:
                                logger.warning(
                                    "Transcription queue full – dropping %.1f s segment.",
                                    len(speech_frames) * config.AUDIO_BLOCK_DURATION_MS / 1000,
                                )
                            with self._lock:
                                self._events["vad_status"] = "processing"
                            logger.debug(
                                "Speech segment finalised: %d frames = %.1f s.",
                                len(speech_frames),
                                len(speech_frames) * config.AUDIO_BLOCK_DURATION_MS / 1000,
                            )
                        else:
                            logger.debug(
                                "Discarded short segment (%d frames < %d min).",
                                len(speech_frames), min_speech_frames,
                            )
                        # Reset
                        speech_frames = []
                        silence_count = 0
                        speech_trigger_count = 0
                        is_speaking = False
                        with self._lock:
                            self._events["speech_detected"] = False
                            if self._events["vad_status"] != "processing":
                                self._events["vad_status"] = "silence"
                else:
                    silence_count = 0

        logger.debug("VAD loop exited.")

    # ------------------------------------------------------------------
    # Transcription thread
    # ------------------------------------------------------------------

    def _transcription_loop(self) -> None:
        """Wait for speech segments, run Whisper, parse commands / dictation."""
        # Block until the model is ready (start() sets this event)
        self._model_loaded.wait(timeout=180)

        while self._running:
            try:
                segment_audio = self._transcription_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if segment_audio is None:
                break

            if self._whisper_model is None:
                continue

            # Convert int16 bytes → float32 array that Whisper expects
            audio_np = (
                np.frombuffer(segment_audio, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            try:
                segments_gen, info = self._whisper_model.transcribe(
                    audio_np,
                    language=config.WHISPER_LANGUAGE,
                    beam_size=5,
                    vad_filter=False,  # we already did our own VAD
                )

                # Materialise generator (segments are lazy)
                text_parts: List[str] = []
                total_log_prob: float = 0.0
                total_no_speech_prob: float = 0.0
                seg_count: int = 0

                for seg in segments_gen:
                    text_parts.append(seg.text)
                    logprob = getattr(seg, "avg_logprob", getattr(seg, "avg_log_prob", -0.5))
                    total_log_prob += logprob
                    total_no_speech_prob += getattr(seg, "no_speech_prob", 0.0)
                    seg_count += 1

                full_text = " ".join(text_parts).strip()

                # ── no_speech_prob gate ───────────────────────────────
                # Whisper's own speech-presence detector.  When the average
                # no_speech_prob across segments is high, Whisper itself
                # determined the audio contained no intelligible speech —
                # discard immediately without routing.
                if seg_count > 0:
                    avg_no_speech_prob = total_no_speech_prob / seg_count
                    if avg_no_speech_prob >= config.WHISPER_NO_SPEECH_THRESHOLD:
                        logger.debug(
                            "Discarding segment: avg_no_speech_prob=%.2f >= %.2f",
                            avg_no_speech_prob, config.WHISPER_NO_SPEECH_THRESHOLD,
                        )
                        with self._lock:
                            self._events["speech_confidence"] = 0.0
                            self._events["vad_status"] = "silence"
                        continue

                # ── Confidence ────────────────────────────────────────
                # avg_log_prob is typically in [-2.0, 0.0].
                # exp() maps it to (0.0, 1.0].  Clamped for safety.
                if seg_count > 0:
                    avg_log_prob = total_log_prob / seg_count
                    confidence = float(
                        min(max(math.exp(avg_log_prob), 0.0), 1.0)
                    )
                else:
                    confidence = 0.0

                # ── Empty / hallucination filter ──────────────────────
                # Filter out empty transcriptions and known Whisper hallucinations
                # that occur when it processes noise or very short audio.
                if not full_text or _is_whisper_hallucination(full_text):
                    logger.debug(
                        "Discarding empty or hallucinated transcription: %r",
                        full_text,
                    )
                    with self._lock:
                        self._events["speech_confidence"] = 0.0
                        self._events["vad_status"] = "silence"
                    continue

                logger.info(
                    "Transcribed: '%s'  (confidence=%.2f)", full_text, confidence,
                )
                self._process_transcription(full_text, confidence)

            except Exception as exc:
                logger.warning("Transcription error: %s", exc)

            # Reset VAD status after processing
            with self._lock:
                if self._events["vad_status"] == "processing":
                    self._events["vad_status"] = "silence"

        logger.debug("Transcription loop exited.")

    # ------------------------------------------------------------------
    # Post-transcription routing
    # ------------------------------------------------------------------

    def _process_transcription(self, text: str, confidence: float) -> None:
        """Route recognised text to command parser or dictation output."""
        now = time.monotonic()

        # Bump the event counter & write common fields
        with self._lock:
            self._event_id += 1
            self._events["timestamp"] = now
            self._events["transcription_text"] = text
            self._events["speech_confidence"] = confidence
            self._events["event_id"] = self._event_id

        # ── Always check dictation-control commands first ─────────────
        cmd_id, cmd_phrase, score = parse_command(text)

        if cmd_id == "start_dictation":
            with self._lock:
                self._dictation_mode = True
                self._events["dictation_mode"] = True
                self._events["command"] = cmd_id
                self._events["command_phrase"] = cmd_phrase
            logger.info("Dictation mode ACTIVATED.")
            return

        if cmd_id == "stop_dictation":
            with self._lock:
                self._dictation_mode = False
                self._events["dictation_mode"] = False
                self._events["command"] = cmd_id
                self._events["command_phrase"] = cmd_phrase
            logger.info("Dictation mode DEACTIVATED.")
            return

        # ── Dictation mode: emit text, skip command parsing ───────────
        if self._dictation_mode:
            with self._lock:
                self._events["dictation_text"] = text
            logger.info("Dictation text: '%s'", text)
            return

        # ── Normal mode: emit matched command (if any) ────────────────
        if cmd_id is not None:
            with self._lock:
                self._events["command"] = cmd_id
                self._events["command_phrase"] = cmd_phrase
            logger.info(
                "Command recognised: %s  (phrase='%s', score=%.2f)",
                cmd_id, cmd_phrase, score,
            )
        else:
            logger.debug(
                "No command matched for: '%s'  (best_score=%.2f)", text, score,
            )
