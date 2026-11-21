"""
calibration/calibration.py - Interactive calibration for the Assistive Control System.

Responsibilities
----------------
1. First-run detection: if calibration/user_settings.json is absent, run the
   ~60-second interactive calibration procedure; otherwise load saved settings.
2. Calibration procedure (5 phases, ~60 s total):
     Phase 1 (15 s) - Neutral head position
     Phase 2 (15 s) - Head movement range
     Phase 3 (10 s) - Deliberate blinks
     Phase 4 (10 s) - Jaw-open gestures
     Phase 5 (10 s) - Quiet mic noise floor
3. Persist settings to calibration/user_settings.json.
4. Apply settings to live config module attributes.
5. Re-calibration via:
     - --calibrate CLI flag (caller's responsibility to invoke CalibrationManager.run())
     - "recalibrate" voice command (handled upstream, same as above)
     - Ctrl+Alt+C hotkey (registered here via pynput)

Public API
----------
  mgr = CalibrationManager()
  if mgr.needs_calibration():
      settings = mgr.run(face_tracker, speech_engine)
  else:
      settings = mgr.load()
  mgr.apply(settings)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger("assistive.calibration")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CALIBRATION_DIR: Path = config.PROJECT_ROOT / "calibration"
_SETTINGS_FILE: Path = _CALIBRATION_DIR / "user_settings.json"

# ---------------------------------------------------------------------------
# Default settings (used when a phase is skipped due to unavailable hardware)
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "neutral_x": 0.5,
    "neutral_y": 0.5,
    "head_range_x": 0.15,
    "head_range_y": 0.10,
    "head_stability_std": 0.005,
    "blink_threshold": 0.55,
    "jaw_threshold": 0.55,
    "mic_noise_floor": 0.015,
    "calibration_date": "",
}


# ---------------------------------------------------------------------------
# CalibrationManager
# ---------------------------------------------------------------------------
class CalibrationManager:
    """
    Manages first-run detection, interactive calibration, settings I/O,
    and applying calibrated values to the live config module.
    """

    def __init__(self) -> None:
        # Hotkey listener (started once, lives for the process lifetime)
        self._hotkey_listener: Optional[Any] = None
        self._recalibrate_requested: threading.Event = threading.Event()
        self._register_hotkey()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def needs_calibration(self) -> bool:
        """Return True when no saved settings file exists."""
        return not _SETTINGS_FILE.exists()

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load settings from calibration/user_settings.json.

        Returns the settings dict, or None if the file is absent or corrupt.
        """
        if not _SETTINGS_FILE.exists():
            logger.info("No saved calibration found at %s.", _SETTINGS_FILE)
            return None
        try:
            with _SETTINGS_FILE.open("r", encoding="utf-8") as fh:
                settings = json.load(fh)
            logger.info(
                "Loaded calibration settings from %s (dated %s).",
                _SETTINGS_FILE,
                settings.get("calibration_date", "unknown"),
            )
            # Back-fill any keys that may be missing from an older file
            for key, default in _DEFAULTS.items():
                settings.setdefault(key, default)
            return settings
        except Exception as exc:
            logger.warning("Failed to read calibration file: %s", exc)
            return None

    def save(self, settings: Dict[str, Any]) -> None:
        """Persist *settings* to calibration/user_settings.json."""
        _CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with _SETTINGS_FILE.open("w", encoding="utf-8") as fh:
                json.dump(settings, fh, indent=2)
            logger.info("Calibration settings saved to %s.", _SETTINGS_FILE)
        except Exception as exc:
            logger.error("Failed to save calibration settings: %s", exc)

    def apply(self, settings: Dict[str, Any]) -> None:
        """
        Write calibrated values back into the live config module so that
        all subsystems pick them up without a restart.
        """
        config.BLINK_SCORE_THRESHOLD = float(settings["blink_threshold"])
        config.JAW_OPEN_THRESHOLD = float(settings["jaw_threshold"])
        config.SPEECH_RMS_THRESHOLD = max(
            float(settings["mic_noise_floor"]) * 1.5, 0.01
        )
        config.CURSOR_DEAD_ZONE = float(settings["head_stability_std"]) * 2

        logger.info(
            "Applied calibration: blink_thr=%.3f jaw_thr=%.3f "
            "speech_rms=%.4f cursor_dead=%.4f",
            config.BLINK_SCORE_THRESHOLD,
            config.JAW_OPEN_THRESHOLD,
            config.SPEECH_RMS_THRESHOLD,
            config.CURSOR_DEAD_ZONE,
        )

    def run(
        self,
        face_tracker: Optional[Any] = None,
        speech_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run the interactive ~60-second calibration procedure.

        Parameters
        ----------
        face_tracker  : FaceTracker instance (or None → face phases skipped)
        speech_engine : SpeechEngine instance (or None → mic phase skipped)

        Returns
        -------
        settings dict (also saved to disk automatically)
        """
        print("\n" + "=" * 60)
        print("  ASSISTIVE SYSTEM CALIBRATION")
        print("  Total time: ~60 seconds")
        print("=" * 60)

        settings: Dict[str, Any] = dict(_DEFAULTS)

        face_ok = self._check_face_available(face_tracker)
        mic_ok = self._check_mic_available(speech_engine)

        if face_ok:
            settings.update(self._phase1_neutral(face_tracker))
            settings.update(self._phase2_head_range(face_tracker))
            settings.update(self._phase3_blink(face_tracker))
            settings.update(self._phase4_jaw(face_tracker))
        else:
            print("\n[SKIP] Camera unavailable – using default face parameters.")
            logger.warning("FaceTracker unavailable; skipping face calibration phases.")

        if mic_ok:
            settings.update(self._phase5_mic_noise(speech_engine))
        else:
            print("\n[SKIP] Microphone unavailable – using default mic noise floor.")
            logger.warning("SpeechEngine unavailable; skipping mic calibration phase.")

        settings["calibration_date"] = datetime.now(timezone.utc).isoformat()

        self.save(settings)
        print("\n[DONE] Calibration complete. Settings saved.\n")
        return settings

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_face_available(face_tracker: Optional[Any]) -> bool:
        """Return True if face_tracker is a live, running FaceTracker."""
        if face_tracker is None:
            return False
        try:
            events = face_tracker.get_latest_events(consume=False)
            # Accept it if the tracker returned any events dict at all
            return isinstance(events, dict)
        except Exception:
            return False

    @staticmethod
    def _check_mic_available(speech_engine: Optional[Any]) -> bool:
        """Return True if speech_engine is a live, running SpeechEngine."""
        if speech_engine is None:
            return False
        try:
            events = speech_engine.get_latest_events(consume=False)
            return isinstance(events, dict)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Calibration phases
    # ------------------------------------------------------------------

    @staticmethod
    def _countdown(duration: int, label: str) -> None:
        """Print a live countdown bar for *duration* seconds."""
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            remaining = duration - elapsed
            if remaining <= 0:
                break
            bar_len = 30
            filled = int(bar_len * elapsed / duration)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"\r  {label} [{bar}] {int(remaining):>2}s remaining",
                end="",
                flush=True,
            )
            time.sleep(0.1)
        print(f"\r  {label} [{'█' * 30}]  0s remaining  ")

    def _collect_face_events(
        self, face_tracker: Any, duration: float, interval: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Collect face events over *duration* seconds, polling at *interval*."""
        samples: List[Dict[str, Any]] = []
        end = time.monotonic() + duration
        while time.monotonic() < end:
            try:
                ev = face_tracker.get_latest_events(consume=False)
                if ev.get("face_detected") and ev.get("head_position"):
                    samples.append(ev)
            except Exception:
                pass
            time.sleep(interval)
        return samples

    def _collect_mic_rms(
        self, speech_engine: Any, duration: float, interval: float = 0.05
    ) -> List[float]:
        """Collect mic noise_level readings over *duration* seconds."""
        readings: List[float] = []
        end = time.monotonic() + duration
        while time.monotonic() < end:
            try:
                ev = speech_engine.get_latest_events(consume=False)
                nl = ev.get("noise_level", 0.0)
                if nl > 0.0:
                    readings.append(float(nl))
            except Exception:
                pass
            time.sleep(interval)
        return readings

    # --- Phase 1 ----------------------------------------------------------
    def _phase1_neutral(self, face_tracker: Any) -> Dict[str, Any]:
        duration = 15
        print("\n" + "-" * 60)
        print("PHASE 1 / 5 — Neutral head position (15 s)")
        print("  Look straight at the camera, stay still.")
        input("  Press ENTER when ready...")

        # Collect samples in background while countdown runs
        samples: List[Dict[str, Any]] = []
        stop_flag = threading.Event()

        def _collect() -> None:
            while not stop_flag.is_set():
                try:
                    ev = face_tracker.get_latest_events(consume=False)
                    if ev.get("face_detected") and ev.get("head_position"):
                        samples.append(ev)
                except Exception:
                    pass
                time.sleep(0.05)

        t = threading.Thread(target=_collect, daemon=True)
        t.start()
        self._countdown(duration, "Phase 1")
        stop_flag.set()
        t.join(timeout=1.0)

        result = {}
        if samples:
            xs = [s["head_position"]["x"] for s in samples]
            ys = [s["head_position"]["y"] for s in samples]
            result["neutral_x"] = float(np.mean(xs))
            result["neutral_y"] = float(np.mean(ys))
            result["head_stability_std"] = float(
                (np.std(xs) + np.std(ys)) / 2.0
            )
            print(
                f"  → neutral ({result['neutral_x']:.4f}, {result['neutral_y']:.4f})"
                f"  std={result['head_stability_std']:.5f}"
            )
            # Let FaceTracker itself know about the calibrated neutral
            try:
                face_tracker.calibrate_neutral()
            except Exception:
                pass
        else:
            print("  ⚠ No face detected – using defaults for Phase 1.")
            logger.warning("Phase 1: no face samples collected.")
            result["neutral_x"] = _DEFAULTS["neutral_x"]
            result["neutral_y"] = _DEFAULTS["neutral_y"]
            result["head_stability_std"] = _DEFAULTS["head_stability_std"]

        return result

    # --- Phase 2 ----------------------------------------------------------
    def _phase2_head_range(self, face_tracker: Any) -> Dict[str, Any]:
        duration = 15
        print("\n" + "-" * 60)
        print("PHASE 2 / 5 — Head movement range (15 s)")
        print("  Move your head slowly left/right/up/down to your comfortable limit.")
        input("  Press ENTER when ready...")

        samples: List[Dict[str, Any]] = []
        stop_flag = threading.Event()

        def _collect() -> None:
            while not stop_flag.is_set():
                try:
                    ev = face_tracker.get_latest_events(consume=False)
                    if ev.get("face_detected") and ev.get("head_position"):
                        samples.append(ev)
                except Exception:
                    pass
                time.sleep(0.05)

        t = threading.Thread(target=_collect, daemon=True)
        t.start()
        self._countdown(duration, "Phase 2")
        stop_flag.set()
        t.join(timeout=1.0)

        result = {}
        if samples:
            xs = [s["head_position"]["x"] for s in samples]
            ys = [s["head_position"]["y"] for s in samples]
            result["head_range_x"] = float(np.max(xs) - np.min(xs))
            result["head_range_y"] = float(np.max(ys) - np.min(ys))
            print(
                f"  → range_x={result['head_range_x']:.4f}"
                f"  range_y={result['head_range_y']:.4f}"
            )
        else:
            print("  ⚠ No face detected – using defaults for Phase 2.")
            logger.warning("Phase 2: no face samples collected.")
            result["head_range_x"] = _DEFAULTS["head_range_x"]
            result["head_range_y"] = _DEFAULTS["head_range_y"]

        return result

    # --- Phase 3 ----------------------------------------------------------
    def _phase3_blink(self, face_tracker: Any) -> Dict[str, Any]:
        duration = 10
        print("\n" + "-" * 60)
        print("PHASE 3 / 5 — Blink calibration (10 s)")
        print("  Blink deliberately 3-5 times (hold each blink ~0.5 s).")
        input("  Press ENTER when ready...")

        peaks: List[float] = []
        stop_flag = threading.Event()

        def _collect() -> None:
            prev_blink = 0.0
            while not stop_flag.is_set():
                try:
                    ev = face_tracker.get_latest_events(consume=False)
                    if ev.get("face_detected"):
                        debug = ev.get("debug", {})
                        bl = debug.get("blink_left_score", 0.0)
                        br = debug.get("blink_right_score", 0.0)
                        score = (bl + br) / 2.0
                        # Rising edge peak detection
                        if score > prev_blink:
                            prev_blink = score
                        elif score < prev_blink * 0.5 and prev_blink > 0.3:
                            # Descending from a peak
                            peaks.append(prev_blink)
                            prev_blink = 0.0
                except Exception:
                    pass
                time.sleep(0.05)

        t = threading.Thread(target=_collect, daemon=True)
        t.start()
        self._countdown(duration, "Phase 3")
        stop_flag.set()
        t.join(timeout=1.0)

        result = {}
        if peaks:
            raw = float(np.mean(peaks))
            result["blink_threshold"] = float(np.clip(raw, 0.3, 0.8))
            print(
                f"  → {len(peaks)} blink peaks, avg={raw:.3f}"
                f"  threshold={result['blink_threshold']:.3f}"
            )
        else:
            print("  ⚠ No blink peaks detected – using defaults for Phase 3.")
            logger.warning("Phase 3: no blink peaks detected.")
            result["blink_threshold"] = _DEFAULTS["blink_threshold"]

        return result

    # --- Phase 4 ----------------------------------------------------------
    def _phase4_jaw(self, face_tracker: Any) -> Dict[str, Any]:
        duration = 10
        print("\n" + "-" * 60)
        print("PHASE 4 / 5 — Jaw-open calibration (10 s)")
        print("  Open your mouth wide 3 times.")
        input("  Press ENTER when ready...")

        peaks: List[float] = []
        stop_flag = threading.Event()

        def _collect() -> None:
            prev_jaw = 0.0
            while not stop_flag.is_set():
                try:
                    ev = face_tracker.get_latest_events(consume=False)
                    if ev.get("face_detected"):
                        debug = ev.get("debug", {})
                        score = float(debug.get("jaw_open_score", 0.0))
                        if score > prev_jaw:
                            prev_jaw = score
                        elif score < prev_jaw * 0.5 and prev_jaw > 0.3:
                            peaks.append(prev_jaw)
                            prev_jaw = 0.0
                except Exception:
                    pass
                time.sleep(0.05)

        t = threading.Thread(target=_collect, daemon=True)
        t.start()
        self._countdown(duration, "Phase 4")
        stop_flag.set()
        t.join(timeout=1.0)

        result = {}
        if peaks:
            raw = float(np.mean(peaks))
            result["jaw_threshold"] = float(np.clip(raw, 0.3, 0.8))
            print(
                f"  → {len(peaks)} jaw peaks, avg={raw:.3f}"
                f"  threshold={result['jaw_threshold']:.3f}"
            )
        else:
            print("  ⚠ No jaw peaks detected – using defaults for Phase 4.")
            logger.warning("Phase 4: no jaw peaks detected.")
            result["jaw_threshold"] = _DEFAULTS["jaw_threshold"]

        return result

    # --- Phase 5 ----------------------------------------------------------
    def _phase5_mic_noise(self, speech_engine: Any) -> Dict[str, Any]:
        duration = 10
        print("\n" + "-" * 60)
        print("PHASE 5 / 5 — Microphone noise floor (10 s)")
        print("  Stay quiet – measuring ambient background noise.")
        input("  Press ENTER when ready...")

        readings: List[float] = []
        stop_flag = threading.Event()

        def _collect() -> None:
            while not stop_flag.is_set():
                try:
                    ev = speech_engine.get_latest_events(consume=False)
                    nl = ev.get("noise_level", 0.0)
                    if nl > 0.0:
                        readings.append(float(nl))
                except Exception:
                    pass
                time.sleep(0.05)

        t = threading.Thread(target=_collect, daemon=True)
        t.start()
        self._countdown(duration, "Phase 5")
        stop_flag.set()
        t.join(timeout=1.0)

        result = {}
        if readings:
            floor = float(np.percentile(readings, 95))
            result["mic_noise_floor"] = floor
            print(f"  → mic noise floor (95th pct) = {floor:.5f}")
        else:
            print("  ⚠ No mic readings collected – using defaults for Phase 5.")
            logger.warning("Phase 5: no mic readings collected.")
            result["mic_noise_floor"] = _DEFAULTS["mic_noise_floor"]

        return result

    # ------------------------------------------------------------------
    # Hotkey registration (Ctrl+Alt+C → request re-calibration)
    # ------------------------------------------------------------------

    def _register_hotkey(self) -> None:
        """Register Ctrl+Alt+C to signal a re-calibration request."""
        try:
            from pynput import keyboard  # type: ignore

            HOTKEY_COMBO = "<ctrl>+<alt>+c"

            def _on_activate() -> None:
                logger.info("Ctrl+Alt+C pressed: re-calibration requested.")
                self._recalibrate_requested.set()

            self._hotkey_listener = keyboard.GlobalHotKeys(
                {HOTKEY_COMBO: _on_activate}
            )
            self._hotkey_listener.daemon = True
            self._hotkey_listener.start()
            logger.debug("Registered Ctrl+Alt+C hotkey for re-calibration.")
        except Exception as exc:
            logger.warning(
                "Could not register Ctrl+Alt+C hotkey (pynput unavailable?): %s", exc
            )

    @property
    def recalibrate_requested(self) -> bool:
        """True if the user pressed Ctrl+Alt+C since the last check."""
        return self._recalibrate_requested.is_set()

    def clear_recalibrate_flag(self) -> None:
        """Reset the re-calibration request flag after acting on it."""
        self._recalibrate_requested.clear()
