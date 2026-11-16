"""
tests/test_face_tracker.py - Automated unit tests for FaceTracker (Stage 2).

All tests use synthetic/mocked inputs so no real webcam or model is needed.
Hardware-dependent tests (real camera, real cursor) are excluded and must
be verified manually via demo_face.py.

Run with:
  .\.venv\Scripts\Activate.ps1
  python -m pytest tests/test_face_tracker.py -v
"""

import sys
import pathlib
import time
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from face.face_tracker import FaceTracker, _OneEuroFilter


# ---------------------------------------------------------------------------
# Helpers to build fake MediaPipe result objects
# ---------------------------------------------------------------------------

def _make_landmark(x: float, y: float, z: float = 0.0) -> MagicMock:
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _make_blendshape(name: str, score: float) -> MagicMock:
    bs = MagicMock()
    bs.category_name = name
    bs.score = score
    return bs


def _make_result(
    nose_x: float = 0.5,
    nose_y: float = 0.5,
    blink_left: float = 0.0,
    blink_right: float = 0.0,
    jaw_open: float = 0.0,
) -> MagicMock:
    """Build a minimal fake FaceLandmarkerResult."""
    # 478 landmarks; we only care about index 4 (nose tip)
    landmarks = [_make_landmark(0.5, 0.5)] * 5
    landmarks[4] = _make_landmark(nose_x, nose_y)

    blendshapes = [
        _make_blendshape("eyeBlinkLeft",  blink_left),
        _make_blendshape("eyeBlinkRight", blink_right),
        _make_blendshape("jawOpen",       jaw_open),
    ]

    result = MagicMock()
    result.face_landmarks = [landmarks]
    result.face_blendshapes = [blendshapes]
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOneEuroFilter(unittest.TestCase):
    """Tests for the 1-Euro low-pass filter."""

    def test_passthrough_on_first_call(self):
        f = _OneEuroFilter()
        out = f(42.0)
        self.assertEqual(out, 42.0)

    def test_smooths_step_input(self):
        f = _OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.0)
        # Feed many frames at 0.0 then jump to 1.0 - output should lag
        for _ in range(20):
            out = f(0.0)
        out = f(1.0)  # sudden jump
        self.assertLess(out, 1.0, "Filter should smooth out step jump")

    def test_reset(self):
        f = _OneEuroFilter()
        f(10.0)
        f.reset()
        out = f(5.0)
        self.assertEqual(out, 5.0, "After reset first call should pass through")


class TestFaceTrackerInit(unittest.TestCase):
    """Tests for FaceTracker initialisation and event structure."""

    def _make_tracker(self) -> FaceTracker:
        """Create a FaceTracker without starting it."""
        return FaceTracker()

    def test_instantiation(self):
        t = self._make_tracker()
        self.assertIsNotNone(t)

    def test_initial_events_structure(self):
        t = self._make_tracker()
        events = t.get_latest_events()
        required_keys = [
            "timestamp", "face_detected", "face_confidence",
            "cursor_delta_x", "cursor_delta_y",
            "left_click", "right_click", "head_position", "debug",
        ]
        for key in required_keys:
            self.assertIn(key, events, f"Missing key: {key}")

    def test_initial_event_types(self):
        t = self._make_tracker()
        events = t.get_latest_events()
        self.assertIsInstance(events["timestamp"], float)
        self.assertIsInstance(events["face_detected"], bool)
        self.assertIsInstance(events["face_confidence"], float)
        self.assertIsInstance(events["cursor_delta_x"], float)
        self.assertIsInstance(events["cursor_delta_y"], float)
        self.assertIsInstance(events["left_click"], bool)
        self.assertIsInstance(events["right_click"], bool)
        self.assertIsInstance(events["head_position"], dict)

    def test_initial_no_face_confidence_zero(self):
        t = self._make_tracker()
        events = t.get_latest_events()
        self.assertFalse(events["face_detected"])
        self.assertEqual(events["face_confidence"], 0.0,
                         "No face should give confidence = 0.0")

    def test_initial_no_click(self):
        t = self._make_tracker()
        events = t.get_latest_events()
        self.assertFalse(events["left_click"])
        self.assertFalse(events["right_click"])


class TestConfidenceRange(unittest.TestCase):
    """Confidence must always be in [0.0, 1.0]."""

    def setUp(self):
        self.tracker = FaceTracker()
        # Seed nose history so stability score can be computed
        self.tracker._neutral_x = 0.5
        self.tracker._neutral_y = 0.5
        for i in range(10):
            self.tracker._nose_history_x.append(0.5 + i * 0.001)
            self.tracker._nose_history_y.append(0.5 + i * 0.001)

    def test_confidence_in_range_good_lighting(self):
        c = self.tracker._compute_confidence(brightness=120.0, now=0.0)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_confidence_in_range_dark_room(self):
        c = self.tracker._compute_confidence(brightness=10.0, now=0.0)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_confidence_in_range_overexposed(self):
        c = self.tracker._compute_confidence(brightness=250.0, now=0.0)
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_no_face_confidence_zero(self):
        """_empty_events() must return confidence 0.0."""
        events = FaceTracker._empty_events()
        self.assertEqual(events["face_confidence"], 0.0)

    def test_good_lighting_higher_than_dark(self):
        c_good = self.tracker._compute_confidence(brightness=120.0, now=0.0)
        c_dark = self.tracker._compute_confidence(brightness=5.0, now=0.0)
        self.assertGreater(c_good, c_dark,
                           "Good lighting should yield higher confidence than darkness")


class TestDeadZone(unittest.TestCase):
    """Verify dead zone suppresses small movements."""

    def setUp(self):
        self.tracker = FaceTracker()
        self.tracker._neutral_x = 0.5
        self.tracker._neutral_y = 0.5
        # Fill history to seed moving average
        for _ in range(10):
            self.tracker._nose_history_x.append(0.5)
            self.tracker._nose_history_y.append(0.5)

    def _get_delta(self, nose_x: float, nose_y: float) -> tuple:
        result = _make_result(nose_x=nose_x, nose_y=nose_y)
        events = self.tracker._build_events(result, brightness=120.0, now=time.monotonic())
        return events["cursor_delta_x"], events["cursor_delta_y"]

    def test_zero_displacement_gives_zero_delta(self):
        dx, dy = self._get_delta(0.5, 0.5)
        self.assertAlmostEqual(dx, 0.0, places=2)
        self.assertAlmostEqual(dy, 0.0, places=2)

    def test_sub_deadzone_gives_zero_delta(self):
        # Move less than dead zone (0.012 of frame)
        tiny = config.CURSOR_DEAD_ZONE * 0.5
        dx, dy = self._get_delta(0.5 + tiny, 0.5 + tiny)
        self.assertAlmostEqual(dx, 0.0, places=1,
                               msg="Sub-deadzone movement should produce ~0 delta")
        self.assertAlmostEqual(dy, 0.0, places=1)

    def test_beyond_deadzone_gives_nonzero_delta(self):
        # Move well beyond dead zone
        big = config.CURSOR_DEAD_ZONE * 5.0
        dx, dy = self._get_delta(0.5 + big, 0.5 + big)
        self.assertNotEqual(dx, 0.0, "Movement beyond dead zone should produce non-zero delta")
        self.assertNotEqual(dy, 0.0)

    def test_delta_clamped_to_max(self):
        # Extreme head movement should be clamped
        dx, dy = self._get_delta(0.0, 0.0)  # far left + up
        self.assertLessEqual(abs(dx), config.CURSOR_MAX_DELTA + 1)
        self.assertLessEqual(abs(dy), config.CURSOR_MAX_DELTA + 1)


class TestCursorDirection(unittest.TestCase):
    """Verify cursor direction conventions."""

    def setUp(self):
        self.tracker = FaceTracker()
        self.tracker._neutral_x = 0.5
        self.tracker._neutral_y = 0.5
        for _ in range(10):
            self.tracker._nose_history_x.append(0.5)
            self.tracker._nose_history_y.append(0.5)

    def _delta(self, nx, ny):
        result = _make_result(nose_x=nx, nose_y=ny)
        events = self.tracker._build_events(result, brightness=120.0, now=time.monotonic())
        return events["cursor_delta_x"], events["cursor_delta_y"]

    def test_head_right_moves_cursor_right(self):
        # In front-facing webcam, head right moves nose towards left of frame (nose_x decreases)
        dx, _ = self._delta(0.5 - 0.1, 0.5)
        self.assertGreater(dx, 0.0,
            "Head moving right (nose_x in frame decreases) should give positive dx (cursor right)")

    def test_head_left_moves_cursor_left(self):
        # In front-facing webcam, head left moves nose towards right of frame (nose_x increases)
        dx, _ = self._delta(0.5 + 0.1, 0.5)
        self.assertLess(dx, 0.0,
            "Head moving left (nose_x in frame increases) should give negative dx (cursor left)")

    def test_head_down_moves_cursor_down(self):
        _, dy = self._delta(0.5, 0.5 + 0.1)
        self.assertGreater(dy, 0.0, "Head moving down (nose_y increases) should give positive dy")

    def test_head_up_moves_cursor_up(self):
        _, dy = self._delta(0.5, 0.5 - 0.1)
        self.assertLess(dy, 0.0, "Head moving up (nose_y decreases) should give negative dy")

    def test_balanced_medium_speed_all_directions(self):
        """Verify right, left, up, and down all produce medium, balanced speeds."""
        def _get_direction_delta(nx, ny):
            tracker = FaceTracker()
            tracker._neutral_x = 0.5
            tracker._neutral_y = 0.5
            res = _make_result(nose_x=nx, nose_y=ny)
            t = time.monotonic()
            for i in range(10):
                ev = tracker._build_events(res, brightness=120.0, now=t + i * 0.033)
            return ev["cursor_delta_x"], ev["cursor_delta_y"]

        # 0.06 horizontal displacement (moderate head turn)
        dx_right, _ = _get_direction_delta(0.44, 0.5)
        dx_left, _ = _get_direction_delta(0.56, 0.5)
        # 0.06 upward displacement (moderate head tilt up)
        _, dy_up = _get_direction_delta(0.5, 0.44)
        # 0.038 downward displacement (moderate downward head tilt, naturally ~1/1.5 of up)
        _, dy_down = _get_direction_delta(0.5, 0.538)

        # All movements should be in comfortable medium range (8 - 15 px/frame)
        self.assertTrue(8.0 <= dx_right <= 15.0, f"Right delta {dx_right} not in medium range [8, 15]")
        self.assertTrue(8.0 <= abs(dx_left) <= 15.0, f"Left delta {dx_left} not in medium range [8, 15]")
        self.assertTrue(8.0 <= abs(dy_up) <= 15.0, f"Up delta {dy_up} not in medium range [8, 15]")
        self.assertTrue(8.0 <= dy_down <= 15.0, f"Down delta {dy_down} not in medium range [8, 15]")

        # Down should be balanced with up (within 20% of each other)
        speed_ratio = dy_down / abs(dy_up)
        self.assertTrue(0.8 <= speed_ratio <= 1.2, f"Down/Up ratio {speed_ratio} not balanced")


class TestBlinkGesture(unittest.TestCase):
    """Tests for bilateral long-blink left-click gesture."""

    def setUp(self):
        self.tracker = FaceTracker()
        self.tracker._neutral_x = 0.5
        self.tracker._neutral_y = 0.5
        for _ in range(10):
            self.tracker._nose_history_x.append(0.5)
            self.tracker._nose_history_y.append(0.5)

    def _send_blink_frames(self, blink_score: float, duration_s: float,
                           start_time: float = 10.0) -> list:
        """Simulate blink frames over duration. Use start_time >> 0 so cooldown check passes."""
        results = []
        steps = max(int(duration_s * 30), 2)
        for i in range(steps):
            t = start_time + i * (1.0 / 30.0)
            result = _make_result(blink_left=blink_score, blink_right=blink_score)
            events = self.tracker._build_events(result, brightness=120.0, now=t)
            results.append(events)
        return results

    def test_short_blink_no_click(self):
        # Blink for less than BLINK_HOLD_SECONDS - should NOT fire
        short = config.BLINK_HOLD_SECONDS * 0.5
        events_list = self._send_blink_frames(0.8, short, start_time=10.0)
        clicks = [e["left_click"] for e in events_list]
        self.assertFalse(any(clicks),
                         "Short blink must not trigger a left-click")

    def test_long_blink_fires_click(self):
        # Blink for longer than BLINK_HOLD_SECONDS - should fire exactly once
        long = config.BLINK_HOLD_SECONDS * 1.5
        events_list = self._send_blink_frames(0.8, long, start_time=10.0)
        clicks = [e["left_click"] for e in events_list]
        self.assertTrue(any(clicks),
                        "Long blink must trigger exactly one left-click")

    def test_long_blink_fires_only_once(self):
        long = config.BLINK_HOLD_SECONDS * 2.0
        events_list = self._send_blink_frames(0.8, long, start_time=10.0)
        clicks = sum(1 for e in events_list if e["left_click"])
        self.assertEqual(clicks, 1, "Long blink must fire click exactly ONCE")

    def test_sub_threshold_blink_no_click(self):
        # Blink score below threshold - never triggers
        long = config.BLINK_HOLD_SECONDS * 2.0
        events_list = self._send_blink_frames(
            config.BLINK_SCORE_THRESHOLD * 0.5, long, start_time=10.0
        )
        clicks = [e["left_click"] for e in events_list]
        self.assertFalse(any(clicks), "Below-threshold blink must not trigger click")

    def test_blink_requires_reset_before_second_click(self):
        """After one click, eyes must open before second click can fire."""
        long = config.BLINK_HOLD_SECONDS * 1.5
        # First blink cycle - fires a click
        self._send_blink_frames(0.8, long, start_time=10.0)

        # Verify blink_reset_ok is now False (waiting for eyes to reopen)
        self.assertFalse(self.tracker._blink_reset_ok,
                         "After click, blink_reset_ok must be False")

        # Without opening eyes, blink again immediately - should NOT fire
        t_offset = 10.0 + long + config.CLICK_COOLDOWN_SECONDS + 0.1
        events_list = self._send_blink_frames(
            0.8, config.BLINK_HOLD_SECONDS * 1.5, start_time=t_offset
        )
        clicks = [e["left_click"] for e in events_list]
        self.assertFalse(any(clicks),
                         "Second blink must not fire until eyes have reopened")

    def test_missing_face_resets_blink(self):
        """No face frames must reset gesture state without firing click."""
        # Start blink but don't hold long enough
        short = config.BLINK_HOLD_SECONDS * 0.4
        self._send_blink_frames(0.8, short, start_time=0.0)

        # Feed no-face frames
        for i in range(5):
            events = self.tracker._build_events(None, brightness=100.0,
                                                now=short + i * 0.033)
        self.assertIsNone(self.tracker._blink_start_time,
                          "No-face frames should reset blink state")


class TestJawGesture(unittest.TestCase):
    """Tests for jaw-open right-click gesture."""

    def setUp(self):
        self.tracker = FaceTracker()
        self.tracker._neutral_x = 0.5
        self.tracker._neutral_y = 0.5
        for _ in range(10):
            self.tracker._nose_history_x.append(0.5)
            self.tracker._nose_history_y.append(0.5)

    def _send_jaw_frames(self, jaw_score: float, duration_s: float,
                         start_time: float = 10.0) -> list:
        """Simulate jaw frames. Use start_time >> 0 so cooldown check passes."""
        results = []
        steps = max(int(duration_s * 30), 2)
        for i in range(steps):
            t = start_time + i / 30.0
            result = _make_result(jaw_open=jaw_score)
            events = self.tracker._build_events(result, brightness=120.0, now=t)
            results.append(events)
        return results

    def test_short_jaw_no_click(self):
        short = config.JAW_HOLD_SECONDS * 0.5
        events_list = self._send_jaw_frames(0.8, short, start_time=10.0)
        clicks = [e["right_click"] for e in events_list]
        self.assertFalse(any(clicks), "Short jaw-open must not trigger right-click")

    def test_long_jaw_fires_click(self):
        long = config.JAW_HOLD_SECONDS * 1.5
        events_list = self._send_jaw_frames(0.8, long, start_time=10.0)
        clicks = [e["right_click"] for e in events_list]
        self.assertTrue(any(clicks), "Long jaw-open must trigger a right-click")

    def test_long_jaw_fires_only_once(self):
        long = config.JAW_HOLD_SECONDS * 2.5
        events_list = self._send_jaw_frames(0.8, long, start_time=10.0)
        clicks = sum(1 for e in events_list if e["right_click"])
        self.assertEqual(clicks, 1, "Jaw-open must fire click exactly ONCE")

    def test_jaw_requires_reset_before_second_click(self):
        """After click, jaw must close before second click."""
        long = config.JAW_HOLD_SECONDS * 1.5
        self._send_jaw_frames(0.8, long, start_time=10.0)

        # Verify jaw_reset_ok is False (waiting for jaw to close)
        self.assertFalse(self.tracker._jaw_reset_ok,
                         "After jaw-click, jaw_reset_ok must be False")

        # Without closing jaw, open again - should NOT fire (reset_ok=False)
        t_offset = 10.0 + long + config.CLICK_COOLDOWN_SECONDS + 0.1
        events_list = self._send_jaw_frames(0.8, long, start_time=t_offset)
        clicks = [e["right_click"] for e in events_list]
        self.assertFalse(any(clicks),
                         "Second jaw-click must not fire until jaw closed below reset threshold")


class TestClickCooldown(unittest.TestCase):
    """Click cooldown prevents rapid repeated clicks."""

    def setUp(self):
        self.tracker = FaceTracker()
        self.tracker._neutral_x = 0.5
        self.tracker._neutral_y = 0.5
        for _ in range(10):
            self.tracker._nose_history_x.append(0.5)
            self.tracker._nose_history_y.append(0.5)

    def test_cooldown_blocks_second_click(self):
        """After a click, a second click within cooldown period must be blocked."""
        # Simulate first click by setting last_click_time to now
        now = time.monotonic()
        self.tracker._last_click_time = now

        # Immediately try to fire blink click - cooldown not elapsed
        fired = self.tracker._update_blink_gesture(0.9, 0.9, now + 0.01)
        self.assertFalse(fired, "Click within cooldown window must be blocked")

    def test_cooldown_allows_click_after_expiry(self):
        """After cooldown expires, click should be allowed again."""
        now = time.monotonic()
        # Set last click time far in the past
        self.tracker._last_click_time = now - config.CLICK_COOLDOWN_SECONDS - 1.0
        self.tracker._blink_start_time = now - config.BLINK_HOLD_SECONDS - 0.1
        self.tracker._blink_fired = False
        self.tracker._blink_reset_ok = True

        fired = self.tracker._update_blink_gesture(0.9, 0.9, now)
        self.assertTrue(fired, "Click should be allowed after cooldown expires")


class TestThreadSafety(unittest.TestCase):
    """get_latest_events() must be safe to call from a different thread."""

    def test_concurrent_reads_do_not_crash(self):
        tracker = FaceTracker()
        results = []
        errors = []

        def reader():
            try:
                for _ in range(100):
                    events = tracker.get_latest_events()
                    results.append(events["face_confidence"])
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(100):
                    with tracker._lock:
                        tracker._events["face_confidence"] = float(i % 2)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)
        t1.start(); t2.start()
        t1.join(); t2.join()

        self.assertEqual(errors, [], f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 100)

    def test_click_flags_consumed_on_read(self):
        """left_click and right_click flags reset to False after first read."""
        tracker = FaceTracker()
        with tracker._lock:
            tracker._events["left_click"] = True
            tracker._events["right_click"] = True

        events1 = tracker.get_latest_events()
        events2 = tracker.get_latest_events()

        self.assertTrue(events1["left_click"],
                        "First read should see left_click=True")
        self.assertFalse(events2["left_click"],
                         "Second read should see left_click=False (consumed)")
        self.assertTrue(events1["right_click"])
        self.assertFalse(events2["right_click"])


class TestCleanShutdown(unittest.TestCase):
    """Verify stop() is safe to call even without start()."""

    def test_stop_without_start(self):
        tracker = FaceTracker()
        try:
            tracker.stop()  # Should not raise
        except Exception as e:
            self.fail(f"stop() raised unexpected exception: {e}")

    def test_double_stop_safe(self):
        tracker = FaceTracker()
        tracker.stop()
        tracker.stop()  # Second stop should be harmless


class TestGetLatestFrame(unittest.TestCase):
    """Verify get_latest_frame thread-safe access."""

    def test_initially_none(self):
        tracker = FaceTracker()
        self.assertIsNone(tracker.get_latest_frame())

    def test_returns_copy_of_frame(self):
        import numpy as np
        tracker = FaceTracker()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_frame[10, 10] = [255, 0, 0]
        with tracker._lock:
            tracker._latest_frame = dummy_frame

        retrieved = tracker.get_latest_frame()
        self.assertIsNotNone(retrieved)
        self.assertTrue(np.array_equal(retrieved, dummy_frame))
        # Ensure it is a copy, not the exact same memory reference
        self.assertIsNot(retrieved, dummy_frame)


if __name__ == "__main__":
    unittest.main(verbosity=2)
