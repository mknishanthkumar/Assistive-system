"""
tests/demo_decision_engine.py - Dry-run simulation for the Stage 5 Decision Engine.

Feeds scripted sensor scenarios into DecisionEngine and prints the full decision
log so you can verify mode switches, reasons, and hysteresis behaviour WITHOUT
needing a camera or microphone.

Scenarios tested
----------------
1.  Noisy room         → mode=FACE  (noise above threshold)
2.  Dark room          → mode=SPEECH (poor lighting, face lost)
3.  Text field focused → mode=SPEECH (dictation context)
4.  Face lost (bad conf) → mode=SPEECH (navigation by voice)
5.  Both reliable      → mode=BOTH  (normal use)
6.  Manual override → SPEECH  (voice command: "switch to voice")
7.  Manual override → FACE    (gesture command: "switch to face")
8.  Return to auto     → obeys rules again
9.  Both unreliable    → mode=PAUSED (warning issued)
10. Hysteresis demo    → shows candidate registered vs committed

Run with:
    .venv\\Scripts\\python.exe tests\\demo_decision_engine.py
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from context.decision_engine import DecisionEngine, Mode

# ---------------------------------------------------------------------------
# Fast hysteresis for the demo (so we don't wait real seconds)
# ---------------------------------------------------------------------------
config.DECISION_HOLD_DURATION_S       = 0.0   # commit on 2nd call
config.DECISION_MIN_SWITCH_INTERVAL_S = 0.0   # no cooldown between switches

# ---------------------------------------------------------------------------
# Console logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)

WIDTH = 68


def banner(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def section(label: str) -> None:
    print()
    print(f"  {'─' * 60}")
    print(f"  {label}")
    print(f"  {'─' * 60}")


def show(label: str, result) -> None:
    override_tag = " [OVERRIDE]" if result.is_override else ""
    print(f"  [{label:30s}]  mode={result.mode.value:6s}{override_tag}")
    print(f"    reason : {result.reason}")
    print()


def eval2(engine, inputs: dict):
    """Call evaluate twice so hysteresis hold (0 s) is satisfied."""
    engine.evaluate(inputs)
    return engine.evaluate(inputs)


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

def quiet_good():
    return dict(noise_level=0.0, face_confidence=0.9, brightness=128.0,
                text_field_focused=False, speech_confidence=0.9)

def noisy():
    return dict(noise_level=0.8, face_confidence=0.9, brightness=128.0,
                text_field_focused=False, speech_confidence=0.9)

def dark():
    return dict(noise_level=0.0, face_confidence=0.9, brightness=10.0,
                text_field_focused=False, speech_confidence=0.9)

def text_field():
    return dict(noise_level=0.0, face_confidence=0.9, brightness=128.0,
                text_field_focused=True, speech_confidence=0.9)

def face_lost():
    return dict(noise_level=0.0, face_confidence=0.1, brightness=128.0,
                text_field_focused=False, speech_confidence=0.9)

def both_unreliable():
    return dict(noise_level=0.8, face_confidence=0.0, brightness=5.0,
                text_field_focused=False, speech_confidence=0.0)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    banner("Stage 5 Decision Engine — Dry-Run Simulation")
    print()
    print(f"  Noise threshold    : {config.DECISION_NOISE_THRESHOLD}")
    print(f"  Face conf threshold: {config.DECISION_FACE_CONF_THRESHOLD}")
    print(f"  Brightness thresh  : {config.DECISION_BRIGHTNESS_THRESHOLD}")
    print(f"  Speech conf thresh : {config.DECISION_SPEECH_CONF_THRESHOLD}")
    print(f"  Hold duration      : {config.DECISION_HOLD_DURATION_S} s")
    print(f"  Switch interval    : {config.DECISION_MIN_SWITCH_INTERVAL_S} s")

    engine = DecisionEngine()

    # ── Scenario 1: Normal operation (BOTH) ───────────────────────────
    section("Scenario 1: Normal operation — good face + good speech + quiet")
    r = eval2(engine, quiet_good())
    show("BOTH_RELIABLE", r)
    assert r.mode == Mode.BOTH, f"Expected BOTH, got {r.mode}"
    print("  ✓ PASS: mode=BOTH")

    # ── Scenario 2: Noisy room ────────────────────────────────────────
    section("Scenario 2: Noisy room — noise=0.8 above threshold")
    r = eval2(engine, noisy())
    show("NOISY_ROOM", r)
    assert r.mode == Mode.FACE, f"Expected FACE, got {r.mode}"
    print("  ✓ PASS: mode=FACE  (speech disabled)")

    # ── Scenario 3: Dark room ─────────────────────────────────────────
    section("Scenario 3: Dark room — brightness=10, face tracking fails")
    r = eval2(engine, dark())
    show("DARK_ROOM", r)
    assert r.mode == Mode.SPEECH, f"Expected SPEECH, got {r.mode}"
    print("  ✓ PASS: mode=SPEECH  (use voice for navigation)")

    # ── Scenario 4: Text field focused + quiet ────────────────────────
    section("Scenario 4: Text field focused + quiet — dictation context")
    r = eval2(engine, text_field())
    show("TEXT_FIELD_FOCUSED", r)
    assert r.mode == Mode.SPEECH, f"Expected SPEECH, got {r.mode}"
    print("  ✓ PASS: mode=SPEECH  (dictation)")

    # ── Scenario 5: Face lost (low confidence) ────────────────────────
    section("Scenario 5: Face lost — face_confidence=0.1")
    r = eval2(engine, face_lost())
    show("FACE_LOST", r)
    assert r.mode == Mode.SPEECH, f"Expected SPEECH, got {r.mode}"
    print("  ✓ PASS: mode=SPEECH  (voice navigation)")

    # ── Scenario 6: Manual override → SPEECH ─────────────────────────
    section('Scenario 6: Manual override — "switch to voice"')
    engine.set_manual_override(Mode.SPEECH)
    # Even in a noisy room, override wins
    r = eval2(engine, noisy())
    show("OVERRIDE_SPEECH_in_NOISY_ROOM", r)
    assert r.mode == Mode.SPEECH, f"Expected SPEECH, got {r.mode}"
    assert r.is_override, "Expected is_override=True"
    print("  ✓ PASS: override=SPEECH  (even in noisy room)")

    # ── Scenario 7: Manual override → FACE ───────────────────────────
    section('Scenario 7: Manual override — "switch to face" (jaw gesture)')
    engine.set_manual_override(Mode.FACE)
    r = eval2(engine, dark())   # would normally be SPEECH
    show("OVERRIDE_FACE_in_DARK_ROOM", r)
    assert r.mode == Mode.FACE, f"Expected FACE, got {r.mode}"
    assert r.is_override, "Expected is_override=True"
    print("  ✓ PASS: override=FACE  (even in dark room)")

    # ── Scenario 8: Return to automatic mode ─────────────────────────
    section('Scenario 8: Return to auto — "switch to auto"')
    engine.clear_manual_override()
    # Dark room → should give SPEECH again (auto mode)
    r = eval2(engine, dark())
    show("AUTO_RESTORED_DARK_ROOM", r)
    assert r.mode == Mode.SPEECH, f"Expected SPEECH after clearing override, got {r.mode}"
    assert not r.is_override, "Expected is_override=False"
    print("  ✓ PASS: auto mode restored, override cleared")

    # ── Scenario 9: Both unreliable → PAUSED ─────────────────────────
    section("Scenario 9: Both unreliable — noise=0.8 + face lost + no speech")
    r = eval2(engine, both_unreliable())
    show("BOTH_UNRELIABLE", r)
    assert r.mode == Mode.PAUSED, f"Expected PAUSED, got {r.mode}"
    print("  ✓ PASS: mode=PAUSED  (warning: improve conditions)")

    # ── Scenario 10: Hysteresis demo ─────────────────────────────────
    section("Scenario 10: Hysteresis demo — hold timer")
    # Reset to real hysteresis values to show it working
    config.DECISION_HOLD_DURATION_S       = 0.1
    config.DECISION_MIN_SWITCH_INTERVAL_S = 0.0
    engine2 = DecisionEngine()

    noisy_inp = noisy()
    # First call: candidate registered, mode unchanged (still BOTH)
    r1 = engine2.evaluate(noisy_inp)
    print(f"  First  evaluate(): mode={r1.mode.value}  "
          f"(candidate registered, hold timer started)")
    assert r1.mode == Mode.BOTH, f"Expected BOTH before hold, got {r1.mode}"

    # Second call immediately: hold NOT yet elapsed → still BOTH
    r2 = engine2.evaluate(noisy_inp)
    print(f"  Second evaluate(): mode={r2.mode.value}  "
          f"(hold < 0.1 s, switch suppressed)")

    # Wait past hold duration
    time.sleep(0.12)

    # Third call: hold elapsed → switch commits
    r3 = engine2.evaluate(noisy_inp)
    print(f"  Third  evaluate(): mode={r3.mode.value}  "
          f"(hold elapsed 0.1 s → switch committed)")
    assert r3.mode == Mode.FACE, f"Expected FACE after hold expired, got {r3.mode}"
    print("  ✓ PASS: hysteresis hold timer working correctly")

    # ── Summary ───────────────────────────────────────────────────────
    banner("All scenarios PASSED")
    print()
    print("  Decision log entries printed above by logging.INFO.")
    print("  Every entry shows: timestamp, mode, prev_mode, reason, is_override.")
    print()
    print("  To use the engine in the live system:")
    print("    python main.py --dry-run")
    print()


if __name__ == "__main__":
    main()

