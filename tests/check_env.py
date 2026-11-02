"""
tests/check_env.py – Environment verification script for Stage 1.

Imports every required library, lists available webcam and microphone
devices, and prints PASS / FAIL for each check.
"""

import sys
import importlib
from typing import List, Tuple

# ── Libraries to verify ─────────────────────
REQUIRED_LIBS: List[Tuple[str, str]] = [
    # (import_name, display_name)
    ("cv2",             "opencv-python"),
    ("numpy",           "numpy"),
    ("mediapipe",       "mediapipe"),
    ("faster_whisper",  "faster-whisper"),
    ("sounddevice",     "sounddevice"),
    ("webrtcvad",       "webrtcvad-wheels"),
    ("pynput",          "pynput"),
    ("pyautogui",       "pyautogui"),
    ("uiautomation",    "uiautomation"),
    ("win32api",        "pywin32"),
    ("PyQt5",           "PyQt5"),
    ("pandas",          "pandas"),
    ("matplotlib",      "matplotlib"),
    ("sklearn",         "scikit-learn"),
]


def check_libraries() -> bool:
    """Import each library and report PASS/FAIL."""
    print("\n" + "=" * 60)
    print("  LIBRARY IMPORT CHECKS")
    print("=" * 60)

    all_ok = True
    for import_name, display_name in REQUIRED_LIBS:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "N/A")
            print(f"  [PASS] {display_name:<22s}  (version: {version})")
        except Exception as e:
            print(f"  [FAIL] {display_name:<22s}  -> {e}")
            all_ok = False

    return all_ok


def check_webcams() -> bool:
    """Try to open camera indices 0–3 and list available ones."""
    print("\n" + "=" * 60)
    print("  WEBCAM DEVICE CHECK")
    print("=" * 60)

    try:
        import cv2  # type: ignore
    except ImportError:
        print("  [FAIL] Cannot import cv2 – skipping webcam check.")
        return False

    found_any = False
    for idx in range(4):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  [PASS] Camera index {idx}: {w}x{h} @ {fps:.0f} FPS")
            found_any = True
            cap.release()
        else:
            print(f"  [----] Camera index {idx}: not available")

    if not found_any:
        print("  [WARN] No webcam detected. Face tracking will not work"
              " until a camera is connected.")
    return found_any


def check_microphones() -> bool:
    """List available audio input devices via sounddevice."""
    print("\n" + "=" * 60)
    print("  MICROPHONE DEVICE CHECK")
    print("=" * 60)

    try:
        import sounddevice as sd  # type: ignore
    except ImportError:
        print("  [FAIL] Cannot import sounddevice – skipping mic check.")
        return False

    devices = sd.query_devices()
    found_any = False
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  [PASS] Input #{i}: {dev['name']}  "
                  f"(channels: {dev['max_input_channels']}, "
                  f"rate: {int(dev['default_samplerate'])} Hz)")
            found_any = True

    if not found_any:
        print("  [WARN] No microphone detected. Speech recognition will"
              " not work until a mic is connected.")
    return found_any


def main() -> None:
    """Run all environment checks and print summary."""
    print("\n" + "#" * 60)
    print("  Assistive Control System – Environment Check")
    print(f"  Python {sys.version}")
    print(f"  Platform: {sys.platform}")
    print("#" * 60)

    lib_ok = check_libraries()
    cam_ok = check_webcams()
    mic_ok = check_microphones()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Libraries : {'PASS' if lib_ok else 'FAIL'}")
    print(f"  Webcam    : {'PASS' if cam_ok else 'WARN - no camera found'}")
    print(f"  Microphone: {'PASS' if mic_ok else 'WARN - no mic found'}")

    if lib_ok:
        print("\n  [OK]  Environment is ready for Stage 2.")
    else:
        print("\n  [ERROR]  Some libraries failed to import. Fix errors"
              " above before proceeding.")

    print("=" * 60 + "\n")
    sys.exit(0 if lib_ok else 1)


if __name__ == "__main__":
    main()
