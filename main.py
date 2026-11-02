"""
main.py – Entry point for the Assistive Control System.

Initialises logging, checks safety settings, and will later
bootstrap face-tracking, speech recognition, and the context engine.
"""

import sys
import logging
from pathlib import Path

# ── Local imports ────────────────────────────
import config

# ── Logging setup ────────────────────────────
def _setup_logging() -> logging.Logger:
    """Configure rotating file + console logging."""
    from logging.handlers import RotatingFileHandler

    logger = logging.getLogger("assistive")
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Rotating file handler
    log_file = config.LOG_DIR / "assistive.log"
    file_h = RotatingFileHandler(
        log_file,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_h.setFormatter(fmt)
    logger.addHandler(file_h)

    return logger


def _apply_safety() -> None:
    """Enable PyAutoGUI fail-safe and register the emergency-stop hotkey."""
    import pyautogui

    pyautogui.FAILSAFE = config.PYAUTOGUI_FAILSAFE
    pyautogui.PAUSE = config.PYAUTOGUI_PAUSE


def main() -> None:
    """Application entry point."""
    logger = _setup_logging()
    logger.info("Assistive Control System starting …")
    logger.info("Python %s on %s", sys.version, sys.platform)

    _apply_safety()
    logger.info("PyAutoGUI fail-safe enabled, emergency hotkey: %s",
                config.EMERGENCY_STOP_HOTKEY)

    # Future stages will initialise subsystems here.
    logger.info("Stage 1 complete – environment verified. Exiting.")


if __name__ == "__main__":
    main()
