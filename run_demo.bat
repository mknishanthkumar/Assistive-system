@echo off
:: run_demo.bat — Double-click launcher for the Assistive Control System Demo Panel.
:: Uses the project .venv Python. Falls back to system Python if venv is absent.

setlocal

set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "LAUNCHER=%SCRIPT_DIR%launcher.py"

if exist "%VENV_PYTHON%" (
    echo Using venv Python: %VENV_PYTHON%
    "%VENV_PYTHON%" "%LAUNCHER%"
) else (
    echo Venv not found — using system Python
    python "%LAUNCHER%"
)

endlocal

