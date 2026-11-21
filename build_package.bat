@echo off
echo Building Assistive Control System package...
call .venv\Scripts\activate.bat
pip install pyinstaller
pyinstaller assistive.spec --clean --noconfirm
echo Done. Output: dist\assistive\assistive.exe
