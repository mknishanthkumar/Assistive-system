# assistive.spec
# PyInstaller spec for the Assistive Control System
# Build: pyinstaller assistive.spec
# Run:   dist/assistive/assistive.exe

import sys
from pathlib import Path
block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[str(Path('.').resolve())],
    binaries=[],
    datas=[
        ('models/*.task', 'models'),
        ('calibration/user_settings.json', 'calibration') if Path('calibration/user_settings.json').exists() else ('', '.'),
    ],
    hiddenimports=[
        'output.overlay', 'output.hotkeys', 'output.undo',
        'calibration.calibration',
        'analysis.summarize_logs',
        'pynput.keyboard._win32', 'pynput.mouse._win32',
        'mediapipe', 'faster_whisper',
        'winsound', 'ctypes', 'tkinter',
        'sounddevice', 'webrtcvad',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='assistive',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='assistive',
)
