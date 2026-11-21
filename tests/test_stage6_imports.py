def test_overlay_import():
    from output.overlay import OverlayManager
    m = OverlayManager()
    assert hasattr(m, 'start')
    assert hasattr(m, 'update')
    assert hasattr(m, 'stop')

def test_hotkeys_import():
    from output.hotkeys import HotkeyManager
    # just check import and instantiation with mocks
    class FakeDE:
        current_mode = None
        def set_manual_override(self, m): pass
        def clear_manual_override(self): pass
    m = HotkeyManager(FakeDE())
    assert hasattr(m, 'start')

def test_undo_import():
    from output.undo import UndoManager
    u = UndoManager()
    assert hasattr(u, 'undo')
    assert u.is_undo_phrase("go back")
    assert u.is_undo_phrase("undo")
    assert u.is_undo_phrase("cancel that")

def test_calibration_import():
    from calibration.calibration import CalibrationManager
    c = CalibrationManager()
    assert hasattr(c, 'needs_calibration')

def test_config_stage6():
    import config
    assert hasattr(config, 'STUDY_CONDITION')
    assert hasattr(config, 'HOTKEY_FORCE_FACE')
    assert hasattr(config, 'OVERLAY_ENABLED')

def test_analysis_import():
    from analysis.summarize_logs import summarize
    # summarize should be callable
    assert callable(summarize)
