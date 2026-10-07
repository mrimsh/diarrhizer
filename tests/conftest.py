"""Shared fixtures for the GUI tests."""

import os

import pytest


@pytest.fixture(scope="session")
def qapp():
    """One offscreen QApplication for every Qt test in the session (widgets
    need a QApplication, not just a QCoreApplication, and Qt allows only one)."""
    pytest.importorskip("PySide6", reason="GUI extra not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from diarrhizer_gui.child_process import kill_all

    app = QApplication.instance() or QApplication([])
    yield app
    # Child processes are not daemons: a test that failed mid-way must not
    # leave one running (or the interpreter would wait for it at exit).
    kill_all()


@pytest.fixture
def ini_settings(tmp_path, monkeypatch):
    """Point every screen's QSettings("Diarrhizer", "DiarrhizerGUI") at a
    throwaway INI file, so tests never touch the real settings store."""
    pytest.importorskip("PySide6", reason="GUI extra not installed")
    from PySide6.QtCore import QSettings

    path = str(tmp_path / "gui-settings.ini")

    def factory(*_args, **_kwargs):
        return QSettings(path, QSettings.Format.IniFormat)

    return path, factory
