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


@pytest.fixture(autouse=True)
def summary_stores(tmp_path, monkeypatch):
    """Never read or write the developer's real <repo>/summary_templates.json
    and summary_providers.json. Returns (templates path, providers path)."""
    from diarrhizer.summary.providers import ENV_PROVIDERS_FILE
    from diarrhizer.summary.templates import ENV_TEMPLATES_FILE

    templates = tmp_path / "summary_templates.json"
    providers = tmp_path / "summary_providers.json"
    monkeypatch.setenv(ENV_TEMPLATES_FILE, str(templates))
    monkeypatch.setenv(ENV_PROVIDERS_FILE, str(providers))
    return templates, providers


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
