"""Opening environments recovers parameters without installing or changing anything."""

from dataclasses import replace
from pathlib import Path

import pytest

from diarrhizer_launcher import existing
from diarrhizer_launcher.config import Settings, State


def test_snapshot_recovers_parameters_without_writes(tmp_path, monkeypatch):
    folder = tmp_path / "venv"
    folder.mkdir()
    (folder / "pyvenv.cfg").write_text("home = python")
    settings = Settings(str(folder), str(tmp_path / "models"), str(tmp_path / "results"), str(tmp_path / "config"))
    state = State(settings, managed=True, ready=True)
    existing.remember_environment(tmp_path / "settings.json", state)
    before = (folder / existing.MARKER).read_bytes()
    monkeypatch.setattr(existing, "check_ready", lambda _: (True, "OK"))
    opened, ready, reason = existing.inspect_environment(folder, None, replace(settings, output="different"))
    assert ready and opened.settings == settings
    assert "загружены" in reason
    assert (folder / existing.MARKER).read_bytes() == before
    assert not (tmp_path / "settings.json").exists()


def test_current_configuration_works_for_legacy_environment_without_marker(tmp_path, monkeypatch):
    folder = tmp_path / "venv"
    folder.mkdir()
    (folder / "pyvenv.cfg").touch()
    settings = Settings(str(folder), "cache", "output", "config")
    current = State(settings, ready=True)
    monkeypatch.setattr(existing, "check_ready", lambda _: (True, "OK"))
    opened, ready, _ = existing.inspect_environment(folder, current, settings)
    assert ready and opened.settings == current.settings
    opened.settings.output = "edited"
    assert current.settings.output == "output"
    assert not (folder / existing.MARKER).exists()


def test_unregistered_environment_probes_python_device_and_labels_unknown_paths(tmp_path, monkeypatch):
    folder = tmp_path / "venv"
    folder.mkdir()
    (folder / "pyvenv.cfg").touch()
    defaults = Settings("old", "cache", "output", "config")
    monkeypatch.setattr(existing, "probe_python", lambda *_: {"base": str(tmp_path / "base-python")})
    monkeypatch.setattr(existing, "capture", lambda *_: '{"torch":"2.3.1+cu121"}')
    monkeypatch.setattr(existing, "check_ready", lambda _: (True, "OK"))
    state, ready, reason = existing.inspect_environment(folder, None, defaults)
    assert ready and state.settings.device == "cuda"
    assert state.settings.output == defaults.output
    assert "предлагаемые" in reason
    assert Path(state.settings.python).parent == tmp_path / "base-python"
    assert not state.managed


def test_corrupt_snapshot_is_not_silently_replaced(tmp_path):
    (tmp_path / "pyvenv.cfg").touch()
    marker = tmp_path / existing.MARKER
    marker.write_text("broken")
    with pytest.raises(ValueError, match="прочитать"):
        existing.inspect_environment(tmp_path, None, Settings("a", "b", "c", "d"))
    assert marker.read_text() == "broken"
