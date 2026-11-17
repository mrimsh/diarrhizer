"""Tk event handling and confirmation checks without installing dependencies."""

import tkinter as tk

import pytest

from diarrhizer_launcher.app import Launcher
from diarrhizer_launcher.config import Settings, State


@pytest.fixture
def ui(tmp_path):
    root = tk.Tk()
    root.withdraw()
    source = tmp_path / "source"
    source.mkdir()
    file = tmp_path / "launcher" / "settings.json"
    settings = Settings(str(tmp_path / ".venv"), str(tmp_path / "cache"),
                        str(tmp_path / "out"), str(tmp_path / "config"))
    State(settings).save(file)
    window = Launcher(root, file, source, probe=False)
    yield window
    for callback in root.tk.call("after", "info"):
        root.after_cancel(callback)
    root.destroy()


def test_confirmation_decline_leaves_existing_environment_and_state_untouched(ui, monkeypatch):
    from pathlib import Path

    folder = Path(ui.state.settings.venv)
    folder.mkdir()
    (folder / "pyvenv.cfg").write_text("old environment")
    previous = ui.state_path.read_bytes()
    prompts = []

    def decline(title, text, **_):
        prompts.append(text)
        return False

    monkeypatch.setattr("diarrhizer_launcher.app.messagebox.askyesno", decline)
    ui.begin()
    assert len(prompts) == 1
    assert "Окружение уже существует" in prompts[0]
    assert ui.state_path.read_bytes() == previous
    assert (folder / "pyvenv.cfg").read_text() == "old environment"
    assert not ui.busy


def test_failure_keeps_log_and_retry_then_enables_all_completion_actions(ui):
    ui.busy = True
    ui.state.failed_step = "dependencies"
    ui.state.error = "Network unavailable"
    ui.state.settings.python = "resolved Python path"
    ui.events.put({"kind": "stage", "index": 3, "total": 6, "message": "Dependencies"})
    ui.events.put({"kind": "log", "message": "Saved diagnostic output"})
    ui.events.put({"kind": "error", "step": "dependencies", "message": "Network unavailable"})
    ui.poll()
    assert not ui.busy
    assert "Повторить:" in ui.setup_button["text"]
    assert ui.variables["python"].get() == "resolved Python path"
    assert "disabled" in ui.launch_button.state()
    assert "Saved diagnostic output" in ui.log.get("1.0", "end")
    ui.state.ready = True
    ui.state.failed_step = ui.state.error = ""
    ui.events.put({"kind": "done"})
    ui.poll()
    for button in (ui.launch_button, ui.shortcut_button, ui.folder_button):
        assert "disabled" not in button.state()
    # Edited, unapplied settings cannot launch an environment with different saved parameters.
    ui.variables["output"].set(ui.state.settings.output + " changed")
    assert "disabled" in ui.launch_button.state()


def test_missing_ready_environment_disables_launch_and_offers_repair(ui):
    ui.scanning = True
    ui.state.ready = True
    ui.events.put({"kind": "discovered", "candidates": [], "gpu": "",
                   "ready": (False, "Environment unavailable")})
    ui.poll()
    assert not ui.scanning
    assert "disabled" in ui.launch_button.state()
    assert "disabled" not in ui.repair_button.state()
    assert ui.status.get() == "Environment unavailable"


def test_opened_environment_loads_fields_and_enables_shortcut_without_saving(ui):
    from dataclasses import replace

    previous = ui.state_path.read_bytes()
    settings = replace(ui.state.settings, venv=ui.state.settings.venv + "-other", output=ui.state.settings.output + "-other")
    ui.variables["venv"].set(settings.venv)
    ui.scanning = True
    ui.events.put({"kind": "opened", "folder": settings.venv, "state": State(settings, ready=True),
                   "ready": True, "message": "Loaded"})
    ui.poll()
    assert ui.variables["output"].get() == settings.output
    assert "disabled" not in ui.shortcut_button.state()
    assert ui.state_path.read_bytes() == previous


def test_editing_ready_environment_paths_saves_without_starting_installation(ui, monkeypatch):
    from pathlib import Path

    from diarrhizer_launcher.existing import MARKER

    folder = Path(ui.state.settings.venv)
    folder.mkdir()
    ui.state.ready = ui.available = True
    ui.variables["output"].set(ui.state.settings.output + "-edited")
    monkeypatch.setattr("diarrhizer_launcher.app.messagebox.askyesno", lambda *_, **__: True)
    ui.begin()
    assert ui.worker is None
    assert State.load(ui.state_path).settings.output.endswith("-edited")
    assert State.load(folder / MARKER).settings.output.endswith("-edited")
    assert "disabled" not in ui.shortcut_button.state()


def test_launcher_update_available_without_ready_environment_and_disabled_during_setup(ui):
    assert "disabled" not in ui.self_update_button.state()
    ui.busy = True
    ui.refresh()
    assert "disabled" in ui.self_update_button.state()


def test_declining_verified_launcher_update_preserves_settings_and_window(ui, tmp_path, monkeypatch):
    from diarrhizer_launcher.self_update import PreparedUpdate

    old = ui.state_path.read_bytes()
    staged = tmp_path / "candidate.exe"
    staged.write_bytes(b"new exe")
    ui.self_updating = True
    monkeypatch.setattr("diarrhizer_launcher.app.messagebox.askyesno", lambda *_, **__: False)
    ui.events.put({"kind": "launcher_prepared", "prepared": PreparedUpdate(tmp_path / "plan.json", staged, "2.10")})
    ui.poll()
    assert not ui.self_updating
    assert not staged.exists()
    assert ui.state_path.read_bytes() == old
    assert "disabled" not in ui.self_update_button.state()


def test_launcher_update_error_keeps_existing_environment_actions(ui):
    ui.available = ui.state.ready = True
    ui.self_updating = True
    ui.events.put({"kind": "launcher_error", "message": "Network unavailable"})
    ui.poll()
    assert not ui.self_updating
    assert "disabled" not in ui.launch_button.state()
    assert "Network unavailable" in ui.status.get()


def test_failed_launcher_replacement_is_reported_once_after_restart(ui, monkeypatch):
    folder = ui.state_path.parent / "launcher-updates"
    folder.mkdir()
    (folder / "last-result.json").write_text('{"status":"failed"}')
    warnings = []
    monkeypatch.setattr("diarrhizer_launcher.app.messagebox.showwarning", lambda *args, **_: warnings.append(args))
    before = ui.state_path.read_bytes()
    ui._report_launcher_update()
    ui._report_launcher_update()
    assert len(warnings) == 1
    assert "не выполнено" in ui.status.get()
    assert ui.state_path.read_bytes() == before
