"""Launcher safety, recovery and Windows entry-point contracts; no external downloads/models."""

import base64
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from diarrhizer_launcher import launch, setup, system
from diarrhizer_launcher.config import LauncherLock, Settings, State


@pytest.fixture
def plan(tmp_path):
    source = tmp_path / "application"
    source.mkdir()
    home = tmp_path / "launcher"
    settings = Settings(str(tmp_path / "runtime"), str(tmp_path / "models"),
                        str(tmp_path / "results"), str(tmp_path / "config"), python=sys.executable)
    return State(settings), home / "settings.json", source


def test_atomic_state_survives_failed_replacement(plan, monkeypatch):
    state, file, _ = plan
    state.save(file)
    state.ready = True
    monkeypatch.setattr(os, "replace", lambda *_: (_ for _ in ()).throw(OSError("locked")))
    with pytest.raises(OSError):
        state.save(file)
    assert not State.load(file).ready


@pytest.mark.parametrize("raw", ['{"schema":99}', '{"schema":1,"settings":{}}', 'not JSON'])
def test_corrupt_state_is_never_replaced(plan, raw):
    _, file, _ = plan
    file.parent.mkdir()
    file.write_text(raw)
    with pytest.raises(ValueError, match="прочитать"):
        State.load(file)
    assert file.read_text() == raw


def test_separate_environment_cannot_contain_app_or_results(plan):
    state, file, source = plan
    state.settings.validate(source, file.parent)
    for target in (source, file.parent, Path(state.settings.output)):
        with pytest.raises(ValueError, match="отдельной папке"):
            replace(state.settings, venv=str(target)).validate(source, file.parent)


def test_only_one_launcher_owns_state(plan):
    _, file, _ = plan
    first = LauncherLock(file.parent)
    try:
        with pytest.raises(RuntimeError, match="уже открыт"):
            LauncherLock(file.parent)
    finally:
        first.close()
    second = LauncherLock(file.parent)
    second.close()


@pytest.mark.parametrize("version,bits", [([3, 13, 1], 64), ([3, 12, 10], 32), ([3, 10, 0], 64)])
def test_reject_unsupported_python(monkeypatch, version, bits):
    monkeypatch.setattr(system, "capture", lambda *_: json.dumps({
        "version": version, "bits": bits, "implementation": "CPython"}))
    with pytest.raises(ValueError, match="CPython 3.11 или 3.12"):
        system.probe_python("python")


def test_reject_global_python_as_venv(plan, monkeypatch):
    state, _, _ = plan
    monkeypatch.setattr(system, "capture", lambda *_: json.dumps({
        "version": [3, 12, 10], "bits": 64, "implementation": "CPython",
        "prefix": str(Path(sys.base_prefix)), "base": str(Path(sys.base_prefix))}))
    with pytest.raises(ValueError, match="не принадлежит"):
        system.probe_python(sys.executable, Path(state.settings.venv))


def test_existing_venv_requires_approval_and_preserves_files(plan, monkeypatch):
    state, file, source = plan
    folder = Path(state.settings.venv)
    folder.mkdir()
    (folder / "pyvenv.cfg").write_text("existing")
    sentinel = folder / "keep.txt"
    sentinel.write_text("user packages")
    worker = setup.SetupWorker(state, file, source, lambda _: None)
    with pytest.raises(ValueError, match="подтверждения"):
        worker._venv()
    assert not state.managed
    monkeypatch.setattr(system, "probe_python", lambda *_: {"version": [3, 12, 10]})
    worker.approve_existing = True
    worker._venv()
    assert state.managed
    assert sentinel.read_text() == "user packages"
    assert (folder / "pyvenv.cfg").read_text() == "existing"


def test_nonvenv_directory_is_not_modified_even_with_approval(plan):
    state, file, source = plan
    folder = Path(state.settings.venv)
    folder.mkdir()
    sentinel = folder / "personal.txt"
    sentinel.write_text("keep")
    worker = setup.SetupWorker(state, file, source, lambda _: None, approve_existing=True)
    with pytest.raises(ValueError, match="не является venv"):
        worker._venv()
    assert list(folder.iterdir()) == [sentinel]


def test_failed_command_is_logged_and_resume_starts_at_failed_stage(plan):
    state, file, source = plan
    calls, events = [], []
    worker = setup.SetupWorker(state, file, source, events.append)
    for name, _ in setup.STEPS:
        setattr(worker, "_" + name, lambda name=name: calls.append(name))

    def fail():
        calls.append("dependencies")
        worker.command([sys.executable, "-I", "-c", "print('installation failed', flush=True); raise SystemExit(7)"])

    worker._dependencies = fail
    worker.run()
    saved = State.load(file)
    assert saved.completed == ["python", "venv", "ffmpeg"]
    assert not saved.ready
    assert saved.failed_step == "dependencies"
    assert "installation failed" in Path(saved.log).read_text(encoding="utf-8")
    assert events[-1]["kind"] == "error"
    worker._dependencies = lambda: calls.append("dependencies")
    calls.clear()
    worker.run()
    assert calls == ["dependencies", "diagnostics", "entrypoint"]
    assert State.load(file).ready
    assert not State.load(file).failed_step
    calls.clear()
    worker.run(repair=True)
    assert calls == [name for name, _ in setup.STEPS]
    assert worker.repair
    worker._dependencies = fail
    worker.run(repair=True)
    saved = State.load(file)
    assert saved.repair
    resumed = setup.SetupWorker(saved, file, source, events.append)
    calls.clear()
    for name, _ in setup.STEPS:
        setattr(resumed, "_" + name, lambda name=name: calls.append(name))
    resumed.run()
    assert resumed.repair
    assert calls == ["dependencies", "diagnostics", "entrypoint"]
    assert not State.load(file).repair


def test_command_timeout_is_retryable_and_redacts_credentials(plan):
    state, file, source = plan
    file.parent.mkdir()
    events = []
    worker = setup.SetupWorker(state, file, source, events.append)
    with pytest.raises(RuntimeError, match="прерван"):
        worker.command([sys.executable, "-I", "-c", "import time; print('hf_private123', flush=True); time.sleep(60)"],
                       timeout=1)
    log = "\n".join(e.get("message", "") for e in events)
    assert "hf_private123" not in log
    assert "[HF_TOKEN]" in log


def test_verified_download_is_cached_and_bad_checksum_never_publishes(api, tmp_path):
    api.body = b"installer bytes" * 100000
    target = tmp_path / "installer.exe"
    digest = hashlib.sha256(api.body).hexdigest()
    events = []
    system.download(api.url, target, digest, events.append)
    assert target.read_bytes() == api.body
    assert events[-1]["received"] == len(api.body)
    system.download(api.url, target, digest, events.append)
    assert len(api.requests) == 1
    target.unlink()
    with pytest.raises(ValueError, match="Контрольная сумма"):
        system.download(api.url, target, "0" * 64, events.append)
    assert not target.exists()
    assert not target.with_suffix(".exe.part").exists()


def test_ffmpeg_archive_cannot_write_outside_component_folder(tmp_path, monkeypatch):
    archive = tmp_path / "ffmpeg.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("release/bin/ffmpeg.exe", b"ffmpeg")
        stream.writestr("release/bin/ffprobe.exe", b"ffprobe")
        stream.writestr("../../outside.txt", b"must not be extracted")
    monkeypatch.setattr(system.urllib.request, "urlopen", lambda *_, **__: io.BytesIO(b"a" * 64))
    monkeypatch.setattr(system, "download", lambda *args: archive)
    executable = Path(system.install_ffmpeg(tmp_path / "home", lambda _: None))
    assert executable.read_bytes() == b"ffmpeg"
    assert executable.with_name("ffprobe.exe").read_bytes() == b"ffprobe"
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("device,variant", [("cpu", "cpu.txt"), ("cuda", "cuda-cu121.txt")])
def test_install_plan_uses_single_supported_profile_and_never_editable_source(plan, monkeypatch, device, variant):
    state, file, source = plan
    state.settings.device = device
    calls = []
    monkeypatch.setattr(system, "probe_python", lambda *_: {})
    worker = setup.SetupWorker(state, file, source, lambda _: None)
    worker.command = lambda args, **kw: calls.append((args, kw))
    worker._dependencies()
    installs = [args for args, _ in calls if "install" in args]
    assert any(str(source / "requirements" / variant) in args for args in installs)
    assert not any(str(source / "requirements" / ("cpu.txt" if device == "cuda" else "cuda-cu121.txt")) in args
                   for args in installs)
    assert all("-e" not in args for args in installs)
    assert str(source) + "[gui]" in installs[-1]
    assert all(args[0] == str(setup.environment_python(state.settings.venv)) for args, _ in calls)
    assert all(kw["env"]["PIP_REQUIRE_VIRTUALENV"] == "true" for _, kw in calls)


def test_start_files_are_unicode_safe_and_spawn_import_does_not_open_gui(plan, monkeypatch):
    state, file, _ = plan
    state.settings.venv += " пробел % $ апостроф'"
    state.ready = True
    launch.write_start_files(file, state)
    text = launch.start_file(file.parent).read_text(encoding="utf-8")
    encoded = text.split("-EncodedCommand ")[1].splitlines()[0]
    decoded = base64.b64decode(encoded).decode("utf-16-le")
    assert state.settings.venv.replace("'", "''") in decoded
    assert "-I" in decoded
    monkeypatch.setattr(launch, "gui_entry", lambda *_: pytest.fail("GUI recursively opened by spawn"))
    runpy.run_path(str(file.parent / "gui_entry.py"), run_name="__mp_main__")


def test_gui_launch_uses_venv_and_selected_paths(plan, monkeypatch):
    state, file, _ = plan
    calls = []
    state.ready = True
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *args, **kw: calls.append((args, kw)))
    launch.start_gui(file, state)
    argv, kw = calls[0]
    assert argv[0][0] == str(setup.environment_python(state.settings.venv, windowed=True))
    assert argv[0][1] == "-I"
    assert kw["env"]["HF_HOME"] == state.settings.cache
    assert kw["env"]["TORCH_HOME"] == str(Path(state.settings.cache) / "torch")
    assert kw["cwd"] == state.settings.config_dir


def test_launcher_defaults_preserve_gui_changes_until_configuration_changes(plan, tmp_path):
    qt = pytest.importorskip("PySide6.QtCore")
    state, _, _ = plan
    store = qt.QSettings(str(tmp_path / "gui.ini"), qt.QSettings.Format.IniFormat)
    store.setValue("new_job/device", "cuda")
    launch.apply_gui_defaults(store, state.settings)
    assert store.value("app/out_dir") == state.settings.output
    assert store.value("app/default_device") == "cpu"
    assert not store.contains("new_job/device")
    store.setValue("app/out_dir", "user changed it")
    launch.apply_gui_defaults(store, state.settings)
    assert store.value("app/out_dir") == "user changed it"
    state.settings.output += " new"
    launch.apply_gui_defaults(store, state.settings)
    assert store.value("app/out_dir") == state.settings.output


@pytest.mark.skipif(os.name != "nt", reason="Windows shortcut contract")
def test_real_shortcut_targets_pythonw_with_unicode_arguments(plan):
    state, file, _ = plan
    state.settings.venv = sys.prefix
    state.ready = True
    file = file.parent / "настройки % $ ' пробел.json"
    shortcut = file.parent / "Diarrhizer тест.lnk"
    launch.create_shortcut(file, state, shortcut)
    script = ("$l=(New-Object -ComObject WScript.Shell).CreateShortcut("
              + launch._literal(str(shortcut)) + "); "
              "@{target=$l.TargetPath;args=$l.Arguments;cwd=$l.WorkingDirectory}|ConvertTo-Json -Compress")
    raw = system.capture(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", launch._encoded(script)])
    assert raw.startswith("{"), repr(raw)
    facts = json.loads(raw)
    assert Path(facts["target"]).resolve() == setup.environment_python(sys.prefix, windowed=True).resolve()
    assert "-I" in facts["args"]
    assert str(file.parent / "gui_entry.py") in facts["args"]
    assert Path(facts["cwd"]).resolve() == Path(state.settings.config_dir).resolve()


def test_launcher_import_does_not_load_processing_or_qt():
    source = Path(__file__).resolve().parents[1] / "src"
    # The package root contains only release identity; no core submodule may load here.
    script = (f"import sys; sys.path.insert(0, {str(source)!r}); import diarrhizer_launcher.app; "
              "assert not any(n in sys.modules for n in ('torch','PySide6','whisperx')); "
              "assert not any(n.startswith('diarrhizer.') for n in sys.modules)")
    subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=20,
                   creationflags=system.CREATE_NO_WINDOW)


def test_missing_python_is_provisioned_only_when_allowed(plan, monkeypatch):
    state, file, source = plan
    state.settings.python = ""
    state.settings.install_python = False
    monkeypatch.setattr(system, "discover_pythons", list)
    monkeypatch.setattr(system, "probe_python", lambda *_: {"version": [3, 12, 10]})
    downloads, commands = [], []
    monkeypatch.setattr(system, "download", lambda *args: downloads.append(args) or source / "installer.exe")
    worker = setup.SetupWorker(state, file, source, lambda _: None)
    worker.command = lambda args, **_: commands.append(args)
    with pytest.raises(ValueError, match="не найден"):
        worker._python()
    assert not downloads
    state.settings.install_python = True
    worker._python()
    assert downloads[0][0] == system.PYTHON_URL
    assert downloads[0][2] == system.PYTHON_SHA256
    assert "InstallAllUsers=0" in commands[0]
    assert "PrependPath=0" in commands[0]
    assert str(file.parent / "components" / "Python312" / "python.exe") == state.settings.python


def test_missing_gpu_stops_cuda_before_environment_changes(plan, monkeypatch):
    state, file, source = plan
    state.settings.device = "cuda"
    monkeypatch.setattr(system, "gpu_description", lambda: "")
    worker = setup.SetupWorker(state, file, source, lambda _: None)
    with pytest.raises(ValueError, match="Выберите CPU"):
        worker._python()
    assert not Path(state.settings.venv).exists()


def test_saved_ready_flag_does_not_hide_missing_environment(plan):
    state, _, _ = plan
    state.ready = True
    ready, reason = setup.check_ready(state)
    assert not ready
    assert "восстановить" in reason


def test_frozen_launcher_in_checkout_uses_updated_sources(tmp_path, monkeypatch):
    from diarrhizer_launcher.config import application_source, source_version

    (tmp_path / '.git').mkdir()
    (tmp_path / 'src/diarrhizer').mkdir(parents=True)
    (tmp_path / 'src/diarrhizer/__init__.py').write_text('__version__ = "2.10"')
    (tmp_path / 'requirements').mkdir()
    (tmp_path / 'requirements/constraints-stable.txt').touch()
    (tmp_path / 'pyproject.toml').touch()
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path / 'bundle'), raising=False)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'dist/Diarrhizer.exe'))
    assert application_source() == tmp_path
    assert source_version(application_source()) == '2.10'
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'elsewhere/bin/Diarrhizer.exe'))
    assert application_source() == tmp_path / 'bundle/application'
    assert application_source(tmp_path / '.venvcuda') == tmp_path


def test_application_only_update_never_reinstalls_dependencies(plan, monkeypatch):
    state, file, source = plan
    (source / 'src/diarrhizer').mkdir(parents=True)
    (source / 'src/diarrhizer/__init__.py').write_text('__version__ = "2.1"')
    monkeypatch.setattr(system, 'probe_python', lambda *_: {})
    worker = setup.SetupWorker(state, file, source, lambda _: None)
    worker.application_only = True
    calls = []
    worker.command = lambda args, **kw: calls.append(args)
    worker._dependencies()
    assert '--no-deps' in calls[0] and '--force-reinstall' in calls[0]
    assert not any('-r' in args or 'ensurepip' in args for args in calls)
    assert calls[-1][-1] == '2.1'
    assert state.application_version == '2.1'
