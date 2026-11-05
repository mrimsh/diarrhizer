"""Launch the installed GUI and create Windows entry points without activation."""

import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from .config import State, atomic_text
from .setup import environment_python, runtime_env
from .system import CREATE_NO_WINDOW, capture


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _encoded(script: str) -> str:
    preamble = ("$ProgressPreference='SilentlyContinue'; "
                "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); ")
    return base64.b64encode((preamble + script).encode("utf-16-le")).decode("ascii")


# [SEMANTIC-BEGIN] LAUNCHER:START
# @purpose: Open the main GUI with the selected environment, paths and first-launch defaults
# @description: Shortcuts call venv/pythonw directly with a generated entry script; the .cmd fallback
#   uses an encoded PowerShell command for Unicode paths. No venv activation or main-app exe is needed.
#   Defaults are applied once per settings revision, preserving subsequent edits in the main GUI.
# @inputs: persisted ready state and an explicitly chosen shortcut destination
# @outputs: GUI process, start files and optional .lnk
# @sideEffects: Local start files/shortcut, child process, GUI settings and append-only GUI error log
# @errors: Invalid environments fail visibly; failed GUI startup remains in logs/gui.log
# @see: LAUNCHER:CONFIG, LAUNCHER:READY, LAUNCHER:SETUP
def start_file(home: Path) -> Path:
    return home / "Start Diarrhizer.cmd"


def write_start_files(state_path: Path, state: State) -> None:
    home = state_path.parent
    entry = home / "gui_entry.py"
    atomic_text(entry, "from pathlib import Path\nimport multiprocessing\n"
                "from diarrhizer_launcher.launch import gui_entry\n"
                "if __name__ == '__main__':\n    multiprocessing.freeze_support()\n"
                f"    gui_entry(Path({str(state_path)!r}))\n")
    pythonw = str(environment_python(state.settings.venv, windowed=True))
    script = ("$ErrorActionPreference='Stop'; Start-Process -WindowStyle Hidden -FilePath "
              + _literal(pythonw) + " -ArgumentList "
              + _literal(subprocess.list2cmdline(["-I", str(entry)]))
              + " -WorkingDirectory " + _literal(state.settings.config_dir))
    atomic_text(start_file(home), "@echo off\npowershell.exe -NoProfile -NonInteractive -WindowStyle Hidden "
                "-EncodedCommand " + _encoded(script) + "\nif errorlevel 1 pause\n")


def start_gui(state_path: Path, state: State) -> subprocess.Popen:
    if not state.ready:
        raise ValueError("Сначала завершите настройку окружения.")
    write_start_files(state_path, state)
    return subprocess.Popen([str(environment_python(state.settings.venv, windowed=True)), "-I",
                             str(state_path.parent / "gui_entry.py")],
                            cwd=state.settings.config_dir, env=runtime_env(state.settings),
                            creationflags=CREATE_NO_WINDOW)


def desktop_folder() -> Path:
    try:
        return Path(capture(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                             _encoded("[Environment]::GetFolderPath('DesktopDirectory')")]))
    except (OSError, subprocess.SubprocessError):
        return Path.home() / "Desktop"


def create_shortcut(state_path: Path, state: State, destination: Path) -> None:
    if not state.ready:
        raise ValueError("Сначала завершите настройку окружения.")
    if destination.suffix.lower() != ".lnk":
        raise ValueError("Укажите имя ярлыка с расширением .lnk.")
    write_start_files(state_path, state)
    pythonw = str(environment_python(state.settings.venv, windowed=True))
    script = ("$ErrorActionPreference='Stop'; $shell=New-Object -ComObject WScript.Shell; "
              "$link=$shell.CreateShortcut(" + _literal(str(destination)) + "); "
              "$link.TargetPath=" + _literal(pythonw) + "; "
              "$link.Arguments=" + _literal(subprocess.list2cmdline(["-I", str(state_path.parent / "gui_entry.py")])) + "; "
              "$link.WorkingDirectory=" + _literal(state.settings.config_dir) + "; "
              "$link.Description='Diarrhizer: основной GUI'; $link.Save()")
    capture(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", _encoded(script)])


def apply_gui_defaults(settings, config) -> None:
    revision = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    if settings.value("app/launcher_revision", "") != revision:
        settings.setValue("app/out_dir", config.output)
        settings.setValue("app/default_device", config.device)
        # New Job otherwise restores its previous device after reading the defaults.
        settings.remove("new_job/device")
        settings.setValue("app/launcher_revision", revision)
        settings.sync()


def gui_entry(state_path: Path) -> None:
    try:
        state = State.load(state_path)
        if state is None or not state.ready:
            raise ValueError("Окружение не готово. Откройте лаунчер Diarrhizer и завершите настройку.")
        if Path(sys.prefix).resolve() != Path(state.settings.venv).resolve():
            raise ValueError("Используется другой Python. Запустите GUI через лаунчер или созданный ярлык.")
        os.environ.update(runtime_env(state.settings))
        # Dynamic imports keep Qt/ML out of the independent launcher bundle.
        qt = importlib.import_module("PySide6.QtCore")
        apply_gui_defaults(qt.QSettings("Diarrhizer", "DiarrhizerGUI"), state.settings)
        app = importlib.import_module("diarrhizer_gui.app")
        raise SystemExit(app.main())
    except Exception as exc:
        log = state_path.parent / "logs" / "gui.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, f"Не удалось открыть Diarrhizer: {exc}\n\nЖурнал: {log}",
                                            "Diarrhizer", 0x10)
        else:
            raise
# [SEMANTIC-END] LAUNCHER:START
