"""Read environment settings without installing or changing anything."""

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

from .config import Settings, State
from .setup import check_ready, environment_python
from .system import capture, probe_python

MARKER = ".diarrhizer-launcher.json"


# [SEMANTIC-BEGIN] LAUNCHER:OPEN_ENVIRONMENT
# @purpose: Open an existing environment and recover its launcher parameters without modifying packages
# @description: Prefer the current configuration, then the environment's settings snapshot. For an
#   unregistered venv inspect Python and wheel metadata; cache/results remain proposed defaults.
#   Reading never saves or adopts a configuration; explicit use/apply persists the selected environment.
# @inputs: selected venv directory, current state and proposed defaults
# @outputs: recovered state, readiness and explanation; optional ready-state snapshot
# @sideEffects: Reads local JSON and runs isolated Python probes; remember_environment writes a snapshot
# @errors: ValueError for corrupt snapshots or incompatible Python; subprocess/OSError failures
# @see: LAUNCHER:UI, LAUNCHER:READY, LAUNCHER:CONFIG
def inspect_environment(folder: Path, current: State | None, defaults: Settings) -> tuple[State, bool, str]:
    folder = folder.resolve()
    if not (folder / "pyvenv.cfg").is_file():
        raise ValueError("В выбранной папке нет Python-окружения (pyvenv.cfg).")
    recovered = False
    if current and Path(current.settings.venv).resolve() == folder:
        state = replace(current, settings=replace(current.settings), completed=list(current.completed))
        recovered = True
    else:
        state = State.load(folder / MARKER)
        if state:
            if Path(state.settings.venv).resolve() != folder:
                raise ValueError("Путь в настройках среды отличается от выбранной папки. "
                                 "Окружение могло быть перемещено; проверьте его Python и настройте заново.")
            recovered = True
        else:
            python = str(environment_python(str(folder)))
            facts = probe_python(python, folder)
            code = ("import importlib.metadata as m,json; "
                    "print(json.dumps({'torch':m.version('torch')}))")
            try:
                version = json.loads(capture([python, "-I", "-c", code]))["torch"]
            except (ValueError, OSError, RuntimeError, subprocess.SubprocessError):
                version = ""
            base = Path(facts["base"]) / ("python.exe" if os.name == "nt" else "bin/python")
            settings = replace(defaults, venv=str(folder), python=str(base),
                               device="cuda" if "+cu" in version else "cpu")
            state = State(settings, ready=True)
    ready, reason = check_ready(state)
    state.ready = ready
    note = "Параметры среды загружены. " if recovered else (
        "Среда обнаружена. Python и режим определены; пути кэша, результатов и FFmpeg — "
        "предлагаемые значения, проверьте их перед использованием. ")
    return state, ready, note + reason


def remember_environment(path: Path, state: State) -> None:
    if state.ready:
        state.save(Path(state.settings.venv) / MARKER)
# [SEMANTIC-END] LAUNCHER:OPEN_ENVIRONMENT
