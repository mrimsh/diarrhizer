"""Restartable setup stages; all long work runs outside the UI thread."""

import json
import os
import re
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from . import system
from .config import State, atomic_text, source_version

STEPS = (
    ("python", "Проверка и подготовка Python"),
    ("venv", "Создание / проверка окружения"),
    ("ffmpeg", "Проверка и подготовка FFmpeg"),
    ("dependencies", "Установка совместимых зависимостей"),
    ("diagnostics", "Диагностика окружения"),
    ("entrypoint", "Создание файла запуска GUI"),
)

STEP_HINTS = {
    "source": "Проверьте доступ к GitHub и повторите загрузку. Старая встроенная версия не устанавливалась.",
    "python": "Проверьте путь к Python 3.11/3.12 x64 и доступ к Интернету для загрузки.",
    "venv": "Проверьте доступ к папке окружения. Для несовместимой или сломанной venv выберите новую пустую папку.",
    "ffmpeg": "Выберите комплект ffmpeg.exe + ffprobe.exe либо разрешите их загрузку.",
    "dependencies": "Проверьте подключение к Интернету и свободное место, затем повторите этот этап.",
    "diagnostics": "Проверьте журнал и нажмите «Проверить и восстановить среду»; при проблемах CUDA можно выбрать CPU.",
    "entrypoint": "Проверьте доступ на запись к папке лаунчера, затем повторите этап.",
}

DEPENDENCY_CHECK = """
import importlib.metadata as metadata
import sys
import torch, torchaudio, ctranslate2
from PySide6.QtWidgets import QApplication
import diarrhizer_gui.app
expected = '2.3.1+' + ('cu121' if sys.argv[1] == 'cuda' else 'cpu')
for name in ('torch', 'torchaudio'):
    actual = metadata.version(name)
    if actual != expected:
        raise RuntimeError(f'{name}: {actual}; expected {expected}')
if sys.argv[1] == 'cuda':
    if not torch.cuda.is_available() or ctranslate2.get_cuda_device_count() < 1:
        raise RuntimeError('CUDA unavailable: check NVIDIA driver or select CPU')
    torch.zeros(1, device='cuda').cpu()
print('Torch / torchaudio / CTranslate2 / PySide6 / GUI: OK')
"""


# [SEMANTIC-BEGIN] LAUNCHER:READY
# @purpose: Resolve runtime paths and detect a finished environment without loading ML in the launcher
# @description: Saved readiness is necessary but not sufficient: the actual venv prefix, supported
#   Python, GUI package and selected torch wheel metadata are checked in its own interpreter.
#   Explicit cache variables keep inherited HF cache overrides from bypassing the selected location.
# @inputs: saved single-environment state
# @outputs: runtime executable/environment mapping; readiness boolean and a user-facing reason
# @sideEffects: One short hidden Python subprocess
# @errors: Missing/broken environments return a reason, never auto-recreate themselves
# @see: LAUNCHER:SETUP, LAUNCHER:START
def environment_python(venv: str, *, windowed: bool = False) -> Path:
    if os.name == "nt":
        return Path(venv) / "Scripts" / ("pythonw.exe" if windowed else "python.exe")
    return Path(venv) / "bin" / "python"


def runtime_env(settings) -> dict[str, str]:
    env = system.child_env()
    env.update(HF_HOME=settings.cache, TORCH_HOME=str(Path(settings.cache) / "torch"),
               HF_HUB_CACHE=str(Path(settings.cache) / "hub"),
               HUGGINGFACE_HUB_CACHE=str(Path(settings.cache) / "hub"),
               DIARRHIZER_CONFIG_DIR=settings.config_dir)
    env.pop("TRANSFORMERS_CACHE", None)
    if settings.ffmpeg:
        env["DIARRHIZER_FFMPEG_PATH"] = settings.ffmpeg
        env["PATH"] = str(Path(settings.ffmpeg).parent) + os.pathsep + env.get("PATH", "")
    return env


def check_ready(state: State) -> tuple[bool, str]:
    if not state.ready:
        return False, state.error or "Окружение ещё не настроено."
    python = str(environment_python(state.settings.venv))
    try:
        system.probe_python(python, Path(state.settings.venv))
        suffix = "cu121" if state.settings.device == "cuda" else "cpu"
        code = ("import importlib.util as u,importlib.metadata as m; "
                "assert u.find_spec('diarrhizer_gui.app'); "
                "assert u.find_spec('PySide6.QtWidgets'); "
                "assert u.find_spec('diarrhizer_launcher.launch'); "
                f"assert m.version('torch') == '2.3.1+{suffix}'; "
                f"assert m.version('torchaudio') == '2.3.1+{suffix}'")
        system.capture([python, "-I", "-c", code])
        return True, "Окружение готово. Можно открыть Diarrhizer."
    except (ValueError, OSError, subprocess.SubprocessError):
        return False, "Сохранённое окружение недоступно или повреждено. Нажмите «Проверить и восстановить среду»."
# [SEMANTIC-END] LAUNCHER:READY


# [SEMANTIC-BEGIN] LAUNCHER:SETUP
# @purpose: Prepare one environment through retryable Python, venv, FFmpeg, install and diagnostic stages
# @description: Checkpoint only successful stages. Existing environments need approval; no directory
#   is deleted or cleared. Resume skips successful stages; repair reruns them and keeps packages/data.
#   Install uses the repository's CPU/cu121 requirements and shared constraints, then a regular install
#   of the selected source snapshot, so downloaded/extracted files are never an editable install source.
#   Successful setup also snapshots parameters in the venv for reopening it by directory.
#   Application-only updates leave dependencies intact; source/installed versions are verified.
#   GUI setup resolves GitHub main before touching the environment; source revisions invalidate checkpoints.
# @inputs: settings state, source payload, event callback, explicit existing-environment approval
# @outputs: ready state, durable log, GUI start files; queue events for progress/errors
# @sideEffects: Local filesystem writes, hidden child processes, optional verified downloads and pip network
# @errors: Each failed step persists its name/log and remains retryable; setup never records credentials
# @see: LAUNCHER:CONFIG, LAUNCHER:PROBES, LAUNCHER:DOWNLOAD, LAUNCHER:START, DIAGNOSTICS:DOCTOR
class SetupWorker:
    def __init__(self, state: State, state_path: Path, source: Path, emit, *, approve_existing: bool = False,
                 remote: bool = False):
        self.state, self.state_path, self.source, self.emit = state, state_path, source, emit
        self.home = state_path.parent
        self.approve_existing = approve_existing
        self.cancel = threading.Event()
        self._stream = None
        self.repair = False
        self.application_only = False
        self.remote = remote

    def log(self, text: str) -> None:
        text = re.sub(r"hf_[A-Za-z0-9]+", "[HF_TOKEN]", text)
        text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[credentials]@", text)
        if self._stream:
            if self._stream.closed:
                with Path(self._stream.name).open("a", encoding="utf-8") as stream:
                    stream.write(text.rstrip() + "\n")
            else:
                self._stream.write(text.rstrip() + "\n")
                self._stream.flush()
        self.emit({"kind": "log", "message": text.rstrip()})

    def command(self, args: list[str], *, env: dict | None = None, timeout: int = 3600) -> None:
        """Stream stdout while a watchdog bounds hangs and cancellation kills the whole child tree."""
        self.log("> " + subprocess.list2cmdline([str(arg) for arg in args]))
        with subprocess.Popen([str(arg) for arg in args], cwd=self.home,
                              env=env or system.child_env(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                              creationflags=system.CREATE_NO_WINDOW) as process:
            timed_out = threading.Event()
            finished = threading.Event()

            def watch():
                remaining = timeout
                while not finished.wait(0.25):
                    remaining -= 0.25
                    if self.cancel.is_set() or remaining <= 0:
                        timed_out.set()
                        try:
                            if os.name == "nt":
                                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                               creationflags=system.CREATE_NO_WINDOW, timeout=20, check=True)
                            else:
                                process.kill()
                        except (OSError, subprocess.SubprocessError):
                            process.kill()
                        return

            watchdog = threading.Thread(target=watch, daemon=True)
            watchdog.start()
            try:
                for line in process.stdout:
                    self.log(line)
                returncode = process.wait()
            finally:
                finished.set()
                watchdog.join(timeout=21)
            if timed_out.is_set():
                raise RuntimeError("Шаг прерван пользователем или превысил время ожидания. Его можно повторить.")
            if returncode != 0:
                raise RuntimeError(f"Команда завершилась с кодом {returncode}. Подробности — в журнале выше.")

    def run(self, *, repair: bool = False, application_only: bool = False) -> None:
        step = "python"
        self.repair = repair or self.state.repair
        self.application_only = application_only or (self.state.update_only and not repair)
        was_ready = self.state.ready
        try:
            self.state.settings.validate(self.source, self.home)
            self.home.mkdir(parents=True, exist_ok=True)
            if not self.state.log:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                self.state.log = str(self.home / "logs" / f"setup-{stamp}.log")
            log_path = Path(self.state.log)
            if not log_path.resolve().is_relative_to((self.home / "logs").resolve()):
                raise ValueError("Некорректный путь к журналу настройки.")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as stream:
                self._stream = stream
                self.state.repair = self.repair
                self.state.update_only = self.application_only
                if self.remote:
                    from .repository import latest_source
                    step = "source"
                    self.emit({"kind": "stage", "index": 0, "total": len(STEPS),
                               "message": "Загрузка актуальных исходников с GitHub (main)"})
                    self.source, revision = latest_source(self.home, self._download_event, self.cancel)
                    self.state.settings.validate(self.source, self.home)
                    receipt = self.home / "source.json"
                    try:
                        recorded = json.loads(receipt.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        recorded = {}
                    previous = recorded.get("revision") if isinstance(recorded, dict) else None
                    if revision != previous:
                        self.state.completed = [s for s in self.state.completed
                                                if s not in {"dependencies", "diagnostics", "entrypoint"}]
                    atomic_text(receipt, json.dumps({"revision": revision, "version": source_version(self.source),
                                                     "path": str(self.source)}))
                    self.log(f"GitHub main: {revision}; версия {source_version(self.source)}")
                    self.emit({"kind": "source", "revision": revision, "version": source_version(self.source)})
                self.state.ready = False
                version = source_version(self.source)
                if application_only or (version and self.state.application_version != version):
                    self.state.completed = [s for s in self.state.completed
                                            if s not in {"dependencies", "diagnostics", "entrypoint"}]
                if repair:
                    self.state.completed.clear()
                self.state.save(self.state_path)
                self.log("\nНастройка Diarrhizer: " + datetime.now().isoformat(timespec="seconds"))
                self.log(f"Источник приложения: {self.source}; версия: {version or 'не указана'}")
                for index, (step, title) in enumerate(STEPS):
                    if self.application_only and step not in {"dependencies", "diagnostics", "entrypoint"}:
                        continue
                    if self.application_only and step == "dependencies":
                        title = "Обновление приложения без переустановки библиотек"
                    if self.cancel.is_set():
                        raise RuntimeError("Настройка остановлена. Её можно продолжить.")
                    self.emit({"kind": "stage", "index": index, "total": len(STEPS), "message": title})
                    if step in self.state.completed:
                        self.log(title + ": уже выполнено")
                        continue
                    self.log("\n" + title)
                    getattr(self, "_" + step)()
                    self.state.completed.append(step)
                    self.state.failed_step = self.state.error = ""
                    self.state.save(self.state_path)
                self.state.ready = True
                self.state.repair = False
                self.state.update_only = False
                from .existing import remember_environment
                remember_environment(self.state_path, self.state)
                self.state.save(self.state_path)
                self.log("Готово. Модели не загружались. HF-токен можно указать в настройках основного GUI.")
                self.emit({"kind": "done"})
        except Exception as exc:  # noqa: BLE001 - worker boundary must persist any failed stage
            self.state.ready = was_ready if step == "source" else False
            self.state.failed_step = step
            self.state.error = str(exc) + ("" if self.cancel.is_set() else " " + STEP_HINTS[step])
            try:
                self.log(f"Ошибка: {exc}")
            except OSError:
                self.emit({"kind": "log", "message": f"Ошибка: {exc}. Запись журнала недоступна."})
            try:
                self.state.save(self.state_path)
            except OSError as save_error:
                self.emit({"kind": "log", "message": f"Не удалось сохранить состояние: {save_error}"})
            self.emit({"kind": "error", "step": step, "message": self.state.error})
        finally:
            self._stream = None

    def _python(self) -> None:
        settings = self.state.settings
        if settings.device == "cuda":
            gpu = system.gpu_description()
            if not gpu:
                raise ValueError("NVIDIA GPU / драйвер не обнаружен. Выберите CPU или установите драйвер NVIDIA "
                                 "и повторите проверку. Доступность CUDA также проверяется после установки Torch.")
            self.log("NVIDIA: " + gpu)
        if not settings.python:
            candidates = system.discover_pythons()
            if candidates:
                settings.python = candidates[0]
            elif settings.install_python:
                target = self.home / "components" / "Python312"
                existing = target / "python.exe"
                if not existing.is_file():
                    installer = system.download(system.PYTHON_URL, self.home / "downloads" / "python-3.12.10.exe",
                                                system.PYTHON_SHA256, self._download_event)
                    self.command([str(installer), "/quiet", "InstallAllUsers=0", "Include_launcher=0",
                                  "Include_test=0", "PrependPath=0", "Shortcuts=0", f"TargetDir={target}"])
                settings.python = str(existing)
            else:
                raise ValueError("Python 3.11/3.12 x64 не найден. Выберите python.exe или разрешите его загрузку.")
        facts = system.probe_python(settings.python)
        self.log("Python " + ".".join(map(str, facts["version"])) + " x64: " + settings.python)

    def _venv(self) -> None:
        settings = self.state.settings
        folder = Path(settings.venv)
        nonempty = folder.exists() and (not folder.is_dir() or any(folder.iterdir()))
        if nonempty:
            if not (folder / "pyvenv.cfg").is_file():
                if not self.state.managed:
                    raise ValueError("Выбранная папка не является venv и не пуста. Выберите отдельную пустую папку.")
            elif not self.state.managed and not self.approve_existing:
                raise ValueError("Существующее окружение требует явного подтверждения перед установкой пакетов.")
            else:
                facts = system.probe_python(str(environment_python(settings.venv)), folder)
                base = system.probe_python(settings.python)
                if facts["version"][:2] != base["version"][:2]:
                    raise ValueError("Версию Python существующей venv менять нельзя. Выберите её версию Python "
                                     "или другую пустую папку окружения.")
        folder.mkdir(parents=True, exist_ok=True)
        self.state.managed = True
        self.state.save(self.state_path)  # Own the empty directory even if venv/ensurepip is interrupted.
        if not (folder / "pyvenv.cfg").is_file():
            self.command([settings.python, "-I", "-m", "venv", str(folder)])
        system.probe_python(str(environment_python(settings.venv)), folder)
        Path(settings.cache).mkdir(parents=True, exist_ok=True)
        Path(settings.output).mkdir(parents=True, exist_ok=True)
        Path(settings.config_dir).mkdir(parents=True, exist_ok=True)

    def _ffmpeg(self) -> None:
        settings = self.state.settings
        executable = settings.ffmpeg or os.environ.get("DIARRHIZER_FFMPEG_PATH") or shutil.which("ffmpeg")
        if not executable:
            if not settings.install_ffmpeg:
                raise ValueError("FFmpeg не найден. Укажите ffmpeg.exe или разрешите его загрузку.")
            executable = system.install_ffmpeg(self.home, self._download_event)
        settings.ffmpeg = str(Path(executable).resolve())
        self.log(system.capture([settings.ffmpeg, "-version"]).splitlines()[0])
        probe = Path(settings.ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        # A matching sibling also works for pyannote's decoders through the prepended PATH.
        if not probe.is_file():
            raise ValueError("Рядом с ffmpeg.exe не найден ffprobe.exe. Выберите полный комплект FFmpeg "
                             "или очистите поле FFmpeg и разрешите загрузку.")
        self.log(system.capture([str(probe), "-version"]).splitlines()[0])

    def _dependencies(self) -> None:
        settings = self.state.settings
        python = str(environment_python(settings.venv))
        system.probe_python(python, Path(settings.venv))
        env = runtime_env(settings)
        env["PIP_REQUIRE_VIRTUALENV"] = "true"
        requirements = self.source / "requirements"
        constraints = str(requirements / "constraints-stable.txt")
        variant = "cuda-cu121.txt" if settings.device == "cuda" else "cpu.txt"
        if self.application_only:
            self.command([python, "-I", "-m", "pip", "install", "--no-deps", "--force-reinstall",
                          "--no-cache-dir", str(self.source)], env=env)
            self._verify_application(python, env)
            return
        self.command([python, "-I", "-m", "ensurepip", "--upgrade"], env=env)
        self.command([python, "-I", "-m", "pip", "install", "--upgrade", "pip", "setuptools<81", "wheel"], env=env)
        reinstall = ["--force-reinstall"] if self.repair else []
        self.command([python, "-I", "-m", "pip", "install", *reinstall, "-c", constraints,
                      "-r", str(requirements / "base.txt"), "-r", str(requirements / variant)], env=env)
        self.command([python, "-I", "-m", "pip", "install", "--no-cache-dir", "-c", constraints,
                      str(self.source) + "[gui]"], env=env)
        self._verify_application(python, env)

    def _verify_application(self, python, env):
        version = source_version(self.source)
        if version:
            code = ("import sys,importlib.metadata as m,diarrhizer; "
                    "print('Installed Diarrhizer:',diarrhizer.__version__,diarrhizer.__file__); "
                    "assert diarrhizer.__version__ == sys.argv[1], 'Runtime version differs from source'; "
                    "assert m.version('diarrhizer') == sys.argv[1], 'Package version differs from source'")
            self.command([python, "-I", "-c", code, version], env=env, timeout=30)
            self.state.application_version = version

    def _diagnostics(self) -> None:
        settings = self.state.settings
        python = str(environment_python(settings.venv))
        system.probe_python(python, Path(settings.venv))
        env = runtime_env(settings)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        self.command([python, "-I", "-m", "pip", "check"], env=env, timeout=120)
        self.command([python, "-I", "-c", DEPENDENCY_CHECK, settings.device], env=env, timeout=180)
        self.command([python, "-I", "-m", "diarrhizer", "doctor", "--profile", "asr"], env=env, timeout=180)

    def _entrypoint(self) -> None:
        from .launch import write_start_files
        write_start_files(self.state_path, self.state)

    def _download_event(self, event) -> None:
        if self.cancel.is_set():
            raise RuntimeError("Загрузка остановлена. Шаг можно повторить.")
        self.emit(event)
# [SEMANTIC-END] LAUNCHER:SETUP
