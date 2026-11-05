"""One local launcher configuration, independent of Diarrhizer and Qt."""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


# [SEMANTIC-BEGIN] LAUNCHER:CONFIG
# @purpose: Persist one setup plan and restartable checkpoints without credentials
# @description: Atomic JSON replacement preserves the last complete state on interruption.
#   A failed/unfinished install is never ready; managed is set only after directory approval.
# @inputs: absolute environment, interpreter, cache, results and FFmpeg paths
# @outputs: settings.json with progress and the durable log location
# @sideEffects: Local filesystem reads/writes; never reads or copies .env
# @errors: ValueError for unsupported/corrupt state; OSError for inaccessible storage
# @see: LAUNCHER:SETUP, LAUNCHER:START
def application_source() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "application"
    return Path(__file__).resolve().parents[2]


def launcher_home() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".local" / "share"
    return base / "Diarrhizer" / "launcher"


@dataclass
class Settings:
    venv: str
    cache: str
    output: str
    config_dir: str
    python: str = ""
    device: str = "cpu"
    ffmpeg: str = ""
    install_python: bool = True
    install_ffmpeg: bool = True

    @classmethod
    def defaults(cls, source: Path, home: Path) -> "Settings":
        installed = getattr(sys, "frozen", False)
        data = home.parent if installed else source
        return cls(str(data / ".venv"), str(home.parent / "models"),
                   str(data / "out"), str(home.parent if installed else source))

    def validate(self, source: Path, home: Path) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Выберите CPU или CUDA.")
        for name in ("venv", "cache", "output", "config_dir"):
            value = getattr(self, name)
            if not value or not Path(value).is_absolute():
                raise ValueError("Укажите абсолютные пути к окружению, кэшу и результатам.")
            if any(c in value for c in '\r\n\0"'):
                raise ValueError("Путь содержит недопустимые символы.")
        venv = Path(self.venv).resolve()
        for protected in (source, home, Path(self.cache), Path(self.output), Path(self.config_dir)):
            if protected.resolve().is_relative_to(venv):
                raise ValueError("Окружение должно находиться в отдельной папке: "
                                 "вне исходников, настроек, кэша и результатов.")


@dataclass
class State:
    settings: Settings
    completed: list[str] = field(default_factory=list)
    managed: bool = False
    ready: bool = False
    repair: bool = False
    failed_step: str = ""
    error: str = ""
    log: str = ""

    @classmethod
    def load(cls, path: Path) -> "State | None":
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise TypeError("invalid state object")
            if data.pop("schema") != 1:
                raise ValueError("unsupported schema")
            data["settings"] = Settings(**data["settings"])
            state = cls(**data)
            if (not isinstance(state.completed, list)
                    or not all(isinstance(item, str) for item in state.completed)
                    or any(type(getattr(state, key)) is not bool for key in ("managed", "ready", "repair"))
                    or any(not isinstance(getattr(state, key), str)
                           for key in ("failed_step", "error", "log"))):
                raise ValueError("invalid state types")
            if any(not isinstance(value, str) for key, value in asdict(state.settings).items()
                   if key not in {"install_python", "install_ffmpeg"}):
                raise ValueError("invalid settings types")
            if any(type(getattr(state.settings, key)) is not bool
                   for key in ("install_python", "install_ffmpeg")):
                raise ValueError("invalid download options")
            return state
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Не удалось прочитать {path}. Сохраните этот файл и восстановите "
                             "его из копии либо переименуйте для новой настройки.") from exc

    def save(self, path: Path) -> None:
        atomic_text(path, json.dumps({"schema": 1, **asdict(self)}, ensure_ascii=False, indent=2))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class LauncherLock:
    """Hold an OS lock, automatically released even when the launcher crashes."""

    def __init__(self, home: Path):
        home.mkdir(parents=True, exist_ok=True)
        self.stream = (home / "launcher.lock").open("a+b")
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("Лаунчер уже открыт. Перейдите в существующее окно.") from None

    def close(self) -> None:
        self.stream.close()
# [SEMANTIC-END] LAUNCHER:CONFIG
