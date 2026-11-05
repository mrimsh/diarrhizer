"""Windows probes and verified component downloads (standard library only)."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import atomic_text

PYTHON_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
# SHA-256 from the PSF Sigstore bundle accompanying this exact installer.
PYTHON_SHA256 = "67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# [SEMANTIC-BEGIN] LAUNCHER:PROBES
# @purpose: Find compatible 64-bit CPython and GPU without importing ML into the launcher
# @description: Inspect real interpreters, including py/PEP 514 registrations; skip Store aliases.
#   A virtual environment must identify its own prefix before it can receive any pip command.
#   Remove host/frozen Python variables from hidden subprocess environments.
# @inputs: executable paths, Windows registry and PATH
# @outputs: interpreter facts, compatible candidates, NVIDIA driver description
# @sideEffects: Short hidden subprocesses and read-only registry access
# @errors: ValueError for unsupported/broken interpreters; subprocess errors for failed probes
# @see: LAUNCHER:SETUP
def child_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "TCL_LIBRARY", "TK_LIBRARY"):
        env.pop(key, None)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PIP_DISABLE_PIP_VERSION_CHECK="1")
    return env


def capture(args: list[str], *, timeout: int = 15) -> str:
    return subprocess.check_output(args, text=True, encoding="utf-8", errors="replace",
                                   stderr=subprocess.STDOUT, timeout=timeout,
                                   env=child_env(), creationflags=CREATE_NO_WINDOW).strip()


def probe_python(executable: str, expected_venv: Path | None = None) -> dict:
    script = ("import json,sys,struct,platform; print(json.dumps(dict("
              "version=list(sys.version_info[:3]),bits=struct.calcsize('P')*8,"
              "implementation=platform.python_implementation(),prefix=sys.prefix,"
              "base=sys.base_prefix,executable=sys.executable)))")
    try:
        facts = json.loads(capture([executable, "-I", "-c", script]))
        if (facts["version"][:2] not in ([3, 11], [3, 12]) or facts["bits"] != 64
                or facts["implementation"] != "CPython"):
            raise ValueError("Нужен CPython 3.11 или 3.12, 64 бит; Python 3.13+ не поддерживается.")
        if expected_venv is not None and (Path(facts["prefix"]).resolve() != expected_venv.resolve()
                                         or facts["prefix"] == facts["base"]):
            raise ValueError("Python не принадлежит выбранной .venv. Установка пакетов остановлена.")
        return facts
    except (OSError, subprocess.SubprocessError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось запустить Python: {executable}. "
                         "Выберите рабочий Python 3.11/3.12 или разрешите его загрузку.") from exc


def discover_pythons() -> list[str]:
    candidates = []
    if not getattr(sys, "frozen", False):
        candidates.append(getattr(sys, "_base_executable", sys.executable))
    for name in ("python", "python3"):
        executable = shutil.which(name)
        if executable and "windowsapps" not in executable.lower():
            candidates.append(executable)
    try:
        output = capture(["py", "-0p"], timeout=10)
        candidates.extend(re.findall(r"([A-Za-z]:\\[^\r\n]*?python(?:w)?\.exe)\s*$", output,
                                     re.MULTILINE | re.IGNORECASE))
    except (OSError, subprocess.SubprocessError):
        pass
    if os.name == "nt":
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hive, r"Software\Python") as vendors:
                    for i in range(winreg.QueryInfoKey(vendors)[0]):
                        with winreg.OpenKey(vendors, winreg.EnumKey(vendors, i)) as versions:
                            for j in range(winreg.QueryInfoKey(versions)[0]):
                                try:
                                    with winreg.OpenKey(versions, winreg.EnumKey(versions, j) + r"\InstallPath") as key:
                                        candidates.append(str(winreg.QueryValueEx(key, "ExecutablePath")[0]))
                                except OSError:
                                    pass
            except OSError:
                pass
    found = []
    for executable in dict.fromkeys(candidates):
        try:
            facts = probe_python(executable)
            actual = facts["executable"]
            if actual not in found:
                found.append(actual)
        except ValueError:
            pass
    return sorted(found, key=lambda p: ("312" not in p.replace(".", ""), p.lower()))


def gpu_description() -> str:
    executable = shutil.which("nvidia-smi")
    if not executable:
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        executable = str(candidate) if candidate.is_file() else "nvidia-smi"
    try:
        text = capture([executable, "--query-gpu=name,driver_version", "--format=csv,noheader"], timeout=10)
        return text if text else ""
    except (OSError, subprocess.SubprocessError):
        return ""
# [SEMANTIC-END] LAUNCHER:PROBES


# [SEMANTIC-BEGIN] LAUNCHER:DOWNLOAD
# @purpose: Download Python/FFmpeg with checksum verification and byte progress
# @description: Partial files never become executable downloads; FFmpeg extraction copies only two
#   named binaries, so archive paths cannot escape the component directory.
# @inputs: HTTPS URL, expected SHA-256, component folder, event callback
# @outputs: verified cached installer or FFmpeg executables
# @sideEffects: Network access and local component/cache files
# @errors: ValueError for checksum/archive errors; network/OSError failures leave a retryable step
# @see: LAUNCHER:SETUP
def download(url: str, target: Path, expected_sha256: str, emit) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("Источник не предоставил корректную контрольную сумму SHA-256.")
    if target.is_file() and _sha256(target) == expected_sha256.lower():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Diarrhizer-Launcher/1"})
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as stream:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while chunk := response.read(512 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                emit({"kind": "download", "received": received, "total": total})
        if digest.hexdigest() != expected_sha256.lower():
            raise ValueError("Контрольная сумма загрузки не совпала. Повторите шаг: файл будет скачан заново.")
        os.replace(partial, target)
        return target
    finally:
        partial.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_ffmpeg(home: Path, emit) -> str:
    with urllib.request.urlopen(FFMPEG_URL + ".sha256", timeout=30) as response:
        checksum = response.read(4096).decode("ascii").split()[0]
    archive = download(FFMPEG_URL, home / "downloads" / "ffmpeg.zip", checksum, emit)
    folder = home / "components" / "ffmpeg" / checksum[:16] / "bin"
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for name in ("ffmpeg.exe", "ffprobe.exe"):
            entries = [p for p in bundle.namelist() if p.endswith("/bin/" + name)]
            if len(entries) != 1:
                raise ValueError(f"Архив FFmpeg не содержит однозначный bin/{name}.")
            target = folder / name
            temporary = target.with_suffix(".tmp")
            with bundle.open(entries[0]) as source, temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            os.replace(temporary, target)
    atomic_text(folder.parent / "source.txt", FFMPEG_URL + "\nSHA256: " + checksum + "\n")
    return str(folder / "ffmpeg.exe")
# [SEMANTIC-END] LAUNCHER:DOWNLOAD
