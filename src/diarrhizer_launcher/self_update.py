"""Verified launcher releases and a separate Windows replacement helper."""

import base64
import ctypes
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import atomic_text
from .repository import REPOSITORY, _open
from .system import CREATE_NO_WINDOW, child_env, download


def _version(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+", value):
        raise ValueError("Релиз должен иметь версию MAJOR.MINOR.")
    return tuple(map(int, value.split(".")))


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# [SEMANTIC-BEGIN] LAUNCHER:SELF_UPDATE
# @purpose: Update the standalone launcher from a newer published GitHub release without touching its venv
# @description: Require a stable numeric release, the exact repository asset and GitHub SHA-256 digest.
#   Verify size, x64 PE and Windows file/product versions before staging beside the current executable.
#   The helper waits for shutdown, atomically replaces with backup, and requires a startup acknowledgement;
#   failed restarts roll back to the previous executable. Existing settings and application packages stay intact.
# @inputs: current launcher version/executable, state path and download events
# @outputs: verified update plan; hidden helper process, backup and durable update log
# @sideEffects: Public HTTPS requests, local staged exe/plan/log files and explicitly confirmed process restart
# @errors: Untrusted/missing digest, malformed version/PE, permissions, network/cancellation or helper failures
# @see: LAUNCHER:UI, LAUNCHER:DOWNLOAD, LAUNCHER:CONFIG
@dataclass(frozen=True)
class Release:
    version: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class PreparedUpdate:
    plan: Path
    staged: Path
    version: str


def latest_launcher(current_version: str) -> Release | None:
    with _open(f"https://api.github.com/repos/{REPOSITORY}/releases/latest") as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("Ответ GitHub слишком большой.")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("draft") is not False or data.get("prerelease") is not False:
        raise ValueError("GitHub не вернул опубликованный стабильный релиз.")
    tag = data.get("tag_name", "")
    version = tag.removeprefix("v") if isinstance(tag, str) else ""
    if _version(version) <= _version(current_version):
        return None
    assets = data.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("Некорректный список файлов релиза.")  # noqa: TRY004 - malformed API response
    matches = [a for a in assets if isinstance(a, dict) and a.get("name") == "Diarrhizer.exe"]
    if len(matches) != 1:
        raise ValueError("В новом релизе нет однозначного файла Diarrhizer.exe.")
    asset = matches[0]
    digest, url, size = asset.get("digest", ""), asset.get("browser_download_url", ""), asset.get("size")
    expected_url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/Diarrhizer.exe"
    if (not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or url != expected_url or type(size) is not int or not 0 < size <= 256 * 1024 * 1024
            or asset.get("state") != "uploaded"):
        raise ValueError("Файл релиза не прошёл проверку URL, размера или контрольной суммы SHA-256.")
    return Release(version, url, digest.removeprefix("sha256:"), size)


def executable_version(path: Path) -> str:
    """Read Windows resources without executing the downloaded program."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise ValueError("Загруженный файл не является Windows exe.")
        offset = struct.unpack_from("<I", header, 60)[0]
        stream.seek(offset)
        if stream.read(6) != b"PE\0\0\x64\x86":
            raise ValueError("Требуется Windows x64 exe.")
    api = ctypes.WinDLL("version", use_last_error=True)
    api.GetFileVersionInfoSizeW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p]
    api.GetFileVersionInfoW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    api.VerQueryValueW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p),
                                 ctypes.POINTER(ctypes.c_uint)]
    size = api.GetFileVersionInfoSizeW(str(path), None)
    buffer = ctypes.create_string_buffer(size)
    pointer, length = ctypes.c_void_p(), ctypes.c_uint()
    if (not size or not api.GetFileVersionInfoW(str(path), 0, size, buffer)
            or not api.VerQueryValueW(buffer, "\\", ctypes.byref(pointer), ctypes.byref(length))
            or length.value < 52):
        raise ValueError("У exe отсутствуют корректные метаданные версии.")
    fields = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
    if fields[0] != 0xFEEF04BD or fields[2] != fields[4] or fields[3] or fields[5]:
        raise ValueError("Версии файла и продукта exe не согласованы.")
    return f"{fields[4] >> 16}.{fields[4] & 65535}"


def prepare_update(release: Release, target: Path, state_path: Path, emit, cancel) -> PreparedUpdate:
    target, state_path = target.resolve(), state_path.resolve()
    token = uuid.uuid4().hex
    folder = state_path.parent / "launcher-updates" / token
    folder.mkdir(parents=True, exist_ok=False)
    log = state_path.parent / "logs" / "launcher-update.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f".{target.stem}.update-{token}.exe")

    def progress(event):
        if cancel.is_set():
            raise RuntimeError("Обновление лаунчера остановлено.")
        if event.get("received", 0) > release.size:
            raise ValueError("Размер загрузки превышает размер файла релиза.")
        emit(event)

    try:
        original = _hash(target)
        candidate = download(release.url, folder / "Diarrhizer.exe", release.sha256, progress)
        if cancel.is_set():
            raise RuntimeError("Обновление лаунчера остановлено.")
        if candidate.stat().st_size != release.size or executable_version(candidate) != release.version:
            raise ValueError("Размер или версия exe отличаются от опубликованного релиза.")
        # Exclusive create both checks destination permissions and avoids clobbering existing files.
        with candidate.open("rb") as source, staged.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        if _hash(staged) != release.sha256:
            raise ValueError("Контрольная сумма подготовленного exe не совпала.")
        plan = folder / "plan.json"
        atomic_text(plan, json.dumps({
            **asdict(release), "target": str(target), "staged": str(staged), "original_sha256": original,
            "backup": str(target.with_name(f".{target.name}.{token}.previous")),
            "state": str(state_path), "ready": str(folder / "helper.ready"), "ack": str(folder / "startup.ack"),
            "log": str(log), "parent_pid": os.getpid(), "result": str(folder / "result.json"),
            "notice": str(folder.parent / "last-result.json"),
        }, ensure_ascii=False))
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"Prepared launcher {release.version}; SHA256 {release.sha256}; target {target}\n")
        return PreparedUpdate(plan, staged, release.version)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def discard_update(prepared: PreparedUpdate) -> None:
    try:
        prepared.staged.unlink(missing_ok=True)
    except OSError:
        pass  # A scanner may temporarily hold this unused candidate; the running exe is unaffected.


def start_replacement(prepared: PreparedUpdate) -> subprocess.Popen:
    plan = json.loads(prepared.plan.read_text(encoding="utf-8"))
    if _hash(prepared.staged) != plan["sha256"] or _hash(plan["target"]) != plan["original_sha256"]:
        raise ValueError("Файл лаунчера изменился после проверки. Повторите обновление.")
    script = "$planPath='" + str(prepared.plan).replace("'", "''") + "';\n" + HELPER_SCRIPT
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    with prepared.plan.with_name("helper-console.log").open("ab") as log:
        process = subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                   env=child_env(), creationflags=CREATE_NO_WINDOW,
                                   stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if Path(plan["ready"]).is_file():
            return process
        if process.poll() is not None:
            break
        time.sleep(0.05)
    # No readiness means the helper has not been allowed to replace anything.
    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError("Не удалось подготовить замену. Текущий лаунчер сохранён; см. logs/launcher-update.log.")


# Kept in the module so the frozen launcher needs no external script/runtime or venv to update itself.
HELPER_SCRIPT = r"""
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$utf8=[Text.UTF8Encoding]::new($false)
$plan=Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$replaced=$false
$parentExited=$false
$child=$null
function Log($message) { [IO.File]::AppendAllText($plan.log, [DateTime]::Now.ToString('s')+' '+$message+[Environment]::NewLine, $utf8) }
function Hash($path) {
    $stream=[IO.File]::OpenRead($path)
    $algorithm=[Security.Cryptography.SHA256]::Create()
    try { [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function Replace-WithRetry($source,$target,$backup) {
    for ($attempt=0; $attempt -lt 120; $attempt++) {
        try { [IO.File]::Replace($source,$target,$backup); return } catch [IO.IOException] { Start-Sleep -Milliseconds 250 }
    }
    throw 'Executable is locked; replacement timed out'
}
function Start-Launcher($ack) {
    $arguments='--state "'+$plan.state+'"'
    if ($ack) { $arguments+=' --update-ack "'+$plan.ack+'"' }
    Start-Process -FilePath $plan.target -ArgumentList $arguments -WorkingDirectory ([IO.Path]::GetDirectoryName($plan.target)) -WindowStyle Normal -PassThru
}
try {
    $parent=[Diagnostics.Process]::GetProcessById([int]$plan.parent_pid)
    if ((Hash $plan.staged) -ne $plan.sha256 -or (Hash $plan.target) -ne $plan.original_sha256) { throw 'Executable hash changed' }
    [IO.File]::WriteAllText($plan.ready,'ready',$utf8)
    if (-not $parent.WaitForExit(120000)) { throw 'Launcher did not exit; original executable retained' }
    $parentExited=$true
    if ((Hash $plan.target) -ne $plan.original_sha256) { throw 'Original executable changed' }
    Replace-WithRetry $plan.staged $plan.target $plan.backup
    $replaced=$true
    Log ('Installed launcher '+$plan.version)
    $child=Start-Launcher $true
    $deadline=[DateTime]::UtcNow.AddSeconds(30)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ([IO.File]::Exists($plan.ack)) {
            [IO.File]::WriteAllText($plan.result,'{"status":"updated"}',$utf8)
            try { [IO.File]::WriteAllText($plan.notice,'{"status":"updated"}',$utf8) } catch { Log 'Could not save update notification' }
            Log 'New launcher acknowledged startup; backup retained'
            exit 0
        }
        if ($child.HasExited) { break }
        Start-Sleep -Milliseconds 100
    }
    throw 'New launcher failed to acknowledge startup'
} catch {
    $failure=$_.Exception.Message
    Log $failure
    try { [IO.File]::WriteAllText($plan.notice,'{"status":"failed"}',$utf8) } catch { Log 'Could not save failure notification' }
    if ($replaced) {
        try {
            if ($child -and -not $child.HasExited) {
                & taskkill.exe /PID $child.Id /T /F 2>&1 | Out-Null
                $child.WaitForExit(10000) | Out-Null
            }
            Replace-WithRetry $plan.backup $plan.target $plan.staged
            Log 'Previous launcher restored'
            Start-Launcher $false | Out-Null
        } catch { Log ('Rollback/restart failed: '+$_.Exception.Message) }
    } elseif ($parentExited) {
        try { Start-Launcher $false | Out-Null } catch { Log ('Restart failed: '+$_.Exception.Message) }
    }
    $result=@{status='failed';message=$failure} | ConvertTo-Json -Compress
    [IO.File]::WriteAllText($plan.result,$result,$utf8)
    exit 1
}
"""
# [SEMANTIC-END] LAUNCHER:SELF_UPDATE
