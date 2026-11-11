"""Download an immutable source snapshot of the public repository's main branch."""

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .config import atomic_text, source_version
from .system import open_https

REPOSITORY = "mrimsh/diarrhizer"
BRANCH = "main"
SOURCE_LABEL = f"GitHub · {REPOSITORY} · {BRANCH}"


def _open(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Diarrhizer-Launcher",
                                                  "Accept": "application/vnd.github+json"})
    return open_https(request)


# [SEMANTIC-BEGIN] LAUNCHER:REPOSITORY
# @purpose: Install current repository code independently of the launcher's compiled source snapshot
# @description: Resolve main to one immutable SHA, download its archive over HTTPS and cache the validated
#   source atomically. Extract only package source/requirements/metadata; reject traversal and symlinks.
#   Network failures never silently fall back to the bundled version or modify an existing environment.
# @inputs: local launcher home, progress callback and cancellation event
# @outputs: cached application-source directory and GitHub commit SHA
# @sideEffects: Public GitHub API/codeload requests and local source-cache writes; no Git or tokens needed
# @errors: Network errors, invalid commit/archive/version, cancellation, download/expansion size limits
# @see: LAUNCHER:SETUP, LAUNCHER:CONFIG
def latest_source(home: Path, emit, cancel) -> tuple[Path, str]:
    with _open(f"https://api.github.com/repos/{REPOSITORY}/commits/{BRANCH}") as response:
        metadata = json.loads(response.read(1024 * 1024))
    revision = metadata.get("sha", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("GitHub не вернул корректный идентификатор коммита.")
    if cancel.is_set():
        raise RuntimeError("Загрузка исходников остановлена.")
    cache = home / "sources"
    target = cache / revision
    if (target / ".download.json").is_file() and source_version(target):
        return target, revision
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="download-", dir=cache) as temporary:
        folder = Path(temporary)
        archive = folder / "source.zip"
        digest = hashlib.sha256()
        with _open(f"https://codeload.github.com/{REPOSITORY}/zip/{revision}") as response, archive.open("wb") as stream:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while chunk := response.read(512 * 1024):
                if cancel.is_set():
                    raise RuntimeError("Загрузка исходников остановлена.")
                received += len(chunk)
                if received > 256 * 1024 * 1024:
                    raise ValueError("Архив репозитория превышает допустимый размер 256 МБ.")
                digest.update(chunk)
                stream.write(chunk)
                emit({"kind": "download", "received": received, "total": total})
        source = folder / "application"
        source.mkdir()
        expanded = 0
        with zipfile.ZipFile(archive) as bundle:
            if len(bundle.infolist()) > 20000:
                raise ValueError("Слишком много файлов в архиве исходников.")
            roots = set()
            for item in bundle.infolist():
                parts = PurePosixPath(item.filename).parts
                if (not parts or ".." in parts or "\\" in item.filename or ":" in item.filename
                        or item.filename.startswith("/") or (item.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError("Небезопасный путь в архиве исходников.")
                roots.add(parts[0])
                relative = PurePosixPath(*parts[1:])
                allowed = (str(relative) in {"pyproject.toml", "README.md", "LICENSE"}
                           or (relative.parts and relative.parts[0] == "src" and relative.suffix == ".py")
                           or (relative.parts and relative.parts[0] == "requirements" and relative.suffix == ".txt"))
                if not allowed or item.is_dir():
                    continue
                expanded += item.file_size
                if expanded > 64 * 1024 * 1024:
                    raise ValueError("Исходники превышают допустимый размер 64 МБ.")
                destination = source.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            if len(roots) != 1:
                raise ValueError("Некорректная структура архива исходников.")
        for required in ("pyproject.toml", "src/diarrhizer_gui/app.py", "requirements/constraints-stable.txt",
                         "requirements/base.txt", "requirements/cpu.txt", "requirements/cuda-cu121.txt"):
            if not (source / required).is_file():
                raise ValueError(f"В архиве отсутствует {required}.")
        version = source_version(source)
        if not version:
            raise ValueError("В архиве не указана версия приложения.")
        atomic_text(source / ".download.json", json.dumps({"revision": revision, "version": version,
                                                           "archive_sha256": digest.hexdigest()}))
        if target.exists():
            raise ValueError("Кэш исходников повреждён. Переименуйте папку " + str(target) + " и повторите загрузку.")
        os.replace(source, target)
    return target, revision
# [SEMANTIC-END] LAUNCHER:REPOSITORY
