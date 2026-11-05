"""Small local artifact contracts: OS write locks, provenance and recoverable file bundles."""

import hashlib
import os
import platform
import threading
from contextlib import contextmanager
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from diarrhizer.utils import read_json, write_json_atomic, write_text_atomic

SCHEMA_VERSION = 1
LOCK_DIR = ".diarrhizer-locks"
_held = threading.local()


# [SEMANTIC-BEGIN] ARTIFACTS:WRITE_LOCK
# @purpose: Allow one writer per job across CLI/GUI processes without stale PID-lock cleanup
# @description: A nonblocking OS file lock is released on close/crash. Lock files live beside jobs so
#   a job can be deleted under its lock. Nested calls on the same thread reuse the lock; other writers fail.
#   Project jobs also hold their outer project lock (outside the movable tree), making scope renames
#   mutually exclusive with CLI/GUI writes without open Windows lock files inside a renamed directory.
# @inputs: job directory
# @outputs: context manager/decorator for mutating operations
# @sideEffects: Creates a byte-sized lock file in the parent's hidden lock directory
# @errors: RuntimeError when busy; OSError on filesystem failure
# @see: PIPELINE:EXECUTE, SUMMARY:JOB, SUMMARY:CHAT
@contextmanager
def job_write_lock(job_dir):
    root = Path(job_dir).resolve()
    project = next((p for p in root.parents if (p / ".diarrhizer-project.json").is_file()), None)
    if project is not None:
        with job_write_lock(project):
            with _directory_lock(root):
                yield
        return
    with _directory_lock(root):
        yield


@contextmanager
def _directory_lock(root):
    key = os.path.normcase(str(root))
    held = getattr(_held, "paths", None)
    if held is None:
        held = _held.paths = set()
    if key in held:
        yield
        return
    folder = root.parent / LOCK_DIR
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / (hashlib.sha256(key.encode()).hexdigest() + ".lock")
    with lock.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Job is busy in another operation: {root}. Retry after it finishes.") from exc
        held.add(key)
        try:
            yield
        finally:
            held.remove(key)
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def locked_job(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        root = kwargs.get("job_dir", args[0] if args else None)
        if root is not None and Path(root).is_dir():
            with job_write_lock(root):
                return function(*args, **kwargs)
        return function(*args, **kwargs)
    return wrapped
# [SEMANTIC-END] ARTIFACTS:WRITE_LOCK


# [SEMANTIC-BEGIN] ARTIFACTS:BUNDLE
# @purpose: Detect and recover partial multi-file writes; publish a manifest only after all files
# @description: Render first, then atomically journal the full bundle, replace each output, publish
#   its hashes last, remove journal. A retry completes the journal without another model call.
#   Readers reject pending/mismatched bundles. Manifest-less legacy files remain readable.
#   This is recoverable publication, not a filesystem-wide atomic transaction or power-loss guarantee.
# @inputs: output folder, flat relative filenames/text, metadata; caller holds the job write lock
# @outputs: versioned manifest, data files, optionally a pending journal after interruption
# @sideEffects: Reads/writes local artifacts; provenance reads installed distribution versions
# @errors: ValueError for unsafe names/corrupt journal; OSError for disk failures
# @see: STAGE:EXPORT, SUMMARY:JOB, ARTIFACTS:WRITE_LOCK
def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def provenance() -> dict:
    packages = {}
    for name in ("diarrhizer", "whisperx", "torch", "torchaudio", "ctranslate2", "faster-whisper",
                 "pyannote.audio", "transformers", "pytorch-lightning"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return {"python": platform.python_version(), "packages": packages}


def _name(value: str) -> str:
    if (not isinstance(value, str) or not value or value in (".", "..")
            or Path(value).name != value or any(c in value for c in '/\\:') or value.startswith(".")):
        raise ValueError(f"Invalid artifact filename: {value!r}")
    return value


def _journal(root: Path, manifest: str) -> Path:
    return root / ("." + _name(manifest) + ".pending.json")


def _publish(root: Path, record: dict) -> None:
    manifest = _name(record["manifest"])
    files = record["files"]
    if not isinstance(files, dict) or not files or manifest in files:
        raise ValueError("Invalid artifact bundle")
    for name, content in files.items():
        _name(name)
        if not isinstance(content, str):
            raise ValueError("Artifact content must be text")
    for name, content in files.items():
        write_text_atomic(root / name, content)
    write_json_atomic(root / manifest, {
        "schema_version": SCHEMA_VERSION, "files": {name: digest(content) for name, content in files.items()},
        "metadata": record.get("metadata", {}),
    })
    _journal(root, manifest).unlink(missing_ok=True)


def write_bundle(root: Path, files: dict[str, str], manifest: str, metadata: dict | None = None) -> None:
    root = Path(root)
    _name(manifest)
    for name in files:
        _name(name)
    if manifest in files or not files or any(not isinstance(text, str) for text in files.values()):
        raise ValueError("Invalid artifact bundle")
    record = {"schema_version": SCHEMA_VERSION, "manifest": manifest, "files": files, "metadata": metadata or {}}
    write_json_atomic(_journal(root, manifest), record)
    _publish(root, record)


def recover_bundles(root: Path) -> int:
    root = Path(root)
    count = 0
    for pending in root.glob(".*.pending.json"):
        record = read_json(pending)
        if (not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION
                or not isinstance(record.get("manifest"), str) or not isinstance(record.get("files"), dict)
                or _journal(root, record["manifest"]) != pending):
            raise ValueError(f"Cannot recover corrupt artifact journal: {pending}")
        _publish(root, record)
        count += 1
    return count


def bundle_valid(root: Path, manifest: str, *, allow_legacy: bool = True) -> bool:
    root = Path(root)
    if _journal(root, manifest).exists():
        return False
    path = root / _name(manifest)
    if not path.exists():
        return allow_legacy
    record = read_json(path)
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION or not record.get("files"):
        return False
    try:
        return all(hashlib.sha256((root / _name(name)).read_bytes()).hexdigest() == sha
                   for name, sha in record["files"].items())
    except (OSError, ValueError, AttributeError):
        return False


@locked_job
def recover_job(job_dir) -> dict:
    root = Path(job_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Job folder not found: {root}")
    return {name: recover_bundles(root / name) for name in ("export", "summary")}
# [SEMANTIC-END] ARTIFACTS:BUNDLE
