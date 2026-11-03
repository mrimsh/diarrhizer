"""Shared utility helpers."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


# [SEMANTIC-BEGIN] UTILS:ATOMIC_IO
# @purpose: Keep single-file writes complete and readable across process interruptions
# @description: Write a sibling temporary file, flush/fsync it, then replace the destination. This
#   does not make several files transactional or guarantee recovery after hardware/power failure.
#   Preserve the supplied line endings so bundle hashes describe exact UTF-8 bytes on Windows too.
# @inputs: text/JSON data and local paths
# @outputs: complete files; read_json returns None for missing/corrupt files
# @sideEffects: Reads/writes files; removes its temporary file on failure
# @errors: OSError for write failures; TypeError for unsupported JSON values
# @see: ARTIFACTS:BUNDLE, CONFIG:ENV
def write_text_atomic(path: str | Path, content: str) -> None:
    """Write text to a file atomically (write to a temp file, then rename into place).

    Guarantees readers never observe a partially-written file if the process
    is interrupted mid-write (Ctrl+C, process crash) - the target path
    either has the old content or the fully-written new content, never a
    truncated mix of both.

    Args:
        path: Destination file path
        content: Text content to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays (and similar) so json.dumps can serialize them."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_json(path: str | Path) -> Any | None:
    """Read a JSON file, or None if it is missing or corrupt."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None


def write_json_atomic(path: str | Path, data: Any) -> None:
    """Serialize data as JSON and write it to a file atomically.

    Args:
        path: Destination file path
        data: JSON-serializable data
    """
    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False, default=to_jsonable))
# [SEMANTIC-END] UTILS:ATOMIC_IO
