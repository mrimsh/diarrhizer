"""Shared utility helpers."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: str | Path, content: str) -> None:
    """Write text to a file atomically (write to a temp file, then rename into place).

    Guarantees readers never observe a partially-written file if the process
    is interrupted mid-write (Ctrl+C, crash, power loss) - the target path
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
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
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
    except (OSError, json.JSONDecodeError):
        return None


def write_json_atomic(path: str | Path, data: Any) -> None:
    """Serialize data as JSON and write it to a file atomically.

    Args:
        path: Destination file path
        data: JSON-serializable data
    """
    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False, default=to_jsonable))
