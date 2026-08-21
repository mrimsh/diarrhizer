"""Minimal .env reader/writer for the handful of keys this GUI persists
there (HF_TOKEN, DIARRHIZER_FFMPEG_PATH) - not a general dotenv
implementation. diarrhizer's core CLI never auto-loads .env (no
python-dotenv anywhere in src/diarrhizer, confirmed by search) - apply_env_file()
is additive, GUI-only behavior. An existing process env var always wins over
a .env value, matching common dotenv convention, so a real env var set
outside the GUI is never silently overridden by a stale .env entry.
"""

import os
import re
from pathlib import Path

# src/diarrhizer_gui/env_file.py -> src/diarrhizer_gui -> src -> repo root.
# Works for this project's editable install: diarrhizer_gui.__file__ points
# at the real source tree, not a site-packages copy.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def read_env_file(path: Path) -> dict:
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE_RE.match(stripped)
        if match:
            key, value = match.groups()
            values[key] = value.strip().strip('"').strip("'")
    return values


def write_env_file(path: Path, updates: dict) -> None:
    """Update or append keys from `updates`, preserving every other line
    (including comments and unrelated keys, e.g. DIARRHIZER_DEVICE) as-is.
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen = set()
    output = []
    for line in lines:
        stripped = line.strip()
        match = None
        if stripped and not stripped.startswith("#"):
            match = _LINE_RE.match(stripped)
        if match and match.group(1) in updates:
            key = match.group(1)
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)

    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")

    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def apply_env_file(path: Path) -> None:
    """Load `path` into os.environ without overriding already-set variables."""
    for key, value in read_env_file(path).items():
        os.environ.setdefault(key, value)
