"""Minimal .env reader/writer shared by the CLI and GUI.

Not a general dotenv implementation. Existing process env vars always win
over a .env value, matching common dotenv convention.
"""

import os
import re
from pathlib import Path

# src/diarrhizer/env_file.py -> src/diarrhizer -> src -> repo root.
# Works for this project's editable install: diarrhizer.__file__ points
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


# [SEMANTIC-BEGIN] CONFIG:ENV
# @purpose: Load project .env files into the process environment
# @description: Precedence is process env > cwd .env > repo-root .env. Used by
#   both the CLI (doctor/run) and the GUI so HF_TOKEN / DIARRHIZER_FFMPEG_PATH
#   from a local .env are visible without a manual export.
# @inputs: optional extra paths; defaults to repo-root .env then cwd .env
# @outputs: None (mutates os.environ via setdefault)
# @sideEffects: Reads .env files, sets missing os.environ keys
# @errors: None (missing files are ignored)
# @see: CLI:ENTRY, DIAGNOSTICS:DOCTOR
def load_project_env() -> None:
    """Load repo-root .env, then cwd .env, without overriding existing vars."""
    values: dict[str, str] = {}
    repo_env = REPO_ROOT / ".env"
    cwd_env = Path.cwd() / ".env"
    if repo_env.exists():
        values.update(read_env_file(repo_env))
    if cwd_env.exists() and cwd_env.resolve() != repo_env.resolve():
        values.update(read_env_file(cwd_env))
    for key, value in values.items():
        os.environ.setdefault(key, value)
# [SEMANTIC-END] CONFIG:ENV
