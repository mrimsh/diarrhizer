"""Minimal .env reader/writer shared by the CLI and GUI.

Not a general dotenv implementation. Existing process env vars always win
over a .env value, matching common dotenv convention.
"""

import os
import re
from pathlib import Path
from diarrhizer.utils import write_text_atomic

# src/diarrhizer/env_file.py -> src/diarrhizer -> src -> repo root.
# Works for this project's editable install: diarrhizer.__file__ points
# at the real source tree, not a site-packages copy.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SOURCE_CHECKOUT = (REPO_ROOT / "pyproject.toml").is_file() and (REPO_ROOT / "src" / "diarrhizer").is_dir()


# [SEMANTIC-BEGIN] CONFIG:DIRECTORY
# @purpose: Keep editable configuration in its checkout and installed-wheel data in the user profile
# @description: DIARRHIZER_CONFIG_DIR overrides the root; per-file overrides remain authoritative.
#   Existing checkout files stay in place; installing a wheel never copies secrets automatically.
# @inputs: environment and package layout
# @outputs: config directory, without creating it
# @sideEffects: None
# @see: CONFIG:ENV, SUMMARY:STORE
def config_dir() -> Path:
    override = os.environ.get("DIARRHIZER_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if _SOURCE_CHECKOUT:
        return REPO_ROOT
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) / "Diarrhizer" if local else Path.home() / ".config" / "diarrhizer"
# [SEMANTIC-END] CONFIG:DIRECTORY

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


# [SEMANTIC-BEGIN] CONFIG:ENV_WRITE
# @purpose: Update local secrets without exposing a truncated .env on process interruption
# @inputs: local .env path and key/value updates
# @outputs: preserved unrelated lines and updated values
# @sideEffects: Atomically replaces .env; no secret values are logged
# @errors: OSError for unreadable/unwritable files
# @see: CONFIG:ENV, UTILS:ATOMIC_IO
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

    write_text_atomic(path, "\n".join(output) + "\n")
# [SEMANTIC-END] CONFIG:ENV_WRITE


def apply_env_file(path: Path) -> None:
    """Load `path` into os.environ without overriding already-set variables."""
    for key, value in read_env_file(path).items():
        os.environ.setdefault(key, value)


# [SEMANTIC-BEGIN] CONFIG:ENV
# @purpose: Load local .env files into the process environment
# @description: Precedence is process env > cwd .env > config directory .env. Used by
#   both the CLI (doctor/run) and the GUI so HF_TOKEN / DIARRHIZER_FFMPEG_PATH
#   from a local .env are visible without a manual export.
# @inputs: CONFIG:DIRECTORY and cwd .env files
# @outputs: None (mutates os.environ via setdefault)
# @sideEffects: Reads .env files, sets missing os.environ keys
# @errors: None (missing files are ignored)
# @see: CLI:ENTRY, DIAGNOSTICS:DOCTOR
def load_project_env() -> None:
    """Load config .env, then cwd .env, without overriding existing vars."""
    values: dict[str, str] = {}
    repo_env = config_dir() / ".env"
    cwd_env = Path.cwd() / ".env"
    if repo_env.exists():
        values.update(read_env_file(repo_env))
    if cwd_env.exists() and cwd_env.resolve() != repo_env.resolve():
        values.update(read_env_file(cwd_env))
    for key, value in values.items():
        os.environ.setdefault(key, value)
# [SEMANTIC-END] CONFIG:ENV
