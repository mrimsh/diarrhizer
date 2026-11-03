"""Reading the JSON files that hold user templates and providers.

Both files have the same outline - {"default": "<name>", "<section>": {"<name>": {...}}} -
and the same rules: a missing file is an empty store, a malformed one is a
ValueError that names the file (a hand-edited typo must not silently turn into
"no templates").
"""

import json
import os
from pathlib import Path
from typing import Any

from diarrhizer.env_file import config_dir


# [SEMANTIC-BEGIN] SUMMARY:STORE
# @purpose: Resolve and validate local template/provider stores without silently dropping corrupt data
# @inputs: per-file environment override, store filename and expected section
# @outputs: config path or validated default name and entries
# @sideEffects: Reads JSON; never writes secrets or repairs malformed input
# @errors: ValueError for unreadable/invalid stores
# @see: CONFIG:DIRECTORY
def resolve_path(env_var: str, filename: str) -> Path:
    """The env var's path if set, else <config directory>/<filename>."""
    override = os.environ.get(env_var)
    return Path(override) if override else config_dir() / filename


def read_entries(path: Path, section: str, label: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """(default name or "", {name: entry object}) from a store file.

    Raises:
        ValueError: unreadable file, wrong outline, or an entry that is not an object
    """
    if not path.exists():
        return "", {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"Cannot read {label} file {path}: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get(section, {}), dict):
        raise ValueError(f"{label.capitalize()} file {path} must be an object with a '{section}' object")
    entries = data.get(section, {})
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: '{name}' must be an object")
    return str(data.get("default") or ""), entries
# [SEMANTIC-END] SUMMARY:STORE
