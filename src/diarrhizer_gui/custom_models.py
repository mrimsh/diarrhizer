"""Persistence for user-added ASR model repo ids, so a custom model warmed up
once on the Models screen keeps showing up next to the built-in presets
(tiny…large-v3) everywhere a model is chosen - the Models table, New Job and
Settings - instead of being forgotten the moment the warm-up finishes.

Stored as one newline-joined string, not as a Python list, on purpose:
QSettings' native Windows-registry backend does not round-trip a list
faithfully (an empty list reads back as None, a one-element list reads back as
a bare str), so storing a list would force the same defensive unwrapping into
every call site. A single string has exactly one shape. Repo ids cannot contain
a newline, so the join is unambiguous.

Entries are only added after a warm-up actually succeeded (see
ModelsScreen._warm_custom_model), which keeps repos WhisperX cannot load - e.g.
a transformers-format checkpoint with no CTranslate2 model.bin - out of the
list rather than letting them fail later, at job time.
"""

from typing import Iterable, List

from PySide6.QtCore import QSettings

from diarrhizer_gui import settings_keys

_SEPARATOR = "\n"


def load_custom_models(settings: QSettings) -> List[str]:
    """Return the remembered repo ids, in the order they were added."""
    raw = settings.value(settings_keys.CUSTOM_ASR_MODELS, "")
    if not isinstance(raw, str):
        # A settings store written by another Qt backend (or hand-edited) can
        # still hand back a list/None here - normalise instead of crashing.
        raw = _SEPARATOR.join(str(item) for item in (raw or []))
    return [line.strip() for line in raw.split(_SEPARATOR) if line.strip()]


def add_custom_model(settings: QSettings, repo_id: str) -> bool:
    """Remember repo_id. Returns False if it was blank or already known."""
    repo_id = repo_id.strip()
    if not repo_id:
        return False
    existing = load_custom_models(settings)
    if repo_id in existing:
        return False
    _store(settings, existing + [repo_id])
    return True


def remove_custom_model(settings: QSettings, repo_id: str) -> None:
    """Forget repo_id. Downloaded files are left alone - this is list-only."""
    existing = load_custom_models(settings)
    remaining = [item for item in existing if item != repo_id]
    if len(remaining) != len(existing):
        _store(settings, remaining)


def is_whisper_repo(repo_id: str) -> bool:
    return "whisper" in repo_id.lower()


def detect_cached_asr_models(cached: Iterable, presets: Iterable[str] = ()) -> List[str]:
    """Repo ids in the HF cache that WhisperX can use as an ASR model.

    Catches models downloaded outside the app (or before it remembered custom
    models): a CTranslate2 checkpoint (model.bin) whose name says whisper.
    Alignment/diarization repos and transformers-format checkpoints are skipped,
    as are the Systran repos the `presets` aliases already stand for.
    """
    preset_repos = {f"Systran/faster-whisper-{alias}" for alias in presets}
    return [
        info.repo_id
        for info in cached
        if info.is_ctranslate2
        and is_whisper_repo(info.repo_id)
        and info.repo_id not in preset_repos
    ]


def asr_model_choices(
    settings: QSettings, presets: List[str], extra: Iterable[str] = ()
) -> List[str]:
    """Presets, remembered custom repo ids, then `extra` (e.g. detected cache
    entries), without duplicates."""
    choices = list(presets)
    for repo_id in [*load_custom_models(settings), *extra]:
        if repo_id not in choices:
            choices.append(repo_id)
    return choices


def _store(settings: QSettings, models: List[str]) -> None:
    settings.setValue(settings_keys.CUSTOM_ASR_MODELS, _SEPARATOR.join(models))
