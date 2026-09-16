"""Speaker name resolution utilities."""

import re

_SPEAKER_RE = re.compile(r"^speaker[_\s-]*(\d+)$", re.IGNORECASE)


def normalize_speaker_id(speaker_id: str) -> str:
    """Canonicalize pyannote/WhisperX labels to the documented Speaker_XX form.

    Pyannote emits SPEAKER_00; docs, CLI help, and --speakers files use Speaker_00.
    """
    match = _SPEAKER_RE.match(str(speaker_id).strip())
    if match:
        return f"Speaker_{int(match.group(1)):02d}"
    return speaker_id


def resolve_speaker_name(speaker_id: str, speakers: dict | None) -> str:
    """Resolve speaker display name from mapping.

    Args:
        speaker_id: The diarization speaker ID (e.g., "Speaker_00" or "SPEAKER_00")
        speakers: Optional mapping {speaker_id: display_name}

    Returns:
        Display name if mapping exists, otherwise the normalized speaker_id
    """
    speaker_id = normalize_speaker_id(speaker_id)
    if not speakers:
        return speaker_id
    if speaker_id in speakers:
        return speakers[speaker_id]
    for key, value in speakers.items():
        if normalize_speaker_id(str(key)) == speaker_id:
            return value
    return speaker_id
