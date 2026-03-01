"""Speaker name resolution utilities."""


def resolve_speaker_name(speaker_id: str, speakers: dict | None) -> str:
    """Resolve speaker display name from mapping.

    Args:
        speaker_id: The diarization speaker ID (e.g., "Speaker_00")
        speakers: Optional mapping {speaker_id: display_name}

    Returns:
        Display name if mapping exists, otherwise the original speaker_id
    """
    if speakers and speaker_id in speakers:
        return speakers[speaker_id]
    return speaker_id
