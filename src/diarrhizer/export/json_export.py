"""JSON export functionality for diarized transcripts."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from diarrhizer.export.speakers import resolve_speaker_name


# [SEMANTIC-BEGIN] EXPORT:JSON
# @purpose: Export merged segments to structured JSON format
# @description: Creates machine-readable JSON with metadata and segments. Adds speaker_name field when config.speakers mapping is provided
# @inputs: segments data from merge stage, config metadata (including optional speakers mapping)
# @outputs: JSON-formatted string
# @sideEffects: None (pure function)
# @errors: None
# @see: STAGE:EXPORT, EXPORT:MARKDOWN
def export_to_json(
    segments: list[dict[str, Any]],
    config: dict[str, Any],
    input_path: str,
) -> str:
    """Export segments to JSON format.

    Output structure:
    {
        "metadata": {
            "input_file": "...",
            "created_at": "...",
            "config": {...}
        },
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker_id": "Speaker_00",
                "speaker_name": "Ivan",
                "text": "Hello world",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "Hello", "speaker_id": "Speaker_00", "speaker_name": "Ivan"},
                    {"start": 0.5, "end": 1.0, "word": "world", "speaker_id": "Speaker_00", "speaker_name": "Ivan"}
                ]
            }
        ]
    }

    Args:
        segments: List of segment dictionaries with start, end, speaker_id, text
        config: Pipeline configuration dictionary
        input_path: Original input file path

    Returns:
        JSON-formatted transcript string
    """
    speakers = config.get("speakers")

    # Enrich segments with speaker_name
    enriched_segments = []
    for seg in segments:
        enriched_seg = dict(seg)
        speaker_id = seg.get("speaker_id", "Speaker_00")
        enriched_seg["speaker_name"] = resolve_speaker_name(speaker_id, speakers)

        # Also enrich word-level details
        if "words" in seg and seg["words"]:
            enriched_words = []
            for word in seg["words"]:
                enriched_word = dict(word)
                word_speaker_id = word.get("speaker_id", speaker_id)
                enriched_word["speaker_name"] = resolve_speaker_name(word_speaker_id, speakers)
                enriched_words.append(enriched_word)
            enriched_seg["words"] = enriched_words

        enriched_segments.append(enriched_seg)

    # Build output structure
    output_data = {
        "metadata": {
            "input_file": input_path,
            "created_at": datetime.now().isoformat(),
            "config": {
                "language": config.get("language", "auto"),
                "device": config.get("device", "cpu"),
                "min_speakers": config.get("min_speakers", 1),
                "max_speakers": config.get("max_speakers", 10),
            },
        },
        "segments": enriched_segments,
    }

    return json.dumps(output_data, indent=2, ensure_ascii=False)


# [SEMANTIC-END] EXPORT:JSON
