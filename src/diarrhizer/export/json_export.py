"""JSON export functionality for diarized transcripts."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# [SEMANTIC-BEGIN] EXPORT:JSON
# @purpose: Export merged segments to structured JSON format
# @description: Creates machine-readable JSON with metadata and segments
# @inputs: segments data from merge stage, config metadata
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
                "text": "Hello world",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "Hello", "speaker_id": "Speaker_00"},
                    {"start": 0.5, "end": 1.0, "word": "world", "speaker_id": "Speaker_00"}
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
        "segments": segments,
    }

    return json.dumps(output_data, indent=2, ensure_ascii=False)


# [SEMANTIC-END] EXPORT:JSON
