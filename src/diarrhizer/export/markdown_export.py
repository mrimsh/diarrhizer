"""Markdown export functionality for diarized transcripts."""

from datetime import datetime
from pathlib import Path
from typing import Any

from diarrhizer.export.speakers import resolve_speaker_name


# [SEMANTIC-BEGIN] EXPORT:MARKDOWN
# @purpose: Export merged segments to human-readable Markdown format
# @description: Creates a transcript with timecodes and speaker labels. Supports speaker name mapping via config.speakers
# @inputs: segments data from merge stage, config metadata (including optional speakers mapping)
# @outputs: Markdown-formatted string
# @sideEffects: None (pure function)
# @errors: None
# @see: STAGE:EXPORT, EXPORT:JSON
def export_to_markdown(
    segments: list[dict[str, Any]],
    config: dict[str, Any],
    input_path: str,
) -> str:
    """Export segments to Markdown format.

    Format:
    - Timecodes in [HH:MM:SS] format
    - Speaker labels followed by transcript text
    - Word-level details if available (optional, indented)

    Args:
        segments: List of segment dictionaries with start, end, speaker_id, text
        config: Pipeline configuration dictionary
        input_path: Original input file path

    Returns:
        Markdown-formatted transcript string
    """
    speakers = config.get("speakers")
    lines: list[str] = []

    # Header with metadata
    lines.append("# Transcription")
    lines.append("")
    lines.append(f"**Input:** {input_path}")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Language:** {config.get('language', 'auto')}")
    lines.append(f"**Device:** {config.get('device', 'cpu')}")
    lines.append("")

    # Segments
    for seg in segments:
        start_time = _format_timestamp(seg.get("start", 0))
        end_time = _format_timestamp(seg.get("end", 0))
        speaker_id = seg.get("speaker_id", "Speaker_00")
        speaker_name = resolve_speaker_name(speaker_id, speakers)
        text = seg.get("text", "")

        # Main segment line
        lines.append(f"[{start_time} → {end_time}] **{speaker_name}:** {text}")

        # Word-level details if available
        words = seg.get("words")
        if words:
            for word in words:
                word_start = _format_timestamp(word.get("start", 0))
                word_text = word.get("word", "")
                word_speaker_id = word.get("speaker_id", speaker_id)
                word_speaker_name = resolve_speaker_name(word_speaker_id, speakers)
                if word_speaker_id != speaker_id:
                    lines.append(f"    - [{word_start}] {word_text} ({word_speaker_name})")
                else:
                    lines.append(f"    - [{word_start}] {word_text}")

        lines.append("")

    return "\n".join(lines)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# [SEMANTIC-END] EXPORT:MARKDOWN
