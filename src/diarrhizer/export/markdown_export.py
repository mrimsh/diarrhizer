"""Markdown export functionality for diarized transcripts."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from diarrhizer.export.turns import format_timestamp as _format_timestamp
from diarrhizer.export.turns import group_turns, speaking_time

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import PipelineConfig


# [SEMANTIC-BEGIN] EXPORT:MARKDOWN
# @purpose: Export merged segments to human-readable Markdown format
# @description: Creates a transcript with timecodes and speaker labels. Supports speaker name mapping via config.speakers.
#   The header lists each speaker's total speaking time and share. The body has one line per speaker
#   turn (EXPORT:SPEAKER_TURNS): consecutive segments of the same person are joined, so a reply reads as
#   one paragraph instead of one line per Whisper phrase. Word-level data lives only in result.json
#   (EXPORT:JSON), which keeps every segment and word's start/end/speaker_id in full.
# @inputs: segments data from merge stage, config metadata (including optional speakers mapping)
# @outputs: Markdown-formatted string
# @sideEffects: None (pure function)
# @errors: None
# @see: STAGE:EXPORT, EXPORT:JSON, EXPORT:TEXT, EXPORT:SPEAKER_TURNS, CONFIG:PIPELINE
def export_to_markdown(
    segments: list[dict[str, Any]],
    config: "PipelineConfig",
    input_path: str,
) -> str:
    """Export segments to Markdown format.

    Format:
    - Metadata header, then a speaking-time summary per speaker
    - One "[HH:MM:SS → HH:MM:SS] **Name:** text" line per speaker turn
    - Word-level timings are in result.json only

    Args:
        segments: List of segment dictionaries with start, end, speaker_id, text
        config: Pipeline configuration
        input_path: Original input file path

    Returns:
        Markdown-formatted transcript string
    """
    speakers = config.speakers
    lines: list[str] = []

    # Header with metadata
    lines.append("# Transcription")
    lines.append("")
    lines.append(f"**Input:** {input_path}")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Language:** {config.language}")
    lines.append(f"**Device:** {config.device}")
    lines.append("")

    totals = speaking_time(segments, speakers)
    total_seconds = sum(totals.values())
    if totals:
        lines.append("**Speakers:**")
        lines.append("")
        for name, seconds in sorted(totals.items(), key=lambda item: -item[1]):
            share = f" ({seconds / total_seconds:.0%})" if total_seconds > 0 else ""
            lines.append(f"- {name} — {_format_timestamp(seconds)}{share}")
        lines.append("")

    for turn in group_turns(segments, speakers):
        start_time = _format_timestamp(turn["start"])
        end_time = _format_timestamp(turn["end"])
        lines.append(f"[{start_time} → {end_time}] **{turn['speaker_name']}:** {turn['text']}")
        lines.append("")

    return "\n".join(lines)


# [SEMANTIC-END] EXPORT:MARKDOWN
