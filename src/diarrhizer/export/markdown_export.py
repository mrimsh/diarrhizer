"""Markdown export functionality for diarized transcripts."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from diarrhizer.export.speakers import resolve_speaker_name

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import PipelineConfig


# [SEMANTIC-BEGIN] EXPORT:MARKDOWN
# @purpose: Export merged segments to human-readable Markdown format
# @description: Creates a transcript with timecodes and speaker labels. Supports speaker name mapping via config.speakers.
#   One line per segment and nothing else: the per-word timestamp dump that used
#   to follow each segment restated timings the segment line already carries and
#   buried the actual text, so word-level data now lives only in result.json
#   (EXPORT:JSON), which keeps every word's start/end/speaker_id in full.
# @inputs: segments data from merge stage, config metadata (including optional speakers mapping)
# @outputs: Markdown-formatted string
# @sideEffects: None (pure function)
# @errors: None
# @see: STAGE:EXPORT, EXPORT:JSON, EXPORT:TEXT, CONFIG:PIPELINE
def export_to_markdown(
    segments: list[dict[str, Any]],
    config: "PipelineConfig",
    input_path: str,
) -> str:
    """Export segments to Markdown format.

    Format:
    - Timecodes in [HH:MM:SS] format
    - Speaker labels followed by transcript text
    - One line per segment; word-level timings are in result.json only

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

    # Segments
    for seg in segments:
        start_time = _format_timestamp(seg.get("start", 0))
        end_time = _format_timestamp(seg.get("end", 0))
        speaker_id = seg.get("speaker_id", "Speaker_00")
        speaker_name = resolve_speaker_name(speaker_id, speakers)
        text = seg.get("text", "")

        # Main segment line
        lines.append(f"[{start_time} → {end_time}] **{speaker_name}:** {text}")
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
