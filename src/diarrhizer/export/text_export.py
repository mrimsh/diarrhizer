"""Plain-text export functionality for diarized transcripts."""

from typing import TYPE_CHECKING, Any

from diarrhizer.export.markdown_export import _format_timestamp
from diarrhizer.export.speakers import resolve_speaker_name

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import PipelineConfig


# [SEMANTIC-BEGIN] EXPORT:TEXT
# @purpose: Export merged segments to a plain-text transcript
# @description: Same timestamps and speaker labels as Markdown, without the
#   markup or the metadata header — the plain .txt counterpart of result.md,
#   for pasting somewhere that won't render Markdown
# @inputs: segments data from merge stage, config metadata (including optional speakers mapping)
# @outputs: Plain-text string
# @sideEffects: None (pure function)
# @errors: None
# @see: STAGE:EXPORT, EXPORT:MARKDOWN, EXPORT:JSON
def export_to_text(
    segments: list[dict[str, Any]],
    config: "PipelineConfig",
    input_path: str,
) -> str:
    """Export segments to plain text (no word-level dump)."""
    speakers = config.speakers
    lines: list[str] = []
    lines.append(f"Input: {input_path}")
    lines.append(f"Language: {config.language}")
    lines.append("")

    for seg in segments:
        start_time = _format_timestamp(seg.get("start", 0))
        end_time = _format_timestamp(seg.get("end", 0))
        speaker_id = seg.get("speaker_id", "Speaker_00")
        speaker_name = resolve_speaker_name(speaker_id, speakers)
        text = seg.get("text", "")
        lines.append(f"[{start_time} → {end_time}] {speaker_name}: {text}")
        lines.append("")

    return "\n".join(lines)
# [SEMANTIC-END] EXPORT:TEXT
