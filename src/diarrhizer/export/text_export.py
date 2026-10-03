"""Plain-text export functionality for diarized transcripts."""

from typing import TYPE_CHECKING, Any

from diarrhizer.export.turns import format_turn_line, group_turns

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import PipelineConfig


# [SEMANTIC-BEGIN] EXPORT:TEXT
# @purpose: Export merged segments to a plain-text transcript
# @description: Same speaker turns, timestamps and speaker labels as Markdown, without the
#   markup or the metadata header — the plain .txt counterpart of result.md,
#   for pasting somewhere that won't render Markdown
# @inputs: segments data from merge stage, config metadata (including optional speakers mapping)
# @outputs: Plain-text string
# @sideEffects: None (pure function)
# @errors: None
# @see: STAGE:EXPORT, EXPORT:MARKDOWN, EXPORT:JSON, EXPORT:SPEAKER_TURNS
def export_to_text(
    segments: list[dict[str, Any]],
    config: "PipelineConfig",
    input_path: str,
) -> str:
    """Export segments to plain text, one line per speaker turn."""
    lines: list[str] = []
    lines.append(f"Input: {input_path}")
    lines.append(f"Language: {config.language}")
    lines.append("")

    for turn in group_turns(segments, config.speakers):
        lines.append(format_turn_line(turn))
        lines.append("")

    return "\n".join(lines)
# [SEMANTIC-END] EXPORT:TEXT
