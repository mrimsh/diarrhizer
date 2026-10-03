"""SRT subtitle export for diarized transcripts."""

from typing import TYPE_CHECKING, Any

from diarrhizer.export.plain_transcript import SENTENCE_END
from diarrhizer.export.speakers import resolve_speaker_name

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import PipelineConfig

# Readable-subtitle limits: a segment longer than either is split into several
# cues at word boundaries (when word timings exist).
MAX_CUE_SECONDS = 7.0
MAX_CUE_CHARS = 84


def _srt_time(seconds: float) -> str:
    millis = max(0, int(round((seconds or 0) * 1000)))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _split_segment(seg: dict[str, Any]) -> list[tuple[float, float, str]]:
    """(start, end, text) cues for one segment: the whole segment when it is
    short enough or has no word timings, else word chunks within the limits."""
    text = (seg.get("text") or "").strip()
    start = float(seg.get("start") or 0.0)
    end = float(seg.get("end") or start)
    words = [w for w in seg.get("words") or [] if (w.get("word") or "").strip()]
    timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    fits = end - start <= MAX_CUE_SECONDS and len(text) <= MAX_CUE_CHARS
    if fits or not timed:
        return [(start, end, text)] if text else []

    cues: list[tuple[float, float, str]] = []
    chunk: list[str] = []
    chunk_start: float | None = None
    chunk_end: float | None = None
    for word in words:
        token = word["word"].strip()
        if chunk_start is None:
            word_start = word.get("start")
            if word_start is not None:
                chunk_start = float(word_start)
            else:
                chunk_start = chunk_end if chunk_end is not None else start
        chunk.append(token)
        if word.get("end") is not None:
            chunk_end = float(word["end"])
        duration = (chunk_end if chunk_end is not None else chunk_start) - chunk_start
        chars = len(" ".join(chunk))
        if (
            duration >= MAX_CUE_SECONDS
            or chars >= MAX_CUE_CHARS
            or (token.endswith(SENTENCE_END) and duration >= MAX_CUE_SECONDS / 2)
        ):
            cues.append((chunk_start, chunk_end if chunk_end is not None else chunk_start, " ".join(chunk)))
            chunk = []
            chunk_start = None
    if chunk:
        cues.append((chunk_start, max(chunk_end or end, chunk_start), " ".join(chunk)))
    return cues


# [SEMANTIC-BEGIN] EXPORT:SRT
# @purpose: Export merged segments as SRT subtitles ("Name: text" per cue)
# @description: One cue per merged segment, prefixed with the speaker's display name. A segment
#   longer than MAX_CUE_SECONDS or MAX_CUE_CHARS is split into several cues at word boundaries
#   using its word timings (preferring sentence ends); without word timings it stays one cue.
#   Segments with empty text are skipped - an empty cue is invalid SRT.
# @inputs: segments data from merge stage, config metadata (including optional speakers mapping)
# @outputs: SRT-formatted string
# @sideEffects: None (pure function)
# @errors: None
# @see: STAGE:EXPORT, EXPORT:TEXT, EXPORT:JSON
def export_to_srt(
    segments: list[dict[str, Any]],
    config: "PipelineConfig",
    input_path: str,
) -> str:
    """Export segments to SRT subtitles."""
    lines: list[str] = []
    index = 0
    for seg in segments:
        name = resolve_speaker_name(seg.get("speaker_id") or "Speaker_00", config.speakers)
        for start, end, text in _split_segment(seg):
            index += 1
            lines.append(str(index))
            lines.append(f"{_srt_time(start)} --> {_srt_time(max(start, end))}")
            lines.append(f"{name}: {text}")
            lines.append("")
    return "\n".join(lines)
# [SEMANTIC-END] EXPORT:SRT
