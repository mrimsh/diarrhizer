"""Readable, speaker-less transcript built from raw ASR segments."""

from typing import Any

# Whisper emits short phrase-sized segments; a pause at least this long between
# two of them starts a new paragraph.
PARAGRAPH_GAP_SECONDS = 2.0
# Past this many characters a paragraph is closed at the next sentence end even
# without a pause, so continuous speech does not become one wall of text.
PARAGRAPH_SOFT_LIMIT = 600
_SENTENCE_END = (".", "!", "?", "\u2026")


# [SEMANTIC-BEGIN] EXPORT:PLAIN_TRANSCRIPT
# @purpose: Turn raw ASR segments into readable paragraphs, before/without diarization
# @description: Joins phrase-sized Whisper segments into running text and breaks
#   paragraphs on pauses (PARAGRAPH_GAP_SECONDS) or, past PARAGRAPH_SOFT_LIMIT
#   characters, at the next sentence end. No timestamps, no speaker labels -
#   the counterpart of result.txt for the "just give me the text" case.
# @inputs: transcript segments (dicts with start, end, text)
# @outputs: Plain-text string, paragraphs separated by a blank line
# @sideEffects: None (pure function)
# @errors: None
# @see: STAGE:TRANSCRIBE, EXPORT:TEXT
def format_plain_transcript(segments: list[dict[str, Any]]) -> str:
    """Join ASR segments into paragraphs of readable text."""
    paragraphs: list[list[str]] = []
    length = 0
    prev_end = None
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start")
        gap = (
            start - prev_end
            if start is not None and prev_end is not None
            else 0.0
        )
        current = paragraphs[-1] if paragraphs else None
        long_enough = current is not None and length >= PARAGRAPH_SOFT_LIMIT
        at_sentence_end = current is not None and current[-1].endswith(_SENTENCE_END)
        if current is None or gap >= PARAGRAPH_GAP_SECONDS or (long_enough and at_sentence_end):
            paragraphs.append([text])
            length = len(text)
        else:
            current.append(text)
            length += len(text) + 1
        prev_end = seg.get("end", prev_end)
    return "\n\n".join(" ".join(p) for p in paragraphs) + ("\n" if paragraphs else "")
# [SEMANTIC-END] EXPORT:PLAIN_TRANSCRIPT
