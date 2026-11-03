"""Normalize WhisperX transcribe/align payloads into our transcript shape."""

import math
from typing import Any


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


# [SEMANTIC-BEGIN] ASR:NORMALIZE_RESULT
# @purpose: Preserve ASR text and word ownership when alignment timestamps are incomplete
# @description: Prefer nested segment words to retain segment_index; invalid/missing word times
#   remain None rather than becoming zero. Flat legacy payloads are still supported.
# @inputs: WhisperX transcribe/align result
# @outputs: text, segments, words with nullable times and optional segment_index
# @sideEffects: None
# @errors: None
# @see: ADAPTER:WHISPERX_ASR, MERGE:ASSIGN_SPEAKERS
def extract_transcript_fields(result: dict) -> dict:
    """Pull text/segments/words out of a WhisperX transcribe or align result.

    whisperx.align() returns segments (words nested) plus word_segments — not
    top-level text/words. Raw transcribe() often has neither top-level field.
    """
    segments = [dict(seg) for seg in (result.get("segments") or []) if isinstance(seg, dict)]

    nested = [(word, index) for index, seg in enumerate(segments) if isinstance(seg, dict)
              for word in (seg.get("words") or [])]
    words = nested or [(word, word.get("segment_index") if isinstance(word, dict) else None)
                       for word in (result.get("word_segments") or result.get("words") or [])]

    cleaned_words = []
    for word, index in words:
        if not isinstance(word, dict):
            continue
        text = word.get("word") or word.get("text") or ""
        start, end = _as_float(word.get("start")), _as_float(word.get("end"))
        if start is None or end is None or end < start:
            start = end = None
        cleaned = {
            "start": start,
            "end": end,
            "word": str(text),
        }
        if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(segments):
            cleaned["segment_index"] = index
        cleaned_words.append(cleaned)

    if nested:
        # The nested representation written to ASR JSON must use the same safe times.
        by_segment: dict[int, list] = {}
        for word in cleaned_words:
            by_segment.setdefault(word["segment_index"], []).append(dict(word))
        for index, seg in enumerate(segments):
            if "words" in seg:
                seg["words"] = by_segment.get(index, [])

    text = (result.get("text") or "").strip()
    if not text:
        text = " ".join(
            (seg.get("text") or "").strip()
            for seg in segments
            if isinstance(seg, dict) and (seg.get("text") or "").strip()
        ).strip()

    return {
        "text": text,
        "segments": segments,
        "words": cleaned_words,
    }
# [SEMANTIC-END] ASR:NORMALIZE_RESULT
