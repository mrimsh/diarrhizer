"""Normalize WhisperX transcribe/align payloads into our transcript shape."""

from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_transcript_fields(result: dict) -> dict:
    """Pull text/segments/words out of a WhisperX transcribe or align result.

    whisperx.align() returns segments (words nested) plus word_segments — not
    top-level text/words. Raw transcribe() often has neither top-level field.
    """
    segments = list(result.get("segments") or [])

    words = result.get("word_segments") or result.get("words")
    if not words:
        words = []
        for seg in segments:
            if isinstance(seg, dict):
                words.extend(seg.get("words") or [])

    cleaned_words = []
    for word in words:
        if not isinstance(word, dict):
            continue
        text = word.get("word") or word.get("text") or ""
        cleaned_words.append({
            "start": _as_float(word.get("start", 0)),
            "end": _as_float(word.get("end", 0)),
            "word": str(text),
        })

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
