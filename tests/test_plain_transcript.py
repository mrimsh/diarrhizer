"""Tests for export.plain_transcript (readable speaker-less ASR text)."""

from diarrhizer.export.plain_transcript import (
    PARAGRAPH_GAP_SECONDS,
    PARAGRAPH_SOFT_LIMIT,
    format_plain_transcript,
)


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_empty_input_gives_empty_string():
    assert format_plain_transcript([]) == ""


def test_close_segments_are_joined_into_one_paragraph():
    out = format_plain_transcript([seg(0, 1, " Привет."), seg(1.2, 2, "Как дела?")])
    assert out == "Привет. Как дела?\n"


def test_long_pause_starts_new_paragraph():
    out = format_plain_transcript(
        [seg(0, 1, "Раз."), seg(1 + PARAGRAPH_GAP_SECONDS, 5, "Два.")]
    )
    assert out == "Раз.\n\nДва.\n"


def test_blank_segments_are_skipped():
    out = format_plain_transcript([seg(0, 1, "  "), seg(1, 2, "Текст.")])
    assert out == "Текст.\n"


def test_long_run_breaks_at_sentence_end():
    sentence = "x" * 300 + "."
    segs = [seg(i, i + 0.9, sentence) for i in range(4)]
    paragraphs = format_plain_transcript(segs).strip().split("\n\n")
    assert len(paragraphs) > 1
    assert all(p.endswith(".") for p in paragraphs)
    assert PARAGRAPH_SOFT_LIMIT <= len(paragraphs[0])
