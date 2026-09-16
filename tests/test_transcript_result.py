"""Tests for WhisperX payload normalization."""

from diarrhizer.adapters.transcript_result import extract_transcript_fields


def test_extracts_word_segments_and_concatenates_text():
    result = extract_transcript_fields({
        "segments": [
            {"start": 0, "end": 1, "text": "hello"},
            {"start": 1, "end": 2, "text": "world"},
        ],
        "word_segments": [
            {"start": 0.0, "end": 0.5, "word": "hello"},
            {"start": 1.0, "end": 1.5, "word": "world"},
        ],
    })
    assert result["text"] == "hello world"
    assert [w["word"] for w in result["words"]] == ["hello", "world"]


def test_flattens_per_segment_words_when_no_top_level_list():
    result = extract_transcript_fields({
        "segments": [{
            "start": 0,
            "end": 1,
            "text": "hi there",
            "words": [
                {"start": 0.0, "end": 0.4, "word": "hi"},
                {"start": 0.4, "end": 1.0, "word": "there"},
            ],
        }],
    })
    assert result["text"] == "hi there"
    assert [w["word"] for w in result["words"]] == ["hi", "there"]


def test_keeps_explicit_top_level_text():
    result = extract_transcript_fields({
        "text": "already here",
        "segments": [{"start": 0, "end": 1, "text": "ignored for text"}],
    })
    assert result["text"] == "already here"


def test_empty_result():
    result = extract_transcript_fields({})
    assert result == {"text": "", "segments": [], "words": []}
