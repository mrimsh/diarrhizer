"""Tests for the speaker-assignment algorithm in diarrhizer.pipeline.stages.merge."""

from diarrhizer.pipeline.stages.merge import assign_speakers, _find_overlapping_speaker


# --- assign_speakers -------------------------------------------------------

def test_empty_transcript_returns_empty_list():
    assert assign_speakers([], [], [{"start": 0, "end": 1, "speaker": "Speaker_00"}]) == []


def test_empty_diarization_defaults_all_segments_to_speaker_00():
    segments = [
        {"start": 0, "end": 1, "text": "hello"},
        {"start": 1, "end": 2, "text": "world"},
    ]
    result = assign_speakers(segments, [], [])
    assert [s["speaker_id"] for s in result] == ["Speaker_00", "Speaker_00"]
    assert [s["text"] for s in result] == ["hello", "world"]


def test_segment_assigned_to_fully_overlapping_speaker():
    segments = [{"start": 0, "end": 5, "text": "hi"}]
    diar = [{"start": 0, "end": 5, "speaker": "Speaker_01"}]
    result = assign_speakers(segments, [], diar)
    assert result[0]["speaker_id"] == "Speaker_01"


def test_segment_assigned_to_speaker_with_max_overlap():
    segments = [{"start": 0, "end": 10, "text": "hi"}]
    diar = [
        {"start": 0, "end": 3, "speaker": "Speaker_00"},   # 3s overlap
        {"start": 3, "end": 10, "speaker": "Speaker_01"},  # 7s overlap
    ]
    result = assign_speakers(segments, [], diar)
    assert result[0]["speaker_id"] == "Speaker_01"


def test_words_get_per_word_speaker_ids():
    segments = [{"start": 0, "end": 10, "text": "hi there"}]
    words = [
        {"start": 0, "end": 1, "word": "hi"},
        {"start": 6, "end": 7, "word": "there"},
    ]
    diar = [
        {"start": 0, "end": 5, "speaker": "Speaker_00"},
        {"start": 5, "end": 10, "speaker": "Speaker_01"},
    ]
    result = assign_speakers(segments, words, diar)
    assert result[0]["words"] == [
        {"start": 0, "end": 1, "word": "hi", "speaker_id": "Speaker_00"},
        {"start": 6, "end": 7, "word": "there", "speaker_id": "Speaker_01"},
    ]


def test_word_outside_all_segments_is_dropped():
    # Documents current behavior: a word must start within [seg_start, seg_end)
    # of some segment to be attributed at all; otherwise it's silently omitted.
    segments = [{"start": 0, "end": 2, "text": "hi"}]
    words = [{"start": 5, "end": 6, "word": "orphan"}]
    diar = [{"start": 0, "end": 10, "speaker": "Speaker_00"}]
    result = assign_speakers(segments, words, diar)
    assert "words" not in result[0]


def test_missing_diarization_keys_default_gracefully():
    segments = [{"start": 0, "end": 1, "text": "hi"}]
    diar = [{"start": 0, "end": 1}]  # no "speaker" key
    result = assign_speakers(segments, [], diar)
    assert result[0]["speaker_id"] == "Speaker_00"


# --- _find_overlapping_speaker ---------------------------------------------

def test_find_overlapping_speaker_no_diarization_returns_default():
    assert _find_overlapping_speaker(0, 5, []) == "Speaker_00"


def test_find_overlapping_speaker_picks_max_overlap():
    diar = [
        {"start": 0, "end": 2, "speaker": "A"},
        {"start": 2, "end": 10, "speaker": "B"},
    ]
    assert _find_overlapping_speaker(0, 10, diar) == "B"


def test_find_overlapping_speaker_gap_before_picks_closest():
    diar = [{"start": 10, "end": 20, "speaker": "A"}]
    # Segment [0, 5) doesn't overlap [10, 20) at all -> falls back to closest
    assert _find_overlapping_speaker(0, 5, diar) == "A"


def test_find_overlapping_speaker_gap_between_two_picks_nearest():
    diar = [
        {"start": 0, "end": 5, "speaker": "A"},
        {"start": 20, "end": 25, "speaker": "B"},
    ]
    # Segment [6, 8) is 1s from A's end (5) and 12s from B's start (20)
    assert _find_overlapping_speaker(6, 8, diar) == "A"
