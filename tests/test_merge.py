"""Tests for the speaker-assignment algorithm in diarrhizer.pipeline.stages.merge."""

import random

from diarrhizer.pipeline.stages.merge import (
    assign_speakers,
    _find_overlapping_speaker,
    _DiarizationSweep,
)


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


def test_word_in_gap_between_two_segments_is_dropped():
    # The word-to-segment two-pointer sweep must not misattribute a word that
    # falls in a gap between segments to either neighbor.
    segments = [
        {"start": 0, "end": 2, "text": "first"},
        {"start": 5, "end": 7, "text": "second"},
    ]
    words = [{"start": 3, "end": 3.5, "word": "orphan"}]
    diar = [{"start": 0, "end": 10, "speaker": "Speaker_00"}]
    result = assign_speakers(segments, words, diar)
    assert "words" not in result[0]
    assert "words" not in result[1]


def test_words_across_multiple_segments_each_map_to_their_own_segment():
    segments = [
        {"start": 0, "end": 2, "text": "first"},
        {"start": 2, "end": 4, "text": "second"},
        {"start": 4, "end": 6, "text": "third"},
    ]
    words = [
        {"start": 0.5, "end": 1, "word": "a"},
        {"start": 1.5, "end": 2, "word": "b"},
        {"start": 2.5, "end": 3, "word": "c"},
        {"start": 5.5, "end": 6, "word": "d"},
    ]
    diar = [{"start": 0, "end": 6, "speaker": "Speaker_00"}]
    result = assign_speakers(segments, words, diar)
    assert [w["word"] for w in result[0]["words"]] == ["a", "b"]
    assert [w["word"] for w in result[1]["words"]] == ["c"]
    assert [w["word"] for w in result[2]["words"]] == ["d"]


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


# --- _DiarizationSweep -------------------------------------------------

def test_sweep_no_diarization_returns_default():
    sweep = _DiarizationSweep([])
    assert sweep.find(0, 5) == "Speaker_00"


def test_sweep_sequential_queries_pick_max_overlap_each_time():
    # Same sweep instance queried repeatedly as queries move forward in time -
    # the window must keep tracking correctly across calls, not just once.
    diar = [
        {"start": 0, "end": 2, "speaker": "A"},
        {"start": 2, "end": 10, "speaker": "B"},
    ]
    sweep = _DiarizationSweep(diar)
    assert sweep.find(0, 1) == "A"
    assert sweep.find(2.5, 3) == "B"
    assert sweep.find(5, 6) == "B"


def test_sweep_gap_before_picks_closest():
    sweep = _DiarizationSweep([{"start": 10, "end": 20, "speaker": "A"}])
    assert sweep.find(0, 5) == "A"


def test_sweep_gap_between_two_picks_nearest():
    diar = [
        {"start": 0, "end": 5, "speaker": "A"},
        {"start": 20, "end": 25, "speaker": "B"},
    ]
    sweep = _DiarizationSweep(diar)
    assert sweep.find(6, 8) == "A"  # 1s from A's end vs 12s from B's start
    assert sweep.find(15, 16) == "B"  # 10s from A's end vs 4s from B's start


def test_sweep_handles_out_of_order_input_by_sorting_internally():
    # Constructor input isn't guaranteed sorted by the caller; the sweep must
    # sort it itself rather than assume diar_segments arrives pre-sorted.
    diar = [
        {"start": 10, "end": 20, "speaker": "LATER"},
        {"start": 0, "end": 5, "speaker": "EARLIER"},
    ]
    sweep = _DiarizationSweep(diar)
    assert sweep.find(0, 5) == "EARLIER"
    assert sweep.find(10, 20) == "LATER"


def test_sweep_long_segment_masking_nested_shorter_expired_segment():
    # Regression case for the trickiest part of the sweep's correctness
    # argument: a long-running segment (A, 0-100) keeps the window's left
    # edge from advancing past a much shorter, already-ended nested segment
    # (B, 1-2). A still genuinely overlaps queries in this range, so the
    # "closest fallback" path (which only looks outside the window) is never
    # reached here - this just confirms overlap selection stays correct.
    diar = [
        {"start": 0, "end": 100, "speaker": "A"},
        {"start": 1, "end": 2, "speaker": "B"},
    ]
    sweep = _DiarizationSweep(diar)
    assert sweep.find(50, 51) == "A"


def test_sweep_many_concurrently_overlapping_segments_picks_max():
    # Stress the window-scan itself: several mutually overlapping segments
    # active at once - the widest-overlap one must still win.
    diar = [
        {"start": 0, "end": 10, "speaker": "wide"},
        {"start": 4, "end": 6, "speaker": "narrow1"},
        {"start": 4.5, "end": 5.5, "speaker": "narrow2"},
    ]
    sweep = _DiarizationSweep(diar)
    # "wide" and "narrow1" both overlap [4, 6) by 2s; ties resolve to
    # whichever is encountered first in start order, i.e. "wide".
    assert sweep.find(4, 6) == "wide"
    assert sweep.find(0, 1) == "wide"  # only "wide" covers this range at all


# --- Differential test: sweep must always agree with the brute-force ------
# reference implementation, across many random chronologically-ordered
# query streams (including overlapping diarization segments and gaps).

def _random_intervals(rng: random.Random, count: int, max_time: float, max_len: float):
    intervals = []
    for _ in range(count):
        start = rng.uniform(0, max_time)
        end = start + rng.uniform(0.1, max_len)
        intervals.append((start, end))
    return intervals


def _random_diar_segments(rng: random.Random, count: int, max_time: float, max_len: float):
    return [
        {"start": s, "end": e, "speaker": f"Speaker_{i % 4:02d}"}
        for i, (s, e) in enumerate(_random_intervals(rng, count, max_time, max_len))
    ]


def test_sweep_matches_brute_force_reference_on_random_inputs():
    for seed in range(300):
        rng = random.Random(seed)
        n_diar = rng.choice([0, 1, 2, 5, 15, 40])
        diar_segments = _random_diar_segments(rng, n_diar, max_time=50, max_len=6)

        n_queries = rng.choice([0, 1, 5, 30])
        # Queries must be issued in non-decreasing start order (the sweep's
        # documented precondition) - generate then sort by start.
        queries = sorted(_random_intervals(rng, n_queries, max_time=55, max_len=3))

        sweep = _DiarizationSweep(diar_segments)
        for start, end in queries:
            expected = _find_overlapping_speaker(start, end, diar_segments)
            actual = sweep.find(start, end)
            assert actual == expected, (
                f"seed={seed} query=({start},{end}) diar={diar_segments} "
                f"expected={expected} actual={actual}"
            )
