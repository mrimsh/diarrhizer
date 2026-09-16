"""Tests for the speaker-assignment algorithm in diarrhizer.pipeline.stages.merge."""

import json
import random
from types import SimpleNamespace

import pytest

from diarrhizer.pipeline.runner import PipelineConfig
from diarrhizer.pipeline.stages.merge import (
    DEFAULT_MIN_TURN_DURATION,
    MergeStage,
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


def test_pyannote_speaker_ids_are_normalized():
    segments = [{"start": 0, "end": 5, "text": "hi"}]
    diar = [{"start": 0, "end": 5, "speaker": "SPEAKER_01"}]
    result = assign_speakers(segments, [], diar)
    assert result[0]["speaker_id"] == "Speaker_01"


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
        {"start": 2, "end": 3, "word": "there"},
    ]
    diar = [{"start": 0, "end": 10, "speaker": "Speaker_00"}]
    result = assign_speakers(segments, words, diar)
    assert result[0]["words"] == [
        {"start": 0, "end": 1, "word": "hi", "speaker_id": "Speaker_00"},
        {"start": 2, "end": 3, "word": "there", "speaker_id": "Speaker_00"},
    ]


def test_homogeneous_segment_keeps_original_text_and_timings():
    # Nothing to split: the ASR text must survive verbatim rather than being
    # rebuilt from word tokens (which would lose the original spacing).
    segments = [{"start": 0, "end": 10, "text": "Hello,   how are you?"}]
    words = [
        {"start": 1, "end": 2, "word": "Hello,"},
        {"start": 2, "end": 3, "word": "how"},
        {"start": 3, "end": 4, "word": "are"},
        {"start": 4, "end": 5, "word": "you?"},
    ]
    diar = [{"start": 0, "end": 10, "speaker": "Speaker_01"}]
    result = assign_speakers(segments, words, diar)
    assert len(result) == 1
    assert result[0]["text"] == "Hello,   how are you?"
    assert (result[0]["start"], result[0]["end"]) == (0, 10)


def test_segment_speaker_comes_from_its_words_not_whole_segment_overlap():
    # Whole-segment overlap says Speaker_00 (6s vs 4s), but every word lands in
    # Speaker_01's stretch. The label must follow the words, or the segment
    # would contradict its own word list.
    segments = [{"start": 0, "end": 10, "text": "hi there"}]
    words = [
        {"start": 7, "end": 8, "word": "hi"},
        {"start": 8, "end": 9, "word": "there"},
    ]
    diar = [
        {"start": 0, "end": 6, "speaker": "Speaker_00"},
        {"start": 6, "end": 10, "speaker": "Speaker_01"},
    ]
    result = assign_speakers(segments, words, diar)
    assert len(result) == 1
    assert result[0]["speaker_id"] == "Speaker_01"


# --- splitting segments that span a speaker change -------------------------

def test_segment_spanning_speaker_change_is_split_into_two():
    segments = [{"start": 0, "end": 10, "text": "da ladno nastolko shutka"}]
    words = [
        {"start": 0, "end": 1, "word": "da"},
        {"start": 1, "end": 2, "word": "ladno"},
        {"start": 6, "end": 7, "word": "nastolko"},
        {"start": 7, "end": 8, "word": "shutka"},
    ]
    diar = [
        {"start": 0, "end": 5, "speaker": "Speaker_05"},
        {"start": 5, "end": 10, "speaker": "Speaker_06"},
    ]
    result = assign_speakers(segments, words, diar)

    assert [s["speaker_id"] for s in result] == ["Speaker_05", "Speaker_06"]
    assert [s["text"] for s in result] == ["da ladno", "nastolko shutka"]
    # Original segment bounds are preserved at the outer edges.
    assert (result[0]["start"], result[0]["end"]) == (0, 2)
    assert (result[1]["start"], result[1]["end"]) == (6, 10)


def test_split_segments_stay_homogeneous():
    # The whole point: no word may carry a speaker_id its segment doesn't.
    segments = [{"start": 0, "end": 12, "text": "a b c d"}]
    words = [
        {"start": 0, "end": 1, "word": "a"},
        {"start": 4, "end": 5, "word": "b"},
        {"start": 8, "end": 9, "word": "c"},
        {"start": 9, "end": 10, "word": "d"},
    ]
    diar = [
        {"start": 0, "end": 3, "speaker": "Speaker_00"},
        {"start": 3, "end": 7, "speaker": "Speaker_01"},
        {"start": 7, "end": 12, "speaker": "Speaker_00"},
    ]
    result = assign_speakers(segments, words, diar)

    assert [s["speaker_id"] for s in result] == ["Speaker_00", "Speaker_01", "Speaker_00"]
    for seg in result:
        assert {w["speaker_id"] for w in seg["words"]} == {seg["speaker_id"]}


def test_split_segments_are_chronological_and_non_overlapping():
    segments = [{"start": 0, "end": 12, "text": "a b c d"}]
    words = [
        {"start": 0, "end": 1, "word": "a"},
        {"start": 4, "end": 5, "word": "b"},
        {"start": 8, "end": 9, "word": "c"},
        {"start": 9, "end": 10, "word": "d"},
    ]
    diar = [
        {"start": 0, "end": 3, "speaker": "Speaker_00"},
        {"start": 3, "end": 7, "speaker": "Speaker_01"},
        {"start": 7, "end": 12, "speaker": "Speaker_00"},
    ]
    result = assign_speakers(segments, words, diar)

    for seg in result:
        assert seg["start"] <= seg["end"]
    for earlier, later in zip(result, result[1:]):
        assert earlier["end"] <= later["start"]


def test_short_speaker_blip_between_same_speaker_is_absorbed():
    # One 0.2s word flips speaker mid-sentence: diarization jitter, not a turn.
    # The sentence must stay in one piece and the word must be relabelled.
    segments = [{"start": 0, "end": 10, "text": "one two three"}]
    words = [
        {"start": 0, "end": 1, "word": "one"},
        {"start": 5.05, "end": 5.25, "word": "two"},
        {"start": 8, "end": 9, "word": "three"},
    ]
    diar = [
        {"start": 0, "end": 5, "speaker": "Speaker_00"},
        {"start": 5, "end": 5.3, "speaker": "Speaker_01"},
        {"start": 5.3, "end": 10, "speaker": "Speaker_00"},
    ]
    result = assign_speakers(segments, words, diar)

    assert len(result) == 1
    assert result[0]["speaker_id"] == "Speaker_00"
    assert result[0]["text"] == "one two three"
    assert [w["speaker_id"] for w in result[0]["words"]] == ["Speaker_00"] * 3


def test_short_blip_at_segment_edge_is_not_absorbed():
    # Only interior runs are smoothed - a short run at the edge has no evidence
    # on both sides, and may well be the tail of a real turn.
    segments = [{"start": 0, "end": 10, "text": "yes go on"}]
    words = [
        {"start": 0, "end": 0.2, "word": "yes"},
        {"start": 6, "end": 7, "word": "go"},
        {"start": 7, "end": 8, "word": "on"},
    ]
    diar = [
        {"start": 0, "end": 5, "speaker": "Speaker_00"},
        {"start": 5, "end": 10, "speaker": "Speaker_01"},
    ]
    result = assign_speakers(segments, words, diar)

    assert [s["speaker_id"] for s in result] == ["Speaker_00", "Speaker_01"]
    assert [s["text"] for s in result] == ["yes", "go on"]


def test_longer_interior_turn_is_not_absorbed():
    # Same shape as the jitter case but the middle run is a real turn -
    # it must survive as its own segment.
    segments = [{"start": 0, "end": 12, "text": "one two three"}]
    words = [
        {"start": 0, "end": 1, "word": "one"},
        {"start": 4, "end": 6, "word": "two"},
        {"start": 9, "end": 10, "word": "three"},
    ]
    diar = [
        {"start": 0, "end": 3, "speaker": "Speaker_00"},
        {"start": 3, "end": 7, "speaker": "Speaker_01"},
        {"start": 7, "end": 12, "speaker": "Speaker_00"},
    ]
    result = assign_speakers(segments, words, diar)

    assert [s["speaker_id"] for s in result] == ["Speaker_00", "Speaker_01", "Speaker_00"]


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


# --- min_turn_duration ------------------------------------------------------

def _blip_case():
    """A 0.2s interior speaker flip - jitter at the default threshold, a real
    turn once the threshold is lowered below it."""
    segments = [{"start": 0, "end": 10, "text": "one two three"}]
    words = [
        {"start": 0, "end": 1, "word": "one"},
        {"start": 5.05, "end": 5.25, "word": "two"},
        {"start": 8, "end": 9, "word": "three"},
    ]
    diar = [
        {"start": 0, "end": 5, "speaker": "Speaker_00"},
        {"start": 5, "end": 5.3, "speaker": "Speaker_01"},
        {"start": 5.3, "end": 10, "speaker": "Speaker_00"},
    ]
    return segments, words, diar


def test_min_turn_duration_zero_splits_on_every_speaker_change():
    segments, words, diar = _blip_case()
    result = assign_speakers(segments, words, diar, min_turn_duration=0)
    assert [s["speaker_id"] for s in result] == ["Speaker_00", "Speaker_01", "Speaker_00"]


def test_raising_min_turn_duration_absorbs_a_turn_the_default_would_keep():
    # A 2s interior turn survives the 0.4s default but not a 3s threshold.
    segments = [{"start": 0, "end": 12, "text": "one two three"}]
    words = [
        {"start": 0, "end": 1, "word": "one"},
        {"start": 4, "end": 6, "word": "two"},
        {"start": 9, "end": 10, "word": "three"},
    ]
    diar = [
        {"start": 0, "end": 3, "speaker": "Speaker_00"},
        {"start": 3, "end": 7, "speaker": "Speaker_01"},
        {"start": 7, "end": 12, "speaker": "Speaker_00"},
    ]
    assert len(assign_speakers(segments, words, diar)) == 3
    assert len(assign_speakers(segments, words, diar, min_turn_duration=3.0)) == 1


def test_assign_speakers_defaults_to_the_documented_threshold():
    segments, words, diar = _blip_case()
    assert assign_speakers(segments, words, diar) == assign_speakers(
        segments, words, diar, min_turn_duration=DEFAULT_MIN_TURN_DURATION
    )


# --- MergeStage.run() -------------------------------------------------------

def _job(job_dir, **config_overrides):
    config = PipelineConfig(job_id="job", input_file="input.wav", **config_overrides)
    return SimpleNamespace(job_dir=job_dir, config=config)


def test_run_without_diarization_file_defaults_all_segments_to_speaker_00(tmp_path):
    # No diarize stage has run: diar/diarization.json doesn't exist at all.
    # This must not raise - it should behave like an empty diarization input.
    job_dir = tmp_path / "job"
    (job_dir / "asr").mkdir(parents=True)
    (job_dir / "asr" / "transcript.json").write_text(json.dumps({
        "segments": [
            {"start": 0, "end": 1, "text": "hello"},
            {"start": 1, "end": 2, "text": "world"},
        ],
        "words": [],
    }), encoding="utf-8")

    assert not (job_dir / "diar" / "diarization.json").exists()

    result = MergeStage().run(_job(job_dir))

    assert result["status"] == "completed"
    segments_path = job_dir / "merged" / "segments.json"
    assert segments_path.exists()
    output = json.loads(segments_path.read_text(encoding="utf-8"))
    assert [s["speaker_id"] for s in output["segments"]] == ["Speaker_00", "Speaker_00"]


def test_run_without_transcript_file_still_raises_file_not_found(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Transcript not found"):
        MergeStage().run(_job(job_dir))


def test_run_with_diarization_file_present_assigns_real_speakers(tmp_path):
    # Regression guard: presence of diar/diarization.json must still be used
    # (not silently ignored now that its absence is tolerated).
    job_dir = tmp_path / "job"
    (job_dir / "asr").mkdir(parents=True)
    (job_dir / "diar").mkdir(parents=True)
    (job_dir / "asr" / "transcript.json").write_text(json.dumps({
        "segments": [{"start": 0, "end": 5, "text": "hi"}],
        "words": [],
    }), encoding="utf-8")
    (job_dir / "diar" / "diarization.json").write_text(json.dumps({
        "segments": [{"start": 0, "end": 5, "speaker": "Speaker_01"}],
    }), encoding="utf-8")

    result = MergeStage().run(_job(job_dir))

    assert result["status"] == "completed"
    output = json.loads((job_dir / "merged" / "segments.json").read_text(encoding="utf-8"))
    assert output["segments"][0]["speaker_id"] == "Speaker_01"


def _split_job_dir(tmp_path):
    """A job dir whose single ASR segment straddles a speaker change, with a
    0.2s interior blip that the threshold decides the fate of."""
    job_dir = tmp_path / "job"
    (job_dir / "asr").mkdir(parents=True)
    (job_dir / "diar").mkdir(parents=True)
    (job_dir / "asr" / "transcript.json").write_text(json.dumps({
        "segments": [{"start": 0, "end": 10, "text": "one two three"}],
        "words": [
            {"start": 0, "end": 1, "word": "one"},
            {"start": 5.05, "end": 5.25, "word": "two"},
            {"start": 8, "end": 9, "word": "three"},
        ],
    }), encoding="utf-8")
    (job_dir / "diar" / "diarization.json").write_text(json.dumps({
        "segments": [
            {"start": 0, "end": 5, "speaker": "Speaker_00"},
            {"start": 5, "end": 5.3, "speaker": "Speaker_01"},
            {"start": 5.3, "end": 10, "speaker": "Speaker_00"},
        ],
    }), encoding="utf-8")
    return job_dir


def test_run_uses_min_turn_duration_from_config(tmp_path):
    job_dir = _split_job_dir(tmp_path)

    MergeStage().run(_job(job_dir, min_turn_duration=0.0))
    output = json.loads((job_dir / "merged" / "segments.json").read_text(encoding="utf-8"))
    assert output["num_segments"] == 3

    MergeStage().run(_job(job_dir, min_turn_duration=1.0))
    output = json.loads((job_dir / "merged" / "segments.json").read_text(encoding="utf-8"))
    assert output["num_segments"] == 1


def test_run_records_min_turn_duration_in_metadata(tmp_path):
    job_dir = _split_job_dir(tmp_path)
    MergeStage().run(_job(job_dir, min_turn_duration=0.75))
    output = json.loads((job_dir / "merged" / "segments.json").read_text(encoding="utf-8"))
    assert output["metadata"]["min_turn_duration"] == 0.75


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
