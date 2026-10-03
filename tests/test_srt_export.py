"""Tests for diarrhizer.export.srt_export (EXPORT:SRT)."""

from diarrhizer.export.srt_export import MAX_CUE_CHARS, MAX_CUE_SECONDS, _srt_time, export_to_srt
from diarrhizer.pipeline.runner import PipelineConfig


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig(job_id="job", input_file="input.wav", **overrides)


def _cues(srt: str) -> list[list[str]]:
    return [block.splitlines() for block in srt.strip().split("\n\n") if block.strip()]


def test_srt_time_format():
    assert _srt_time(0) == "00:00:00,000"
    assert _srt_time(3661.5) == "01:01:01,500"
    assert _srt_time(None) == "00:00:00,000"


def test_one_cue_per_short_segment_with_speaker_name():
    segments = [
        {"start": 0, "end": 1.25, "speaker_id": "Speaker_00", "text": " Привет "},
        {"start": 1.5, "end": 3, "speaker_id": "SPEAKER_01", "text": "Здравствуйте"},
    ]
    cues = _cues(export_to_srt(segments, _config(speakers={"Speaker_00": "Иван"}), "in.wav"))
    assert cues == [
        ["1", "00:00:00,000 --> 00:00:01,250", "Иван: Привет"],
        ["2", "00:00:01,500 --> 00:00:03,000", "Speaker_01: Здравствуйте"],
    ]


def test_empty_segments_are_skipped():
    segments = [
        {"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "  "},
        {"start": 1, "end": 2, "speaker_id": "Speaker_00", "text": "текст"},
    ]
    cues = _cues(export_to_srt(segments, _config(), "in.wav"))
    assert [cue[0] for cue in cues] == ["1"]


def test_long_segment_is_split_at_word_boundaries():
    words = [
        {"start": float(i), "end": float(i) + 0.9, "word": f"w{i}", "speaker_id": "Speaker_00"}
        for i in range(20)
    ]
    segment = {
        "start": 0.0,
        "end": 19.9,
        "speaker_id": "Speaker_00",
        "text": " ".join(w["word"] for w in words),
        "words": words,
    }
    cues = _cues(export_to_srt([segment], _config(), "in.wav"))
    assert len(cues) > 1
    texts = [cue[2].removeprefix("Speaker_00: ") for cue in cues]
    # Every word survives, in order, exactly once.
    assert " ".join(texts) == segment["text"]
    for cue in cues:
        start, end = cue[1].split(" --> ")
        assert start <= end


def test_long_segment_without_word_timings_stays_one_cue():
    text = "x" * (MAX_CUE_CHARS + 10)
    segment = {"start": 0, "end": MAX_CUE_SECONDS * 3, "speaker_id": "Speaker_00", "text": text}
    cues = _cues(export_to_srt([segment], _config(), "in.wav"))
    assert len(cues) == 1
    assert cues[0][2] == f"Speaker_00: {text}"


def test_words_with_missing_timings_are_kept():
    words = [{"start": float(i), "end": float(i) + 0.9, "word": f"w{i}"} for i in range(12)]
    words.insert(5, {"word": "1990", "start": None, "end": None})
    segment = {
        "start": 0.0,
        "end": 12.0,
        "speaker_id": "Speaker_00",
        "text": " ".join(w["word"] for w in words),
        "words": words,
    }
    cues = _cues(export_to_srt([segment], _config(), "in.wav"))
    assert " ".join(c[2].removeprefix("Speaker_00: ") for c in cues) == segment["text"]
