"""Tests for diarrhizer.export.text_export.

result.txt is the plain counterpart of result.md: same timestamps, same
speaker resolution, no markup and no metadata header beyond input/language.
"""

from diarrhizer.export.text_export import export_to_text
from diarrhizer.pipeline.runner import PipelineConfig


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig(job_id="job", input_file="input.wav", **overrides)


def _body_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("[")]


def test_header_carries_input_path_and_language():
    config = _config(language="ru", device="cpu")
    out = export_to_text([], config, "C:/calls/input.wav")
    assert "Input: C:/calls/input.wav" in out
    assert "Language: ru" in out


def test_segment_line_has_timestamps_speaker_and_text():
    segments = [{"start": 0, "end": 61, "speaker_id": "Speaker_00", "text": "привет"}]
    out = export_to_text(segments, _config(language="ru", device="cpu"), "input.wav")
    assert "[00:00:00 → 00:01:01] Speaker_00: привет" in out


def test_uses_speaker_mapping():
    segments = [{"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "hello"}]
    config = _config(language="en", device="cpu", speakers={"Speaker_00": "Ivan"})
    out = export_to_text(segments, config, "input.wav")
    assert "Ivan: hello" in out
    assert "Speaker_00" not in out


def test_normalizes_pyannote_speaker_ids():
    """Diarization can still hand us SPEAKER_00; the documented form is Speaker_00."""
    segments = [{"start": 0, "end": 1, "speaker_id": "SPEAKER_01", "text": "hello"}]
    out = export_to_text(segments, _config(language="en", device="cpu"), "input.wav")
    assert "Speaker_01: hello" in out
    assert "SPEAKER_01" not in out


def test_has_no_markdown_markup():
    segments = [{"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "hello"}]
    out = export_to_text(segments, _config(language="en", device="cpu"), "input.wav")
    assert "**" not in out
    assert not out.lstrip().startswith("#")


def test_omits_word_level_dump():
    """Word timings live in result.json only - one line per segment here too."""
    segments = [
        {
            "start": 0,
            "end": 2,
            "speaker_id": "Speaker_01",
            "text": "hello there",
            "words": [
                {"start": 0, "end": 1, "word": "hello", "speaker_id": "Speaker_01"},
                {"start": 1, "end": 2, "word": "there", "speaker_id": "Speaker_01"},
            ],
        }
    ]
    out = export_to_text(segments, _config(language="en", device="cpu"), "input.wav")
    assert _body_lines(out) == ["[00:00:00 → 00:00:02] Speaker_01: hello there"]


def test_one_line_per_segment_in_order():
    segments = [
        {"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "first"},
        {"start": 1, "end": 2, "speaker_id": "Speaker_01", "text": "second"},
        {"start": 2, "end": 3, "speaker_id": "Speaker_00", "text": "third"},
    ]
    out = export_to_text(segments, _config(language="en", device="cpu"), "input.wav")
    assert _body_lines(out) == [
        "[00:00:00 → 00:00:01] Speaker_00: first",
        "[00:00:01 → 00:00:02] Speaker_01: second",
        "[00:00:02 → 00:00:03] Speaker_00: third",
    ]


def test_empty_segments_still_produce_a_header():
    out = export_to_text([], _config(language="en", device="cpu"), "input.wav")
    assert "Input: input.wav" in out
    assert _body_lines(out) == []


def test_missing_fields_fall_back_to_defaults():
    """A segment without start/end/speaker_id/text must not raise."""
    out = export_to_text([{}], _config(language="en", device="cpu"), "input.wav")
    assert "[00:00:00 → 00:00:00] Speaker_00: " in out
