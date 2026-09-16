"""Tests for diarrhizer.export.markdown_export."""

from diarrhizer.export.markdown_export import _format_timestamp, export_to_markdown
from diarrhizer.pipeline.runner import PipelineConfig


def test_format_timestamp_zero():
    assert _format_timestamp(0) == "00:00:00"


def test_format_timestamp_seconds_and_minutes():
    assert _format_timestamp(61) == "00:01:01"


def test_format_timestamp_hours():
    assert _format_timestamp(3661) == "01:01:01"


def test_format_timestamp_truncates_fractional_seconds():
    assert _format_timestamp(61.9) == "00:01:01"


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig(job_id="job", input_file="input.wav", **overrides)


def test_export_to_markdown_uses_speaker_mapping():
    segments = [{"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "hello"}]
    config = _config(language="en", device="cpu", speakers={"Speaker_00": "Ivan"})
    md = export_to_markdown(segments, config, "input.wav")
    assert "**Ivan:** hello" in md
    assert "Speaker_00" not in md


def test_export_to_markdown_without_mapping_uses_speaker_id():
    segments = [{"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "hello"}]
    config = _config(language="en", device="cpu")
    md = export_to_markdown(segments, config, "input.wav")
    assert "**Speaker_00:** hello" in md


def test_export_to_markdown_omits_word_level_dump():
    """Word timings live in result.json only - result.md stays one line per
    segment, including when a word's speaker differs from the segment's."""
    segments = [
        {
            "start": 0,
            "end": 2,
            "speaker_id": "Speaker_01",
            "text": "hello there",
            "words": [
                {"start": 0, "end": 1, "word": "hello", "speaker_id": "Speaker_00"},
                {"start": 1, "end": 2, "word": "there", "speaker_id": "Speaker_01"},
            ],
        }
    ]
    config = _config(language="en", device="cpu")
    md = export_to_markdown(segments, config, "input.wav")

    assert "[00:00:00 → 00:00:02] **Speaker_01:** hello there" in md
    assert "    - " not in md
    assert "(Speaker_00)" not in md
    # The segment line is the only body line - header lines plus one segment.
    assert [line for line in md.splitlines() if line.startswith("[")] == [
        "[00:00:00 → 00:00:02] **Speaker_01:** hello there"
    ]
