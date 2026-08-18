"""Tests for diarrhizer.export.markdown_export."""

from diarrhizer.export.markdown_export import _format_timestamp, export_to_markdown


def test_format_timestamp_zero():
    assert _format_timestamp(0) == "00:00:00"


def test_format_timestamp_seconds_and_minutes():
    assert _format_timestamp(61) == "00:01:01"


def test_format_timestamp_hours():
    assert _format_timestamp(3661) == "01:01:01"


def test_format_timestamp_truncates_fractional_seconds():
    assert _format_timestamp(61.9) == "00:01:01"


def test_export_to_markdown_uses_speaker_mapping():
    segments = [{"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "hello"}]
    config = {"language": "en", "device": "cpu", "speakers": {"Speaker_00": "Ivan"}}
    md = export_to_markdown(segments, config, "input.wav")
    assert "**Ivan:** hello" in md
    assert "Speaker_00" not in md


def test_export_to_markdown_without_mapping_uses_speaker_id():
    segments = [{"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "hello"}]
    config = {"language": "en", "device": "cpu"}
    md = export_to_markdown(segments, config, "input.wav")
    assert "**Speaker_00:** hello" in md
