"""Tests for diarrhizer.export.speakers.resolve_speaker_name."""

from diarrhizer.export.speakers import resolve_speaker_name


def test_no_mapping_returns_speaker_id():
    assert resolve_speaker_name("Speaker_00", None) == "Speaker_00"


def test_empty_mapping_returns_speaker_id():
    assert resolve_speaker_name("Speaker_00", {}) == "Speaker_00"


def test_mapped_speaker_returns_display_name():
    mapping = {"Speaker_00": "Ivan", "Speaker_01": "Maria"}
    assert resolve_speaker_name("Speaker_01", mapping) == "Maria"


def test_unmapped_speaker_in_nonempty_mapping_returns_speaker_id():
    mapping = {"Speaker_00": "Ivan"}
    assert resolve_speaker_name("Speaker_05", mapping) == "Speaker_05"
