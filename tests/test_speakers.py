"""Tests for diarrhizer.export.speakers.resolve_speaker_name."""

from diarrhizer.export.speakers import normalize_speaker_id, resolve_speaker_name


def test_normalize_pyannote_speaker_id():
    assert normalize_speaker_id("SPEAKER_00") == "Speaker_00"
    assert normalize_speaker_id("Speaker_01") == "Speaker_01"
    assert normalize_speaker_id("speaker_2") == "Speaker_02"
    assert normalize_speaker_id("custom") == "custom"


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


def test_resolve_matches_pyannote_id_against_documented_mapping():
    mapping = {"Speaker_00": "Ivan"}
    assert resolve_speaker_name("SPEAKER_00", mapping) == "Ivan"


def test_resolve_matches_mapping_written_with_pyannote_keys():
    mapping = {"SPEAKER_00": "Ivan"}
    assert resolve_speaker_name("Speaker_00", mapping) == "Ivan"
