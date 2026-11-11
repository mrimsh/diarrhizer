"""Reading old merged artifacts must preserve text without rewriting the source."""

import json

import pytest

from diarrhizer.pipeline.settings import read_segments


def _write(tmp_path, segments):
    path = tmp_path / "segments.json"
    path.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    return path


@pytest.mark.parametrize("start,end", [(1065.176, 1065.156), (1153.36, 1152.1), (1189.066, 1187.926)])
@pytest.mark.parametrize("versioned", [False, True])
def test_legacy_reversed_interval_is_readable_without_losing_text_or_rewriting(tmp_path, caplog, start, end, versioned):
    segments = [{"start": 0, "end": 1, "text": "Before", "speaker_id": "Speaker_00"},
                {"start": start, "end": end, "text": "Original   text!", "speaker_id": "Speaker_01",
                 "extra": "keep", "alignment_warning": "Previous warning",
                 "words": [{"word": "Original", "start": start, "end": end}, {"word": "text!"}]},
                {"start": start + 2, "end": start + 4, "text": "After", "speaker_id": "Speaker_00"}]
    path = _write(tmp_path, segments)
    if versioned:
        path.write_text(json.dumps({"schema_version": 1, "segments": segments}), encoding="utf-8")
    before = path.read_bytes(), path.stat().st_mtime_ns

    result = read_segments(path)

    assert [s["text"] for s in result] == [s["text"] for s in segments]
    assert [s["speaker_id"] for s in result] == [s["speaker_id"] for s in segments]
    assert result[0] == segments[0] and result[2] == segments[2]
    assert result[1]["start"] == result[1]["end"] == start
    assert result[1]["extra"] == "keep"
    assert "Previous warning" in result[1]["alignment_warning"]
    assert str(end) in result[1]["alignment_warning"]
    assert all(w["start"] is None and w["end"] is None for w in result[1]["words"])
    assert [w["word"] for w in result[1]["words"]] == ["Original", "text!"]
    assert "segment 1" in caplog.text
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("start,end", [(None, 1), (1, None), (-1, 0), (1, -1),
                                      (float("nan"), 1), (1, float("inf")), ("1", 2), (False, 1)])
def test_unusable_timing_still_reports_segment_index_and_bounds(tmp_path, start, end):
    path = _write(tmp_path, [{"start": 0, "end": 1, "text": "ok"},
                             {"start": start, "end": end, "text": "bad"}])
    with pytest.raises(ValueError, match="invalid segment timestamps .*segment 1, start=.*end="):
        read_segments(path)


def test_valid_and_zero_duration_results_are_unchanged(tmp_path):
    segments = [{"start": 1, "end": 2, "text": "ok", "speaker_id": "Speaker_00"},
                {"start": 3, "end": 3, "text": "point", "speaker_id": "Speaker_01"}]
    assert read_segments(_write(tmp_path, segments)) == segments


@pytest.mark.parametrize("segment", ["bad", {"start": 0, "end": 1, "text": []},
                                    {"start": 0, "end": 1, "text": "ok", "speaker_id": 3}])
def test_malformed_result_is_not_treated_as_recoverable_timing(tmp_path, segment):
    with pytest.raises(ValueError):
        read_segments(_write(tmp_path, [segment]))
