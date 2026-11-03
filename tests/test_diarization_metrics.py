"""Analytic DER regressions: independent durations, overlap, label assignment and evaluation masks."""
import importlib
import json
from pathlib import Path

import pytest
from pyannote.core import Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


@pytest.fixture
def der(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    return importlib.import_module("diarization_metrics")


def ann(module, *intervals):
    return module.annotation([dict(start=a, end=b, speaker=label) for a, b, label in intervals], "sample")


def uem(*intervals):
    return Timeline([Segment(a, b) for a, b in intervals], uri="sample")


def test_permuted_speakers_and_overlap_are_perfect(der):
    ref = ann(der, (0, 4, "A"), (2, 6, "B"))
    hyp = ann(der, (0, 4, "Y"), (2, 6, "X"))
    score = der.score_der(ref, hyp, uem((0, 7)))
    assert score["der"] == 0 and score["reference_speaker_seconds"] == 8
    assert score["overlap"]["evaluated_seconds"] == 2
    assert score["overlap"]["reference_speaker_seconds"] == 4
    assert score["speaker_mapping"] == {"X": "B", "Y": "A"}


def test_missed_simultaneous_speaker_is_counted(der):
    score = der.score_der(ann(der, (0, 4, "A"), (2, 6, "B")), ann(der, (0, 4, "X")), uem((0, 7)))
    assert score["missed_speech_seconds"] == 4 and score["der"] == .5
    assert score["overlap"]["missed_speech_seconds"] == 2
    assert score["non_overlap"]["missed_speech_seconds"] == 2


def test_three_simultaneous_speakers_and_extra_hypothesis_speech(der):
    ref = ann(der, (0, 4, "A"), (1, 3, "B"), (2, 3, "C"))
    score = der.score_der(ref, ann(der, (0, 4, "X"), (1, 3, "Y")), uem((0, 4)))
    assert score["der"] == pytest.approx(1 / 7)
    assert score["overlap"]["reference_speaker_seconds"] == 5 and score["overlap"]["der"] == .2
    ref = ann(der, (0, 2, "A"), (0, 2, "B"))
    hyp = ann(der, (0, 2, "X"), (0, 2, "Y"), (0, 2, "Z"))
    score = der.score_der(ref, hyp, uem((0, 2)))
    assert score["overlap"]["false_alarm_seconds"] == 2 and score["der"] == .5


def test_overlap_breakdown_uses_global_mapping_not_optimistic_local_remapping(der):
    ref = ann(der, (0, 10, "A"), (10, 20, "B"), (2, 3, "B"))
    hyp = ann(der, (0, 10, "X"), (10, 20, "Y"), (2, 3, "Z"))
    score = der.score_der(ref, hyp, uem((0, 20)))
    assert score["der"] == pytest.approx(1 / 21)
    assert score["overlap"]["confusion_seconds"] == 1 and score["overlap"]["der"] == .5
    assert score["non_overlap"]["der"] == 0 and score["unmapped_hypothesis_speakers"] == ["Z"]


def test_all_error_components_have_analytic_durations(der):
    ref = ann(der, (0, 4, "A"), (4, 8, "B"))
    hyp = ann(der, (0, 3, "X"), (4, 7, "Y"), (7, 8, "X"), (8, 10, "Y"))
    score = der.score_der(ref, hyp, uem((0, 10)))
    assert score["missed_speech_seconds"] == 1
    assert score["false_alarm_seconds"] == 2
    assert score["confusion_seconds"] == 1
    assert score["correct_seconds"] == 6 and score["der"] == .5


def test_unmapped_original_label_cannot_accidentally_match_reference(der):
    score = der.score_der(ann(der, (0, 10, "A"), (10, 12, "B")),
                          ann(der, (0, 10, "X"), (10, 11, "Y"), (11, 12, "B")), uem((0, 12)))
    assert score["confusion_seconds"] == 1 and score["der"] == pytest.approx(1 / 12)


def test_duplicate_same_speaker_is_union_not_extra_speech(der):
    ref = ann(der, (0, 3, "A"), (0, 3, "A"), (2, 4, "A"))
    score = der.score_der(ref, ann(der, (0, 4, "X"), (1, 3, "X")), uem((0, 4)))
    assert score["der"] == 0 and score["reference_speaker_seconds"] == 4
    assert score["overlap"]["reference_speaker_seconds"] == 0


def test_uem_excludes_unannotated_regions_and_controls_mapping(der):
    ref = ann(der, (0, 10, "A"), (10, 12, "B"))
    hyp = ann(der, (0, 12, "X"), (12, 20, "Y"))
    score = der.score_der(ref, hyp, uem((10, 12)))
    assert score["der"] == 0 and score["speaker_mapping"] == {"X": "B"}
    assert score["evaluated_seconds"] == 2


def test_collar_is_total_width_and_is_removed_once(der):
    score = der.score_der(ann(der, (1, 3, "A")), ann(der, (.8, 3.2, "X")), uem((0, 4)), collar=.5)
    assert score["der"] == 0
    assert score["reference_speaker_seconds"] == 1.5 and score["evaluated_seconds"] == 3


def test_skip_overlap_is_explicit_and_retains_raw_overlap_duration(der):
    score = der.score_der(ann(der, (0, 4, "A"), (2, 6, "B")), ann(der, (0, 4, "X")),
                          uem((0, 7)), skip_overlap=True)
    assert score["reference_overlap_seconds_before_exclusions"] == 2
    assert score["overlap"]["evaluated_seconds"] == 0 and score["overlap"]["der"] is None
    assert score["reference_speaker_seconds"] == 4 and score["policy"]["skip_overlap"] is True


@pytest.mark.parametrize("hypothesis", [(), ((0, 1, "X"),)])
def test_silence_has_no_invented_denominator(der, hypothesis):
    score = der.score_der(ann(der), ann(der, *hypothesis), uem((0, 2)))
    assert score["der"] is None and score["reference_speaker_seconds"] == 0
    assert score["false_alarm_seconds"] == len(hypothesis)


def test_collar_can_exclude_everything_without_failure(der):
    score = der.score_der(ann(der, (0, .1, "A")), ann(der), uem((0, .1)), collar=1)
    assert score["der"] is None and score["evaluated_seconds"] == 0


def test_der_is_not_clamped_at_one(der):
    score = der.score_der(ann(der, (1, 2, "A")), ann(der, (0, 4, "X"), (0, 4, "Y")), uem((0, 4)))
    assert score["false_alarm_seconds"] == 7 and score["der"] == 7


def test_weighted_corpus_der_and_incompatible_policies(der):
    first = der.score_der(ann(der, (0, 1, "A")), ann(der), uem((0, 1)))
    second = der.score_der(ann(der, (0, 9, "A")), ann(der, (0, 9, "X")), uem((0, 9)))
    assert der.aggregate_der([first, second])["der"] == .1
    assert der.aggregate_der([]) is None
    altered = der.score_der(ann(der, (0, 9, "A")), ann(der), uem((0, 9)), collar=.5)
    with pytest.raises(ValueError, match="different DER"):
        der.aggregate_der([first, altered])


@pytest.mark.parametrize("collar,skip", [(0, False), (.5, False), (0, True), (.5, True)])
def test_matches_standard_pyannote_der_and_components_are_additive(der, collar, skip):
    ref = ann(der, (0, 4, "A"), (2, 6, "B"), (7, 9, "A"))
    hyp = ann(der, (.2, 5, "X"), (3, 6.3, "Y"), (7.5, 9.3, "Y"))
    regions = uem((0, 8), (8.5, 10))
    actual = der.score_der(ref, hyp, regions, collar=collar, skip_overlap=skip)
    expected = DiarizationErrorRate(collar=collar, skip_overlap=skip)(ref, hyp, uem=regions, detailed=True)
    assert actual["der"] == pytest.approx(expected["diarization error rate"])
    for key, original in der.COMPONENTS.items():
        assert actual[key] == pytest.approx(expected[original])
        assert actual[key] == pytest.approx(actual["overlap"][key] + actual["non_overlap"][key])


@pytest.mark.parametrize("start,end", [(0, 0), (2, 1), (-1, 1), (True, 2), (0, float("nan")), (0, float("inf"))])
def test_invalid_intervals_are_not_silently_discarded(der, start, end):
    with pytest.raises(ValueError):
        ann(der, (start, end, "A"))


@pytest.mark.parametrize("collar", [-1, float("inf"), float("nan"), True])
def test_invalid_policy_is_rejected(der, collar):
    with pytest.raises(ValueError):
        der.score_der(ann(der), ann(der), uem((0, 1)), collar=collar)


def test_empty_uem_is_rejected(der):
    with pytest.raises(ValueError, match="explicit nonempty"):
        der.score_der(ann(der), ann(der), Timeline())


@pytest.mark.parametrize("row", ["SPEAKER wrong 1 0 1 <NA> <NA> A <NA> <NA>",
                                 "SPEAKER sample 2 0 1 <NA> <NA> A <NA> <NA>",
                                 "SPEAKER sample 1 nan 1 <NA> <NA> A <NA> <NA>",
                                 "SPEAKER sample 1 0 1 <NA> <NA> <NA> <NA> <NA>", "malformed"])
def test_malformed_or_wrong_recording_rttm_is_rejected(der, tmp_path, row):
    path = tmp_path / "reference.rttm"
    path.write_text(row, encoding="utf-8")
    with pytest.raises(ValueError):
        der.read_rttm(path, "sample")


def test_readers_and_cli_preserve_simultaneous_tracks(der, tmp_path):
    ref, hyp, regions, report = [tmp_path / p for p in ("ref.rttm", "diar.json", "test.uem", "score.json")]
    ref.write_text("SPEAKER sample 1 0 2 <NA> <NA> A <NA> <NA>\nSPEAKER sample 1 0 2 <NA> <NA> B <NA> <NA>\n")
    hyp.write_text(json.dumps({"stage": "diarize", "segments": [{"start": 0, "end": 2, "speaker": "X"}]}))
    regions.write_text("sample 1 0 1\nsample 1 0.5 3\n")
    der.main(["--reference", str(ref), "--hypothesis", str(hyp), "--uem", str(regions), "--uri", "sample", "--report", str(report)])
    result = json.loads(report.read_text())
    assert result["der"] == .5 and result["reference_speaker_seconds"] == 4 and result["evaluated_seconds"] == 3
    hyp.write_text(json.dumps({"segments": [{"start": 0, "end": 2, "speaker_id": "X"}]}))
    with pytest.raises(ValueError, match="raw diar"):
        der.read_hypothesis(hyp, "sample")
