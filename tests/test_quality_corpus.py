"""Quality-tool regressions run without downloaded models or installed speech voices."""
import hashlib
import importlib
import json
import wave
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def quality(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    return tuple(importlib.import_module(name) for name in
                 ("benchmark_asr", "benchmark_suite", "build_quality_corpus"))


def marker(word, start=None, speaker=None):
    return {"word": word, "start": start, "speaker_id": speaker}


def test_timing_reports_only_exact_matches_with_coverage(quality):
    benchmark, _, _ = quality
    ref = [marker("Привет,", 0), marker("мир", 1), marker("друг", 2), marker("снова", 3)]
    hyp = [marker("привет", .1), marker("иной", 1), marker("друг", 2.3), marker("снова", float("nan"))]
    result = benchmark.word_metrics(ref, hyp)
    assert result["matched_words"] == 3
    assert result["lexical_coverage"] == .75 and result["timing_coverage"] == .5
    assert result["onset_mae_seconds"] == pytest.approx(.2)
    assert result["onset_p95_seconds"] == pytest.approx(.3)
    assert benchmark.word_metrics([], hyp)["timing_coverage"] is None
    assert benchmark.word_metrics(ref, [marker("lost")])["onset_mae_seconds"] is None


@pytest.mark.parametrize("invalid", [None, True, -1, float("inf"), "1"])
def test_invalid_word_time_is_not_scored(quality, invalid):
    benchmark, _, _ = quality
    assert benchmark.word_metrics([marker("a", 0)], [marker("a", invalid)])["timed_matches"] == 0
    assert benchmark.missing_times([{"start": invalid, "end": 2}]) == 1


def test_speaker_permutation_and_missing_labels_are_explicit(quality):
    benchmark, _, _ = quality
    ref = [marker("one", 0, "A"), marker("two", 1, "B"), marker("three", 2, "A")]
    hyp = [marker("one", 0, "S1"), marker("two", 1, "S0"), marker("three", 2)]
    result = benchmark.word_metrics(ref, hyp, speakers=True)
    assert result["speaker_mapping"] == {"S0": "B", "S1": "A"}
    assert result["speaker_error_rate"] == pytest.approx(1 / 3)
    assert result["speaker_scored_words"] == 3
    assert benchmark.word_metrics(ref, hyp)["speaker_error_rate"] is None
    for item in hyp:
        item["speaker_id"] = None
    assert benchmark.word_metrics(ref, hyp, speakers=True)["speaker_error_rate"] == 1


def test_edit_matching_handles_insertions_deletions_and_repeated_words(quality):
    benchmark, _, _ = quality
    ref = [marker("a", 0), marker("b", 1), marker("c", 2), marker("a", 3)]
    hyp = [marker("extra"), marker("a", .1), marker("b", 1.1), marker("a", 3.1)]
    pairs, total = benchmark.matched_words(ref, hyp)
    assert total == 4 and [(a["start"], b["start"]) for a, b in pairs] == [(0, .1), (1, 1.1), (3, 3.1)]


def test_ambiguous_edit_alignment_has_stable_diagonal_tie_order(quality):
    benchmark, _, _ = quality
    pairs, _ = benchmark.matched_words([marker(w) for w in "a b a".split()],
                                       [marker(w) for w in "extra a a".split()])
    assert len(pairs) == 1  # Two substitutions tie with insertion+deletion; diagonal wins.


def test_aggregate_uses_edit_denominators_and_keeps_failures(quality):
    _, suite, _ = quality
    def run(edits, words):
        return {"model": "tiny", "status": "completed", "wall_seconds": 1, "audio_seconds": 2,
                "metrics": {"word_edits": edits, "reference_words": words, "char_edits": edits,
                            "reference_chars": words}}
    runs = [run(1, 1), run(0, 9), {**run(0, 0), "metrics": None}, {"model": "tiny", "status": "failed"}]
    result = suite.aggregate(runs)["tiny"]
    assert result["wer"] == result["cer"] == .1
    assert result["completed"] == 3 and result["failed"] == 1 and result["rtf"] == .5


def manifest(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"synthetic")
    case = {"id": "sample", "audio": audio.name, "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            "reference_text": "one", "reference_words": [marker("one", 0)], "language": "en"}
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"schema_version": 1, "cases": [case]}), encoding="utf-8")
    return path, case


@pytest.mark.parametrize("damage", ["hash", "escape", "duplicate", "unknown", "schema"])
def test_suite_rejects_invalid_corpus_before_inference(quality, tmp_path, damage):
    _, suite, _ = quality
    path, case = manifest(tmp_path)
    if damage == "hash":
        case["sha256"] = "0" * 64
    elif damage == "escape":
        case["audio"] = "../outside.wav"
    cases = [case, case] if damage == "duplicate" else [case]
    path.write_text(json.dumps({"schema_version": 9 if damage == "schema" else 1, "cases": cases}), encoding="utf-8")
    with pytest.raises(ValueError):
        suite.select_cases(path, ["missing"] if damage == "unknown" else None)


def test_suite_retains_success_before_failed_repeat(quality, tmp_path, monkeypatch):
    _, suite, _ = quality
    path, _ = manifest(tmp_path)
    calls = []
    def benchmark(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic failure")
        return {"metrics": None, "wall_seconds": 1, "audio_seconds": 2,
                "export_preserves_normalized_text": True, "asr_metadata": {"alignment_status": "aligned"}}
    monkeypatch.setattr(suite, "run_benchmark", benchmark)
    report = tmp_path / "report.json"
    assert suite.main([str(path), "--out", str(tmp_path / "out"), "--report", str(report), "--repeat", "2"]) == 1
    data = json.loads(report.read_text(encoding="utf-8"))
    assert [r["status"] for r in data["runs"]] == ["completed", "failed"]
    assert "synthetic failure" in data["runs"][1]["error"]


@pytest.mark.parametrize("reference,hypothesis,scored,insertions", [("", "", True, 0), ("", "ghost", True, 1),
                                                                    ("one two", "one", False, None)])
def test_silence_and_overlap_have_no_misleading_wer(quality, tmp_path, monkeypatch, reference, hypothesis, scored, insertions):
    benchmark, _, _ = quality
    for folder in ("audio", "asr", "export"):
        (tmp_path / folder).mkdir()
    audio = tmp_path / "audio" / "normalized.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000)
    transcript = {"segments": [{"text": hypothesis, "words": []}]}
    for name in ("asr/transcript.json", "export/result.json"):
        (tmp_path / name).write_text(json.dumps(transcript), encoding="utf-8")
    monkeypatch.setattr(benchmark, "run_pipeline", lambda **kwargs: {"job_dir": str(tmp_path)})
    result = benchmark.run_benchmark(audio, reference, tmp_path, score_text=scored)
    assert result["performance"]["wall_seconds"] == result["wall_seconds"]
    assert result["performance"]["ram"]["peak_rss_bytes"] > 0
    assert result["metrics"] is None and result["word_metrics"] is None
    assert result["silence_insertions"] == insertions
    assert bool(result["unscored_reason"]) is (not scored)


def test_composition_keeps_onsets_gaps_and_speakers(quality):
    _, _, corpus = quality
    units = {name: (np.ones(1600) * .1, {"text": name, "voice": name, "words": [marker(name, .02)]})
             for name in ("first", "second")}
    pcm, reference = corpus.compose({"parts": [{"unit": "first"}, {"unit": "second", "gap_before": 2, "speaker": "B"}]}, units)
    assert len(pcm) == 51200 and not pcm[10000:40000].any()
    assert [w["start"] for w in reference["reference_words"]] == pytest.approx([.52, 2.62])
    assert [w["speaker_id"] for w in reference["reference_words"]] == ["A", "B"]
    assert reference["reference_text"] == "first second"


def test_noise_stereo_and_silence_are_deterministic(quality):
    _, _, corpus = quality
    units = {"a": (np.ones(1600) * .1, {"text": "a", "voice": "test", "words": [marker("a", 0)]})}
    case = {"parts": [{"unit": "a"}], "snr_db": 12, "seed": 1729, "gain_db": -18, "stereo": True}
    first, _ = corpus.compose(case, units)
    second, _ = corpus.compose(case, units)
    assert np.array_equal(first, second)
    assert first.shape[1] == 2 and np.max(np.abs(first[:, 0] - 2 * first[:, 1])) <= 1
    different, _ = corpus.compose({**case, "seed": 1730}, units)
    assert not np.array_equal(first, different)
    silence, reference = corpus.compose({"parts": [], "duration": 8}, {})
    assert len(silence) == 128000 and not silence.any() and reference["reference_words"] == []
