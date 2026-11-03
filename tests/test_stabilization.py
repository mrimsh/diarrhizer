"""Data-integrity regressions from the A1–A5 audit, using only synthetic jobs."""

import json
import os
import sys

import pytest

from diarrhizer.adapters.transcript_result import extract_transcript_fields
from diarrhizer.export.srt_export import export_to_srt
from diarrhizer.pipeline import runner
from diarrhizer.pipeline.cache import source_fingerprint
from diarrhizer.pipeline.runner import JobContext, PipelineConfig, run_pipeline
from diarrhizer.pipeline.settings import read_job_settings, recorded_stage_names
from diarrhizer.pipeline.stages.convert import ConvertStage
from diarrhizer.pipeline.stages.export import ExportStage
from diarrhizer.pipeline.stages.merge import assign_speakers
from diarrhizer.summary.providers import default_key_env


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("names", [("Сервис один", "Сервис два"), ("a-b", "a b")])
def test_generated_key_variables_do_not_collide(names):
    first, second = map(default_key_env, names)
    assert first != second
    assert first == default_key_env(names[0])
    assert default_key_env(names[0], [first.lower()]) != first


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("bad_times", [{}, {"start": None, "end": None},
                                      {"start": float("nan"), "end": 12},
                                      {"start": 12, "end": float("inf")},
                                      {"start": 13, "end": 11}])
def test_partial_alignment_preserves_every_token_and_unknown_times(position, bad_times):
    words = [{"start": 10, "end": 11, "word": "first"},
             {"start": 11, "end": 12, "word": "2026"},
             {"start": 12, "end": 14, "word": "second"}]
    words[position] = {"word": words[position]["word"], **bad_times}
    normalized = extract_transcript_fields({"segments": [
        {"start": 10, "end": 14, "text": "first 2026 second", "words": words}]})
    assert normalized["words"][position]["start"] is None
    assert normalized["words"][position]["segment_index"] == 0
    diar = [{"start": 10, "end": 12, "speaker": "Speaker_00"},
            {"start": 12, "end": 14, "speaker": "Speaker_01"}]
    result = assign_speakers(normalized["segments"], normalized["words"], diar)
    assert [seg["text"] for seg in result] == ["first 2026 second"]
    assert result[0]["alignment_warning"]
    assert [word["word"] for word in result[0]["words"]] == ["first", "2026", "second"]


def test_legacy_flat_words_cannot_remove_missing_token_in_merge_or_srt():
    text = "first " + "2026 " * 20 + "second"
    segments = [{"start": 10, "end": 30, "text": text}]
    words = [{"start": 10, "end": 11, "word": "first"},
             {"word": "2026"}, {"start": 28, "end": 30, "word": "second"}]
    result = assign_speakers(segments, words, [{"start": 10, "end": 20, "speaker": "Speaker_00"},
                                             {"start": 20, "end": 30, "speaker": "Speaker_01"}])
    assert result[0]["text"] == text
    assert text in export_to_srt(result, PipelineConfig("job", "fake.wav"), "fake.wav")


def test_unordered_word_times_keep_original_text():
    segments = [{"start": 0, "end": 10, "text": "a b"}]
    words = [{"start": 8, "end": 9, "word": "a"}, {"start": 1, "end": 2, "word": "b"}]
    result = assign_speakers(segments, words, [{"start": 0, "end": 5, "speaker": "Speaker_00"},
                                            {"start": 5, "end": 10, "speaker": "Speaker_01"}])
    assert result[0]["text"] == "a b" and result[0]["alignment_warning"]


@pytest.mark.parametrize("replacement", ["same_path", "different_path", "relocated_same_bytes"])
def test_convert_cache_verifies_content_even_with_old_mtime(tmp_path, replacement):
    source = tmp_path / "old.wav"
    source.write_bytes(b"old source")
    job_dir = tmp_path / "job"
    write(job_dir / "meta" / "run.json", {"config": {"audio_profile": "raw"},
          "source_fingerprint": source_fingerprint(source)})
    (job_dir / "audio").mkdir()
    (job_dir / "audio" / "normalized.wav").write_bytes(b"normalized")
    original_stat = source.stat()
    if replacement != "same_path":
        source = tmp_path / "new.wav"
    source.write_bytes(b"old source" if replacement == "relocated_same_bytes" else b"new source")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    job = JobContext(source, job_dir, PipelineConfig("job", str(source)))
    assert not ConvertStage().is_cache_valid(job)


def test_legacy_conversion_without_identity_is_not_trusted(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"original")
    job_dir = tmp_path / "job"
    write(job_dir / "meta" / "run.json", {"input_path": str(source), "config": {"audio_profile": "raw"}})
    (job_dir / "audio").mkdir()
    (job_dir / "audio" / "normalized.wav").write_bytes(b"normalized")
    assert not ConvertStage().is_cache_valid(JobContext(source, job_dir, PipelineConfig("job", str(source))))


def seed_job(tmp_path):
    job = tmp_path / "job"
    write(job / "meta" / "run.json", {"input_path": str(tmp_path / "gone.wav"), "pipeline_config": {
        "language": "ru", "device": "cpu", "asr_model": "small", "min_speakers": 2, "max_speakers": 3,
        "asr_initial_prompt": "Names", "stages": ["convert", "transcribe", "merge", "export"]}})
    write(job / "merged" / "segments.json", {"segments": [
        {"start": 0, "end": 1, "text": "hello", "speaker_id": "Speaker_00"}]})
    write(job / "speakers.json", {"Speaker_00": "Alice"})
    return job


def test_resume_reexport_keeps_config_names_and_cache(tmp_path):
    job = seed_job(tmp_path)
    first = run_pipeline(job_dir=job, out_dir=tmp_path, stages=[ExportStage()])
    data = json.loads((job / "export" / "result.json").read_text(encoding="utf-8"))
    assert data["metadata"]["config"] == {"language": "ru", "device": "cpu", "min_speakers": 2, "max_speakers": 3}
    assert data["segments"][0]["speaker_name"] == "Alice"
    second = run_pipeline(job_dir=job, out_dir=tmp_path, stages=[ExportStage()])
    assert first["stages"][0]["status"] == "completed"
    assert second["stages"] == [{"stage": "export", "status": "cached"}]
    run_pipeline(job_dir=job, out_dir=tmp_path, stages=[ExportStage()], speakers={})
    assert json.loads((job / "export" / "result.json").read_text(encoding="utf-8"))["metadata"]["speakers"] == {}
    assert read_job_settings(job)["speakers"] == {}
    run_pipeline(job_dir=job, out_dir=tmp_path, stages=[ExportStage()], speakers={"Speaker_00": "Bob"})
    assert read_job_settings(job)["speakers"] == {"Speaker_00": "Bob"}


def test_resume_explicit_overrides_and_saved_stage_list(tmp_path, monkeypatch):
    job = seed_job(tmp_path)
    captured = {}
    monkeypatch.setattr(runner, "_run_pipeline", lambda **kwargs: captured.update(kwargs) or kwargs)
    monkeypatch.setattr(runner, "build_stages", lambda names: list(names))
    run_pipeline(job_dir=job, out_dir=tmp_path, asr_model="base", asr_initial_prompt=None)
    assert captured["stages"] == ["convert", "transcribe", "merge", "export"]
    assert captured["asr_model"] == "base" and captured["asr_initial_prompt"] is None
    assert captured["language"] == "ru" and captured["min_speakers"] == 2


@pytest.mark.parametrize("payload", [[], {"pipeline_config": []}, {"pipeline_config": "broken"},
                                     {"pipeline_config": {"min_speakers": "two"}},
                                     {"pipeline_config": {"asr_beam_size": float("nan")}}])
def test_malformed_recorded_configuration_is_rejected(tmp_path, payload):
    write(tmp_path / "meta" / "run.json", payload)
    with pytest.raises(ValueError):
        read_job_settings(tmp_path)


def test_cli_resume_passes_only_explicit_settings(tmp_path, monkeypatch):
    from diarrhizer import cli
    job = seed_job(tmp_path)
    captured = {}
    monkeypatch.setattr(cli, "load_project_env", lambda: None)
    monkeypatch.setattr(cli, "_configure_logging", lambda: None)
    def run(**kwargs):
        captured.update(kwargs)
        return {"job_id": "job", "job_dir": str(job)}
    monkeypatch.setattr(runner, "run_pipeline", run)
    monkeypatch.setattr(sys, "argv", ["diarrhizer", "run", "--job-dir", str(job), "--asr-model", "base"])
    assert cli.main() == 0
    assert captured["asr_model"] == "base"
    assert "language" not in captured and "speakers" not in captured and "audio_format" not in captured
    assert [stage.NAME for stage in captured["stages"]] == list(recorded_stage_names(job))


def test_malformed_speakers_file_has_readable_cli_error(tmp_path, monkeypatch, capsys):
    from diarrhizer import cli
    bad = tmp_path / "speakers.json"
    bad.write_text("{invalid", encoding="utf-8")
    monkeypatch.setattr(cli, "load_project_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["diarrhizer", "run", "missing.wav", "--speakers", str(bad)])
    assert cli.main() == 1
    assert "Cannot read speakers file" in capsys.readouterr().err


@pytest.mark.parametrize("cached", [True, False])
def test_speaker_backup_happens_only_when_diarization_really_runs(tmp_path, cached):
    job = seed_job(tmp_path)
    write(job / "export" / "result.json", {"metadata": {"speakers": {"Speaker_00": "Alice"}}})
    class Diarization:
        NAME = "diarize"
        def is_cache_valid(self, context):
            return cached
        def get_output_paths(self, directory):
            return {}
        def run(self, context):
            assert json.loads((job / "speakers.json").read_text(encoding="utf-8")) == {}
            assert context.config.speakers is None
            return {"stage": self.NAME, "status": "completed"}
    run_pipeline(job_dir=job, out_dir=tmp_path, stages=[Diarization()])
    assert read_job_settings(job)["speakers"] == ({"Speaker_00": "Alice"} if cached else {})
    assert (job / "speakers.before-rediarize.json").exists() is not cached
    assert ("diarize" in recorded_stage_names(job)) is not cached


def test_programmatic_hotwords_setting_fails_before_asr(tmp_path):
    from diarrhizer.pipeline.stages.transcribe import TranscribeStage
    config = PipelineConfig("job", "synthetic.wav", asr_hotwords_file="words.txt")
    with pytest.raises(ValueError, match="asr_hotwords_file is unsupported"):
        TranscribeStage().run(JobContext(tmp_path / "synthetic.wav", tmp_path, config))
