"""ASR hints and supported VAD controls: no models, network or private recordings."""

import json

import pytest

from diarrhizer.asr_settings import RU_CALL_PROMPT, compose_asr_prompt, validate_vad_options
from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.settings import read_job_settings


@pytest.mark.parametrize("prompt", [None, "", "  Existing prompt.\n", "Имена: Анна, GitHub"])
def test_default_profile_preserves_existing_custom_prompt(prompt):
    assert compose_asr_prompt(initial_prompt=prompt) == prompt


def test_profile_sample_custom_prompt_and_glossary_have_stable_order():
    result = compose_asr_prompt("ru-call", "  Обсуждаем API. ", "\nАнна\n.NET, GitHub\n\n")
    assert result == RU_CALL_PROMPT + "\n\nОбсуждаем API.\n\nТермины и имена: Анна, .NET, GitHub"
    assert compose_asr_prompt("ru-call") == RU_CALL_PROMPT
    assert compose_asr_prompt(glossary="API") == "Термины и имена: API"


@pytest.mark.parametrize("kwargs", [{"profile": "unknown"}, {"initial_prompt": 3}, {"glossary": ["API"]}])
def test_invalid_prompt_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        compose_asr_prompt(**kwargs)


@pytest.mark.parametrize("onset,offset,chunk", [
    (0, .3, 30), (1, .3, 30), (float("nan"), .3, 30), (.5, float("inf"), 30),
    (.5, 0, 30), (.3, .4, 30), (True, .3, 30), (.5, .3, 0), (.5, .3, 31),
    (.5, .3, 15.0), (.5, .3, True),
])
def test_invalid_vad_controls_are_rejected(onset, offset, chunk):
    with pytest.raises(ValueError):
        validate_vad_options(onset, offset, chunk)


def test_pipeline_rejects_invalid_vad_before_creating_job_artifacts(tmp_path):
    source = tmp_path / "call.wav"
    source.write_bytes(b"synthetic")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="offset"):
        run_pipeline(input_path=source, out_dir=out, stages=[], asr_vad_onset=.2, asr_vad_offset=.3)
    assert not out.exists()


def test_saved_profile_glossary_and_vad_restore_with_explicit_overrides(tmp_path, monkeypatch):
    from diarrhizer.pipeline import runner

    job = tmp_path / "job"
    (job / "meta").mkdir(parents=True)
    saved = {"asr_profile": "ru-call", "asr_glossary": "Анна\n.NET", "asr_initial_prompt": "API.",
             "asr_vad_onset": .4, "asr_vad_offset": .25, "asr_chunk_size": 18}
    (job / "meta/run.json").write_text(json.dumps({"pipeline_config": saved}), encoding="utf-8")
    captured = []
    monkeypatch.setattr(runner, "_run_pipeline", lambda **kw: captured.append(kw) or {})
    run_pipeline(job_dir=job, out_dir=tmp_path, stages=[])
    assert all(captured[-1][key] == value for key, value in saved.items())
    run_pipeline(job_dir=job, out_dir=tmp_path, stages=[], asr_profile="default", asr_glossary=None,
                 asr_vad_onset=.6, asr_chunk_size=30)
    assert captured[-1]["asr_profile"] == "default" and captured[-1]["asr_glossary"] is None
    assert captured[-1]["asr_vad_onset"] == .6 and captured[-1]["asr_vad_offset"] == .25
    assert captured[-1]["asr_chunk_size"] == 30


@pytest.mark.parametrize("saved", [{"asr_vad_onset": 0}, {"asr_chunk_size": 31},
                                   {"asr_profile": "other"}, {"asr_glossary": []}])
def test_corrupt_recorded_asr_settings_are_rejected(tmp_path, saved):
    with pytest.raises(ValueError):
        read_job_settings(tmp_path, {"pipeline_config": saved})


def test_conversion_and_stage_rerun_record_profile_glossary_and_vad_contents(tmp_path):
    import dataclasses
    from pathlib import Path
    from types import SimpleNamespace
    from diarrhizer.pipeline.runner import JobContext, PipelineConfig, _record_stage_settings
    from diarrhizer.pipeline.stages.convert import ConvertStage

    source = tmp_path / "source.wav"
    source.write_bytes(b"synthetic input")
    saved = {"asr_profile": "ru-call", "asr_glossary": "Анна\nAPI", "asr_vad_onset": .45,
             "asr_vad_offset": .3, "asr_chunk_size": 18}
    config = PipelineConfig(job_id="job", input_file=str(source), **saved)
    job = JobContext(source, tmp_path / "job", config)
    stage = ConvertStage()
    stage._ffmpeg_adapter = SimpleNamespace(convert_to_wav=lambda **kw: Path(kw["output_path"]).write_bytes(b"WAV"))
    stage.run(job)
    recorded = read_job_settings(job.job_dir)
    assert all(recorded[key] == value for key, value in saved.items())
    changed = dataclasses.replace(config, asr_profile="default", asr_glossary=None, asr_chunk_size=12)
    _record_stage_settings(job.job_dir, "transcribe", changed)
    recorded = read_job_settings(job.job_dir)
    assert recorded["asr_profile"] == "default" and recorded["asr_glossary"] is None
    assert recorded["asr_chunk_size"] == 12 and recorded["asr_vad_onset"] == .45
