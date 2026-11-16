"""Config-aware cache checks on convert/transcribe/diarize stages."""

import json
import dataclasses

import pytest

from diarrhizer.pipeline.runner import JobContext, PipelineConfig
from diarrhizer.pipeline.cache import source_fingerprint
from diarrhizer.pipeline.stages.convert import ConvertStage
from diarrhizer.pipeline.stages.diarize import DiarizeStage
from diarrhizer.pipeline.stages.transcribe import TranscribeStage


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (bytes, bytearray)):
        path.write_bytes(data)
    elif isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _job(tmp_path, job_dir, **config_kw) -> JobContext:
    src = tmp_path / "input.wav"
    if not src.exists():
        src.write_bytes(b"src")
    config = PipelineConfig(job_id=job_dir.name, input_file=str(src), **config_kw)
    return JobContext(input_path=src, job_dir=job_dir, config=config)


def test_convert_cache_invalid_when_audio_profile_changes(tmp_path):
    job_dir = tmp_path / "job"
    job = _job(tmp_path, job_dir, audio_profile="voice-call")
    _write(job_dir / "audio" / "normalized.wav", b"wav")
    _write(job_dir / "meta" / "run.json", {"config": {"audio_profile": "raw"}})
    assert ConvertStage().is_cache_valid(job) is False


def test_convert_cache_valid_when_profile_matches(tmp_path):
    job_dir = tmp_path / "job"
    job = _job(tmp_path, job_dir, audio_profile="raw")
    _write(job_dir / "audio" / "normalized.wav", b"wav")
    _write(job_dir / "meta" / "run.json", {"config": {"audio_profile": "raw"},
                                           "source_fingerprint": source_fingerprint(job.input_path)})
    assert ConvertStage().is_cache_valid(job) is True


def test_transcribe_cache_invalid_when_model_changes(tmp_path):
    job_dir = tmp_path / "job"
    _write(job_dir / "audio" / "normalized.wav", b"wav")
    _write(job_dir / "asr" / "transcript.json", {
        "model": "large-v3",
        "compute_type": None,
        "metadata": {
            "cache_version": TranscribeStage.CACHE_VERSION,
            "asr_profile": "default", "glossary_sha256": None, "initial_prompt_sha256": None,
            "vad_onset": .5, "vad_offset": .363, "chunk_size": 30,
            "language_setting": "auto",
            "beam_size": 5,
            "temperature": 0.0,
            "condition_on_previous_text": True,
            "vad_filter": True,
            "vad_min_silence_ms": 1000,
        },
    })
    job = _job(tmp_path, job_dir, asr_model="base")
    assert TranscribeStage().is_cache_valid(job) is False


def test_transcribe_cache_valid_when_asr_config_matches(tmp_path):
    job_dir = tmp_path / "job"
    _write(job_dir / "audio" / "normalized.wav", b"wav")
    _write(job_dir / "asr" / "transcript.json", {
        "model": "large-v3",
        "compute_type": None,
        "metadata": {
            "cache_version": TranscribeStage.CACHE_VERSION,
            "asr_profile": "default", "glossary_sha256": None, "initial_prompt_sha256": None,
            "vad_onset": .5, "vad_offset": .363, "chunk_size": 30,
            "language_setting": "auto",
            "beam_size": 5,
            "temperature": 0.0,
            "condition_on_previous_text": True,
            "vad_filter": True,
            "vad_min_silence_ms": 1000,
        },
    })
    job = _job(tmp_path, job_dir, asr_model="large-v3")
    assert TranscribeStage().is_cache_valid(job) is True
    path = job_dir / "asr" / "transcript.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["metadata"]["cache_version"] = 2  # Before validated alignment loading/provenance.
    _write(path, data)
    assert TranscribeStage().is_cache_valid(job) is False
    del data["metadata"]["cache_version"]
    _write(path, data)
    assert TranscribeStage().is_cache_valid(job) is False


def test_diarize_cache_invalid_when_speaker_range_changes(tmp_path):
    job_dir = tmp_path / "job"
    _write(job_dir / "audio" / "normalized.wav", b"wav")
    _write(job_dir / "diar" / "diarization.json", {
        "segments": [],
        "metadata": {"min_speakers": 1, "max_speakers": 10},
    })
    job = _job(tmp_path, job_dir, min_speakers=2, max_speakers=4)
    assert DiarizeStage().is_cache_valid(job) is False


def test_transcribe_persists_alignment_model_provenance(tmp_path, monkeypatch):
    from types import SimpleNamespace

    job_dir = tmp_path / "job"
    _write(job_dir / "audio" / "normalized.wav", b"wav")
    info = {"name": "test/model", "revision": "synthetic-revision", "backend": "huggingface",
            "loading": {"all_keys_matched": True, "renamed_tensors_verified": 2}}
    unloaded = []
    adapter = SimpleNamespace(transcribe=lambda **kw: {"segments": [], "alignment_model": info,
                                                       "alignment_status": "aligned"},
                              unload=lambda: unloaded.append(True))
    monkeypatch.setattr("diarrhizer.pipeline.stages.transcribe.WhisperXAdapter", lambda **kw: adapter)
    TranscribeStage().run(_job(tmp_path, job_dir))
    data = json.loads((job_dir / "asr/transcript.json").read_text(encoding="utf-8"))
    assert data["metadata"]["alignment_model"] == info
    assert data["metadata"]["cache_version"] == 4 and unloaded == [True]


@pytest.mark.parametrize("changed", [
    {"asr_profile": "default"}, {"asr_glossary": "Другое имя"},
    {"asr_initial_prompt": "Другой prompt"}, {"asr_vad_onset": .6},
    {"asr_vad_offset": .2}, {"asr_chunk_size": 12},
])
def test_profile_glossary_and_real_vad_invalidate_asr_cache(tmp_path, monkeypatch, changed):
    from types import SimpleNamespace
    from diarrhizer.asr_settings import compose_asr_prompt

    job_dir = tmp_path / "job"
    _write(job_dir / "audio/normalized.wav", b"synthetic")
    calls = []
    def make_adapter(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(transcribe=lambda **kw: {"text": "Да.", "segments": [], "words": []},
                               unload=lambda: None)
    monkeypatch.setattr("diarrhizer.pipeline.stages.transcribe.WhisperXAdapter", make_adapter)
    job = _job(tmp_path, job_dir, asr_profile="ru-call", asr_glossary="Анна\nAPI",
               asr_initial_prompt="Рабочая встреча.", asr_vad_onset=.4, asr_vad_offset=.3, asr_chunk_size=18)
    stage = TranscribeStage()
    stage.run(job)
    assert stage.is_cache_valid(job)
    assert calls[0]["initial_prompt"] == compose_asr_prompt("ru-call", "Рабочая встреча.", "Анна\nAPI")
    assert (calls[0]["vad_onset"], calls[0]["vad_offset"], calls[0]["chunk_size"]) == (.4, .3, 18)
    if changed == {"asr_profile": "default"}:
        data_path = job_dir / "asr/transcript.json"
        original = json.loads(data_path.read_text(encoding="utf-8"))
        for key in ("initial_prompt_sha256", "asr_profile", "glossary_sha256", "vad_onset", "vad_offset", "chunk_size"):
            incomplete = {**original, "metadata": {k: v for k, v in original["metadata"].items() if k != key}}
            _write(data_path, incomplete)
            assert not stage.is_cache_valid(job), key
        _write(data_path, original)
    job.config = dataclasses.replace(job.config, **changed)
    assert not stage.is_cache_valid(job)
