"""Config-aware cache checks on convert/transcribe/diarize stages."""

import json

from diarrhizer.pipeline.runner import JobContext, PipelineConfig
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
    _write(job_dir / "meta" / "run.json", {"config": {"audio_profile": "raw"}})
    assert ConvertStage().is_cache_valid(job) is True


def test_transcribe_cache_invalid_when_model_changes(tmp_path):
    job_dir = tmp_path / "job"
    _write(job_dir / "audio" / "normalized.wav", b"wav")
    _write(job_dir / "asr" / "transcript.json", {
        "model": "large-v3",
        "compute_type": None,
        "metadata": {
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


def test_diarize_cache_invalid_when_speaker_range_changes(tmp_path):
    job_dir = tmp_path / "job"
    _write(job_dir / "audio" / "normalized.wav", b"wav")
    _write(job_dir / "diar" / "diarization.json", {
        "segments": [],
        "metadata": {"min_speakers": 1, "max_speakers": 10},
    })
    job = _job(tmp_path, job_dir, min_speakers=2, max_speakers=4)
    assert DiarizeStage().is_cache_valid(job) is False
