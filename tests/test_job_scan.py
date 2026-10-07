"""Tests for diarrhizer_gui.job_scan: history rows and GUI:JOB_PLANS.

job_scan is Qt-free (stdlib + diarrhizer.audio_formats), so these run
without a display.
"""

import json
from pathlib import Path

from diarrhizer.audio_formats import AudioFormat
from diarrhizer_gui.job_scan import (
    _parse_job_date,
    rediarize_plan,
    resume_plan,
    scan_jobs,
)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _job(tmp_path: Path, *, stages_done, stages=None, audio_format=None, extra_config=None) -> Path:
    job_dir = tmp_path / "out" / "call_20260102_030405"
    config = {
        "language": "ru",
        "device": "cpu",
        "min_speakers": 2,
        "max_speakers": 3,
        "asr_model": "small",
        "asr_initial_prompt": "Диаррайзер, ООО Ромашка",
        "audio_profile": "voice-call",
        "audio_format": "mp3-q5",
    }
    if stages is not None:
        config["stages"] = stages
    config.update(extra_config or {})
    meta = {"input_path": "D:/calls/call.m4a", "pipeline_config": config, "config": {}}
    if audio_format is not None:
        meta["config"]["audio_format"] = audio_format
    _write(job_dir / "meta" / "run.json", meta)
    files = {
        "convert": "audio/normalized.wav",
        "transcribe": "asr/transcript.json",
        "diarize": "diar/diarization.json",
        "merge": "merged/segments.json",
        "export": "export/result.md",
    }
    for name in stages_done:
        target = job_dir / files[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    return job_dir


def test_resume_starts_at_the_first_missing_stage_with_recorded_settings(tmp_path):
    job_dir = _job(tmp_path, stages_done=["convert", "transcribe"])

    mode, kwargs = resume_plan(job_dir)

    assert mode == "full"
    assert kwargs["from_stage"] == "diarize"
    assert kwargs["job_dir"] == job_dir
    assert kwargs["out_dir"] == job_dir.parent
    assert kwargs["input_path"] == "D:/calls/call.m4a"
    assert kwargs["language"] == "ru"
    assert (kwargs["min_speakers"], kwargs["max_speakers"]) == (2, 3)
    assert kwargs["asr_model"] == "small"
    assert kwargs["asr_initial_prompt"] == "Диаррайзер, ООО Ромашка"
    assert kwargs["audio_profile"] == "voice-call"
    # The recorded format's *name* is not a run_pipeline argument on its own.
    assert "audio_format" not in kwargs


def test_resume_of_a_recognition_only_job_skips_diarize(tmp_path):
    job_dir = _job(
        tmp_path,
        stages_done=["convert", "transcribe"],
        stages=["convert", "transcribe", "merge", "export"],
    )
    mode, kwargs = resume_plan(job_dir)
    assert mode == "asr_only"
    assert kwargs["from_stage"] == "merge"


def test_complete_job_has_no_resume_plan(tmp_path):
    job_dir = _job(tmp_path, stages_done=["convert", "transcribe", "diarize", "merge", "export"])
    assert resume_plan(job_dir) is None


def test_job_without_meta_cannot_be_resumed(tmp_path):
    job_dir = tmp_path / "out" / "half_20260101_000000"
    (job_dir / "audio").mkdir(parents=True)
    assert resume_plan(job_dir) is None


def test_resume_uses_the_full_recorded_audio_format(tmp_path):
    fmt = AudioFormat(name="my-opus", codec="libopus", extension="opus", bitrate="16k", keep_wav=False)
    job_dir = _job(tmp_path, stages_done=[], audio_format=fmt.to_dict())

    _, kwargs = resume_plan(job_dir)

    assert kwargs["from_stage"] == "convert"
    assert kwargs["audio_format"] == fmt


def test_min_turn_duration_falls_back_to_the_merge_metadata(tmp_path):
    job_dir = _job(tmp_path, stages_done=["convert", "transcribe", "diarize"])
    _write(job_dir / "merged" / "segments.json", {"metadata": {"min_turn_duration": 0.8}})
    (job_dir / "export").mkdir()
    _, kwargs = resume_plan(job_dir)
    assert kwargs["from_stage"] == "export"
    assert kwargs["min_turn_duration"] == 0.8


def test_rediarize_keeps_the_transcript_and_drops_old_names(tmp_path):
    job_dir = _job(
        tmp_path,
        stages_done=["convert", "transcribe", "merge", "export"],
        stages=["convert", "transcribe", "merge", "export"],
        extra_config={"min_turn_duration": 0.6},
    )

    mode, kwargs = rediarize_plan(job_dir, 2, 2)

    assert mode == "full"
    assert kwargs["from_stage"] == "diarize"
    assert (kwargs["min_speakers"], kwargs["max_speakers"]) == (2, 2)
    assert kwargs["speakers"] is None
    assert kwargs["min_turn_duration"] == 0.6


def test_rediarize_needs_a_transcript(tmp_path):
    job_dir = _job(tmp_path, stages_done=["convert"])
    assert rediarize_plan(job_dir, 1, 2) is None


def test_scan_marks_resumable_and_complete_jobs(tmp_path):
    _job(tmp_path, stages_done=["convert", "transcribe"])
    (jobs,) = scan_jobs(tmp_path / "out")
    assert jobs.resumable is True
    assert jobs.complete is False
    assert jobs.stage_done["transcribe"] is True
    assert jobs.stage_done["diarize"] is False


def test_job_date_parses_with_a_clash_suffix(tmp_path):
    date = _parse_job_date("call_20260102_030405_2", tmp_path)
    assert (date.year, date.month, date.day, date.hour, date.minute, date.second) == (2026, 1, 2, 3, 4, 5)
