"""Tests for diarrhizer_gui.job_scan: history rows and GUI:JOB_PLANS.

job_scan is Qt-free (stdlib + diarrhizer.audio_formats), so these run
without a display.
"""

import json
import os
import time
from pathlib import Path

from diarrhizer.audio_formats import AudioFormat
import pytest

from diarrhizer.pipeline.runner import STAGE_SETTINGS, run_pipeline
from diarrhizer.pipeline.stages.export import ExportStage
from diarrhizer.pipeline.stages.merge import MergeStage
from diarrhizer_gui.job_scan import (
    RECORDED_RUN_KEYS,
    SPEAKERS_SET_ASIDE,
    _parse_job_date,
    read_speaker_names,
    rediarize_plan,
    restart_plan,
    restart_settings,
    restart_stages,
    restartable_stages,
    resume_plan,
    scan_jobs,
    set_aside_speaker_names,
)

ALL_STAGES = ["convert", "transcribe", "diarize", "merge", "export"]


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


def test_restart_rebuilds_the_stage_and_what_is_built_on_it():
    assert restart_stages("full", "convert") == ALL_STAGES
    # Diarization only reads the audio: a new transcript does not touch it.
    assert restart_stages("full", "transcribe") == ["transcribe", "merge", "export"]
    assert restart_stages("full", "diarize") == ["diarize", "merge", "export"]
    assert restart_stages("full", "merge") == ["merge", "export"]
    assert restart_stages("full", "export") == ["export"]
    assert restart_stages("asr_only", "convert") == ["convert", "transcribe", "merge", "export"]
    assert restart_stages("asr_only", "diarize") == []


def test_restart_plan_forces_the_stage_with_the_recorded_settings(tmp_path):
    job_dir = _job(tmp_path, stages_done=ALL_STAGES)

    mode, kwargs = restart_plan(job_dir, "transcribe")

    assert mode == "full"
    assert (kwargs["from_stage"], kwargs["force_stage"]) == ("transcribe", "transcribe")
    assert kwargs["job_dir"] == job_dir
    assert kwargs["input_path"] == "D:/calls/call.m4a"
    assert kwargs["language"] == "ru"
    assert (kwargs["min_speakers"], kwargs["max_speakers"]) == (2, 3)
    assert kwargs["asr_initial_prompt"] == "Диаррайзер, ООО Ромашка"


def test_restart_keeps_saved_names_unless_diarization_is_redone(tmp_path):
    job_dir = _job(tmp_path, stages_done=ALL_STAGES)
    _write(job_dir / "speakers.json", {"SPEAKER_00": "Иван", "Speaker_01": "Speaker_01", "Speaker_02": " "})

    for stage in ("transcribe", "merge", "export"):
        _, kwargs = restart_plan(job_dir, stage)
        assert kwargs["speakers"] == {"Speaker_00": "Иван"}, stage
    for stage in ("convert", "diarize"):
        _, kwargs = restart_plan(job_dir, stage)
        assert kwargs["speakers"] is None, stage


def test_restart_needs_every_earlier_stage_to_have_its_output(tmp_path):
    job_dir = _job(tmp_path, stages_done=["convert", "transcribe"])

    assert restart_plan(job_dir, "transcribe") is not None
    assert restart_plan(job_dir, "diarize") is not None  # the stage "continue" would start with
    assert restart_plan(job_dir, "merge") is None
    assert restart_plan(job_dir, "export") is None


def test_restart_of_a_recognition_only_job_has_no_diarize_stage(tmp_path):
    job_dir = _job(
        tmp_path,
        stages_done=["convert", "transcribe", "merge", "export"],
        stages=["convert", "transcribe", "merge", "export"],
    )
    assert restart_plan(job_dir, "diarize") is None
    mode, kwargs = restart_plan(job_dir, "merge")
    assert (mode, kwargs["force_stage"]) == ("asr_only", "merge")


def test_restart_needs_a_recorded_source_file(tmp_path):
    job_dir = tmp_path / "out" / "half_20260101_000000"
    (job_dir / "audio").mkdir(parents=True)
    (job_dir / "audio" / "normalized.wav").write_bytes(b"x")
    assert restart_plan(job_dir, "convert") is None


def test_scan_lists_the_restartable_stages(tmp_path):
    _job(tmp_path, stages_done=["convert", "transcribe"])
    (job,) = scan_jobs(tmp_path / "out")
    assert job.input_path == "D:/calls/call.m4a"
    assert restartable_stages(job) == ["convert", "transcribe", "diarize"]


def test_speaker_names_are_read_and_set_aside(tmp_path):
    _write(tmp_path / "speakers.json", {"SPEAKER_01": " Мария ", "Speaker_02": "Speaker_02", "Speaker_03": ""})
    assert read_speaker_names(tmp_path) == {"Speaker_01": "Мария"}

    set_aside_speaker_names(tmp_path)
    assert not (tmp_path / "speakers.json").exists()
    assert (tmp_path / SPEAKERS_SET_ASIDE).exists()
    set_aside_speaker_names(tmp_path)  # nothing left to move: no error
    assert read_speaker_names(tmp_path) == {}


def _finished_merge_job(tmp_path: Path) -> tuple:
    """A job whose merge and export really ran (the two pure-Python stages), with
    a saved speaker name, and whose files are all clearly older than "now"."""
    job_dir = _job(tmp_path, stages_done=ALL_STAGES)
    _write(job_dir / "asr" / "transcript.json", {"segments": [{"start": 0, "end": 1, "text": "Привет"}], "words": []})
    _write(job_dir / "diar" / "diarization.json", {"segments": [{"start": 0, "end": 1, "speaker": "Speaker_00"}]})
    _write(job_dir / "speakers.json", {"Speaker_00": "Иван"})
    stages = [MergeStage(), ExportStage()]
    run_pipeline(out_dir=job_dir.parent, job_dir=job_dir, stages=stages, speakers={"Speaker_00": "Иван"})
    # Back-to-back writes can share a file-time tick; make the earlier run clearly older.
    long_ago = time.time() - 60
    for path in job_dir.rglob("*.*"):
        if path.is_file() and path.parent.name != "meta":
            os.utime(path, (long_ago, long_ago))
    return job_dir, stages


def test_restarted_merge_rebuilds_the_export_and_keeps_the_names(tmp_path):
    """The plan fed to the real runner: merge is forced, export follows from the
    newer segments.json, and the saved name still reaches result.md."""
    job_dir, stages = _finished_merge_job(tmp_path)

    _, kwargs = restart_plan(job_dir, "merge")
    result = run_pipeline(stages=stages, **kwargs)

    assert {s["stage"]: s["status"] for s in result["stages"]} == {"merge": "completed", "export": "completed"}
    assert "Иван" in (job_dir / "export" / "result.md").read_text(encoding="utf-8")


def test_restart_settings_are_those_of_the_recomputed_stages():
    assert restart_settings("full", "export") == []
    assert restart_settings("full", "merge") == ["min_turn_duration"]
    assert restart_settings("full", "diarize") == ["device", "min_speakers", "max_speakers", "min_turn_duration"]
    transcribe = restart_settings("full", "transcribe")
    assert "asr_model" in transcribe and "min_turn_duration" in transcribe
    assert "min_speakers" not in transcribe and "audio_profile" not in transcribe
    assert transcribe.count("device") == 1
    assert "asr_model" in restart_settings("asr_only", "convert")
    assert "min_speakers" not in restart_settings("asr_only", "convert")
    convert = restart_settings("full", "convert")
    assert {"audio_profile", "audio_format", "language", "min_speakers", "min_turn_duration"} <= set(convert)


def test_every_recorded_setting_belongs_to_a_stage():
    # Plans read the audio format back from the full format convert records, not from
    # RECORDED_RUN_KEYS; every other recorded key must be some stage's setting (and vice versa).
    offered = {key for keys in STAGE_SETTINGS.values() for key in keys}
    assert set(RECORDED_RUN_KEYS) | {"audio_format"} == offered


def test_restart_overrides_replace_the_recorded_settings(tmp_path):
    job_dir = _job(tmp_path, stages_done=ALL_STAGES)

    _, kwargs = restart_plan(job_dir, "transcribe", {"asr_model": "tiny", "asr_initial_prompt": None})

    assert kwargs["asr_model"] == "tiny"
    assert kwargs["asr_initial_prompt"] is None  # cleared, not "keep the recorded prompt"
    assert kwargs["language"] == "ru"  # the rest stays as recorded
    assert kwargs["force_stage"] == "transcribe"


@pytest.mark.parametrize(
    "stage, setting",
    [
        ("transcribe", "min_speakers"),  # diarization is not recomputed: the change would not apply
        ("export", "language"),
        ("merge", "asr_model"),
        ("diarize", "asr_model"),
        ("convert", "force_stage"),  # not a setting at all
    ],
)
def test_restart_refuses_settings_of_stages_it_does_not_recompute(tmp_path, stage, setting):
    job_dir = _job(tmp_path, stages_done=ALL_STAGES)
    with pytest.raises(ValueError, match=setting):
        restart_plan(job_dir, stage, {setting: 1})


def test_restart_with_changed_settings_is_recorded_in_the_job(tmp_path):
    """The run itself updates meta/run.json, so History and the next restart see
    the settings the merge was really made with."""
    job_dir, stages = _finished_merge_job(tmp_path)
    assert json.loads((job_dir / "meta" / "run.json").read_text(encoding="utf-8"))["pipeline_config"][
        "asr_model"
    ] == "small"

    _, kwargs = restart_plan(job_dir, "merge", {"min_turn_duration": 0.9})
    run_pipeline(stages=stages, **kwargs)

    recorded = json.loads((job_dir / "meta" / "run.json").read_text(encoding="utf-8"))["pipeline_config"]
    assert recorded["min_turn_duration"] == 0.9
    assert recorded["asr_model"] == "small"  # not a merge setting: untouched
    assert json.loads((job_dir / "merged" / "segments.json").read_text(encoding="utf-8"))["metadata"][
        "min_turn_duration"
    ] == 0.9
    # ... and the next plan starts from it.
    _, again = restart_plan(job_dir, "export")
    assert again["min_turn_duration"] == 0.9


def test_job_date_parses_with_a_clash_suffix(tmp_path):
    date = _parse_job_date("call_20260102_030405_2", tmp_path)
    assert (date.year, date.month, date.day, date.hour, date.minute, date.second) == (2026, 1, 2, 3, 4, 5)
