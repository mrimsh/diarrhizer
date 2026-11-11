"""Recording proposals and edits must never replace the processing timestamp or rewrite transcripts."""

from datetime import datetime
from pathlib import Path

import pytest

from diarrhizer.recording_time import job_recording_time, local_datetime, suggest_recording_time
from diarrhizer.projects import (
    JOB_FILE, move_jobs, plan_job_moves, read_config, render_name, reserve_job,
)
from diarrhizer_gui.job_scan import scan_jobs
from diarrhizer.utils import write_json_atomic


@pytest.mark.parametrize("name, expected", [
    ("Meeting 2026-10-02 15-04-33.mp4", datetime(2026, 10, 2, 15, 4, 33)),
    ("Meeting_2026_10_02_15_04_33.wav", datetime(2026, 10, 2, 15, 4, 33)),
    ("2026.10.02 15.04.33.mkv", datetime(2026, 10, 2, 15, 4, 33)),
    ("2026-10-02-15-04-33.mp4", datetime(2026, 10, 2, 15, 4, 33)),
    ("2026.10.02.15.04.33.mp4", datetime(2026, 10, 2, 15, 4, 33)),
    ("call_20261002_150433.mp3", datetime(2026, 10, 2, 15, 4, 33)),
    ("2026-10-02T15:04:33.webm", datetime(2026, 10, 2, 15, 4, 33)),
    ("02.10.2026 15-04-33.flac", datetime(2026, 10, 2, 15, 4, 33)),
    ("2026-10-02 15-04.wav", datetime(2026, 10, 2, 15, 4)),
    ("2026-10-02.wav", datetime(2026, 10, 2)),
    ("2024-02-29 12-00-00.wav", datetime(2024, 2, 29, 12)),
    ("meeting.wav", None),
    ("2026-02-30 12-00-00.wav", None),
    ("2026-10-02 25-04-33.wav", None),
    ("2026-10-02 and 2026-10-03.wav", None),
    ("02-03-2026.wav", None),  # Ambiguous day/month: ask the user through the editor instead.
])
def test_filename_proposals(name, expected):
    assert suggest_recording_time(name) == expected


@pytest.mark.parametrize("invalid", ["not a date", "2026-02-30T12:00:00", "2026-10-02T12:00:00+03:00"])
def test_explicit_invalid_local_time_fails_before_reserving_job(tmp_path, invalid):
    with pytest.raises(ValueError):
        reserve_job(tmp_path / "out", Path("call.mp4"), project="Project", recorded_at=invalid)
    assert not (tmp_path / "out").exists()


def test_recording_time_names_new_job_without_replacing_creation_time(tmp_path):
    job = reserve_job(tmp_path, Path("call.mp4"), project="Project", team="Coders",
                      naming_template="PROJECT-YYYY_MM_DD-HH_MM_SS-TEAM", recorded_at="2019-04-12T08:03:59")
    assert job.name == "Project-2019_04_12-08_03_59-Coders"
    identity = read_config(job / JOB_FILE)
    assert identity["recorded_at"] == "2019-04-12T08:03:59"
    assert identity["created_at"] != identity["recorded_at"]
    assert scan_jobs(tmp_path)[0].sort_key == datetime(2019, 4, 12, 8, 3, 59)
    assert "08:03:59" in scan_jobs(tmp_path)[0].date_label
    assert render_name("{{date}}-{{time}}", date="2019_04_12", time="08_03_59") == "2019_04_12-08_03_59"


def test_existing_job_can_edit_recording_time_in_place_without_touching_transcript(tmp_path):
    job = reserve_job(tmp_path, Path("call.wav"), project="Project", recorded_at="2026-11-01T10:00:00")
    identity = read_config(job / JOB_FILE)
    old_audio = job / "audio" / "working" / "call.wav"
    old_audio.parent.mkdir(parents=True)
    old_audio.write_bytes(b"retained working audio")
    transcript = job / "asr" / "transcript.json"
    write_json_atomic(transcript, {"text": "Keep original words"})
    old_transcript = transcript.read_bytes()
    options = {"project": "Project", "rename": True, "recorded_at": "2020-05-06T07:08:09",
               "naming_template": "PROJECT-YYYY_MM_DD-HH_MM_SS"}
    preview = plan_job_moves(tmp_path, [job], **options)
    assert read_config(job / JOB_FILE) == identity
    new = move_jobs(tmp_path, [job], **options)[job]
    assert new == preview[job]
    updated = read_config(new / JOB_FILE)
    assert updated["created_at"] == identity["created_at"]
    assert updated["recorded_at"] == "2020-05-06T07:08:09"
    assert (new / "audio" / "working" / f"{new.name}.wav").read_bytes() == b"retained working audio"
    assert (new / "asr" / "transcript.json").read_bytes() == old_transcript
    assert job_recording_time(new) == datetime(2020, 5, 6, 7, 8, 9)


def test_legacy_fallback_preserves_hours_minutes_seconds(tmp_path):
    job = tmp_path / "call_20261002_150433"
    job.mkdir()
    assert job_recording_time(job) == datetime(2026, 10, 2, 15, 4, 33)
    assert local_datetime("2026-10-02 15:04:33").isoformat() == "2026-10-02T15:04:33"
