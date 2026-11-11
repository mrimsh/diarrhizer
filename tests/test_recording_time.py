"""Recording proposals and edits must never replace the processing timestamp or rewrite transcripts."""

from datetime import datetime
from pathlib import Path

import pytest

from diarrhizer.projects import (
    JOB_FILE,
    move_jobs,
    plan_job_moves,
    read_config,
    render_name,
    reserve_job,
)
from diarrhizer.recording_time import (
    job_recording_time,
    local_datetime,
    suggest_recording_time,
)
from diarrhizer.utils import write_json_atomic
from diarrhizer_gui.job_scan import scan_jobs


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


@pytest.mark.parametrize("template, name", [
    ("DD.MM.YYYY_HH-mm-SS", "02.10.2026_15-04-33"),
    ("DD_MM_YY_HH-mm-SS", "02_10_26_15-04-33"),
    ("HHmmSS_DDMMYYYY", "150433_02102026"),
    ("YYYYMMDDHHmmSS", "20261002150433"),
    ("DD--MM__YYYY [HH+mm(SS)]", "02--10__2026 [15+04(33)]"),
    ("DD.MM.YYYY_YY_HH-mm-SS", "02.10.2026_26_15-04-33"),
])
def test_atomic_name_and_input_formats_round_trip(template, name):
    assert render_name(template, date="2026_10_02", time="15_04_33") == name
    assert suggest_recording_time(f"Call {name}.wav", template) == datetime(2026, 10, 2, 15, 4, 33)


@pytest.mark.parametrize("template, expected", [
    ("YYYY", "2026"), ("YY", "26"), ("MM", "10"), ("DD", "02"),
    ("HH", "15"), ("mm", "04"), ("SS", "33"),
    ("YYYY_MM_DD-HH_MM_SS", "2026_10_02-15_10_33"),
])
def test_each_date_time_token_is_independent_and_case_sensitive(template, expected):
    assert render_name(template, date="2026_10_02", time="15_04_33") == expected


def test_token_like_inserted_values_are_literal_and_keep_separators():
    assert render_name("PROJECT--DD.MM.YY_HH-mm-SS__TEAM-CUSTOM-SOURCE", project="YYYY",
                       team="MM", custom="mm", source="HH", date="2026_10_02", time="15_04_33") == (
                           "YYYY--02.10.26_15-04-33__MM-mm-HH")
    assert render_name("PROJECT-DD--MM__YYYY-TEAM", date="2026_10_02") == "02--10__2026"


@pytest.mark.parametrize("name, template, expected", [
    ("31_12_99.wav", "DD_MM_YY", datetime(2099, 12, 31)),
    ("01_01_00.wav", "DD_MM_YY", datetime(2000, 1, 1)),
    ("29.02.24_03-59-01.wav", "DD.MM.YY_HH-mm-SS", datetime(2024, 2, 29, 3, 59, 1)),
    ("29.02.23_03-59-01.wav", "DD.MM.YY_HH-mm-SS", None),
    ("02.10.2026_25-04-33.wav", "DD.MM.YYYY_HH-mm-SS", None),
    ("02.10.2026_15-60-33.wav", "DD.MM.YYYY_HH-mm-SS", None),
    ("02.10.2026_15-04-60.wav", "DD.MM.YYYY_HH-mm-SS", None),
    ("02.10.2026_25.wav", "DD.MM.YYYY_YY", None),
    ("02.10.2026_11.wav", "DD.MM.YYYY_MM", None),
    ("02.10.2026 and 03.10.2026.wav", "DD.MM.YYYY", None),
    ("02x10x2026.wav", "DD.MM.YYYY", None),  # Literal separators, not regular expressions.
    ("102.10.2026.wav", "DD.MM.YYYY", None),
    ("02.10.20260.wav", "DD.MM.YYYY", None),
])
def test_input_format_validates_calendar_time_duplicates_and_ambiguity(name, template, expected):
    assert suggest_recording_time(name, template) == expected


@pytest.mark.parametrize("template", ["DD.mm.YYYY", "HH-mm-SS", "yyyy-MM-DD"])
def test_input_format_requires_a_complete_case_sensitive_date(template):
    with pytest.raises(ValueError, match="YYYY или YY, MM"):
        suggest_recording_time("02.10.2026.wav", template)


@pytest.mark.parametrize("invalid", ["not a date", "2026-02-30T12:00:00", "2026-10-02T12:00:00+03:00"])
def test_explicit_invalid_local_time_fails_before_reserving_job(tmp_path, invalid):
    with pytest.raises(ValueError):
        reserve_job(tmp_path / "out", Path("call.mp4"), project="Project", recorded_at=invalid)
    assert not (tmp_path / "out").exists()


def test_recording_time_names_new_job_without_replacing_creation_time(tmp_path):
    job = reserve_job(tmp_path, Path("call.mp4"), project="Project", team="Coders",
                      naming_template="PROJECT-DD.MM.YYYY_HH-mm-SS-TEAM", recorded_at="2019-04-12T08:03:59")
    assert job.name == "Project-12.04.2019_08-03-59-Coders"
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
               "naming_template": "PROJECT-DD.MM.YY_HH-mm-SS"}
    preview = plan_job_moves(tmp_path, [job], **options)
    assert read_config(job / JOB_FILE) == identity
    new = move_jobs(tmp_path, [job], **options)[job]
    assert new == preview[job]
    assert new.name == "Project-06.05.20_07-08-09"
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
