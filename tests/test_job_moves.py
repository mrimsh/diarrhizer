"""Batch transfers preserve job artifacts, reject unsafe targets and roll back ordinary failures."""

from pathlib import Path

import pytest

from diarrhizer import projects
from diarrhizer.artifacts import bundle_valid
from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.stages.export import ExportStage
from diarrhizer.projects import (
    JobMoveError, change_participant, ensure_scope, export_path, job_context, move_jobs,
    participants, read_config, reserve_job, save_scope,
)
from diarrhizer.summary.service import read_transcript
from diarrhizer.utils import write_json_atomic
from diarrhizer_gui.job_scan import restart_plan, scan_jobs


def _files(job):
    return {str(file.relative_to(job)): file.read_bytes() for file in job.rglob("*") if file.is_file()}


def _legacy(root, name):
    job = root / name
    job.mkdir(parents=True)
    (job / "notes.txt").write_text("keep me", encoding="utf-8")
    return job


@pytest.mark.parametrize("existing", [False, True])
def test_move_mixed_batch_preserves_exports_identity_and_independent_rosters(tmp_path, existing):
    root = tmp_path / "out"
    legacy = _legacy(root, "legacy_20260101_120000")
    portable = reserve_job(root, Path("recording.mp4"), project="Old", team="Coders")
    change_participant(portable.parent, "Alice")
    if existing:
        change_participant(ensure_scope(root, "New", "Designers"), "Bob")
    for job in (legacy, portable):
        write_json_atomic(job / "meta" / "run.json", {"input_path": str(tmp_path / "missing.wav"),
                                                       "pipeline_config": {"device": "cpu"}})
        write_json_atomic(job / "merged" / "segments.json", {
            "segments": [{"start": 0, "end": 1, "text": "Hello", "speaker_id": "Speaker_00"}],
        })
        write_json_atomic(job / "speakers.json", {"Speaker_00": "Alice"})
        write_json_atomic(job / "chat" / "conversation.json", {"question": "A previous question"})
        (job / "original").mkdir()
        (job / "original" / "recording.mp4").write_bytes(b"original media")
        run_pipeline(job_dir=job, out_dir=root, stages=[ExportStage()])
    snapshots = {job: _files(job) for job in (legacy, portable)}
    moved = move_jobs(root, [legacy, portable], project="New", team="Designers")
    for previous, current in moved.items():
        assert current == root / "New" / "Designers" / previous.name
        assert not previous.exists()
        assert _files(current) == snapshots[previous]
        assert "Alice" in read_transcript(current)
        assert bundle_valid(current / "export", "manifest.json")
        assert export_path(current, "md").exists()
        assert job_context(current) == (root / "New", root / "New" / "Designers")
    assert participants(root / "Old" / "Coders") == ["Alice"]
    assert participants(root / "New" / "Designers") == (["Bob"] if existing else [])
    assert {(job.project, job.team) for job in scan_jobs(root)} == {("New", "Designers")}


def test_move_to_project_root_ignores_default_team_and_can_return_to_library(tmp_path):
    source = reserve_job(tmp_path, Path("call.wav"), project="Old", team="Coders")
    project = ensure_scope(tmp_path, "New", "Default").parent
    save_scope(project, {"default_team": "Default"})
    destination = move_jobs(tmp_path, [source], project="New")[source]
    assert destination.parent == project
    root_job = move_jobs(tmp_path, [destination])[destination]
    assert root_job.parent == tmp_path


def test_duplicate_names_get_suffixes_and_no_op_members_are_kept(tmp_path):
    a = _legacy(tmp_path, "same")
    b = _legacy(ensure_scope(tmp_path, "Old"), "same")
    target = ensure_scope(tmp_path, "Target", "Team")
    collision = _legacy(target, "same")
    before = _files(collision)
    moved = move_jobs(tmp_path, [a, b, a, collision], project="Target", team="Team")
    assert moved == {a: target / "same_2", b: target / "same_3"}
    assert _files(collision) == before
    assert move_jobs(tmp_path, list(moved.values()), project="Target", team="Team") == {}


@pytest.mark.parametrize("invalid", ["root", "scope", "outside"])
def test_invalid_sources_fail_before_destination_creation(tmp_path, invalid):
    root = tmp_path / "out"
    job = _legacy(root, "call")
    scope = ensure_scope(root, "Project")
    outside = _legacy(tmp_path, "outside")
    source = {"root": root, "scope": scope, "outside": outside}[invalid]
    with pytest.raises(ValueError):
        move_jobs(root, [job, source], project="New", team="Team")
    assert job.exists() and not (root / "New").exists()


def test_invalid_destination_does_not_move_or_create_scopes(tmp_path):
    job = _legacy(tmp_path, "call")
    for project, team in (("../outside", ""), ("New", "bad/team"), ("", "Team"), ("call", "Team")):
        with pytest.raises(ValueError):
            move_jobs(tmp_path, [job], project=project, team=team)
        assert job.exists() and not (tmp_path / "New").exists()


def test_move_updates_internal_run_paths_and_resuming_legacy_job(tmp_path):
    job = _legacy(tmp_path, "legacy")
    audio = job / "audio" / "normalized.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"audio")
    write_json_atomic(job / "meta" / "run.json", {
        "input_path": str(audio), "output_path": str(audio), "output_paths": [str(audio)],
        "archive_paths": [], "source_fingerprint": {"path": str(audio), "sha256": "kept"},
        "pipeline_config": {"stages": ["convert", "transcribe", "merge", "export"]},
    })
    destination = move_jobs(tmp_path, [job], project="New")[job]
    expected = str(destination / "audio" / audio.name)
    data = read_config(destination / "meta" / "run.json")
    assert data["input_path"] == data["output_path"] == data["source_fingerprint"]["path"] == expected
    assert data["source_fingerprint"]["sha256"] == "kept"
    assert data["output_paths"] == [expected]
    assert restart_plan(destination, "convert")[1]["input_path"] == expected


def test_selected_no_op_job_keeps_its_reference_to_another_moved_job(tmp_path):
    source = _legacy(tmp_path, "source")
    staying = _legacy(ensure_scope(tmp_path, "Target"), "staying")
    write_json_atomic(staying / "meta" / "run.json", {"input_path": str(source / "notes.txt")})
    moved = move_jobs(tmp_path, [source, staying], project="Target")
    assert set(moved) == {source}
    assert read_config(staying / "meta" / "run.json")["input_path"] == str(moved[source] / "notes.txt")


def test_rename_failure_rolls_back_entire_batch(tmp_path, monkeypatch):
    a, b = _legacy(tmp_path, "a"), _legacy(tmp_path, "b")
    original = Path.rename
    def fail_second(self, target):
        if self == b:
            raise OSError("synthetic locked recording")
        return original(self, target)
    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="locked recording"):
        move_jobs(tmp_path, [a, b], project="New", team="Team")
    assert a.exists() and b.exists()
    assert not (tmp_path / "New" / "Team" / "a").exists()


def test_metadata_write_failure_restores_original_bytes_and_locations(tmp_path, monkeypatch):
    a, b = _legacy(tmp_path, "a"), _legacy(tmp_path, "b")
    for job in (a, b):
        write_json_atomic(job / "meta" / "run.json", {"input_path": str(job / "notes.txt")})
    snapshots = {job: _files(job) for job in (a, b)}
    original = projects.write_json_atomic
    writes = []
    def fail_second(path, data):
        if Path(path).name == "run.json":
            writes.append(path)
            if len(writes) == 2:
                raise OSError("synthetic full disk")
        return original(path, data)
    monkeypatch.setattr(projects, "write_json_atomic", fail_second)
    with pytest.raises(OSError, match="full disk"):
        move_jobs(tmp_path, [a, b], project="New")
    assert _files(a) == snapshots[a]
    assert _files(b) == snapshots[b]


def test_incomplete_rollback_reports_actual_remaining_paths(tmp_path, monkeypatch):
    a, b = _legacy(tmp_path, "a"), _legacy(tmp_path, "b")
    original = Path.rename
    def fail(self, target):
        if self == b or Path(target) == a:
            raise OSError("synthetic locked directory")
        return original(self, target)
    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(JobMoveError) as caught:
        move_jobs(tmp_path, [a, b], project="New")
    assert caught.value.moved == {a: tmp_path / "New" / "a"}
    assert b.exists() and not a.exists()
    assert (caught.value.moved[a] / "notes.txt").read_text() == "keep me"
