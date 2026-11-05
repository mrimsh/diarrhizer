"""Portable library behavior: hierarchy, names, renames, rosters and old jobs."""

from pathlib import Path

import pytest

from diarrhizer.projects import (
    DEFAULT_TEMPLATE, JOB_FILE, PROJECT_FILE, TEAM_FILE, change_participant, ensure_scope,
    export_path, inherited_settings, iter_job_dirs, job_context, list_scopes, participants,
    read_config, rename_scope, render_name, reserve_job, save_scope, scope_path,
)
from diarrhizer_gui.job_scan import delete_job, scan_jobs


def test_name_template_optional_values_and_literal_custom():
    assert render_name(DEFAULT_TEMPLATE, project="MySuperProject", team="Coders", date="2026_11_05") == "MySuperProject-2026_11_05-Coders"
    assert render_name(project="Проект", date="2026_11_05") == "Проект-2026_11_05"
    assert render_name("{{project}}-{{date}}-{{team}}-{{custom}}", project="А", date="2026_11_05",
                       custom="TEAM") == "А-2026_11_05-TEAM"


@pytest.mark.parametrize("name", ["../escape", "a/b", "a\\b", "CON", "NUL.txt", "name.", "bad:name", ".hidden", "", "x "])
def test_invalid_scope_names_never_create_folders(tmp_path, name):
    with pytest.raises(ValueError):
        ensure_scope(tmp_path / "out", name, "team")
    assert not (tmp_path / "out").exists()


def test_default_team_inheritance_and_explicit_root(tmp_path):
    project = ensure_scope(tmp_path, "Project")
    ensure_scope(tmp_path, "Project", "Coders")
    save_scope(project, {"default_team": "Coders", "naming_template": "PROJECT-TEAM-CUSTOM",
                         "storage": {"working": "template"}})
    a = reserve_job(tmp_path, Path("call.mp4"), project="Project", custom="one")
    b = reserve_job(tmp_path, Path("call.mp4"), project="Project", team="", custom="two")
    assert a.parent == project / "Coders"
    assert b.parent == project
    assert a.name == "Project-Coders-one"
    assert b.name == "Project-two"
    assert read_config(a / JOB_FILE)["storage"]["working"] == "template"
    assert inherited_settings(tmp_path, "Project", "Coders")["naming_template"] == "PROJECT-TEAM-CUSTOM"


def test_rosters_are_explicit_and_independent(tmp_path):
    project = ensure_scope(tmp_path, "Project")
    team = ensure_scope(tmp_path, "Project", "Coders")
    other = ensure_scope(tmp_path, "Project", "Designers")
    change_participant(project, "Root")
    change_participant(team, "Alice")
    change_participant(team, "Alice")
    assert participants(team) == ["Alice"]
    assert participants(project) == ["Root"]
    assert participants(other) == []
    change_participant(team, "Alice", remove=True)
    assert participants(team) == []


def test_scope_rename_moves_jobs_and_updates_default_team(tmp_path):
    team = ensure_scope(tmp_path, "Project", "Coders")
    save_scope(team.parent, {"default_team": "Coders"})
    job = reserve_job(tmp_path, Path("source.mp4"), project="Project", team="Coders")
    change_participant(team, "Alice")
    renamed = rename_scope(team, "Engineers")
    assert not team.exists()
    assert participants(renamed) == ["Alice"]
    assert read_config(renamed.parent / PROJECT_FILE)["default_team"] == "Engineers"
    project = rename_scope(renamed.parent, "Renamed Project")
    moved = project / renamed.name / job.name
    assert moved.is_dir()
    assert export_path(moved, "md").name == f"{job.name}.md"
    assert job_context(moved) == (project, project / "Engineers")
    assert list_scopes(tmp_path) == ["Renamed Project"]
    assert read_config(project / "Engineers" / TEAM_FILE)["name"] == "Engineers"


def test_scan_and_safe_delete_mixed_hierarchy_with_collisions(tmp_path):
    source = Path("source.mp4")
    a = reserve_job(tmp_path, source, project="Project", team="Coders")
    b = reserve_job(tmp_path, source, project="Project", team="Coders")
    c = reserve_job(tmp_path, source, project="Project", team="")
    legacy = tmp_path / "old_20260101_120000"
    legacy.mkdir()
    assert b.name == a.name + "_2"
    assert set(iter_job_dirs(tmp_path)) == {a, b, c, legacy}
    summaries = scan_jobs(tmp_path)
    assert {(j.project, j.team) for j in summaries} == {("Project", "Coders"), ("Project", ""), ("", "")}
    for invalid in (tmp_path, a.parent, c.parent, tmp_path.parent):
        with pytest.raises(ValueError):
            delete_job(tmp_path, invalid)
    delete_job(tmp_path, b)
    assert a.exists() and c.exists() and not b.exists()


def test_scope_creation_refuses_job_collision_and_team_without_project(tmp_path):
    (tmp_path / "Project").mkdir()
    with pytest.raises(ValueError):
        ensure_scope(tmp_path, "Project")
    with pytest.raises(ValueError):
        scope_path(tmp_path, "", "Coders")
    assert not (tmp_path / "Project" / PROJECT_FILE).exists()


def test_rename_refuses_collision_without_changing_settings(tmp_path):
    a = ensure_scope(tmp_path, "A")
    b = ensure_scope(tmp_path, "B")
    with pytest.raises(ValueError):
        rename_scope(a, "B")
    assert read_config(a / PROJECT_FILE)["name"] == "A"
    assert read_config(b / PROJECT_FILE)["name"] == "B"
