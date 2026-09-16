"""Tests for project .env loading."""

import os

from diarrhizer import env_file


def test_load_project_env_cwd_overrides_repo_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("HF_TOKEN=from_repo\nSHARED=repo\n", encoding="utf-8")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".env").write_text("HF_TOKEN=from_cwd\n", encoding="utf-8")

    monkeypatch.setattr(env_file, "REPO_ROOT", repo)
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("SHARED", raising=False)

    env_file.load_project_env()

    assert os.environ["HF_TOKEN"] == "from_cwd"
    assert os.environ["SHARED"] == "repo"


def test_load_project_env_does_not_override_process_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("HF_TOKEN=from_file\n", encoding="utf-8")
    monkeypatch.setattr(env_file, "REPO_ROOT", repo)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("HF_TOKEN", "already_set")

    env_file.load_project_env()

    assert os.environ["HF_TOKEN"] == "already_set"
