"""Tests for diarrhizer.diagnostics.doctor.check_ffmpeg.

check_ffmpeg() must mirror FFmpegAdapter's resolution order (env var > PATH -
doctor has no constructor argument to check) so that `python -m diarrhizer
doctor` doesn't report "not found" when DIARRHIZER_FFMPEG_PATH is set.
"""

import pytest

from diarrhizer.adapters import ffmpeg as ffmpeg_module
from diarrhizer.adapters.ffmpeg import ENV_FFMPEG_PATH
from diarrhizer.diagnostics import doctor


def make_fake_ffmpeg(tmp_path, name="ffmpeg.exe"):
    path = tmp_path / name
    path.write_text("fake binary", encoding="utf-8")
    return path


def test_check_ffmpeg_reports_env_var_path_when_set(tmp_path, monkeypatch):
    env_ffmpeg = make_fake_ffmpeg(tmp_path)
    monkeypatch.setenv(ENV_FFMPEG_PATH, str(env_ffmpeg))
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: None)

    name, passed, message = doctor.check_ffmpeg()

    assert name == "FFmpeg"
    assert passed is True
    assert str(env_ffmpeg) in message
    assert ENV_FFMPEG_PATH in message


def test_check_ffmpeg_falls_back_to_path_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV_FFMPEG_PATH, raising=False)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "C:/on/path/ffmpeg.exe")

    name, passed, message = doctor.check_ffmpeg()

    assert passed is True
    assert "C:/on/path/ffmpeg.exe" in message


def test_check_ffmpeg_fails_with_clear_message_when_unresolved(monkeypatch):
    monkeypatch.delenv(ENV_FFMPEG_PATH, raising=False)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: None)

    name, passed, message = doctor.check_ffmpeg()

    assert passed is False
    assert ENV_FFMPEG_PATH in message


def test_check_ffmpeg_reports_clear_error_for_invalid_env_var(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.exe"
    monkeypatch.setenv(ENV_FFMPEG_PATH, str(missing))
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "C:/on/path/ffmpeg.exe")

    name, passed, message = doctor.check_ffmpeg()

    assert passed is False
    assert str(missing) in message
