"""Tests for diarrhizer.adapters.ffmpeg FFmpeg path resolution.

Covers resolve_ffmpeg_path()'s priority order (explicit arg > DIARRHIZER_FFMPEG_PATH
env var > PATH) and FFmpegAdapter's use of it. No real ffmpeg binary is required -
"ffmpeg" stand-ins are just files created in tmp_path.
"""

import re

import pytest

from diarrhizer.adapters import ffmpeg as ffmpeg_module
from diarrhizer.adapters.ffmpeg import ENV_FFMPEG_PATH, FFmpegAdapter, resolve_ffmpeg_path


def make_fake_ffmpeg(tmp_path, name="ffmpeg.exe"):
    path = tmp_path / name
    path.write_text("fake binary", encoding="utf-8")
    return path


# --- resolve_ffmpeg_path: priority order ------------------------------------


def test_explicit_arg_wins_over_env_and_path(tmp_path, monkeypatch):
    explicit = make_fake_ffmpeg(tmp_path, "explicit.exe")
    env_ffmpeg = make_fake_ffmpeg(tmp_path, "env.exe")
    monkeypatch.setenv(ENV_FFMPEG_PATH, str(env_ffmpeg))
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "C:/on/path/ffmpeg.exe")

    assert resolve_ffmpeg_path(explicit) == str(explicit)


def test_env_var_wins_over_path_when_no_explicit_arg(tmp_path, monkeypatch):
    env_ffmpeg = make_fake_ffmpeg(tmp_path, "env.exe")
    monkeypatch.setenv(ENV_FFMPEG_PATH, str(env_ffmpeg))
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "C:/on/path/ffmpeg.exe")

    assert resolve_ffmpeg_path() == str(env_ffmpeg)


def test_falls_back_to_path_when_no_explicit_and_no_env(monkeypatch):
    monkeypatch.delenv(ENV_FFMPEG_PATH, raising=False)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "C:/on/path/ffmpeg.exe")

    assert resolve_ffmpeg_path() == "C:/on/path/ffmpeg.exe"


def test_returns_none_when_nothing_resolves(monkeypatch):
    monkeypatch.delenv(ENV_FFMPEG_PATH, raising=False)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: None)

    assert resolve_ffmpeg_path() is None


# --- resolve_ffmpeg_path: invalid overrides fail loudly, no silent fallback -


def test_invalid_explicit_arg_raises_and_does_not_fall_back(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.exe"
    env_ffmpeg = make_fake_ffmpeg(tmp_path, "env.exe")
    monkeypatch.setenv(ENV_FFMPEG_PATH, str(env_ffmpeg))
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "C:/on/path/ffmpeg.exe")

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        resolve_ffmpeg_path(missing)


def test_invalid_env_var_raises_and_does_not_fall_back_to_path(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.exe"
    monkeypatch.setenv(ENV_FFMPEG_PATH, str(missing))
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "C:/on/path/ffmpeg.exe")

    with pytest.raises(FileNotFoundError, match=ENV_FFMPEG_PATH):
        resolve_ffmpeg_path()


def test_explicit_arg_pointing_at_directory_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_FFMPEG_PATH, raising=False)

    with pytest.raises(FileNotFoundError):
        resolve_ffmpeg_path(tmp_path)


# --- FFmpegAdapter integration ----------------------------------------------


def test_adapter_uses_explicit_ffmpeg_path(tmp_path, monkeypatch):
    explicit = make_fake_ffmpeg(tmp_path)
    monkeypatch.delenv(ENV_FFMPEG_PATH, raising=False)

    adapter = FFmpegAdapter(ffmpeg_path=explicit)

    assert adapter.ffmpeg_path == str(explicit)


def test_adapter_uses_env_var_when_no_explicit_path(tmp_path, monkeypatch):
    env_ffmpeg = make_fake_ffmpeg(tmp_path)
    monkeypatch.setenv(ENV_FFMPEG_PATH, str(env_ffmpeg))

    adapter = FFmpegAdapter()

    assert adapter.ffmpeg_path == str(env_ffmpeg)


def test_adapter_falls_back_to_path(monkeypatch):
    monkeypatch.delenv(ENV_FFMPEG_PATH, raising=False)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "C:/on/path/ffmpeg.exe")

    adapter = FFmpegAdapter()

    assert adapter.ffmpeg_path == "C:/on/path/ffmpeg.exe"


def test_adapter_invalid_explicit_path_raises_runtime_error_not_silent_fallback(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.exe"
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "C:/on/path/ffmpeg.exe")

    with pytest.raises(RuntimeError, match=re.escape(str(missing))):
        FFmpegAdapter(ffmpeg_path=missing)


def test_adapter_raises_runtime_error_mentioning_all_resolution_options_when_unresolved(monkeypatch):
    monkeypatch.delenv(ENV_FFMPEG_PATH, raising=False)
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError) as exc_info:
        FFmpegAdapter()

    message = str(exc_info.value)
    assert "PATH" in message
    assert ENV_FFMPEG_PATH in message
    assert "ffmpeg_path" in message
