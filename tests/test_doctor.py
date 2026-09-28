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


def test_check_cudnn_skips_on_cpu_only_torch(monkeypatch):
    class _Cuda:
        @staticmethod
        def is_available():
            return False

    class _Torch:
        cuda = _Cuda
        __file__ = "unused"

    monkeypatch.setitem(__import__("sys").modules, "torch", _Torch)
    name, passed, message = doctor.check_cudnn()
    assert name == "cuDNN"
    assert passed is True
    assert "CPU-only" in message


def test_check_ffmpeg_reports_clear_error_for_invalid_env_var(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.exe"
    monkeypatch.setenv(ENV_FFMPEG_PATH, str(missing))
    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: "C:/on/path/ffmpeg.exe")

    name, passed, message = doctor.check_ffmpeg()

    assert passed is False
    assert str(missing) in message


# --- check_audio_formats ------------------------------------------------------


@pytest.fixture
def audio_formats_file(tmp_path, monkeypatch):
    from diarrhizer.audio_formats import ENV_AUDIO_FORMATS_FILE

    path = tmp_path / "audio_formats.json"
    monkeypatch.setenv(ENV_AUDIO_FORMATS_FILE, str(path))
    monkeypatch.setattr(doctor, "resolve_ffmpeg_path", lambda: "ffmpeg")
    return path


def test_check_audio_formats_passes_when_the_default_encoder_exists(audio_formats_file, monkeypatch):
    audio_formats_file.write_text('{"default": "mp3-q5"}', encoding="utf-8")
    monkeypatch.setattr(doctor, "list_audio_encoders", lambda path: {"pcm_s16le", "flac", "libmp3lame"})

    name, passed, message = doctor.check_audio_formats()

    assert passed
    assert "mp3-q5" in message
    assert "unavailable in this FFmpeg build: opus-24k" in message


def test_check_audio_formats_fails_when_the_default_encoder_is_missing(audio_formats_file, monkeypatch):
    audio_formats_file.write_text('{"default": "opus-24k"}', encoding="utf-8")
    monkeypatch.setattr(doctor, "list_audio_encoders", lambda path: {"pcm_s16le"})

    name, passed, message = doctor.check_audio_formats()

    assert not passed
    assert "libopus" in message


def test_check_audio_formats_reports_a_broken_file(audio_formats_file):
    audio_formats_file.write_text("{broken", encoding="utf-8")

    name, passed, message = doctor.check_audio_formats()

    assert not passed
    assert str(audio_formats_file) in message
