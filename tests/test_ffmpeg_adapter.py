"""Tests for diarrhizer.adapters.ffmpeg FFmpeg path resolution.

Covers resolve_ffmpeg_path()'s priority order (explicit arg > DIARRHIZER_FFMPEG_PATH
env var > PATH) and FFmpegAdapter's use of it. No real ffmpeg binary is required -
"ffmpeg" stand-ins are just files created in tmp_path.
"""

import re
import shutil
import subprocess
from pathlib import Path

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


# --- split-stereo: mono downmix + per-channel extras -------------------------


def _fake_ffmpeg_run(cmd, capture_output=True, text=True, check=True, timeout=None):
    """Stand-in for subprocess.run that just materializes whatever output
    file the command was going to write (last arg), instead of invoking a
    real ffmpeg binary.
    """
    output_path = Path(cmd[-1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake wav data")
    return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")


def test_split_stereo_also_writes_standard_mono_downmix(tmp_path, monkeypatch):
    """split-stereo must still produce the same audio/normalized.wav every
    other profile does (in addition to the left/right extras), since
    transcribe/diarize read that path unconditionally regardless of profile.
    """
    ffmpeg_path = make_fake_ffmpeg(tmp_path)
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"fake media")
    output_path = tmp_path / "job" / "audio" / "normalized.wav"

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", _fake_ffmpeg_run)

    adapter = FFmpegAdapter(ffmpeg_path=ffmpeg_path)
    result = adapter.convert_to_wav(
        input_file, output_path, audio_profile=FFmpegAdapter.PROFILE_SPLIT_STEREO
    )

    expected = [
        output_path,
        output_path.parent / "normalized_left.wav",
        output_path.parent / "normalized_right.wav",
    ]
    assert result == expected
    for path in expected:
        assert path.exists(), f"expected {path} to be written"


# --- audio profiles: real ffmpeg invocation catches invalid filter syntax ---
#
# Every other test in this file fakes subprocess.run, so none of them can
# catch "the constructed -af string is invalid ffmpeg syntax" - that's
# exactly how denoise-light's afftdn=nr=12:nt=auto shipped and stayed broken:
# nt (noise_type) is an enum (white/vinyl/shellac/custom), "auto" was never a
# valid value, and ffmpeg only rejects it once actually invoked. These tests
# run a real ffmpeg binary end to end and are skipped where one isn't on PATH.

REQUIRES_REAL_FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="requires a real ffmpeg binary on PATH"
)

_FIXTURE_AUDIO = Path(__file__).resolve().parent.parent / "test_speech.mp3"


@REQUIRES_REAL_FFMPEG
@pytest.mark.skipif(not _FIXTURE_AUDIO.exists(), reason="test_speech.mp3 fixture not present")
@pytest.mark.parametrize(
    "profile",
    [
        FFmpegAdapter.PROFILE_RAW,
        FFmpegAdapter.PROFILE_VOICE_CALL,
        FFmpegAdapter.PROFILE_DENOISE_LIGHT,
        FFmpegAdapter.PROFILE_SPLIT_STEREO,
    ],
)
def test_audio_profile_filters_are_valid_ffmpeg_syntax(tmp_path, profile):
    output_path = tmp_path / "normalized.wav"
    adapter = FFmpegAdapter()

    result = adapter.convert_to_wav(_FIXTURE_AUDIO, output_path, audio_profile=profile)

    written = result if isinstance(result, list) else [result]
    for path in written:
        path = Path(path)
        assert path.exists(), f"expected {path} to be written"
        assert path.stat().st_size > 0, f"{path} was written but is empty"
