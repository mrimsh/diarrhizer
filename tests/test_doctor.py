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


# --- check_python_version -----------------------------------------------------


def _python(monkeypatch, major, minor, micro=0):
    from types import SimpleNamespace

    monkeypatch.setattr(doctor.sys, "version_info", SimpleNamespace(major=major, minor=minor, micro=micro))


@pytest.mark.parametrize("minor", [11, 12])
def test_check_python_version_accepts_the_supported_pythons(monkeypatch, minor):
    _python(monkeypatch, 3, minor, 4)
    name, passed, message = doctor.check_python_version()
    assert passed and f"3.{minor}.4" in message


def test_check_python_version_rejects_one_that_is_too_old(monkeypatch):
    _python(monkeypatch, 3, 10, 9)
    name, passed, message = doctor.check_python_version()
    assert not passed and "need 3.11 or 3.12" in message


@pytest.mark.parametrize("minor", [13, 14])
def test_check_python_version_rejects_one_too_new_for_the_pinned_stack(monkeypatch, minor):
    """whisperx 3.3.1 has Requires-Python <3.13 and torch 2.3.1 no wheels past 3.12: pip cannot install
    the stack there and only says "No matching distribution found" - doctor has to name the real cause.
    """
    _python(monkeypatch, 3, minor, 1)
    name, passed, message = doctor.check_python_version()
    assert not passed
    assert "too new" in message and "Python 3.12" in message


def test_a_wrong_python_is_a_blocking_failure():
    assert doctor.is_blocking_failure("Python version", False)


def test_pyproject_excludes_the_pythons_the_stack_cannot_install_on():
    import re

    from diarrhizer.env_file import REPO_ROOT

    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'requires-python\s*=\s*">=3\.11,<3\.13"', text)


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


# --- check_summarization ------------------------------------------------------


def _save_providers(*providers):
    from diarrhizer.summary.providers import ProviderStore

    store = ProviderStore.load()
    for provider in providers:
        store.put(provider)
    store.save()


def test_check_summarization_passes_with_no_providers_configured():
    name, passed, message = doctor.check_summarization()

    assert passed
    assert "no providers configured" in message


def test_check_summarization_wants_cli_commands_on_path_and_api_keys_set(monkeypatch):
    from diarrhizer.summary.providers import KIND_CODEX_CLI, KIND_OPENAI_API, Provider

    _save_providers(
        Provider(name="ChatGPT", kind=KIND_CODEX_CLI),
        Provider(name="gw", kind=KIND_OPENAI_API, endpoint="https://x/v1", model="m", api_key_env="GW_KEY"),
    )
    monkeypatch.delenv("GW_KEY", raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda command: None)

    name, passed, message = doctor.check_summarization()

    assert not passed
    assert "'ChatGPT': command codex not found" in message
    assert "'gw': GW_KEY is not set" in message

    monkeypatch.setenv("GW_KEY", "sk-x")
    monkeypatch.setattr(doctor.shutil, "which", lambda command: f"C:/bin/{command}.exe")
    name, passed, message = doctor.check_summarization()
    assert passed
    assert "ChatGPT, gw" in message
    assert "sk-x" not in message


def test_check_summarization_reports_a_broken_file(summary_stores):
    summary_stores[1].write_text("{broken", encoding="utf-8")

    name, passed, message = doctor.check_summarization()

    assert not passed
    assert str(summary_stores[1]) in message


def test_a_summarization_problem_is_only_a_warning():
    assert not doctor.is_blocking_failure("Summarization", False)
    assert doctor.is_blocking_failure("FFmpeg", False)


def _fake_checks(*results):
    return [lambda result=result: result for result in results]


def test_optional_check_failure_is_only_a_warning(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "ALL_CHECKS",
        _fake_checks(("FFmpeg", True, "ok"), ("torchcodec", False, "Not installed")),
    )
    assert doctor.run_doctor_checks() is True
    out = capsys.readouterr().out
    assert "[WARN]: torchcodec" in out
    assert "1 optional warning" in out


def test_required_check_failure_fails_the_run(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "ALL_CHECKS",
        _fake_checks(("Critical imports", False, "whisperx: boom"), ("torchcodec", True, "")),
    )
    assert doctor.run_doctor_checks() is False
    out = capsys.readouterr().out
    assert "[FAIL]: Critical imports" in out
    assert "RECOMMENDED ACTION" in out


def test_run_all_checks_returns_every_result_in_order(monkeypatch):
    monkeypatch.setattr(
        doctor, "ALL_CHECKS", _fake_checks(("A", True, ""), ("B", False, "x"))
    )
    assert doctor.run_all_checks() == [("A", True, ""), ("B", False, "x")]
