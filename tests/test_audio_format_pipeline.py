"""End-to-end tests for audio formats (CONFIG:AUDIO_FORMATS) through the pipeline.

Same approach as tests/test_split_stereo_pipeline.py: run_pipeline() with the
FFmpeg subprocess and the WhisperX adapters faked, so it runs without ffmpeg,
torch or pyannote. Covers what a user choosing "keep only a small archive"
relies on: the WAV is dropped once the job completes (and only then), later
runs stay cached and decode the archive, and changing the storage format of a
finished job does not re-run ASR/diarization.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from diarrhizer.adapters import ffmpeg as ffmpeg_module
from diarrhizer.audio_formats import ENV_AUDIO_FORMATS_FILE
from diarrhizer.pipeline.runner import JobContext, PipelineConfig, run_pipeline
from diarrhizer.pipeline.stages.convert import ConvertStage
from diarrhizer.pipeline.stages.diarize import DiarizeStage
from diarrhizer.pipeline.stages.export import ExportStage
from diarrhizer.pipeline.stages.merge import MergeStage
from diarrhizer.pipeline.stages.transcribe import TranscribeStage

FFMPEG_CALLS: list = []
DECODED: list = []


def _fake_ffmpeg_run(cmd, capture_output=True, text=True, check=True, timeout=None):
    FFMPEG_CALLS.append(list(cmd))
    output_path = Path(cmd[-1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake audio")
    return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")


class _FakeWhisperXAdapter:
    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio_path, language=None):
        DECODED.append(("transcribe", Path(audio_path).name))
        return {
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
            "words": [
                {"start": 0.0, "end": 0.5, "word": "hello"},
                {"start": 0.5, "end": 1.0, "word": "world"},
            ],
            "language": "en",
        }

    def unload(self) -> None:
        return


class _FakeWhisperXDiarizeAdapter:
    def __init__(self, *args, **kwargs):
        pass

    def diarize(self, audio_path):
        DECODED.append(("diarize", Path(audio_path).name))
        return {
            "segments": [{"start": 0.0, "end": 1.0, "speaker": "Speaker_00"}],
            "num_speakers": 1,
            "speakers": ["Speaker_00"],
        }

    def unload(self) -> None:
        return


@pytest.fixture(autouse=True)
def fake_heavy_deps(tmp_path, monkeypatch):
    FFMPEG_CALLS.clear()
    DECODED.clear()
    monkeypatch.setenv(ENV_AUDIO_FORMATS_FILE, str(tmp_path / "audio_formats.json"))
    monkeypatch.setattr(ffmpeg_module, "resolve_ffmpeg_path", lambda explicit=None: "ffmpeg")
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", _fake_ffmpeg_run)
    monkeypatch.setattr(
        "diarrhizer.pipeline.stages.transcribe.WhisperXAdapter", _FakeWhisperXAdapter
    )
    monkeypatch.setattr(
        "diarrhizer.pipeline.stages.diarize.WhisperXDiarizeAdapter", _FakeWhisperXDiarizeAdapter
    )


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path / "call.mp4"
    path.write_bytes(b"fake media")
    # Older than anything the pipeline writes, so mtime checks are unambiguous.
    os.utime(path, (1_000_000, 1_000_000))
    return path


def _stages():
    return [ConvertStage(), TranscribeStage(), DiarizeStage(), MergeStage(), ExportStage()]


def _run(input_file, tmp_path, job_dir=None, **kwargs) -> tuple[dict, Path]:
    result = run_pipeline(
        input_path=input_file,
        out_dir=tmp_path / "out",
        stages=_stages(),
        job_dir=job_dir,
        device="cpu",
        **kwargs,
    )
    return {s["stage"]: s["status"] for s in result["stages"]}, Path(result["job_dir"])


def _audio_files(job_dir: Path) -> list:
    return sorted(p.name for p in (job_dir / "audio").iterdir())


ALL_CACHED = {s: "cached" for s in ("convert", "transcribe", "diarize", "merge", "export")}


def test_default_format_keeps_only_the_working_wav(input_file, tmp_path):
    statuses, job_dir = _run(input_file, tmp_path)

    assert set(statuses.values()) == {"completed"}
    assert _audio_files(job_dir) == ["normalized.wav"]
    assert len(FFMPEG_CALLS) == 1


def test_lossy_format_drops_the_wav_once_the_job_completes(input_file, tmp_path):
    statuses, job_dir = _run(input_file, tmp_path, audio_format="mp3-q5")

    assert set(statuses.values()) == {"completed"}
    # ASR and diarization still got the lossless WAV...
    assert DECODED == [("transcribe", "normalized.wav"), ("diarize", "normalized.wav")]
    # ...but only the archive is left on disk.
    assert _audio_files(job_dir) == ["archive.mp3"]
    archive_cmd = FFMPEG_CALLS[1]
    assert archive_cmd[archive_cmd.index("-c:a") + 1] == "libmp3lame"

    meta = json.loads((job_dir / "meta" / "run.json").read_text(encoding="utf-8"))
    assert meta["config"]["audio_format"]["name"] == "mp3-q5"
    assert meta["pipeline_config"]["audio_format"] == "mp3-q5"
    assert meta["archive_paths"] == [str(job_dir / "audio" / "archive.mp3")]

    # Nothing changed: the dropped WAV must not count as a missing output.
    statuses, _ = _run(input_file, tmp_path, job_dir=job_dir, audio_format="mp3-q5")
    assert statuses == ALL_CACHED


def test_keep_wav_keeps_both_files(input_file, tmp_path):
    _, job_dir = _run(input_file, tmp_path, audio_format="mp3-q5", keep_wav=True)

    assert _audio_files(job_dir) == ["archive.mp3", "normalized.wav"]
    statuses, _ = _run(input_file, tmp_path, job_dir=job_dir, audio_format="mp3-q5", keep_wav=True)
    assert statuses == ALL_CACHED


def test_a_partial_run_keeps_the_wav_for_the_stages_still_to_come(input_file, tmp_path):
    statuses, job_dir = _run(input_file, tmp_path, audio_format="mp3-q5", to_stage="convert")

    assert statuses["convert"] == "completed"
    assert _audio_files(job_dir) == ["archive.mp3", "normalized.wav"]

    statuses, _ = _run(input_file, tmp_path, job_dir=job_dir, audio_format="mp3-q5")
    assert statuses["convert"] == "cached"
    assert DECODED[0] == ("transcribe", "normalized.wav")
    assert _audio_files(job_dir) == ["archive.mp3"]


def test_changing_the_format_of_a_finished_job_does_not_redo_asr(input_file, tmp_path):
    _, job_dir = _run(input_file, tmp_path, audio_format="mp3-q5")
    FFMPEG_CALLS.clear()

    statuses, _ = _run(input_file, tmp_path, job_dir=job_dir, audio_format="opus-24k")

    assert statuses == {**ALL_CACHED, "convert": "completed"}
    # Only the new archive was encoded; the dropped WAV was not brought back.
    assert [Path(cmd[-1]).name for cmd in FFMPEG_CALLS] == ["archive.opus"]
    assert _audio_files(job_dir) == ["archive.opus"]


def test_changing_the_format_reuses_an_existing_wav_untouched(input_file, tmp_path):
    _, job_dir = _run(input_file, tmp_path, audio_format="mp3-q5", keep_wav=True)
    wav = job_dir / "audio" / "normalized.wav"
    wav_mtime = wav.stat().st_mtime_ns
    FFMPEG_CALLS.clear()

    statuses, _ = _run(input_file, tmp_path, job_dir=job_dir, audio_format="wav")

    assert statuses == {**ALL_CACHED, "convert": "completed"}
    assert FFMPEG_CALLS == []
    assert wav.stat().st_mtime_ns == wav_mtime
    assert _audio_files(job_dir) == ["normalized.wav"]


def test_rerun_after_the_wav_was_dropped_decodes_the_archive(input_file, tmp_path):
    _, job_dir = _run(input_file, tmp_path, audio_format="mp3-q5")
    DECODED.clear()

    # A different ASR model forces transcribe to run again.
    statuses, _ = _run(input_file, tmp_path, job_dir=job_dir, audio_format="mp3-q5", asr_model="small")

    assert statuses["convert"] == "cached"
    assert statuses["transcribe"] == "completed"
    assert DECODED == [("transcribe", "archive.mp3")]


def test_a_newer_source_file_rewrites_the_wav_and_invalidates_asr(input_file, tmp_path):
    _, job_dir = _run(input_file, tmp_path, audio_format="mp3-q5")
    DECODED.clear()
    os.utime(input_file, None)  # "the recording was replaced"

    statuses, _ = _run(input_file, tmp_path, job_dir=job_dir, audio_format="mp3-q5")

    assert statuses["convert"] == "completed"
    assert statuses["transcribe"] == "completed"
    assert DECODED[0] == ("transcribe", "normalized.wav")
    assert _audio_files(job_dir) == ["archive.mp3"]


def test_switching_away_from_split_stereo_removes_its_extras(input_file, tmp_path):
    _, job_dir = _run(input_file, tmp_path, audio_format="flac", audio_profile="split-stereo")
    assert _audio_files(job_dir) == ["archive.flac", "archive_left.flac", "archive_right.flac"]
    pan_filters = [cmd[cmd.index("-af") + 1] for cmd in FFMPEG_CALLS if "-af" in cmd]
    assert "pan=mono|c0=c0" in pan_filters and "pan=mono|c0=c1" in pan_filters

    _run(input_file, tmp_path, job_dir=job_dir, audio_format="flac", audio_profile="raw")

    assert _audio_files(job_dir) == ["archive.flac"]


def test_custom_profile_from_the_store(input_file, tmp_path):
    (tmp_path / "audio_formats.json").write_text(
        json.dumps({
            "default": "tiny",
            "profiles": {"tiny": {"codec": "libopus", "extension": "ogg", "bitrate": "12k"}},
        }),
        encoding="utf-8",
    )

    # None = the store's default, exactly what the CLI passes without --audio-format.
    _, job_dir = _run(input_file, tmp_path, audio_format=None)

    # A custom profile keeps the WAV unless it says otherwise (keep_wav defaults to true).
    assert _audio_files(job_dir) == ["archive.ogg", "normalized.wav"]
    cmd = FFMPEG_CALLS[1]
    assert cmd[cmd.index("-b:a") + 1] == "12k"


def test_unknown_format_fails_before_creating_the_job(input_file, tmp_path):
    with pytest.raises(ValueError, match="Unknown audio format"):
        _run(input_file, tmp_path, audio_format="nope")

    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_finalize_never_deletes_the_only_copy(tmp_path):
    job_dir = tmp_path / "job"
    (job_dir / "audio").mkdir(parents=True)
    (job_dir / "meta").mkdir()
    wav = job_dir / "audio" / "normalized.wav"
    wav.write_bytes(b"wav")
    (job_dir / "meta" / "run.json").write_text(
        json.dumps({"config": {
            "audio_profile": "raw",
            "audio_format": {"name": "mp3-q5", "codec": "libmp3lame", "extension": "mp3",
                             "quality": 5, "keep_wav": False},
        }}),
        encoding="utf-8",
    )
    config = PipelineConfig(job_id="job", input_file=str(tmp_path / "x.mp4"))
    job = JobContext(input_path=tmp_path / "x.mp4", job_dir=job_dir, config=config)

    ConvertStage().finalize(job)  # archive.mp3 is missing

    assert wav.exists()
