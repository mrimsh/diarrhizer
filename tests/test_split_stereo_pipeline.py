"""End-to-end test for the split-stereo audio profile through the full pipeline.

Exercises convert -> transcribe -> diarize -> merge -> export with
audio_profile="split-stereo", using fakes for the FFmpeg subprocess calls and
the WhisperX ASR/diarization adapters, so it runs without a real ffmpeg
binary, torch, whisperx, or pyannote installed. Modeled on tests/test_runner.py's
approach of driving run_pipeline() directly and asserting on stage statuses.

Regression coverage: split-stereo used to only write normalized_left.wav/
normalized_right.wav, leaving transcribe (which unconditionally reads
audio/normalized.wav) to fail with FileNotFoundError, and ConvertStage's
is_cache_valid() to always report stale for this profile since it only ever
checked normalized.wav. Both are exercised here: the first run must reach
export successfully, and the second (unchanged) run must report every stage,
including convert, as cached.
"""

import json
import subprocess
from pathlib import Path

import pytest

from diarrhizer.adapters import ffmpeg as ffmpeg_module
from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.stages.convert import ConvertStage
from diarrhizer.pipeline.stages.transcribe import TranscribeStage
from diarrhizer.pipeline.stages.diarize import DiarizeStage
from diarrhizer.pipeline.stages.merge import MergeStage
from diarrhizer.pipeline.stages.export import ExportStage


def _fake_ffmpeg_run(cmd, capture_output=True, text=True, check=True, timeout=None):
    """Stand-in for subprocess.run that materializes whatever output file the
    command was going to write (last arg) instead of invoking real ffmpeg.
    """
    output_path = Path(cmd[-1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake wav data")
    return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")


class _FakeWhisperXAdapter:
    """Stand-in for WhisperXAdapter that skips loading any real ASR model."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio_path, language=None):
        return {
            "text": "hello world",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
            "words": [
                {"start": 0.0, "end": 0.5, "word": "hello"},
                {"start": 0.5, "end": 1.0, "word": "world"},
            ],
            "language": "en",
        }


class _FakeWhisperXDiarizeAdapter:
    """Stand-in for WhisperXDiarizeAdapter that skips loading pyannote and
    doesn't require an HF token.
    """

    def __init__(self, *args, **kwargs):
        pass

    def diarize(self, audio_path):
        return {
            "segments": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
            "num_speakers": 1,
            "speakers": ["SPEAKER_00"],
        }


@pytest.fixture
def fake_heavy_deps(monkeypatch):
    """Replace FFmpeg subprocess calls and the WhisperX ASR/diarization
    adapters with fakes, so the pipeline runs end to end without ffmpeg,
    torch, whisperx, or pyannote actually installed/available.
    """
    monkeypatch.setattr(ffmpeg_module, "resolve_ffmpeg_path", lambda explicit=None: "ffmpeg")
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", _fake_ffmpeg_run)
    monkeypatch.setattr(
        "diarrhizer.pipeline.stages.transcribe.WhisperXAdapter", _FakeWhisperXAdapter
    )
    monkeypatch.setattr(
        "diarrhizer.pipeline.stages.diarize.WhisperXDiarizeAdapter", _FakeWhisperXDiarizeAdapter
    )


def _statuses(result: dict) -> dict:
    return {s["stage"]: s["status"] for s in result["stages"]}


def _stages():
    return [ConvertStage(), TranscribeStage(), DiarizeStage(), MergeStage(), ExportStage()]


def test_split_stereo_runs_end_to_end_through_export(tmp_path, fake_heavy_deps):
    input_file = tmp_path / "call.mp4"
    input_file.write_bytes(b"fake media")
    out_dir = tmp_path / "out"

    result = run_pipeline(
        input_path=input_file,
        out_dir=out_dir,
        stages=_stages(),
        device="cpu",
        audio_profile="split-stereo",
    )

    assert _statuses(result) == {
        "convert": "completed",
        "transcribe": "completed",
        "diarize": "completed",
        "merge": "completed",
        "export": "completed",
    }

    job_dir = Path(result["job_dir"])

    # convert: standard mono downmix always present, plus split-stereo extras
    assert (job_dir / "audio" / "normalized.wav").exists()
    assert (job_dir / "audio" / "normalized_left.wav").exists()
    assert (job_dir / "audio" / "normalized_right.wav").exists()

    meta = json.loads((job_dir / "meta" / "run.json").read_text(encoding="utf-8"))
    assert meta["config"]["audio_profile"] == "split-stereo"
    assert len(meta["output_paths"]) == 3

    # transcribe/diarize/merge/export all ran against the mono downmix as usual
    transcript = json.loads((job_dir / "asr" / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["text"] == "hello world"

    diarization = json.loads((job_dir / "diar" / "diarization.json").read_text(encoding="utf-8"))
    assert diarization["num_speakers"] == 1

    segments = json.loads((job_dir / "merged" / "segments.json").read_text(encoding="utf-8"))
    assert segments["num_segments"] == 1
    assert segments["segments"][0]["speaker_id"] == "SPEAKER_00"
    assert segments["segments"][0]["text"] == "hello world"

    md = (job_dir / "export" / "result.md").read_text(encoding="utf-8")
    assert "hello world" in md

    # Re-running with no changes must be fully cached, including convert -
    # the is_cache_valid() fix: split-stereo's own extra artifacts (left/
    # right) must be recognized as part of the cache, not just the shared
    # normalized.wav.
    result2 = run_pipeline(
        input_path=input_file,
        out_dir=out_dir,
        stages=_stages(),
        device="cpu",
        audio_profile="split-stereo",
        job_dir=job_dir,
    )
    assert _statuses(result2) == {
        "convert": "cached",
        "transcribe": "cached",
        "diarize": "cached",
        "merge": "cached",
        "export": "cached",
    }
