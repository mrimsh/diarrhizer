"""Stage pre-flight checks (PIPELINE:RUNNER, STAGE:DIARIZE).

Diarization reads the HF token only when it starts, i.e. after the whole
transcription; preflight() moves that check before any stage runs.
"""

import json
import os

import pytest

from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.stages.diarize import DiarizeStage


@pytest.fixture(autouse=True)
def no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)


class _Recorder:
    """A stage that only records that it ran."""

    NAME = "transcribe"

    def __init__(self):
        self.ran = False

    def run(self, job):
        self.ran = True
        return {"stage": self.NAME, "status": "completed"}

    def is_cache_valid(self, job):
        return False

    def get_artifact_paths(self, job_dir):
        return {}

    def get_output_paths(self, job_dir):
        return {}


def test_missing_token_fails_before_any_stage_runs(tmp_path):
    media = tmp_path / "call.wav"
    media.write_bytes(b"x")
    first = _Recorder()
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="HuggingFace token"):
        run_pipeline(input_path=media, out_dir=out, stages=[first, DiarizeStage()])

    assert first.ran is False
    assert not out.exists() or not any(out.iterdir())


def _seed_cached_diarization(job_dir):
    (job_dir / "audio").mkdir(parents=True)
    (job_dir / "diar").mkdir()
    wav = job_dir / "audio" / "normalized.wav"
    wav.write_bytes(b"x")
    os.utime(wav, (1_000_000, 1_000_000))
    (job_dir / "diar" / "diarization.json").write_text(
        json.dumps({"segments": [], "metadata": {"min_speakers": 1, "max_speakers": 10}}),
        encoding="utf-8",
    )


def test_cached_diarization_needs_no_token(tmp_path):
    job_dir = tmp_path / "out" / "job"
    _seed_cached_diarization(job_dir)

    result = run_pipeline(
        input_path=tmp_path / "gone.wav", out_dir=job_dir.parent, stages=[DiarizeStage()], job_dir=job_dir
    )

    assert result["stages"] == [{"stage": "diarize", "status": "cached"}]


def test_forced_diarization_still_needs_a_token(tmp_path):
    job_dir = tmp_path / "out" / "job"
    _seed_cached_diarization(job_dir)

    with pytest.raises(RuntimeError, match="HuggingFace token"):
        run_pipeline(
            input_path=tmp_path / "gone.wav",
            out_dir=job_dir.parent,
            stages=[DiarizeStage()],
            job_dir=job_dir,
            force_stage="diarize",
        )


def test_diarize_outside_the_stage_range_is_not_checked(tmp_path):
    media = tmp_path / "call.wav"
    media.write_bytes(b"x")
    first = _Recorder()

    run_pipeline(
        input_path=media,
        out_dir=tmp_path / "out",
        stages=[first, DiarizeStage()],
        to_stage="transcribe",
    )

    assert first.ran is True
