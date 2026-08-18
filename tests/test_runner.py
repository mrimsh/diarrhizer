"""Tests for diarrhizer.pipeline.runner: job id generation and cache/force behavior.

Uses the real MergeStage/ExportStage (pure Python, no torch/ffmpeg dependency)
to exercise run_pipeline's caching, forcing, and cascading-invalidation logic
end to end without needing any ML dependencies installed.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from diarrhizer.pipeline.runner import generate_job_id, run_pipeline
from diarrhizer.pipeline.stages.merge import MergeStage
from diarrhizer.pipeline.stages.export import ExportStage


# --- generate_job_id ---------------------------------------------------

def test_generate_job_id_format():
    fixed_now = datetime(2026, 1, 2, 3, 4, 5)
    with mock.patch("diarrhizer.pipeline.runner.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_now
        job_id = generate_job_id("D:/records/meeting.mp4")
    assert job_id == "meeting_20260102_030405"


def test_generate_job_id_uses_stem_only():
    with mock.patch("diarrhizer.pipeline.runner.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 1, 1, 0, 0, 0)
        job_id = generate_job_id("call.recording.mp3")
    assert re.match(r"^call\.recording_\d{8}_\d{6}$", job_id)


# --- run_pipeline: basic validation --------------------------------------

def test_min_speakers_greater_than_max_raises_value_error(tmp_path):
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"fake")
    with pytest.raises(ValueError, match="min_speakers"):
        run_pipeline(
            input_path=input_file,
            out_dir=tmp_path / "out",
            stages=[],
            min_speakers=5,
            max_speakers=2,
        )


def test_missing_input_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            input_path=tmp_path / "does_not_exist.wav",
            out_dir=tmp_path / "out",
            stages=[],
        )


# --- run_pipeline: caching / forcing / cascade, via real merge+export ----

def _seed_job_dir(job_dir: Path, transcript_text: str = "OLD TEXT") -> None:
    """Pre-populate a job dir with the artifacts merge/export expect as inputs."""
    (job_dir / "asr").mkdir(parents=True, exist_ok=True)
    (job_dir / "diar").mkdir(parents=True, exist_ok=True)
    (job_dir / "asr" / "transcript.json").write_text(json.dumps({
        "segments": [{"start": 0, "end": 1, "text": transcript_text}],
        "words": [],
    }), encoding="utf-8")
    (job_dir / "diar" / "diarization.json").write_text(json.dumps({
        "segments": [{"start": 0, "end": 1, "speaker": "Speaker_00"}],
    }), encoding="utf-8")


@pytest.fixture
def fixed_job(tmp_path):
    """A (input_file, out_dir, job_dir) triple with a monkeypatched, stable job id."""
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    job_id = "input_20260101_000000"
    job_dir = out_dir / job_id
    _seed_job_dir(job_dir)
    patcher = mock.patch("diarrhizer.pipeline.runner.generate_job_id", return_value=job_id)
    patcher.start()
    yield input_file, out_dir, job_dir
    patcher.stop()


def _run(input_file, out_dir, **kwargs):
    return run_pipeline(
        input_path=input_file,
        out_dir=out_dir,
        stages=[MergeStage(), ExportStage()],
        **kwargs,
    )


def _statuses(result: dict) -> dict:
    return {s["stage"]: s["status"] for s in result["stages"]}


def test_second_run_with_no_changes_is_fully_cached(fixed_job):
    input_file, out_dir, job_dir = fixed_job
    _run(input_file, out_dir)
    result2 = _run(input_file, out_dir)
    assert _statuses(result2) == {"merge": "cached", "export": "cached"}


def test_updated_transcript_cascades_to_merge_and_export_without_force(fixed_job):
    input_file, out_dir, job_dir = fixed_job
    _run(input_file, out_dir)

    # Simulate transcribe having re-run with different content - no --force
    # flags involved at all, just a newer input artifact.
    transcript_path = job_dir / "asr" / "transcript.json"
    new_mtime = transcript_path.stat().st_mtime + 5
    transcript_path.write_text(json.dumps({
        "segments": [{"start": 0, "end": 1, "text": "NEW TEXT"}],
        "words": [],
    }), encoding="utf-8")
    import os
    os.utime(transcript_path, (new_mtime, new_mtime))

    result2 = _run(input_file, out_dir)
    assert _statuses(result2) == {"merge": "completed", "export": "completed"}

    segments = json.loads((job_dir / "merged" / "segments.json").read_text(encoding="utf-8"))
    assert segments["segments"][0]["text"] == "NEW TEXT"
    md = (job_dir / "export" / "result.md").read_text(encoding="utf-8")
    assert "NEW TEXT" in md


def test_force_stage_merge_does_not_delete_its_own_inputs(fixed_job):
    """Regression test: --force-stage merge used to delete transcript.json and
    diarization.json (merge's *inputs*, owned by other stages) because the
    force-deletion loop walked get_artifact_paths() instead of
    get_output_paths(). That made the very next line in merge's run() raise
    FileNotFoundError, crashing the whole pipeline.
    """
    input_file, out_dir, job_dir = fixed_job
    _run(input_file, out_dir)

    transcript_path = job_dir / "asr" / "transcript.json"
    diar_path = job_dir / "diar" / "diarization.json"
    assert transcript_path.exists() and diar_path.exists()

    result2 = _run(input_file, out_dir, force_stage="merge")

    assert transcript_path.exists(), "merge's input transcript.json must survive --force-stage merge"
    assert diar_path.exists(), "merge's input diarization.json must survive --force-stage merge"
    # Forcing merge rewrites segments.json with a fresh mtime, which should in
    # turn cascade into export recomputing too (export's cached result.md/.json
    # are now older than their segments.json input).
    assert _statuses(result2) == {"merge": "completed", "export": "completed"}


def test_force_stage_export_does_not_delete_its_input_segments(fixed_job):
    input_file, out_dir, job_dir = fixed_job
    _run(input_file, out_dir)

    segments_path = job_dir / "merged" / "segments.json"
    assert segments_path.exists()

    result2 = _run(input_file, out_dir, force_stage="export")

    assert segments_path.exists(), "export's input segments.json must survive --force-stage export"
    assert _statuses(result2)["export"] == "completed"
    # merge's own cache was untouched and still valid, so it should be reported as cached
    assert _statuses(result2)["merge"] == "cached"


def test_global_force_recomputes_every_stage(fixed_job):
    input_file, out_dir, job_dir = fixed_job
    _run(input_file, out_dir)

    result2 = _run(input_file, out_dir, force=True)

    assert _statuses(result2) == {"merge": "completed", "export": "completed"}


# --- run_pipeline: --job-dir resume ----------------------------------------

def test_job_dir_resume_recovers_input_path_from_meta(tmp_path):
    """--job-dir lets a run be resumed without re-passing the input file, by
    recovering it from a prior convert stage's meta/run.json - the mechanism
    that makes e.g. --from-stage export a real "re-export only" feature.
    """
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    job_dir = out_dir / "resumed_job"
    _seed_job_dir(job_dir)
    (job_dir / "meta").mkdir(parents=True, exist_ok=True)
    (job_dir / "meta" / "run.json").write_text(
        json.dumps({"input_path": str(input_file)}), encoding="utf-8"
    )

    result = run_pipeline(
        out_dir=out_dir,
        stages=[MergeStage(), ExportStage()],
        job_dir=job_dir,
    )

    assert result["job_id"] == "resumed_job"
    assert _statuses(result) == {"merge": "completed", "export": "completed"}


def test_job_dir_resume_without_input_or_meta_raises_clear_error(tmp_path):
    out_dir = tmp_path / "out"
    job_dir = out_dir / "no_meta_job"
    _seed_job_dir(job_dir)

    with pytest.raises(FileNotFoundError, match="input_path is required"):
        run_pipeline(
            out_dir=out_dir,
            stages=[MergeStage(), ExportStage()],
            job_dir=job_dir,
        )


def test_job_dir_explicit_input_overrides_meta(tmp_path):
    """An explicitly passed input_path wins over whatever meta/run.json
    recorded, so a moved/renamed source file can be re-pointed at.
    """
    old_input = tmp_path / "old.wav"
    old_input.write_bytes(b"fake")
    new_input = tmp_path / "new.wav"
    new_input.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    job_dir = out_dir / "repointed_job"
    _seed_job_dir(job_dir)
    (job_dir / "meta").mkdir(parents=True, exist_ok=True)
    (job_dir / "meta" / "run.json").write_text(
        json.dumps({"input_path": str(old_input)}), encoding="utf-8"
    )

    run_pipeline(
        input_path=new_input,
        out_dir=out_dir,
        stages=[MergeStage(), ExportStage()],
        job_dir=job_dir,
    )

    data = json.loads((job_dir / "export" / "result.json").read_text(encoding="utf-8"))
    assert data["metadata"]["input_file"] == str(new_input)


# --- run_pipeline: --from-stage / --to-stage -------------------------------

def test_to_stage_merge_skips_export(fixed_job):
    input_file, out_dir, job_dir = fixed_job
    _run(input_file, out_dir)

    result2 = _run(input_file, out_dir, to_stage="merge")

    assert _statuses(result2) == {"merge": "cached", "export": "skipped"}


def test_from_stage_export_skips_merge(fixed_job):
    input_file, out_dir, job_dir = fixed_job
    _run(input_file, out_dir)

    result2 = _run(input_file, out_dir, from_stage="export")

    assert _statuses(result2) == {"merge": "skipped", "export": "cached"}


class _ExplodingStage:
    """Stub earlier-than-merge stage that fails the test if the runner ever
    calls it. Used to prove --from-stage genuinely skips earlier stages
    entirely (never even cache-checks them), rather than merely happening to
    start at index 0 in a reduced pipeline.
    """

    NAME = "explode"

    def run(self, job):
        raise AssertionError("should have been skipped by --from-stage")

    def is_cache_valid(self, job_dir):
        raise AssertionError("should have been skipped by --from-stage")

    def get_artifact_paths(self, job_dir):
        return {}

    def get_output_paths(self, job_dir):
        return {}


def test_from_stage_merge_skips_earlier_stage_and_fails_clearly_on_empty_job_dir(tmp_path):
    """Requirement: selecting a stage range whose first active stage's
    job-dir inputs don't exist yet must fail with a clear, actionable error
    (e.g. "please run the transcribe stage first") - not a confusing crash -
    while genuinely skipping (never calling) the stages before it.
    """
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"fake")
    out_dir = tmp_path / "out"
    job_dir = out_dir / "empty_job"
    job_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Transcript not found"):
        run_pipeline(
            input_path=input_file,
            out_dir=out_dir,
            stages=[_ExplodingStage(), MergeStage(), ExportStage()],
            job_dir=job_dir,
            from_stage="merge",
        )


def test_from_stage_after_to_stage_raises_value_error(fixed_job):
    input_file, out_dir, job_dir = fixed_job

    with pytest.raises(ValueError, match="--from-stage"):
        _run(input_file, out_dir, from_stage="export", to_stage="merge")


def test_unknown_stage_name_not_in_supplied_stages_raises_value_error(fixed_job):
    """"convert" is a real pipeline stage, but this test's reduced pipeline
    (like _run's) only supplies [MergeStage(), ExportStage()], so it must be
    rejected as unknown *for this stages sequence* rather than silently
    accepted - the validation is against the stages actually passed in, not
    a hardcoded stage list.
    """
    input_file, out_dir, job_dir = fixed_job

    with pytest.raises(ValueError, match="Unknown --from-stage 'convert'"):
        _run(input_file, out_dir, from_stage="convert")


def test_force_stage_outside_range_has_no_effect(fixed_job):
    """--force-stage naming a stage that --from-stage/--to-stage excludes
    from the active range must not touch that stage at all: range-skipping
    happens before the should_force check in the runner loop.
    """
    input_file, out_dir, job_dir = fixed_job
    _run(input_file, out_dir)
    segments_path = job_dir / "merged" / "segments.json"
    mtime_before = segments_path.stat().st_mtime

    result2 = _run(input_file, out_dir, from_stage="export", force_stage="merge")

    assert _statuses(result2) == {"merge": "skipped", "export": "cached"}
    assert segments_path.stat().st_mtime == mtime_before
