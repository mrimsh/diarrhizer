"""Tests that pipeline/stage progress is emitted via `logging`, not print(),
carrying the stage name in each record's `extra` (see PIPELINE:RUNNER,
STAGE:MERGE, STAGE:EXPORT) so a GUI-style logging.Handler can filter/route
per stage without parsing message text.

Uses the real MergeStage/ExportStage (pure Python, no torch/ffmpeg
dependency), same as tests/test_runner.py's fixed_job fixture.
"""

import json
import logging
from pathlib import Path
from unittest import mock

import pytest

from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.stages.merge import MergeStage
from diarrhizer.pipeline.stages.export import ExportStage


def _seed_job_dir(job_dir: Path) -> None:
    """Pre-populate a job dir with the artifacts merge/export expect as inputs."""
    (job_dir / "asr").mkdir(parents=True, exist_ok=True)
    (job_dir / "diar").mkdir(parents=True, exist_ok=True)
    (job_dir / "asr" / "transcript.json").write_text(json.dumps({
        "segments": [{"start": 0, "end": 1, "text": "HELLO"}],
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


class _CollectingHandler(logging.Handler):
    """Minimal handler that just remembers every record it receives, the
    same shape a GUI would plug in to receive live progress.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def collected_records():
    """Attach a collecting handler to the "diarrhizer" package logger for the
    duration of the test and yield the list of records it captured.
    """
    handler = _CollectingHandler()
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("diarrhizer")
    original_level = logger.level
    original_propagate = logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_each_stage_emits_at_least_one_record_with_stage_in_extra(fixed_job, collected_records):
    input_file, out_dir, job_dir = fixed_job

    run_pipeline(
        input_path=input_file,
        out_dir=out_dir,
        stages=[MergeStage(), ExportStage()],
    )

    stages_seen = {
        record.stage for record in collected_records if hasattr(record, "stage")
    }
    assert stages_seen == {"merge", "export"}


def test_no_progress_is_printed_to_stdout(fixed_job, collected_records, capsys):
    """Progress must flow through logging only - nothing left over on stdout
    from the print()-based reporting this replaced.
    """
    input_file, out_dir, job_dir = fixed_job

    run_pipeline(
        input_path=input_file,
        out_dir=out_dir,
        stages=[MergeStage(), ExportStage()],
    )

    captured = capsys.readouterr()
    assert captured.out == ""
