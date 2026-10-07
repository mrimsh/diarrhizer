"""Tests for the GUI's child-process plumbing (GUI:CHILD_PROCESS,
GUI:PIPELINE_WORKER, GUI:ENVIRONMENT_PROBE) with real spawned processes.

Targets live at module level here because a spawned child imports them by
module name (multiprocessing passes the parent's sys.path along).
"""

import json
import os
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402

from diarrhizer_gui.child_process import ChildProcess  # noqa: E402
from diarrhizer_gui.pipeline_worker import PipelineWorker  # noqa: E402

TIMEOUT_MS = 60_000


def _wait_for(signal, timeout_ms=TIMEOUT_MS):
    """Run the event loop until `signal` fires; return its arguments."""
    loop = QEventLoop()
    received = []

    def on_signal(*args):
        received.append(args)
        loop.quit()

    signal.connect(on_signal)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    signal.disconnect(on_signal)
    assert received, "timed out waiting for the signal"
    return received[0]


def report_values(queue, *values):
    for value in values:
        queue.put(("value", value))


def sleep_forever(queue, *_args):
    queue.put(("started",))
    while True:
        time.sleep(0.1)


def crash_hard(queue, *_args):
    os._exit(3)


def test_messages_arrive_in_order_before_exit(qapp):
    child = ChildProcess(report_values, ("a", "b", "c"))
    messages = []
    child.message.connect(messages.append)
    child.start()
    (code,) = _wait_for(child.exited)
    assert code == 0
    assert messages == [("value", "a"), ("value", "b"), ("value", "c")]
    assert not child.is_running()


def test_kill_stops_a_running_child(qapp):
    child = ChildProcess(sleep_forever)
    child.start()
    assert _wait_for(child.message) == (("started",),)
    child.kill()
    (code,) = _wait_for(child.exited)
    assert code != 0


def test_worker_reports_a_silent_crash_as_failure(qapp, monkeypatch):
    import diarrhizer_gui.pipeline_worker as worker_module

    monkeypatch.setattr(worker_module, "run_pipeline_task", crash_hard)
    worker = PipelineWorker("full", {})
    worker.start()
    error_type, message = _wait_for(worker.failed)
    assert error_type == "ProcessCrashed"
    assert "(код 3)" in message


def test_worker_cancel_emits_cancelled(qapp, monkeypatch):
    import diarrhizer_gui.pipeline_worker as worker_module

    monkeypatch.setattr(worker_module, "run_pipeline_task", sleep_forever)
    worker = PipelineWorker("full", {})
    worker.start()
    QTimer.singleShot(1500, worker.cancel)
    assert _wait_for(worker.cancelled) == ()


def _seed_export_job(job_dir):
    (job_dir / "merged").mkdir(parents=True)
    (job_dir / "meta").mkdir()
    (job_dir / "merged" / "segments.json").write_text(
        json.dumps({"segments": [
            {"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "Привет."},
            {"start": 1, "end": 2, "speaker_id": "Speaker_01", "text": "Здравствуйте."},
        ]}),
        encoding="utf-8",
    )
    (job_dir / "meta" / "run.json").write_text(
        json.dumps({"input_path": str(job_dir / "call.wav")}), encoding="utf-8"
    )


def test_worker_runs_the_real_pipeline_in_a_child(qapp, tmp_path):
    job_dir = tmp_path / "out" / "call_20260101_000000"
    _seed_export_job(job_dir)
    worker = PipelineWorker(
        "full",
        {
            "job_dir": job_dir,
            "out_dir": job_dir.parent,
            "from_stage": "export",
            "to_stage": "export",
            "speakers": {"Speaker_00": "Иван"},
        },
    )
    progress = []
    worker.progress.connect(lambda stage, message: progress.append((stage, message)))
    worker.start()
    (result,) = _wait_for(worker.finished)

    assert result["job_dir"] == str(job_dir)
    assert "Иван: Привет." in (job_dir / "export" / "result.txt").read_text(encoding="utf-8")
    assert (job_dir / "export" / "result.srt").exists()
    assert any(stage == "export" for stage, _ in progress)


def test_worker_reports_pipeline_errors(qapp, tmp_path):
    worker = PipelineWorker("full", {"out_dir": tmp_path, "input_path": tmp_path / "missing.wav"})
    worker.start()
    error_type, message = _wait_for(worker.failed)
    assert error_type == "FileNotFoundError"
    assert "missing.wav" in message
