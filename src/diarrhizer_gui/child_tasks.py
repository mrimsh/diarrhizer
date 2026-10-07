"""Work the GUI runs in a spawned child process (see child_process.py).

Qt-free on purpose: a spawned child starts a fresh interpreter and imports
this module to find its target, so everything here must be cheap to import.
The heavy imports (torch, whisperx, pyannote) happen inside the targets,
i.e. only in the child - the GUI process never loads the ML stack.

Each target's first argument is a multiprocessing queue; the child reports
through it with small picklable tuples:
    ("progress", stage, message)       - one pipeline log line
    ("finished", result_dict)          - run_pipeline() returned
    ("failed", error_type, message)    - run_pipeline() raised
    ("checks", [(name, passed, msg)])  - doctor results
"""

import logging
import os
import sys


def build_stages(mode: str) -> list:
    """Build the stage list for a pipeline mode. Imports stage classes lazily."""
    from diarrhizer.pipeline.stages.convert import ConvertStage
    from diarrhizer.pipeline.stages.export import ExportStage
    from diarrhizer.pipeline.stages.merge import MergeStage
    from diarrhizer.pipeline.stages.transcribe import TranscribeStage

    if mode == "asr_only":
        return [ConvertStage(), TranscribeStage(), MergeStage(), ExportStage()]

    if mode == "full":
        from diarrhizer.pipeline.stages.diarize import DiarizeStage

        return [
            ConvertStage(),
            TranscribeStage(),
            DiarizeStage(),
            MergeStage(),
            ExportStage(),
        ]

    raise ValueError(f"Unknown pipeline mode: {mode!r}")


def _ensure_std_streams() -> None:
    """A child of pythonw.exe has no console: sys.stdout/stderr are None, and
    tqdm progress bars (model downloads, pyannote) crash writing to None."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


class _QueueLogHandler(logging.Handler):
    """Forwards "diarrhizer" log records (stage progress) to the parent."""

    def __init__(self, queue) -> None:
        super().__init__()
        self._queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put(("progress", getattr(record, "stage", "-"), record.getMessage()))


# [SEMANTIC-BEGIN] GUI:CHILD_TASKS
# @purpose: Entry points for the GUI's child processes: one pipeline job, or the doctor checks
# @description: run_pipeline_task() runs run_pipeline() with the given stage mode and kwargs and reports
#   progress/outcome over the queue; it catches the same exceptions as the CLI, so only a hard crash
#   (e.g. a native CUDA/cuDNN abort) ends the process without a final message - the parent reports that
#   as a crash. run_doctor_task() runs doctor.run_all_checks() in a fresh interpreter, so the GUI never
#   imports torch/whisperx itself and a re-check sees packages installed since the GUI started.
# @inputs: queue, mode + run_pipeline() kwargs (picklable) | queue
# @outputs: tuples on the queue (see module docstring)
# @sideEffects: Everything run_pipeline does (files, GPU, model downloads); doctor's imports and probes
# @errors: None raised - reported over the queue
# @see: GUI:CHILD_PROCESS, GUI:PIPELINE_WORKER, PIPELINE:RUNNER, DIAGNOSTICS:DOCTOR
def run_pipeline_task(queue, mode: str, kwargs: dict) -> None:
    _ensure_std_streams()
    from diarrhizer.pipeline.runner import run_pipeline

    logger = logging.getLogger("diarrhizer")
    logger.addHandler(_QueueLogHandler(queue))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        result = run_pipeline(stages=build_stages(mode), **kwargs)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        queue.put(("failed", type(exc).__name__, str(exc)))
        return
    except Exception as exc:  # matches cli.py's catch-all for unexpected failures
        queue.put(("failed", type(exc).__name__, f"{type(exc).__name__}: {exc}"))
        return
    queue.put(("finished", result))


def run_doctor_task(queue) -> None:
    _ensure_std_streams()
    from diarrhizer.diagnostics import doctor

    queue.put(("checks", doctor.run_all_checks()))
# [SEMANTIC-END] GUI:CHILD_TASKS
