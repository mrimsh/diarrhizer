"""Background execution of run_pipeline() plus a logging-to-Qt-signal bridge.

Heavy imports (diarrhizer.pipeline.*, diarrhizer.adapters.*) stay inside
build_stages()/PipelineWorker.run() - never at module level - so importing
this module (which happens as soon as the GUI wires up New Job/Monitor)
never pulls in torch/whisperx. Same reason cli.py defers those imports:
"so that doctor never pulls in torch/whisperx as a side effect".
"""

import logging

from PySide6.QtCore import QObject, Signal


def build_stages(mode: str) -> list:
    """Build the stage list for a pipeline mode. Imports stage classes lazily."""
    from diarrhizer.pipeline.stages.convert import ConvertStage
    from diarrhizer.pipeline.stages.transcribe import TranscribeStage

    if mode == "asr_only":
        return [ConvertStage(), TranscribeStage()]

    if mode == "full":
        from diarrhizer.pipeline.stages.diarize import DiarizeStage
        from diarrhizer.pipeline.stages.export import ExportStage
        from diarrhizer.pipeline.stages.merge import MergeStage

        return [
            ConvertStage(),
            TranscribeStage(),
            DiarizeStage(),
            MergeStage(),
            ExportStage(),
        ]

    raise ValueError(f"Unknown pipeline mode: {mode!r}")


class SignalBridge(QObject):
    """Lives in the main thread. The worker's logging handler emits through
    this so progress crosses the thread boundary via Qt's normal queued
    signal delivery, instead of a worker thread touching widgets directly.
    """

    progress = Signal(str, str)  # stage, message


class SignalLogHandler(logging.Handler):
    """Routes "diarrhizer" logger records to a SignalBridge - same idea as
    examples/subscribe_to_progress.py's ProgressHandler, but emitting a Qt
    signal instead of printing.
    """

    def __init__(self, bridge: SignalBridge) -> None:
        super().__init__()
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        stage = getattr(record, "stage", "-")
        self._bridge.progress.emit(stage, record.getMessage())


class PipelineWorker(QObject):
    """Runs run_pipeline() off the UI thread.

    Create in the main thread, move to a QThread, connect
    thread.started -> worker.run(). Errors are reported via the failed
    signal rather than parsed from logs: runner.py's own except-block around
    stage.run() still uses print(), not logger, so the log stream carries no
    structured error marker - this catches the same exception cli.py does.
    """

    finished = Signal(dict)
    failed = Signal(str, str)  # error type name, message

    def __init__(self, mode: str, kwargs: dict, bridge: SignalBridge) -> None:
        super().__init__()
        self._mode = mode
        self._kwargs = kwargs
        self._bridge = bridge

    def run(self) -> None:
        from diarrhizer.pipeline.runner import run_pipeline

        logger = logging.getLogger("diarrhizer")
        handler = SignalLogHandler(self._bridge)
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            stages = build_stages(self._mode)
            result = run_pipeline(stages=stages, **self._kwargs)
            self.finished.emit(result)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            self.failed.emit(type(exc).__name__, str(exc))
        except Exception as exc:  # matches cli.py's catch-all for unexpected failures
            self.failed.emit(type(exc).__name__, f"{type(exc).__name__}: {exc}")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
