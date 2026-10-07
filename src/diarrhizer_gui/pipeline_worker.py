"""One pipeline job, run in a child process (child_process.py) and reported
through Qt signals.

Nothing heavy is imported here or in child_tasks.py: the stage modules (and
with them torch/whisperx) are imported only inside the child.
"""

from typing import Optional

from PySide6.QtCore import QObject, Signal

from diarrhizer_gui.child_process import ChildProcess
from diarrhizer_gui.child_tasks import build_stages, run_pipeline_task  # noqa: F401  (re-exported)

CRASH_HINT = (
    "Процесс обработки аварийно завершился (код {code}) и не сообщил об ошибке. "
    "Обычно это сбой нативной библиотеки (CUDA/cuDNN/CTranslate2) или нехватка памяти. "
    "Проверьте экран «Доктор», попробуйте устройство cpu или модель поменьше. "
    "Готовые этапы сохранены — задание можно продолжить из истории."
)


# [SEMANTIC-BEGIN] GUI:PIPELINE_WORKER
# @purpose: Run one run_pipeline() job off the GUI process and report progress/outcome as Qt signals
# @description: Wraps ChildProcess + run_pipeline_task. Exactly one of finished/failed/cancelled fires per
#   job. cancel() kills the child at once (mid-stage): every stage writes its outputs atomically, so a killed
#   job keeps the stages that already finished and can be resumed (GUI:JOB_PLANS). A child that dies without
#   reporting (native crash) is turned into failed("ProcessCrashed", ...) with a hint, instead of taking the
#   GUI down with it.
# @inputs: mode ("full" | "asr_only"), run_pipeline() kwargs (picklable)
# @outputs: signals progress(stage, message), finished(result dict), failed(type, message), cancelled()
# @sideEffects: Spawns and may kill a child process that writes the job's artifacts
# @errors: Reported via failed, never raised
# @see: GUI:CHILD_PROCESS, GUI:CHILD_TASKS, PIPELINE:RUNNER
class PipelineWorker(QObject):
    progress = Signal(str, str)  # stage, message
    finished = Signal(dict)
    failed = Signal(str, str)  # error type name, message
    cancelled = Signal()

    def __init__(self, mode: str, kwargs: dict, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._process = ChildProcess(run_pipeline_task, (mode, kwargs), self)
        self._process.message.connect(self._on_message)
        self._process.exited.connect(self._on_exited)
        self._result: Optional[dict] = None
        self._failure: Optional[tuple] = None
        self._cancel_requested = False

    def start(self) -> None:
        self._process.start()

    def cancel(self) -> None:
        self._cancel_requested = True
        self._process.kill()

    def is_running(self) -> bool:
        return self._process.is_running()

    def _on_message(self, item) -> None:
        kind = item[0]
        if kind == "progress":
            self.progress.emit(item[1], item[2])
        elif kind == "finished":
            self._result = item[1]
        elif kind == "failed":
            self._failure = (item[1], item[2])

    def _on_exited(self, code: int) -> None:
        if self._result is not None:
            self.finished.emit(self._result)
        elif self._failure is not None:
            self.failed.emit(*self._failure)
        elif self._cancel_requested:
            self.cancelled.emit()
        else:
            self.failed.emit("ProcessCrashed", CRASH_HINT.format(code=code))
# [SEMANTIC-END] GUI:PIPELINE_WORKER
