"""One model call - a summary or a provider check - run in a child process
(child_process.py) and reported through Qt signals.

Nothing heavy is imported: the summary modules are stdlib-only.
"""

from typing import Callable, Optional

from PySide6.QtCore import QObject, Signal

from diarrhizer_gui.child_process import ChildProcess

CRASH_HINT = (
    "Процесс обращения к модели аварийно завершился (код {code}) и не сообщил об ошибке."
)


# [SEMANTIC-BEGIN] GUI:LLM_WORKER
# @purpose: Run one summary / provider-check task off the GUI process and report its outcome as Qt signals
# @description: Wraps ChildProcess around one of the child_tasks targets (run_summary_task,
#   run_provider_check_task). Exactly one of finished/failed/cancelled fires. cancel() kills the child
#   and everything it started - including a subscription CLI's own processes - at once; nothing is written
#   for a cancelled summary, because the reply is stored only after the provider answered. A child that
#   dies without reporting becomes failed("ProcessCrashed", ...).
# @inputs: target (a child_tasks function), its picklable arguments
# @outputs: signals finished(result dict), failed(type, message), cancelled()
# @sideEffects: Spawns and may kill a child process that talks to the provider
# @errors: Reported via failed, never raised
# @see: GUI:CHILD_PROCESS, GUI:CHILD_TASKS, GUI:SUMMARY_DIALOG, GUI:SUMMARY_SETTINGS
class LlmWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str, str)  # error type name, message
    cancelled = Signal()

    def __init__(self, target: Callable, args: tuple, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._process = ChildProcess(target, args, self)
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
        if item[0] == "finished":
            self._result = item[1]
        elif item[0] == "failed":
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
# [SEMANTIC-END] GUI:LLM_WORKER
