"""Background environment check shared by every screen that needs to know
what the machine can do (Doctor, and CUDA availability for New Job,
Settings and Models).

The checks import torch and whisperx - seconds when warm, close to a minute
on a cold start - so they run once in a child process at startup instead of
blocking the window from appearing, and the GUI process itself never loads
the ML stack.
"""

from typing import Optional

from PySide6.QtCore import QObject, Signal

from diarrhizer.diagnostics.doctor import is_blocking_failure
from diarrhizer_gui.child_process import ChildProcess
from diarrhizer_gui.child_tasks import run_doctor_task


# [SEMANTIC-BEGIN] GUI:ENVIRONMENT_PROBE
# @purpose: Run the doctor checks off the GUI process and share the results with all screens
# @description: run() spawns run_doctor_task (GUI:CHILD_TASKS); `finished` carries the (name, passed, message)
#   list. cuda_available stays None until the first result, then reflects the "CUDA" check, so screens can
#   enable/disable the cuda device once it is known. A child that dies without reporting yields a single
#   failed pseudo-check instead of an empty screen.
# @inputs: (none)
# @outputs: signals started(), finished(list); properties results, running, cuda_available, failure_count
# @sideEffects: Spawns a child process (imports torch/whisperx there, runs ffmpeg -encoders)
# @errors: None raised
# @see: DIAGNOSTICS:DOCTOR, GUI:CHILD_PROCESS
class EnvironmentProbe(QObject):
    started = Signal()
    finished = Signal(list)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._process: Optional[ChildProcess] = None
        self._results: Optional[list] = None
        self._pending: Optional[list] = None

    @property
    def running(self) -> bool:
        return self._process is not None

    @property
    def results(self) -> Optional[list]:
        return self._results

    @property
    def cuda_available(self) -> Optional[bool]:
        return self._check("CUDA")[0]

    @property
    def cuda_message(self) -> str:
        return self._check("CUDA")[1]

    @property
    def failure_count(self) -> int:
        return sum(1 for name, passed, _ in self._results or [] if is_blocking_failure(name, passed))

    def _check(self, wanted: str) -> tuple:
        for name, passed, message in self._results or []:
            if name == wanted:
                return passed, message
        return None, ""

    def run(self) -> None:
        if self._process is not None:
            return
        self._pending = None
        self._process = ChildProcess(run_doctor_task, (), self)
        self._process.message.connect(self._on_message)
        self._process.exited.connect(self._on_exited)
        self._process.start()
        self.started.emit()

    def _on_message(self, item) -> None:
        if item[0] == "checks":
            self._pending = [tuple(result) for result in item[1]]

    def _on_exited(self, code: int) -> None:
        process, self._process = self._process, None
        if process is not None:
            process.deleteLater()
        if self._pending is None:
            self._pending = [
                (
                    "Диагностика",
                    False,
                    f"Процесс проверки аварийно завершился (код {code}) — вероятно, "
                    "сломана установка torch/whisperx. Запустите `python -m diarrhizer doctor` в консоли.",
                )
            ]
        self._results = self._pending
        self.finished.emit(self._results)
# [SEMANTIC-END] GUI:ENVIRONMENT_PROBE
