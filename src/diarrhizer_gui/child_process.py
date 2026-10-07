"""A spawned child process the Qt event loop can watch and kill.

Why a process and not a QThread: a QThread cannot be stopped mid-stage, a
native crash inside CTranslate2/cuDNN takes the whole GUI down with it, and
the ML stack's memory (RAM and VRAM) is only fully returned when the process
that loaded it exits. A child process fixes all three.

The parent never blocks: a QTimer drains the child's queue on the GUI thread
and re-emits each item as a Qt signal.
"""

import multiprocessing
import queue as queue_module
import subprocess
import sys
from typing import Callable, Optional

from PySide6.QtCore import QObject, QTimer, Signal

POLL_INTERVAL_MS = 100

# Children that have not exited yet; kill_all() stops them when the GUI quits.
_LIVE: set = set()


def _kill_tree(pid: int) -> None:
    """Kill a process and everything it started (FFmpeg runs as a grandchild)."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


# [SEMANTIC-BEGIN] GUI:CHILD_PROCESS
# @purpose: Run a function in a spawned child process and relay what it reports to the Qt event loop
# @description: target(queue, *args) runs in a fresh interpreter ("spawn" - the only safe start method next to
#   Qt). A QTimer polls the queue on the GUI thread and emits `message` per item; once the process is gone and
#   the queue drained, `exited` fires exactly once with the exit code. kill() stops the whole process tree
#   (taskkill /T on Windows) - FFmpeg is a grandchild and would otherwise keep writing into the job folder.
#   Not a daemon on purpose: a daemonic process may not start multiprocessing children of its own, which a
#   future torch/pyannote data loader could do; instead kill_all() (wired to QApplication.aboutToQuit) stops
#   every child still running when the GUI exits, so none outlives it or blocks interpreter shutdown.
# @inputs: target (a module-level function, importable in the child), picklable args
# @outputs: Qt signals message(object), exited(int)
# @sideEffects: Starts/kills OS processes
# @errors: start() propagates OSError if the process cannot be spawned
# @see: GUI:CHILD_TASKS, GUI:PIPELINE_WORKER, GUI:ENVIRONMENT_PROBE
class ChildProcess(QObject):
    message = Signal(object)
    exited = Signal(int)

    def __init__(self, target: Callable, args: tuple = (), parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._target = target
        self._args = args
        self._context = multiprocessing.get_context("spawn")
        self._queue = None
        self._process = None
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._poll)
        self._done = False

    def start(self) -> None:
        self._queue = self._context.Queue()
        self._process = self._context.Process(
            target=self._target, args=(self._queue, *self._args), daemon=False
        )
        self._process.start()
        _LIVE.add(self)
        self._timer.start()

    def is_running(self) -> bool:
        return self._process is not None and not self._done

    def kill(self) -> None:
        if self._process is None or not self._process.is_alive():
            return
        _kill_tree(self._process.pid)
        if self._process.is_alive():
            self._process.terminate()

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue_module.Empty:
                return
            except Exception:
                # A killed child can leave half a message in the pipe.
                return
            self.message.emit(item)

    def _poll(self) -> None:
        if self._done:
            return
        self._drain()
        if self._process.is_alive():
            return
        # Anything the child put right before exiting is in the pipe by now.
        self._drain()
        self._timer.stop()
        self._done = True
        _LIVE.discard(self)
        self._process.join(timeout=1)
        code = self._process.exitcode
        self.exited.emit(code if code is not None else -1)
# [SEMANTIC-END] GUI:CHILD_PROCESS


def kill_all() -> None:
    """Kill every child process that is still running (GUI shutdown)."""
    for child in list(_LIVE):
        child.kill()
