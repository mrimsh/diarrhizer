"""Main application window: sidebar navigation + stacked screens, and the
job queue every screen hands its runs to (New Job, History's "continue",
Result's "re-run diarization")."""

import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from diarrhizer_gui.environment import EnvironmentProbe
from diarrhizer_gui.job_queue import CANCELLED, DONE, FAILED, RUNNING, JobQueue
from diarrhizer_gui.pipeline_worker import PipelineWorker
from diarrhizer_gui.screens.doctor_screen import DoctorScreen
from diarrhizer_gui.screens.history_screen import HistoryScreen
from diarrhizer_gui.screens.models_screen import ModelsScreen
from diarrhizer_gui.screens.monitor_screen import STAGE_TITLES, MonitorScreen, format_elapsed
from diarrhizer_gui.screens.new_job_screen import NewJobScreen
from diarrhizer_gui.screens.result_screen import ResultScreen
from diarrhizer_gui.screens.settings_screen import SettingsScreen

NAV_ITEMS = [
    ("Новое задание", "new_job"),
    ("Выполнение", "monitor"),
    ("История заданий", "history"),
    ("Модели", "models"),
    ("Доктор", "doctor"),
    ("Настройки", "settings"),
]
NAV_LABELS = dict((key, label) for label, key in NAV_ITEMS)
ROWS = {key: row for row, (_label, key) in enumerate(NAV_ITEMS)}

APP_TITLE = "Diarrhizer"


# [SEMANTIC-BEGIN] GUI:MAIN_WINDOW
# @purpose: Host the screens and run every pipeline job the GUI starts, one at a time
# @description: Screens emit lists of QueuedRun; enqueue() adds them to the JobQueue and starts the next one
#   in a PipelineWorker (child process). New jobs get their folder from new_job_dir() when they start, so
#   the window knows each job's folder up front (History won't delete a folder that is being written).
#   The Monitor screen is a sidebar item, so a running batch can be left and revisited; the sidebar label,
#   the window title and a taskbar flash report progress and completion. Closing the window while a job
#   runs asks first and kills the child. Environment checks run once at startup in the background
#   (GUI:ENVIRONMENT_PROBE); the Doctor item shows a warning badge when a required check failed.
# @inputs: user actions on the screens
# @outputs: pipeline jobs, screen navigation
# @sideEffects: Spawns/kills child processes; files dropped on the window are added to New Job
# @errors: None raised; job failures are shown on the Monitor screen
# @see: GUI:JOB_QUEUE, GUI:PIPELINE_WORKER, GUI:ENVIRONMENT_PROBE, GUI:JOB_PLANS
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1150, 760)
        self.setAcceptDrops(True)

        self._probe = EnvironmentProbe(self)
        self._queue = JobQueue()
        self._worker: Optional[PipelineWorker] = None
        self._stop_after_current = False
        self._item_started_at = 0.0
        self._current_stage: Optional[str] = None

        self._nav = QListWidget()
        self._nav.setObjectName("NavList")
        self._nav.setFixedWidth(220)

        self._stack = QStackedWidget()
        self._build_screens()

        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)

        self._job_status = QPushButton()
        self._job_status.setObjectName("JobStatusButton")
        self._job_status.setCursor(Qt.CursorShape.PointingHandCursor)
        self._job_status.clicked.connect(self._show_monitor)
        self._job_status.hide()

        credit = QLabel("By MRIMSH, L. L. MRIMSH and all other LLMs")
        credit.setObjectName("CreditLabel")
        credit.setAlignment(Qt.AlignCenter)
        credit.setWordWrap(True)

        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        sidebar_layout.addWidget(self._nav, stretch=1)
        sidebar_layout.addWidget(self._job_status)
        sidebar_layout.addWidget(credit)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(sidebar)
        layout.addWidget(self._stack, stretch=1)
        self.setCentralWidget(central)

        self._probe.started.connect(self._on_probe_started)
        self._probe.finished.connect(self._on_probe_finished)
        self._nav.setCurrentRow(ROWS["new_job"])
        # After the window is up: the checks import torch/whisperx in a child.
        QTimer.singleShot(0, self._probe.run)

    def _build_screens(self) -> None:
        self._new_job = NewJobScreen(self._probe)
        self._new_job.runs_requested.connect(self.enqueue)

        self._monitor = MonitorScreen()
        self._monitor.back_requested.connect(self._go_to_history)
        self._monitor.view_result_requested.connect(self._show_result)
        self._monitor.cancel_requested.connect(self._on_cancel_requested)

        self._history = HistoryScreen(is_busy=self._is_job_busy)
        self._history.view_result_requested.connect(self._show_result)
        self._history.runs_requested.connect(self.enqueue)

        self._result = ResultScreen()
        self._result.back_requested.connect(self._go_to_history)
        self._result.runs_requested.connect(self.enqueue)

        self._doctor = DoctorScreen(self._probe)
        self._settings_screen = SettingsScreen(self._probe)
        self._settings_screen.defaults_changed.connect(self._new_job.apply_defaults)

        screens = {
            "new_job": self._new_job,
            "monitor": self._monitor,
            "history": self._history,
            "models": ModelsScreen(self._probe),
            "doctor": self._doctor,
            "settings": self._settings_screen,
        }
        for label, key in NAV_ITEMS:
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(0, 40))
            self._nav.addItem(item)
            self._stack.addWidget(screens[key])

        # Not a nav destination - reached from History, Monitor or a finished
        # job; the nav row selection is left wherever it was.
        self._stack.addWidget(self._result)

    # --- navigation ----------------------------------------------------------

    def _go_to_history(self) -> None:
        self._select(ROWS["history"])

    def _show_monitor(self) -> None:
        self._select(ROWS["monitor"])

    def _select(self, row: int) -> None:
        if self._nav.currentRow() == row:
            # Coming back from the Result screen: currentRowChanged won't fire.
            self._stack.setCurrentIndex(row)
        else:
            self._nav.setCurrentRow(row)

    def _show_result(self, job_dir) -> None:
        self._result.load(Path(job_dir))
        self._stack.setCurrentWidget(self._result)

    def _set_nav_label(self, key: str, text: str) -> None:
        self._nav.item(ROWS[key]).setText(text)

    # --- environment checks --------------------------------------------------

    def _on_probe_started(self) -> None:
        self._set_nav_label("doctor", f"{NAV_LABELS['doctor']}  …")

    def _on_probe_finished(self, _results: list) -> None:
        failures = self._probe.failure_count
        label = NAV_LABELS["doctor"] + (f"  ⚠ {failures}" if failures else "")
        self._set_nav_label("doctor", label)
        self._nav.item(ROWS["doctor"]).setToolTip(
            f"Проблем в окружении: {failures}" if failures else "Окружение в порядке"
        )

    # --- drag & drop: files dropped anywhere go to New Job ---------------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        added = self._new_job.add_paths(paths)
        if added:
            self._select(ROWS["new_job"])
        event.acceptProposedAction()

    # --- queue ---------------------------------------------------------------

    def _is_job_busy(self, job_dir: Path) -> bool:
        try:
            return job_dir.resolve() in self._queue.busy_job_dirs()
        except OSError:
            return False

    def enqueue(self, runs: list) -> None:
        if not runs:
            return
        was_active = self._queue.active()
        self._queue.add(runs)
        self._monitor.refresh_queue(self._queue.items)
        self._show_monitor()
        if not was_active:
            self._start_next()
        self._update_progress_labels()

    def _start_next(self) -> None:
        item = self._queue.next_pending()
        if item is None:
            self._on_batch_finished()
            return

        if not item.kwargs.get("job_dir"):
            # Light import (no torch): the runner module itself only needs the stdlib.
            from diarrhizer.pipeline.runner import new_job_dir

            item.kwargs["job_dir"] = new_job_dir(item.kwargs["out_dir"], item.kwargs["input_path"])
            item.job_dir = Path(item.kwargs["job_dir"])

        item.status = RUNNING
        self._item_started_at = time.monotonic()
        self._current_stage = None
        position, total = self._queue.position(item)
        self._monitor.begin_item(item, position, total)
        self._monitor.refresh_queue(self._queue.items)

        worker = PipelineWorker(item.mode, item.kwargs, self)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_item_finished)
        worker.failed.connect(self._on_item_failed)
        worker.cancelled.connect(self._on_item_cancelled)
        self._worker = worker
        try:
            worker.start()
        except OSError as exc:
            self._end_item(FAILED, f"Не удалось запустить процесс обработки: {exc}")
            return
        self._new_job.set_busy(True)
        self._update_progress_labels()

    def _on_progress(self, stage: str, message: str) -> None:
        started = self._monitor.on_progress(stage, message)
        if started is not None:
            self._current_stage = started
            self._update_progress_labels()

    def _on_item_finished(self, result: dict) -> None:
        duration = result.get("total_duration_seconds")
        detail = f"Заняло {format_elapsed(duration)}." if duration is not None else ""
        self._end_item(DONE, detail)

    def _on_item_failed(self, error_type: str, message: str) -> None:
        self._end_item(FAILED, message if error_type == "ProcessCrashed" else f"{error_type}: {message}")

    def _on_item_cancelled(self) -> None:
        self._end_item(CANCELLED, "Остановлено пользователем.")

    def _end_item(self, status: str, detail: str) -> None:
        item = self._queue.current()
        if item is not None:
            item.status = status
            item.detail = detail
            self._monitor.end_item(item)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._stop_after_current:
            self._stop_after_current = False
            self._queue.cancel_pending()
        self._monitor.refresh_queue(self._queue.items)
        self._history.refresh()
        self._start_next()

    def _on_batch_finished(self) -> None:
        self._monitor.batch_finished(self._queue.items)
        self._monitor.refresh_queue(self._queue.items)
        self._new_job.set_busy(False)
        self._update_progress_labels()
        counts = self._queue.counts()
        if counts[FAILED]:
            self._job_status.setText(f"✗ Ошибок: {counts[FAILED]} — подробнее")
        elif counts[DONE]:
            self._job_status.setText("✓ Обработка завершена")
        else:
            self._job_status.setText("■ Обработка остановлена")
        self._job_status.show()
        # Flash the taskbar button: long jobs are usually left running in the background.
        QApplication.alert(self)

    def _on_cancel_requested(self) -> None:
        if self._worker is None:
            return
        pending = self._queue.pending_count()
        if pending:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle("Остановить обработку")
            box.setText(
                f"Остановить текущее задание? В очереди ещё {pending}.\n\n"
                "Готовые этапы сохранятся — задание можно продолжить из «Истории заданий»."
            )
            stop_all = box.addButton("Остановить всё", QMessageBox.ButtonRole.AcceptRole)
            skip = box.addButton("Только текущее", QMessageBox.ButtonRole.ActionRole)
            box.addButton("Не останавливать", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked is stop_all:
                self._stop_after_current = True
            elif clicked is not skip:
                return
        else:
            answer = QMessageBox.question(
                self,
                "Остановить обработку",
                "Остановить текущее задание?\n\n"
                "Готовые этапы сохранятся — задание можно продолжить из «Истории заданий».",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        if self._worker is not None:
            self._monitor.set_stopping()
            self._worker.cancel()

    def _update_progress_labels(self) -> None:
        item = self._queue.current()
        if item is None:
            self.setWindowTitle(APP_TITLE)
            self._set_nav_label("monitor", NAV_LABELS["monitor"])
            return
        position, total = self._queue.position(item)
        counter = f" {position}/{total}" if total > 1 else ""
        stage = STAGE_TITLES.get(self._current_stage, "запуск")
        self.setWindowTitle(f"{stage}{counter} · {item.label} — {APP_TITLE}")
        self._set_nav_label("monitor", f"{NAV_LABELS['monitor']}  ▶{counter}")
        self._job_status.setText(f"▶ {stage}{counter}: {item.label}")
        self._job_status.setToolTip("Открыть экран «Выполнение»")
        self._job_status.show()

    # --- closing -------------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            answer = QMessageBox.question(
                self,
                "Идёт обработка",
                "Задание ещё выполняется. Прервать его и выйти?\n\n"
                "Готовые этапы сохранятся — задание можно будет продолжить из «Истории заданий».",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._queue.cancel_pending()
            self._worker.cancel()
        event.accept()
# [SEMANTIC-END] GUI:MAIN_WINDOW
