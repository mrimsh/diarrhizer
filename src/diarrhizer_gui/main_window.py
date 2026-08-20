"""Main application window: sidebar navigation + stacked screens."""

from PySide6.QtCore import QSize, QThread
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from diarrhizer_gui.pipeline_worker import PipelineWorker, SignalBridge
from diarrhizer_gui.screens.doctor_screen import DoctorScreen
from diarrhizer_gui.screens.history_screen import HistoryScreen
from diarrhizer_gui.screens.monitor_screen import MonitorScreen
from diarrhizer_gui.screens.new_job_screen import NewJobScreen
from diarrhizer_gui.screens.placeholder_screen import PlaceholderScreen

NAV_ITEMS = [
    ("Доктор", "doctor"),
    ("Модели", "models"),
    ("Новое задание", "new_job"),
    ("История заданий", "history"),
    ("Настройки", "settings"),
]

HISTORY_ROW = [key for _label, key in NAV_ITEMS].index("history")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Diarrhizer")
        self.resize(1100, 720)

        self._nav = QListWidget()
        self._nav.setObjectName("NavList")
        self._nav.setFixedWidth(220)

        self._stack = QStackedWidget()
        self._build_screens()

        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setCurrentRow(0)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._nav)
        layout.addWidget(self._stack, stretch=1)
        self.setCentralWidget(central)

        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._bridge: SignalBridge | None = None
        # Tracked separately from self._thread: thread.finished triggers
        # thread.deleteLater(), which destroys the underlying C++ QThread
        # object independently of Python's refcounting. self._thread would
        # then be a dangling wrapper - calling .isRunning() on it raises
        # "Internal C++ object already deleted", not a clean False.
        self._job_running = False

    def _build_screens(self) -> None:
        self._new_job = NewJobScreen()
        self._new_job.run_requested.connect(self._on_run_requested)

        self._monitor = MonitorScreen()
        self._monitor.back_requested.connect(self._go_to_history)

        screens = {
            "doctor": DoctorScreen(),
            "models": PlaceholderScreen("Модели"),
            "new_job": self._new_job,
            "history": HistoryScreen(),
            "settings": PlaceholderScreen("Настройки"),
        }
        for label, key in NAV_ITEMS:
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(0, 40))
            self._nav.addItem(item)
            self._stack.addWidget(screens[key])

        # Not a nav destination - only reachable by starting a run from
        # "Новое задание"; the nav row selection is left wherever it was.
        self._stack.addWidget(self._monitor)

    def _go_to_history(self) -> None:
        self._nav.setCurrentRow(HISTORY_ROW)

    def _on_run_requested(self, mode: str, kwargs: dict) -> None:
        if self._job_running:
            return
        self._job_running = True

        self._monitor.reset(mode)
        self._stack.setCurrentWidget(self._monitor)

        bridge = SignalBridge()
        worker = PipelineWorker(mode, kwargs, bridge)
        thread = QThread(self)

        worker.moveToThread(thread)
        bridge.progress.connect(self._monitor.on_progress)
        worker.finished.connect(self._monitor.on_finished)
        worker.failed.connect(self._monitor.on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._bridge = bridge
        self._worker = worker
        self._thread = thread
        thread.start()

    def _on_thread_finished(self) -> None:
        self._job_running = False
        self._thread = None
        self._worker = None
        self._bridge = None
