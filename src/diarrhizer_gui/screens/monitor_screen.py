"""Monitor screen ("Выполнение"): live view of the running batch.

Stage state is inferred from the progress log text (see classify()) because
runner.py/the stage modules only carry the stage name structurally via
`extra={"stage": ...}` - the message itself is still free text. classify()
is a pure function so it can be checked without Qt. Errors are NOT inferred
from logs - they arrive only via PipelineWorker's `failed` signal.

The screen is a sidebar destination, so a running batch can be left and
come back to at any time; MainWindow drives it through begin_item() /
end_item() / batch_finished().
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diarrhizer_gui.job_queue import (
    CANCELLED,
    DONE,
    FAILED,
    RUNNING,
    STATUS_ICONS,
    STATUS_LABELS,
    QueuedRun,
)
from diarrhizer_gui.job_scan import ASR_ONLY_STAGES, FULL_STAGES

STAGE_NAMES = FULL_STAGES

STAGE_TITLES = {
    "convert": "Конвертация",
    "transcribe": "Распознавание",
    "diarize": "Диаризация",
    "merge": "Объединение",
    "export": "Экспорт",
}

STATE_COLORS = {
    "pending": ("#ffffff", "#c3c8d3", "#7d8394"),  # fill, border, label
    "running": ("#a85520", "#a85520", "#1a1d24"),
    "cached": ("#2f7d52", "#2f7d52", "#1a1d24"),
    "done": ("#2f7d52", "#2f7d52", "#1a1d24"),
    "skipped": ("#ffffff", "#c3c8d3", "#7d8394"),
    "error": ("#b23b35", "#b23b35", "#1a1d24"),
}

STATE_LABELS = {
    "pending": "ожидание",
    "running": "выполняется",
    "cached": "из кэша",
    "done": "готово",
    "skipped": "пропущен",
    "error": "ошибка",
}

STATUS_COLORS = {DONE: "#2f7d52", FAILED: "#b23b35", CANCELLED: "#7d8394", RUNNING: "#a85520"}


def classify(stage: str, message: str) -> str | None:
    """Map one progress line for `stage` to a state, or None if the line
    isn't a state-transition marker (e.g. the pipeline banner lines).
    """
    if f"--- Stage: {stage} (skipped" in message:
        return "skipped"
    if f"Stage {stage}: using cached output" in message:
        return "cached"
    if f"Stage {stage}: running..." in message or f"Stage {stage}: forcing recompute" in message:
        return "running"
    if f"[{stage}] Completed in" in message:
        return "done"
    if f"--- Stage: {stage} ---" in message:
        return "running"
    return None


def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} с"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} мин {secs:02d} с"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes:02d} мин"


class StageStep(QWidget):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

        self._dot = QLabel()
        self._dot.setFixedSize(22, 22)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel(STAGE_TITLES.get(name, name))
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.setSpacing(4)
        layout.addWidget(self._dot, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._label)

        self.set_state("pending")

    def set_state(self, state: str) -> None:
        self.state = state
        fill, border, label_color = STATE_COLORS[state]
        self._dot.setStyleSheet(
            f"background: {fill}; border: 2px solid {border}; border-radius: 11px;"
        )
        self._label.setStyleSheet(f"color: {label_color}; font-size: 11px;")
        self._dot.setToolTip(f"{self._name}: {STATE_LABELS[state]}")


class MonitorScreen(QWidget):
    back_requested = Signal()
    view_result_requested = Signal(Path)
    cancel_requested = Signal()

    def __init__(self) -> None:
        super().__init__()

        title = QLabel("Выполнение")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._steps: dict[str, StageStep] = {}
        stepper_row = QHBoxLayout()
        for name in STAGE_NAMES:
            step = StageStep(name)
            self._steps[name] = step
            stepper_row.addWidget(step)
        self._stepper = QWidget()
        self._stepper.setLayout(stepper_row)

        self._queue_list = QListWidget()
        self._queue_list.setMaximumHeight(130)
        self._queue_list.setToolTip("Двойной щелчок по готовому заданию открывает результат")
        self._queue_list.itemDoubleClicked.connect(self._on_queue_double_click)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setMaximumBlockCount(20000)

        self._stop_button = QPushButton("Остановить")
        self._stop_button.setToolTip(
            "Прервать обработку сразу. Уже готовые этапы сохраняются —\n"
            "задание можно продолжить из «Истории заданий»."
        )
        self._stop_button.clicked.connect(self.cancel_requested)

        self._view_result_button = QPushButton("Просмотреть результат")
        self._view_result_button.clicked.connect(self._emit_view_result)

        self._back_button = QPushButton("К истории заданий")
        self._back_button.clicked.connect(self.back_requested)

        footer = QHBoxLayout()
        footer.addWidget(self._stop_button)
        footer.addStretch(1)
        footer.addWidget(self._view_result_button)
        footer.addWidget(self._back_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addWidget(self._status_label)
        layout.addWidget(self._stepper)
        layout.addWidget(self._queue_list)
        layout.addWidget(self._log, stretch=1)
        layout.addLayout(footer)

        self._current_stage: str | None = None
        self._result_job_dir: Path | None = None
        self._items: list = []
        self.show_idle()

    # --- state ---------------------------------------------------------------

    def show_idle(self) -> None:
        self._set_status(
            "Сейчас ничего не обрабатывается. Запустите задание на экране «Новое задание» "
            "или продолжите незавершённое из «Истории заданий».",
            "#7d8394",
        )
        self._stepper.hide()
        self._queue_list.hide()
        self._stop_button.hide()
        self._view_result_button.hide()

    def begin_item(self, item: QueuedRun, position: int, total: int) -> None:
        """Prepare the stepper for the run that is starting."""
        stages_in_run = set(FULL_STAGES if item.mode == "full" else ASR_ONLY_STAGES)
        for name, step in self._steps.items():
            in_run = name in stages_in_run
            step.set_state("pending" if in_run else "skipped")
            step.setEnabled(in_run)
        self._stepper.show()
        self._stop_button.show()
        self._stop_button.setEnabled(True)
        self._view_result_button.hide()
        self._current_stage = None
        self._result_job_dir = None

        prefix = f"Задание {position} из {total}: " if total > 1 else ""
        self._set_status(f"{prefix}{item.label} — выполняется…", "#565c6b")
        if self._log.blockCount() > 1:
            self._log.appendPlainText("")
        self._log.appendPlainText(f"===== {prefix}{item.label} =====")

    def on_progress(self, stage: str, message: str) -> Optional[str]:
        """Log one line; returns the stage that just started running, if any."""
        self._log.appendPlainText(f"[{stage}] {message}" if stage != "-" else message)

        state = classify(stage, message)
        if state is not None and stage in self._steps:
            self._steps[stage].set_state(state)
            if state == "running":
                self._current_stage = stage
                return stage
        return None

    def end_item(self, item: QueuedRun) -> None:
        """Reflect how the current run ended (item.status/detail are already set)."""
        if item.status == FAILED:
            if self._current_stage in self._steps:
                self._steps[self._current_stage].set_state("error")
            self._log.appendPlainText(f"ОШИБКА: {item.detail}")
        elif item.status == CANCELLED:
            if self._current_stage in self._steps:
                self._steps[self._current_stage].set_state("error")
            self._log.appendPlainText("Остановлено пользователем.")
        elif item.status == DONE and item.job_dir is not None:
            if (item.job_dir / "export" / "result.json").exists():
                self._result_job_dir = item.job_dir

    def batch_finished(self, items: list) -> None:
        self._stop_button.hide()
        done = [item for item in items if item.status == DONE]
        failed = [item for item in items if item.status == FAILED]
        cancelled = [item for item in items if item.status == CANCELLED]

        if len(items) == 1:
            item = items[0]
            if item.status == DONE:
                self._set_status(f"Готово: {item.label}. {item.detail}", "#2f7d52")
            elif item.status == FAILED:
                self._set_status(f"Ошибка: {item.label}\n{item.detail}", "#b23b35")
            else:
                self._set_status(
                    f"Остановлено: {item.label}. Готовые этапы сохранены — "
                    "продолжить можно из «Истории заданий».",
                    "#7d8394",
                )
        else:
            parts = [f"готово {len(done)} из {len(items)}"]
            if failed:
                parts.append(f"с ошибкой {len(failed)}")
            if cancelled:
                parts.append(f"отменено {len(cancelled)}")
            color = "#b23b35" if failed else "#2f7d52" if done else "#7d8394"
            self._set_status("Пакет завершён: " + ", ".join(parts) + ".", color)

        last_done = next((i for i in reversed(done) if i.job_dir is not None), None)
        if last_done is not None and (last_done.job_dir / "export" / "result.json").exists():
            self._result_job_dir = last_done.job_dir
        self._view_result_button.setVisible(self._result_job_dir is not None)

    def set_stopping(self) -> None:
        self._stop_button.setEnabled(False)
        self._set_status("Останавливаю…", "#7d8394")

    def refresh_queue(self, items: list) -> None:
        self._items = list(items)
        self._queue_list.clear()
        for item in items:
            text = f"{STATUS_ICONS[item.status]}  {item.label} — {STATUS_LABELS[item.status]}"
            if item.status == FAILED and item.detail:
                text += f": {item.detail.splitlines()[0][:120]}"
            entry = QListWidgetItem(text)
            entry.setToolTip(item.detail or str(item.job_dir or ""))
            color = STATUS_COLORS.get(item.status)
            if color:
                entry.setForeground(QColor(color))
            self._queue_list.addItem(entry)
        self._queue_list.setVisible(len(items) > 1)

    # --- helpers -------------------------------------------------------------

    def _set_status(self, text: str, color: str) -> None:
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"color: {color};")

    def _emit_view_result(self) -> None:
        if self._result_job_dir is not None:
            self.view_result_requested.emit(self._result_job_dir)

    def _on_queue_double_click(self, entry: QListWidgetItem) -> None:
        row = self._queue_list.row(entry)
        if 0 <= row < len(self._items):
            item = self._items[row]
            if item.status == DONE and item.job_dir is not None:
                self.view_result_requested.emit(item.job_dir)
