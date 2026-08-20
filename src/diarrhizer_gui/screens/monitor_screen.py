"""Monitor screen: live view of a running pipeline.

Stage state is inferred from the progress log text (see classify()) because
runner.py/the stage modules only carry the stage name structurally via
`extra={"stage": ...}` - the message itself is still free text. classify()
is a pure function so it can be checked without Qt. Errors are NOT inferred
from logs (runner.py's except-block around stage.run() still uses print(),
not logger) - they arrive only via PipelineWorker's `failed` signal.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from diarrhizer_gui.job_scan import STAGE_ARTIFACTS

STAGE_NAMES = [name for name, _ in STAGE_ARTIFACTS]

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


class StageStep(QWidget):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

        self._dot = QLabel()
        self._dot.setFixedSize(22, 22)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel(name)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.setSpacing(4)
        layout.addWidget(self._dot, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._label)

        self.set_state("pending")

    def set_state(self, state: str) -> None:
        fill, border, label_color = STATE_COLORS[state]
        self._dot.setStyleSheet(
            f"background: {fill}; border: 2px solid {border}; border-radius: 11px;"
        )
        self._label.setStyleSheet(f"color: {label_color}; font-size: 11px;")
        self._dot.setToolTip(f"{self._name}: {STATE_LABELS[state]}")


class MonitorScreen(QWidget):
    back_requested = Signal()
    view_result_requested = Signal(Path)

    def __init__(self) -> None:
        super().__init__()

        title = QLabel("Мониторинг задания")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        self._status_label = QLabel()
        self._status_label.setStyleSheet("color: #565c6b;")

        self._steps: dict[str, StageStep] = {}
        stepper_row = QHBoxLayout()
        for name in STAGE_NAMES:
            step = StageStep(name)
            self._steps[name] = step
            stepper_row.addWidget(step)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))

        self._back_button = QPushButton("Назад к списку заданий")
        self._back_button.clicked.connect(self.back_requested)
        self._back_button.setEnabled(False)

        self._view_result_button = QPushButton("Просмотреть результат")
        self._view_result_button.clicked.connect(self._emit_view_result)
        self._view_result_button.hide()

        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(self._view_result_button)
        footer.addWidget(self._back_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addWidget(self._status_label)
        layout.addLayout(stepper_row)
        layout.addWidget(self._log, stretch=1)
        layout.addLayout(footer)

        self._current_stage: str | None = None
        self._result_job_dir: Path | None = None

    def reset(self, mode: str) -> None:
        """Prepare the screen for a fresh run. `mode` decides which stages
        actually participate ("asr_only" runs only convert+transcribe) -
        the rest are shown as not participating (pending, never updated).
        """
        stages_in_run = set(STAGE_NAMES) if mode == "full" else {"convert", "transcribe"}
        for name, step in self._steps.items():
            step.set_state("pending")
            step.setEnabled(name in stages_in_run)
        self._log.clear()
        self._status_label.setText("Выполняется…")
        self._status_label.setStyleSheet("color: #565c6b;")
        self._back_button.setEnabled(False)
        self._view_result_button.hide()
        self._current_stage = None
        self._result_job_dir = None

    def on_progress(self, stage: str, message: str) -> None:
        self._log.appendPlainText(f"[{stage}] {message}" if stage != "-" else message)

        state = classify(stage, message)
        if state is not None and stage in self._steps:
            self._steps[stage].set_state(state)
            if state == "running":
                self._current_stage = stage

    def on_finished(self, result: dict) -> None:
        duration = result.get("total_duration_seconds")
        if duration is not None:
            self._status_label.setText(f"Готово за {duration:.1f} с")
        else:
            self._status_label.setText("Готово")
        self._back_button.setEnabled(True)

        job_dir = Path(result["job_dir"])
        if (job_dir / "export" / "result.json").exists():
            self._result_job_dir = job_dir
            self._view_result_button.show()

    def _emit_view_result(self) -> None:
        if self._result_job_dir is not None:
            self.view_result_requested.emit(self._result_job_dir)

    def on_failed(self, error_type: str, message: str) -> None:
        if self._current_stage is not None and self._current_stage in self._steps:
            self._steps[self._current_stage].set_state("error")
        self._status_label.setText(f"Ошибка ({error_type}): {message}")
        self._status_label.setStyleSheet("color: #b23b35;")
        self._back_button.setEnabled(True)
