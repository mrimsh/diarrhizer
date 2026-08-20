"""History screen: lists job folders under a configurable out/ root.

Pure filesystem read via job_scan.py - no pipeline execution involved, so
there's nothing to run in a background thread here.
"""

import os
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from diarrhizer_gui import settings_keys
from diarrhizer_gui.job_scan import STAGE_ARTIFACTS, JobSummary, scan_jobs

STAGE_LABELS = [name[0].upper() for name, _ in STAGE_ARTIFACTS]  # C T D M E

DONE_COLOR = "#2f7d52"
PENDING_COLOR = "#c3c8d3"


class StageDots(QWidget):
    def __init__(self, stage_done: dict) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        for (name, _), letter in zip(STAGE_ARTIFACTS, STAGE_LABELS):
            done = stage_done.get(name, False)
            dot = QLabel(letter)
            dot.setFixedSize(20, 20)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            color = DONE_COLOR if done else PENDING_COLOR
            dot.setToolTip(f"{name}: {'готово' if done else 'нет'}")
            dot.setStyleSheet(
                f"border-radius: 10px; background: {color}; color: white; "
                f"font-size: 10px; font-weight: 600;"
            )
            layout.addWidget(dot)
        layout.addStretch(1)


class RowActions(QWidget):
    open_result_requested = Signal(Path)

    def __init__(self, job: JobSummary) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        open_folder = QPushButton("Папка")
        open_folder.clicked.connect(lambda: os.startfile(str(job.job_dir)))
        layout.addWidget(open_folder)

        open_result = QPushButton("Результат")
        open_result.setEnabled(job.result_path is not None)
        if job.result_path is not None:
            job_dir = job.job_dir
            open_result.clicked.connect(lambda: self.open_result_requested.emit(job_dir))
        layout.addWidget(open_result)

        layout.addStretch(1)


class HistoryScreen(QWidget):
    COLUMNS = ["Дата", "Файл", "Язык", "Спикеры", "Модель", "Этапы", "Действия"]
    view_result_requested = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self._settings = QSettings("Diarrhizer", "DiarrhizerGUI")

        title = QLabel("История заданий")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        self._path_field = QLineEdit()
        self._path_field.setReadOnly(True)

        browse_button = QPushButton("Обзор…")
        browse_button.clicked.connect(self._browse)

        refresh_button = QPushButton("Обновить")
        refresh_button.clicked.connect(self._refresh)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Папка результатов:"))
        path_row.addWidget(self._path_field, stretch=1)
        path_row.addWidget(browse_button)
        path_row.addWidget(refresh_button)

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        self._empty_label = QLabel(
            "Заданий пока нет — запустите обработку или укажите другую папку."
        )
        self._empty_label.setStyleSheet("color: #7d8394;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addLayout(path_row)
        layout.addWidget(self._table, stretch=1)
        layout.addWidget(self._empty_label)

        default_out = str(Path.cwd() / "out")
        out_dir = self._settings.value(settings_keys.OUT_DIR, default_out)
        self._path_field.setText(out_dir)

        self._refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh()

    def _browse(self) -> None:
        current = self._path_field.text()
        chosen = QFileDialog.getExistingDirectory(self, "Папка результатов", current)
        if chosen:
            self._path_field.setText(chosen)
            self._settings.setValue(settings_keys.OUT_DIR, chosen)
            self._refresh()

    def _refresh(self) -> None:
        out_dir = Path(self._path_field.text())
        jobs = scan_jobs(out_dir)

        self._table.setRowCount(0)
        self._empty_label.setVisible(not jobs)
        self._table.setVisible(bool(jobs))

        for row, job in enumerate(jobs):
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(job.date_label))
            self._table.setItem(row, 1, QTableWidgetItem(job.input_name))
            self._table.setItem(row, 2, QTableWidgetItem(job.language))
            self._table.setItem(row, 3, QTableWidgetItem(job.speakers))
            self._table.setItem(row, 4, QTableWidgetItem(job.asr_model))
            self._table.setCellWidget(row, 5, StageDots(job.stage_done))
            row_actions = RowActions(job)
            row_actions.open_result_requested.connect(self.view_result_requested)
            self._table.setCellWidget(row, 6, row_actions)
