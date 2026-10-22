"""History screen: lists job folders under a configurable out/ root.

Pure filesystem read via job_scan.py. An unfinished job (e.g. stopped by the
user, or failed at diarization for want of an HF token) gets a "Продолжить"
button that resumes it from the first missing stage with its recorded
settings (GUI:JOB_PLANS) - the work already done is not repeated. Any job gets
a "Перезапустить этап" menu: the chosen stage is recomputed from scratch, and
so are the stages built on it, with the recorded settings - which a dialog
(GUI:RESTART_DIALOG) lets the user change first.
"""

import os
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from diarrhizer_gui import settings_keys
from diarrhizer_gui.job_queue import QueuedRun
from diarrhizer_gui.job_scan import (
    SPEAKERS_SET_ASIDE,
    STAGE_ARTIFACTS,
    JobSummary,
    delete_job,
    read_speaker_names,
    restart_drops_speaker_names,
    restart_plan,
    restart_settings,
    restart_stages,
    restartable_stages,
    resume_plan,
    scan_jobs,
    set_aside_speaker_names,
    stages_for_mode,
)
from diarrhizer_gui.screens.new_job_screen import hf_token_present
from diarrhizer_gui.screens.restart_dialog import RestartDialog

STAGE_LABELS = [name[0].upper() for name, _ in STAGE_ARTIFACTS]  # C T D M E
STAGE_TITLES = {
    "convert": "конвертация",
    "transcribe": "распознавание",
    "diarize": "диаризация",
    "merge": "объединение",
    "export": "экспорт",
}

DONE_COLOR = "#2f7d52"
PENDING_COLOR = "#c3c8d3"
MISSING_COLOR = "#d9a441"


class StageDots(QWidget):
    def __init__(self, job: JobSummary) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        for (name, _), letter in zip(STAGE_ARTIFACTS, STAGE_LABELS):
            done = job.stage_done.get(name, False)
            skipped = name == "diarize" and job.mode == "asr_only"
            dot = QLabel(letter)
            dot.setFixedSize(20, 20)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if done:
                color, state = DONE_COLOR, "готово"
            elif skipped:
                color, state = PENDING_COLOR, "не выполнялась (только распознавание)"
            elif job.complete:
                color, state = PENDING_COLOR, "нет"
            else:
                color, state = MISSING_COLOR, "не выполнено"
            dot.setToolTip(f"{STAGE_TITLES[name]}: {state}")
            dot.setStyleSheet(
                f"border-radius: 10px; background: {color}; color: white; "
                f"font-size: 10px; font-weight: 600;"
            )
            layout.addWidget(dot)
        layout.addStretch(1)


class RowActions(QWidget):
    open_result_requested = Signal(Path)
    resume_requested = Signal(object)
    restart_requested = Signal(object, str)  # job, stage name
    delete_requested = Signal(object)

    def __init__(self, job: JobSummary, busy: bool) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        open_folder = QPushButton("Папка")
        open_folder.clicked.connect(lambda: os.startfile(str(job.job_dir)))
        layout.addWidget(open_folder)

        if job.complete:
            open_result = QPushButton("Результат")
            job_dir = job.job_dir
            open_result.clicked.connect(lambda: self.open_result_requested.emit(job_dir))
            layout.addWidget(open_result)
        else:
            resume = QPushButton("Продолжить")
            resume.setEnabled(job.resumable and not busy)
            resume.setToolTip(
                "Выполняется сейчас" if busy
                else "Довыполнить недостающие этапы с исходными настройками задания"
                if job.resumable
                else "Задание не дошло до конца конвертации — запустите его заново"
            )
            resume.clicked.connect(lambda: self.resume_requested.emit(job))
            layout.addWidget(resume)

        layout.addWidget(self._restart_button(job, busy))

        delete = QPushButton("Удалить")
        delete.setEnabled(not busy)
        delete.setToolTip(
            "Задание сейчас выполняется" if busy else "Удалить папку задания вместе со всеми файлами"
        )
        delete.clicked.connect(lambda: self.delete_requested.emit(job))
        layout.addWidget(delete)

        layout.addStretch(1)

    def _restart_button(self, job: JobSummary, busy: bool) -> QPushButton:
        """"Перезапустить этап ▾": one menu entry per stage the job runs; a stage
        whose inputs are missing (an earlier stage has no output) is greyed out."""
        available = restartable_stages(job)
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        for name in stages_for_mode(job.mode):
            action = menu.addAction(STAGE_TITLES[name].capitalize())
            action.setEnabled(name in available)
            action.setToolTip(
                "Заново выполнятся: " + ", ".join(STAGE_TITLES[s] for s in restart_stages(job.mode, name))
                if name in available
                else "Нет результатов предыдущих этапов"
            )
            action.triggered.connect(lambda checked=False, stage=name: self.restart_requested.emit(job, stage))

        button = QPushButton("Перезапустить этап")
        button.setMenu(menu)
        button.setEnabled(bool(available) and not busy)
        button.setToolTip(
            "Выполняется сейчас" if busy
            else "Заново выполнить выбранный этап и зависящие от него, с настройками задания"
            if available
            else "Задание не дошло до конца конвертации — запустите его заново"
        )
        return button


class HistoryScreen(QWidget):
    COLUMNS = ["Дата", "Файл", "Язык", "Спикеры", "Модель", "Этапы", "Действия"]
    view_result_requested = Signal(Path)
    runs_requested = Signal(list)  # [QueuedRun]

    def __init__(self, is_busy: Optional[Callable[[Path], bool]] = None, probe=None) -> None:
        super().__init__()
        self._settings = QSettings("Diarrhizer", "DiarrhizerGUI")
        self._is_busy = is_busy or (lambda job_dir: False)
        self._probe = probe  # EnvironmentProbe: restart settings disable cuda when it is unavailable
        self._jobs: list = []

        title = QLabel("История заданий")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        self._path_field = QLineEdit()
        self._path_field.setReadOnly(True)

        browse_button = QPushButton("Обзор…")
        browse_button.clicked.connect(self._browse)

        refresh_button = QPushButton("Обновить")
        refresh_button.clicked.connect(self.refresh)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Папка результатов:"))
        path_row.addWidget(self._path_field, stretch=1)
        path_row.addWidget(browse_button)
        path_row.addWidget(refresh_button)

        self._filter_field = QLineEdit()
        self._filter_field.setPlaceholderText("Фильтр по имени файла…")
        self._filter_field.setClearButtonEnabled(True)
        self._filter_field.textChanged.connect(self._apply_filter)
        self._summary_label = QLabel()
        self._summary_label.setStyleSheet("color: #7d8394;")
        filter_row = QHBoxLayout()
        filter_row.addWidget(self._filter_field, stretch=1)
        filter_row.addWidget(self._summary_label)

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setToolTip("Двойной щелчок по строке открывает результат")
        self._table.cellDoubleClicked.connect(self._on_double_click)
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
        layout.addLayout(filter_row)
        layout.addWidget(self._table, stretch=1)
        layout.addWidget(self._empty_label)

        default_out = str(Path.cwd() / "out")
        out_dir = self._settings.value(settings_keys.OUT_DIR, default_out)
        self._path_field.setText(out_dir)

        self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # The folder may have been changed on New Job/Settings meanwhile.
        out_dir = self._settings.value(settings_keys.OUT_DIR, self._path_field.text())
        self._path_field.setText(out_dir)
        self.refresh()

    def _browse(self) -> None:
        current = self._path_field.text()
        chosen = QFileDialog.getExistingDirectory(self, "Папка результатов", current)
        if chosen:
            self._path_field.setText(chosen)
            self._settings.setValue(settings_keys.OUT_DIR, chosen)
            self.refresh()

    def refresh(self) -> None:
        out_dir = Path(self._path_field.text())
        jobs = scan_jobs(out_dir)
        self._jobs = jobs

        self._table.setRowCount(0)
        self._empty_label.setVisible(not jobs)
        self._table.setVisible(bool(jobs))

        for row, job in enumerate(jobs):
            busy = self._is_busy(job.job_dir)
            self._table.insertRow(row)
            date_item = QTableWidgetItem(job.date_label)
            self._table.setItem(row, 0, date_item)
            name_item = QTableWidgetItem(job.input_name)
            name_item.setToolTip(str(job.job_dir))
            if not job.complete:
                name_item.setText(f"{job.input_name}  — {'выполняется' if busy else 'не завершено'}")
            self._table.setItem(row, 1, name_item)
            self._table.setItem(row, 2, QTableWidgetItem(job.language))
            self._table.setItem(row, 3, QTableWidgetItem(job.speakers))
            self._table.setItem(row, 4, QTableWidgetItem(job.asr_model))
            self._table.setCellWidget(row, 5, StageDots(job))
            row_actions = RowActions(job, busy)
            row_actions.open_result_requested.connect(self.view_result_requested)
            row_actions.resume_requested.connect(self._resume)
            row_actions.restart_requested.connect(self._restart)
            row_actions.delete_requested.connect(self._delete)
            self._table.setCellWidget(row, 6, row_actions)

        self._apply_filter()

    def _apply_filter(self) -> None:
        needle = self._filter_field.text().strip().lower()
        shown = 0
        for row, job in enumerate(self._jobs):
            hidden = bool(needle) and needle not in job.input_name.lower() and needle not in job.job_id.lower()
            self._table.setRowHidden(row, hidden)
            shown += not hidden
        unfinished = sum(1 for job in self._jobs if not job.complete)
        text = f"{shown} из {len(self._jobs)}" if needle else f"всего {len(self._jobs)}"
        if unfinished:
            text += f", не завершено {unfinished}"
        self._summary_label.setText(text)

    def _on_double_click(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._jobs) and self._jobs[row].complete:
            self.view_result_requested.emit(self._jobs[row].job_dir)

    def _resume(self, job: JobSummary) -> None:
        plan = resume_plan(job.job_dir)
        if plan is None:
            QMessageBox.information(
                self, "Продолжить задание", "Этому заданию нечего довыполнять."
            )
            self.refresh()
            return
        mode, kwargs = plan
        if kwargs["from_stage"] == "convert" and not Path(kwargs["input_path"]).exists():
            QMessageBox.warning(
                self,
                "Продолжить задание",
                f"Нужно заново сконвертировать исходный файл, но его нет на месте:\n"
                f"{kwargs['input_path']}",
            )
            return
        label = f"{job.input_name} (продолжение с этапа «{STAGE_TITLES[kwargs['from_stage']]}»)"
        self.runs_requested.emit([QueuedRun(mode=mode, kwargs=kwargs, label=label)])

    def _restart(self, job: JobSummary, stage: str) -> None:
        title = "Перезапустить этап"
        plan = restart_plan(job.job_dir, stage)
        if plan is None:
            QMessageBox.information(
                self, title, "Этот этап сейчас нельзя перезапустить: нет результатов предыдущих этапов."
            )
            self.refresh()
            return
        mode, kwargs = plan
        rebuilt = restart_stages(mode, stage)
        if "convert" in rebuilt and not Path(kwargs["input_path"]).exists():
            QMessageBox.warning(
                self,
                title,
                f"Нужно заново сконвертировать исходный файл, но его нет на месте:\n{kwargs['input_path']}",
            )
            return
        # Diarization is part of the restart (or of what it invalidates): same
        # check the Result screen makes before "Пересчитать спикеров".
        if "diarize" in rebuilt and not hf_token_present():
            QMessageBox.warning(
                self, title, "Для диаризации нужен HF-токен — задайте его на экране «Настройки»."
            )
            return

        drops_names = restart_drops_speaker_names(mode, stage)
        intro = (
            f"Перезапустить этап «{STAGE_TITLES[stage]}» задания «{job.input_name}» ({job.date_label})?\n\n"
            f"Заново выполнятся: {', '.join(STAGE_TITLES[name] for name in rebuilt)}. "
            "Прежние результаты этих этапов будут удалены."
        )
        if drops_names and read_speaker_names(job.job_dir):
            intro += (
                "\n\nИмена спикеров будут сброшены: новая диаризация нумерует спикеров заново "
                f"(прежние имена сохранятся в {SPEAKERS_SET_ASIDE})."
            )
        dialog = RestartDialog(
            self, intro, restart_settings(mode, stage), kwargs, self._settings, self._probe
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        # Planned again with the user's changes (restart_plan checks them against the stage).
        plan = restart_plan(job.job_dir, stage, dialog.overrides())
        if plan is None:
            QMessageBox.information(self, title, "Задание изменилось: этот этап больше нельзя перезапустить.")
            self.refresh()
            return
        mode, kwargs = plan
        if drops_names:
            set_aside_speaker_names(job.job_dir)
        label = f"{job.input_name} (перезапуск этапа «{STAGE_TITLES[stage]}»)"
        self.runs_requested.emit([QueuedRun(mode=mode, kwargs=kwargs, label=label)])

    def _delete(self, job: JobSummary) -> None:
        answer = QMessageBox.question(
            self,
            "Удалить задание",
            f"Удалить задание «{job.input_name}» ({job.date_label})?\n\n"
            f"Папка {job.job_dir} будет удалена безвозвратно.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            delete_job(Path(self._path_field.text()), job.job_dir)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Не удалось удалить", str(exc))
        self.refresh()
