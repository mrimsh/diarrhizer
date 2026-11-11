"""History screen: lists job folders under a configurable out/ root.

Pure filesystem read via job_scan.py. An unfinished job (e.g. stopped by the
user, or failed at diarization for want of an HF token) gets a "Продолжить"
button that resumes it from the first missing stage with its recorded
settings (GUI:JOB_PLANS) - the work already done is not repeated. Any job gets
a "Перезапустить этап" menu: the chosen stage is recomputed from scratch, and
so are the stages built on it, with the recorded settings - which a dialog
(GUI:RESTART_DIALOG) lets the user change first.
"""

import json
import os
import re
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QItemSelectionModel, QMimeData, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
    QTreeWidget,
    QTreeWidgetItem,
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
    stages_for_mode,
)
from diarrhizer_gui.screens.new_job_screen import hf_token_present
from diarrhizer_gui.screens.restart_dialog import RestartDialog
from diarrhizer_gui.screens.project_panel import ScopeDialog
from diarrhizer_gui.screens.move_jobs_dialog import MoveJobsDialog
from diarrhizer.projects import JobMoveError, job_context, list_scopes, move_jobs, scope_path

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


_SPEAKER_RANGE = re.compile(r"^\s*(\d+)\D+(\d+)\s*$")
FILE_COLUMN = 1
ACTIONS_COLUMN = 6


def column_sort_key(job: JobSummary, column: int):
    """What the table is ordered by in `column`; None = no value (always listed last)."""
    if column == 0:
        return job.sort_key
    if column == 1:
        return None if job.input_name == "—" else job.input_name.casefold()
    if column == 2:
        return None if job.language == "—" else job.language.casefold()
    if column == 3:
        match = _SPEAKER_RANGE.match(job.speakers)
        return (int(match.group(1)), int(match.group(2))) if match else None
    if column == 4:
        return None if job.asr_model == "—" else job.asr_model.casefold()
    if column == 5:
        return sum(1 for done in job.stage_done.values() if done)
    return None


def sort_jobs(jobs: list, column: int, descending: bool) -> list:
    """Stable sort by a column; jobs without a value stay at the end in either direction."""
    keyed = [(column_sort_key(job, column), job) for job in jobs]
    present = sorted((x for x in keyed if x[0] is not None), key=lambda x: x[0], reverse=descending)
    return [job for _, job in present] + [job for key, job in keyed if key is None]


# [SEMANTIC-BEGIN] GUI:JOB_DRAG
# @purpose: Drag selected history jobs onto project/team groups through the shared move dialog
# @description: Only internal drags are accepted; drops emit frozen source paths and destination names.
#   Qt never rearranges tree nodes itself, so cancel/failed disk moves cannot misrepresent the library.
# @inputs: selected-path getter, pointer target group (blank viewport means library root)
# @outputs: move_dropped(paths, project, team)
# @sideEffects: Native drag UI only; no filesystem writes
# @errors: Invalid/external drops are ignored
# @see: GUI:HISTORY_TABLE, GUI:MOVE_JOBS
class JobTree(QTreeWidget):
    move_dropped = Signal(object, str, str)

    def __init__(self, paths):
        super().__init__()
        self._paths = paths
        self._drag_paths = set()
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    def startDrag(self, _actions):
        self._drag_paths = self._paths()
        if not self._drag_paths:
            return
        mime = QMimeData()
        mime.setData("application/x-diarrhizer-jobs", b"jobs")
        drag = QDrag(self)
        drag.setMimeData(mime)
        try:
            drag.exec(Qt.DropAction.MoveAction)
        finally:
            self._drag_paths = set()

    def drop_destination(self, item):
        if item is None:
            return ("", "")
        return item.data(0, Qt.ItemDataRole.UserRole + 2)

    def dragEnterEvent(self, event):
        if event.source() is self and self._drag_paths:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if (event.source() is self and self._drag_paths
                and self.drop_destination(self.itemAt(event.position().toPoint())) is not None):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        destination = self.drop_destination(self.itemAt(event.position().toPoint()))
        if event.source() is not self or not self._drag_paths or destination is None:
            event.ignore()
            return
        paths = set(self._drag_paths)
        event.acceptProposedAction()
        # Open the modal picker after the native drop handler has returned.
        QTimer.singleShot(0, lambda: self.move_dropped.emit(paths, *destination))
# [SEMANTIC-END] GUI:JOB_DRAG


# [SEMANTIC-BEGIN] GUI:HISTORY_TABLE
# @purpose: List the jobs of the results folder with sorting by column and user-set column widths
# @description: self._jobs is always in the order shown, so self._jobs[row] is the job of table row `row`
#   (double click, filter, row actions). QTableWidget's own sorting is off - cell widgets would not move
#   with the rows; a click on a sortable header (all but "Действия") re-sorts self._jobs in Python and
#   rebuilds the table. Default order: date, newest first. Empty values go last in both directions.
#   The sort column/direction and the widths the user dragged are kept in QSettings
#   Default view is a project/team tree; independent toggles hide either grouping or show the flat table.
#   Group settings edit the on-disk scope, including directory renames guarded against queued writers.
#   (settings_keys.HISTORY_*); values that do not fit the table are ignored. "Файл" fills the free width
#   until the user sets its width. Multiple selected jobs survive refresh, sort and grouping changes by path.
#   Selecting a tree group includes its filtered descendants; hidden table rows never join a move.
#   Batch moves recheck queued/busy jobs, preserve artifacts and notify live views of changed paths.
#   Internal drops and the toolbar open the same optional destination-template renaming preview.
#   The separate date/name action edits one job in place, including its recording wall time.
# @inputs: job folders under the results folder; header clicks; header drags
# @outputs: view_result_requested, runs_requested, jobs_moved(old-to-new paths)
# @sideEffects: reads/writes QSettings; deletes or moves job folders on explicit request
# @errors: Bad display settings fall back to defaults; scan/config I/O errors are shown without escaping Qt slots
class HistoryScreen(QWidget):
    COLUMNS = ["Дата", "Файл", "Язык", "Спикеры", "Модель", "Этапы", "Действия"]
    view_result_requested = Signal(Path)
    runs_requested = Signal(list)  # [QueuedRun]
    scope_renamed = Signal(Path, Path)
    jobs_moved = Signal(object)  # dict[Path, Path]; QVariantMap would discard non-string keys.

    def __init__(self, is_busy: Optional[Callable[[Path], bool]] = None, probe=None) -> None:
        super().__init__()
        self._settings = QSettings("Diarrhizer", "DiarrhizerGUI")
        self._is_busy = is_busy or (lambda job_dir: False)
        self._probe = probe  # EnvironmentProbe: restart settings disable cuda when it is unavailable
        self._jobs: list = []  # in display order
        self._scanned: list = []
        self._rebuilding = False
        self._display_grouped = True
        self._sort_column, self._sort_descending = self._load_sort()
        self._user_widths = self._load_widths()
        self._adjusting = False  # our own resizes are not the user's
        self._width_timer = QTimer(self)
        self._width_timer.setSingleShot(True)
        self._width_timer.setInterval(400)
        self._width_timer.timeout.connect(self._save_widths)

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
        self._filter_field.setPlaceholderText("Поиск по заданию, файлу, проекту или команде…")
        self._filter_field.setClearButtonEnabled(True)
        self._filter_field.textChanged.connect(lambda _text: self._apply_filter())
        self._summary_label = QLabel()
        self._summary_label.setStyleSheet("color: #7d8394;")
        filter_row = QHBoxLayout()
        filter_row.addWidget(self._filter_field, stretch=1)
        filter_row.addWidget(self._summary_label)
        self._group_projects = QCheckBox("По проектам")
        self._group_teams = QCheckBox("По командам")
        for box, key in ((self._group_projects, settings_keys.HISTORY_GROUP_PROJECTS),
                         (self._group_teams, settings_keys.HISTORY_GROUP_TEAMS)):
            box.setChecked(self._settings.value(key, True, type=bool))
            box.toggled.connect(self._grouping_changed)
            filter_row.addWidget(box)
        edit_group = QPushButton("Параметры группы…")
        edit_group.clicked.connect(self._edit_group)
        filter_row.addWidget(edit_group)
        self._tree = JobTree(self._selected_job_paths)
        self._tree.move_dropped.connect(self._drop_jobs)
        self._tree.setColumnCount(len(self.COLUMNS))
        self._tree.setHeaderLabels(self.COLUMNS)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemDoubleClicked.connect(self._tree_double_click)
        self._tree.header().sectionClicked.connect(self._on_header_clicked)
        self._tree.header().setSectionsClickable(True)
        self._tree.setColumnWidth(0, 210)
        self._tree.setColumnWidth(1, 270)

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setToolTip("Двойной щелчок по строке открывает результат")
        self._table.cellDoubleClicked.connect(self._on_double_click)
        header = self._table.horizontalHeader()
        header.setMinimumSectionSize(50)
        header.setStretchLastSection(False)
        for column in range(ACTIONS_COLUMN):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(ACTIONS_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        self._show_sort_indicator()
        header.sectionClicked.connect(self._on_header_clicked)
        header.sectionResized.connect(self._on_section_resized)
        self._adjusting = True
        for column, width in self._user_widths.items():
            header.resizeSection(column, width)
        self._adjusting = False
        self._table.viewport().installEventFilter(self)
        self._tree.itemSelectionChanged.connect(self._update_selection)
        self._table.itemSelectionChanged.connect(self._update_selection)
        self._selection_label = QLabel("Выбрано: 0")
        self._move_button = QPushButton("Переместить / переименовать…")
        self._move_button.setEnabled(False)
        self._move_button.clicked.connect(self._move_selected)
        self._rename_button = QPushButton("Изменить дату / имя…")
        self._rename_button.setEnabled(False)
        self._rename_button.clicked.connect(self._rename_selected)
        self._move_status = QLabel()
        self._move_status.setWordWrap(True)
        selection_row = QHBoxLayout()
        selection_row.addWidget(self._selection_label)
        selection_row.addWidget(self._move_button)
        selection_row.addWidget(self._rename_button)
        selection_row.addWidget(self._move_status, 1)
        selection_hint = QLabel("Выделите задания (Ctrl/Shift) и перетащите на проект/команду в дереве "
                               "или нажмите «Переместить / переименовать…».")
        selection_hint.setWordWrap(True)

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
        layout.addLayout(selection_row)
        layout.addWidget(selection_hint)
        layout.addWidget(self._tree, stretch=1)
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

    # --- sorting and widths ----------------------------------------------------

    def _load_sort(self) -> tuple:
        """(column, descending) from QSettings; the default is the date, newest first."""
        try:
            column = int(self._settings.value(settings_keys.HISTORY_SORT_COLUMN))
            order = self._settings.value(settings_keys.HISTORY_SORT_ORDER)
            if 0 <= column < ACTIONS_COLUMN and order in ("asc", "desc"):
                return column, order == "desc"
        except (TypeError, ValueError):
            pass
        return 0, True

    def _load_widths(self) -> dict:
        """{column: width} the user set; anything that does not fit this table means none."""
        try:
            data = json.loads(self._settings.value(settings_keys.HISTORY_COLUMN_WIDTHS) or "")
            if data["columns"] != len(self.COLUMNS):
                return {}
            widths = {int(column): int(width) for column, width in data["widths"].items()}
        except (TypeError, ValueError, KeyError, AttributeError):
            return {}
        if all(0 <= c < ACTIONS_COLUMN and 40 <= w <= 5000 for c, w in widths.items()):
            return widths
        return {}

    def _save_widths(self) -> None:
        data = {"columns": len(self.COLUMNS), "widths": {str(c): w for c, w in self._user_widths.items()}}
        self._settings.setValue(settings_keys.HISTORY_COLUMN_WIDTHS, json.dumps(data))

    def _show_sort_indicator(self) -> None:
        order = Qt.SortOrder.DescendingOrder if self._sort_descending else Qt.SortOrder.AscendingOrder
        self._table.horizontalHeader().setSortIndicator(self._sort_column, order)

    def _on_header_clicked(self, column: int) -> None:
        if column >= ACTIONS_COLUMN:
            return
        if column == self._sort_column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column, self._sort_descending = column, False
        self._show_sort_indicator()
        self._settings.setValue(settings_keys.HISTORY_SORT_COLUMN, self._sort_column)
        self._settings.setValue(settings_keys.HISTORY_SORT_ORDER, "desc" if self._sort_descending else "asc")
        self._populate(self._scanned)

    def _on_section_resized(self, column: int, _old: int, new: int) -> None:
        if self._adjusting or column >= ACTIONS_COLUMN:
            return
        self._user_widths[column] = new
        self._width_timer.start()

    def _fit_widths(self) -> None:
        """Columns the user has not sized: to their contents; "Файл" takes what is left."""
        header = self._table.horizontalHeader()
        self._adjusting = True
        try:
            for column in range(ACTIONS_COLUMN):
                if column != FILE_COLUMN and column not in self._user_widths:
                    self._table.resizeColumnToContents(column)
            if FILE_COLUMN not in self._user_widths:
                others = sum(header.sectionSize(c) for c in range(len(self.COLUMNS)) if c != FILE_COLUMN)
                header.resizeSection(FILE_COLUMN, max(150, self._table.viewport().width() - others))
        finally:
            self._adjusting = False

    def eventFilter(self, obj, event) -> bool:
        if obj is self._table.viewport() and event.type() == QEvent.Type.Resize:
            self._fit_widths()
        return super().eventFilter(obj, event)

    def refresh(self) -> None:
        try:
            self._scanned = scan_jobs(Path(self._path_field.text()))
        except (OSError, ValueError, UnicodeError) as exc:
            self._summary_label.setText(f"Не удалось прочитать историю: {exc}")
            return
        self._populate(self._scanned)

    def _populate(self, jobs: list) -> None:
        """Rebuild the table from `jobs` in the chosen order, keeping the selected job selected."""
        selected = self._selected_job_paths()
        # Keep table selection too for the existing flat-table controls while the tree is shown.
        table_selected = self._table_selected_paths()
        self._rebuilding = True
        jobs = sort_jobs(jobs, self._sort_column, self._sort_descending)
        self._jobs = jobs

        self._table.setRowCount(0)
        self._empty_label.setVisible(not jobs)
        self._table.setVisible(bool(jobs) and not self._grouped())
        self._tree.setVisible(bool(jobs) and self._grouped())

        for row, job in enumerate(jobs):
            busy = self._is_busy(job.job_dir)
            self._table.insertRow(row)
            date_item = QTableWidgetItem(job.date_label)
            self._table.setItem(row, 0, date_item)
            display_name = job.job_id if (job.job_dir / "job.json").is_file() else job.input_name
            name_item = QTableWidgetItem(display_name)
            name_item.setToolTip(str(job.job_dir))
            if not job.complete:
                name_item.setText(f"{display_name}  — {'выполняется' if busy else 'не завершено'}")
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

        self._fit_widths()
        self._display_grouped = self._grouped()
        self._apply_filter(selected)
        self._select_table_paths(table_selected if self._display_grouped else selected)
        self._rebuilding = False
        self._update_selection()

    def _selected_job_id(self) -> Optional[str]:
        rows = self._table.selectionModel().selectedRows()
        if rows and 0 <= rows[0].row() < len(self._jobs):
            return str(self._jobs[rows[0].row()].job_dir)
        return None

    def _apply_filter(self, selected_paths=None) -> None:
        needle = self._filter_field.text().strip().lower()
        shown = 0
        for row, job in enumerate(self._jobs):
            hidden = bool(needle) and needle not in " ".join((job.input_name, job.job_id, job.project, job.team)).lower()
            self._table.setRowHidden(row, hidden)
            shown += not hidden
        unfinished = sum(1 for job in self._jobs if not job.complete)
        text = f"{shown} из {len(self._jobs)}" if needle else f"всего {len(self._jobs)}"
        if unfinished:
            text += f", не завершено {unfinished}"
        self._summary_label.setText(text)
        self._populate_tree(selected_paths)
        self._update_selection()

    def _grouped(self):
        return self._group_projects.isChecked() or self._group_teams.isChecked()

    def _grouping_changed(self):
        self._settings.setValue(settings_keys.HISTORY_GROUP_PROJECTS, self._group_projects.isChecked())
        self._settings.setValue(settings_keys.HISTORY_GROUP_TEAMS, self._group_teams.isChecked())
        self._populate(self._scanned)

    def _populate_tree(self, selected_paths=None):
        selected_paths = self._tree_selected_paths() if selected_paths is None else selected_paths
        rebuilding = self._rebuilding
        self._rebuilding = True
        collapsed = set()
        def remember(item):
            if item.childCount() and not item.isExpanded():
                collapsed.add(item.data(0, Qt.ItemDataRole.UserRole))
            for i in range(item.childCount()):
                remember(item.child(i))
        for i in range(self._tree.topLevelItemCount()):
            remember(self._tree.topLevelItem(i))
        self._tree.clear()
        if not self._grouped():
            self._rebuilding = rebuilding
            return
        groups = {}
        needle = self._filter_field.text().strip().casefold()
        for job in self._jobs:
            if needle and needle not in " ".join((job.input_name, job.job_id, job.project, job.team)).casefold():
                continue
            parent = self._tree.invisibleRootItem()
            project_dir, team_dir = job_context(job.job_dir)
            parts = []
            if self._group_projects.isChecked():
                parts.append(((job.project,), job.project or "Без проекта", project_dir))
            if self._group_teams.isChecked():
                parts.append(((job.project, job.team), job.team or "Без команды", team_dir or project_dir))
            for key, label, directory in parts:
                if key not in groups:
                    group = QTreeWidgetItem([label])
                    group.setData(0, Qt.ItemDataRole.UserRole, str(directory) if directory else repr(key))
                    group.setData(0, Qt.ItemDataRole.UserRole + 1, "scope" if directory else "empty")
                    group.setData(0, Qt.ItemDataRole.UserRole + 2, (job.project, job.team if len(key) > 1 else ""))
                    parent.addChild(group)
                    group.setExpanded(str(directory) not in collapsed)
                    groups[key] = group
                parent = groups[key]
            leaf = QTreeWidgetItem([job.date_label, job.job_id, job.language, job.speakers, job.asr_model])
            leaf.setToolTip(1, f"{job.input_name}\n{job.job_dir}")
            leaf.setData(0, Qt.ItemDataRole.UserRole, str(job.job_dir))
            leaf.setData(0, Qt.ItemDataRole.UserRole + 1, "job")
            parent.addChild(leaf)
            self._tree.setItemWidget(leaf, 5, StageDots(job))
            actions = RowActions(job, self._is_busy(job.job_dir))
            actions.open_result_requested.connect(self.view_result_requested)
            actions.resume_requested.connect(self._resume)
            actions.restart_requested.connect(self._restart)
            actions.delete_requested.connect(self._delete)
            self._tree.setItemWidget(leaf, 6, actions)
            if job.job_dir in selected_paths:
                self._tree.setCurrentItem(leaf, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
                leaf.setSelected(True)
        if not needle:
            root = Path(self._path_field.text())
            for project in list_scopes(root):
                parent = self._tree.invisibleRootItem()
                if self._group_projects.isChecked():
                    key = (project,)
                    if key not in groups:
                        group = QTreeWidgetItem([project])
                        group.setData(0, Qt.ItemDataRole.UserRole, str(root / project))
                        group.setData(0, Qt.ItemDataRole.UserRole + 1, "scope")
                        group.setData(0, Qt.ItemDataRole.UserRole + 2, (project, ""))
                        parent.addChild(group)
                        group.setExpanded(True)
                        groups[key] = group
                    parent = groups[key]
                if self._group_teams.isChecked():
                    for team in list_scopes(root / project, teams=True):
                        key = (project, team)
                        if key not in groups:
                            group = QTreeWidgetItem([team])
                            group.setData(0, Qt.ItemDataRole.UserRole, str(root / project / team))
                            group.setData(0, Qt.ItemDataRole.UserRole + 1, "scope")
                            group.setData(0, Qt.ItemDataRole.UserRole + 2, (project, team))
                            parent.addChild(group)
                            groups[key] = group
        if self._grouped() and self._tree.topLevelItemCount():
            self._tree.show()
            self._empty_label.hide()
        self._tree.resizeColumnToContents(6)
        self._rebuilding = rebuilding

    def _table_selected_paths(self) -> set[Path]:
        return {self._jobs[index.row()].job_dir for index in self._table.selectionModel().selectedRows()
                if 0 <= index.row() < len(self._jobs) and not self._table.isRowHidden(index.row())}

    def _tree_selected_paths(self) -> set[Path]:
        paths = set()
        def collect(item):
            if item.data(0, Qt.ItemDataRole.UserRole + 1) == "job":
                paths.add(Path(item.data(0, Qt.ItemDataRole.UserRole)))
            for index in range(item.childCount()):
                collect(item.child(index))
        for item in self._tree.selectedItems():
            collect(item)
        return paths

    def _selected_job_paths(self) -> set[Path]:
        return self._tree_selected_paths() if self._display_grouped else self._table_selected_paths()

    def _select_table_paths(self, paths):
        self._table.clearSelection()
        selection = self._table.selectionModel()
        for row, job in enumerate(self._jobs):
            if job.job_dir in paths and not self._table.isRowHidden(row):
                index = self._table.model().index(row, 0)
                selection.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
                selection.select(index,
                                 QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    def _select_job_paths(self, paths):
        self._select_table_paths(paths)
        self._tree.clearSelection()
        def select(item):
            if item.data(0, Qt.ItemDataRole.UserRole + 1) == "job" and Path(item.data(0, Qt.ItemDataRole.UserRole)) in paths:
                self._tree.setCurrentItem(item, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
                item.setSelected(True)
                parent = item.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
            for index in range(item.childCount()):
                select(item.child(index))
        for index in range(self._tree.topLevelItemCount()):
            select(self._tree.topLevelItem(index))
        self._update_selection()

    def _update_selection(self):
        if self._rebuilding:
            return
        paths = self._selected_job_paths()
        busy = any(self._is_busy(p) for p in paths)
        self._selection_label.setText(f"Выбрано: {len(paths)}")
        self._move_button.setEnabled(bool(paths) and not busy)
        self._rename_button.setEnabled(len(paths) == 1 and not busy)
        self._move_button.setToolTip("Среди выбранных есть выполняющиеся или ожидающие задания" if busy else
                                    "Переместить выбранные задания в проект и команду")

    def _move_selected(self):
        self._move_paths(self._selected_job_paths())

    def _drop_jobs(self, paths, project, team):
        self._move_paths(paths, (project, team))

    def _rename_selected(self):
        paths = self._selected_job_paths()
        if len(paths) == 1:
            self._move_paths(paths, edit_recording=True)

    def _move_paths(self, paths, destination_hint=None, edit_recording=False):
        if not paths:
            return
        try:
            self._check_move_busy(paths)
            root = Path(self._path_field.text())
            jobs = [job for job in self._jobs if job.job_dir in paths]
            project = jobs[0].project if len({job.project for job in jobs}) == 1 else ""
            team = jobs[0].team if project and len({job.team for job in jobs}) == 1 else ""
            if destination_hint is not None:
                project, team = destination_hint
            dialog = MoveJobsDialog(self, root, len(paths), project, team, jobs=[job.job_dir for job in jobs])
            if edit_recording:
                dialog.edit_recording()
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            project, team = dialog.destination()
            # The queue may have advanced while the dialog was open.
            self._check_move_busy(paths)
            destination = scope_path(root, project, team)
            if self._is_busy(destination):
                raise ValueError("В папке назначения есть выполняющиеся или ожидающие задания")
            moved = move_jobs(root, [job.job_dir for job in jobs], project=project, team=team, **dialog.rename_options())
        except (OSError, ValueError, RuntimeError) as exc:
            if isinstance(exc, JobMoveError) and exc.moved:
                self.jobs_moved.emit(exc.moved)
            QMessageBox.warning(self, "Переместить задания", str(exc))
            self.refresh()
            return
        if moved:
            self.jobs_moved.emit(moved)
        self.refresh()
        self._select_job_paths({moved.get(p.resolve(), p) for p in paths})
        self._move_status.setText(f"Перемещено заданий: {len(moved)}" if moved else
                                  "Выбранные задания уже находятся в этой папке")

    def _check_move_busy(self, paths):
        if any(self._is_busy(p) for p in paths):
            raise ValueError("Нельзя переместить выполняющиеся или ожидающие задания")

    def _tree_double_click(self, item, _column):
        if item.data(0, Qt.ItemDataRole.UserRole + 1) == "job":
            directory = Path(item.data(0, Qt.ItemDataRole.UserRole))
            if any(j.job_dir == directory and j.complete for j in self._jobs):
                self.view_result_requested.emit(directory)

    def _edit_group(self):
        item = self._tree.currentItem()
        if item is None or item.data(0, Qt.ItemDataRole.UserRole + 1) != "scope":
            QMessageBox.information(self, "Параметры группы", "Выберите проект или команду в дереве истории")
            return
        try:
            dialog = ScopeDialog(self, Path(item.data(0, Qt.ItemDataRole.UserRole)), self._is_busy)
            dialog.scope_renamed.connect(self.scope_renamed)
            dialog.exec()
            self.refresh()
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Параметры группы", str(exc))

    def _on_double_click(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._jobs) and self._jobs[row].complete:
            self.view_result_requested.emit(self._jobs[row].job_dir)

    def _resume(self, job: JobSummary) -> None:
        try:
            plan = resume_plan(job.job_dir)
        except (OSError, ValueError, UnicodeError) as exc:
            QMessageBox.warning(self, "Продолжить задание", f"Не удалось прочитать задание: {exc}")
            return
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
        try:
            plan = restart_plan(job.job_dir, stage)
        except (OSError, ValueError, UnicodeError) as exc:
            QMessageBox.warning(self, title, f"Не удалось прочитать задание: {exc}")
            return
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
        try:
            plan = restart_plan(job.job_dir, stage, dialog.overrides())
        except (OSError, ValueError, UnicodeError) as exc:
            QMessageBox.warning(self, title, f"Не удалось подготовить перезапуск: {exc}")
            return
        if plan is None:
            QMessageBox.information(self, title, "Задание изменилось: этот этап больше нельзя перезапустить.")
            self.refresh()
            return
        mode, kwargs = plan
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
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Не удалось удалить", str(exc))
        self.refresh()
# [SEMANTIC-END] GUI:HISTORY_TABLE
