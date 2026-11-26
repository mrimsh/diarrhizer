"""Inline editor for the result screen's saved segment list."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.export.turns import format_timestamp
from diarrhizer.transcript import (
    change_speaker,
    change_text,
    merge_segments,
    speaker_id,
    split_segment,
)


def _time(seconds):
    millis = round(seconds * 1000)
    return f"{format_timestamp(millis // 1000)}.{millis % 1000:03d}"


# [SEMANTIC-BEGIN] GUI:TRANSCRIPT_EDITOR
# @purpose: Edit a result's individual timed segments with the recording available for reference.
# @description: Only the selected row owns input widgets. Draft edits replace individual dictionaries.
#   Cursor positions come from the Qt document, including Unicode characters outside the BMP.
# @inputs: saved segment list, job speakers and a callable returning the player's position in seconds
# @outputs: draft segments and requests to play, save or discard
# @sideEffects: Qt dialogs for split time and final speaker; no file writes
# @see: TRANSCRIPT:EDIT, GUI:RESULT_SCREEN, TRANSCRIPT:REVISION
class TranscriptEditor(QWidget):
    save_requested = Signal()
    cancel_requested = Signal()
    play_requested = Signal(float, float)

    def __init__(self, player_position):
        super().__init__()
        self.player_position = player_position
        self.segments = []
        self.original = []
        self.speakers = {}
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Начало", "Конец", "Спикер", "Текст реплики"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.currentCellChanged.connect(self._select_row)
        self.text_field = None
        self.speaker_field = None
        self.split = QPushButton("Разделить реплику")
        self.split.clicked.connect(self._split)
        self.merge = QPushButton("Объединить реплики")
        self.merge.setToolTip("Объединить выбранную реплику со следующей")
        self.merge.clicked.connect(self._merge)
        self.play = QPushButton("▶ Реплика")
        self.play.clicked.connect(self._play)
        save = QPushButton("Сохранить изменения")
        save.clicked.connect(self.save_requested)
        cancel = QPushButton("Отменить изменения")
        cancel.clicked.connect(self.cancel_requested)
        actions = QHBoxLayout()
        for button in (self.play, self.split, self.merge):
            actions.addWidget(button)
        actions.addStretch()
        actions.addWidget(save)
        actions.addWidget(cancel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addLayout(actions)

    def begin(self, segments, speakers, can_play):
        # Legacy readers default omitted bounds to zero. Manual snapshots require both fields.
        self.original = [{"start": 0, "end": 0, **seg} for seg in segments]
        self.segments = list(self.original)
        self.speakers = speakers
        self.play.setEnabled(can_play)
        self._populate(0)

    def clear(self):
        self.original = []
        self.segments = []
        self.speakers = {}
        self._populate(0)

    @property
    def dirty(self):
        return self.segments != self.original

    def _populate(self, selected):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.segments))
        self.text_field = self.speaker_field = None
        for row, seg in enumerate(self.segments):
            for col, value in enumerate((_time(seg.get("start", 0)), _time(seg.get("end", 0)),
                                         self.speakers[speaker_id(seg)], seg.get("text", ""))):
                self.table.setItem(row, col, QTableWidgetItem(value))
        self.table.blockSignals(False)
        if self.segments:
            self.table.setCurrentCell(selected, 3)

    def _select_row(self, row, column, previous, previous_column):
        if row == previous or row < 0:
            return
        if previous >= 0:
            self.table.removeCellWidget(previous, 2)
            self.table.removeCellWidget(previous, 3)
            self.table.setRowHeight(previous, self.table.verticalHeader().defaultSectionSize())
        seg = self.segments[row]
        field = self.text_field = QPlainTextEdit()
        field.setTabChangesFocus(True)
        field.setPlainText(seg.get("text", ""))
        field.textChanged.connect(lambda: self._text_changed(row, field.toPlainText()))
        self.table.setCellWidget(row, 3, field)
        combo = self.speaker_field = QComboBox()
        for sid, label in self.speakers.items():
            combo.addItem(label, sid)
        combo.setCurrentIndex(combo.findData(speaker_id(seg)))
        combo.currentIndexChanged.connect(lambda: self._speaker_changed(row, combo.currentData()))
        self.table.setCellWidget(row, 2, combo)
        self.table.setRowHeight(row, max(64, field.fontMetrics().height() * 3))
        self.merge.setEnabled(row + 1 < len(self.segments))
        self.split.setEnabled(seg.get("end", 0) > seg.get("start", 0))

    def _text_changed(self, row, text):
        self.segments[row] = change_text(self.segments[row], text)
        self.table.item(row, 3).setText(text)

    def _speaker_changed(self, row, sid):
        self.segments[row] = change_speaker(self.segments[row], sid)
        self.table.item(row, 2).setText(self.speakers[sid])

    def _play(self):
        seg = self.segments[self.table.currentRow()]
        self.play_requested.emit(seg["start"], seg["end"])

    def _split(self):
        row = self.table.currentRow()
        seg = self.segments[row]
        # QTextCursor counts UTF-16 units. Measuring its selected prefix gives a Python index.
        cursor = self.text_field.textCursor()
        cursor.clearSelection()
        cursor.setPosition(0, QTextCursor.MoveMode.KeepAnchor)
        offset = len(cursor.selectedText())
        start, end = seg["start"], seg["end"]
        initial = self.player_position()
        if initial is None or not start < initial < end:
            initial = (start + end) / 2
        try:
            split_segment(seg, offset, initial)
            at, ok = QInputDialog.getDouble(self, "Разделить реплику",
                                            f"Момент разреза в секундах ({_time(start)} – {_time(end)})",
                                            initial, start, end, 6)
            if not ok:
                return
            left, right = split_segment(seg, offset, at)
        except ValueError as exc:
            QMessageBox.warning(self, "Разделить реплику", str(exc))
            return
        self.segments[row] = left
        # An overlapping voice may start between the two halves.
        next_row = row + 1
        while next_row < len(self.segments) and self.segments[next_row]["start"] < at:
            next_row += 1
        self.segments.insert(next_row, right)
        self._populate(next_row)

    def _merge(self):
        row = self.table.currentRow()
        first, second = self.segments[row:row + 2]
        sid = speaker_id(first)
        if sid != speaker_id(second):
            ids = list(self.speakers)
            labels = list(self.speakers.values())
            choice, ok = QInputDialog.getItem(self, "Объединить реплики", "Спикер объединённой реплики",
                                              labels, ids.index(sid), False)
            if not ok:
                return
            sid = ids[labels.index(choice)]
        self.segments[row:row + 2] = [merge_segments(first, second, sid)]
        self._populate(row)
# [SEMANTIC-END] GUI:TRANSCRIPT_EDITOR
