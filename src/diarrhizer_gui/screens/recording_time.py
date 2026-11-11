"""Shared editable recording wall time with explicit filename proposal and Now actions."""

from datetime import datetime

from PySide6.QtCore import QDateTime, Signal
from PySide6.QtWidgets import QDateTimeEdit, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from diarrhizer.recording_time import local_datetime, suggest_recording_time


# [SEMANTIC-BEGIN] GUI:RECORDING_TIME
# @purpose: Let users review/edit recording date/time, accept a filename proposal or insert current time
# @inputs: initial local timestamp and optional source filename
# @outputs: changed signal and ISO local value; no processing/job metadata mutations
# @sideEffects: Widget state only; no filesystem writes or settings persistence
# @errors: Invalid explicit timestamps raise ValueError before changing the editor
# @see: JOB:RECORDING_TIME, GUI:MOVE_JOBS, GUI:PROJECT_SELECTION
class RecordingTimeEditor(QWidget):
    changed = Signal()

    def __init__(self, value=None, source=None):
        super().__init__()
        self._source = source
        self.edit = QDateTimeEdit()
        self.edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.edit.setCalendarPopup(True)
        self._proposal = QPushButton("Из имени файла")
        self._proposal.clicked.connect(self.from_filename)
        self._now = QPushButton("Сейчас")
        self._now.clicked.connect(lambda: self.set_value(datetime.now(), emit=True))
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit)
        row.addWidget(self._proposal)
        row.addWidget(self._now)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.hint)
        self.set_value(value or datetime.now())
        self.set_source(source)
        self.edit.dateTimeChanged.connect(self.changed)

    def value(self):
        return self.edit.dateTime().toPython().replace(tzinfo=None, microsecond=0).isoformat()

    def set_value(self, value, *, emit=False):
        value = local_datetime(value)
        self.edit.blockSignals(True)
        self.edit.setDateTime(QDateTime(value))
        self.edit.blockSignals(False)
        if emit:
            self.changed.emit()

    def set_source(self, source):
        self._source = source
        proposal = suggest_recording_time(source) if source else None
        self._proposal.setEnabled(proposal is not None)
        self.hint.setText(f"Из имени предложено: {proposal:%Y-%m-%d %H:%M:%S}. Проверьте значение."
                          if proposal else "Дата в имени не распознана однозначно — укажите вручную или нажмите «Сейчас».")

    def from_filename(self):
        proposal = suggest_recording_time(self._source) if self._source else None
        if proposal is not None:
            self.set_value(proposal, emit=True)
# [SEMANTIC-END] GUI:RECORDING_TIME
