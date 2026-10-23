"""Dialog that shows the request a template would send for a job - built, not sent.

Opened from the template editor (any finished job under the results folder) and from the summary dialog (the
job it is about). The text is what summary.service.preview_prompt() returns, i.e. exactly what a summary run
would give the model; nothing leaves the machine and nothing is stored.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from diarrhizer.summary import service
from diarrhizer.summary.wildcards import UnknownWildcardError
from diarrhizer_gui import settings_keys, wildcard_text
from diarrhizer_gui.job_scan import scan_jobs

ERROR_COLOR = "#b23b35"
WARN_COLOR = "#a85520"
MUTED_COLOR = "#7d8394"


def completed_jobs(settings: QSettings) -> tuple[list[tuple[str, Path]], Path]:
    """([(title, job folder)], results folder): the jobs under the results folder that have a result, newest first."""
    out_dir = Path(str(settings.value(settings_keys.OUT_DIR, str(Path.cwd() / "out"))))
    jobs = [
        (f"{job.input_name} · {job.date_label}", job.job_dir)
        for job in scan_jobs(out_dir)
        if job.complete
    ]
    return jobs, out_dir


# [SEMANTIC-BEGIN] GUI:PROMPT_PREVIEW
# @purpose: Show the request text a template produces for a job, without sending it to a model
# @description: A job to pick (the ones given, or just the one the caller is about), an optional note for
#   {{note}} and the assembled request in a read-only box, rebuilt whenever the job or the note changes. It
#   is service.preview_prompt() - the code summarize_job() uses - so the text is what a run would send. The
#   template's prompt is taken as text, so a template that is not saved yet can be previewed. Problems are
#   said in words in the box's place: no finished job, no result.md, an unknown wildcard (named, with its
#   line); a template without {{transcript}} or a note without {{note}} gets the same warnings as in the
#   editor and the summary dialog. Reads the job's files; opens no network connection.
# @inputs: prompt text, template name, [(title, job folder)], optional starting note
# @outputs: None (the dialog only shows)
# @sideEffects: Reads the job's export/result.md and the files the wildcards read
# @errors: Shown in the dialog, never raised
# @see: SUMMARY:JOB, SUMMARY:WILDCARDS, GUI:SUMMARY_SETTINGS, GUI:SUMMARY_DIALOG
class PromptPreviewDialog(QDialog):
    def __init__(
        self,
        parent,
        prompt: str,
        template_name: str,
        jobs: list[tuple[str, Path]],
        note: str = "",
        empty_hint: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Собранный запрос")
        self.resize(860, 680)
        self._prompt = prompt
        self._template_name = template_name
        self._empty_hint = empty_hint or "Нет готовых заданий: сначала обработайте запись."

        title = QLabel(f"Запрос по шаблону «{template_name}»" if template_name else "Запрос по шаблону")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        title.setTextFormat(Qt.TextFormat.PlainText)
        intro = QLabel("Это текст, который получила бы модель. Ничего не отправляется и не сохраняется.")
        intro.setStyleSheet(f"color: {MUTED_COLOR};")

        self._job_combo = QComboBox()
        for job_title, job_dir in jobs:
            self._job_combo.addItem(job_title, str(job_dir))
        self._job_combo.setEnabled(len(jobs) > 1)
        self._job_combo.currentIndexChanged.connect(self._render)

        self._note_edit = QPlainTextEdit()
        self._note_edit.setPlainText(note)
        self._note_edit.setFixedHeight(64)
        self._note_edit.setPlaceholderText("Текст для {{note}} — необязательно")
        self._note_edit.textChanged.connect(self._render)

        form = QFormLayout()
        form.addRow("Задание:", self._job_combo)
        form.addRow("Заметка:", self._note_edit)

        self._info = QLabel()
        self._info.setWordWrap(True)
        self._info.setTextFormat(Qt.TextFormat.PlainText)
        self._info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setFont(QFont("Consolas", 10))

        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self._info)
        layout.addWidget(self._view, stretch=1)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

        self._render()

    def _say(self, text: str, color: Optional[str]) -> None:
        self._info.setText(text)
        self._info.setStyleSheet(f"color: {color or MUTED_COLOR};")
        self._info.setVisible(bool(text))

    def _render(self) -> None:
        job_dir = self._job_combo.currentData()
        if not job_dir:
            self._view.clear()
            self._say(self._empty_hint, WARN_COLOR)
            return
        note = self._note_edit.toPlainText()
        try:
            rendered = service.preview_prompt(job_dir, self._prompt, note, self._template_name)
        except ValueError as e:  # an unknown wildcard (said in Russian), or an empty result.md
            self._view.clear()
            message = wildcard_text.unknown_message(self._prompt) if isinstance(e, UnknownWildcardError) else ""
            self._say(message or str(e), ERROR_COLOR)
            return
        except OSError as e:
            self._view.clear()
            self._say(str(e), ERROR_COLOR)
            return
        self._view.setPlainText(rendered.text)
        size = f"{len(rendered.text):,}".replace(",", " ")
        warnings = [
            message
            for message in (
                wildcard_text.missing_transcript_message(self._prompt),
                wildcard_text.note_placement_message(self._prompt, note),
            )
            if message
        ]
        self._say("\n".join([f"{size} симв. Ничего не отправлено.", *warnings]), WARN_COLOR if warnings else None)
# [SEMANTIC-END] GUI:PROMPT_PREVIEW
