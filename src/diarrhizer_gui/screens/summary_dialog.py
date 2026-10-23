"""Dialog that makes a summary of one finished job and shows the job's earlier ones.

Opened from the Result screen. The model call runs in a child process
(LlmWorker) so the GUI stays responsive and the request can be cancelled; the
reply is stored in the job's summary/ folder by diarrhizer.summary.service and
shown here as Markdown. Templates and providers come from the same JSON files
the "Саммаризация" settings screen edits.
"""

import os
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from diarrhizer.summary import service
from diarrhizer.summary.providers import KIND_CLAUDE_CLI, KIND_CODEX_CLI, Provider, ProviderStore
from diarrhizer.summary.providers import store_path as providers_path
from diarrhizer.summary.templates import PromptTemplate, TemplateStore
from diarrhizer.summary.templates import store_path as templates_path
from diarrhizer_gui import provider_text, wildcard_text
from diarrhizer_gui.child_tasks import run_summary_task
from diarrhizer_gui.llm_worker import LlmWorker
from diarrhizer_gui.screens.monitor_screen import format_elapsed
from diarrhizer_gui.screens.prompt_preview_dialog import PromptPreviewDialog
from diarrhizer_gui.screens.summary_settings_screen import KIND_SHORT

OK_COLOR = "#2f7d52"
ERROR_COLOR = "#b23b35"
WARN_COLOR = "#a85520"
MUTED_COLOR = "#7d8394"

NOTE_VIEW_CHARS = 600  # how much of a summary's note the viewer shows (the whole one is in its tooltip and .json)

EMPTY_HTML = (
    f'<p style="color:{MUTED_COLOR};">Саммари для этого задания ещё не создавались. '
    "Выберите шаблон и провайдера и нажмите «Создать саммари».</p>"
)


def destination(provider: Provider) -> str:
    """Where the transcript goes when `provider` is used, for the note under the buttons."""
    if provider.kind == KIND_CODEX_CLI:
        return "в OpenAI через Codex CLI (аккаунт ChatGPT)"
    if provider.kind == KIND_CLAUDE_CLI:
        return "в Anthropic через Claude Code (аккаунт Claude)"
    return f"на {provider.endpoint}"


# [SEMANTIC-BEGIN] GUI:SUMMARY_DIALOG
# @purpose: Let the user summarize a finished job with a chosen template and provider, and read the job's summaries
# @description: Template and provider default to the ones marked as default in their stores. A multi-line
#   "Заметка (необязательно)" field - empty each time the dialog opens, never remembered - is passed to the
#   run as the note for {{note}}; when the chosen template has no {{note}} a warning next to the field says the
#   note will be added at the end of the request. "Показать запрос" opens GUI:PROMPT_PREVIEW for this job, the
#   template and the note. "Создать саммари"
#   starts an LlmWorker (run_summary_task) and turns into "Отменить" while it runs - cancelling or closing the
#   dialog kills the child and its tree and stores nothing. A note under the buttons says exactly where the
#   transcript will be sent. Finished summaries are listed newest first from summary/*.json (a line carries the
#   start of the note the summary was made with) and rendered as
#   Markdown, with the whole note under them; the raw text can be copied or opened as a file. What a request used is shown as far as the
#   provider reported it - the model that answered, tokens, a cost (with where the figure comes from) and the
#   plan's limits - both when it finishes and with every stored summary. A malformed templates file falls back to the
#   built-in templates with a visible error; a malformed or empty providers file disables the button and says
#   where to fix it. The dialog is modal: a request is a matter of seconds to minutes.
# @inputs: job_dir with export/result.md, the template/provider stores
# @outputs: summary/<timestamp>_<template>.md and .json in the job folder
# @sideEffects: Sends the transcript to the chosen provider; spawns/kills a child process; writes summary files
# @errors: Provider failures are shown in the dialog, never raised
# @see: SUMMARY:JOB, SUMMARY:WILDCARDS, GUI:PROMPT_PREVIEW, GUI:LLM_WORKER, GUI:SUMMARY_SETTINGS, CONFIG:SUMMARY_TEMPLATES, CONFIG:SUMMARY_PROVIDERS, CONFIG:PROVIDER_INFO
class SummaryDialog(QDialog):
    def __init__(self, parent, job_dir: Path, job_title: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Саммаризация")
        self.resize(900, 720)
        self._job_dir = Path(job_dir)
        self._job_title = job_title
        self._worker: Optional[LlmWorker] = None
        self._started_at = 0.0
        self._current_text = ""
        self._transcript_chars = 0

        title = QLabel(job_title)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._template_combo = QComboBox()
        self._provider_combo = QComboBox()
        self._provider_combo.currentIndexChanged.connect(self._update_target)
        self._create_button = QPushButton("Создать саммари")
        self._create_button.clicked.connect(self._create_or_cancel)
        self._preview_button = QPushButton("Показать запрос")
        self._preview_button.setToolTip("Показать текст запроса, как он уйдёт модели, — ничего не отправляя")
        self._preview_button.clicked.connect(self._show_request)

        # Empty each time the dialog opens: a note belongs to one run, so it is not remembered.
        self._note_edit = QPlainTextEdit()
        self._note_edit.setFixedHeight(72)
        self._note_edit.setPlaceholderText(
            "Свой текст для этого запуска — попадёт в запрос туда, где в шаблоне стоит {{note}}"
        )
        self._note_warning = QLabel()
        self._note_warning.setWordWrap(True)
        self._note_warning.setStyleSheet(f"color: {WARN_COLOR};")
        self._note_warning.setVisible(False)

        form = QFormLayout()
        form.addRow("Шаблон:", self._template_combo)
        form.addRow("Провайдер:", self._provider_combo)
        form.addRow("Заметка (необязательно):", self._note_edit)
        form.addRow("", self._note_warning)

        self._target_label = QLabel()
        self._target_label.setWordWrap(True)
        self._target_label.setStyleSheet(f"color: {MUTED_COLOR};")

        self._summary_combo = QComboBox()
        self._summary_combo.currentIndexChanged.connect(self._on_summary_selected)
        self._open_button = QPushButton("Открыть файл")
        self._open_button.clicked.connect(self._open_file)
        self._copy_button = QPushButton("Копировать")
        self._copy_button.setToolTip("Текст саммари (Markdown) — в буфер обмена")
        self._copy_button.clicked.connect(self._copy)
        saved_row = QHBoxLayout()
        saved_row.addWidget(QLabel("Сохранённые:"))
        saved_row.addWidget(self._summary_combo, stretch=1)
        saved_row.addWidget(self._open_button)
        saved_row.addWidget(self._copy_button)

        self._meta_label = QLabel()
        self._meta_label.setWordWrap(True)
        self._meta_label.setTextFormat(Qt.TextFormat.PlainText)  # it shows the user's note: no markup
        self._meta_label.setStyleSheet(f"color: {MUTED_COLOR};")

        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._view = QTextBrowser()
        self._view.setOpenExternalLinks(True)
        self._view.setStyleSheet("QTextBrowser { font-size: 13px; padding: 6px; }")

        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addWidget(self._create_button)
        buttons.addWidget(self._preview_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addWidget(self._target_label)
        layout.addLayout(buttons)
        layout.addWidget(self._status)
        layout.addLayout(saved_row)
        layout.addWidget(self._meta_label)
        layout.addWidget(self._view, stretch=1)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        self._load_options()
        self._template_combo.currentIndexChanged.connect(self._update_note_warning)
        self._note_edit.textChanged.connect(self._update_note_warning)
        self._refresh_summaries()

    # --- options ---------------------------------------------------------------

    def _load_options(self) -> None:
        problems = []
        try:
            templates = TemplateStore.load()
        except ValueError as e:
            problems.append(str(e))
            templates = TemplateStore(path=templates_path())  # built-ins only
        try:
            providers = ProviderStore.load()
        except ValueError as e:
            problems.append(str(e))
            providers = ProviderStore(path=providers_path())

        self._templates = templates.templates()
        for name, template in self._templates.items():
            self._template_combo.addItem(name, name)
            self._template_combo.setItemData(
                self._template_combo.count() - 1, template.description, Qt.ItemDataRole.ToolTipRole
            )
        self._template_combo.setCurrentIndex(max(0, self._template_combo.findData(templates.default().name)))

        self._providers = providers.providers
        for name, provider in self._providers.items():
            self._provider_combo.addItem(f"{name}  ·  {KIND_SHORT[provider.kind]}", name)
        default = providers.default()
        if default is not None:
            self._provider_combo.setCurrentIndex(max(0, self._provider_combo.findData(default.name)))

        try:
            self._transcript_chars = len(service.read_transcript(self._job_dir))
        except (FileNotFoundError, ValueError) as e:
            problems.append(str(e))
            self._transcript_chars = 0

        ready = bool(self._providers) and self._transcript_chars > 0
        self._create_button.setEnabled(ready)
        self._template_combo.setEnabled(ready)
        self._note_edit.setEnabled(self._transcript_chars > 0)
        self._preview_button.setEnabled(self._transcript_chars > 0)
        self._provider_combo.setEnabled(bool(self._providers))
        if not self._providers and not problems:
            problems.append(
                "Не настроено ни одного провайдера — добавьте его на экране «Саммаризация» "
                "(вкладка «Провайдеры»): аккаунт ChatGPT или Claude либо API с ключом."
            )
        self._set_status("\n".join(problems), ok=False if problems else None)
        self._update_target()

    def _current_template(self) -> PromptTemplate:
        return self._templates[self._template_combo.currentData()]

    def _current_provider(self) -> Optional[Provider]:
        return self._providers.get(self._provider_combo.currentData())

    def _update_target(self) -> None:
        provider = self._current_provider()
        if provider is None or not self._transcript_chars:
            self._target_label.clear()
            return
        size = f"{self._transcript_chars:,}".replace(",", " ")
        self._target_label.setText(
            f"Протокол (export/result.md, {size} симв.) будет отправлен {destination(provider)}."
        )

    def _update_note_warning(self) -> None:
        """Say so when the note typed has no place in the chosen template: it would be added at the end."""
        message = wildcard_text.note_placement_message(
            self._current_template().prompt, self._note_edit.toPlainText()
        )
        self._note_warning.setText(message)
        self._note_warning.setVisible(bool(message))

    def _show_request(self) -> None:
        template = self._current_template()
        PromptPreviewDialog(
            self, template.prompt, template.name, [(self._job_title, self._job_dir)],
            note=self._note_edit.toPlainText(),
        ).exec()

    # --- creating a summary ----------------------------------------------------------

    def _create_or_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            return
        provider = self._current_provider()
        if provider is None:
            return
        template = self._current_template()
        worker = LlmWorker(
            run_summary_task,
            (str(self._job_dir), template.to_dict(), provider.to_dict(), self._note_edit.toPlainText()),
            self,
        )
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self._started_at = time.monotonic()
        self._set_busy(True)
        self._tick()
        self._timer.start()
        try:
            worker.start()
        except OSError as e:
            self._end(f"Не удалось запустить запрос: {e}", ok=False)

    def _set_busy(self, busy: bool) -> None:
        self._template_combo.setEnabled(not busy)
        self._provider_combo.setEnabled(not busy)
        self._note_edit.setEnabled(not busy)
        self._create_button.setText("Отменить" if busy else "Создать саммари")

    def _tick(self) -> None:
        provider = self._current_provider()
        name = provider.name if provider else "провайдеру"
        self._set_status(
            f"Запрос к «{name}»… {format_elapsed(time.monotonic() - self._started_at)}. "
            "Длинные расшифровки обрабатываются минуты.",
            ok=None,
        )

    def _on_finished(self, result: dict) -> None:
        note = ""
        ok = True
        if result.get("truncated"):
            ok = False
            note = (
                " Внимание: ответ обрезан по лимиту длины — увеличьте «Макс. токенов ответа» "
                "в настройках провайдера."
            )
        used = "".join("\n" + line for line in provider_text.run_lines(result))  # model, tokens, cost, limits
        self._end(
            f"Готово за {format_elapsed(result.get('duration_seconds', 0))}. "
            f"Сохранено: summary/{Path(result['path']).name}.{note}{used}",
            ok=ok,
        )
        self._refresh_summaries(select=result["path"])

    def _on_failed(self, _error_type: str, message: str) -> None:
        self._end(message, ok=False)

    def _on_cancelled(self) -> None:
        self._end("Запрос отменён, ничего не сохранено.", ok=None)

    def _end(self, message: str, ok: Optional[bool]) -> None:
        self._timer.stop()
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._set_busy(False)
        self._set_status(message, ok=ok)

    # --- stored summaries ------------------------------------------------------------

    def _refresh_summaries(self, select: Optional[str] = None) -> None:
        summaries = service.list_summaries(self._job_dir)
        self._summaries = {s["path"]: s for s in summaries}
        self._summary_combo.blockSignals(True)
        self._summary_combo.clear()
        for summary in summaries:
            self._summary_combo.addItem(service.summary_title(summary), summary["path"])
            note = service.summary_note(summary)
            if note:  # the list shows the start of it; the whole note is in the tooltip
                self._summary_combo.setItemData(self._summary_combo.count() - 1, note, Qt.ItemDataRole.ToolTipRole)
        self._summary_combo.setCurrentIndex(max(0, self._summary_combo.findData(select)) if select else 0)
        self._summary_combo.blockSignals(False)
        for widget in (self._summary_combo, self._open_button, self._copy_button):
            widget.setEnabled(bool(summaries))
        if summaries:
            self._show_summary(self._summary_combo.currentData())
        else:
            self._current_text = ""
            self._meta_label.clear()
            self._view.setHtml(EMPTY_HTML)

    def _on_summary_selected(self) -> None:
        path = self._summary_combo.currentData()
        if path:
            self._show_summary(path)

    def _show_summary(self, path: str) -> None:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            self._current_text = ""
            self._view.setPlainText(f"Не удалось прочитать {path}: {e}")
            return
        self._current_text = text
        self._view.setMarkdown(text)
        self._view.verticalScrollBar().setValue(0)

        meta = self._summaries.get(path, {})
        provider = meta.get("provider") or {}
        parts = [f"Шаблон «{meta.get('template', '?')}»", f"провайдер «{provider.get('name', '?')}»"]
        if provider.get("model") and not meta.get("model"):  # a summary from before the model was recorded
            parts.append(f"модель {provider['model']}")
        if meta.get("duration_seconds") is not None:
            parts.append(f"заняло {format_elapsed(meta['duration_seconds'])}")
        if meta.get("truncated"):
            parts.append("ответ обрезан по лимиту длины")
        lines = [" · ".join(parts)]
        note = service.summary_note(meta)
        if note:  # the summary was made with a note of the user's
            lines.append("Заметка: " + (note if len(note) <= NOTE_VIEW_CHARS else note[:NOTE_VIEW_CHARS] + "…"))
        # What the request used, as far as the provider reported it: model, tokens, cost, limits.
        self._meta_label.setText("\n".join([*lines, *provider_text.run_lines(meta)]))

    def _open_file(self) -> None:
        path = self._summary_combo.currentData()
        if path and Path(path).exists():
            os.startfile(path)

    def _copy(self) -> None:
        if self._current_text:
            QApplication.clipboard().setText(self._current_text)
            self._set_status("Текст саммари скопирован в буфер обмена.", ok=True)

    # --- misc -------------------------------------------------------------------------

    def _set_status(self, text: str, ok: Optional[bool]) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {MUTED_COLOR if ok is None else OK_COLOR if ok else ERROR_COLOR};")
        self._status.setVisible(bool(text))

    def reject(self) -> None:
        if self._worker is not None:
            answer = QMessageBox.question(
                self,
                "Идёт запрос",
                "Запрос к модели ещё выполняется. Прервать его и закрыть окно?\nНичего не сохранится.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._timer.stop()
            self._worker.cancel()
        super().reject()
# [SEMANTIC-END] GUI:SUMMARY_DIALOG
