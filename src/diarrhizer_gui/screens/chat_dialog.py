"""Dialog for a conversation with a language model about one finished job.

Opened from the Result screen. The job's transcript is the context; each question and answer is a turn of a
stored dialogue (diarrhizer.summary.chat, files in the job's chat/ folder). The request runs in a child process
(LlmWorker + run_chat_task) so the window stays responsive and the request can be cancelled; the answer arrives
whole, there is no streaming. Templates and providers come from the same JSON files as the rest of the
"Работа с языковой моделью" section.
"""

import dataclasses
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.summary import chat, service
from diarrhizer.summary.providers import KIND_CODEX_CLI, Provider, ProviderStore
from diarrhizer.summary.providers import store_path as providers_path
from diarrhizer.summary.templates import TemplateStore
from diarrhizer.summary.templates import store_path as templates_path
from diarrhizer_gui import chat_text, provider_text
from diarrhizer_gui.child_tasks import run_chat_task
from diarrhizer_gui.llm_worker import LlmWorker
from diarrhizer_gui.screens.monitor_screen import format_elapsed
from diarrhizer_gui.screens.summary_settings_screen import KIND_SHORT

OK_COLOR = "#2f7d52"
ERROR_COLOR = "#b23b35"
WARN_COLOR = "#a85520"
MUTED_COLOR = "#7d8394"

EMPTY_HTML = (
    f'<p style="color:{MUTED_COLOR};">В этом диалоге ещё нет ходов. Напишите вопрос ниже или выберите быстрый '
    "запрос из шаблона и нажмите «Отправить».</p>"
)


# [SEMANTIC-BEGIN] GUI:CHAT_DIALOG
# @purpose: Talk to a language model about a finished job: stored dialogues, questions, what each turn used
# @description: Beside the job's list of dialogues (new, delete, export to Markdown) the dialog shows the turns of
#   the selected one and an input for the next question; template, provider and model can be changed between
#   turns (each turn records its own). A button per template puts that template's instruction into the input as
#   a quick question. What the context is stays in view: whose transcript, its size, the estimated tokens and the
#   provider's limit if set, and how the chosen provider handles it (its prompt cache, a Codex CLI session, or the
#   whole history sent again); after a turn its tokens (with the cache parts), cost and what was sent are shown,
#   and the dialogue's total. Before sending, the size is checked (summary.chat.check_context): over the
#   provider's limit nothing is sent, and a panel offers the options as buttons - cut the transcript to a window,
#   leave out the oldest turns, or change the model; whatever a turn left out is written under it. The request
#   runs in a child process (LlmWorker, run_chat_task) with a waiting indicator and a cancel button; closing
#   during a request asks first. The answer is shown whole (no streaming).
# @inputs: job_dir with export/result.md, the template/provider stores
# @outputs: chat/<id>.json (and .md on export) in the job folder
# @sideEffects: Sends the transcript and the dialogue to the chosen provider; spawns/kills a child process; writes files
# @errors: Provider failures and bad input are shown in the dialog, never raised
# @see: SUMMARY:CHAT, GUI:LLM_WORKER, GUI:CHILD_TASKS, GUI:SUMMARY_SETTINGS, GUI:SUMMARY_DIALOG
class ChatDialog(QDialog):
    def __init__(self, parent, job_dir: Path, job_title: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Диалог с моделью")
        self.resize(1080, 780)
        self._job_dir = Path(job_dir)
        self._job_title = job_title
        self._worker: Optional[LlmWorker] = None
        self._started_at = 0.0
        self._conversation: Optional[chat.Conversation] = None
        self._transcript_chars = 0
        self._templates: dict = {}
        self._providers: dict = {}
        self._quick_buttons: dict[str, QPushButton] = {}

        title = QLabel(job_title)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        title.setTextFormat(Qt.TextFormat.PlainText)

        self._context_label = QLabel()
        self._context_label.setWordWrap(True)
        self._context_label.setTextFormat(Qt.TextFormat.PlainText)
        self._context_label.setStyleSheet(f"color: {MUTED_COLOR};")

        # --- left: the job's dialogues ---
        self._list = QListWidget()
        self._list.setFixedWidth(250)
        self._list.currentItemChanged.connect(self._on_item_changed)
        self._new_button = QPushButton("Новый диалог")
        self._new_button.clicked.connect(self._new)
        self._delete_button = QPushButton("Удалить")
        self._delete_button.clicked.connect(self._delete)
        self._export_button = QPushButton("Экспорт в Markdown…")
        self._export_button.clicked.connect(self._export)
        left = QVBoxLayout()
        left.addWidget(self._list, stretch=1)
        left.addWidget(self._new_button)
        left.addWidget(self._delete_button)
        left.addWidget(self._export_button)

        # --- right: settings, history, the question ---
        self._template_combo = QComboBox()
        self._provider_combo = QComboBox()
        self._provider_combo.currentIndexChanged.connect(self._update_context)
        self._model_field = QLineEdit()
        self._model_field.setPlaceholderText("модель провайдера по умолчанию — или впишите другую")
        self._note_field = QLineEdit()
        self._note_field.setPlaceholderText("для нового диалога: текст для {{note}}, фиксируется на весь диалог")
        self._session_check = QCheckBox("Хранить сессию Codex (иначе вся история уходит каждый раз)")
        self._session_check.setChecked(True)
        self._session_check.toggled.connect(self._update_context)
        form = QFormLayout()
        form.addRow("Шаблон:", self._template_combo)
        form.addRow("Провайдер:", self._provider_combo)
        form.addRow("Модель:", self._model_field)
        form.addRow("Заметка:", self._note_field)
        form.addRow("", self._session_check)

        self._history = QTextBrowser()
        self._history.setOpenExternalLinks(True)
        self._history.setStyleSheet("QTextBrowser { font-size: 13px; padding: 6px; }")

        # The panel for a context that is too long: what happened, and one button per way out.
        self._warning_box = QFrame()
        self._warning_box.setStyleSheet(f"QFrame {{ border: 1px solid {WARN_COLOR}; border-radius: 4px; }}")
        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setTextFormat(Qt.TextFormat.PlainText)
        self._warning_label.setStyleSheet(f"color: {WARN_COLOR}; border: none;")
        self._warning_buttons = QHBoxLayout()
        warning_layout = QVBoxLayout(self._warning_box)
        warning_layout.addWidget(self._warning_label)
        warning_layout.addLayout(self._warning_buttons)
        self._warning_box.setVisible(False)

        self._quick_row = QHBoxLayout()
        quick_holder = QWidget()
        quick_holder.setLayout(self._quick_row)
        self._quick_scroll = QScrollArea()
        self._quick_scroll.setWidget(quick_holder)
        self._quick_scroll.setWidgetResizable(True)
        self._quick_scroll.setFixedHeight(52)
        self._quick_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._input = QPlainTextEdit()
        self._input.setFixedHeight(84)
        self._input.setPlaceholderText("Вопрос к модели (Ctrl+Enter — отправить)")
        QShortcut(QKeySequence("Ctrl+Return"), self._input).activated.connect(self._send_or_cancel)
        self._send_button = QPushButton("Отправить")
        self._send_button.clicked.connect(self._send_or_cancel)
        send_row = QHBoxLayout()
        send_row.addWidget(self._input, stretch=1)
        send_row.addWidget(self._send_button)

        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._totals_label = QLabel()
        self._totals_label.setWordWrap(True)
        self._totals_label.setTextFormat(Qt.TextFormat.PlainText)
        self._totals_label.setStyleSheet(f"color: {MUTED_COLOR};")

        right = QVBoxLayout()
        right.addLayout(form)
        right.addWidget(self._history, stretch=1)
        right.addWidget(self._warning_box)
        right.addWidget(QLabel("Быстрые запросы из шаблонов:"))
        right.addWidget(self._quick_scroll)
        right.addLayout(send_row)
        right.addWidget(self._status)
        right.addWidget(self._totals_label)

        body = QHBoxLayout()
        body.addLayout(left)
        body.addLayout(right, stretch=1)
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self._context_label)
        layout.addLayout(body, stretch=1)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        self._load_options()
        self._reload_list()

    # --- options -----------------------------------------------------------------------

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
            button = QPushButton(name)
            button.setToolTip(template.description or template.prompt[:300])
            button.clicked.connect(lambda checked=False, n=name: self._use_quick(n))
            self._quick_buttons[name] = button
            self._quick_row.addWidget(button)
        self._quick_row.addStretch(1)
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
        if not self._providers and not problems:
            problems.append(
                "Не настроено ни одного провайдера — добавьте его на экране «Работа с языковой моделью» "
                "(вкладка «Провайдеры»): аккаунт ChatGPT или Claude либо API с ключом."
            )
        self._ready = bool(self._providers) and self._transcript_chars > 0
        self._set_status("\n".join(problems), ok=False if problems else None)
        self._update_context()

    def _current_provider(self) -> Optional[Provider]:
        provider = self._providers.get(self._provider_combo.currentData())
        model = self._model_field.text().strip()
        if provider is not None and model and model != provider.model:
            provider = dataclasses.replace(provider, model=model)
        return provider

    def _update_context(self) -> None:
        """What the context is: whose transcript, how big, the limit, and how this provider handles it."""
        provider = self._current_provider()
        if not self._transcript_chars:
            self._context_label.setText("")
            return
        tokens = chat.estimate_tokens(self._transcript_chars)
        lines = [
            f"Контекст: транскрипт задания «{self._job_title}» (export/result.md): "
            f"{chat_text.number(self._transcript_chars)} симв., ≈ {chat_text.number(tokens)} токенов."
        ]
        if provider is not None:
            limit = f"предел {chat_text.number(provider.context_tokens)} токенов" if provider.context_tokens else (
                "предел не задан — размер не проверяется"
            )
            how = chat_text.how_context_works(provider, self._session_check.isChecked())
            lines.append(f"{how[:1].upper() + how[1:]}; {limit}.")
        conversation = self._conversation
        if conversation is not None and (conversation.window or conversation.omitted_turns):
            cut = chat_text.truncation(chat.truncation_notes(conversation))
            lines.append(cut)
        self._context_label.setText("\n".join(lines))
        self._session_check.setVisible(provider is not None and provider.kind == KIND_CODEX_CLI)

    # --- the dialogues ------------------------------------------------------------------

    def _reload_list(self, select: Optional[str] = None) -> None:
        conversations = chat.list_conversations(self._job_dir)
        wanted = select or (self._conversation.id if self._conversation else None)
        self._list.blockSignals(True)
        self._list.clear()
        for conversation in conversations:
            item = QListWidgetItem(f"{conversation.title or conversation.id}  ·  ходов {len(conversation.turns)}")
            item.setData(Qt.ItemDataRole.UserRole, conversation.id)
            item.setToolTip(f"{conversation.template} · {conversation.created_at}")
            self._list.addItem(item)
            if conversation.id == wanted:
                self._list.setCurrentItem(item)
        if self._list.currentItem() is None and self._list.count():
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        current = self._list.currentItem()
        self._select(current.data(Qt.ItemDataRole.UserRole) if current else None)

    def _on_item_changed(self, current: Optional[QListWidgetItem], _previous) -> None:
        if current is not None:
            self._select(current.data(Qt.ItemDataRole.UserRole))

    def _select(self, conversation_id: Optional[str]) -> None:
        self._conversation = None
        if conversation_id:
            try:
                self._conversation = chat.load_conversation(self._job_dir, conversation_id)
            except (FileNotFoundError, ValueError) as e:
                self._set_status(str(e), ok=False)
        self._hide_warning()
        self._show_conversation()
        self._set_busy(self._worker is not None)

    def _show_conversation(self) -> None:
        conversation = self._conversation
        if conversation is None or not conversation.turns:
            self._history.setHtml(EMPTY_HTML)
        else:
            self._history.setMarkdown(self._history_markdown(conversation))
            self._history.verticalScrollBar().setValue(self._history.verticalScrollBar().maximum())
        if conversation is not None:
            name = conversation.provider.get("name")
            if name in self._providers and self._worker is None:
                self._provider_combo.setCurrentIndex(self._provider_combo.findData(name))
            self._totals_label.setText(chat_text.totals(chat.conversation_totals(conversation)))
        else:
            self._totals_label.setText("")
        self._update_context()

    @staticmethod
    def _history_markdown(conversation: chat.Conversation) -> str:
        parts = []
        if conversation.note:
            parts.append("*Заметка диалога:* " + conversation.note.replace("\n", " "))
        for number, turn in enumerate(conversation.turns, 1):
            who = f"{turn.provider.get('name', '?')} · {turn.model or '?'} · {turn.asked_at}"
            quoted = "\n".join("> " + line for line in turn.question.splitlines() or [""])
            parts.append(f"**{number}. Вы** ({who})\n\n{quoted}\n\n**Модель:**\n\n{turn.answer}")
            lines = provider_text.run_lines({"usage": turn.usage, "cost_usd": turn.cost_usd, "cost_note": turn.cost_note})
            mode = chat_text.context_mode(turn.context_mode)
            if mode:
                size = f", {chat_text.number(turn.context_chars)} симв." if turn.context_chars is not None else ""
                lines.append(f"контекст: {mode}{size}")
            if turn.truncated:
                lines.append("ответ обрезан по лимиту длины")
            cut = chat_text.truncation(turn.truncation)
            if cut:
                lines.append(cut)
            if lines:
                parts.append("*" + " · ".join(lines) + "*")
        return "\n\n---\n\n".join(parts)

    def _new(self) -> Optional[chat.Conversation]:
        provider = self._current_provider()
        template = self._templates.get(self._template_combo.currentData())
        if not self._ready or provider is None or template is None:
            return None
        try:
            conversation = chat.create_conversation(
                self._job_dir, template, provider, note=self._note_field.text()
            )
        except (FileNotFoundError, ValueError, OSError) as e:
            self._set_status(str(e), ok=False)
            return None
        self._conversation = conversation
        self._reload_list(select=conversation.id)
        warnings = [chat_text.warning(w) for w in conversation.warnings]
        self._set_status("\n".join(["Новый диалог создан.", *warnings]), ok=None if not warnings else False)
        if warnings:
            self._status.setStyleSheet(f"color: {WARN_COLOR};")
        return conversation

    def _delete(self) -> None:
        if self._conversation is None or self._worker is not None:
            return
        if QMessageBox.question(
            self, "Удалить диалог", f"Удалить диалог «{self._conversation.title or self._conversation.id}»?"
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            chat.delete_conversation(self._job_dir, self._conversation.id)
        except (FileNotFoundError, OSError) as e:
            self._set_status(str(e), ok=False)
            return
        self._conversation = None
        self._reload_list()
        self._set_status("Диалог удалён.", ok=True)

    def _export(self) -> None:
        if self._conversation is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Экспорт диалога", str(self._job_dir / "chat" / f"{self._conversation.id}.md"), "Markdown (*.md)"
        )
        if not path:
            return
        try:
            written = chat.export_conversation(self._job_dir, self._conversation.id, path)
        except (FileNotFoundError, ValueError, OSError) as e:
            self._set_status(str(e), ok=False)
            return
        self._set_status(f"Диалог сохранён: {written}", ok=True)

    def _use_quick(self, template_name: str) -> None:
        """Put a template's instruction into the input as the question (to read, change and send)."""
        template = self._templates.get(template_name)
        if template is None:
            return
        note = self._conversation.note if self._conversation else self._note_field.text()
        try:
            self._input.setPlainText(chat.quick_question(template, self._job_dir, note))
        except ValueError as e:
            self._set_status(str(e), ok=False)
            return
        self._input.setFocus()

    # --- asking ----------------------------------------------------------------------------

    def _send_or_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            return
        self._start()

    def _start(self, drop_turns: int = 0, window: Optional[tuple] = None) -> None:
        if not self._ready:
            return
        provider = self._current_provider()
        if provider is None:
            return
        if self._conversation is None and self._new() is None:
            return
        conversation = self._conversation
        question = self._input.toPlainText()
        if not question.strip() and conversation.turns:
            self._set_status("Напишите вопрос.", ok=False)
            return
        try:
            check = chat.check_context(
                self._job_dir, conversation.id, question, provider, drop_turns=drop_turns, window=window
            )
        except (FileNotFoundError, ValueError) as e:
            self._set_status(str(e), ok=False)
            return
        if check.exceeds:
            self._show_warning(check, provider)
            return
        self._hide_warning()
        worker = LlmWorker(
            run_chat_task,
            (str(self._job_dir), conversation.id, question, provider.to_dict(),
             self._session_check.isChecked(), drop_turns, list(window) if window else None),
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

    def _show_warning(self, check: chat.ContextCheck, provider: Provider) -> None:
        self._clear_warning_buttons()
        self._warning_label.setText(chat_text.too_long(check, provider.name))
        for option in check.options:
            button = QPushButton(chat_text.option_label(option))
            button.clicked.connect(lambda checked=False, o=option: self._take_option(o))
            self._warning_buttons.addWidget(button)
        cancel = QPushButton("Отмена")
        cancel.clicked.connect(self._hide_warning)
        self._warning_buttons.addWidget(cancel)
        self._warning_box.setVisible(True)

    def _clear_warning_buttons(self) -> None:
        while self._warning_buttons.count():
            item = self._warning_buttons.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _hide_warning(self) -> None:
        self._clear_warning_buttons()
        self._warning_box.setVisible(False)

    def _take_option(self, option: chat.ContextOption) -> None:
        if option.kind == "drop_turns":
            self._start(drop_turns=option.data["count"])
        elif option.kind == "window":
            self._start(window=(option.data["start"], option.data["end"]))
        else:  # another model: nothing to do but let the user pick it
            self._hide_warning()
            self._set_status("Выберите другого провайдера или модель и отправьте вопрос снова.", ok=None)
            self._provider_combo.setFocus()

    def _set_busy(self, busy: bool) -> None:
        enabled = not busy
        for widget in (self._list, self._new_button, self._delete_button, self._export_button, self._template_combo,
                       self._provider_combo, self._model_field, self._note_field, self._input):
            widget.setEnabled(enabled and (self._ready or widget is self._list))
        self._delete_button.setEnabled(enabled and self._conversation is not None)
        self._export_button.setEnabled(enabled and self._conversation is not None)
        self._send_button.setText("Отменить" if busy else "Отправить")
        self._send_button.setEnabled(busy or self._ready)

    def _tick(self) -> None:
        provider = self._current_provider()
        name = provider.name if provider else "провайдеру"
        self._set_status(
            f"Запрос к «{name}»… {format_elapsed(time.monotonic() - self._started_at)}. "
            "Ответ придёт целиком, длинные ответы занимают минуты.",
            ok=None,
        )

    def _on_finished(self, result: dict) -> None:
        turn = chat.Turn.from_dict(result["turn"])
        lines = [f"Готово за {format_elapsed(turn.duration_seconds)}."]
        lines += provider_text.run_lines({"usage": turn.usage, "cost_usd": turn.cost_usd, "cost_note": turn.cost_note})
        mode = chat_text.context_mode(turn.context_mode)
        if mode:
            lines.append(f"Контекст: {mode}.")
        cut = chat_text.truncation(turn.truncation)
        if cut:
            lines.append(cut)
        if turn.truncated:
            lines.append("Внимание: ответ обрезан по лимиту длины — увеличьте «Макс. токенов ответа» провайдера.")
        self._input.clear()
        self._end("\n".join(lines), ok=not (turn.truncated or cut))
        self._reload_list(select=result["conversation_id"])

    def _on_failed(self, _error_type: str, message: str) -> None:
        self._end(message, ok=False)

    def _on_cancelled(self) -> None:
        self._end("Запрос отменён, ход не добавлен.", ok=None)

    def _end(self, message: str, ok: Optional[bool]) -> None:
        self._timer.stop()
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._set_busy(False)
        self._set_status(message, ok=ok)

    # --- misc ----------------------------------------------------------------------------

    def _set_status(self, text: str, ok: Optional[bool]) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {MUTED_COLOR if ok is None else OK_COLOR if ok else ERROR_COLOR};")
        self._status.setVisible(bool(text))

    def reject(self) -> None:
        if self._worker is not None:
            answer = QMessageBox.question(
                self,
                "Идёт запрос",
                "Запрос к модели ещё выполняется. Прервать его и закрыть окно?\nХод не будет добавлен.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._timer.stop()
            self._worker.cancel()
        super().reject()
# [SEMANTIC-END] GUI:CHAT_DIALOG
