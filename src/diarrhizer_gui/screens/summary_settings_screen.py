"""Summarization setup: prompt templates and model providers.

Both live in JSON files shared with the CLI (summary_templates.json,
summary_providers.json - see diarrhizer.summary). An API key is the one thing
that is not kept there: like the Hugging Face token it goes to the repo-root
.env (env_file.py) and into os.environ at once, and the provider only names
the variable. Summaries themselves are made from a finished job's Result
screen (summary_dialog.py).

Only PySide6 + the stdlib-only diarrhizer.summary modules at import time.
"""

import os
import shlex
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from diarrhizer.summary.providers import (
    CLI_KINDS,
    DEFAULT_ANTHROPIC_ENDPOINT,
    DEFAULT_COMMANDS,
    DEFAULT_TIMEOUT_SECONDS,
    KIND_ANTHROPIC_API,
    KIND_CLAUDE_CLI,
    KIND_CODEX_CLI,
    KIND_OPENAI_API,
    KINDS,
    Provider,
    ProviderStore,
    default_key_env,
)
from diarrhizer.summary.templates import PromptTemplate, TemplateStore
from diarrhizer_gui import env_file
from diarrhizer_gui.child_tasks import run_provider_check_task
from diarrhizer_gui.llm_worker import LlmWorker

ENV_PATH = env_file.REPO_ROOT / ".env"

OK_COLOR = "#2f7d52"
ERROR_COLOR = "#b23b35"
MUTED_COLOR = "#7d8394"

KIND_LABELS = {
    KIND_CODEX_CLI: "ChatGPT — аккаунт с подпиской (Codex CLI)",
    KIND_CLAUDE_CLI: "Claude — аккаунт с подпиской (Claude Code CLI)",
    KIND_OPENAI_API: "OpenAI-совместимый API (по ключу)",
    KIND_ANTHROPIC_API: "Anthropic API (по ключу)",
}
KIND_SHORT = {
    KIND_CODEX_CLI: "ChatGPT, подписка",
    KIND_CLAUDE_CLI: "Claude, подписка",
    KIND_OPENAI_API: "OpenAI-совместимый API",
    KIND_ANTHROPIC_API: "Anthropic API",
}
KIND_HINTS = {
    KIND_CODEX_CLI: (
        "Используется вход, выполненный в Codex CLI (тарифный план ChatGPT): один раз выполните в "
        "терминале «codex login». Пароль и токены подписки Diarrhizer не хранит и не видит. "
        "Расшифровка отправляется в OpenAI через Codex."
    ),
    KIND_CLAUDE_CLI: (
        "Используется вход, выполненный в Claude Code (подписка Claude): один раз выполните в "
        "терминале «claude auth login». Пароль и токены подписки Diarrhizer не хранит и не видит. "
        "Расшифровка отправляется в Anthropic через Claude Code."
    ),
    KIND_OPENAI_API: (
        "Любой сервис с OpenAI-совместимым API: OpenAI, OpenRouter, DeepSeek, Groq, а также локальные "
        "Ollama, LM Studio, vLLM. Endpoint — базовый адрес (например https://api.openai.com/v1). "
        "Ключ хранится в .env, а не в summary_providers.json; для локального сервера его можно не указывать."
    ),
    KIND_ANTHROPIC_API: (
        "Anthropic Messages API по ключу. Ключ хранится в .env, а не в summary_providers.json."
    ),
}
MODEL_PLACEHOLDERS = {
    KIND_CODEX_CLI: "необязательно — модель по умолчанию у Codex",
    KIND_CLAUDE_CLI: "необязательно — модель по умолчанию у Claude Code, например sonnet",
    KIND_OPENAI_API: "например gpt-4o-mini, deepseek-chat, llama3.1",
    KIND_ANTHROPIC_API: "например claude-sonnet-5-5",
}
NEW_PROVIDER_NAMES = {
    KIND_CODEX_CLI: "ChatGPT",
    KIND_CLAUDE_CLI: "Claude",
    KIND_OPENAI_API: "OpenAI API",
    KIND_ANTHROPIC_API: "Anthropic API",
}
DEFAULT_ENDPOINTS = {
    KIND_OPENAI_API: "https://api.openai.com/v1",
    KIND_ANTHROPIC_API: DEFAULT_ANTHROPIC_ENDPOINT,
}


def _set_status(label: QLabel, message: str, ok: Optional[bool]) -> None:
    """ok=True green, False red, None muted."""
    label.setText(message)
    label.setStyleSheet(f"color: {MUTED_COLOR if ok is None else OK_COLOR if ok else ERROR_COLOR};")


def _status_label() -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {MUTED_COLOR};")
    return label


def _optional_number(text: str, cast, label: str):
    """None for a blank field, else the number; ValueError says which field is wrong."""
    text = text.strip().replace(",", ".")
    if not text:
        return None
    try:
        return cast(text)
    except ValueError:
        raise ValueError(f"{label}: нужно число, а не {text!r}") from None


def _unique_name(base: str, taken) -> str:
    name, counter = base, 2
    while name in taken:
        name = f"{base} {counter}"
        counter += 1
    return name


def _confirm_discard(parent: QWidget) -> bool:
    return QMessageBox.question(
        parent,
        "Несохранённые изменения",
        "В форме есть несохранённые изменения. Продолжить без сохранения?",
    ) == QMessageBox.StandardButton.Yes


class TemplatesTab(QWidget):
    """List of templates on the left, the selected one on the right."""

    def __init__(self) -> None:
        super().__init__()
        self._store: Optional[TemplateStore] = None
        # Name of the stored template shown in the form; None = a new, unsaved one.
        self._editing: Optional[str] = None
        # The form's values as they were filled in; the form is "dirty" once they differ.
        self._baseline: tuple = ("", "", "")

        self._list = QListWidget()
        self._list.setFixedWidth(260)
        self._list.currentItemChanged.connect(self._on_item_changed)

        self._name_field = QLineEdit()
        self._description_field = QLineEdit()
        self._description_field.setPlaceholderText("необязательно — подсказка в списке шаблонов")
        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setMinimumHeight(240)
        self._prompt_edit.setPlaceholderText(
            "Что сделать с расшифровкой, например:\n"
            "«Составь протокол встречи: участники, решения, задачи с ответственными и сроками.»"
        )
        hint = _muted_label(
            "К промпту автоматически добавляется расшифровка разговора (export/result.md) — "
            "включать её в шаблон не нужно."
        )

        form = QFormLayout()
        form.addRow("Название:", self._name_field)
        form.addRow("Описание:", self._description_field)
        form.addRow("Промпт:", self._prompt_edit)
        form.addRow("", hint)

        self._new_button = QPushButton("Новый")
        self._new_button.clicked.connect(self._new)
        self._duplicate_button = QPushButton("Дублировать")
        self._duplicate_button.setToolTip("Новый шаблон на основе выбранного — например, чтобы изменить встроенный")
        self._duplicate_button.clicked.connect(self._duplicate)
        self._save_button = QPushButton("Сохранить")
        self._save_button.clicked.connect(self._save)
        self._delete_button = QPushButton("Удалить")
        self._delete_button.clicked.connect(self._delete)
        self._default_button = QPushButton("Сделать по умолчанию")
        self._default_button.clicked.connect(self._make_default)
        buttons = QHBoxLayout()
        for button in (self._new_button, self._duplicate_button, self._save_button,
                       self._delete_button, self._default_button):
            buttons.addWidget(button)
        buttons.addStretch(1)

        self._status = _status_label()

        right = QVBoxLayout()
        right.addLayout(form)
        right.addLayout(buttons)
        right.addWidget(self._status)
        right.addStretch(1)

        layout = QHBoxLayout(self)
        layout.addWidget(self._list)
        layout.addLayout(right, stretch=1)

        self.reload()

    # --- state ---------------------------------------------------------------

    def reload(self, select: Optional[str] = None) -> None:
        """Re-read the store, rebuild the list and show `select` (default: what was being edited)."""
        try:
            self._store = TemplateStore.load()
        except ValueError as e:
            self._store = None
            self._list.clear()
            self._set_enabled(False)
            _set_status(self._status, str(e), ok=False)
            return
        self._set_enabled(True)
        wanted = select or self._editing or self._store.default().name
        self._list.blockSignals(True)
        self._list.clear()
        for name, template in self._store.templates().items():
            suffix = " (встроенный)" if self._store.is_builtin(name) else ""
            prefix = "✓ " if name == self._store.default().name else ""
            item = QListWidgetItem(f"{prefix}{name}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setToolTip(template.description)
            self._list.addItem(item)
            if name == wanted:
                self._list.setCurrentItem(item)
        if self._list.currentItem() is None:
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        self._show(self._list.currentItem().data(Qt.ItemDataRole.UserRole))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._dirty():  # an unsaved form survives a visit to another screen
            self.reload()

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self._list, self._name_field, self._description_field, self._prompt_edit,
                       self._new_button, self._duplicate_button, self._save_button,
                       self._delete_button, self._default_button):
            widget.setEnabled(enabled)

    def _show(self, name: str) -> None:
        template = self._store.get(name)
        self._editing = name
        self._name_field.setText(template.name)
        self._description_field.setText(template.description)
        self._prompt_edit.setPlainText(template.prompt)
        self._baseline = self._snapshot()
        self._update_buttons()
        if self._store.is_builtin(name):
            _set_status(
                self._status, "Встроенный шаблон нельзя изменить — «Дублировать» создаст его копию.", ok=None
            )
        else:
            self._status.clear()

    def _update_buttons(self) -> None:
        builtin = self._editing is not None and self._store is not None and self._store.is_builtin(self._editing)
        for field in (self._name_field, self._description_field):
            field.setReadOnly(builtin)
        self._prompt_edit.setReadOnly(builtin)
        self._save_button.setEnabled(not builtin)
        self._duplicate_button.setEnabled(self._editing is not None)
        self._delete_button.setEnabled(self._editing is not None and not builtin)
        is_default = self._editing is not None and self._store.default().name == self._editing
        self._default_button.setEnabled(self._editing is not None and not is_default)

    def _snapshot(self) -> tuple:
        return (self._name_field.text(), self._description_field.text(), self._prompt_edit.toPlainText())

    def _dirty(self) -> bool:
        return self._store is not None and self._snapshot() != self._baseline

    # --- actions ---------------------------------------------------------------

    def _on_item_changed(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if current is None:
            return
        if self._dirty() and not _confirm_discard(self):
            self._list.blockSignals(True)
            self._list.setCurrentItem(previous)
            self._list.blockSignals(False)
            return
        self._show(current.data(Qt.ItemDataRole.UserRole))

    def _start_new_form(self, name: str, description: str, prompt: str) -> None:
        self._list.blockSignals(True)
        self._list.setCurrentRow(-1)
        self._list.blockSignals(False)
        self._editing = None
        self._name_field.setText(name)
        self._description_field.setText(description)
        self._prompt_edit.setPlainText(prompt)
        self._baseline = self._snapshot()
        self._update_buttons()
        self._name_field.setFocus()
        _set_status(self._status, "Новый шаблон: заполните поля и нажмите «Сохранить».", ok=None)

    def _new(self) -> None:
        if self._store is None or (self._dirty() and not _confirm_discard(self)):
            return
        self._start_new_form("", "", "")

    def _duplicate(self) -> None:
        if self._store is None or self._editing is None:
            return
        if self._dirty() and not _confirm_discard(self):
            return
        source = self._store.get(self._editing)
        name = _unique_name(f"{source.name} (копия)", self._store.templates())
        self._start_new_form(name, source.description, source.prompt)

    def _save(self) -> None:
        if self._store is None:
            return
        try:
            template = PromptTemplate(
                name=self._name_field.text().strip(),
                prompt=self._prompt_edit.toPlainText(),
                description=self._description_field.text().strip(),
            )
            if template.name != self._editing and template.name in self._store.templates():
                raise ValueError(f"Шаблон «{template.name}» уже есть — выберите другое название")
        except ValueError as e:
            _set_status(self._status, str(e), ok=False)
            return
        renamed_from = self._editing if self._editing != template.name else None
        was_default = renamed_from is not None and self._store.default_name == renamed_from
        try:
            self._store.put(template)
            if renamed_from is not None:
                self._store.remove(renamed_from)
                if was_default:
                    self._store.default_name = template.name
            self._store.save()
        except (ValueError, OSError) as e:
            _set_status(self._status, str(e), ok=False)
            self._store = TemplateStore.load()  # forget the change that could not be written
            return
        self._editing = template.name
        self.reload(select=template.name)
        _set_status(self._status, f"Шаблон «{template.name}» сохранён в {self._store.path.name}", ok=True)

    def _delete(self) -> None:
        if self._store is None or self._editing is None or self._store.is_builtin(self._editing):
            return
        name = self._editing
        if QMessageBox.question(
            self, "Удалить шаблон", f"Удалить шаблон «{name}»? Уже созданные саммари не изменятся."
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self._store.remove(name)
            self._store.save()
        except (ValueError, OSError) as e:
            _set_status(self._status, str(e), ok=False)
            return
        self._editing = None
        self.reload()
        _set_status(self._status, f"Шаблон «{name}» удалён", ok=True)

    def _make_default(self) -> None:
        if self._store is None or self._editing is None:
            return
        name = self._editing
        try:
            self._store.set_default(name)
            self._store.save()
        except (ValueError, OSError) as e:
            _set_status(self._status, str(e), ok=False)
            return
        self.reload(select=name)
        _set_status(self._status, f"«{name}» — шаблон по умолчанию (GUI и CLI)", ok=True)


class ProvidersTab(QWidget):
    """List of providers on the left, the selected one on the right."""

    def __init__(self) -> None:
        super().__init__()
        self._store: Optional[ProviderStore] = None
        # Name of the stored provider shown in the form; None = a new, unsaved one.
        self._editing: Optional[str] = None
        # The form's values as they were filled in; the form is "dirty" once they differ.
        self._baseline: tuple = ()
        self._worker: Optional[LlmWorker] = None

        self._list = QListWidget()
        self._list.setFixedWidth(260)
        self._list.currentItemChanged.connect(self._on_item_changed)

        add_menu = QMenu(self)
        for kind in KINDS:
            action = add_menu.addAction(KIND_LABELS[kind])
            action.triggered.connect(lambda checked=False, k=kind: self._new(k))
        self._add_button = QToolButton()
        self._add_button.setText("Добавить провайдера")
        self._add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._add_button.setMenu(add_menu)

        left = QVBoxLayout()
        left.addWidget(self._list, stretch=1)
        left.addWidget(self._add_button)

        # --- form ---
        self._name_field = QLineEdit()
        self._kind_combo = QComboBox()
        for kind in KINDS:
            self._kind_combo.addItem(KIND_LABELS[kind], kind)
        self._kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        self._kind_hint = _muted_label("")

        self._command_field = QLineEdit()
        self._extra_args_field = QLineEdit()
        self._extra_args_field.setPlaceholderText('необязательно, например: -c model_reasoning_effort="low"')
        self._model_field = QLineEdit()
        self._endpoint_field = QLineEdit()
        self._endpoint_field.setPlaceholderText("https://api.example.com/v1")

        self._key_field = QLineEdit()
        self._key_field.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_toggle = QPushButton("Показать")
        self._key_toggle.setCheckable(True)
        self._key_toggle.toggled.connect(self._toggle_key_visibility)
        key_row = QWidget()
        key_layout = QHBoxLayout(key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.addWidget(self._key_field, stretch=1)
        key_layout.addWidget(self._key_toggle)
        self._key_row = key_row
        self._key_env_field = QLineEdit()
        self._key_env_field.setToolTip("Имя переменной окружения, из которой читается ключ")
        self._temperature_field = QLineEdit()
        self._temperature_field.setPlaceholderText("по умолчанию сервиса")
        self._max_tokens_field = QLineEdit()
        self._max_tokens_field.setPlaceholderText("по умолчанию сервиса")
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(1, 24 * 3600)
        self._timeout_spin.setSuffix(" с")
        self._timeout_spin.setValue(DEFAULT_TIMEOUT_SECONDS)
        self._timeout_spin.setToolTip("Сколько ждать ответ: длинные расшифровки обрабатываются минуты")

        self._form = QFormLayout()
        self._form.addRow("Название:", self._name_field)
        self._form.addRow("Тип:", self._kind_combo)
        self._form.addRow("", self._kind_hint)
        self._form.addRow("Команда:", self._command_field)
        self._form.addRow("Endpoint:", self._endpoint_field)
        self._form.addRow("Модель:", self._model_field)
        self._form.addRow("Ключ API:", self._key_row)
        self._form.addRow("Переменная ключа:", self._key_env_field)
        self._form.addRow("Доп. аргументы:", self._extra_args_field)
        self._form.addRow("Температура:", self._temperature_field)
        self._form.addRow("Макс. токенов ответа:", self._max_tokens_field)
        self._form.addRow("Таймаут:", self._timeout_spin)
        self._cli_widgets = (self._command_field, self._extra_args_field)
        self._api_widgets = (
            self._endpoint_field, self._key_row, self._key_env_field,
            self._temperature_field, self._max_tokens_field,
        )

        self._save_button = QPushButton("Сохранить")
        self._save_button.clicked.connect(self._save)
        self._delete_button = QPushButton("Удалить")
        self._delete_button.clicked.connect(self._delete)
        self._default_button = QPushButton("Сделать по умолчанию")
        self._default_button.clicked.connect(self._make_default)
        self._test_button = QPushButton("Проверить")
        self._test_button.setToolTip(
            "Отправить модели короткий тестовый запрос с текущими значениями формы (без сохранения). "
            "Для подписки это расходует ничтожную долю лимита."
        )
        self._test_button.clicked.connect(self._test_or_cancel)
        buttons = QHBoxLayout()
        for button in (self._save_button, self._delete_button, self._default_button, self._test_button):
            buttons.addWidget(button)
        buttons.addStretch(1)

        self._status = _status_label()

        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setContentsMargins(0, 0, 8, 0)
        form_layout.addLayout(self._form)
        form_layout.addLayout(buttons)
        form_layout.addWidget(self._status)
        form_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(form_widget)

        layout = QHBoxLayout(self)
        layout.addLayout(left)
        layout.addWidget(scroll, stretch=1)

        self._apply_kind(KIND_CODEX_CLI)
        self._baseline = self._snapshot()
        self.reload()

    # --- state ---------------------------------------------------------------

    def reload(self, select: Optional[str] = None) -> None:
        """Re-read the store, rebuild the list and show `select` (default: what was being edited)."""
        try:
            self._store = ProviderStore.load()
        except ValueError as e:
            self._store = None
            self._list.clear()
            self._set_enabled(False)
            _set_status(self._status, str(e), ok=False)
            return
        self._set_enabled(True)
        default = self._store.default()
        wanted = select or self._editing or (default.name if default else None)
        self._list.blockSignals(True)
        self._list.clear()
        for name, provider in self._store.providers.items():
            prefix = "✓ " if default is not None and name == default.name else ""
            item = QListWidgetItem(f"{prefix}{name}  ·  {KIND_SHORT[provider.kind]}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._list.addItem(item)
            if name == wanted:
                self._list.setCurrentItem(item)
        if self._list.currentItem() is None and self._list.count():
            self._list.setCurrentRow(0)
        self._list.blockSignals(False)
        current = self._list.currentItem()
        if current is not None:
            self._show(current.data(Qt.ItemDataRole.UserRole))
        else:
            self._editing = None
            self._update_buttons()
            _set_status(
                self._status,
                "Провайдеров пока нет — нажмите «Добавить провайдера» и выберите тип: "
                "аккаунт ChatGPT / Claude (подписка) или API с ключом.",
                ok=None,
            )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._dirty():  # an unsaved form survives a visit to another screen
            self.reload()

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self._list, self._add_button, self._name_field, self._kind_combo, self._command_field,
                       self._extra_args_field, self._model_field, self._endpoint_field, self._key_field,
                       self._key_env_field, self._temperature_field, self._max_tokens_field,
                       self._timeout_spin, self._save_button, self._delete_button, self._default_button,
                       self._test_button):
            widget.setEnabled(enabled)

    def _apply_kind(self, kind: str) -> None:
        is_cli = kind in CLI_KINDS
        for widget in self._cli_widgets:
            self._form.setRowVisible(widget, is_cli)
        for widget in self._api_widgets:
            self._form.setRowVisible(widget, not is_cli)
        self._kind_hint.setText(KIND_HINTS[kind])
        self._model_field.setPlaceholderText(MODEL_PLACEHOLDERS[kind])
        self._command_field.setPlaceholderText(
            f"{DEFAULT_COMMANDS[kind]} — или полный путь к исполняемому файлу" if is_cli else ""
        )

    def _on_kind_changed(self) -> None:
        kind = self._kind_combo.currentData()
        self._apply_kind(kind)
        if kind in DEFAULT_ENDPOINTS and not self._endpoint_field.text().strip():
            self._endpoint_field.setText(DEFAULT_ENDPOINTS[kind])

    def _toggle_key_visibility(self, checked: bool) -> None:
        self._key_field.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password)
        self._key_toggle.setText("Скрыть" if checked else "Показать")

    def _fill(self, provider: Provider) -> None:
        self._name_field.setText(provider.name)
        self._kind_combo.blockSignals(True)
        self._kind_combo.setCurrentIndex(self._kind_combo.findData(provider.kind))
        self._kind_combo.blockSignals(False)
        self._apply_kind(provider.kind)
        self._command_field.setText(provider.command)
        self._extra_args_field.setText(shlex.join(provider.extra_args))
        self._model_field.setText(provider.model)
        self._endpoint_field.setText(provider.endpoint)
        self._key_env_field.setText(provider.api_key_env)
        self._key_field.setText(provider.api_key())
        self._temperature_field.setText("" if provider.temperature is None else f"{provider.temperature:g}")
        self._max_tokens_field.setText("" if provider.max_tokens is None else str(provider.max_tokens))
        self._timeout_spin.setValue(provider.timeout_seconds)
        self._baseline = self._snapshot()

    def _fill_blank(self, kind: str, name: str) -> None:
        self._name_field.setText(name)
        self._kind_combo.blockSignals(True)
        self._kind_combo.setCurrentIndex(self._kind_combo.findData(kind))
        self._kind_combo.blockSignals(False)
        self._apply_kind(kind)
        for field in (self._command_field, self._extra_args_field, self._model_field, self._key_field,
                      self._key_env_field, self._temperature_field, self._max_tokens_field):
            field.clear()
        self._endpoint_field.setText(DEFAULT_ENDPOINTS.get(kind, ""))
        if kind == KIND_ANTHROPIC_API:  # the one API kind that cannot go without a key
            self._key_env_field.setText(default_key_env(name))
        self._timeout_spin.setValue(DEFAULT_TIMEOUT_SECONDS)
        self._baseline = self._snapshot()

    def _show(self, name: str) -> None:
        self._editing = name
        self._fill(self._store.get(name))
        self._update_buttons()
        self._status.clear()

    def _update_buttons(self) -> None:
        stored = self._editing is not None
        self._delete_button.setEnabled(stored)
        is_default = stored and self._store is not None and self._store.default_name == self._editing
        self._default_button.setEnabled(stored and not is_default)

    def _snapshot(self) -> tuple:
        return (
            self._name_field.text(), self._kind_combo.currentData(), self._command_field.text(),
            self._extra_args_field.text(), self._model_field.text(), self._endpoint_field.text(),
            self._key_field.text(), self._key_env_field.text(), self._temperature_field.text(),
            self._max_tokens_field.text(), self._timeout_spin.value(),
        )

    def _dirty(self) -> bool:
        return self._store is not None and self._snapshot() != self._baseline

    # --- form <-> Provider -------------------------------------------------------

    def _provider_from_fields(self) -> Provider:
        """The provider the form describes; ValueError says what is wrong."""
        name = self._name_field.text().strip()
        kind = self._kind_combo.currentData()
        common = dict(
            name=name,
            kind=kind,
            model=self._model_field.text().strip(),
            timeout_seconds=self._timeout_spin.value(),
        )
        if kind in CLI_KINDS:
            try:
                extra_args = tuple(shlex.split(self._extra_args_field.text()))
            except ValueError as e:
                raise ValueError(f"Доп. аргументы: {e}") from None
            return Provider(command=self._command_field.text().strip(), extra_args=extra_args, **common)
        key_env = self._key_env_field.text().strip()
        if not key_env and (self._key_field.text().strip() or kind == KIND_ANTHROPIC_API):
            key_env = default_key_env(name)
        return Provider(
            endpoint=self._endpoint_field.text().strip(),
            api_key_env=key_env,
            temperature=_optional_number(self._temperature_field.text(), float, "Температура"),
            max_tokens=_optional_number(self._max_tokens_field.text(), int, "Макс. токенов ответа"),
            **common,
        )

    # --- actions ---------------------------------------------------------------

    def _on_item_changed(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if current is None:
            return
        if self._dirty() and not _confirm_discard(self):
            self._list.blockSignals(True)
            self._list.setCurrentItem(previous)
            self._list.blockSignals(False)
            return
        self._show(current.data(Qt.ItemDataRole.UserRole))

    def _new(self, kind: str) -> None:
        if self._store is None or (self._dirty() and not _confirm_discard(self)):
            return
        self._list.blockSignals(True)
        self._list.setCurrentRow(-1)
        self._list.blockSignals(False)
        self._editing = None
        name = _unique_name(NEW_PROVIDER_NAMES[kind], self._store.providers)
        self._fill_blank(kind, name)
        self._update_buttons()
        self._name_field.setFocus()
        self._name_field.selectAll()
        _set_status(self._status, "Новый провайдер: заполните поля и нажмите «Сохранить».", ok=None)

    def _save(self) -> None:
        if self._store is None:
            return
        try:
            provider = self._provider_from_fields()
            if provider.name != self._editing and provider.name in self._store.providers:
                raise ValueError(f"Провайдер «{provider.name}» уже есть — выберите другое название")
        except ValueError as e:
            _set_status(self._status, str(e), ok=False)
            return

        key = self._key_field.text().strip()
        if not provider.is_cli and provider.api_key_env and key and key != provider.api_key():
            try:
                env_file.write_env_file(ENV_PATH, {provider.api_key_env: key})
            except OSError as e:
                _set_status(self._status, f"Не удалось записать ключ в {ENV_PATH.name}: {e}", ok=False)
                return
            os.environ[provider.api_key_env] = key

        renamed_from = self._editing if self._editing != provider.name else None
        was_default = renamed_from is not None and self._store.default_name == renamed_from
        try:
            self._store.put(provider)
            if renamed_from is not None:
                self._store.remove(renamed_from)
                if was_default:
                    self._store.default_name = provider.name
            self._store.save()
        except (ValueError, OSError) as e:
            _set_status(self._status, str(e), ok=False)
            self._store = ProviderStore.load()  # forget the change that could not be written
            return
        self._editing = provider.name
        self.reload(select=provider.name)
        note = f" Ключ записан в {ENV_PATH.name}." if key and not provider.is_cli else ""
        _set_status(self._status, f"Провайдер «{provider.name}» сохранён.{note}", ok=True)

    def _delete(self) -> None:
        if self._store is None or self._editing is None:
            return
        name = self._editing
        if QMessageBox.question(
            self,
            "Удалить провайдера",
            f"Удалить провайдера «{name}»? Ключ в {ENV_PATH.name} останется на месте.",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self._store.remove(name)
            self._store.save()
        except (ValueError, OSError) as e:
            _set_status(self._status, str(e), ok=False)
            return
        self._editing = None
        self.reload()
        _set_status(self._status, f"Провайдер «{name}» удалён", ok=True)

    def _make_default(self) -> None:
        if self._store is None or self._editing is None:
            return
        name = self._editing
        try:
            self._store.set_default(name)
            self._store.save()
        except (ValueError, OSError) as e:
            _set_status(self._status, str(e), ok=False)
            return
        self.reload(select=name)
        _set_status(self._status, f"«{name}» — провайдер по умолчанию (GUI и CLI)", ok=True)

    # --- connection check ----------------------------------------------------------

    def _test_or_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            return
        try:
            provider = self._provider_from_fields()
        except ValueError as e:
            _set_status(self._status, str(e), ok=False)
            return
        key = self._key_field.text().strip()
        # The key may be typed but not saved yet: the child gets it for this one call.
        env = {provider.api_key_env: key} if not provider.is_cli and provider.api_key_env and key else {}
        worker = LlmWorker(run_provider_check_task, (provider.to_dict(), env), self)
        worker.finished.connect(self._on_test_finished)
        worker.failed.connect(self._on_test_failed)
        worker.cancelled.connect(self._on_test_cancelled)
        self._worker = worker
        self._test_button.setText("Отменить проверку")
        _set_status(self._status, f"Проверка «{provider.name}»… (до {provider.timeout_seconds} с)", ok=None)
        try:
            worker.start()
        except OSError as e:
            self._end_test(f"Не удалось запустить проверку: {e}", ok=False)

    def _on_test_finished(self, result: dict) -> None:
        self._end_test(f"✓ Провайдер отвечает. Ответ: {result.get('reply', '')}", ok=True)

    def _on_test_failed(self, _error_type: str, message: str) -> None:
        self._end_test(message, ok=False)

    def _on_test_cancelled(self) -> None:
        self._end_test("Проверка отменена.", ok=None)

    def _end_test(self, message: str, ok: Optional[bool]) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._test_button.setText("Проверить")
        _set_status(self._status, message, ok=ok)


# [SEMANTIC-BEGIN] GUI:SUMMARY_SETTINGS
# @purpose: Let the user create prompt templates and configure the providers that write summaries
# @description: Two tabs over the shared JSON stores (CONFIG:SUMMARY_TEMPLATES / CONFIG:SUMMARY_PROVIDERS).
#   Templates: built-ins are read-only ("Дублировать" copies one into an editable template), a user template
#   can be renamed, deleted or made the default. Providers are created from four presets (ChatGPT / Claude
#   subscription via their CLIs, OpenAI-compatible API, Anthropic API); an API key typed in the form is written
#   to the repo-root .env under the provider's variable and into os.environ, never to the JSON file. "Проверить"
#   sends a trivial prompt through a child process (LlmWorker) using the form's current values, saved or not,
#   and can be cancelled. An unsaved form survives a visit to another screen and asks before being discarded
#   by a selection change. A malformed store file is shown as an error and the tab is disabled instead of
#   overwriting the user's hand-written file.
# @inputs: user edits; summary_templates.json, summary_providers.json, .env
# @outputs: the two JSON stores, API keys in .env
# @sideEffects: Writes the stores and .env, os.environ; spawns a child process for the connection check
# @errors: Store/validation errors are shown in the tab's status line
# @see: CONFIG:SUMMARY_TEMPLATES, CONFIG:SUMMARY_PROVIDERS, GUI:LLM_WORKER, GUI:SUMMARY_DIALOG, CONFIG:ENV
class SummarySettingsScreen(QWidget):
    """Sidebar screen "Саммаризация": the templates and providers tabs."""

    def __init__(self) -> None:
        super().__init__()
        title = QLabel("Саммаризация")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        intro = _muted_label(
            "Саммари создаётся из готового результата: на экране результата задания нажмите «Саммаризация…», "
            "выберите шаблон и провайдера. Здесь они настраиваются; те же файлы использует команда "
            "diarrhizer summarize."
        )

        self.templates_tab = TemplatesTab()
        self.providers_tab = ProvidersTab()
        tabs = QTabWidget()
        tabs.addTab(self.templates_tab, "Шаблоны промптов")
        tabs.addTab(self.providers_tab, "Провайдеры")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(tabs, stretch=1)
# [SEMANTIC-END] GUI:SUMMARY_SETTINGS
