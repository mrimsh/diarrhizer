"""Account panel of a subscription provider (ChatGPT via Codex, Claude via Claude Code).

One button does what the state calls for - sign in, cancel a sign-in in progress, sign out - and the
sign-in is the vendor's own: OpenAI's page for ChatGPT (the address comes from `codex app-server` and is
opened here), Anthropic's page for Claude (the Claude Code CLI opens it itself). No password, token or
session ever passes through the app. Below the account, the plan's limits as the provider reports them.

Every action runs in a child process (LlmWorker + run_account_task), so a sign-in that waits for the
browser never blocks the window and "Отменить вход" really stops it.
"""

import html
from typing import Callable, Optional

from PySide6.QtCore import QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from diarrhizer.summary.providers import KIND_CLAUDE_CLI, KIND_CODEX_CLI, Provider
from diarrhizer_gui import provider_text
from diarrhizer_gui.child_tasks import run_account_task
from diarrhizer_gui.llm_worker import LlmWorker

OK_COLOR = "#2f7d52"
WARN_COLOR = "#a85520"
ERROR_COLOR = "#b23b35"
MUTED_COLOR = "#7d8394"

SIGN_IN_LABELS = {KIND_CODEX_CLI: "Войти через ChatGPT", KIND_CLAUDE_CLI: "Войти через Claude"}
VENDOR_PAGES = {KIND_CODEX_CLI: "OpenAI", KIND_CLAUDE_CLI: "Anthropic"}

# States of the panel.
UNKNOWN, CHECKING, SIGNED_OUT, SIGNED_IN, SIGNING_IN, BUSY = range(6)


def open_url(url: str) -> bool:
    """Open a sign-in address in the user's browser. Only https: the address comes from another program."""
    return url.startswith("https://") and QDesktopServices.openUrl(QUrl(url))


# [SEMANTIC-BEGIN] GUI:ACCOUNT_PANEL
# @purpose: Show who a subscription provider is signed in as, sign in/out with one button, and show the plan's limits
# @description: The state decides the one button: signed out -> "Войти через ChatGPT/Claude", signing in ->
#   "Отменить вход" (kills the child and the tool with it), signed in -> "Выйти" (after a confirmation). Status
#   is checked when the provider is shown (set_kind) and on demand. A ChatGPT sign-in gets its address from the
#   tool and opens it here (https only); a Claude sign-in is opened by the CLI itself, the panel only waits.
#   The panel never sees a credential - the tools keep their own. When signed in with Codex the limits are
#   read at once (a local query); Claude has no limits query, so "Обновить лимиты" sends a one-word request and
#   says so. An account that is signed in with an API key rather than the subscription is flagged, because its
#   requests would be billed per token. Limits are windows (percent used as a bar, reset time), credits and
#   an "exhausted" warning; nothing is shown that the provider did not report.
# @inputs: provider_source (a Provider built from the form), user clicks
# @outputs: signal signed_in_changed(bool)
# @sideEffects: Spawns child processes that start the vendor CLI; opens the browser; a sign-in/out changes the tool's stored credentials
# @errors: Failures are shown in the panel's status line
# @see: ADAPTER:LLM_ACCOUNTS, GUI:LLM_WORKER, GUI:CHILD_TASKS, GUI:SUMMARY_SETTINGS, CONFIG:PROVIDER_INFO
class AccountPanel(QGroupBox):
    signed_in_changed = Signal(bool)

    def __init__(self, provider_source: Callable[[], Provider]) -> None:
        super().__init__("Аккаунт")
        self._provider_source = provider_source
        self._kind = KIND_CODEX_CLI
        self._state = UNKNOWN
        self._account: Optional[dict] = None
        self._was_signed_in = False  # what signed_in_changed last reported
        self._worker: Optional[LlmWorker] = None
        self._action = ""

        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._button = QPushButton()
        self._button.clicked.connect(self._on_button)
        self._recheck_button = QPushButton("Проверить вход")
        self._recheck_button.setToolTip("Узнать у инструмента, выполнен ли вход и в какой аккаунт")
        self._recheck_button.clicked.connect(self.refresh_status)
        buttons = QHBoxLayout()
        buttons.addWidget(self._button)
        buttons.addWidget(self._recheck_button)
        buttons.addStretch(1)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: {MUTED_COLOR};")
        self._hint.setTextFormat(Qt.TextFormat.RichText)
        self._hint.linkActivated.connect(open_url)

        self._limits_title = QLabel("Лимиты тарифа")
        self._limits_title.setStyleSheet("font-weight: 600; padding-top: 6px;")
        self._limits_grid = QGridLayout()
        self._limits_grid.setColumnStretch(1, 1)
        self._limits_note = QLabel()
        self._limits_note.setWordWrap(True)
        self._limits_note.setStyleSheet(f"color: {MUTED_COLOR};")
        self._limits_button = QPushButton("Обновить лимиты")
        self._limits_button.clicked.connect(self.refresh_limits)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addLayout(buttons)
        layout.addWidget(self._hint)
        layout.addWidget(self._limits_title)
        layout.addLayout(self._limits_grid)
        layout.addWidget(self._limits_note)
        layout.addWidget(self._limits_button, alignment=Qt.AlignmentFlag.AlignLeft)

        self.set_kind(KIND_CODEX_CLI, check=False)

    # --- public ---------------------------------------------------------------------

    def set_kind(self, kind: str, check: bool = True) -> None:
        """Another provider kind is shown: forget the previous account and limits, and ask the tool again."""
        self._abandon_worker()  # a sign-in or query of the previous provider no longer matters
        self._kind = kind
        self._account = None
        self._was_signed_in = False
        self._state = UNKNOWN
        self._status.clear()
        self._hint.clear()
        self._show_limits(None)
        self._update_ui()
        if check:
            self.refresh_status()

    def signed_in(self) -> bool:
        return self._state == SIGNED_IN

    def is_busy(self) -> bool:
        return self._worker is not None

    def refresh_status(self) -> None:
        self._start("status", CHECKING)

    def refresh_limits(self) -> None:
        self._start("limits", self._state)

    # --- actions --------------------------------------------------------------------------

    def _on_button(self) -> None:
        if self._state == SIGNING_IN:
            if self._worker is not None:
                self._worker.cancel()
        elif self._state == SIGNED_IN:
            answer = QMessageBox.question(
                self,
                "Выйти из аккаунта",
                f"Выйти из аккаунта {VENDOR_PAGES[self._kind]} в этом инструменте?\n\n"
                "Он станет недоступен и для других программ, которые им пользуются.",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._start("logout", BUSY)
        else:
            self._start("login", SIGNING_IN)

    def _start(self, action: str, state: int) -> None:
        if self._worker is not None:
            return
        try:
            provider = self._provider_source()
        except ValueError as e:
            self._set_status(str(e), ERROR_COLOR)
            return
        worker = LlmWorker(run_account_task, (action, provider.to_dict(), {}), self)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        worker.event.connect(self._on_event)
        self._worker, self._action = worker, action
        self._state = state
        if action == "login":
            self._set_status("Выполняется вход…", MUTED_COLOR)
        elif action == "status":
            self._set_status("Проверка входа…", MUTED_COLOR)
        elif action == "limits":
            self._limits_note.setText(
                "Читаю лимиты…" + (" Claude отвечает на короткий запрос — это займёт несколько секунд."
                                   if self._kind == KIND_CLAUDE_CLI else "")
            )
        self._update_ui()
        try:
            worker.start()
        except OSError as e:
            self._end(f"Не удалось запустить инструмент: {e}", ERROR_COLOR, SIGNED_OUT)

    def _on_event(self, name: str, payload: str) -> None:
        if name == "url":
            link = f'<a href="{html.escape(payload, quote=True)}">открыть страницу входа</a>'
            if open_url(payload):
                self._hint.setText(
                    f"Страница входа {VENDOR_PAGES[self._kind]} открыта в браузере. "
                    f"Завершите вход там — окно обновится само. Если страница не открылась, {link}."
                )
            else:
                self._hint.setText(f"Не удалось открыть браузер: {link}, затем завершите вход там.")
        elif name == "waiting":
            self._hint.setText(
                f"Claude Code открыл страницу входа {VENDOR_PAGES[self._kind]} в браузере. "
                "Завершите вход там — окно обновится само."
            )

    def _on_finished(self, result: dict) -> None:
        action = self._action
        self._finish_worker()
        if action == "limits":
            self._show_limits(result.get("limits"))
            self._update_ui()
            return
        self._hint.clear()
        self._set_account(result.get("account"))
        if self.signed_in() != self._was_signed_in:
            self._was_signed_in = self.signed_in()
            self.signed_in_changed.emit(self._was_signed_in)
        if self.signed_in() and action in ("login", "status") and self._kind == KIND_CODEX_CLI:
            self.refresh_limits()  # a local query for Codex; Claude would need a request, so it waits for the button
        elif not self.signed_in() or self._limits_grid.count() == 0:
            # Nobody signed in has no limits; for Claude (no automatic query) the note says how to get them.
            # Limits already on screen stay when the sign-in is merely re-checked.
            self._show_limits(None)

    def _on_failed(self, _error_type: str, message: str) -> None:
        action = self._action
        self._finish_worker()
        if action == "limits":
            self._limits_note.setText(message)
            self._update_ui()
            return
        self._hint.clear()
        self._end(
            ("Вход не выполнен: " if action == "login" else "") + message,
            ERROR_COLOR,
            SIGNED_IN if self._account and self._account.get("signed_in") and action != "login" else SIGNED_OUT,
        )

    def _on_cancelled(self) -> None:
        action = self._action
        self._finish_worker()
        self._hint.clear()
        if action == "limits":
            self._limits_note.clear()
            self._update_ui()
            return
        self._end("Вход отменён.", MUTED_COLOR, SIGNED_OUT)

    def _finish_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _abandon_worker(self) -> None:
        """Stop a running action and stop listening to it: its late signals must not touch what comes next."""
        worker, self._worker = self._worker, None
        if worker is None:
            return
        for signal in (worker.finished, worker.failed, worker.cancelled, worker.event):
            try:
                signal.disconnect()
            except (RuntimeError, TypeError):
                pass
        worker.cancel()  # kills the child and the vendor tool with it
        worker.deleteLater()

    def _end(self, message: str, color: str, state: int) -> None:
        self._state = state
        self._set_status(message, color)
        self._update_ui()

    # --- showing --------------------------------------------------------------------------

    def _set_account(self, account: Optional[dict]) -> None:
        self._account = account
        signed_in = bool(account and account.get("signed_in"))
        self._state = SIGNED_IN if signed_in else SIGNED_OUT
        text = provider_text.account_text(account)
        color = OK_COLOR if signed_in else MUTED_COLOR
        if signed_in and (account or {}).get("method") == "API key":
            text += (
                "\nВход выполнен по API-ключу, а не по подписке: запросы будут оплачиваться по токенам. "
                "Для подписки выйдите и войдите заново."
            )
            color = WARN_COLOR
        self._set_status(text, color)
        self._update_ui()

    def _set_status(self, text: str, color: str) -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color};")

    def _show_limits(self, limits: Optional[dict]) -> None:
        while self._limits_grid.count():
            item = self._limits_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        if limits is None:
            self._limits_note.setText(
                "Claude сообщает лимиты только вместе с ответом на запрос: «Обновить лимиты» отправит короткий запрос."
                if self._kind == KIND_CLAUDE_CLI and self._state == SIGNED_IN else ""
            )
            return
        for row, window in enumerate(limits.get("windows") or []):
            used = window.get("used_percent")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(round(used)) if isinstance(used, (int, float)) else 0)
            bar.setFormat("%p%" if isinstance(used, (int, float)) else (window.get("status") or "нет данных"))
            bar.setTextVisible(True)
            if isinstance(used, (int, float)) and used >= 90:
                bar.setStyleSheet(f"QProgressBar::chunk {{ background: {ERROR_COLOR}; }}")
            elif isinstance(used, (int, float)) and used >= 70:
                bar.setStyleSheet(f"QProgressBar::chunk {{ background: {WARN_COLOR}; }}")
            reset = QLabel(provider_text.reset_text(window.get("resets_at")))
            reset.setStyleSheet(f"color: {MUTED_COLOR};")
            self._limits_grid.addWidget(QLabel(provider_text.window_name(window)), row, 0)
            self._limits_grid.addWidget(bar, row, 1)
            self._limits_grid.addWidget(reset, row, 2)
        extra = [line for line in provider_text.limits_lines({**limits, "windows": []})]
        if not limits.get("windows") and not extra:
            extra = ["Провайдер не сообщил лимиты."]
        self._limits_note.setText("\n".join(extra))

    def _update_ui(self) -> None:
        busy = self._worker is not None
        state = self._state
        if state == SIGNING_IN:
            self._button.setText("Отменить вход")
        elif state == SIGNED_IN:
            self._button.setText("Выйти")
        else:
            self._button.setText(SIGN_IN_LABELS.get(self._kind, "Войти"))
        # While something is running only a sign-in can be cancelled; the rest waits.
        self._button.setEnabled(not busy or state == SIGNING_IN)
        self._recheck_button.setEnabled(not busy)
        self._limits_button.setEnabled(not busy and state == SIGNED_IN)
        # The plan's limits belong to whoever is signed in: nobody, nothing to show.
        for widget in (self._limits_title, self._limits_button):
            widget.setVisible(state == SIGNED_IN)
        self._limits_note.setVisible(state == SIGNED_IN and bool(self._limits_note.text()))
        self._limits_button.setToolTip(
            "Отправит модели короткий запрос — Claude сообщает лимиты только вместе с ответом"
            if self._kind == KIND_CLAUDE_CLI else "Спросить у Codex, сколько лимита использовано"
        )
# [SEMANTIC-END] GUI:ACCOUNT_PANEL
