"""Offscreen tests of the summarization GUI: the settings tabs (GUI:SUMMARY_SETTINGS),
the summary dialog (GUI:SUMMARY_DIALOG), the child-process tasks and worker behind them
(GUI:CHILD_TASKS, GUI:LLM_WORKER) and the Result screen's entry point.

The model call is replaced by a recorder; one test runs the whole chain in a really
spawned child process against a stand-in executable.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from vendor_stub import make_vendor

pytest.importorskip("PySide6", reason="GUI extra not installed")

from PySide6.QtCore import QEventLoop, QObject, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from diarrhizer.adapters import llm  # noqa: E402
from diarrhizer.adapters.llm import LLMError, LLMReply  # noqa: E402
from diarrhizer.summary import service  # noqa: E402
from diarrhizer.summary.provider_info import Account, LimitWindow, Limits, ModelOption, Usage  # noqa: E402
from diarrhizer.summary.providers import (  # noqa: E402
    KIND_ANTHROPIC_API,
    KIND_CLAUDE_CLI,
    KIND_CODEX_CLI,
    KIND_OPENAI_API,
    Provider,
    ProviderStore,
)
from diarrhizer.summary.templates import BUILTIN_TEMPLATES, PromptTemplate, TemplateStore  # noqa: E402
from diarrhizer_gui import child_tasks  # noqa: E402
from diarrhizer_gui.llm_worker import LlmWorker  # noqa: E402
from diarrhizer_gui.screens import (  # noqa: E402
    account_panel,
    result_screen,
    summary_dialog,
    summary_settings_screen,
)
from diarrhizer_gui.screens.model_combo import DEFAULT_LABEL, ModelCombo  # noqa: E402

REAL_OPEN_URL = account_panel.open_url  # before the autouse fixture below replaces it

TRANSCRIPT = "# Transcription\n\n[00:00:01 → 00:00:05] **Иван:** Привет, начнём встречу.\n"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """The API key goes to a throwaway .env, never to the repo's."""
    monkeypatch.setattr(summary_settings_screen, "ENV_PATH", tmp_path / ".env")
    return tmp_path / ".env"


def _store_providers(*providers: Provider) -> None:
    store = ProviderStore.load()
    for provider in providers:
        store.put(provider)
    store.save()


def _job_with_protocol(tmp_path: Path) -> Path:
    job = tmp_path / "out" / "meeting_20261022_101500"
    (job / "export").mkdir(parents=True)
    (job / "export" / "result.md").write_text(TRANSCRIPT, encoding="utf-8")
    return job


class FakeWorker(QObject):
    """Stands in for LlmWorker: records what it was asked and lets the test decide the outcome."""

    finished = Signal(dict)
    failed = Signal(str, str)
    cancelled = Signal()
    event = Signal(str, str)
    instances: list = []

    def __init__(self, target, args, parent=None):
        super().__init__(parent)
        self.target, self.args = target, args
        self.started = False
        self.cancel_called = False
        FakeWorker.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancel_called = True
        self.cancelled.emit()

    def run_like_the_child(self):
        """Do what the child process would and report it as the worker would."""

        class Queue:
            items: list = []

            def put(self, item):
                self.items.append(item)

        queue = Queue()
        queue.items = []
        self.target(queue, *self.args)
        kind, *rest = queue.items[-1]
        (self.finished if kind == "finished" else self.failed).emit(*rest)


class PanelWorker(FakeWorker):
    """The account panel's workers, kept apart from the tab's own (FakeWorker.instances)."""

    instances: list = []

    def __init__(self, target, args, parent=None):
        super().__init__(target, args, parent)
        FakeWorker.instances.remove(self)  # registered by the base class in the wrong list
        PanelWorker.instances.append(self)


@pytest.fixture(autouse=True)
def isolated_accounts(monkeypatch):
    """No test starts a real codex/claude or opens a browser: the account panel gets stand-in workers.
    Returns the list of addresses the panel "opened"."""
    PanelWorker.instances = []
    opened: list = []
    monkeypatch.setattr(account_panel, "LlmWorker", PanelWorker)
    monkeypatch.setattr(account_panel, "open_url", lambda url: opened.append(url) or True)
    return opened


@pytest.fixture
def fake_worker(monkeypatch):
    FakeWorker.instances = []
    monkeypatch.setattr(summary_settings_screen, "LlmWorker", FakeWorker)
    monkeypatch.setattr(summary_dialog, "LlmWorker", FakeWorker)
    return FakeWorker


@pytest.fixture
def model(monkeypatch):
    class Model:
        reply: object = LLMReply("## Итоги\n- начали встречу")
        providers: list = []

    Model.providers = []

    def fake_complete(provider, prompt):
        Model.providers.append(provider)
        if isinstance(Model.reply, Exception):
            raise Model.reply
        return Model.reply

    monkeypatch.setattr(llm, "complete", fake_complete)
    return Model


def _yes(monkeypatch, answer=QMessageBox.StandardButton.Yes):
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *args, **kwargs: answer))


# --- templates tab ---------------------------------------------------------------------------


def test_builtin_templates_are_listed_and_cannot_be_edited_or_deleted(qapp):
    tab = summary_settings_screen.TemplatesTab()

    assert tab._list.count() == len(BUILTIN_TEMPLATES)
    assert tab._prompt_edit.isReadOnly() and tab._name_field.isReadOnly()
    assert not tab._save_button.isEnabled()
    assert not tab._delete_button.isEnabled()
    assert tab._duplicate_button.isEnabled()


def test_duplicating_a_builtin_and_saving_makes_a_user_template(qapp, summary_stores):
    tab = summary_settings_screen.TemplatesTab()

    tab._duplicate_button.click()
    assert tab._name_field.text() == "Краткое резюме (копия)"
    assert not tab._prompt_edit.isReadOnly()
    tab._prompt_edit.setPlainText("Выпиши только задачи.")
    tab._save_button.click()

    saved = TemplateStore.load().get("Краткое резюме (копия)")
    assert saved.prompt == "Выпиши только задачи."
    assert saved.description == BUILTIN_TEMPLATES["Краткое резюме"].description
    assert tab._list.currentItem().data(Qt.ItemDataRole.UserRole) == "Краткое резюме (копия)"
    assert tab._delete_button.isEnabled()


def test_a_new_template_needs_a_name_and_a_prompt_and_a_free_name(qapp, summary_stores):
    tab = summary_settings_screen.TemplatesTab()
    tab._new_button.click()

    tab._save_button.click()
    assert "name" in tab._status.text().lower()
    tab._name_field.setText("Итоги")
    tab._save_button.click()
    assert "prompt" in tab._status.text().lower()
    tab._prompt_edit.setPlainText("Подведи итоги.")
    tab._name_field.setText("Краткое резюме")  # taken by a built-in
    tab._save_button.click()
    assert "уже есть" in tab._status.text()
    assert not summary_stores[0].exists()

    tab._name_field.setText("Итоги")
    tab._save_button.click()
    assert TemplateStore.load().get("Итоги").prompt == "Подведи итоги."


def test_renaming_a_template_keeps_it_the_default_and_drops_the_old_name(qapp, summary_stores):
    store = TemplateStore.load()
    store.put(PromptTemplate(name="Старое", prompt="p"))
    store.set_default("Старое")
    store.save()
    tab = summary_settings_screen.TemplatesTab()
    tab.reload(select="Старое")

    tab._name_field.setText("Новое")
    tab._save_button.click()

    store = TemplateStore.load()
    assert "Старое" not in store.templates() and "Новое" in store.templates()
    assert store.default().name == "Новое"


def test_deleting_and_defaulting_a_template(qapp, summary_stores, monkeypatch):
    store = TemplateStore.load()
    store.put(PromptTemplate(name="Мой", prompt="p"))
    store.save()
    tab = summary_settings_screen.TemplatesTab()
    tab.reload(select="Мой")

    tab._default_button.click()
    assert TemplateStore.load().default().name == "Мой"
    assert not tab._default_button.isEnabled()

    _yes(monkeypatch, QMessageBox.StandardButton.No)
    tab._delete_button.click()
    assert "Мой" in TemplateStore.load().templates()

    _yes(monkeypatch)
    tab._delete_button.click()
    store = TemplateStore.load()
    assert "Мой" not in store.templates()
    assert store.default().name == "Краткое резюме"


def test_switching_templates_with_unsaved_edits_asks_first(qapp, summary_stores, monkeypatch):
    store = TemplateStore.load()
    store.put(PromptTemplate(name="Мой", prompt="p"))
    store.save()
    tab = summary_settings_screen.TemplatesTab()
    tab.reload(select="Мой")
    tab._prompt_edit.setPlainText("правка")

    _yes(monkeypatch, QMessageBox.StandardButton.No)
    tab._list.setCurrentRow(0)  # a built-in
    assert tab._editing == "Мой" and tab._prompt_edit.toPlainText() == "правка"
    assert tab._list.currentItem().data(Qt.ItemDataRole.UserRole) == "Мой"

    _yes(monkeypatch)
    tab._list.setCurrentRow(0)
    assert tab._editing == "Краткое резюме"
    assert TemplateStore.load().get("Мой").prompt == "p"  # the edit was never saved


def test_a_broken_templates_file_disables_the_tab_instead_of_being_overwritten(qapp, summary_stores):
    summary_stores[0].write_text("{broken", encoding="utf-8")
    tab = summary_settings_screen.TemplatesTab()
    assert not tab._save_button.isEnabled() and not tab._new_button.isEnabled()
    assert "Cannot read" in tab._status.text()
    assert summary_stores[0].read_text(encoding="utf-8") == "{broken"


# --- providers tab ------------------------------------------------------------------------------


def _preset(tab, kind):
    tab._new(kind)
    return tab


def test_with_no_providers_the_tab_explains_how_to_start(qapp):
    tab = summary_settings_screen.ProvidersTab()
    assert tab._list.count() == 0
    assert "Добавить провайдера" in tab._status.text()
    assert not tab._delete_button.isEnabled()


def test_a_cli_provider_form_shows_only_its_own_fields(qapp):
    tab = _preset(summary_settings_screen.ProvidersTab(), KIND_CLAUDE_CLI)
    form = tab._form

    assert tab._name_field.text() == "Claude"
    assert form.isRowVisible(tab._command_field) and form.isRowVisible(tab._extra_args_field)
    assert not form.isRowVisible(tab._endpoint_field) and not form.isRowVisible(tab._key_row)
    assert "правилам Anthropic" in tab._kind_hint.text()  # why the sign-in is the tool's own, not the app's
    assert form.isRowVisible(tab._account_panel)  # a subscription has an account

    tab._new(KIND_OPENAI_API)
    assert not form.isRowVisible(tab._command_field)
    assert not form.isRowVisible(tab._account_panel) and form.isRowVisible(tab._prices_row)
    assert form.isRowVisible(tab._endpoint_field) and form.isRowVisible(tab._key_row)
    assert tab._endpoint_field.text() == "https://api.openai.com/v1"


def test_saving_subscription_providers_stores_no_secrets(qapp, summary_stores):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_CODEX_CLI)
    tab._extra_args_field.setText('-c model_reasoning_effort="low"')
    tab._save_button.click()
    _preset(tab, KIND_CLAUDE_CLI)
    tab._model_field.setText("sonnet")
    tab._save_button.click()

    store = ProviderStore.load()
    assert store.names() == ["ChatGPT", "Claude"]
    assert store.default().name == "ChatGPT"
    assert store.get("ChatGPT").extra_args == ("-c", "model_reasoning_effort=low")
    assert store.get("Claude").model == "sonnet"
    assert tab._list.count() == 2 and "✓" in tab._list.item(0).text()


def test_an_api_key_goes_to_env_not_to_the_providers_file(qapp, summary_stores, isolated_env, monkeypatch):
    monkeypatch.delenv("DIARRHIZER_LLM_KEY_OPENROUTER", raising=False)
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_OPENAI_API)
    tab._name_field.setText("OpenRouter")
    tab._endpoint_field.setText("https://openrouter.ai/api/v1")
    tab._model_field.setText("anthropic/claude-sonnet")
    tab._key_field.setText("sk-or-secret")
    tab._temperature_field.setText("0,3")
    tab._max_tokens_field.setText("4000")
    tab._save_button.click()

    provider = ProviderStore.load().get("OpenRouter")
    assert provider.api_key_env == "DIARRHIZER_LLM_KEY_OPENROUTER"
    assert (provider.temperature, provider.max_tokens) == (0.3, 4000)
    assert "sk-or-secret" not in summary_stores[1].read_text(encoding="utf-8")
    assert "DIARRHIZER_LLM_KEY_OPENROUTER=sk-or-secret" in isolated_env.read_text(encoding="utf-8")
    assert provider.api_key() == "sk-or-secret"  # usable at once, no restart
    assert "Ключ записан" in tab._status.text()

    # Reopening the provider shows the key again (masked) from the environment.
    tab.reload(select="OpenRouter")
    assert tab._key_field.text() == "sk-or-secret"
    assert tab._key_field.echoMode() == tab._key_field.EchoMode.Password


def test_a_local_openai_compatible_server_needs_no_key(qapp, summary_stores, isolated_env):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_OPENAI_API)
    tab._name_field.setText("Ollama")
    tab._endpoint_field.setText("http://localhost:11434/v1")
    tab._model_field.setText("llama3.1")
    tab._save_button.click()

    provider = ProviderStore.load().get("Ollama")
    assert provider.api_key_env == "" and not isolated_env.exists()


def test_the_anthropic_api_gets_a_key_variable_even_before_a_key_is_typed(qapp, summary_stores):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_ANTHROPIC_API)
    tab._model_field.setText("claude-sonnet-5-5")
    tab._save_button.click()
    assert ProviderStore.load().get("Anthropic API").api_key_env == "DIARRHIZER_LLM_KEY_ANTHROPIC_API"


@pytest.mark.parametrize(
    "setup, message",
    [
        (lambda tab: tab._model_field.setText(""), "model"),
        (lambda tab: tab._temperature_field.setText("hot"), "Температура"),
        (lambda tab: tab._max_tokens_field.setText("1.5"), "Макс. токенов"),
        (lambda tab: tab._endpoint_field.setText("api.example.com"), "http"),
        (lambda tab: tab._name_field.setText(""), "name"),
    ],
)
def test_invalid_api_provider_fields_are_reported_and_nothing_is_saved(qapp, summary_stores, setup, message):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_OPENAI_API)
    tab._name_field.setText("gw")
    tab._model_field.setText("m")
    setup(tab)
    tab._save_button.click()
    assert message in tab._status.text()
    assert not summary_stores[1].exists()


def test_renaming_deleting_and_defaulting_providers(qapp, summary_stores, monkeypatch):
    _store_providers(Provider(name="a", kind=KIND_CODEX_CLI), Provider(name="b", kind=KIND_CLAUDE_CLI))
    tab = summary_settings_screen.ProvidersTab()

    tab.reload(select="a")
    tab._name_field.setText("a2")
    tab._save_button.click()
    store = ProviderStore.load()
    assert store.names() == ["b", "a2"] or store.names() == ["a2", "b"]
    assert store.default().name == "a2"  # the rename followed the default

    tab.reload(select="b")
    tab._default_button.click()
    assert ProviderStore.load().default().name == "b"

    _yes(monkeypatch)
    tab._delete_button.click()
    store = ProviderStore.load()
    assert store.names() == ["a2"] and store.default().name == "a2"


def test_a_provider_cannot_take_an_existing_name(qapp, summary_stores):
    _store_providers(Provider(name="a", kind=KIND_CODEX_CLI), Provider(name="b", kind=KIND_CLAUDE_CLI))
    tab = summary_settings_screen.ProvidersTab()
    tab.reload(select="b")
    tab._name_field.setText("a")
    tab._save_button.click()
    assert "уже есть" in tab._status.text()
    assert ProviderStore.load().names() == ["a", "b"]


def test_the_connection_check_uses_the_forms_values_and_a_typed_key(qapp, summary_stores, fake_worker):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_OPENAI_API)
    tab._name_field.setText("gw")
    tab._model_field.setText("m")
    tab._key_field.setText("sk-typed")

    tab._test_button.click()

    (worker,) = fake_worker.instances
    assert worker.started
    provider_dict, env = worker.args
    assert provider_dict["model"] == "m" and provider_dict["api_key_env"] == "DIARRHIZER_LLM_KEY_GW"
    assert env == {"DIARRHIZER_LLM_KEY_GW": "sk-typed"}  # handed to the child for this call only
    assert "sk-typed" not in json.dumps(provider_dict)
    assert not summary_stores[1].exists()  # checking does not save
    assert tab._test_button.text() == "Отменить проверку"

    worker.finished.emit({"reply": "ok"})
    assert "✓" in tab._status.text() and "ok" in tab._status.text()
    assert tab._test_button.text() == "Проверить"


def test_a_failed_check_shows_the_reason_and_can_be_cancelled(qapp, summary_stores, fake_worker):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_CODEX_CLI)

    tab._test_button.click()
    fake_worker.instances[0].failed.emit("LLMError", "Provider 'ChatGPT': codex exited with code 1")
    assert "exited with code 1" in tab._status.text()

    tab._test_button.click()
    tab._test_button.click()  # second click while running = cancel
    assert fake_worker.instances[1].cancel_called
    assert "отменена" in tab._status.text()


def test_an_invalid_form_is_not_sent_for_checking(qapp, summary_stores, fake_worker):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_OPENAI_API)  # no model yet
    tab._test_button.click()
    assert fake_worker.instances == []
    assert "model" in tab._status.text()


def test_an_unsaved_provider_form_survives_leaving_the_screen(qapp, summary_stores):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_OPENAI_API)
    tab._name_field.setText("draft")
    tab.hide()
    tab.show()
    assert tab._name_field.text() == "draft"


# --- child tasks ----------------------------------------------------------------------------------


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_summary_task_reports_the_stored_summary(tmp_path, model):
    job = _job_with_protocol(tmp_path)
    queue = _Queue()
    template = PromptTemplate(name="Итоги", prompt="Подведи итоги.")
    provider = Provider(name="ChatGPT", kind=KIND_CODEX_CLI)

    child_tasks.run_summary_task(queue, str(job), template.to_dict(), provider.to_dict())

    ((kind, result),) = queue.items
    assert kind == "finished"
    assert result["text"] == "## Итоги\n- начали встречу"
    assert Path(result["path"]).exists()
    assert model.providers == [provider]


def test_summary_task_reports_failures_instead_of_raising(tmp_path, model):
    job = _job_with_protocol(tmp_path)
    model.reply = LLMError("Provider 'ChatGPT': codex exited with code 1")
    queue = _Queue()

    child_tasks.run_summary_task(
        queue, str(job), PromptTemplate(name="t", prompt="p").to_dict(),
        Provider(name="ChatGPT", kind=KIND_CODEX_CLI).to_dict(),
    )
    assert queue.items == [("failed", "LLMError", "Provider 'ChatGPT': codex exited with code 1")]

    queue = _Queue()
    child_tasks.run_summary_task(  # a job without a protocol
        queue, str(tmp_path / "nowhere"), PromptTemplate(name="t", prompt="p").to_dict(),
        Provider(name="ChatGPT", kind=KIND_CODEX_CLI).to_dict(),
    )
    assert queue.items[0][:2] == ("failed", "FileNotFoundError")


def test_provider_check_task_sets_the_typed_key_for_the_call(monkeypatch, model):
    monkeypatch.delenv("CHECK_KEY", raising=False)
    seen = {}
    monkeypatch.setattr(llm, "complete", lambda provider, prompt: seen.update(key=provider.api_key()) or LLMReply("ok"))
    queue = _Queue()
    provider = Provider(name="gw", kind=KIND_OPENAI_API, endpoint="http://x/v1", model="m", api_key_env="CHECK_KEY")

    child_tasks.run_provider_check_task(queue, provider.to_dict(), {"CHECK_KEY": "typed"})

    ((kind, result),) = queue.items
    assert kind == "finished" and result["reply"] == "ok"
    assert seen["key"] == "typed"


# --- summary dialog ------------------------------------------------------------------------------------


@pytest.fixture
def configured():
    templates = TemplateStore.load()
    templates.put(PromptTemplate(name="Мой шаблон", prompt="Сделай мой разбор."))
    templates.set_default("Мой шаблон")
    templates.save()
    _store_providers(
        Provider(name="ChatGPT", kind=KIND_CODEX_CLI),
        Provider(name="gw", kind=KIND_OPENAI_API, endpoint="https://api.example.com/v1", model="m"),
    )


def test_dialog_offers_the_defaults_and_says_where_the_text_goes(qapp, tmp_path, configured):
    dialog = summary_dialog.SummaryDialog(None, _job_with_protocol(tmp_path), "meeting.mp4")

    assert dialog._template_combo.currentData() == "Мой шаблон"
    assert dialog._provider_combo.currentData() == "ChatGPT"
    assert dialog._create_button.isEnabled()
    assert "через Codex CLI (аккаунт ChatGPT)" in dialog._target_label.text()
    assert "export/result.md" in dialog._target_label.text()

    dialog._provider_combo.setCurrentIndex(dialog._provider_combo.findData("gw"))
    assert "на https://api.example.com/v1" in dialog._target_label.text()
    assert "Протокол (export/result.md, " in dialog._target_label.text()  # the comma survived the formatting
    assert "будут" not in dialog._status.text()
    assert dialog._view.toPlainText().startswith("Саммари для этого задания ещё не создавались")


def test_dialog_without_providers_says_where_to_add_one(qapp, tmp_path):
    dialog = summary_dialog.SummaryDialog(None, _job_with_protocol(tmp_path), "meeting.mp4")
    assert not dialog._create_button.isEnabled()
    assert "Провайдеры" in dialog._status.text()


def test_dialog_survives_broken_stores_and_a_missing_protocol(qapp, tmp_path, summary_stores, configured):
    summary_stores[0].write_text("{broken", encoding="utf-8")
    dialog = summary_dialog.SummaryDialog(None, _job_with_protocol(tmp_path), "meeting.mp4")
    assert "Cannot read summary templates file" in dialog._status.text()
    assert dialog._template_combo.count() == len(BUILTIN_TEMPLATES)  # built-ins still usable
    assert dialog._create_button.isEnabled()

    empty_job = tmp_path / "out" / "empty_job"
    empty_job.mkdir(parents=True)
    dialog = summary_dialog.SummaryDialog(None, empty_job, "x")
    assert not dialog._create_button.isEnabled()
    assert "result.md" in dialog._status.text()


def test_creating_a_summary_runs_the_task_and_shows_the_stored_result(qapp, tmp_path, configured, fake_worker, model):
    job = _job_with_protocol(tmp_path)
    dialog = summary_dialog.SummaryDialog(None, job, "meeting.mp4")

    dialog._create_button.click()
    (worker,) = fake_worker.instances
    assert worker.started and worker.target is child_tasks.run_summary_task
    job_arg, template_dict, provider_dict = worker.args
    assert job_arg == str(job)
    assert template_dict["name"] == "Мой шаблон" and provider_dict["name"] == "ChatGPT"
    assert dialog._create_button.text() == "Отменить"
    assert not dialog._template_combo.isEnabled() and not dialog._provider_combo.isEnabled()

    worker.run_like_the_child()

    assert dialog._create_button.text() == "Создать саммари" and dialog._provider_combo.isEnabled()
    assert "Готово" in dialog._status.text() and "summary/" in dialog._status.text()
    assert "начали встречу" in dialog._view.toPlainText()  # rendered Markdown, not raw
    assert "##" not in dialog._view.toPlainText()
    assert "Мой шаблон" in dialog._meta_label.text() and "ChatGPT" in dialog._meta_label.text()
    assert dialog._summary_combo.count() == 1
    (stored,) = service.list_summaries(job)
    assert stored["template"] == "Мой шаблон"

    # A second run is listed first and shown; the first stays selectable.
    model.reply = LLMReply("Второе саммари")
    dialog._create_button.click()
    fake_worker.instances[1].run_like_the_child()
    assert dialog._summary_combo.count() == 2
    assert "Второе саммари" in dialog._view.toPlainText()
    dialog._summary_combo.setCurrentIndex(1)
    assert "начали встречу" in dialog._view.toPlainText()


def test_a_failed_summary_shows_the_reason_and_leaves_the_dialog_usable(qapp, tmp_path, configured, fake_worker, model):
    dialog = summary_dialog.SummaryDialog(None, _job_with_protocol(tmp_path), "meeting.mp4")
    model.reply = LLMError("Provider 'ChatGPT': codex exited with code 1:\nNot logged in")

    dialog._create_button.click()
    fake_worker.instances[0].run_like_the_child()

    assert "Not logged in" in dialog._status.text()
    assert dialog._create_button.isEnabled() and dialog._create_button.text() == "Создать саммари"
    assert dialog._summary_combo.count() == 0


def test_a_cut_off_reply_is_flagged(qapp, tmp_path, configured, fake_worker, model):
    dialog = summary_dialog.SummaryDialog(None, _job_with_protocol(tmp_path), "meeting.mp4")
    model.reply = LLMReply("Начало ответа", truncated=True)
    dialog._create_button.click()
    fake_worker.instances[0].run_like_the_child()
    assert "обрезан" in dialog._status.text()
    assert "обрезан" in dialog._meta_label.text()


def test_cancelling_a_running_request_stores_nothing(qapp, tmp_path, configured, fake_worker):
    job = _job_with_protocol(tmp_path)
    dialog = summary_dialog.SummaryDialog(None, job, "meeting.mp4")

    dialog._create_button.click()
    dialog._create_button.click()  # now "Отменить"

    assert fake_worker.instances[0].cancel_called
    assert "отменён" in dialog._status.text()
    assert dialog._create_button.text() == "Создать саммари"
    assert service.list_summaries(job) == []


def test_closing_during_a_request_asks_and_then_cancels(qapp, tmp_path, configured, fake_worker, monkeypatch):
    dialog = summary_dialog.SummaryDialog(None, _job_with_protocol(tmp_path), "meeting.mp4")
    dialog.show()
    dialog._create_button.click()
    worker = fake_worker.instances[0]

    _yes(monkeypatch, QMessageBox.StandardButton.No)
    dialog.reject()
    assert dialog.isVisible() and not worker.cancel_called

    _yes(monkeypatch)
    dialog.reject()
    assert worker.cancel_called and not dialog.isVisible()


def test_copy_puts_the_raw_markdown_on_the_clipboard(qapp, tmp_path, configured, fake_worker, model):
    dialog = summary_dialog.SummaryDialog(None, _job_with_protocol(tmp_path), "meeting.mp4")
    dialog._create_button.click()
    fake_worker.instances[0].run_like_the_child()

    dialog._copy_button.click()

    assert QApplication.clipboard().text().startswith("## Итоги")


def test_dialog_lists_summaries_made_earlier_from_the_cli(qapp, tmp_path, configured, model):
    job = _job_with_protocol(tmp_path)
    service.summarize_job(job, TemplateStore.load().get("Мой шаблон"), ProviderStore.load().get("gw"))

    dialog = summary_dialog.SummaryDialog(None, job, "meeting.mp4")

    assert dialog._summary_combo.count() == 1
    assert "Мой шаблон · gw" in dialog._summary_combo.itemText(0)
    assert "начали встречу" in dialog._view.toPlainText()


# --- the Result screen's entry point ---------------------------------------------------------------------


SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "Привет.", "speaker_id": "Speaker_00"},
    {"start": 5.0, "end": 9.0, "text": "Здравствуйте.", "speaker_id": "Speaker_01"},
]


def _result_job(tmp_path: Path) -> Path:
    job = tmp_path / "out" / "call_20260101_000000"
    (job / "merged").mkdir(parents=True)
    (job / "merged" / "segments.json").write_text(json.dumps({"segments": SEGMENTS}), encoding="utf-8")
    (job / "meta").mkdir()
    (job / "meta" / "run.json").write_text(
        json.dumps({"input_path": str(tmp_path / "call.wav"),
                    "pipeline_config": {"language": "ru", "device": "cpu", "min_speakers": 2, "max_speakers": 2}}),
        encoding="utf-8",
    )
    return job


class _DialogSpy:
    opened: list = []

    def __init__(self, parent, job_dir, title):
        self.args = (job_dir, title)

    def exec(self):
        _DialogSpy.opened.append(self.args)


@pytest.fixture
def dialog_spy(monkeypatch):
    _DialogSpy.opened = []
    monkeypatch.setattr(result_screen, "SummaryDialog", _DialogSpy)
    return _DialogSpy


def test_the_summary_button_needs_a_protocol(qapp, tmp_path):
    job = _result_job(tmp_path)
    screen = result_screen.ResultScreen()

    screen.load(job)
    assert not screen._summary_button.isEnabled()

    result_screen.reexport(job, {})
    screen.load(job)
    assert screen._summary_button.isEnabled()


def test_the_summary_button_opens_the_dialog_for_the_job(qapp, tmp_path, dialog_spy):
    job = _result_job(tmp_path)
    result_screen.reexport(job, {})
    screen = result_screen.ResultScreen()
    screen.load(job)

    screen._summary_button.click()

    assert dialog_spy.opened == [(job, "call.wav")]


def _choose_button(monkeypatch, label: str):
    """Make the 'unsaved names' box answer with the button labelled `label`.

    By label, not position: QMessageBox.buttons() follows the platform's button layout, not the order added.
    """
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(
        QMessageBox, "clickedButton",
        lambda self: next(button for button in self.buttons() if button.text() == label),
    )


@pytest.mark.parametrize(
    "choice, opens, saved",
    [("Сохранить", True, True), ("Продолжить без сохранения", True, False), ("Отмена", False, False)],
)
def test_unsaved_speaker_names_are_offered_for_saving_before_summarizing(
    qapp, tmp_path, dialog_spy, monkeypatch, choice, opens, saved
):
    job = _result_job(tmp_path)
    result_screen.reexport(job, {})
    screen = result_screen.ResultScreen()
    screen.load(job)
    screen._speaker_fields["Speaker_01"].setText("Мария")
    _choose_button(monkeypatch, choice)

    screen._summary_button.click()

    assert bool(dialog_spy.opened) is opens
    assert (job / "speakers.json").exists() is saved
    if saved:
        assert "Мария" in (job / "export" / "result.md").read_text(encoding="utf-8")


# --- really spawned ---------------------------------------------------------------------------------------


FAKE_CLI = '''
import json, os, sys, time
sys.stdin.buffer.read()
marker = os.environ.get("FAKE_MARKER")
if marker:
    with open(marker, "w") as f:
        f.write(str(os.getpid()))
    time.sleep(120)
event = {"type": "result", "subtype": "success", "is_error": False, "result": "Ответ из дочернего процесса",
         "total_cost_usd": 0.002, "usage": {"input_tokens": 5, "output_tokens": 3}}
sys.stdout.buffer.write((json.dumps(event, ensure_ascii=False) + chr(10)).encode("utf-8"))
'''


def _fake_cli(tmp_path: Path) -> str:
    script = tmp_path / "fake_cli.py"
    script.write_text(FAKE_CLI, encoding="utf-8")
    if sys.platform == "win32":
        launcher = tmp_path / "claude.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
    else:
        launcher = tmp_path / "claude"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(0o755)
    return str(launcher)


def _wait_for_any(*signals, timeout_ms=60_000):
    loop = QEventLoop()
    outcome = []
    for name, signal in signals:
        signal.connect(lambda *args, n=name: (outcome.append((n, args)), loop.quit()))
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    assert outcome, "timed out waiting for the worker"
    return outcome[0]


def test_a_provider_check_runs_through_a_real_child_process(qapp, tmp_path):
    provider = Provider(name="Claude", kind=KIND_CLAUDE_CLI, command=_fake_cli(tmp_path), timeout_seconds=30)
    worker = LlmWorker(child_tasks.run_provider_check_task, (provider.to_dict(), {}))
    worker.start()

    name, args = _wait_for_any(("finished", worker.finished), ("failed", worker.failed))

    assert name == "finished"
    (result,) = args
    assert result["reply"] == "Ответ из дочернего процесса"
    assert result["usage"]["input_tokens"] == 5 and result["cost_usd"] == 0.002


def test_a_real_summary_job_in_a_child_process(qapp, tmp_path):
    job = _job_with_protocol(tmp_path)
    provider = Provider(name="Claude", kind=KIND_CLAUDE_CLI, command=_fake_cli(tmp_path), timeout_seconds=30)
    template = PromptTemplate(name="Итоги", prompt="Подведи итоги.")
    worker = LlmWorker(child_tasks.run_summary_task, (str(job), template.to_dict(), provider.to_dict()))
    worker.start()

    name, (result,) = _wait_for_any(("finished", worker.finished), ("failed", worker.failed))

    assert name == "finished"
    assert Path(result["path"]).read_text(encoding="utf-8") == "Ответ из дочернего процесса\n"
    assert [s["template"] for s in service.list_summaries(job)] == ["Итоги"]
    assert result["usage"]["output_tokens"] == 3 and result["cost_usd"] == 0.002


@pytest.mark.skipif(sys.platform != "win32", reason="checks the Windows process tree kill")
def test_cancelling_kills_the_cli_started_by_the_child(qapp, tmp_path, monkeypatch):
    marker = tmp_path / "pid.txt"
    monkeypatch.setenv("FAKE_MARKER", str(marker))  # the spawned child inherits it
    provider = Provider(name="Claude", kind=KIND_CLAUDE_CLI, command=_fake_cli(tmp_path), timeout_seconds=120)
    worker = LlmWorker(child_tasks.run_provider_check_task, (provider.to_dict(), {}))
    worker.start()

    deadline = time.monotonic() + 40
    while not marker.exists() and time.monotonic() < deadline:  # the stand-in CLI is up and waiting
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec()
    assert marker.exists(), "the stand-in CLI never started"
    cli_pid = int(marker.read_text())

    worker.cancel()
    name, _args = _wait_for_any(("cancelled", worker.cancelled), ("failed", worker.failed), ("finished", worker.finished))

    assert name == "cancelled"
    time.sleep(0.5)
    listing = subprocess.run(
        ["tasklist", "/FI", f"PID eq {cli_pid}", "/NH"], capture_output=True, text=True
    ).stdout
    assert str(cli_pid) not in listing, "the CLI outlived the cancelled request"


# --- the model list -------------------------------------------------------------------------------------


def test_the_model_list_shows_names_returns_ids_and_keeps_a_typed_id(qapp):
    combo = ModelCombo()
    assert combo.count() == 1 and combo.text() == ""  # only "the provider's default" until a list arrives

    combo.set_models(
        [ModelOption(id="gpt-6-astra", name="GPT-6-Astra", description="Frontier"), ModelOption(id="gpt-6-sol")],
        allow_default=True,
    )
    assert [combo.itemText(i) for i in range(combo.count())] == [
        DEFAULT_LABEL, "GPT-6-Astra (gpt-6-astra)", "gpt-6-sol"
    ]
    combo.setCurrentIndex(1)
    assert combo.text() == "gpt-6-astra"  # the id, not the label
    assert combo.option().description == "Frontier"

    combo.setText("my-private-model")  # a provider's list is a convenience, not a gate
    assert combo.text() == "my-private-model" and combo.option() is None
    combo.set_models([ModelOption(id="gpt-7")], allow_default=True)
    assert combo.text() == "my-private-model"  # a new list does not eat what the user typed

    combo.setText("")
    assert combo.text() == "" and combo.currentText() == DEFAULT_LABEL


def test_api_providers_have_no_default_entry_and_a_provider_default_keeps_its_own_name(qapp):
    combo = ModelCombo()
    combo.set_models([ModelOption(id="gpt-4o-mini"), ModelOption(id="", name="ignored")], allow_default=False)
    assert combo.count() == 1 and combo.text() == "" and combo.currentText() == ""  # a model must be chosen

    combo.set_models(
        [ModelOption(id="", name="Default (recommended)", default=True), ModelOption(id="sonnet", name="Sonnet")],
        allow_default=True,
    )
    assert combo.itemText(0) == f"{DEFAULT_LABEL} — Default (recommended)" and combo.count() == 2  # none added twice


def test_the_default_entry_says_which_model_the_provider_currently_defaults_to(qapp):
    combo = ModelCombo()
    combo.set_models(
        [ModelOption(id="gpt-6-astra", name="GPT-6-Astra", default=True), ModelOption(id="gpt-6-sol", name="GPT-6-Sol")],
        allow_default=True,
    )

    assert combo.itemText(1) == "GPT-6-Astra (gpt-6-astra) — по умолчанию"
    assert combo.option().description == "Сейчас по умолчанию: GPT-6-Astra"  # the "(по умолчанию)" entry is selected
    combo.setText("gpt-6-astra")
    assert combo.text() == "gpt-6-astra"  # the marker is in the label only; the value is the plain id

    combo.set_models([ModelOption(id="gpt-6-sol")], allow_default=True)  # a list that names no default
    assert combo.itemData(0, Qt.ItemDataRole.ToolTipRole) == ""


# --- the account panel -------------------------------------------------------------------------------------

SIGNED_IN = {"signed_in": True, "email": "me@example.com", "plan": "plus", "method": "ChatGPT", "organization": ""}
SIGNED_OUT = {"signed_in": False, "email": "", "plan": "", "method": "", "organization": ""}
LIMITS = {
    "windows": [
        {"label": "", "used_percent": 18.0, "window_minutes": 10080, "resets_at": 1793293782.0, "status": ""},
        {"label": "", "used_percent": 95.0, "window_minutes": 300, "resets_at": 1793200000.0, "status": ""},
    ],
    "credits": "none", "allowed": False, "note": "",
}


def _panel(kind=KIND_CODEX_CLI):
    # The provider the panel acts on follows its kind, as the tab's form does.
    panel = account_panel.AccountPanel(lambda: Provider(name="P", kind=panel._kind))
    panel.set_kind(kind)
    return panel


def _last() -> PanelWorker:
    return PanelWorker.instances[-1]


def _signed_out_panel(kind=KIND_CODEX_CLI):
    panel = _panel(kind)
    _last().finished.emit({"account": SIGNED_OUT})
    return panel


def test_the_panel_checks_the_sign_in_as_soon_as_a_provider_is_shown(qapp):
    panel = _panel()

    (worker,) = PanelWorker.instances
    action, provider, env = worker.args
    assert action == "status" and provider["kind"] == "codex-cli" and env == {}
    assert worker.target is child_tasks.run_account_task and worker.started
    assert "Проверка входа" in panel._status.text() and not panel._button.isEnabled()

    worker.finished.emit({"account": SIGNED_OUT})
    assert panel._status.text() == "Вход не выполнен"
    assert panel._button.text() == "Войти через ChatGPT" and panel._button.isEnabled()
    assert not panel._limits_button.isEnabled()  # nothing to ask about the plan of nobody


def test_limits_are_only_offered_to_somebody_who_is_signed_in(qapp):
    panel = _signed_out_panel()
    panel.show()
    assert panel._limits_title.isHidden() and panel._limits_button.isHidden() and panel._limits_note.isHidden()

    panel._button.click()
    _last().finished.emit({"account": SIGNED_IN})
    _last().finished.emit({"limits": LIMITS})
    assert not panel._limits_title.isHidden() and not panel._limits_button.isHidden()
    assert not panel._limits_note.isHidden()


def test_one_button_signs_in_hands_the_address_to_the_browser_and_shows_the_account(qapp, isolated_accounts):
    panel = _signed_out_panel()
    changes = []
    panel.signed_in_changed.connect(changes.append)

    panel._button.click()
    login = _last()
    assert login.args[0] == "login" and login.started
    assert panel._button.text() == "Отменить вход" and panel._button.isEnabled()  # the same button, now cancel

    login.event.emit("url", "https://auth.example/authorize?state=1&x=2")
    assert isolated_accounts == ["https://auth.example/authorize?state=1&x=2"]
    assert "открыта в браузере" in panel._hint.text() and "OpenAI" in panel._hint.text()
    assert "&amp;x=2" in panel._hint.text()  # the fallback link is escaped for the label's rich text

    login.finished.emit({"account": SIGNED_IN})
    assert changes == [True] and panel.signed_in()
    assert panel._button.text() == "Выйти"
    assert "me@example.com" in panel._status.text() and "тариф Plus" in panel._status.text()
    assert panel._hint.text() == ""
    assert _last().args[0] == "limits"  # Codex answers a limits query locally, so it is asked at once


def test_claude_sign_in_only_waits_for_the_browser_the_tool_opened(qapp, isolated_accounts):
    panel = _signed_out_panel(KIND_CLAUDE_CLI)
    assert panel._button.text() == "Войти через Claude"

    panel._button.click()
    _last().event.emit("waiting", "")

    assert isolated_accounts == []  # Claude Code opens its own page; the app does not second-guess it
    assert "Anthropic" in panel._hint.text() and "Завершите вход" in panel._hint.text()


def test_a_sign_in_in_progress_can_be_cancelled(qapp):
    panel = _signed_out_panel()
    panel._button.click()
    login = _last()

    panel._button.click()  # "Отменить вход"

    assert login.cancel_called
    assert panel._status.text() == "Вход отменён."
    assert panel._button.text() == "Войти через ChatGPT" and panel._button.isEnabled() and not panel.signed_in()


def test_a_failed_sign_in_says_why(qapp):
    panel = _signed_out_panel()
    panel._button.click()
    _last().failed.emit("LLMError", "Provider 'P': sign-in failed: access denied")
    assert panel._status.text().startswith("Вход не выполнен: ") and "access denied" in panel._status.text()
    assert not panel.signed_in() and panel._button.text() == "Войти через ChatGPT"


def test_signing_out_asks_first(qapp, monkeypatch):
    panel = _signed_out_panel()
    panel._button.click()
    _last().finished.emit({"account": SIGNED_IN})
    assert not panel._button.isEnabled()  # busy with the automatic limits query: nothing to click yet
    _last().finished.emit({"limits": LIMITS})
    PanelWorker.instances.clear()
    changes = []
    panel.signed_in_changed.connect(changes.append)

    _yes(monkeypatch, QMessageBox.StandardButton.No)
    panel._button.click()
    assert PanelWorker.instances == [] and panel.signed_in()

    _yes(monkeypatch)
    panel._button.click()
    assert _last().args[0] == "logout"
    _last().finished.emit({"account": SIGNED_OUT})
    assert changes == [False] and not panel.signed_in() and panel._status.text() == "Вход не выполнен"
    assert panel._limits_grid.count() == 0  # the plan of nobody has no limits


def test_limits_are_shown_as_bars_with_reset_times_credits_and_a_warning(qapp):
    panel = _signed_out_panel()
    panel._button.click()
    _last().finished.emit({"account": SIGNED_IN})
    limits = _last()
    assert limits.args[0] == "limits"

    limits.finished.emit({"limits": LIMITS})

    assert panel._limits_grid.count() == 6  # a name, a bar and a reset time for each of the two windows
    assert panel._limits_grid.itemAtPosition(0, 0).widget().text() == "7 дн."
    bar = panel._limits_grid.itemAtPosition(0, 1).widget()
    assert bar.value() == 18 and bar.format() == "%p%"
    assert "сброс" in panel._limits_grid.itemAtPosition(0, 2).widget().text()
    assert panel._limits_grid.itemAtPosition(1, 0).widget().text() == "5 ч"
    assert panel._limits_grid.itemAtPosition(1, 1).widget().value() == 95
    assert "Кредитов нет" in panel._limits_note.text() and "Лимит тарифа исчерпан" in panel._limits_note.text()


def test_claude_limits_need_a_request_and_the_panel_says_so(qapp):
    panel = _signed_out_panel(KIND_CLAUDE_CLI)
    panel._button.click()
    _last().finished.emit({"account": {**SIGNED_IN, "method": "claude.ai", "plan": "max"}})

    assert len(PanelWorker.instances) == 2  # status, login - and no automatic limits query: it would be a request
    assert "короткий запрос" in panel._limits_note.text()
    assert panel._limits_button.isEnabled()

    panel._limits_button.click()
    asked = _last()
    assert asked.args[0] == "limits" and asked.args[1]["kind"] == "claude-cli"
    asked.finished.emit({"limits": {"windows": [{"label": "five_hour", "used_percent": 82.0, "window_minutes": 300,
                                                 "resets_at": None, "status": "allowed_warning"}],
                                    "credits": "", "allowed": True, "note": ""}})
    assert panel._limits_grid.itemAtPosition(0, 0).widget().text() == "5 ч"  # not the raw "five_hour"
    assert panel._limits_grid.itemAtPosition(0, 1).widget().value() == 82


def test_a_sign_in_by_api_key_is_flagged_because_it_is_not_the_subscription(qapp):
    panel = _signed_out_panel()
    panel._button.click()
    _last().finished.emit({"account": {**SIGNED_IN, "method": "API key", "email": ""}})
    assert "по API-ключу" in panel._status.text() and "по токенам" in panel._status.text()


def test_switching_the_provider_abandons_a_running_action_and_its_late_answer(qapp):
    panel = _signed_out_panel()
    panel._button.click()
    login = _last()

    panel.set_kind(KIND_CLAUDE_CLI)

    assert login.cancel_called  # the tool of the provider no longer shown is killed
    new_check = _last()
    assert new_check is not login and new_check.args[1]["kind"] == "claude-cli"
    login.finished.emit({"account": SIGNED_IN})  # too late: it must not become the new provider's account
    assert not panel.signed_in() and panel._status.text() == "Проверка входа…"


def test_only_https_addresses_are_ever_opened(qapp):
    assert REAL_OPEN_URL("http://insecure.example/login") is False
    assert REAL_OPEN_URL("file:///C:/Windows/notepad.exe") is False
    assert REAL_OPEN_URL("javascript:alert(1)") is False


# --- the providers tab with accounts, models and prices ------------------------------------------------------------


def test_choosing_a_subscription_provider_checks_its_sign_in_once(qapp, summary_stores):
    _store_providers(Provider(name="ChatGPT", kind=KIND_CODEX_CLI), Provider(name="Claude", kind=KIND_CLAUDE_CLI))
    tab = summary_settings_screen.ProvidersTab()

    (first,) = PanelWorker.instances
    assert first.args[0] == "status" and first.args[1]["kind"] == "codex-cli"
    assert tab._form.isRowVisible(tab._account_panel)

    tab.reload(select="ChatGPT")  # the same provider again: nothing to re-check
    assert len(PanelWorker.instances) == 1

    tab._list.setCurrentRow(1)
    assert len(PanelWorker.instances) == 2 and PanelWorker.instances[1].args[1]["kind"] == "claude-cli"
    assert first.cancel_called  # what was asked about ChatGPT is dropped

    tab._new(KIND_OPENAI_API)  # an API provider has a key, not an account
    assert len(PanelWorker.instances) == 2 and not tab._form.isRowVisible(tab._account_panel)


def test_models_are_fetched_from_the_provider_once_signed_in_and_remembered(qapp, summary_stores, fake_worker):
    _store_providers(Provider(name="ChatGPT", kind=KIND_CODEX_CLI))
    tab = summary_settings_screen.ProvidersTab()

    tab._account_panel.signed_in_changed.emit(True)

    (worker,) = fake_worker.instances
    assert worker.args[0] == "models" and worker.args[1]["kind"] == "codex-cli"
    assert tab._models_button.text() == "Отмена"
    worker.finished.emit({"models": [
        ModelOption(id="gpt-6-astra", name="GPT-6-Astra", description="Frontier", default=True).to_dict(),
        ModelOption(id="gpt-6-sol", name="GPT-6-Sol").to_dict(),
    ]})
    assert "Моделей у провайдера: 2" in tab._status.text() and tab._models_button.text() == "Список моделей"
    assert tab._model_field.count() == 3  # the provider's default and its two models

    tab._model_field.setCurrentIndex(1)
    assert tab._model_info.text() == "Frontier"
    tab._save_button.click()
    assert ProviderStore.load().get("ChatGPT").model == "gpt-6-astra"  # the id is stored, not the label
    assert tab._model_field.text() == "gpt-6-astra"

    tab._account_panel.signed_in_changed.emit(True)  # known now: no second fetch
    assert len(fake_worker.instances) == 1


def test_a_failed_model_fetch_is_reported_and_can_be_cancelled(qapp, summary_stores, fake_worker):
    _store_providers(Provider(name="ChatGPT", kind=KIND_CODEX_CLI))
    tab = summary_settings_screen.ProvidersTab()

    tab._models_button.click()
    fake_worker.instances[0].failed.emit("LLMError", "Provider 'ChatGPT': codex did not answer model/list")
    assert "model/list" in tab._status.text() and tab._model_field.count() == 1

    tab._models_button.click()
    tab._models_button.click()  # "Отмена"
    assert fake_worker.instances[1].cancel_called and "отменена" in tab._status.text()


def test_api_models_are_listed_with_the_endpoint_and_the_typed_key_without_saving_anything(
    qapp, summary_stores, fake_worker
):
    tab = summary_settings_screen.ProvidersTab()
    tab._new(KIND_OPENAI_API)
    tab._key_field.setText("sk-typed")

    tab._models_button.click()

    (worker,) = fake_worker.instances
    action, provider, env = worker.args
    assert action == "models" and provider["endpoint"] == "https://api.openai.com/v1"
    assert provider["model"] == "-"  # a placeholder: listing needs no model, validation does
    assert provider["api_key_env"] == "DIARRHIZER_LLM_KEY_OPENAI_API"
    assert env == {"DIARRHIZER_LLM_KEY_OPENAI_API": "sk-typed"} and "sk-typed" not in json.dumps(provider)
    assert not summary_stores[1].exists()

    worker.finished.emit({"models": [ModelOption(id="gpt-4o-mini").to_dict()]})
    assert tab._model_field.count() == 1 and tab._model_field.itemText(0) == "gpt-4o-mini"  # no "default" for an API

    tab._endpoint_field.setText("")
    tab._models_button.click()
    assert "endpoint" in tab._status.text() and len(fake_worker.instances) == 1  # nothing to ask without one


def test_prices_are_saved_with_an_api_provider_and_come_back(qapp, summary_stores):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_OPENAI_API)
    tab._name_field.setText("gw")
    tab._model_field.setText("gpt-4o-mini")
    tab._price_in_field.setText("0,15")
    tab._price_out_field.setText("0.60")
    tab._save_button.click()

    provider = ProviderStore.load().get("gw")
    assert (provider.price_input, provider.price_output) == (0.15, 0.6)
    tab.reload(select="gw")
    assert tab._price_in_field.text() == "0.15" and tab._price_out_field.text() == "0.6"
    assert tab._form.isRowVisible(tab._prices_row)

    tab._price_in_field.setText("cheap")
    tab._save_button.click()
    assert "Цена за вход" in tab._status.text()


def test_the_check_reports_the_model_tokens_cost_and_limits(qapp, summary_stores, fake_worker):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_CODEX_CLI)
    tab._test_button.click()

    fake_worker.instances[0].finished.emit({
        "reply": "ok", "model": "gpt-6-astra",
        "usage": {"input_tokens": 13712, "output_tokens": 5, "cached_tokens": 1000, "reasoning_tokens": 0},
        "cost_usd": 0.0123, "cost_note": llm.CLAUDE_COST_NOTE, "limits": LIMITS,
    })

    text = tab._status.text()
    assert "Провайдер отвечает. Ответ: ok" in text
    assert "модель gpt-6-astra" in text
    assert "токены: вход 13 712 (из кэша 1 000), выход 5" in text
    assert "стоимость ≈ $0.0123 (оценка Claude Code по тарифам API; подписка по токенам не списывается)" in text
    assert "Лимит 7 дн.: использовано 18%" in text and "Лимит тарифа исчерпан" in text


# --- the summary dialog shows what the request used --------------------------------------------------------------------


def test_the_dialog_shows_the_model_tokens_and_limits_of_a_finished_request(
    qapp, tmp_path, configured, fake_worker, model
):
    dialog = summary_dialog.SummaryDialog(None, _job_with_protocol(tmp_path), "meeting.mp4")
    model.reply = LLMReply(
        "Итоги", model="gpt-6-astra", usage=Usage(13712, 400, 1000, 0),
        limits=Limits(windows=(LimitWindow(used_percent=18.0, window_minutes=10080),)),
    )

    dialog._create_button.click()
    fake_worker.instances[0].run_like_the_child()

    status = dialog._status.text()
    assert "модель gpt-6-astra" in status
    assert "токены: вход 13 712 (из кэша 1 000), выход 400" in status
    assert "Лимит 7 дн.: использовано 18%" in status
    assert "стоимость" not in status  # Codex reports none, and none is invented
    meta = dialog._meta_label.text()  # ... and the stored summary keeps saying so
    assert "токены: вход 13 712" in meta and "Лимит 7 дн." in meta


def test_the_dialog_shows_an_estimated_cost_with_where_it_comes_from(qapp, tmp_path, configured, fake_worker, model):
    dialog = summary_dialog.SummaryDialog(None, _job_with_protocol(tmp_path), "meeting.mp4")
    model.reply = LLMReply("Итоги", usage=Usage(240, 7), cost_usd=0.0123, cost_note=llm.CLAUDE_COST_NOTE)

    dialog._create_button.click()
    fake_worker.instances[0].run_like_the_child()

    assert "стоимость ≈ $0.0123 (оценка Claude Code по тарифам API" in dialog._status.text()


# --- the account task ----------------------------------------------------------------------------------------------------


def test_the_account_task_reports_every_action_as_plain_dicts(monkeypatch):
    from diarrhizer.adapters import llm_accounts

    account = Account(signed_in=True, email="me@example.com", plan="plus", method="ChatGPT")
    monkeypatch.setattr(llm_accounts, "account_status", lambda p: account)
    monkeypatch.setattr(llm_accounts, "logout", lambda p: Account(signed_in=False))
    monkeypatch.setattr(llm_accounts, "list_models", lambda p: [ModelOption(id="m", name="M")])
    monkeypatch.setattr(llm_accounts, "read_limits", lambda p: Limits(windows=(LimitWindow(used_percent=5.0),)))
    provider = Provider(name="ChatGPT", kind=KIND_CODEX_CLI).to_dict()

    results = {}
    for action in ("status", "logout", "models", "limits"):
        queue = _Queue()
        child_tasks.run_account_task(queue, action, provider, {})
        ((kind, result),) = queue.items
        assert kind == "finished"
        results[action] = result

    assert results["status"] == {"account": account.to_dict()}
    assert results["logout"]["account"]["signed_in"] is False
    assert results["models"] == {"models": [{"id": "m", "name": "M", "description": "", "default": False, "resolved": ""}]}
    assert results["limits"]["limits"]["windows"][0]["used_percent"] == 5.0


def test_the_account_task_passes_the_sign_in_events_to_the_parent(monkeypatch):
    from diarrhizer.adapters import llm_accounts

    def login(provider, on_event=None, should_cancel=None):
        on_event("url", "https://auth.example/authorize")
        return Account(signed_in=True)

    monkeypatch.setattr(llm_accounts, "login", login)
    queue = _Queue()

    child_tasks.run_account_task(queue, "login", Provider(name="c", kind=KIND_CODEX_CLI).to_dict(), {})

    assert queue.items == [
        ("event", "url", "https://auth.example/authorize"),
        ("finished", {"account": Account(signed_in=True).to_dict()}),
    ]


def test_the_account_task_reports_failures_and_unknown_actions_instead_of_raising(monkeypatch):
    from diarrhizer.adapters import llm_accounts

    def broken(provider):
        raise LLMError("Provider 'c': codex not found")

    monkeypatch.setattr(llm_accounts, "account_status", broken)
    provider = Provider(name="c", kind=KIND_CODEX_CLI).to_dict()

    queue = _Queue()
    child_tasks.run_account_task(queue, "status", provider, {})
    assert queue.items == [("failed", "LLMError", "Provider 'c': codex not found")]

    queue = _Queue()
    child_tasks.run_account_task(queue, "teleport", provider, {})
    assert queue.items[0][:2] == ("failed", "ValueError")


def test_the_account_task_adds_the_typed_key_to_the_environment_of_the_call(monkeypatch):
    from diarrhizer.adapters import llm_accounts

    monkeypatch.delenv("TYPED_KEY", raising=False)
    seen = {}
    monkeypatch.setattr(llm_accounts, "list_models", lambda p: seen.update(key=p.api_key()) or [])
    provider = Provider(name="gw", kind=KIND_OPENAI_API, endpoint="http://x/v1", model="-", api_key_env="TYPED_KEY")

    child_tasks.run_account_task(_Queue(), "models", provider.to_dict(), {"TYPED_KEY": "sk-typed"})

    assert seen["key"] == "sk-typed"


def test_account_actions_run_in_a_real_child_process_and_report_events_and_results(qapp, tmp_path, monkeypatch):
    vendor = make_vendor(tmp_path, monkeypatch)
    claude = Provider(name="Claude", kind=KIND_CLAUDE_CLI, command=vendor.make("claude"), timeout_seconds=30)
    worker = LlmWorker(child_tasks.run_account_task, ("login", claude.to_dict(), {}))
    events = []
    worker.event.connect(lambda name, payload: events.append((name, payload)))
    worker.start()

    name, (result,) = _wait_for_any(("finished", worker.finished), ("failed", worker.failed))

    assert name == "finished", result
    assert events == [("waiting", "")]  # an event crossed the process boundary before the result
    assert result["account"]["signed_in"] and result["account"]["plan"] == "max"

    codex = Provider(name="ChatGPT", kind=KIND_CODEX_CLI, command=vendor.make("codex"), timeout_seconds=30)
    worker = LlmWorker(child_tasks.run_account_task, ("models", codex.to_dict(), {}))
    worker.start()
    name, (result,) = _wait_for_any(("finished", worker.finished), ("failed", worker.failed))
    assert name == "finished", result
    assert [model["id"] for model in result["models"]] == ["gpt-6-astra", "gpt-6-sol"]


# --- a tool that is installed but not found -------------------------------------------------------------------


def test_the_hint_of_a_subscription_provider_says_what_to_do_when_its_tool_is_not_found(qapp, summary_stores):
    tab = summary_settings_screen.ProvidersTab()
    for kind in (KIND_CODEX_CLI, KIND_CLAUDE_CLI):
        tab._new(kind)
        hint = tab._kind_hint.text()
        assert "полный путь" in hint and "«Команда»" in hint and "docs/troubleshooting.md" in hint


def test_a_command_pasted_with_quotes_is_saved_without_them(qapp, summary_stores):
    tab = summary_settings_screen.ProvidersTab()
    _preset(tab, KIND_CODEX_CLI)
    tab._command_field.setText('"C:\\Program Files\\OpenAI\\codex.exe"')

    tab._save_button.click()

    assert ProviderStore.load().get("ChatGPT").command == "C:\\Program Files\\OpenAI\\codex.exe"
    assert tab._command_field.text() == "C:\\Program Files\\OpenAI\\codex.exe"  # and the form shows it clean
