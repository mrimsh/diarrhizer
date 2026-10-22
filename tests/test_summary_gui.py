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

pytest.importorskip("PySide6", reason="GUI extra not installed")

from PySide6.QtCore import QEventLoop, QObject, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from diarrhizer.adapters import llm  # noqa: E402
from diarrhizer.adapters.llm import LLMError, LLMReply  # noqa: E402
from diarrhizer.summary import service  # noqa: E402
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
from diarrhizer_gui.screens import result_screen, summary_dialog, summary_settings_screen  # noqa: E402

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
    assert "claude auth login" in tab._kind_hint.text()

    tab._new(KIND_OPENAI_API)
    assert not form.isRowVisible(tab._command_field)
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

    assert queue.items == [("finished", {"reply": "ok"})]
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
import os, sys, time
sys.stdin.buffer.read()
marker = os.environ.get("FAKE_MARKER")
if marker:
    with open(marker, "w") as f:
        f.write(str(os.getpid()))
    time.sleep(120)
sys.stdout.buffer.write("Ответ из дочернего процесса".encode("utf-8"))
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

    assert (name, args) == ("finished", ({"reply": "Ответ из дочернего процесса"},))


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
