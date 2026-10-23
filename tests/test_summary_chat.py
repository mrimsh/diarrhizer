"""Tests for summary.chat (SUMMARY:CHAT) and `diarrhizer chat` (CLI:CHAT). The provider is a recorder: these tests
are about what is sent on every turn, what is stored and how the dialogues are kept apart.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

from diarrhizer import cli
from diarrhizer.adapters import llm
from diarrhizer.adapters.llm import LLMError, LLMReply, Message
from diarrhizer.summary import chat, service, wildcards
from diarrhizer.summary.provider_info import Usage
from diarrhizer.summary.providers import KIND_ANTHROPIC_API, KIND_CODEX_CLI, KIND_OPENAI_API, Provider, ProviderStore
from diarrhizer.summary.templates import PromptTemplate, TemplateStore

TRANSCRIPT = "[00:00:01] **Иван:** Привет, начнём встречу.\n[00:00:09] **Мария:** Давайте.\n"
FENCED = f"<transcript>\n{TRANSCRIPT.strip()}\n</transcript>"
TEMPLATE = PromptTemplate(name="Разбор", prompt="Ты разбираешь встречи. Отвечай по существу.")
GW = Provider(name="gw", kind=KIND_OPENAI_API, endpoint="https://api.example.com/v1", model="m-1")
CODEX = Provider(name="ChatGPT", kind=KIND_CODEX_CLI)


@pytest.fixture
def job_dir(tmp_path):
    job = tmp_path / "out" / "meeting_20261022_101500"
    (job / "export").mkdir(parents=True)
    (job / "export" / "result.md").write_text(TRANSCRIPT, encoding="utf-8")
    return job


@pytest.fixture
def model(monkeypatch):
    """Replace complete_chat. `model.replies` is what it returns in turn; `model.calls` records what it was given."""

    class Model:
        replies: list = []
        calls: list = []

    Model.replies, Model.calls = [], []

    def fake(provider, messages, keep_session=False, session_id=""):
        Model.calls.append({"provider": provider, "messages": list(messages), "keep": keep_session, "session": session_id})
        reply = Model.replies.pop(0) if Model.replies else LLMReply(f"Ответ {len(Model.calls)}")
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "complete_chat", fake)
    return Model


@pytest.fixture
def warned():
    messages = []

    class Collect(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                messages.append(record.getMessage())

    handler = Collect()
    logger = logging.getLogger("diarrhizer.summary")
    logger.addHandler(handler)
    yield messages
    logger.removeHandler(handler)


def new(job_dir, template=TEMPLATE, provider=GW, **kwargs):
    return chat.create_conversation(job_dir, template, provider, **kwargs)


# --- the prefix --------------------------------------------------------------------------------------------


def test_the_prefix_is_the_instruction_then_the_transcript_marked_as_a_cache_breakpoint(job_dir):
    conversation = new(job_dir)

    assert conversation.prefix == [
        {"text": "Ты разбираешь встречи. Отвечай по существу.\n\n", "cache": False},
        {"text": FENCED, "cache": True},
    ]
    system = conversation.system_message()
    assert system.role == "system" and system.blocks == (
        llm.Block("Ты разбираешь встречи. Отвечай по существу.\n\n"), llm.Block(FENCED, cache=True)
    )


def test_a_template_may_place_the_transcript_in_the_middle(job_dir):
    template = PromptTemplate(name="Т", prompt="Встреча «{{job_name}}».\n{{transcript}}\nСделай итоги.")

    conversation = new(job_dir, template)

    assert [b["cache"] for b in conversation.prefix] == [False, True, False]  # the breakpoint is the transcript's
    assert "".join(b["text"] for b in conversation.prefix) == f"Встреча «meeting».\n{FENCED}\nСделай итоги."


def test_the_blocks_run_together_are_what_render_prompt_gives(job_dir):
    for prompt in ("Итоги.", "{{note}}\n{{transcript}}\nЕщё.", "A {{transcript}} B {{transcript}} C"):
        ctx = wildcards.PromptContext(TRANSCRIPT.strip(), "заметка", job_dir)
        blocks = wildcards.render_prompt_blocks(prompt, ctx, "Т")
        rendered = wildcards.render_prompt(prompt, ctx, "Т")
        assert "".join(b.text for b in blocks.blocks) == rendered.text.rstrip("\n")
        assert blocks.warnings == rendered.warnings and blocks.wildcards_used == rendered.wildcards_used


def test_the_prefix_is_byte_for_byte_the_same_on_every_turn(job_dir, model):
    conversation = new(job_dir, note="только сроки")
    prefix = conversation.system_message()

    chat.ask(job_dir, conversation.id, "Кто был на встрече?", GW)
    # Whatever changes meanwhile - the template, the transcript - the dialogue keeps the context it began with.
    (job_dir / "export" / "result.md").write_text("Совсем другой текст.\n", encoding="utf-8")
    chat.ask(job_dir, conversation.id, "А что решили?", GW)
    chat.ask(job_dir, conversation.id, "Кто отвечает?", GW)

    sent = [call["messages"] for call in model.calls]
    assert [messages[0] for messages in sent] == [prefix, prefix, prefix]
    assert json.dumps([m.blocks for m in (s[0] for s in sent)], default=repr) == json.dumps([prefix.blocks] * 3, default=repr)
    assert [len(messages) for messages in sent] == [2, 4, 6]  # prefix + question, then the turns so far each time
    assert [m.text for m in sent[2][1:]] == ["Кто был на встрече?", "Ответ 1", "А что решили?", "Ответ 2", "Кто отвечает?"]
    assert [m.role for m in sent[2]] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_the_system_part_of_an_anthropic_request_is_identical_each_turn_with_one_breakpoint(api, monkeypatch, job_dir):
    monkeypatch.setenv("ANT_KEY", "k")
    provider = Provider(name="ant", kind=KIND_ANTHROPIC_API, endpoint=api.url, model="claude-x", api_key_env="ANT_KEY",
                        timeout_seconds=10)
    api.body = {"content": [{"type": "text", "text": "ответ"}],
                "usage": {"input_tokens": 20, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 900,
                          "output_tokens": 5}}
    conversation = new(job_dir, provider=provider)

    chat.ask(job_dir, conversation.id, "Первый?", provider)
    api.body["usage"].update(cache_read_input_tokens=900, cache_creation_input_tokens=0)
    turn = chat.ask(job_dir, conversation.id, "Второй?", provider)

    first, second = (request["body"] for request in api.requests)
    assert first["system"] == second["system"]  # byte for byte
    assert [b.get("cache_control") for b in first["system"]] == [None, {"type": "ephemeral"}]
    assert second["messages"][:2] == [{"role": "user", "content": "Первый?"}, {"role": "assistant", "content": "ответ"}]
    assert turn.usage["cached_tokens"] == 900  # the second turn read the first one's cache


# --- storing -----------------------------------------------------------------------------------------------


def test_a_dialogue_is_stored_beside_the_summaries_and_read_back_with_every_turn(job_dir, model):
    model.replies = [
        LLMReply("Пришли двое.", model="m-1-2026", usage=Usage(1000, 40, cached_tokens=0, cache_write_tokens=900,
                                                               uncached_input_tokens=100),
                 cost_usd=0.01, cost_note="estimate", context_mode=llm.CONTEXT_FULL_HISTORY, context_chars=950),
        LLMReply("Решили начать.", truncated=True, usage=Usage(1200, 30, cached_tokens=900, cache_write_tokens=0,
                                                                uncached_input_tokens=300)),
    ]
    conversation = new(job_dir, note="  про людей ")

    chat.ask(job_dir, conversation.id, "Кто пришёл?", GW)
    chat.ask(job_dir, conversation.id, "Что решили?", GW)

    path = job_dir / "chat" / f"{conversation.id}.json"
    assert path.exists() and not (job_dir / "summary").exists()
    stored = chat.load_conversation(job_dir, conversation.id)
    assert (stored.template, stored.note, stored.title) == ("Разбор", "про людей", "Кто пришёл?")
    assert stored.provider == {"name": "gw", "kind": "openai-api", "model": "m-1"}
    first, second = stored.turns
    assert (first.question, first.answer, first.model) == ("Кто пришёл?", "Пришли двое.", "m-1-2026")
    assert first.usage["cache_write_tokens"] == 900 and first.cost_usd == 0.01
    assert first.context_mode == llm.CONTEXT_FULL_HISTORY and first.context_chars == 950
    assert second.usage["cached_tokens"] == 900 and second.truncated is True
    assert stored.prefix == conversation.prefix and first.asked_at
    assert chat.load_conversation(job_dir, conversation.id).to_dict() == stored.to_dict()  # round trip
    assert not list((job_dir / "chat").glob("*.tmp"))  # written atomically


def test_a_failed_request_adds_no_turn(job_dir, model):
    conversation = new(job_dir)
    model.replies = [LLMError("Provider 'gw': HTTP 401")]

    with pytest.raises(LLMError, match="401"):
        chat.ask(job_dir, conversation.id, "Вопрос?", GW)

    assert chat.load_conversation(job_dir, conversation.id).turns == []


def test_a_job_without_a_result_cannot_start_a_dialogue(tmp_path):
    with pytest.raises(FileNotFoundError, match="result.md"):
        chat.create_conversation(tmp_path, TEMPLATE, GW)
    assert not (tmp_path / "chat").exists()


def test_the_first_question_may_be_empty_and_runs_the_templates_instruction(job_dir, model):
    conversation = new(job_dir)

    chat.ask(job_dir, conversation.id, "  ", GW)

    assert model.calls[0]["messages"][-1].text == chat.FIRST_QUESTION
    with pytest.raises(ValueError, match="empty"):
        chat.ask(job_dir, conversation.id, "", GW)  # later turns need one
    assert len(chat.load_conversation(job_dir, conversation.id).turns) == 1


def test_bad_and_unknown_ids(job_dir):
    for bad in ("../x", "a/b", "", "a b"):
        with pytest.raises(ValueError):
            chat.load_conversation(job_dir, bad)
    with pytest.raises(FileNotFoundError, match="Dialogue not found"):
        chat.load_conversation(job_dir, "nope")
    (job_dir / "chat").mkdir()
    (job_dir / "chat" / "broken.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="not a readable dialogue"):
        chat.load_conversation(job_dir, "broken")


# --- several dialogues -------------------------------------------------------------------------------------


def test_a_job_keeps_independent_dialogues(job_dir, model):
    first = new(job_dir, title="Про людей")
    second = new(job_dir, PromptTemplate(name="Другой", prompt="Только факты."), note="N")
    chat.ask(job_dir, first.id, "Кто?", GW)
    chat.ask(job_dir, first.id, "И ещё?", GW)
    chat.ask(job_dir, second.id, "Что?", GW)

    assert first.id != second.id
    listed = chat.list_conversations(job_dir)
    assert [c.id for c in listed] == [second.id, first.id]  # the one used last comes first
    by_id = {c.id: c for c in listed}
    assert [t.question for t in by_id[first.id].turns] == ["Кто?", "И ещё?"] and by_id[first.id].title == "Про людей"
    assert [t.question for t in by_id[second.id].turns] == ["Что?"]
    assert by_id[second.id].prefix != by_id[first.id].prefix  # each has its own context
    assert [m.text for m in model.calls[2]["messages"][1:]] == ["Что?"]  # the other dialogue's turns are not in it


def test_a_dialogue_can_be_deleted_without_touching_the_others(job_dir, model):
    keep, drop = new(job_dir), new(job_dir)
    chat.export_conversation(job_dir, drop.id)

    chat.delete_conversation(job_dir, drop.id)

    assert [c.id for c in chat.list_conversations(job_dir)] == [keep.id]
    assert not (job_dir / "chat" / f"{drop.id}.md").exists()
    with pytest.raises(FileNotFoundError):
        chat.delete_conversation(job_dir, drop.id)
    assert chat.list_conversations(job_dir / "nowhere") == []


# --- the provider may change --------------------------------------------------------------------------------


def test_the_provider_and_model_may_change_in_the_middle_and_each_turn_records_its_own(job_dir, model):
    other = Provider(name="gw2", kind=KIND_OPENAI_API, endpoint="https://b.example.com/v1", model="m-2")
    model.replies = [LLMReply("А"), LLMReply("Б", model="m-2-0101"), LLMReply("В")]
    conversation = new(job_dir)

    chat.ask(job_dir, conversation.id, "1?", GW)
    chat.ask(job_dir, conversation.id, "2?", other)
    chat.ask(job_dir, conversation.id, "3?", Provider(name="gw", kind=KIND_OPENAI_API, endpoint=GW.endpoint, model="m-9"))

    stored = chat.load_conversation(job_dir, conversation.id)
    assert [t.provider["name"] for t in stored.turns] == ["gw", "gw2", "gw"]
    assert [t.provider["model"] for t in stored.turns] == ["m-1", "m-2", "m-9"]
    assert [t.model for t in stored.turns] == ["m-1", "m-2-0101", "m-9"]  # what answered, else what was asked for
    assert stored.provider["model"] == "m-9"  # the dialogue's current provider is the last one used
    assert [call["provider"].name for call in model.calls] == ["gw", "gw2", "gw"]
    assert model.calls[2]["messages"][2].text == "А"  # the history travels with the dialogue, not the provider


def test_a_codex_session_is_kept_and_resumed_only_while_it_has_seen_every_turn(job_dir, model):
    S = llm.CONTEXT_SESSION
    model.replies = [
        LLMReply("a1", session_id="S1", context_mode=llm.CONTEXT_FULL_HISTORY),
        LLMReply("a2", session_id="S1", context_mode=S),
        LLMReply("a3"),  # another provider answers: the session misses this turn
        LLMReply("a4", session_id="S2", context_mode=llm.CONTEXT_FULL_HISTORY),  # so the history is sent in a new one
        LLMReply("a5", session_id="S2", context_mode=S),
    ]
    conversation = new(job_dir, provider=CODEX)

    for provider in (CODEX, CODEX, GW, CODEX, CODEX):
        chat.ask(job_dir, conversation.id, "?", provider)

    assert [(c["keep"], c["session"]) for c in model.calls] == [
        (True, ""), (True, "S1"), (False, ""), (True, ""), (True, "S2"),
    ]
    stored = chat.load_conversation(job_dir, conversation.id)
    assert (stored.session_id, stored.session_kind, stored.session_turns) == ("S2", KIND_CODEX_CLI, 5)


def test_a_cli_session_can_be_switched_off(job_dir, model):
    conversation = new(job_dir, provider=CODEX)
    chat.ask(job_dir, conversation.id, "?", CODEX, keep_cli_session=False)
    assert (model.calls[0]["keep"], model.calls[0]["session"]) == (False, "")


# --- the note ahead of the transcript -------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt, note, cache_warning",
    [
        ("Пояснение: {{note}}\n{{transcript}}", "про сроки", True),  # allowed, but said
        ("{{transcript}}\nПояснение: {{note}}", "про сроки", False),  # after the transcript: no problem
        ("Пояснение: {{note}}\n{{transcript}}", "", False),  # nothing in the note, nothing to differ
        ("Итоги.", "про сроки", True),  # no place of its own: it is added before the appended transcript
        ("Итоги. {{transcript}}", "про сроки", False),  # added at the end, after the transcript
        ("Итоги.", "", False),
    ],
)
def test_a_note_before_the_transcript_is_allowed_with_a_warning_about_the_cache(job_dir, warned, prompt, note, cache_warning):
    conversation = new(job_dir, PromptTemplate(name="Т", prompt=prompt), note=note)

    said = [w for w in conversation.warnings if "prompt cache works only within this dialogue" in w]
    assert bool(said) is cache_warning
    assert (len(warned) >= 1) is bool(cache_warning or (note and "{{note}}" not in prompt))
    assert chat.load_conversation(job_dir, conversation.id).warnings == conversation.warnings  # stored with it


def test_the_note_is_fixed_when_the_dialogue_is_created(job_dir, model):
    conversation = new(job_dir, PromptTemplate(name="Т", prompt="Ввод: {{note}}\n{{transcript}}"), note="первая")

    chat.ask(job_dir, conversation.id, "?", GW)

    assert "Ввод: первая" in model.calls[0]["messages"][0].text
    assert chat.load_conversation(job_dir, conversation.id).note == "первая"


# --- export --------------------------------------------------------------------------------------------------


def test_a_dialogue_exports_as_markdown_with_what_each_turn_used(job_dir, model):
    model.replies = [
        LLMReply("Пришли двое.", model="m-1", usage=Usage(1000, 40, cached_tokens=600, cache_write_tokens=100,
                                                         uncached_input_tokens=300),
                 cost_usd=0.0123, cost_note="estimate", context_mode=llm.CONTEXT_FULL_HISTORY, context_chars=950),
    ]
    conversation = new(job_dir, title="Встреча", note="Про людей\nи сроки")
    chat.ask(job_dir, conversation.id, "Кто пришёл?", GW)

    text = chat.export_markdown(chat.load_conversation(job_dir, conversation.id))

    assert text.startswith("# Встреча\n")
    assert "- Template: Разбор" in text and "- Provider: gw (m-1)" in text
    assert "> Про людей\n> и сроки" in text
    assert "## 1. Кто пришёл?" in text and "Пришли двое." in text
    assert "Tokens: 1,000 in (600 cached, 100 written to cache), 40 out" in text
    assert "Cost: about $0.0123" in text and "Context: full-history, 950 chars" in text
    path = chat.export_conversation(job_dir, conversation.id)
    assert path == job_dir / "chat" / f"{conversation.id}.md" and path.read_text(encoding="utf-8") == text
    other = chat.export_conversation(job_dir, conversation.id, job_dir / "out.md")
    assert other.read_text(encoding="utf-8") == text


# --- the command ---------------------------------------------------------------------------------------------


@pytest.fixture
def run_cli(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_project_env", lambda: None)
    monkeypatch.setattr(cli, "_configure_logging", lambda: None)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["diarrhizer", "chat", *map(str, args)])
        code = cli.main()
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return run


@pytest.fixture
def configured(summary_stores):
    templates = TemplateStore.load()
    templates.put(TEMPLATE)
    templates.set_default("Разбор")
    templates.save()
    providers = ProviderStore.load()
    providers.put(GW)
    providers.put(Provider(name="gw2", kind=KIND_OPENAI_API, endpoint="https://b.example.com/v1", model="m-2"))
    providers.save()


def test_chat_starts_a_dialogue_asks_prints_and_shows_what_the_turn_used(run_cli, job_dir, model, configured):
    model.replies = [LLMReply("Пришли двое.", usage=Usage(1000, 40, cached_tokens=600, uncached_input_tokens=400),
                              cost_usd=0.5, cost_note="estimate", context_mode=llm.CONTEXT_FULL_HISTORY,
                              context_chars=900)]

    code, out, err = run_cli(job_dir, "Кто пришёл?", "--note", "про людей", "--print")

    assert code == 0, err
    (conversation,) = chat.list_conversations(job_dir)
    assert f"New dialogue {conversation.id}" in out and "turn 1" in out
    assert "Tokens: 1,000 in (600 cached), 40 out" in out and "Cost: about $0.5000" in out
    assert "Context: full-history, 900 chars" in out and "Пришли двое." in out
    assert conversation.note == "про людей" and model.calls[0]["provider"].name == "gw"


def test_chat_continues_the_last_dialogue_or_a_named_one_with_another_provider_and_model(run_cli, job_dir, model, configured):
    run_cli(job_dir, "Первый?")
    (first,) = chat.list_conversations(job_dir)

    code, out, err = run_cli(job_dir, "Второй?", "--last", "--provider", "gw2", "--model", "m-9")
    assert code == 0 and "turn 2" in out, err
    code, _out, err = run_cli(job_dir, "Третий?", "--conversation", first.id)  # --model was for the one turn
    assert code == 0, err

    stored = chat.load_conversation(job_dir, first.id)
    assert [t.question for t in stored.turns] == ["Первый?", "Второй?", "Третий?"]
    assert [(t.provider["name"], t.provider["model"]) for t in stored.turns] == [("gw", "m-1"), ("gw2", "m-9"), ("gw2", "m-2")]
    assert len(chat.list_conversations(job_dir)) == 1  # continued, not restarted


def test_chat_without_a_question_on_a_new_dialogue_runs_the_templates_instruction(run_cli, job_dir, model, configured):
    code, _out, err = run_cli(job_dir)
    assert code == 0, err
    assert model.calls[0]["messages"][-1].text == chat.FIRST_QUESTION


def test_chat_lists_exports_and_deletes_dialogues(run_cli, job_dir, model, configured, tmp_path):
    run_cli(job_dir, "Первый?", "--title", "Мой разговор")
    (conversation,) = chat.list_conversations(job_dir)

    _code, out, _err = run_cli(job_dir, "--list")
    assert conversation.id in out and "1 turn(s)" in out and "Мой разговор" in out

    target = tmp_path / "dialogue.md"
    code, out, _err = run_cli(job_dir, "--last", "--export", target)
    assert code == 0 and target.read_text(encoding="utf-8").startswith("# Мой разговор")
    code, out, _err = run_cli(job_dir, "Второй?", "--last", "--export", target)  # a question, then the export
    assert "## 2. Второй?" in target.read_text(encoding="utf-8")

    code, out, _err = run_cli(job_dir, "--delete", conversation.id)
    assert code == 0 and chat.list_conversations(job_dir) == []
    _code, out, _err = run_cli(job_dir, "--list")
    assert "No dialogues" in out


def test_chat_reports_mistakes_without_a_traceback(run_cli, job_dir, model, configured):
    run_cli(job_dir, "Первый?")
    cases = [
        (["Вопрос?", "--conversation", "nope"], "Dialogue not found"),
        (["--last"], "Pass the question"),
        (["--export", "-"], "--export needs"),
        (["Вопрос?", "--provider", "nope"], "Unknown summary provider 'nope'"),
        (["Вопрос?", "--template", "nope"], "Unknown summary template 'nope'"),
    ]
    before = len(model.calls)
    for args, message in cases:
        code, _out, err = run_cli(job_dir, *args)
        assert code == 1 and message in err, (args, err)
    assert len(model.calls) == before  # nothing was sent


def test_chat_an_unknown_wildcard_stops_before_anything_is_sent_or_stored(run_cli, job_dir, model, summary_stores, configured):
    templates = TemplateStore.load()
    templates.put(PromptTemplate(name="Ошибка", prompt="Для {{client}}."))
    templates.save()

    code, _out, err = run_cli(job_dir, "Вопрос?", "--template", "Ошибка")

    assert code == 1 and "Unknown wildcard {{client}}" in err
    assert model.calls == [] and chat.list_conversations(job_dir) == []


def test_chat_shows_a_provider_failure_and_keeps_the_dialogue(run_cli, job_dir, model, configured):
    model.replies = [LLMError("Provider 'gw': HTTP 401 from http://x: bad key")]
    code, _out, err = run_cli(job_dir, "Вопрос?")
    assert code == 1 and "HTTP 401" in err
    (conversation,) = chat.list_conversations(job_dir)  # the dialogue exists, with no turn: ask again to continue
    assert conversation.turns == []
