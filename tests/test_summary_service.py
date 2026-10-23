"""Tests for summary.service (SUMMARY:JOB) and the `diarrhizer summarize`
command (CLI:SUMMARIZE). The model is replaced by a recorder: these tests are
about what is sent, what is stored and how failures are reported.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

from diarrhizer import cli
from diarrhizer.adapters import llm
from diarrhizer.adapters.llm import LLMError, LLMReply
from diarrhizer.summary import service
from diarrhizer.summary.provider_info import LimitWindow, Limits, Usage
from diarrhizer.summary.providers import KIND_CODEX_CLI, KIND_OPENAI_API, Provider, ProviderStore
from diarrhizer.summary.templates import PromptTemplate, TemplateStore
from diarrhizer.summary.wildcards import WILDCARDS, UnknownWildcardError, Wildcard

TRANSCRIPT = "# Transcription\n\n[00:00:01 → 00:00:05] **Иван:** Привет, начнём встречу.\n"
TEMPLATE = PromptTemplate(name="Итоги", prompt="Подведи итоги встречи.")
PROVIDER = Provider(name="ChatGPT", kind=KIND_CODEX_CLI)


@pytest.fixture
def job_dir(tmp_path):
    job = tmp_path / "out" / "meeting_20261022_101500"
    (job / "export").mkdir(parents=True)
    (job / "export" / "result.md").write_text(TRANSCRIPT, encoding="utf-8")
    return job


@pytest.fixture
def model(monkeypatch):
    """Replace the provider call. `model.reply` is what it returns (or raises); `model.prompts` records the calls."""

    class Model:
        reply: object = LLMReply("## Итоги\n- решили начать")
        prompts: list = []
        providers: list = []

    Model.prompts = []
    Model.providers = []

    def fake_complete(provider, prompt):
        Model.prompts.append(prompt)
        Model.providers.append(provider)
        if isinstance(Model.reply, Exception):
            raise Model.reply
        return Model.reply

    monkeypatch.setattr(llm, "complete", fake_complete)
    return Model


# --- building the request ------------------------------------------------------------


def test_the_prompt_is_the_template_followed_by_the_fenced_transcript():
    prompt = service.build_prompt("  Подведи итоги.  \n", "строка 1\nстрока 2")
    assert prompt == "Подведи итоги.\n\n<transcript>\nстрока 1\nстрока 2\n</transcript>\n"


def test_the_transcript_sent_is_result_md_as_it_is(job_dir, model):
    service.summarize_job(job_dir, TEMPLATE, PROVIDER)
    assert model.prompts == [service.build_prompt(TEMPLATE.prompt, TRANSCRIPT.strip())]
    assert "**Иван:**" in model.prompts[0]


def test_a_job_without_a_result_cannot_be_summarized(tmp_path, model):
    with pytest.raises(FileNotFoundError, match="result.md"):
        service.summarize_job(tmp_path, TEMPLATE, PROVIDER)
    assert model.prompts == []


def test_an_empty_result_is_refused_before_asking_the_model(job_dir, model):
    (job_dir / "export" / "result.md").write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        service.summarize_job(job_dir, TEMPLATE, PROVIDER)
    assert model.prompts == []


# --- storing the result --------------------------------------------------------------------


def test_the_reply_is_stored_verbatim_with_what_produced_it(job_dir, model):
    provider = Provider(name="gw", kind=KIND_OPENAI_API, endpoint="http://x/v1", model="gpt-x")
    model.reply = LLMReply("## Итоги\n- решили начать", truncated=True)

    result = service.summarize_job(job_dir, TEMPLATE, provider)

    md = Path(result["path"])
    assert md.parent == job_dir / "summary"
    assert md.read_text(encoding="utf-8") == "## Итоги\n- решили начать\n"
    assert md.name.endswith("_Итоги.md")
    meta = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["template"] == "Итоги"
    assert meta["prompt"] == TEMPLATE.prompt
    assert meta["provider"] == {"name": "gw", "kind": "openai-api", "model": "gpt-x"}
    assert meta["source"] == "export/result.md"
    assert meta["transcript_chars"] == len(TRANSCRIPT.strip())
    assert meta["truncated"] is True
    assert result["text"] == "## Итоги\n- решили начать" and result["truncated"] is True


def test_what_the_request_used_is_stored_with_the_summary(job_dir, model):
    limits = Limits(windows=(LimitWindow(label="", used_percent=82.0, window_minutes=300, resets_at=1793293782.0,
                                         status="allowed_warning"),), allowed=True)
    model.reply = LLMReply("Итоги", model="claude-sonnet-5", usage=Usage(240, 7, 200, 3), cost_usd=0.0123,
                           cost_note="estimated by Claude Code", limits=limits)

    result = service.summarize_job(job_dir, TEMPLATE, Provider(name="Claude", kind="claude-cli"))

    meta = json.loads(Path(result["meta_path"]).read_text(encoding="utf-8"))
    assert meta["model"] == "claude-sonnet-5"
    assert meta["usage"] == {"input_tokens": 240, "output_tokens": 7, "cached_tokens": 200, "reasoning_tokens": 3}
    assert meta["cost_usd"] == 0.0123 and meta["cost_note"] == "estimated by Claude Code"
    assert Limits.from_dict(meta["limits"]) == limits
    assert result["usage"] == meta["usage"] and result["limits"] == meta["limits"]


def test_a_provider_that_reports_nothing_leaves_those_fields_empty(job_dir, model):
    result = service.summarize_job(job_dir, TEMPLATE, PROVIDER)
    meta = json.loads(Path(result["meta_path"]).read_text(encoding="utf-8"))
    assert meta["usage"] is None and meta["cost_usd"] is None and meta["limits"] is None
    assert meta["model"] == ""  # neither the reply nor the provider names a model


def test_codex_limits_are_read_after_the_request_and_a_failure_to_read_them_is_not_fatal(
    job_dir, model, monkeypatch, real_plan_limits
):
    asked = []

    def read_limits(provider):
        asked.append(provider.name)
        return Limits(windows=(LimitWindow(used_percent=18.0, window_minutes=10080),), credits="none")

    monkeypatch.setattr(service.llm_accounts, "read_limits", read_limits)
    result = service.summarize_job(job_dir, TEMPLATE, PROVIDER)  # PROVIDER is a Codex provider
    assert asked == ["ChatGPT"] and result["limits"]["windows"][0]["used_percent"] == 18.0

    def broken(provider):
        raise LLMError("app-server did not answer")

    monkeypatch.setattr(service.llm_accounts, "read_limits", broken)
    result = service.summarize_job(job_dir, TEMPLATE, PROVIDER)
    assert result["limits"] is None and Path(result["path"]).exists()  # the summary was made regardless


def test_limits_that_came_with_the_reply_are_not_asked_for_again(job_dir, model, monkeypatch, real_plan_limits):
    monkeypatch.setattr(service.llm_accounts, "read_limits", lambda provider: pytest.fail("a second request"))
    model.reply = LLMReply("x", limits=Limits(windows=(LimitWindow(used_percent=5.0),)))
    assert service.summarize_job(job_dir, TEMPLATE, PROVIDER)["limits"]["windows"][0]["used_percent"] == 5.0
    # ... and Claude / API providers are never asked separately: that would be a second model request.
    model.reply = LLMReply("x")
    assert service.summarize_job(job_dir, TEMPLATE, Provider(name="Claude", kind="claude-cli"))["limits"] is None


def test_describe_usage_says_only_what_was_reported():
    assert service.describe_usage({}) == []
    lines = service.describe_usage({
        "usage": {"input_tokens": 13712, "output_tokens": 5, "cached_tokens": 1000},
        "cost_usd": 0.0123, "cost_note": "estimated by Claude Code",
        "limits": {"windows": [{"label": "", "used_percent": 18.0, "window_minutes": 10080, "resets_at": None,
                                "status": ""}], "credits": "none", "allowed": False, "note": ""},
    })
    assert lines == [
        "Tokens: 13,712 in (1,000 cached), 5 out",
        "Cost: about $0.0123 (estimated by Claude Code)",
        "Limit 7 d: 18% used",
        "Credits: none",
        "The plan's usage is exhausted",
    ]


def test_format_window_names_the_usual_lengths():
    assert [service.format_window(m) for m in (300, 10080, 45, 90, None, 0)] == ["5 h", "7 d", "45 min", "90 min", "", ""]


def test_summaries_never_overwrite_each_other(job_dir, model, monkeypatch):
    class FrozenNow:
        @staticmethod
        def now():
            import datetime
            return datetime.datetime(2026, 10, 22, 12, 0, 0)

    monkeypatch.setattr(service, "datetime", FrozenNow)
    model.reply = LLMReply("первый")
    first = service.summarize_job(job_dir, TEMPLATE, PROVIDER)
    model.reply = LLMReply("второй")
    second = service.summarize_job(job_dir, TEMPLATE, PROVIDER)

    assert first["path"] != second["path"]
    assert second["path"].endswith("_Итоги_2.md")
    assert len(list((job_dir / "summary").glob("*.md"))) == 2


def test_a_failing_provider_leaves_nothing_behind(job_dir, model):
    model.reply = LLMError("Provider 'ChatGPT': codex exited with code 1")
    with pytest.raises(LLMError):
        service.summarize_job(job_dir, TEMPLATE, PROVIDER)
    assert not (job_dir / "summary").exists()


def test_list_summaries_returns_newest_first_and_skips_orphans(job_dir, model):
    folder = job_dir / "summary"
    folder.mkdir()
    for stem, template in (("20261022_100000_A", "A"), ("20261022_110000_B", "B")):
        (folder / f"{stem}.md").write_text(template, encoding="utf-8")
        (folder / f"{stem}.json").write_text(
            json.dumps({"created_at": "2026-10-22T10:00:00", "template": template,
                        "provider": {"name": "p"}}), encoding="utf-8")
    (folder / "20261022_120000_orphan.json").write_text("{}", encoding="utf-8")  # no .md
    (folder / "20261022_130000_broken.md").write_text("x", encoding="utf-8")
    (folder / "20261022_130000_broken.json").write_text("{not json", encoding="utf-8")

    summaries = service.list_summaries(job_dir)

    assert [s["template"] for s in summaries] == ["B", "A"]
    assert summaries[0]["path"].endswith("20261022_110000_B.md")
    assert service.summary_title(summaries[1]) == "22.10.2026 10:00 · A · p"
    assert service.list_summaries(job_dir / "nowhere") == []


def test_summary_title_survives_missing_metadata():
    assert service.summary_title({}) == "? · ? · ?"
    assert service.summary_title({"created_at": "garbage", "template": "T"}) == "? · T · ?"


def test_check_provider_returns_a_short_reply_and_what_the_request_used(model):
    model.reply = LLMReply("ok", model="m-1", usage=Usage(input_tokens=9, output_tokens=1), cost_usd=0.5,
                           cost_note="estimate", limits=Limits(windows=(LimitWindow(used_percent=12.0),)))

    result = service.check_provider(PROVIDER)

    assert result["reply"] == "ok" and result["model"] == "m-1"
    assert result["usage"]["input_tokens"] == 9 and result["cost_usd"] == 0.5
    assert result["limits"]["windows"][0]["used_percent"] == 12.0
    assert model.prompts == [service.CHECK_PROMPT]
    model.reply = LLMReply("x" * 500)
    assert len(service.check_provider(PROVIDER)["reply"]) == 201
    # Nothing reported, nothing invented:
    assert service.check_provider(PROVIDER)["usage"] is None and service.check_provider(PROVIDER)["limits"] is None


def test_resolve_job_dir_accepts_the_folder_or_a_file_in_its_export(job_dir, tmp_path):
    assert service.resolve_job_dir(job_dir) == job_dir
    assert service.resolve_job_dir(job_dir / "export" / "result.md") == job_dir
    with pytest.raises(FileNotFoundError):
        service.resolve_job_dir(tmp_path / "missing")
    stray = tmp_path / "notes.md"
    stray.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="export"):
        service.resolve_job_dir(stray)


# --- the command ----------------------------------------------------------------------------


@pytest.fixture
def run_cli(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_project_env", lambda: None)
    monkeypatch.setattr(cli, "_configure_logging", lambda: None)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["diarrhizer", "summarize", *map(str, args)])
        code = cli.main()
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return run


@pytest.fixture
def configured(summary_stores):
    templates = TemplateStore.load()
    templates.put(PromptTemplate(name="Мой шаблон", prompt="Сделай мой разбор."))
    templates.save()
    providers = ProviderStore.load()
    providers.put(Provider(name="ChatGPT", kind=KIND_CODEX_CLI))
    providers.put(Provider(name="Claude", kind="claude-cli"))
    providers.save()


def test_summarize_uses_the_default_template_and_provider(run_cli, job_dir, model, configured):
    code, out, err = run_cli(job_dir)

    assert code == 0, err
    assert "Summary saved:" in out
    assert model.providers[0].name == "ChatGPT"  # the first provider became the default
    assert "О чём был разговор" in model.prompts[0]  # the built-in default template
    assert len(service.list_summaries(job_dir)) == 1


def test_summarize_takes_the_template_and_provider_by_name_and_can_print(run_cli, job_dir, model, configured):
    model.reply = LLMReply("Готово: итоги")
    code, out, _err = run_cli(job_dir / "export" / "result.md", "--template", "Мой шаблон",
                              "--provider", "Claude", "--print")

    assert code == 0
    assert model.providers[0].name == "Claude"
    assert model.prompts[0].startswith("Сделай мой разбор.")
    assert "Готово: итоги" in out


@pytest.mark.parametrize(
    "args, message",
    [
        (["--template", "nope"], "Unknown summary template 'nope'"),
        (["--provider", "nope"], "Unknown summary provider 'nope'. Available: ChatGPT, Claude"),
    ],
)
def test_summarize_reports_an_unknown_name_and_sends_nothing(run_cli, job_dir, model, configured, args, message):
    code, _out, err = run_cli(job_dir, *args)
    assert code == 1
    assert message in err
    assert model.prompts == []


def test_summarize_without_providers_says_how_to_add_one(run_cli, job_dir, model, summary_stores):
    code, _out, err = run_cli(job_dir)
    assert code == 1
    assert "No summary provider is configured" in err
    assert str(summary_stores[1]) in err


def test_summarize_reports_a_provider_failure_without_a_traceback(run_cli, job_dir, model, configured):
    model.reply = LLMError("Provider 'ChatGPT': HTTP 401 from http://x: bad key")
    code, out, err = run_cli(job_dir)
    assert code == 1
    assert "HTTP 401" in err and "Summary saved" not in out
    assert not (job_dir / "summary").exists()


def test_summarize_needs_a_job_unless_listing(run_cli, model):
    code, _out, err = run_cli()
    assert code == 1 and "job folder" in err


def test_summarize_reports_a_missing_job(run_cli, tmp_path, model, configured):
    code, _out, err = run_cli(tmp_path / "missing")
    assert code == 1 and "Job folder not found" in err


def test_summarize_list_shows_templates_and_providers(run_cli, configured):
    code, out, _err = run_cli("--list")
    assert code == 0
    assert "* Краткое резюме (built-in)" in out
    assert "Мой шаблон (yours)" in out
    assert "* ChatGPT (codex-cli)" in out and "  Claude (claude-cli)" in out


def test_summarize_reports_a_broken_store_file(run_cli, job_dir, model, summary_stores):
    summary_stores[1].write_text("{broken", encoding="utf-8")
    code, _out, err = run_cli(job_dir)
    assert code == 1 and "Cannot read summary providers file" in err


# --- wildcards in templates: {{transcript}}, {{note}} and the job's facts ----------------------------------
#
# The fakes are the same as above: the model is a recorder, so these tests compare what would be sent.

FENCED = f"<transcript>\n{TRANSCRIPT.strip()}\n</transcript>"


def template_with(prompt: str) -> PromptTemplate:
    return PromptTemplate(name="Свой", prompt=prompt)


@pytest.fixture
def warned():
    """The messages the summary service logged as warnings (the CLI shows its logger, a library caller does not)."""
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


@pytest.fixture
def rich_job(job_dir):
    """A job with what the pipeline records besides the transcript: run.json, segments, saved speaker names."""
    for folder in ("meta", "merged", "asr"):
        (job_dir / folder).mkdir()

    def write(relative, data):
        (job_dir / relative).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    write("meta/run.json", {
        "input_path": "D:\\Записи\\Созвон с Иваном.mp3",
        "start_time": "2026-10-23T09:00:00",
        "pipeline_config": {"language": "ru"},
    })
    write("merged/segments.json", {"segments": [
        {"start": 0.0, "end": 5.2, "speaker_id": "SPEAKER_00", "text": "Привет"},
        {"start": 5.2, "end": 3725.9, "speaker_id": "SPEAKER_01", "text": "Здравствуйте"},
        {"start": 3725.9, "end": 3730.0, "speaker_id": "SPEAKER_00", "text": "Пока"},
    ]})
    write("speakers.json", {"SPEAKER_00": "Иван", "SPEAKER_01": "  "})  # the second one was not named
    return job_dir


def test_a_template_places_the_transcript_and_the_note_where_it_wants(job_dir, model):
    template = template_with("{{note}}\n\nСтенограмма:\n{{transcript}}\nСделай итоги.")

    result = service.summarize_job(job_dir, template, PROVIDER, note="Только решения")

    assert model.prompts == [f"Только решения\n\nСтенограмма:\n{FENCED}\nСделай итоги.\n"]
    assert result["warnings"] == []


def test_a_template_without_the_transcript_wildcard_gets_it_at_the_end_as_before(job_dir, model):
    service.summarize_job(job_dir, template_with("Итоги встречи «{{job_name}}»."), PROVIDER)
    assert model.prompts == [f"Итоги встречи «meeting».\n\n{FENCED}\n"]
    assert service.build_prompt("Подведи итоги.", "x") == "Подведи итоги.\n\n<transcript>\nx\n</transcript>\n"


def test_a_note_for_a_template_without_the_note_wildcard_is_kept_and_reported(job_dir, model, warned):
    result = service.summarize_job(job_dir, template_with("Подведи итоги."), PROVIDER, note="  Важны сроки.\n")

    assert model.prompts == [f"Подведи итоги.\n\nВажны сроки.\n\n{FENCED}\n"]  # before the transcript, trimmed
    (warning,) = result["warnings"]
    assert "'Свой'" in warning and "{{note}}" in warning and "before the transcript" in warning
    assert warned == [f"[summary] Warning: {warning}"]  # the log says it too, not only the result
    assert json.loads(Path(result["meta_path"]).read_text(encoding="utf-8"))["user_fields"] == {"note": "Важны сроки."}


def test_a_note_for_a_template_that_places_the_transcript_itself_goes_to_the_end(job_dir, model):
    result = service.summarize_job(job_dir, template_with("{{transcript}}\nСделай итоги."), PROVIDER, note="Важны сроки.")

    assert model.prompts == [f"{FENCED}\nСделай итоги.\n\nВажны сроки.\n"]
    assert "at the end of the prompt" in result["warnings"][0]


@pytest.mark.parametrize("note", [None, "", "  \n"])
def test_no_note_means_no_warning_and_an_empty_value(job_dir, model, warned, note):
    plain = service.summarize_job(job_dir, template_with("Итоги."), PROVIDER, note=note)
    with_wildcard = service.summarize_job(job_dir, template_with("Итоги. [{{note}}]"), PROVIDER, note=note)

    assert plain["warnings"] == with_wildcard["warnings"] == [] and warned == []
    assert model.prompts == [f"Итоги.\n\n{FENCED}\n", f"Итоги. []\n\n{FENCED}\n"]


def test_an_unknown_wildcard_is_refused_by_name_before_anything_is_sent(job_dir, model):
    template = template_with("Итоги.\nДля {{client}} и {{ note }}.")

    with pytest.raises(UnknownWildcardError) as excinfo:
        service.summarize_job(job_dir, template, PROVIDER)

    message = str(excinfo.value)
    assert isinstance(excinfo.value, ValueError) and excinfo.value.name == "client"
    assert "Unknown wildcard {{client}} in line 2 of the template 'Свой'" in message
    for name in WILDCARDS:  # the valid ones are listed
        assert "{{" + name + "}}" in message
    assert model.prompts == [] and not (job_dir / "summary").exists()


def test_a_backslash_makes_the_double_braces_literal(job_dir, model):
    template = template_with("Пиши \\{{transcript}} как есть и \\{{ ещё \\{{note}}.")

    service.summarize_job(job_dir, template, PROVIDER, note="заметка")

    assert model.prompts[0].startswith("Пиши {{transcript}} как есть и {{ ещё {{note}}.")  # the backslash is consumed
    assert model.prompts[0].count("<transcript>") == 1  # an escaped wildcard is not a wildcard: appended once


def test_single_braces_and_things_that_are_not_names_are_left_alone(job_dir, model):
    prompt = 'Ответь JSON: {"themes": ["a"], "owner": {"name": "x"}}, шаблон {{"a": 1}} и {{ 2 слова }} и }} и {{{note}}}.'

    service.summarize_job(job_dir, template_with(prompt), PROVIDER, note="N")

    assert model.prompts[0].startswith(prompt.replace("{{{note}}}", "{N}"))  # only {{note}} inside {} was a wildcard


def test_a_wildcard_may_stand_many_times_and_values_are_not_scanned_for_wildcards(job_dir, model):
    template = template_with("{{ note }} / {{note}} / {{ job_name}} / {{job_name }}")

    service.summarize_job(job_dir, template, PROVIDER, note="{{transcript}} и \\{{x}}")

    assert model.prompts[0].startswith("{{transcript}} и \\{{x}} / {{transcript}} и \\{{x}} / meeting / meeting")
    assert model.prompts[0].count("<transcript>") == 1  # the note's "{{transcript}}" stayed text


def test_the_transcript_may_be_placed_twice(job_dir, model):
    service.summarize_job(job_dir, template_with("{{transcript}} ... {{transcript}}"), PROVIDER)
    assert model.prompts == [f"{FENCED} ... {FENCED}\n"]


def test_a_transcript_that_looks_like_a_template_is_sent_as_it_is(job_dir, model):
    (job_dir / "export" / "result.md").write_text("Он сказал {{note}} и {{ничего}}.\n", encoding="utf-8")
    service.summarize_job(job_dir, template_with("Итоги."), PROVIDER, note="N")
    assert "Он сказал {{note}} и {{ничего}}." in model.prompts[0]


def test_the_wildcards_that_describe_the_job_come_from_what_the_job_recorded(rich_job, model):
    template = template_with("{{job_name}}|{{date}}|{{duration}}|{{speakers}}|{{language}}")

    service.summarize_job(rich_job, template, PROVIDER)

    # The name from run.json's input path; the date from the folder (when the job was made); the length to the end
    # of the last phrase; a speaker that was never named keeps its id; the language from the job's settings.
    assert model.prompts[0].startswith("Созвон с Иваном|2026-10-22|01:02:10|Иван, Speaker_01|ru\n")


def test_a_job_without_records_gives_empty_values_not_errors(job_dir, model):
    template = template_with("{{job_name}}|{{date}}|{{duration}}|{{speakers}}|{{language}}")

    result = service.summarize_job(job_dir, template, PROVIDER)

    assert model.prompts[0].startswith("meeting|2026-10-22|||\n")  # the name and date are in the folder's name
    assert result["warnings"] == []


def test_the_date_falls_back_to_the_start_time_and_the_name_to_the_folder(tmp_path, model):
    job = tmp_path / "scan"
    (job / "export").mkdir(parents=True)
    (job / "export" / "result.md").write_text(TRANSCRIPT, encoding="utf-8")
    (job / "meta").mkdir()
    (job / "meta" / "run.json").write_text('{"start_time": "2026-03-04T05:06:07"}', encoding="utf-8")

    service.summarize_job(job, template_with("{{job_name}}|{{date}}"), PROVIDER)

    assert model.prompts[0].startswith("scan|2026-03-04\n")


def test_with_the_language_on_auto_the_detected_one_is_used(rich_job, model):
    (rich_job / "meta" / "run.json").write_text('{"pipeline_config": {"language": "auto"}}', encoding="utf-8")
    template = template_with("[{{language}}]")

    service.summarize_job(rich_job, template, PROVIDER)
    (rich_job / "asr" / "transcript.json").write_text('{"language": "en", "segments": []}', encoding="utf-8")
    service.summarize_job(rich_job, template, PROVIDER)

    assert [prompt.split("\n")[0] for prompt in model.prompts] == ["[]", "[en]"]  # "auto" is not a language


def test_files_of_the_wrong_shape_do_not_break_a_summary(rich_job, model):
    (rich_job / "meta" / "run.json").write_text("[1, 2]", encoding="utf-8")
    (rich_job / "merged" / "segments.json").write_text('{"segments": [1, {"end": "soon"}, null]}', encoding="utf-8")
    (rich_job / "speakers.json").write_text("{broken", encoding="utf-8")

    service.summarize_job(rich_job, template_with("{{job_name}}|{{duration}}|{{speakers}}|{{language}}"), PROVIDER)

    # Only the one phrase that is a dict counts, and without a speaker id it is Speaker_00, as everywhere else.
    assert model.prompts[0].startswith("meeting||Speaker_00|\n")


def test_the_registry_is_the_one_place_that_defines_wildcards(job_dir, model, run_cli, monkeypatch):
    monkeypatch.setitem(WILDCARDS, "mood", Wildcard("mood", "How the call felt.", lambda ctx: "тёплое"))

    service.summarize_job(job_dir, template_with("Настроение: {{mood}}."), PROVIDER)
    with pytest.raises(UnknownWildcardError, match=r"\{\{mood\}\}"):  # and it is in the list of valid ones
        service.summarize_job(job_dir, template_with("{{nope}}"), PROVIDER)
    _code, out, _err = run_cli("--list-wildcards")

    assert model.prompts[0].startswith("Настроение: тёплое.")
    assert "{{mood}}" in out and "How the call felt." in out


def test_every_registered_wildcard_has_a_description_and_is_described_in_the_docs():
    from diarrhizer.env_file import REPO_ROOT

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (REPO_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for name, wildcard in WILDCARDS.items():
        assert wildcard.name == name and wildcard.description.strip() and wildcard.description_ru.strip()  # + the GUI's line
        assert "{{" + name + "}}" in readme, f"README.md does not describe {{{{{name}}}}}"
        assert "{{" + name + "}}" in architecture, f"docs/architecture.md does not describe {{{{{name}}}}}"


# --- what is stored about the wildcards --------------------------------------------------------------------


def test_the_summary_records_the_users_fields_and_the_wildcards_the_template_names(job_dir, model):
    template = template_with("{{note}} {{job_name}} {{note}}")

    result = service.summarize_job(job_dir, template, PROVIDER, note="Фокус на сроках")

    meta = json.loads(Path(result["meta_path"]).read_text(encoding="utf-8"))
    assert meta["user_fields"] == {"note": "Фокус на сроках"}
    assert meta["wildcards_used"] == ["note", "job_name"]  # each once, in order of appearance
    assert meta["prompt"] == template.prompt  # the template as written, not the filled-in text
    assert result["user_fields"] == meta["user_fields"] and result["wildcards_used"] == meta["wildcards_used"]


def test_without_a_note_the_field_is_there_but_empty(job_dir, model):
    result = service.summarize_job(job_dir, TEMPLATE, PROVIDER)

    meta = json.loads(Path(result["meta_path"]).read_text(encoding="utf-8"))
    assert meta["user_fields"] == {"note": ""} and meta["wildcards_used"] == []  # the transcript was appended, not named


def test_a_summary_stored_before_wildcards_existed_reads_as_it_did(job_dir):
    folder = job_dir / "summary"
    folder.mkdir()
    old = {"created_at": "2026-10-22T10:00:00", "template": "Итоги", "prompt": "Подведи итоги.",
           "provider": {"name": "ChatGPT", "kind": "codex-cli", "model": ""}, "source": "export/result.md",
           "transcript_chars": 10, "duration_seconds": 3.2, "truncated": False, "model": "gpt-x",
           "usage": {"input_tokens": 120, "output_tokens": 30}, "cost_usd": None, "cost_note": None, "limits": None}
    (folder / "20261022_100000_Итоги.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    (folder / "20261022_100000_Итоги.md").write_text("Итоги\n", encoding="utf-8")

    (summary,) = service.list_summaries(job_dir)

    assert summary["template"] == "Итоги" and summary["prompt"] == "Подведи итоги." and summary["model"] == "gpt-x"
    assert summary["user_fields"] == {} and summary["wildcards_used"] == []  # nothing was recorded, nothing is invented
    assert service.summary_title(summary) == "22.10.2026 10:00 · Итоги · ChatGPT"
    assert service.describe_usage(summary) == ["Tokens: 120 in, 30 out"]


# --- the note and the wildcard list on the command line -------------------------------------------------


def test_summarize_list_wildcards_needs_no_job_and_no_stores(run_cli, model):
    code, out, err = run_cli("--list-wildcards")

    assert code == 0 and err == "" and model.prompts == []
    for name, wildcard in WILDCARDS.items():
        assert "{{" + name + "}}" in out
    assert "\\{{" in out  # how to write a literal one
    assert "{{transcript}}" in out and "{{note}}" in out and "{{language}}" in out


def test_summarize_note_goes_into_the_prompt_and_is_stored(run_cli, job_dir, model, configured):
    code, out, err = run_cli(job_dir, "--template", "Мой шаблон", "--note", "Фокус на сроках")

    assert code == 0, err
    assert "and a note (15 chars)" in out
    assert model.prompts[0].startswith("Сделай мой разбор.\n\nФокус на сроках\n\n<transcript>")
    (stored,) = service.list_summaries(job_dir)
    assert stored["user_fields"] == {"note": "Фокус на сроках"}


def test_summarize_warns_about_a_note_the_template_has_no_place_for(run_cli, job_dir, model, configured, warned):
    code, _out, err = run_cli(job_dir, "--template", "Мой шаблон", "--note", "Фокус на сроках")

    assert code == 0, err
    assert len(warned) == 1 and "has no {{note}}" in warned[0]  # the CLI prints its logger; the text is not lost either


def test_summarize_note_file_is_read_as_utf8_or_as_powershell_writes_it(run_cli, job_dir, model, configured, tmp_path):
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):  # utf-16 = `echo text > note.txt` in Windows PowerShell 5
        note = tmp_path / f"note_{encoding}.txt"
        note.write_bytes("Строка 1\nСтрока 2\n".encode(encoding))
        code, _out, err = run_cli(job_dir, "--template", "Мой шаблон", "--note-file", note)
        assert code == 0, err
        assert "Строка 1\nСтрока 2\n\n<transcript>" in model.prompts[-1]
    assert len(model.prompts) == 3


def test_summarize_reports_a_missing_or_binary_note_file_and_sends_nothing(run_cli, job_dir, model, configured, tmp_path):
    code, _out, err = run_cli(job_dir, "--note-file", tmp_path / "nope.txt")
    assert code == 1 and "Note file not found" in err

    binary = tmp_path / "note.bin"
    binary.write_bytes(b"\xff\xfe\x00\xd8\x00")  # a BOM, then half a surrogate pair
    code, _out, err = run_cli(job_dir, "--note-file", binary)
    assert code == 1 and "not UTF-8" in err
    assert model.prompts == []


def test_summarize_takes_either_note_or_note_file_not_both(run_cli, job_dir, model, configured, tmp_path, capsys):
    note = tmp_path / "note.txt"
    note.write_text("x", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        run_cli(job_dir, "--note", "a", "--note-file", note)

    assert excinfo.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
    assert model.prompts == []


def test_summarize_reports_an_unknown_wildcard_of_a_template_and_sends_nothing(run_cli, job_dir, model, configured):
    templates = TemplateStore.load()
    templates.put(PromptTemplate(name="С ошибкой", prompt="Итоги для {{client}}."))
    templates.save()

    code, _out, err = run_cli(job_dir, "--template", "С ошибкой")

    assert code == 1
    assert "Unknown wildcard {{client}} in line 1 of the template 'С ошибкой'" in err and "{{transcript}}" in err
    assert model.prompts == [] and not (job_dir / "summary").exists()


# --- finding wildcards, previewing the request, the note in a title -------------------------------------------


def test_a_job_folder_renamed_after_a_clash_still_gives_its_date_and_name(tmp_path, model):
    job = tmp_path / "out" / "meeting_20261022_101500_2"  # new_job_dir() adds _2 when the name is taken
    (job / "export").mkdir(parents=True)
    (job / "export" / "result.md").write_text(TRANSCRIPT, encoding="utf-8")

    service.summarize_job(job, template_with("{{job_name}}|{{date}}"), PROVIDER)

    assert model.prompts[0].startswith("meeting|2026-10-22\n")


def test_a_wildcard_does_not_span_lines(job_dir, model):
    service.summarize_job(job_dir, template_with("a {{\nnote\n}} b {{ note\t}}"), PROVIDER, note="N")

    assert model.prompts[0].startswith("a {{\nnote\n}} b N\n")  # the first is not a wildcard, the second is


def test_find_wildcards_gives_every_reference_with_its_place(job_dir):
    from diarrhizer.summary.wildcards import find_wildcards

    prompt = "Для {{client}}\n\\{{note}} и {{ note }}, {{transcript}}"

    refs = find_wildcards(prompt)

    assert [(r.name, r.known, r.line) for r in refs] == [("client", False, 1), ("note", True, 2), ("transcript", True, 2)]
    assert [prompt[r.start:r.end] for r in refs] == ["{{client}}", "{{ note }}", "{{transcript}}"]  # the escaped one is not
    assert find_wildcards("без wildcards {} {{}} {{1}}") == []


def test_preview_prompt_is_what_a_run_sends_and_sends_nothing(job_dir, model):
    prompt = "Итоги «{{job_name}}»: {{note}}"

    preview = service.preview_prompt(job_dir, prompt, "про сроки", "Свой")
    assert model.prompts == []  # built, not sent

    service.summarize_job(job_dir, template_with(prompt), PROVIDER, note="про сроки")
    assert model.prompts == [preview.text]
    assert preview.wildcards_used == ["job_name", "note"] and preview.warnings == []


def test_preview_prompt_reports_the_same_problems_as_a_run(job_dir, tmp_path):
    with pytest.raises(UnknownWildcardError, match=r"\{\{client\}\}"):
        service.preview_prompt(job_dir, "{{client}}")
    with pytest.raises(FileNotFoundError, match="result.md"):
        service.preview_prompt(tmp_path / "nowhere", "x")
    assert service.preview_prompt(job_dir, "Итоги.", "заметка").warnings  # a note with no place: warned


def test_a_summary_line_carries_the_start_of_its_note():
    summary = {"created_at": "2026-10-22T10:00:00", "template": "T", "provider": {"name": "p"}}
    base = "22.10.2026 10:00 · T · p"

    assert service.summary_title(summary) == base
    assert service.summary_title({**summary, "user_fields": {"note": ""}}) == base
    assert service.summary_title({**summary, "user_fields": {"note": "про\nсроки \n и людей"}}) == base + " · «про сроки и людей»"
    long = service.summary_title({**summary, "user_fields": {"note": "x" * 100}})
    assert long == base + " · «" + "x" * service.NOTE_TITLE_CHARS + "…»"
    for damaged in ("garbage", ["note"], {"note": 5}, None):  # a damaged file is not a crash
        assert service.summary_title({**summary, "user_fields": damaged}) == base
        assert service.summary_note({**summary, "user_fields": damaged}) == ""
    assert service.summary_note({**summary, "user_fields": {"note": "  текст \n"}}) == "текст"
