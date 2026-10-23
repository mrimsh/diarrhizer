"""Tests for summary.service (SUMMARY:JOB) and the `diarrhizer summarize`
command (CLI:SUMMARIZE). The model is replaced by a recorder: these tests are
about what is sent, what is stored and how failures are reported.
"""

import json
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
