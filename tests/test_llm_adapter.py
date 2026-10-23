"""Tests for adapters.llm (ADAPTER:LLM): the HTTP providers against a local
server, the subscription CLIs against stand-in executables. Nothing here
touches the network or a real model.
"""

import json
import sys
import time

import pytest

from diarrhizer.adapters import llm
from diarrhizer.adapters.llm import Block, LLMError, Message, complete, complete_chat
from diarrhizer.summary.provider_info import Usage
from diarrhizer.summary.providers import (
    KIND_ANTHROPIC_API,
    KIND_CLAUDE_CLI,
    KIND_CODEX_CLI,
    KIND_OPENAI_API,
    Provider,
)

PROMPT = "Сделай резюме.\n\n<transcript>\nПривет, мир\n</transcript>\n"


def _openai_reply(text="Резюме готово", finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}]}


def _openai(api, **overrides) -> Provider:
    fields = dict(name="gw", kind=KIND_OPENAI_API, endpoint=f"{api.url}/v1", model="gpt-x",
                  timeout_seconds=10)
    return Provider(**{**fields, **overrides})


def _anthropic(api, **overrides) -> Provider:
    fields = dict(name="ant", kind=KIND_ANTHROPIC_API, endpoint=api.url, model="claude-x",
                  api_key_env="ANT_KEY", timeout_seconds=10)
    return Provider(**{**fields, **overrides})


# --- OpenAI-compatible -----------------------------------------------------------


def test_openai_sends_the_prompt_as_one_user_message_with_a_bearer_key(api, monkeypatch):
    monkeypatch.setenv("GW_KEY", "sk-secret")
    api.body = _openai_reply()

    reply = complete(_openai(api, api_key_env="GW_KEY", temperature=0.2, max_tokens=900), PROMPT)

    assert (reply.text, reply.truncated) == ("Резюме готово", False)
    (request,) = api.requests
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer sk-secret"
    assert request["headers"]["user-agent"].startswith("diarrhizer")
    assert request["body"] == {
        "model": "gpt-x",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.2,
        "max_tokens": 900,
    }


def test_openai_omits_optional_parameters_and_the_auth_header_when_unset(api):
    api.body = _openai_reply()
    complete(_openai(api), PROMPT)
    (request,) = api.requests
    assert set(request["body"]) == {"model", "messages"}
    assert "authorization" not in request["headers"]


def test_openai_accepts_the_full_method_url_as_endpoint(api):
    api.body = _openai_reply()
    complete(_openai(api, endpoint=f"{api.url}/custom/chat/completions/"), PROMPT)
    assert api.requests[0]["path"] == "/custom/chat/completions"


def test_openai_reports_the_model_and_token_usage_and_no_cost_without_prices(api):
    api.body = {**_openai_reply(), "model": "gpt-x-2026-01-01",
                "usage": {"prompt_tokens": 1200, "completion_tokens": 300,
                          "prompt_tokens_details": {"cached_tokens": 1000},
                          "completion_tokens_details": {"reasoning_tokens": 50}}}

    reply = complete(_openai(api), PROMPT)

    assert reply.model == "gpt-x-2026-01-01"  # what the server says, not the alias that was asked for
    assert reply.usage == Usage(
        input_tokens=1200, output_tokens=300, cached_tokens=1000, reasoning_tokens=50, uncached_input_tokens=200
    )  # OpenAI counts the cached tokens inside prompt_tokens; it has no cache writes to report
    assert reply.cost_usd is None and reply.cost_note == ""  # a price nobody told us is not guessed


def test_openai_cost_comes_from_the_prices_set_on_the_provider(api):
    api.body = {**_openai_reply(), "usage": {"prompt_tokens": 2_000_000, "completion_tokens": 500_000}}

    reply = complete(_openai(api, price_input=0.15, price_output=0.60), PROMPT)

    assert reply.cost_usd == pytest.approx(2 * 0.15 + 0.5 * 0.60)
    assert "prices set for this provider" in reply.cost_note


def test_a_cost_the_provider_reports_wins_over_the_prices(api):
    """OpenRouter-style servers put the cost into usage."""
    api.body = {**_openai_reply(), "usage": {"prompt_tokens": 10, "completion_tokens": 10, "cost": 0.00042}}

    reply = complete(_openai(api, price_input=100.0, price_output=100.0), PROMPT)

    assert reply.cost_usd == pytest.approx(0.00042)
    assert reply.cost_note == "reported by the provider"


def test_anthropic_reports_usage_with_cache_tokens_counted_as_input(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    api.body = {"content": [{"type": "text", "text": "ok"}], "model": "claude-x-20260101",
                "usage": {"input_tokens": 100, "cache_read_input_tokens": 400, "cache_creation_input_tokens": 50,
                          "output_tokens": 20}}

    reply = complete(_anthropic(api, price_input=3.0, price_output=15.0), PROMPT)

    assert reply.model == "claude-x-20260101"
    assert reply.usage == Usage(
        input_tokens=550, output_tokens=20, cached_tokens=400, cache_write_tokens=50, uncached_input_tokens=100
    )
    assert reply.cost_usd == pytest.approx((550 * 3.0 + 20 * 15.0) / 1_000_000)  # no cache prices: all input at one


def test_openai_flags_a_reply_cut_off_by_the_length_limit(api):
    api.body = _openai_reply("Начало ответа", finish="length")
    assert complete(_openai(api), PROMPT).truncated is True


def test_openai_joins_content_parts(api):
    api.body = {"choices": [{"message": {"content": [{"type": "text", "text": "А"}, {"type": "text", "text": "Б"}]}}]}
    assert complete(_openai(api), PROMPT).text == "АБ"


def test_a_missing_api_key_fails_before_any_request(api):
    with pytest.raises(LLMError, match="GW_KEY"):
        complete(_openai(api, api_key_env="GW_KEY"), PROMPT)
    assert api.requests == []


def test_http_errors_carry_the_servers_message_but_never_the_key(api, monkeypatch):
    monkeypatch.setenv("GW_KEY", "sk-secret")
    api.status = 401
    api.body = {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}}

    with pytest.raises(LLMError) as excinfo:
        complete(_openai(api, api_key_env="GW_KEY"), PROMPT)

    assert "HTTP 401" in str(excinfo.value)
    assert "Incorrect API key provided" in str(excinfo.value)
    assert "sk-secret" not in str(excinfo.value)


def test_an_error_body_that_is_not_json_is_shown_shortened(api):
    api.status = 502
    api.body = b"<html>Bad gateway</html>"
    with pytest.raises(LLMError, match="HTTP 502.*Bad gateway"):
        complete(_openai(api), PROMPT)


def test_an_unreachable_server_is_an_llm_error(api):
    provider = _openai(api)
    api.close()
    with pytest.raises(LLMError, match="cannot reach"):
        complete(provider, PROMPT)


def test_a_slow_server_times_out_with_a_hint(api):
    api.body = _openai_reply()
    api.delay = 3
    with pytest.raises(LLMError, match="no reply .* within 1 s"):
        complete(_openai(api, timeout_seconds=1), PROMPT)


@pytest.mark.parametrize("body", [b"plain text", b"[1, 2]", b'{"choices": []}', b'{"unexpected": true}'])
def test_unexpected_reply_shapes_are_reported_not_raised_as_key_errors(api, body):
    api.body = body
    with pytest.raises(LLMError):
        complete(_openai(api), PROMPT)


def test_an_empty_reply_is_an_error(api):
    api.body = _openai_reply("   ")
    with pytest.raises(LLMError, match="empty reply"):
        complete(_openai(api), PROMPT)


# --- Anthropic ---------------------------------------------------------------------


def test_anthropic_sends_its_own_headers_and_a_default_token_limit(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "ant-secret")
    api.body = {"content": [{"type": "text", "text": "Часть 1. "}, {"type": "text", "text": "Часть 2."}],
                "stop_reason": "end_turn"}

    reply = complete(_anthropic(api, temperature=0), PROMPT)

    assert (reply.text, reply.truncated) == ("Часть 1. Часть 2.", False)
    (request,) = api.requests
    assert request["path"] == "/v1/messages"
    assert request["headers"]["x-api-key"] == "ant-secret"
    assert request["headers"]["anthropic-version"] == llm.ANTHROPIC_VERSION
    assert "authorization" not in request["headers"]
    assert request["body"] == {
        "model": "claude-x",
        "max_tokens": llm.ANTHROPIC_DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
    }


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/messages"])
def test_anthropic_endpoint_may_be_the_host_the_v1_base_or_the_full_url(api, monkeypatch, suffix):
    monkeypatch.setenv("ANT_KEY", "k")
    api.body = {"content": [{"type": "text", "text": "ok"}]}
    complete(_anthropic(api, endpoint=api.url + suffix), PROMPT)
    assert api.requests[0]["path"] == "/v1/messages"


def test_anthropic_reports_a_max_tokens_stop_and_uses_the_configured_limit(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    api.body = {"content": [{"type": "text", "text": "обрыв"}], "stop_reason": "max_tokens"}
    reply = complete(_anthropic(api, max_tokens=100), PROMPT)
    assert reply.truncated is True
    assert api.requests[0]["body"]["max_tokens"] == 100


def test_anthropic_error_message_is_extracted(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    api.status = 400
    api.body = {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long"}}
    with pytest.raises(LLMError, match="HTTP 400.*prompt is too long"):
        complete(_anthropic(api), PROMPT)


# --- stand-in CLIs -------------------------------------------------------------------

FAKE_TOOL = '''
import json, os, sys, time
mode = os.environ.get("FAKE_MODE", "ok")
raw = sys.stdin.buffer.read()
argv = sys.argv[1:]
record = {"argv": argv, "stdin": raw.decode("utf-8"), "cwd": os.getcwd(), "env": dict(os.environ)}
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(record, ensure_ascii=False) + "\\n")

ISOLATION = {"--safe-mode", "--ignore-user-config", "--ephemeral"}
if mode == "reject-isolation" and ISOLATION & set(argv):
    sys.stderr.write("error: unexpected argument '--ignore-user-config' found\\n")
    sys.exit(2)
if mode == "sleep":
    time.sleep(60)
if mode == "fail-cp1251":
    sys.stderr.buffer.write("Ошибка: нет доступа\\n".encode("cp1251"))
    sys.exit(3)
if mode == "fail-codex-banner":
    sys.stderr.write("OpenAI Codex v9\\n--------\\nmodel: x\\n--------\\n")
    err = '{"type":"error","status":400,"error":{"message":"The model is not supported"}}'
    sys.stderr.write("ERROR: " + err + "\\nERROR: " + err + "\\n")
    sys.exit(1)
if mode == "fail-login":
    sys.stdout.write("Not logged in - Please run /login\\n")
    sys.exit(1)

def emit(*events):
    for event in events:
        sys.stdout.buffer.write((json.dumps(event, ensure_ascii=False) + "\\n").encode("utf-8"))


if mode == "fail-login-json":  # what Claude Code prints with --output-format stream-json when nobody is signed in
    emit({"type": "system", "subtype": "init", "model": "claude-opus-5[1m]"},
         {"type": "result", "subtype": "success", "is_error": True, "result": "Not logged in \\u00b7 Please run /login",
          "total_cost_usd": 0, "usage": {"input_tokens": 0, "output_tokens": 0}})
    sys.exit(1)
if mode == "fail-codex-json":  # a model the account cannot use: events on stdout, nothing on stderr
    nested = '{"type":"error","status":400,"error":{"message":"The model is not supported for this account"}}'
    emit({"type": "thread.started"}, {"type": "error", "message": nested},
         {"type": "turn.failed", "error": {"message": nested}})
    sys.exit(1)

if "exec" in argv:  # codex: the final message goes to the --output-last-message file, events to stdout (--json)
    path = argv[argv.index("--output-last-message") + 1]
    with open(path, "w", encoding="utf-8") as out:
        out.write("  Ответ codex  \\n")
    sys.stderr.write("banner on stderr\\n")
    emit({"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"},
         {"type": "item.completed", "item": {"type": "agent_message", "text": "Ответ codex"}},
         {"type": "turn.completed", "usage": {"input_tokens": 13712, "cached_input_tokens": 1000,
                                                 "output_tokens": 5, "reasoning_output_tokens": 2}})
else:  # claude: stream-json events; the reply is in the final "result" one
    emit({"type": "system", "subtype": "init", "model": "claude-sonnet-5"},
         {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "utilization": 0.25}},
         {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed_warning", "resetsAt": 1793293782,
                                                           "utilization": 0.82, "rateLimitType": "five_hour"}},
         {"type": "result", "subtype": "success", "is_error": False, "result": "Ответ claude\\n",
          "stop_reason": "end_turn", "total_cost_usd": 0.0123,
          "usage": {"input_tokens": 10, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 30,
                    "output_tokens": 7, "output_tokens_details": {"thinking_tokens": 3}},
          "modelUsage": {"claude-sonnet-5": {"inputTokens": 10, "outputTokens": 7, "costUSD": 0.0123}}})
'''


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """(make, log): make(name) -> path of a stand-in executable; log() -> its recorded invocations."""
    script = tmp_path / "fake_tool.py"
    script.write_text(FAKE_TOOL, encoding="utf-8")
    log_path = tmp_path / "invocations.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log_path))

    def make(name: str) -> str:
        if sys.platform == "win32":
            launcher = tmp_path / f"{name}.cmd"
            launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
        else:
            launcher = tmp_path / name
            launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
            launcher.chmod(0o755)
        return str(launcher)

    def log() -> list:
        if not log_path.exists():
            return []
        return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    return make, log


def _claude(make, **overrides) -> Provider:
    return Provider(**{**dict(name="Claude", kind=KIND_CLAUDE_CLI, command=make("claude"), timeout_seconds=30),
                       **overrides})


def _codex(make, **overrides) -> Provider:
    return Provider(**{**dict(name="ChatGPT", kind=KIND_CODEX_CLI, command=make("codex"), timeout_seconds=30),
                       **overrides})


def test_claude_gets_the_prompt_on_stdin_with_tools_off_in_an_empty_directory(fake_cli, tmp_path):
    make, log = fake_cli
    reply = complete(_claude(make, model="sonnet", extra_args=("--effort", "low")), PROMPT)

    assert reply.text == "Ответ claude"
    (call,) = log()
    assert call["stdin"] == PROMPT  # Cyrillic and newlines intact, not squeezed through argv
    argv = call["argv"]
    assert argv[:2] == ["-p", "--tools"] and argv[2] == ""
    assert {"--no-session-persistence", "--safe-mode", "--verbose"} <= set(argv)
    assert argv[argv.index("--output-format") + 1] == "stream-json"  # the reply comes back with what it used
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[-2:] == ["--effort", "low"]
    assert PROMPT not in argv
    # Not run from the repo or the job folder: the tool must not load their CLAUDE.md/AGENTS.md.
    assert call["cwd"] != str(tmp_path) and "diarrhizer-llm-" in call["cwd"]


def test_claude_reports_model_tokens_estimated_cost_and_the_limit_status(fake_cli):
    make, _log = fake_cli

    reply = complete(_claude(make), PROMPT)

    assert reply.model == "claude-sonnet-5"  # the model that did the work (modelUsage), not the alias asked for
    # Cached and cache-writing tokens are input too: 10 plain + 200 read + 30 written.
    assert reply.usage == Usage(
        input_tokens=240, output_tokens=7, cached_tokens=200, reasoning_tokens=3,
        cache_write_tokens=30, uncached_input_tokens=10,
    )
    assert reply.cost_usd == pytest.approx(0.0123)
    assert "estimated" in reply.cost_note and "not billed per token" in reply.cost_note  # a subscription is not a bill
    (window,) = reply.limits.windows  # the last rate_limit_event wins
    assert window.status == "allowed_warning"
    assert window.used_percent == pytest.approx(82)  # utilization 0.82 is a fraction
    assert window.window_minutes == 300 and window.resets_at == 1793293782
    assert reply.limits.allowed is True


def test_claude_limits_above_one_are_taken_as_a_percentage_and_rejection_means_not_allowed():
    limits = llm._claude_limits({"status": "rejected", "utilization": 100, "rateLimitType": "seven_day"})
    (window,) = limits.windows
    assert window.used_percent == 100 and window.window_minutes == 10080
    assert limits.allowed is False


def test_codex_reply_comes_from_the_output_file_not_the_noisy_streams(fake_cli):
    make, log = fake_cli
    reply = complete(_codex(make, model="gpt-5"), PROMPT)

    assert reply.text == "Ответ codex"
    (call,) = log()
    argv = call["argv"]
    assert argv[:3] == ["exec", "--json", "--skip-git-repo-check"]  # --json: the events carry the token usage
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert {"--ignore-user-config", "--ephemeral"} <= set(argv)
    assert argv[argv.index("--model") + 1] == "gpt-5"
    assert argv[-1] == "-"  # prompt from stdin
    assert call["stdin"] == PROMPT
    assert reply.model == "gpt-5"
    # Codex counts cached tokens inside input_tokens; it reports no cost.
    assert reply.usage == Usage(
        input_tokens=13712, output_tokens=5, cached_tokens=1000, reasoning_tokens=2, uncached_input_tokens=12712
    )  # and no cache writes: Codex does not report them
    assert reply.cost_usd is None and reply.limits is None


def test_codex_without_a_reply_file_falls_back_to_the_final_message_event(tmp_path):
    reply = llm._codex_reply(
        Provider(name="c", kind=KIND_CODEX_CLI),
        '{"type":"item.completed","item":{"type":"agent_message","text":"first"}}\n'
        'not json at all\n'
        '{"type":"item.completed","item":{"type":"agent_message","text":"final"}}\n',
        tmp_path / "missing.txt",
    )
    assert reply.text == "final"


def test_subscription_clis_do_not_inherit_api_keys_that_would_switch_the_billing(fake_cli, monkeypatch):
    make, log = fake_cli
    for variable in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY", "KEEP_ME"):
        monkeypatch.setenv(variable, "x")

    complete(_claude(make), PROMPT)
    complete(_codex(make), PROMPT)

    claude_env, codex_env = (call["env"] for call in log())
    assert "ANTHROPIC_API_KEY" not in claude_env and "ANTHROPIC_AUTH_TOKEN" not in claude_env
    assert "OPENAI_API_KEY" not in codex_env and "CODEX_API_KEY" not in codex_env
    assert claude_env["KEEP_ME"] == codex_env["KEEP_ME"] == "x"
    assert claude_env["OPENAI_API_KEY"] == "x"  # only the variables of its own account are dropped


def test_a_version_that_rejects_the_isolation_flags_is_retried_without_them(fake_cli, monkeypatch):
    make, log = fake_cli
    monkeypatch.setenv("FAKE_MODE", "reject-isolation")

    assert complete(_codex(make), PROMPT).text == "Ответ codex"

    first, second = log()
    assert "--ignore-user-config" in first["argv"]
    assert not {"--ignore-user-config", "--ephemeral", "--safe-mode"} & set(second["argv"])


def test_other_failures_are_not_retried(fake_cli, monkeypatch):
    make, log = fake_cli
    monkeypatch.setenv("FAKE_MODE", "fail-login")
    with pytest.raises(LLMError, match="Not logged in"):
        complete(_claude(make), PROMPT)
    assert len(log()) == 1


def test_claude_not_signed_in_is_read_from_its_result_event(fake_cli, monkeypatch):
    make, log = fake_cli
    monkeypatch.setenv("FAKE_MODE", "fail-login-json")
    with pytest.raises(LLMError) as excinfo:
        complete(_claude(make), PROMPT)
    message = str(excinfo.value)
    assert "Not logged in" in message and "Please run /login" in message
    assert '"type"' not in message  # the JSON events themselves are not dumped on the user
    assert "claude auth login" in message
    assert len(log()) == 1


def test_codex_json_errors_are_reduced_to_the_providers_message(fake_cli, monkeypatch):
    make, _log = fake_cli
    monkeypatch.setenv("FAKE_MODE", "fail-codex-json")
    with pytest.raises(LLMError) as excinfo:
        complete(_codex(make), PROMPT)
    message = str(excinfo.value)
    assert message.count("The model is not supported for this account") == 1  # error + turn.failed say it once
    assert "thread.started" not in message


def test_a_failed_cli_reports_code_output_and_how_to_sign_in(fake_cli, monkeypatch):
    make, _log = fake_cli
    monkeypatch.setenv("FAKE_MODE", "fail-login")
    with pytest.raises(LLMError) as excinfo:
        complete(_claude(make), PROMPT)
    message = str(excinfo.value)
    assert "exited with code 1" in message
    assert "Not logged in" in message
    assert "claude auth login" in message


def test_output_in_the_console_code_page_is_decoded_not_replaced(fake_cli, monkeypatch):
    make, _log = fake_cli
    monkeypatch.setenv("FAKE_MODE", "fail-cp1251")
    monkeypatch.setattr(llm.locale, "getpreferredencoding", lambda _do_setlocale=True: "cp1251")
    with pytest.raises(LLMError, match="Ошибка: нет доступа"):
        complete(_codex(make), PROMPT)


def test_codex_errors_are_reduced_to_the_cause(fake_cli, monkeypatch):
    make, _log = fake_cli
    monkeypatch.setenv("FAKE_MODE", "fail-codex-banner")
    with pytest.raises(LLMError) as excinfo:
        complete(_codex(make), PROMPT)
    message = str(excinfo.value)
    assert message.count("The model is not supported") == 1  # the echoed duplicate is dropped
    assert "OpenAI Codex v9" not in message


def test_a_missing_command_is_explained(tmp_path):
    provider = Provider(name="ChatGPT", kind=KIND_CODEX_CLI, command=str(tmp_path / "no-such-codex"))
    with pytest.raises(LLMError, match="not found"):
        complete(provider, PROMPT)


def test_a_command_that_is_not_on_path_says_so_and_points_at_the_instruction():
    provider = Provider(name="ChatGPT", kind=KIND_CODEX_CLI, command="definitely-not-installed-codex")
    with pytest.raises(LLMError) as excinfo:
        llm.find_executable(provider)
    message = str(excinfo.value)
    assert "command 'definitely-not-installed-codex' not found" in message and "not on PATH" in message
    assert "full path" in message and "docs/troubleshooting.md" in message


def test_a_path_with_no_file_behind_it_is_reported_as_such(tmp_path):
    missing = tmp_path / "OpenAI" / "codex.exe"
    with pytest.raises(LLMError) as excinfo:
        llm.find_executable(Provider(name="ChatGPT", kind=KIND_CODEX_CLI, command=str(missing)))
    assert "not found - there is no such file" in str(excinfo.value)
    assert "not on PATH" not in str(excinfo.value)  # it is a path: PATH has nothing to do with it


def test_a_folder_instead_of_a_program_is_explained(tmp_path):
    with pytest.raises(LLMError) as excinfo:
        llm.find_executable(Provider(name="ChatGPT", kind=KIND_CODEX_CLI, command=str(tmp_path)))
    assert "is a folder, not a program" in str(excinfo.value) and "codex.exe or codex.cmd" in str(excinfo.value)


def test_a_command_pasted_with_the_quotes_of_copy_as_path_still_runs(fake_cli):
    make, log = fake_cli
    quoted = f'"{make("claude")}"'

    reply = complete(_claude(make, command=quoted), PROMPT)

    assert reply.text == "Ответ claude" and len(log()) == 1


def test_the_instruction_the_error_points_at_is_in_the_docs():
    """The message names a section of docs/troubleshooting.md; if the section is renamed, this fails."""
    from diarrhizer.env_file import REPO_ROOT

    with pytest.raises(LLMError) as excinfo:
        llm.find_executable(Provider(name="ChatGPT", kind=KIND_CODEX_CLI, command="definitely-not-installed-codex"))
    section = "command 'codex' not found although Codex is installed"
    assert section in str(excinfo.value)
    docs = (REPO_ROOT / "docs" / "troubleshooting.md").read_text(encoding="utf-8")
    headings = [line[3:].replace("`", "") for line in docs.splitlines() if line.startswith("## ")]  # as a reader sees them
    assert section in headings
    assert "rundll32 sysdm.cpl,EditEnvironmentVariables" in docs and "Get-Command codex -All" in docs


def test_a_cli_that_never_answers_is_killed_at_the_timeout(fake_cli, monkeypatch):
    make, _log = fake_cli
    monkeypatch.setenv("FAKE_MODE", "sleep")
    started = time.monotonic()
    with pytest.raises(LLMError, match="no reply within 1 s"):
        complete(_claude(make, timeout_seconds=1), PROMPT)
    assert time.monotonic() - started < 20  # not the tool's 60 s sleep


# --- conversations: complete_chat ------------------------------------------------------------------


TRANSCRIPT_BLOCK = "<transcript>\nПривет, мир\n</transcript>"


def _conversation() -> list:
    return [
        Message("system", "Ты — аккуратный редактор протоколов."),
        Message("user", [Block(TRANSCRIPT_BLOCK, cache=True), Block("Подведи итоги.")]),
        Message("assistant", "Итоги: начали встречу."),
        Message("user", "А кто что должен сделать?"),
    ]


EPHEMERAL = {"type": "ephemeral"}


def test_anthropic_gets_the_messages_a_system_prompt_and_a_cache_breakpoint_on_the_flagged_block(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    api.body = {"content": [{"type": "text", "text": "ok"}]}

    complete_chat(_anthropic(api, temperature=0.3), _conversation())

    (request,) = api.requests
    assert request["path"] == "/v1/messages"
    assert request["body"] == {
        "model": "claude-x",
        "max_tokens": llm.ANTHROPIC_DEFAULT_MAX_TOKENS,
        "system": "Ты — аккуратный редактор протоколов.",  # system messages are not messages for this API
        "messages": [
            {"role": "user", "content": [  # given blocks: sent as blocks; the flagged one is the breakpoint
                {"type": "text", "text": TRANSCRIPT_BLOCK, "cache_control": EPHEMERAL},
                {"type": "text", "text": "Подведи итоги."},
            ]},
            {"role": "assistant", "content": "Итоги: начали встречу."},  # given a string: sent as a string
            {"role": "user", "content": "А кто что должен сделать?"},
        ],
        "temperature": 0.3,
    }


def test_anthropic_system_messages_become_a_string_or_blocks_with_their_own_breakpoint(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    api.body = {"content": [{"type": "text", "text": "ok"}]}
    user = Message("user", "Привет")

    complete_chat(_anthropic(api), [Message("system", "Правило 1."), Message("system", "Правило 2."), user])
    complete_chat(_anthropic(api), [Message("system", [Block("Длинные правила.", cache=True)]),
                                    Message("system", "Короткое."), user])
    complete_chat(_anthropic(api), [user])

    strings, blocks, none = (request["body"] for request in api.requests)
    assert strings["system"] == "Правило 1.\n\nПравило 2."
    assert blocks["system"] == [{"type": "text", "text": "Длинные правила.", "cache_control": EPHEMERAL},
                                {"type": "text", "text": "Короткое."}]
    assert "system" not in none
    assert none["messages"] == [{"role": "user", "content": "Привет"}]  # a plain string stays a string


def test_an_openai_compatible_api_gets_plain_messages_and_nothing_about_caching(api):
    api.body = _openai_reply()

    complete_chat(_openai(api), _conversation())

    (request,) = api.requests
    assert request["body"] == {
        "model": "gpt-x",
        "messages": [
            {"role": "system", "content": "Ты — аккуратный редактор протоколов."},
            {"role": "user", "content": TRANSCRIPT_BLOCK + "Подведи итоги."},  # the blocks run together
            {"role": "assistant", "content": "Итоги: начали встречу."},
            {"role": "user", "content": "А кто что должен сделать?"},
        ],
    }
    assert "cache" not in json.dumps(request["body"])  # the server caches a repeated prefix by itself


def test_messages_may_be_given_as_dicts_as_they_cross_a_queue_or_a_file(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    api.body = {"content": [{"type": "text", "text": "ok"}]}

    complete_chat(_anthropic(api), [{"role": "user", "content": [{"text": "префикс", "cache": True}, "вопрос"]}])

    assert api.requests[0]["body"]["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "префикс", "cache_control": EPHEMERAL}, {"type": "text", "text": "вопрос"},
    ]}]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [Message("system", "только правила")],
        "строка вместо списка",
        [{"role": "user"}],
        [{"role": "user", "content": "x", "cache": True}],
        [{"role": "robot", "content": "x"}],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": []}],
        [{"role": "user", "content": [{"text": "x", "cache": "yes"}]}],
        [{"role": "user", "content": [{"text": "x", "ttl": 5}]}],
        [{"role": "user", "content": [{"text": ""}]}],
        [{"role": "user", "content": {"text": "x"}}],
        [42],
    ],
)
def test_malformed_messages_are_refused_before_anything_is_sent(api, monkeypatch, messages):
    monkeypatch.setenv("ANT_KEY", "k")
    with pytest.raises(ValueError):
        complete_chat(_anthropic(api), messages)
    assert api.requests == []


def test_a_message_is_its_blocks_and_its_text_is_them_run_together():
    plain = Message("user", "текст")
    blocks = Message("user", [Block("а"), "б", {"text": "в", "cache": True}])

    assert plain.blocks == (Block("текст"),) and plain.text == "текст"
    assert blocks.blocks == (Block("а"), Block("б"), Block("в", cache=True)) and blocks.text == "абв"
    assert Message.from_value({"role": "assistant", "content": "x"}) == Message("assistant", "x")


@pytest.mark.parametrize("kind", ["anthropic", "openai"])
def test_complete_is_one_user_message_of_complete_chat(api, monkeypatch, kind):
    monkeypatch.setenv("ANT_KEY", "k")
    provider = _anthropic(api) if kind == "anthropic" else _openai(api)
    api.body = {"content": [{"type": "text", "text": "ok"}]} if kind == "anthropic" else _openai_reply("ok")

    via_complete = complete(provider, PROMPT)
    via_chat = complete_chat(provider, [Message("user", PROMPT)])

    assert via_complete == via_chat
    first, second = (request["body"] for request in api.requests)
    assert first == second  # byte for byte the request `complete` has always sent
    assert first["messages"] == [{"role": "user", "content": PROMPT}]


def test_complete_hands_its_prompt_to_complete_chat(monkeypatch):
    seen = []
    monkeypatch.setattr(llm, "complete_chat", lambda provider, messages: seen.append((provider.name, messages)) or "reply")
    provider = Provider(name="gw", kind=KIND_OPENAI_API, endpoint="http://x/v1", model="m")

    assert complete(provider, "вопрос") == "reply"
    assert seen == [("gw", [Message("user", "вопрос")])]


def test_a_cli_gets_a_lone_user_message_verbatim_and_a_longer_conversation_written_out(fake_cli):
    make, log = fake_cli

    complete_chat(_claude(make), [Message("user", PROMPT)])
    complete_chat(_claude(make), _conversation())

    lone, history = (call["stdin"] for call in log())
    assert lone == PROMPT
    assert history == (
        "System:\nТы — аккуратный редактор протоколов.\n\n"
        f"User:\n{TRANSCRIPT_BLOCK}Подведи итоги.\n\n"  # blocks run together; no cache flag means anything here
        "Assistant:\nИтоги: начали встречу.\n\n"
        "User:\nА кто что должен сделать?"
    )  # the whole conversation, every time


# --- usage and the prompt cache ------------------------------------------------------------------------


def test_anthropic_cache_counters_are_reported_separately_and_input_is_their_sum(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    api.body = {"content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 100, "cache_read_input_tokens": 400, "cache_creation_input_tokens": 50,
                          "output_tokens": 20}}

    usage = complete_chat(_anthropic(api), _conversation()).usage

    assert (usage.cached_tokens, usage.cache_write_tokens, usage.uncached_input_tokens) == (400, 50, 100)
    assert usage.input_tokens == 550 and usage.output_tokens == 20


def test_a_reported_zero_is_a_zero_and_a_missing_counter_is_none(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    api.body = {"content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 12, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 1200,
                          "output_tokens": 3}}
    first_request = complete_chat(_anthropic(api), _conversation()).usage
    assert first_request.cached_tokens == 0 and first_request.cache_write_tokens == 1200  # the cache was filled

    api.body = {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 100, "output_tokens": 20}}
    no_cache_info = complete_chat(_anthropic(api), _conversation()).usage
    assert no_cache_info.cached_tokens is None and no_cache_info.cache_write_tokens is None  # not 0: unknown
    assert (no_cache_info.input_tokens, no_cache_info.uncached_input_tokens) == (100, 100)


def test_openai_cached_tokens_are_part_of_prompt_tokens_and_writes_are_unknown(api):
    api.body = {**_openai_reply(), "usage": {"prompt_tokens": 2000, "completion_tokens": 10,
                                             "prompt_tokens_details": {"cached_tokens": 1500}}}

    usage = complete_chat(_openai(api), _conversation()).usage

    assert (usage.input_tokens, usage.cached_tokens, usage.uncached_input_tokens) == (2000, 1500, 500)
    assert usage.cache_write_tokens is None  # OpenAI does not report writes


def test_openai_usage_without_cache_details_leaves_the_cache_fields_empty(api):
    api.body = {**_openai_reply(), "usage": {"prompt_tokens": 2000, "completion_tokens": 10}}
    usage = complete_chat(_openai(api), _conversation()).usage
    assert (usage.cached_tokens, usage.cache_write_tokens, usage.uncached_input_tokens) == (None, None, None)
    assert usage.input_tokens == 2000

    api.body = _openai_reply()  # no usage at all
    assert complete_chat(_openai(api), _conversation()).usage is None


def test_a_gateway_that_reports_cache_writes_has_them_taken_out_of_the_uncached_part(api):
    api.body = {**_openai_reply(), "usage": {"prompt_tokens": 3000, "completion_tokens": 10,
                                             "prompt_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 500}}}

    usage = complete_chat(_openai(api), _conversation()).usage

    assert (usage.cached_tokens, usage.cache_write_tokens, usage.uncached_input_tokens) == (1000, 500, 1500)


def test_usage_records_keep_their_old_shape_and_old_ones_load():
    assert Usage(input_tokens=100, output_tokens=5).to_dict() == {
        "input_tokens": 100, "output_tokens": 5, "reasoning_tokens": 0,
    }  # what was not reported is not written
    full = Usage(550, 20, cached_tokens=400, reasoning_tokens=1, cache_write_tokens=50, uncached_input_tokens=100)
    assert Usage.from_dict(full.to_dict()) == full

    old = {"input_tokens": 13712, "output_tokens": 5, "cached_tokens": 1000, "reasoning_tokens": 2}
    assert Usage.from_dict(old) == Usage(13712, 5, 1000, 2)  # no cache writes recorded: None
    assert Usage.from_dict({"input_tokens": 9, "output_tokens": 1}).cached_tokens is None
    assert Usage.from_dict({**old, "something_new": 1}) == Usage.from_dict(old)  # an unknown key is no crash


# --- the cost of a request with a cache -------------------------------------------------------------------


CACHE_PRICES = dict(price_input=3.0, price_output=15.0, price_cache_read=0.3, price_cache_write=3.75)


def _anthropic_cache_reply(api):
    api.body = {"content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 100, "cache_read_input_tokens": 400, "cache_creation_input_tokens": 50,
                          "output_tokens": 20}}


def test_cache_tokens_are_charged_at_their_own_prices_when_those_are_set(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    _anthropic_cache_reply(api)

    reply = complete_chat(_anthropic(api, **CACHE_PRICES), _conversation())

    assert reply.cost_usd == pytest.approx((100 * 3.0 + 400 * 0.3 + 50 * 3.75 + 20 * 15.0) / 1_000_000)
    assert "prices set for this provider" in reply.cost_note


def test_without_cache_prices_the_cost_is_what_it_was(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    _anthropic_cache_reply(api)

    reply = complete_chat(_anthropic(api, price_input=3.0, price_output=15.0), _conversation())

    assert reply.cost_usd == pytest.approx((550 * 3.0 + 20 * 15.0) / 1_000_000)  # every input token at the one price


def test_a_cache_token_without_a_price_of_its_own_is_charged_as_input(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "k")
    _anthropic_cache_reply(api)

    only_read = complete_chat(_anthropic(api, price_input=3.0, price_output=15.0, price_cache_read=0.3), _conversation())
    only_write = complete_chat(_anthropic(api, price_input=3.0, price_output=15.0, price_cache_write=3.75), _conversation())

    assert only_read.cost_usd == pytest.approx(((100 + 50) * 3.0 + 400 * 0.3 + 20 * 15.0) / 1_000_000)
    assert only_write.cost_usd == pytest.approx(((100 + 400) * 3.0 + 50 * 3.75 + 20 * 15.0) / 1_000_000)


def test_cache_prices_are_used_for_an_openai_compatible_api_too(api):
    api.body = {**_openai_reply(), "usage": {"prompt_tokens": 1200, "completion_tokens": 300,
                                             "prompt_tokens_details": {"cached_tokens": 1000}}}

    reply = complete_chat(_openai(api, price_input=1.0, price_output=4.0, price_cache_read=0.1), _conversation())

    assert reply.cost_usd == pytest.approx((200 * 1.0 + 1000 * 0.1 + 300 * 4.0) / 1_000_000)


def test_cache_prices_do_nothing_for_tokens_the_provider_did_not_report(api):
    api.body = {**_openai_reply(), "usage": {"prompt_tokens": 1200, "completion_tokens": 300}}

    reply = complete_chat(_openai(api, **CACHE_PRICES), _conversation())

    assert reply.cost_usd == pytest.approx((1200 * 3.0 + 300 * 15.0) / 1_000_000)  # no split known: as before


def test_a_cost_the_provider_reports_still_wins_over_cache_prices(api):
    api.body = {**_openai_reply(), "usage": {"prompt_tokens": 10, "completion_tokens": 10, "cost": 0.00042,
                                             "prompt_tokens_details": {"cached_tokens": 5}}}

    reply = complete_chat(_openai(api, **CACHE_PRICES), _conversation())

    assert reply.cost_usd == pytest.approx(0.00042) and reply.cost_note == "reported by the provider"
