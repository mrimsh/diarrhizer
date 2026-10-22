"""Tests for adapters.llm (ADAPTER:LLM): the HTTP providers against a local
server, the subscription CLIs against stand-in executables. Nothing here
touches the network or a real model.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from diarrhizer.adapters import llm
from diarrhizer.adapters.llm import LLMError, complete
from diarrhizer.summary.providers import (
    KIND_ANTHROPIC_API,
    KIND_CLAUDE_CLI,
    KIND_CODEX_CLI,
    KIND_OPENAI_API,
    Provider,
)

PROMPT = "Сделай резюме.\n\n<transcript>\nПривет, мир\n</transcript>\n"


# --- a local stand-in for an HTTP API --------------------------------------------


class FakeApi:
    """Records every request and answers each with the configured status/body."""

    def __init__(self) -> None:
        self.requests: list = []
        self.status = 200
        self.body: object = {}
        self.delay = 0.0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = self.rfile.read(length)
                outer.requests.append(
                    {"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                     "body": json.loads(payload)}
                )
                time.sleep(outer.delay)
                raw = outer.body if isinstance(outer.body, bytes) else json.dumps(outer.body).encode("utf-8")
                try:
                    self.send_response(outer.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except OSError:  # the client gave up (timeout test)
                    pass

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def api(monkeypatch):
    # A system proxy must not swallow requests to the local server.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    server = FakeApi()
    yield server
    server.close()


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

if "exec" in argv:  # codex: the final message goes to the --output-last-message file
    path = argv[argv.index("--output-last-message") + 1]
    with open(path, "w", encoding="utf-8") as out:
        out.write("  Ответ codex  \\n")
    sys.stderr.write("banner on stderr\\n")
else:  # claude: the reply is stdout
    sys.stdout.buffer.write("Ответ claude\\n".encode("utf-8"))
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
    assert {"--no-session-persistence", "--safe-mode"} <= set(argv)
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[-2:] == ["--effort", "low"]
    assert PROMPT not in argv
    # Not run from the repo or the job folder: the tool must not load their CLAUDE.md/AGENTS.md.
    assert call["cwd"] != str(tmp_path) and "diarrhizer-llm-" in call["cwd"]


def test_codex_reply_comes_from_the_output_file_not_the_noisy_streams(fake_cli):
    make, log = fake_cli
    reply = complete(_codex(make, model="gpt-5"), PROMPT)

    assert reply.text == "Ответ codex"
    (call,) = log()
    argv = call["argv"]
    assert argv[:2] == ["exec", "--skip-git-repo-check"]
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert {"--ignore-user-config", "--ephemeral"} <= set(argv)
    assert argv[argv.index("--model") + 1] == "gpt-5"
    assert argv[-1] == "-"  # prompt from stdin
    assert call["stdin"] == PROMPT


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


def test_a_cli_that_never_answers_is_killed_at_the_timeout(fake_cli, monkeypatch):
    make, _log = fake_cli
    monkeypatch.setenv("FAKE_MODE", "sleep")
    started = time.monotonic()
    with pytest.raises(LLMError, match="no reply within 1 s"):
        complete(_claude(make, timeout_seconds=1), PROMPT)
    assert time.monotonic() - started < 20  # not the tool's 60 s sleep
