"""Adapter for the language models that write summaries.

Subscription providers are driven through the vendors' own command-line tools
(`codex exec`, `claude -p`) in non-interactive mode, with the prompt on stdin
so it never meets a command-line length limit or Windows quoting rules. API
providers are plain HTTPS JSON calls made with urllib - no SDK dependency.
"""

import json
import locale
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from diarrhizer.summary.providers import (
    KIND_ANTHROPIC_API,
    KIND_CLAUDE_CLI,
    KIND_CODEX_CLI,
    KIND_OPENAI_API,
    Provider,
)

logger = logging.getLogger("diarrhizer.summary")

USER_AGENT = "diarrhizer/0.1"
ANTHROPIC_VERSION = "2023-06-01"
# The Messages API insists on a limit; this is only used when the provider sets none.
ANTHROPIC_DEFAULT_MAX_TOKENS = 8192

# How much of a failed tool's output goes into the error message.
_ERROR_TAIL_CHARS = 1500

# Variables that make a subscription CLI bill an API account instead. They are dropped from the
# child's environment: a provider of kind *-cli means "the account the user is signed in with",
# and an API key kept in .env for an API provider must not silently switch that.
_API_KEY_VARIABLES = {
    KIND_CLAUDE_CLI: ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    KIND_CODEX_CLI: ("OPENAI_API_KEY", "CODEX_API_KEY"),
}

# Flags that keep the tool away from the user's own setup - hooks that fire on every run (Codex'
# `notify`), MCP servers and plugins to start, a default model or reasoning effort picked for coding
# work (or one the account cannot use with `exec` at all). Sign-in is unaffected. A version that does
# not know a flag is retried without them.
_ISOLATION_FLAGS = {
    KIND_CLAUDE_CLI: ("--safe-mode",),
    KIND_CODEX_CLI: ("--ignore-user-config", "--ephemeral"),
}
_UNKNOWN_FLAG_RE = re.compile(r"unknown option|unexpected argument|unrecognized (option|argument)", re.I)

_LOGIN_HINTS = {
    KIND_CLAUDE_CLI: "If it is a sign-in problem, run `claude auth login` in a terminal.",
    KIND_CODEX_CLI: "If it is a sign-in problem, run `codex login` in a terminal.",
}


class LLMError(RuntimeError):
    """A provider could not produce a reply; the message says why and what to try."""


@dataclass(frozen=True)
class LLMReply:
    """What a provider answered.

    Attributes:
        text: The reply
        truncated: The provider cut the reply off at its length limit
    """

    text: str
    truncated: bool = False


# [SEMANTIC-BEGIN] ADAPTER:LLM
# @purpose: Send one prompt to a configured provider and return the model's reply
# @description: codex-cli runs `codex exec --sandbox read-only` and claude-cli runs `claude -p --tools ""`,
#   both with the prompt on stdin, in an empty temporary working directory (so neither tool wanders into or
#   loads the context of the repo), without the API-key variables that would bypass the subscription and with
#   the _ISOLATION_FLAGS that skip the user's own tool setup (retried without them if the installed version
#   rejects a flag). Tool output is decoded as UTF-8, else in the console code page (Codex prints cp1251 on
#   Russian Windows). A timeout kills the whole process tree - the Windows .cmd shims start the real tool as a
#   grandchild. openai-api and anthropic-api are single non-streaming HTTPS POSTs: the endpoint may be a base
#   URL or the full method URL, the key is read from the provider's api_key_env at call time and never appears
#   in an error message. Errors are always LLMError with an actionable text.
# @inputs: provider (CONFIG:SUMMARY_PROVIDERS), prompt text (already containing the transcript)
# @outputs: LLMReply(text, truncated)
# @sideEffects: Starts the vendor CLI (which talks to its service) or sends the prompt over the network;
#   creates and removes a temporary directory
# @errors: LLMError (tool missing, not signed in, HTTP/network failure, timeout, empty or malformed reply)
# @see: CONFIG:SUMMARY_PROVIDERS, SUMMARY:JOB, GUI:CHILD_TASKS
def complete(provider: Provider, prompt: str) -> LLMReply:
    if provider.is_cli:
        reply = _complete_cli(provider, prompt)
    elif provider.kind == KIND_OPENAI_API:
        reply = _complete_openai(provider, prompt)
    elif provider.kind == KIND_ANTHROPIC_API:
        reply = _complete_anthropic(provider, prompt)
    else:  # Provider validates its kind; this guards a kind added without a backend.
        raise LLMError(f"Provider '{provider.name}': unsupported kind {provider.kind!r}")
    if not reply.text.strip():
        raise LLMError(f"Provider '{provider.name}' returned an empty reply")
    return LLMReply(text=reply.text.strip(), truncated=reply.truncated)
# [SEMANTIC-END] ADAPTER:LLM


# --- subscription CLIs ---------------------------------------------------------


def _complete_cli(provider: Provider, prompt: str) -> LLMReply:
    executable = _find_executable(provider)
    with tempfile.TemporaryDirectory(prefix="diarrhizer-llm-", ignore_cleanup_errors=True) as workdir:
        reply_file = Path(workdir) / "reply.txt"
        for isolated in (True, False):
            argv = _cli_argv(provider, executable, reply_file, isolated)
            code, stdout, stderr = _run_cli(provider, argv, prompt, workdir)
            if code == 0:
                break
            if not (isolated and _UNKNOWN_FLAG_RE.search(stdout + stderr)):
                raise _cli_failure(provider, code, stdout, stderr)
            logger.warning("[summary] %s rejected an option, retrying without them", executable)
        if provider.kind == KIND_CODEX_CLI and reply_file.exists():
            return LLMReply(text=_decode(reply_file.read_bytes()))
    return LLMReply(text=stdout)


def _cli_argv(provider: Provider, executable: str, reply_file: Path, isolated: bool) -> list[str]:
    isolation = list(_ISOLATION_FLAGS[provider.kind]) if isolated else []
    if provider.kind == KIND_CLAUDE_CLI:
        argv = [executable, "-p", "--tools", "", "--no-session-persistence", "--output-format", "text"]
    else:
        argv = [executable, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never",
                "--output-last-message", str(reply_file)]
    argv += isolation
    if provider.model:
        argv += ["--model", provider.model]
    argv += list(provider.extra_args)
    if provider.kind == KIND_CODEX_CLI:
        argv.append("-")  # the prompt comes from stdin
    return argv


def _find_executable(provider: Provider) -> str:
    command = provider.executable
    found = shutil.which(command)
    if not found:
        raise LLMError(
            f"Provider '{provider.name}': command '{command}' not found. Install it and sign in, "
            "or give the full path to the executable in the provider's settings."
        )
    return found


def _run_cli(provider: Provider, argv: list[str], prompt: str, workdir: str) -> tuple[int, str, str]:
    """Run the tool with `prompt` on stdin. Returns (exit code, stdout, stderr)."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _API_KEY_VARIABLES.get(provider.kind, ())
    }
    logger.info("[summary] Running %s (timeout %ds)", Path(argv[0]).name, provider.timeout_seconds)
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as e:
        raise LLMError(f"Provider '{provider.name}': cannot start {argv[0]}: {e}") from e
    try:
        stdout, stderr = process.communicate(prompt.encode("utf-8"), timeout=provider.timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        try:
            process.communicate(timeout=5)  # reap it; never wait on a survivor holding the pipes
        except subprocess.TimeoutExpired:
            pass
        raise LLMError(
            f"Provider '{provider.name}': no reply within {provider.timeout_seconds} s "
            "(raise the timeout in the provider's settings for long transcripts)"
        ) from None
    return process.returncode, _decode(stdout), _decode(stderr)


def _decode(data: bytes) -> str:
    """UTF-8, else the console code page (a tool that prints in cp1251 is not valid UTF-8)."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False), "replace")


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill the tool and what it started (npm's .cmd shim runs the real tool as a child)."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    process.kill()


def _cli_failure(provider: Provider, code: int, stdout: str, stderr: str) -> LLMError:
    # Claude Code prints "Not logged in" on stdout, Codex its errors on stderr: look at both.
    output = "\n".join(part.strip() for part in (stderr, stdout) if part.strip())
    # Codex wraps the cause in a banner and echoes it as "ERROR: {json}" lines at the end.
    errors = list(dict.fromkeys(
        _error_detail(line[len("ERROR:"):].strip().encode("utf-8"))
        for line in output.splitlines()
        if line.startswith("ERROR:")
    ))
    if errors:
        output = "\n".join(errors)
    elif len(output) > _ERROR_TAIL_CHARS:
        output = "…" + output[-_ERROR_TAIL_CHARS:]
    message = f"Provider '{provider.name}': {provider.executable} exited with code {code}"
    if output:
        message += f":\n{output}"
    hint = _LOGIN_HINTS.get(provider.kind)
    return LLMError(f"{message}\n{hint}" if hint else message)


# --- HTTP APIs -------------------------------------------------------------------


def _complete_openai(provider: Provider, prompt: str) -> LLMReply:
    body: dict = {"model": provider.model, "messages": [{"role": "user", "content": prompt}]}
    if provider.temperature is not None:
        body["temperature"] = provider.temperature
    if provider.max_tokens is not None:
        body["max_tokens"] = provider.max_tokens
    headers = {}
    key = _require_key(provider)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = _post_json(provider, _method_url(provider.endpoint, "chat/completions"), headers, body)
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):  # some servers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return LLMReply(text=content or "", truncated=choice.get("finish_reason") == "length")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise LLMError(f"Provider '{provider.name}': unexpected reply shape: {_short(data)}") from None


def _complete_anthropic(provider: Provider, prompt: str) -> LLMReply:
    body: dict = {
        "model": provider.model,
        "max_tokens": provider.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if provider.temperature is not None:
        body["temperature"] = provider.temperature
    headers = {"x-api-key": _require_key(provider), "anthropic-version": ANTHROPIC_VERSION}
    data = _post_json(provider, _method_url(provider.endpoint, "v1/messages"), headers, body)
    try:
        text = "".join(
            block.get("text", "")
            for block in data["content"]
            if isinstance(block, dict) and block.get("type", "text") == "text"
        )
        return LLMReply(text=text, truncated=data.get("stop_reason") == "max_tokens")
    except (KeyError, TypeError, AttributeError):
        raise LLMError(f"Provider '{provider.name}': unexpected reply shape: {_short(data)}") from None


def _method_url(endpoint: str, method: str) -> str:
    """endpoint + method, unless the endpoint already ends with it.

    A base URL that already carries the version ("https://api.anthropic.com/v1")
    is not given a second "/v1".
    """
    base = endpoint.rstrip("/")
    if base.endswith("/" + method):
        return base
    if method.startswith("v1/") and base.endswith("/v1"):
        return f"{base}/{method[len('v1/'):]}"
    return f"{base}/{method}"


def _require_key(provider: Provider) -> str:
    """The API key; LLMError if the provider names a variable that is empty."""
    if provider.api_key_env and not provider.api_key():
        raise LLMError(
            f"Provider '{provider.name}': the API key is not set - environment variable "
            f"{provider.api_key_env} is empty. Enter the key in the provider's settings or in .env."
        )
    return provider.api_key()


def _post_json(provider: Provider, url: str, headers: dict, body: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Some gateways (Cloudflare in front of many APIs) reject urllib's default agent.
            "User-Agent": USER_AGENT,
            **headers,
        },
        method="POST",
    )
    logger.info("[summary] POST %s (model %s, timeout %ds)", url, provider.model, provider.timeout_seconds)
    try:
        with urllib.request.urlopen(request, timeout=provider.timeout_seconds) as response:
            raw = response.read()
    except urllib.error.HTTPError as e:
        raise LLMError(
            f"Provider '{provider.name}': HTTP {e.code} from {url}: {_error_detail(e.read())}"
        ) from None
    except (TimeoutError, socket.timeout):
        raise LLMError(
            f"Provider '{provider.name}': no reply from {url} within {provider.timeout_seconds} s "
            "(raise the timeout in the provider's settings for long transcripts)"
        ) from None
    except urllib.error.URLError as e:
        raise LLMError(f"Provider '{provider.name}': cannot reach {url}: {e.reason}") from None
    try:
        data = json.loads(raw)
    except ValueError:
        raise LLMError(f"Provider '{provider.name}': {url} did not answer with JSON: {_short(raw)}") from None
    if not isinstance(data, dict):
        raise LLMError(f"Provider '{provider.name}': unexpected reply shape: {_short(data)}")
    return data


def _error_detail(raw: bytes) -> str:
    """The message of an API error body ({"error": {"message": ...}}), else the start of the body."""
    try:
        error = json.loads(raw).get("error")
    except (ValueError, AttributeError):
        error = None
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if isinstance(error, str) and error:
        return error
    return _short(raw)


def _short(value, limit: int = 500) -> str:
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"
