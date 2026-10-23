"""Adapter for the language models that write summaries.

Subscription providers are driven through the vendors' own command-line tools
(`codex exec`, `claude -p`) in non-interactive mode, with the prompt on stdin
so it never meets a command-line length limit or Windows quoting rules. The
tools are asked for machine-readable output (`codex --json`, Claude's
`stream-json`) so a reply comes back with what it used. API providers are
plain HTTPS JSON calls made with urllib - no SDK dependency.

A request is a conversation (complete_chat: messages with roles, content as text or
blocks, a block may mark a cache breakpoint); complete() is a single user message.

Signing in, the model list and the plan's limits are in llm_accounts.py.
"""

import dataclasses
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from diarrhizer.summary.provider_info import LimitWindow, Limits, Usage
from diarrhizer.summary.providers import (
    KIND_ANTHROPIC_API,
    KIND_CLAUDE_CLI,
    KIND_CODEX_CLI,
    KIND_OPENAI_API,
    Provider,
)

logger = logging.getLogger("diarrhizer.summary")

USER_AGENT = "diarrhizer/0.1"
# What provider checks send: a request that costs next to nothing and proves the sign-in/key works.
CHECK_PROMPT = "Ответь одним словом: ok"
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
ISOLATION_FLAGS = {
    KIND_CLAUDE_CLI: ("--safe-mode",),
    KIND_CODEX_CLI: ("--ignore-user-config", "--ephemeral"),
}
UNKNOWN_FLAG_RE = re.compile(r"unknown option|unexpected argument|unrecognized (option|argument)", re.I)

_LOGIN_HINTS = {
    KIND_CLAUDE_CLI: "If it is a sign-in problem, sign in again: `claude auth login` (or the provider's sign-in button in the GUI).",
    KIND_CODEX_CLI: "If it is a sign-in problem, sign in again: `codex login` (or the provider's sign-in button in the GUI).",
}

# Claude's rate_limit_event names the window by type; the length is not in the event.
_CLAUDE_WINDOW_MINUTES = {"five_hour": 300, "seven_day": 10080, "seven_day_opus": 10080, "seven_day_sonnet": 10080}

CLAUDE_COST_NOTE = "estimated by Claude Code from API list prices; a subscription is not billed per token"
PROVIDER_COST_NOTE = "reported by the provider"
PRICE_COST_NOTE = "estimated from the prices set for this provider"


class LLMError(RuntimeError):
    """A provider could not produce a reply; the message says why and what to try."""


@dataclass(frozen=True)
class LLMReply:
    """What a provider answered, and what the request used.

    Attributes:
        text: The reply
        truncated: The provider cut the reply off at its length limit
        model: The model that answered, when the provider says
        usage: Token counts, when the provider reports them
        cost_usd: What the request cost in USD, when known (see cost_note for how it is known)
        cost_note: Where cost_usd comes from, e.g. an estimate that is not a bill
        limits: The plan's limits as the request saw them, when the provider reports them with a reply
        session_id: A CLI session that holds this conversation, to pass back to continue it ("" = none kept)
        context_mode: What was sent: CONTEXT_SESSION (only the new message, the CLI's session has the rest) or
            CONTEXT_FULL_HISTORY (the whole conversation)
        context_chars: Characters of message text in the request - the size of the context that was sent
        cache_available: Whether the provider can reuse what it has already seen: True for a resumed CLI session,
            False for a CLI that was sent the whole history, None when this is not known (API providers say it
            in usage)
    """

    text: str
    truncated: bool = False
    model: str = ""
    usage: Optional[Usage] = None
    cost_usd: Optional[float] = None
    cost_note: str = ""
    limits: Optional[Limits] = None
    session_id: str = ""
    context_mode: str = ""
    context_chars: Optional[int] = None
    cache_available: Optional[bool] = None


CONTEXT_SESSION = "cli-session"
CONTEXT_FULL_HISTORY = "full-history"
# What Codex says when the session to resume is not on disk (checked against codex-cli 0.155.1).
SESSION_LOST_RE = re.compile(r"no rollout found|thread/resume failed", re.I)


ROLES = ("system", "user", "assistant")
_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


@dataclass(frozen=True)
class Block:
    """A piece of a message's text.

    `cache` marks a cache breakpoint: the request up to the end of this block is worth caching. The Anthropic
    API is told so (cache_control); an OpenAI-compatible API caches a repeated prefix by itself, so nothing is
    sent there; a CLI provider gets plain text.
    """

    text: str
    cache: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError(f"a message block needs non-empty text, got {self.text!r}")
        if not isinstance(self.cache, bool):
            raise ValueError(f"a block's cache flag must be true or false, got {self.cache!r}")

    @classmethod
    def from_value(cls, value) -> "Block":
        """A Block, a plain string, or a dict {"text": ..., "cache": true|false}."""
        if isinstance(value, Block):
            return value
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping) and set(value) <= {"text", "cache"}:
            return cls(value.get("text"), value.get("cache", False))
        raise ValueError(f"a message block is a string or {{'text': ..., 'cache': ...}}, got {value!r}")


@dataclass(frozen=True)
class Message:
    """One message of a conversation: a role and its content, a string or a list of Blocks."""

    role: str
    content: "str | tuple[Block, ...]"

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"a message's role must be one of {', '.join(ROLES)}, got {self.role!r}")
        if isinstance(self.content, str):
            if not self.content:
                raise ValueError(f"a {self.role} message needs content")
            return
        if isinstance(self.content, Mapping) or not isinstance(self.content, Sequence):
            raise ValueError(f"a message's content is a string or a list of blocks, got {self.content!r}")
        blocks = tuple(Block.from_value(block) for block in self.content)
        if not blocks:
            raise ValueError(f"a {self.role} message needs content")
        object.__setattr__(self, "content", blocks)

    @property
    def blocks(self) -> "tuple[Block, ...]":
        return (Block(self.content),) if isinstance(self.content, str) else self.content

    @property
    def text(self) -> str:
        """The message as one string: its blocks run together."""
        return "".join(block.text for block in self.blocks)

    @classmethod
    def from_value(cls, value) -> "Message":
        """A Message, or a dict {"role": ..., "content": ...} (what crosses a process queue or a JSON file)."""
        if isinstance(value, Message):
            return value
        if isinstance(value, Mapping) and set(value) == {"role", "content"}:
            return cls(value["role"], value["content"])
        raise ValueError(f"a message is a Message or {{'role': ..., 'content': ...}}, got {value!r}")


def coerce_messages(messages: "Sequence[Message | Mapping]") -> list[Message]:
    """The messages as Message objects; ValueError says what is wrong with them."""
    if isinstance(messages, (str, bytes, Mapping)) or not isinstance(messages, Sequence):
        raise ValueError(f"messages must be a list of messages, got {type(messages).__name__}")
    chat = [Message.from_value(message) for message in messages]
    if not any(message.role != "system" for message in chat):
        raise ValueError("messages need at least one user or assistant message")
    return chat


def messages_as_text(messages: "Sequence[Message]") -> str:
    """A conversation as the one text request a CLI takes - all of it, every time.

    A lone user message is sent as it is (what `complete` always sent). Anything longer is written out with
    the role of each message ahead of it.
    """
    if len(messages) == 1 and messages[0].role == "user":
        return messages[0].text
    return "\n\n".join(f"{_ROLE_LABELS[message.role]}:\n{message.text}" for message in messages)


# [SEMANTIC-BEGIN] ADAPTER:LLM
# @purpose: Send a conversation (or one prompt) to a configured provider and return the model's reply with what it used
# @description: complete_chat() takes messages - roles system/user/assistant, content a string or a list of
#   Blocks, a Block may carry a cache flag (a cache breakpoint) - and complete(provider, prompt) is its thin
#   wrapper: one user message. anthropic-api gets the messages and a top-level `system` (a string, or a list of
#   text blocks when a system message has blocks), a content list for every message that was given blocks,
#   and cache_control {"type": "ephemeral"} on each block flagged cache; openai-compatible APIs get plain
#   messages (a message's blocks run together) and nothing about caching, which they do by repeated prefix.
#   A CLI provider gets the whole conversation as one text request every time (a lone user message verbatim,
#   a longer conversation with a "System:/User:/Assistant:" label ahead of each message). Codex can keep its
#   session: keep_session=True writes it to disk (no --ephemeral) and the reply carries session_id; passing
#   session_id resumes it with `codex exec resume` and sends only the messages after the last assistant one - if
#   that session is gone the whole conversation goes again in a new session. Claude Code is always sent the whole
#   history (its resume was not run here). Every reply says what happened: context_mode (cli-session /
#   full-history), context_chars (size of the context sent) and cache_available (docs/cli-sessions.md).
#   codex-cli runs `codex exec --json --sandbox read-only` and claude-cli runs `claude -p --tools ""
#   --output-format stream-json`, both with the prompt on stdin, in an empty temporary working directory (so
#   neither tool wanders into or loads the context of the repo), without the API-key variables that would
#   bypass the subscription and with the ISOLATION_FLAGS that skip the user's own tool setup (retried without
#   them if the installed version rejects a flag). Their JSON events carry the reply, token usage, and - for
#   Claude - an estimated cost and the plan's rate-limit status; Codex reports tokens only. Tool output is
#   decoded as UTF-8, else in the console code page (Codex prints cp1251 on Russian Windows). A timeout kills
#   the whole process tree - the Windows .cmd shims start the real tool as a grandchild. openai-api and
#   anthropic-api are single non-streaming HTTPS POSTs: the endpoint may be a base URL or the full method URL,
#   the key is read from the provider's api_key_env at call time and never appears in an error message; usage
#   comes from the reply - input and output tokens and, where the provider says them, the cache parts of the
#   input (read from the cache, written to it, neither; None when not reported, never a made-up 0) - and cost
#   from the provider (OpenRouter) or from the prices set on the Provider (cache tokens at their own prices
#   when those are set, else at the input price).
#   Errors are always LLMError with an actionable text; a tool that cannot be started says which of the three it is
#   (a bare name that is not on PATH, a path with no file behind it, a folder) and names the section of
#   docs/troubleshooting.md that explains how to find the executable and tell the app where it is.
# @inputs: provider (CONFIG:SUMMARY_PROVIDERS), messages (Message / Block, or the same as dicts) or a prompt text (already containing the transcript)
# @outputs: LLMReply(text, truncated, model, usage, cost_usd, cost_note, limits)
# @sideEffects: Starts the vendor CLI (which talks to its service) or sends the messages over the network;
#   creates and removes a temporary directory
# @errors: LLMError (tool missing, not signed in, HTTP/network failure, timeout, empty or malformed reply); ValueError for malformed messages
# @see: CONFIG:SUMMARY_PROVIDERS, CONFIG:PROVIDER_INFO, ADAPTER:LLM_ACCOUNTS, SUMMARY:JOB, GUI:CHILD_TASKS
def complete_chat(
    provider: Provider,
    messages: "Sequence[Message | Mapping]",
    keep_session: bool = False,
    session_id: str = "",
) -> LLMReply:
    chat = coerce_messages(messages)
    if provider.is_cli:
        reply = _complete_cli(provider, chat, keep_session, session_id)
    else:
        if provider.kind == KIND_OPENAI_API:
            reply = _complete_openai(provider, chat)
        elif provider.kind == KIND_ANTHROPIC_API:
            reply = _complete_anthropic(provider, chat)
        else:  # Provider validates its kind; this guards a kind added without a backend.
            raise LLMError(f"Provider '{provider.name}': unsupported kind {provider.kind!r}")
        reply = dataclasses.replace(
            reply, context_mode=CONTEXT_FULL_HISTORY, context_chars=sum(len(m.text) for m in chat)
        )
    if not reply.text.strip():
        raise LLMError(f"Provider '{provider.name}' returned an empty reply")
    return dataclasses.replace(reply, text=reply.text.strip())


def complete(provider: Provider, prompt: str) -> LLMReply:
    """One prompt, one reply: a conversation of one user message."""
    return complete_chat(provider, [Message("user", prompt)])
# [SEMANTIC-END] ADAPTER:LLM


# --- subscription CLIs ---------------------------------------------------------


def _complete_cli(provider: Provider, chat: list[Message], keep_session: bool, session_id: str) -> LLMReply:
    """One CLI request. Codex can keep its session (see the module's docs/cli-sessions.md): then a request with
    a session_id sends only the messages after the last assistant one; if that session is gone the whole
    conversation goes again, in a new session. Claude Code always gets the whole conversation."""
    executable = find_executable(provider)
    can_keep = provider.kind == KIND_CODEX_CLI and (keep_session or bool(session_id))
    full = messages_as_text(chat)
    if can_keep and session_id:
        new = messages_as_text(_messages_since_last_reply(chat))
        try:
            reply = _run_cli(provider, executable, new, session_id)
            return dataclasses.replace(
                reply, session_id=reply.session_id or session_id, context_mode=CONTEXT_SESSION,
                context_chars=len(new), cache_available=True,
            )
        except LLMError as e:
            if not SESSION_LOST_RE.search(str(e)):
                raise
            logger.warning("[summary] %s: session %s is gone, sending the whole conversation", provider.name, session_id)
    reply = _run_cli(provider, executable, full, "", keep=can_keep)
    return dataclasses.replace(
        reply,
        session_id=reply.session_id if can_keep else "",  # an ephemeral run has an id but nothing to resume
        context_mode=CONTEXT_FULL_HISTORY,
        context_chars=len(full),
        cache_available=None if can_keep else False,
    )


def _messages_since_last_reply(chat: list[Message]) -> list[Message]:
    """What the session has not seen: the messages after the last assistant message (system ones never resent)."""
    last = max((i for i, message in enumerate(chat) if message.role == "assistant"), default=-1)
    tail = [message for message in chat[last + 1:] if message.role != "system"]
    return tail or chat[-1:]


def _run_cli(
    provider: Provider, executable: str, prompt: str, session_id: str, keep: bool = False
) -> LLMReply:
    keep = keep or bool(session_id)
    with tempfile.TemporaryDirectory(prefix="diarrhizer-llm-", ignore_cleanup_errors=True) as workdir:
        reply_file = Path(workdir) / "reply.txt"
        for isolated in (True, False):
            argv = _cli_argv(provider, executable, reply_file, isolated, keep, session_id)
            code, stdout, stderr = run_cli(provider, argv, prompt, workdir)
            if code == 0 and provider.kind == KIND_CLAUDE_CLI and _claude_result(stdout).get("is_error"):
                code = 1  # Claude reports a failure inside the run (e.g. not signed in) as an error result
            if code == 0:
                break
            if not (isolated and UNKNOWN_FLAG_RE.search(stdout + stderr)):
                raise _cli_failure(provider, code, _failure_text(provider, stdout), stderr)
            logger.warning("[summary] %s rejected an option, retrying without them", executable)
        if provider.kind == KIND_CLAUDE_CLI:
            return _claude_reply(stdout)
        return _codex_reply(provider, stdout, reply_file)


def _cli_argv(
    provider: Provider, executable: str, reply_file: Path, isolated: bool, keep: bool = False, session_id: str = ""
) -> list[str]:
    isolation = list(ISOLATION_FLAGS[provider.kind]) if isolated else []
    if keep:  # a session that is to be resumed must be written to disk: no --ephemeral
        isolation = [flag for flag in isolation if flag != "--ephemeral"]
    if provider.kind == KIND_CLAUDE_CLI:
        argv = [executable, "-p", "--tools", "", "--no-session-persistence",
                "--output-format", "stream-json", "--verbose"]
    elif session_id:  # `exec resume` takes no --sandbox/--color: the session keeps the policy it started with
        argv = [executable, "exec", "resume", "--json", "--skip-git-repo-check", "--output-last-message", str(reply_file)]
    else:
        argv = [executable, "exec", "--json", "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never",
                "--output-last-message", str(reply_file)]
    argv += isolation
    if provider.model:
        argv += ["--model", provider.model]
    argv += list(provider.extra_args)
    if provider.kind == KIND_CODEX_CLI:
        if session_id:
            argv.append(session_id)
        argv.append("-")  # the prompt comes from stdin
    return argv


def _json_lines(text: str) -> list[dict]:
    """The JSON objects among the lines of a tool's output (banners and noise are skipped)."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _claude_result(stdout: str) -> dict:
    """Claude's final "result" event, {} if the run produced none."""
    results = [event for event in _json_lines(stdout) if event.get("type") == "result"]
    return results[-1] if results else {}


def _claude_reply(stdout: str) -> LLMReply:
    events = _json_lines(stdout)
    result = _claude_result(stdout)
    model = next((e.get("model", "") for e in events if e.get("type") == "system" and e.get("subtype") == "init"), "")
    model_usage = result.get("modelUsage") or {}
    if model_usage:
        # The model that did the work, when subagents or helpers used others as well.
        model = max(model_usage, key=lambda m: _count(model_usage[m], "inputTokens") + _count(model_usage[m], "outputTokens"))
    usage = _anthropic_usage(result.get("usage") or {})
    cost = result.get("total_cost_usd")
    limit_events = [e.get("rate_limit_info") for e in events if e.get("type") == "rate_limit_event"]
    return LLMReply(
        text=str(result.get("result") or ""),
        truncated=result.get("stop_reason") == "max_tokens",
        model=model if model and not model.startswith("<") else "",
        usage=usage if usage.total else None,
        cost_usd=float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
        cost_note=CLAUDE_COST_NOTE if isinstance(cost, (int, float)) else "",
        limits=_claude_limits(limit_events[-1]) if limit_events and isinstance(limit_events[-1], dict) else None,
    )


def _claude_limits(info: dict) -> Limits:
    """Limits from Claude's rate_limit_event. The documented fields are status, resetsAt and utilization;
    the window type is read when the event carries one. The scale of `utilization` is not documented: a
    value up to 1 is taken as a fraction, a larger one as a percentage."""
    utilization = info.get("utilization")
    percent = None
    if isinstance(utilization, (int, float)) and not isinstance(utilization, bool):
        percent = utilization * 100 if utilization <= 1 else float(utilization)
    kind = str(info.get("rateLimitType") or "")
    resets = info.get("resetsAt")
    status = str(info.get("status") or "")
    return Limits(
        windows=(LimitWindow(
            label=kind,
            used_percent=percent,
            window_minutes=_CLAUDE_WINDOW_MINUTES.get(kind),
            resets_at=float(resets) if isinstance(resets, (int, float)) and not isinstance(resets, bool) else None,
            status=status,
        ),),
        allowed=None if not status else status != "rejected",
    )


def _codex_reply(provider: Provider, stdout: str, reply_file: Path) -> LLMReply:
    events = _json_lines(stdout)
    messages = [
        e["item"].get("text", "") for e in events
        if e.get("type") == "item.completed" and isinstance(e.get("item"), dict)
        and e["item"].get("type") == "agent_message"
    ]
    # The reply file holds the final message; the event is the fallback if codex wrote no file.
    text = decode_output(reply_file.read_bytes()) if reply_file.exists() else (messages[-1] if messages else "")
    completed = [e.get("usage") for e in events if e.get("type") == "turn.completed" and isinstance(e.get("usage"), dict)]
    usage = None
    if completed:
        data = completed[-1]
        usage = _inclusive_usage(  # Codex counts the cached tokens inside input_tokens, as OpenAI does
            _count(data, "input_tokens"), _count(data, "output_tokens"), _opt_count(data, "cached_input_tokens"),
            _opt_count(data, "cache_write_input_tokens"), _count(data, "reasoning_output_tokens"),
        )
    thread = next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), "")
    return LLMReply(
        text=text, model=provider.model, usage=usage if usage and usage.total else None,
        session_id=thread if isinstance(thread, str) else "",
    )


def _count(data: dict, key: str) -> int:
    value = data.get(key)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _opt_count(data: dict, key: str) -> Optional[int]:
    """The number under `key`, None when the provider does not report it (which is not the same as 0)."""
    value = data.get(key)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _anthropic_usage(raw: dict) -> Usage:
    """Usage from Anthropic's `usage` (the API's, and Claude Code's result event, which has the same keys).

    Anthropic reports the input in three separate counters: what was neither read from nor written to the
    cache (input_tokens), what the cache served and what the request wrote to it. All three are input.
    """
    uncached = _opt_count(raw, "input_tokens")
    read = _opt_count(raw, "cache_read_input_tokens")
    write = _opt_count(raw, "cache_creation_input_tokens")
    details = raw.get("output_tokens_details")
    return Usage(
        input_tokens=(uncached or 0) + (read or 0) + (write or 0),
        output_tokens=_count(raw, "output_tokens"),
        cached_tokens=read,
        reasoning_tokens=_count(details, "thinking_tokens") if isinstance(details, dict) else 0,
        cache_write_tokens=write,
        uncached_input_tokens=uncached,
    )


def _inclusive_usage(
    total_input: int, output: int, read: Optional[int], write: Optional[int], reasoning: int
) -> Usage:
    """Usage in the OpenAI shape, where the input total already contains the cache's parts.

    The uncached part is what is left of the input once the cache's parts are taken out - known only when the
    provider reports how much of it the cache served.
    """
    uncached = max(total_input - read - (write or 0), 0) if read is not None else None
    return Usage(
        input_tokens=total_input,
        output_tokens=output,
        cached_tokens=read,
        reasoning_tokens=reasoning,
        cache_write_tokens=write,
        uncached_input_tokens=uncached,
    )


def find_executable(provider: Provider) -> str:
    """Full path of the provider's CLI; LLMError if it is not installed."""
    command = provider.executable
    found = shutil.which(command)
    if found:
        return found
    where = "See docs/troubleshooting.md, section \"command 'codex' not found although Codex is installed\"."
    if os.path.isdir(command):
        raise LLMError(
            f"Provider '{provider.name}': '{command}' is a folder, not a program - give the path of the "
            f"executable inside it (for example codex.exe or codex.cmd). {where}"
        )
    if "\\" in command or "/" in command:
        raise LLMError(
            f"Provider '{provider.name}': command '{command}' not found - there is no such file. "
            f"Check the path in the provider's command field. {where}"
        )
    raise LLMError(
        f"Provider '{provider.name}': command '{command}' not found - it is not on PATH. Install it, put its "
        f"folder on PATH, or give the full path to the executable in the provider's command field. {where}"
    )


def subscription_env(provider: Provider) -> dict:
    """The environment for a subscription CLI: ours, minus the API keys that would switch its billing."""
    return {
        key: value
        for key, value in os.environ.items()
        if key not in _API_KEY_VARIABLES.get(provider.kind, ())
    }


def run_cli(
    provider: Provider, argv: list[str], prompt: str, workdir: str, timeout: Optional[int] = None
) -> tuple[int, str, str]:
    """Run the tool with `prompt` on stdin. Returns (exit code, stdout, stderr).

    Raises:
        LLMError: the tool cannot be started, or does not finish within `timeout` (default: the provider's)
    """
    timeout = provider.timeout_seconds if timeout is None else timeout
    logger.info("[summary] Running %s (timeout %ds)", Path(argv[0]).name, timeout)
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
            env=subscription_env(provider),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as e:
        raise LLMError(f"Provider '{provider.name}': cannot start {argv[0]}: {e}") from e
    try:
        stdout, stderr = process.communicate(prompt.encode("utf-8"), timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_tree(process)
        try:
            process.communicate(timeout=5)  # reap it; never wait on a survivor holding the pipes
        except subprocess.TimeoutExpired:
            pass
        raise LLMError(
            f"Provider '{provider.name}': no reply within {timeout} s "
            "(raise the timeout in the provider's settings for long transcripts)"
        ) from None
    return process.returncode, decode_output(stdout), decode_output(stderr)


def decode_output(data: bytes) -> str:
    """UTF-8, else the console code page (a tool that prints in cp1251 is not valid UTF-8)."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False), "replace")


def kill_tree(process: subprocess.Popen) -> None:
    """Kill the tool and what it started (npm's .cmd shim runs the real tool as a child)."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    process.kill()


def _failure_text(provider: Provider, stdout: str) -> str:
    """What a failed tool said on stdout, in words: its JSON events reduced to the error they carry."""
    events = _json_lines(stdout)
    if not events:
        return stdout
    if provider.kind == KIND_CLAUDE_CLI:
        result = _claude_result(stdout)
        return str(result.get("result") or "") if result.get("is_error") or result.get("result") else ""
    messages = []
    for event in events:
        if event.get("type") == "error":
            messages.append(event.get("message"))
        elif event.get("type") == "turn.failed" and isinstance(event.get("error"), dict):
            messages.append(event["error"].get("message"))
    # The message is often the provider's JSON error body, as text.
    return "\n".join(dict.fromkeys(
        _error_detail(str(message).encode("utf-8")) for message in messages if message
    ))


def _cli_failure(provider: Provider, code: int, stdout: str, stderr: str) -> LLMError:
    # Claude Code says "Not logged in" in its result event, Codex its errors in events or on stderr: look at all.
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


def _complete_openai(provider: Provider, messages: list[Message]) -> LLMReply:
    # Plain messages. A cache breakpoint is not sent: these servers cache a repeated prefix by themselves.
    body: dict = {
        "model": provider.model,
        "messages": [{"role": message.role, "content": message.text} for message in messages],
    }
    if provider.temperature is not None:
        body["temperature"] = provider.temperature
    if provider.max_tokens is not None:
        body["max_tokens"] = provider.max_tokens
    data = request_json(provider, method_url(provider.endpoint, "chat/completions"), _openai_headers(provider), body)
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):  # some servers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text, truncated = content or "", choice.get("finish_reason") == "length"
    except (KeyError, IndexError, TypeError, AttributeError):
        raise LLMError(f"Provider '{provider.name}': unexpected reply shape: {_short(data)}") from None
    raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    prompt_details = raw.get("prompt_tokens_details") if isinstance(raw.get("prompt_tokens_details"), dict) else {}
    reasoning = raw.get("completion_tokens_details") if isinstance(raw.get("completion_tokens_details"), dict) else {}
    usage = _inclusive_usage(
        _count(raw, "prompt_tokens"), _count(raw, "completion_tokens"),
        _opt_count(prompt_details, "cached_tokens"),
        _opt_count(prompt_details, "cache_write_tokens"),  # OpenAI has none; some gateways report it
        _count(reasoning, "reasoning_tokens"),
    )
    cost, note = _api_cost(provider, usage, raw.get("cost"))
    return LLMReply(
        text=text, truncated=truncated, model=str(data.get("model") or provider.model),
        usage=usage if usage.total else None, cost_usd=cost, cost_note=note,
    )


def _anthropic_body(provider: Provider, messages: list[Message]) -> dict:
    """The Messages API request: system messages go to the top-level `system`, the rest to `messages`.

    Content given as a string is sent as a string; content given as blocks as a list of text blocks, each
    flagged cache getting cache_control {"type": "ephemeral"} (the cache breakpoint).
    """

    def blocks(message: Message) -> list[dict]:
        return [
            {"type": "text", "text": block.text, **({"cache_control": {"type": "ephemeral"}} if block.cache else {})}
            for block in message.blocks
        ]

    system = [message for message in messages if message.role == "system"]
    body: dict = {
        "model": provider.model,
        "max_tokens": provider.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
        "messages": [
            {"role": message.role, "content": message.content if isinstance(message.content, str) else blocks(message)}
            for message in messages if message.role != "system"
        ],
    }
    if system:
        if all(isinstance(message.content, str) for message in system):
            body["system"] = "\n\n".join(message.content for message in system)
        else:
            body["system"] = [block for message in system for block in blocks(message)]
    if provider.temperature is not None:
        body["temperature"] = provider.temperature
    return body


def _complete_anthropic(provider: Provider, messages: list[Message]) -> LLMReply:
    body = _anthropic_body(provider, messages)
    data = request_json(provider, method_url(provider.endpoint, "v1/messages"), _anthropic_headers(provider), body)
    try:
        text = "".join(
            block.get("text", "")
            for block in data["content"]
            if isinstance(block, dict) and block.get("type", "text") == "text"
        )
        truncated = data.get("stop_reason") == "max_tokens"
    except (KeyError, TypeError, AttributeError):
        raise LLMError(f"Provider '{provider.name}': unexpected reply shape: {_short(data)}") from None
    usage = _anthropic_usage(data.get("usage") if isinstance(data.get("usage"), dict) else {})
    cost, note = _api_cost(provider, usage, None)
    return LLMReply(
        text=text, truncated=truncated, model=str(data.get("model") or provider.model),
        usage=usage if usage.total else None, cost_usd=cost, cost_note=note,
    )


def _api_cost(provider: Provider, usage: Usage, reported) -> tuple[Optional[float], str]:
    """(USD, where it comes from): the provider's own figure if the reply has one, else the user's prices."""
    if isinstance(reported, (int, float)) and not isinstance(reported, bool):
        return float(reported), PROVIDER_COST_NOTE
    estimate = provider.estimate_cost(
        usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.cache_write_tokens
    )
    return (estimate, PRICE_COST_NOTE) if estimate is not None else (None, "")


def _openai_headers(provider: Provider) -> dict:
    key = _require_key(provider)
    return {"Authorization": f"Bearer {key}"} if key else {}


def _anthropic_headers(provider: Provider) -> dict:
    return {"x-api-key": _require_key(provider), "anthropic-version": ANTHROPIC_VERSION}


def api_headers(provider: Provider) -> dict:
    """Authentication headers for a call to an API provider's endpoint."""
    return _anthropic_headers(provider) if provider.kind == KIND_ANTHROPIC_API else _openai_headers(provider)


def method_url(endpoint: str, method: str) -> str:
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


def request_json(provider: Provider, url: str, headers: dict, body: Optional[dict] = None) -> dict:
    """POST `body` as JSON (GET when there is none) and return the JSON object that comes back."""
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            # Some gateways (Cloudflare in front of many APIs) reject urllib's default agent.
            "User-Agent": USER_AGENT,
            **headers,
        },
        method="GET" if body is None else "POST",
    )
    logger.info("[summary] %s %s (model %s, timeout %ds)", request.get_method(), url, provider.model, provider.timeout_seconds)
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
