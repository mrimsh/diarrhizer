"""Accounts of the summary providers: who is signed in, signing in and out,
the models on offer, the plan's limits.

Why the vendors' own tools, and nothing of ours in between:
  - OpenAI documents `codex app-server` as the way to put Codex - sign-in with a
    ChatGPT account included - inside another product: `account/login/start`
    hands back the address of OpenAI's own sign-in page and the tool completes
    the OAuth exchange itself.
  - Anthropic does not allow other applications to offer a Claude.ai sign-in or to
    collect, store or relay Claude.ai credentials; "sign-in to a Claude account
    must complete through Anthropic's own flow". So the app only starts the
    unmodified `claude auth login`, which opens Anthropic's page in the browser.
In both cases the tokens stay in the tool's own store: this module never reads,
receives or writes a credential.

API providers have no account here, only a model list (GET <endpoint>/models).
"""

import json
import logging
import queue
import subprocess
import tempfile
import threading
import time
from typing import Callable, Optional

from diarrhizer.adapters import llm
from diarrhizer.adapters.llm import LLMError
from diarrhizer.summary.provider_info import Account, LimitWindow, Limits, ModelOption
from diarrhizer.summary.providers import KIND_ANTHROPIC_API, KIND_CLAUDE_CLI, KIND_CODEX_CLI, Provider

logger = logging.getLogger("diarrhizer.summary")

# How long a sign-in may wait for the user to finish it in the browser.
LOGIN_TIMEOUT_SECONDS = 600
# Reading the account, the model list or the limits is a quick local or one-call matter.
QUERY_TIMEOUT_SECONDS = 60
# How long to wait for a reader thread to see the end of a pipe once its tool is dead.
READER_JOIN_SECONDS = 5

CLIENT_INFO = {"name": "diarrhizer", "title": "Diarrhizer", "version": "0.1.0"}

# Callbacks: on_event(name, payload) - "url" (open this address in a browser) or "waiting" (the tool opened
# the browser itself); should_cancel() - true once the user gave up.
EventCallback = Callable[[str, str], None]
CancelCallback = Callable[[], bool]


class LoginCancelled(LLMError):
    """The user gave up on a sign-in."""


def _never() -> bool:
    return False


def _ignore(_name: str, _payload: str) -> None:
    pass


# [SEMANTIC-BEGIN] ADAPTER:LLM_ACCOUNTS
# @purpose: Sign a subscription provider in or out, and read its account, model list and plan limits
# @description: codex-cli talks to `codex app-server` (JSON-RPC, one JSON object per line on stdio):
#   account/read, account/login/start (-> the ChatGPT sign-in address, opened by the GUI; completion arrives as
#   an account/login/completed notification), account/login/cancel, account/logout, model/list (paged),
#   account/rateLimits/read (usage windows with percent used and reset time, credits). claude-cli uses
#   `claude auth status --json`, `claude auth login --claudeai` (the CLI opens the browser itself and waits for
#   the localhost callback) and `claude auth logout`; its model list comes from the stream-json control
#   protocol's `initialize` request, which Claude Code answers without a model request. Claude has no
#   limits query: read_limits() sends a one-word request and returns the rate-limit status that came with the
#   reply. API providers: the model list from GET <endpoint>/models. Every call runs in an empty temporary
#   folder with the subscription environment of ADAPTER:LLM; a sign-in can be cancelled (the GUI kills the
#   child process, which also frees the tool's local callback port). A tool's tokens are never read.
# @inputs: provider (CONFIG:SUMMARY_PROVIDERS)
# @outputs: Account, list of ModelOption, Limits (CONFIG:PROVIDER_INFO)
# @sideEffects: Starts the vendor CLI; a sign-in stores the tool's own credentials and may open a browser
#   (the tool's doing); a sign-out removes them; model listing / limits for Claude / API providers touch the network
# @errors: LLMError (tool missing or not signed in, protocol or HTTP failure, timeout), LoginCancelled
# @see: ADAPTER:LLM, CONFIG:PROVIDER_INFO, CONFIG:SUMMARY_PROVIDERS, GUI:ACCOUNT_PANEL, GUI:CHILD_TASKS
def account_status(provider: Provider) -> Account:
    """Who the provider's tool is signed in as (Account(signed_in=False) when nobody)."""
    if provider.kind == KIND_CODEX_CLI:
        with _CodexClient(provider) as client:
            return _codex_account(client.request("account/read", {"refreshToken": False}))
    if provider.kind == KIND_CLAUDE_CLI:
        return _claude_account(provider)
    raise LLMError(f"Provider '{provider.name}' is an API provider: it has a key, not an account")


def login(
    provider: Provider, on_event: EventCallback = _ignore, should_cancel: CancelCallback = _never
) -> Account:
    """Run the vendor's own sign-in and wait for the user to finish it in the browser.

    Raises:
        LoginCancelled: should_cancel() became true
        LLMError: the sign-in failed or was not finished within LOGIN_TIMEOUT_SECONDS
    """
    if provider.kind == KIND_CODEX_CLI:
        return _codex_login(provider, on_event, should_cancel)
    if provider.kind == KIND_CLAUDE_CLI:
        return _claude_login(provider, on_event, should_cancel)
    raise LLMError(f"Provider '{provider.name}' is an API provider: there is nothing to sign in to")


def logout(provider: Provider) -> Account:
    """Sign the provider's tool out (removes the tool's own stored credentials)."""
    if provider.kind == KIND_CODEX_CLI:
        with _CodexClient(provider) as client:
            client.request("account/logout")
        return Account(signed_in=False)
    if provider.kind == KIND_CLAUDE_CLI:
        _run_claude(provider, ["auth", "logout"])
        return _claude_account(provider)
    raise LLMError(f"Provider '{provider.name}' is an API provider: there is nothing to sign out of")


def list_models(provider: Provider) -> list[ModelOption]:
    """The models the provider offers. An entry with id "" stands for the provider's own default."""
    if provider.kind == KIND_CODEX_CLI:
        with _CodexClient(provider) as client:
            return _codex_models(client)
    if provider.kind == KIND_CLAUDE_CLI:
        return _claude_models(provider)
    return _api_models(provider)


def read_limits(provider: Provider) -> Limits:
    """The plan's usage windows and credits.

    Codex reports them on request. Claude does not: a one-word request is sent and the status that comes back
    with the reply is returned (it costs next to nothing, but it is a request).
    """
    if provider.kind == KIND_CODEX_CLI:
        with _CodexClient(provider) as client:
            return _codex_limits(client.request("account/rateLimits/read"))
    if provider.kind == KIND_CLAUDE_CLI:
        reply = llm.complete(provider, llm.CHECK_PROMPT)
        return reply.limits or Limits(note="Claude Code reported no limits with this request")
    return Limits(note="An API provider has no plan limits; its service applies rate limits per key")
# [SEMANTIC-END] ADAPTER:LLM_ACCOUNTS


# --- a child that speaks one JSON object per line ------------------------------------


class _JsonProcess:
    """A tool talking JSON lines on stdin/stdout (Codex app-server, Claude's stream-json)."""

    _EOF = object()

    def __init__(self, provider: Provider, argv: list[str], workdir: str) -> None:
        self._provider = provider
        self._messages: "queue.Queue" = queue.Queue()
        self._stderr: list[str] = []
        try:
            self._process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=workdir,
                env=llm.subscription_env(provider),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as e:
            raise LLMError(f"Provider '{provider.name}': cannot start {argv[0]}: {e}") from e
        self._readers = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    def _read_stdout(self) -> None:
        for raw in iter(self._process.stdout.readline, b""):
            line = llm.decode_output(raw).strip()
            if line.startswith("{"):
                try:
                    self._messages.put(json.loads(line))
                except ValueError:
                    pass
        self._messages.put(self._EOF)

    def _read_stderr(self) -> None:
        for raw in iter(self._process.stderr.readline, b""):
            self._stderr.append(llm.decode_output(raw).rstrip())

    def send(self, message: dict) -> None:
        try:
            self._process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
            self._process.stdin.flush()
        except OSError:
            raise LLMError(self._ended("closed its input")) from None

    def get(self, timeout: float) -> Optional[dict]:
        """The next JSON message, None if none arrived within `timeout`; LLMError once the tool is gone."""
        try:
            item = self._messages.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is self._EOF:
            self._messages.put(self._EOF)  # later callers see the end as well
            raise LLMError(self._ended("exited"))
        return item

    def _ended(self, what: str) -> str:
        tail = "\n".join(line for line in self._stderr[-8:] if line.strip())
        return f"Provider '{self._provider.name}': {self._provider.executable} {what} unexpectedly" + (
            f":\n{tail}" if tail else ""
        )

    def close(self) -> None:
        """Stop the tool and release its pipes. The pipes are closed only once the reader threads have seen the
        end of them: closing a pipe under a thread that is still reading it is not safe."""
        if self._process.poll() is None:
            llm.kill_tree(self._process)
        for reader in self._readers:
            reader.join(timeout=READER_JOIN_SECONDS)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            try:
                stream.close()
            except OSError:
                pass

    def __enter__(self) -> "_JsonProcess":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --- Codex: app-server ------------------------------------------------------------------


class _CodexClient:
    """A JSON-RPC session with `codex app-server` (the protocol Codex's own IDE extension uses)."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self._workdir = tempfile.TemporaryDirectory(prefix="diarrhizer-llm-", ignore_cleanup_errors=True)
        try:
            self._proc = _JsonProcess(provider, [llm.find_executable(provider), "app-server"], self._workdir.name)
        except BaseException:
            self._workdir.cleanup()
            raise
        self._next_id = 0
        self._responses: dict = {}
        self._notifications: list = []
        try:
            self.request("initialize", {"clientInfo": CLIENT_INFO})
            self._proc.send({"method": "initialized"})
        except BaseException:
            self.close()
            raise

    def request(self, method: str, params: Optional[dict] = None, timeout: float = QUERY_TIMEOUT_SECONDS) -> dict:
        self._next_id += 1
        request_id = self._next_id
        message: dict = {"id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self._proc.send(message)
        deadline = time.monotonic() + timeout
        while request_id not in self._responses:
            if time.monotonic() > deadline:
                raise LLMError(f"Provider '{self._provider.name}': codex did not answer {method} within {timeout:.0f} s")
            self._pump(1.0)
        response = self._responses.pop(request_id)
        if "error" in response:
            error = response["error"]
            detail = error.get("message") if isinstance(error, dict) else error
            raise LLMError(f"Provider '{self._provider.name}': codex {method} failed: {detail}")
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    def wait_notification(self, method: str, timeout: float) -> Optional[dict]:
        """The params of the next `method` notification, None if none came within `timeout`."""
        deadline = time.monotonic() + timeout
        while True:
            for index, note in enumerate(self._notifications):
                if note.get("method") == method:
                    del self._notifications[index]
                    return note.get("params") if isinstance(note.get("params"), dict) else {}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._pump(min(remaining, 1.0))

    def _pump(self, timeout: float) -> None:
        message = self._proc.get(timeout)
        if message is None:
            return
        if "method" in message and "id" in message:  # a request from the server: this client offers nothing
            self._proc.send({"id": message["id"], "error": {"code": -32601, "message": "not supported by this client"}})
        elif "method" in message:
            self._notifications.append(message)
        elif "id" in message:
            self._responses[message["id"]] = message

    def close(self) -> None:
        self._proc.close()
        self._workdir.cleanup()

    def __enter__(self) -> "_CodexClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _codex_account(result: dict) -> Account:
    account = result.get("account")
    if not isinstance(account, dict):
        return Account(signed_in=False)
    kind = account.get("type")
    if kind == "chatgpt":
        return Account(
            signed_in=True,
            email=str(account.get("email") or ""),
            plan=str(account.get("planType") or ""),
            method="ChatGPT",
        )
    return Account(signed_in=True, method="API key" if kind == "apiKey" else str(kind or ""))


def _codex_login(provider: Provider, on_event: EventCallback, should_cancel: CancelCallback) -> Account:
    with _CodexClient(provider) as client:
        started = client.request("account/login/start", {"type": "chatgpt"})
        url, login_id = started.get("authUrl"), started.get("loginId")
        if not url:
            raise LLMError(f"Provider '{provider.name}': codex returned no sign-in address")
        on_event("url", str(url))
        deadline = time.monotonic() + LOGIN_TIMEOUT_SECONDS
        while True:
            done = client.wait_notification("account/login/completed", 1.0)
            if done is not None:
                if done.get("success"):
                    return _codex_account(client.request("account/read", {"refreshToken": False}))
                raise LLMError(f"Provider '{provider.name}': sign-in failed: {done.get('error') or 'no reason given'}")
            expired = time.monotonic() > deadline
            if should_cancel() or expired:
                try:
                    client.request("account/login/cancel", {"loginId": login_id}, timeout=10)
                except LLMError:
                    pass
                if expired:
                    raise LLMError(f"Provider '{provider.name}': the sign-in was not finished within {LOGIN_TIMEOUT_SECONDS} s")
                raise LoginCancelled(f"Provider '{provider.name}': sign-in cancelled")


def _codex_models(client: _CodexClient) -> list[ModelOption]:
    models: list[ModelOption] = []
    cursor = None
    while True:
        result = client.request("model/list", {"cursor": cursor} if cursor else {})
        for item in result.get("data") or []:
            if not isinstance(item, dict) or item.get("hidden"):
                continue
            model_id = str(item.get("model") or item.get("id") or "")
            if model_id:
                models.append(ModelOption(
                    id=model_id,
                    name=str(item.get("displayName") or ""),
                    description=str(item.get("description") or ""),
                    default=bool(item.get("isDefault")),
                ))
        cursor = result.get("nextCursor")
        if not cursor:
            return models


def _codex_limits(result: dict) -> Limits:
    by_id = result.get("rateLimitsByLimitId")
    snapshots = list(by_id.values()) if isinstance(by_id, dict) and by_id else [result.get("rateLimits")]
    windows = []
    credits = ""
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        # The main limit is "codex"; an extra one (a model family) carries its own name.
        name = str(snapshot.get("limitName") or ("" if snapshot.get("limitId") in (None, "codex") else snapshot.get("limitId")))
        for key in ("primary", "secondary"):
            window = snapshot.get(key)
            if isinstance(window, dict):
                windows.append(LimitWindow(
                    label=name,
                    used_percent=_number(window.get("usedPercent")),
                    window_minutes=_integer(window.get("windowDurationMins")),
                    resets_at=_number(window.get("resetsAt")),
                ))
        if not credits and isinstance(snapshot.get("credits"), dict):
            data = snapshot["credits"]
            credits = "unlimited" if data.get("unlimited") else str(data.get("balance") or "0") if data.get("hasCredits") else "none"
    allowed = result.get("ordinaryUsageAllowed")
    return Limits(windows=tuple(windows), credits=credits, allowed=allowed if isinstance(allowed, bool) else None)


def _number(value) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _integer(value) -> Optional[int]:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# --- Claude: auth commands and the control protocol ---------------------------------------


def _run_claude(provider: Provider, args: list[str]) -> tuple[str, str]:
    """Run `claude <args>`; returns (stdout, stderr). Raises LLMError on failure to start or time out."""
    executable = llm.find_executable(provider)
    with tempfile.TemporaryDirectory(prefix="diarrhizer-llm-", ignore_cleanup_errors=True) as workdir:
        code, stdout, stderr = llm.run_cli(provider, [executable, *args], "", workdir, timeout=QUERY_TIMEOUT_SECONDS)
    if code != 0 and not stdout.strip():
        raise LLMError(f"Provider '{provider.name}': claude {' '.join(args)} exited with code {code}: {stderr.strip()[-500:]}")
    return stdout, stderr


def _claude_account(provider: Provider) -> Account:
    # `auth status` prints JSON whether or not anybody is signed in (and exits non-zero when nobody is).
    stdout, stderr = _run_claude(provider, ["auth", "status", "--json"])
    try:
        data = json.loads(stdout)
    except ValueError:
        raise LLMError(f"Provider '{provider.name}': unexpected output of `claude auth status`: {(stdout or stderr).strip()[:300]}") from None
    if not isinstance(data, dict) or not data.get("loggedIn"):
        return Account(signed_in=False)
    method = str(data.get("authMethod") or "")
    if str(data.get("apiProvider") or "firstParty") != "firstParty":
        method = str(data["apiProvider"])
    return Account(
        signed_in=True,
        email=str(data.get("email") or ""),
        plan=str(data.get("subscriptionType") or ""),
        method=method,
        organization=str(data.get("orgName") or data.get("organization") or ""),
    )


def _claude_login(provider: Provider, on_event: EventCallback, should_cancel: CancelCallback) -> Account:
    executable = llm.find_executable(provider)
    with tempfile.TemporaryDirectory(prefix="diarrhizer-llm-", ignore_cleanup_errors=True) as workdir:
        try:
            process = subprocess.Popen(
                [executable, "auth", "login", "--claudeai"],
                stdin=subprocess.PIPE,  # kept open: the tool may ask for a pasted code
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=workdir,
                env=llm.subscription_env(provider),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as e:
            raise LLMError(f"Provider '{provider.name}': cannot start {executable}: {e}") from e
        output: list[str] = []
        reader = threading.Thread(
            target=lambda: [output.append(llm.decode_output(raw).rstrip()) for raw in iter(process.stdout.readline, b"")],
            daemon=True,
        )
        reader.start()
        on_event("waiting", "")  # the tool opens the browser itself
        deadline = time.monotonic() + LOGIN_TIMEOUT_SECONDS
        try:
            while process.poll() is None:
                if should_cancel():
                    raise LoginCancelled(f"Provider '{provider.name}': sign-in cancelled")
                if time.monotonic() > deadline:
                    raise LLMError(f"Provider '{provider.name}': the sign-in was not finished within {LOGIN_TIMEOUT_SECONDS} s")
                time.sleep(0.5)
        except BaseException:
            llm.kill_tree(process)
            raise
        finally:
            reader.join(timeout=READER_JOIN_SECONDS)  # it ends with the tool's output; only then close the pipes
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
        if process.returncode != 0:
            tail = "\n".join(line for line in output[-6:] if line.strip())
            raise LLMError(f"Provider '{provider.name}': sign-in failed" + (f":\n{tail}" if tail else ""))
    return _claude_account(provider)


def _claude_models(provider: Provider) -> list[ModelOption]:
    executable = llm.find_executable(provider)
    base = [executable, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--tools", "", "--no-session-persistence"]
    with tempfile.TemporaryDirectory(prefix="diarrhizer-llm-", ignore_cleanup_errors=True) as workdir:
        for isolated in (True, False):
            argv = base + (list(llm.ISOLATION_FLAGS[KIND_CLAUDE_CLI]) if isolated else [])
            try:
                with _JsonProcess(provider, argv, workdir) as proc:
                    proc.send({"type": "control_request", "request_id": "diarrhizer-init", "request": {"subtype": "initialize"}})
                    models = _wait_models(proc, provider)
            except LLMError as e:
                if isolated and llm.UNKNOWN_FLAG_RE.search(str(e)):
                    continue  # an older Claude Code without one of the isolation flags
                raise
            return models
    return []  # unreachable: the second pass either returns or raises


def _wait_models(proc: _JsonProcess, provider: Provider) -> list[ModelOption]:
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        message = proc.get(1.0)
        if not message or message.get("type") != "control_response":
            continue
        response = message.get("response") or {}
        if response.get("request_id") != "diarrhizer-init":
            continue
        if response.get("subtype") != "success":
            raise LLMError(f"Provider '{provider.name}': claude refused the model query: {response.get('error')}")
        models = []
        for item in (response.get("response") or {}).get("models") or []:
            if not isinstance(item, dict) or "value" not in item:
                continue
            value = str(item["value"])
            models.append(ModelOption(
                # Claude's "default" entry stands for "no --model": the tool's own choice.
                id="" if value == "default" else value,
                name=str(item.get("displayName") or ""),
                description=str(item.get("description") or ""),
                default=value == "default",
                resolved=str(item.get("resolvedModel") or ""),
            ))
        return models
    raise LLMError(f"Provider '{provider.name}': claude did not answer the model query within {QUERY_TIMEOUT_SECONDS} s")


# --- API providers: GET <endpoint>/models --------------------------------------------------


def _api_models(provider: Provider) -> list[ModelOption]:
    if provider.kind == KIND_ANTHROPIC_API:
        url = llm.method_url(provider.endpoint, "v1/models") + "?limit=1000"
    else:
        endpoint = provider.endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            endpoint = endpoint[: -len("/chat/completions")]
        url = llm.method_url(endpoint, "models")
    data = llm.request_json(provider, url, llm.api_headers(provider))
    items = data.get("data") if isinstance(data.get("data"), list) else data.get("models")
    models = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and (item.get("id") or item.get("name")):
            models.append(ModelOption(
                id=str(item.get("id") or item.get("name")),
                name=str(item.get("display_name") or ""),
                description=str(item.get("description") or ""),
            ))
    if provider.kind != KIND_ANTHROPIC_API:
        models.sort(key=lambda m: m.id)  # OpenAI-style lists come in no useful order; Anthropic's is newest first
    return models
