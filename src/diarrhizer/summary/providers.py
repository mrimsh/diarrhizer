"""Model providers that write summaries.

Four kinds:
  - codex-cli / claude-cli: the official `codex` / `claude` command-line tools,
    which use the ChatGPT / Claude subscription the user is already signed in
    to there. Nothing of the subscription is stored or handled by Diarrhizer.
  - openai-api: any OpenAI-compatible chat/completions endpoint (OpenAI,
    OpenRouter, DeepSeek, Groq, Ollama, LM Studio, vLLM, ...).
  - anthropic-api: the Anthropic Messages API.

Providers and the default choice live in a JSON file shared by the CLI and
the GUI. It never holds a secret: an API provider only names the environment
variable its key is read from (the GUI keeps the key itself in the project's
.env, like the Hugging Face token).
"""

import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from diarrhizer.summary import store_file
from diarrhizer.utils import write_json_atomic

# Overrides where providers are stored (default: <repo root>/summary_providers.json).
ENV_PROVIDERS_FILE = "DIARRHIZER_SUMMARY_PROVIDERS_FILE"
PROVIDERS_FILENAME = "summary_providers.json"

KIND_CODEX_CLI = "codex-cli"
KIND_CLAUDE_CLI = "claude-cli"
KIND_OPENAI_API = "openai-api"
KIND_ANTHROPIC_API = "anthropic-api"
KINDS = (KIND_CODEX_CLI, KIND_CLAUDE_CLI, KIND_OPENAI_API, KIND_ANTHROPIC_API)
CLI_KINDS = (KIND_CODEX_CLI, KIND_CLAUDE_CLI)

DEFAULT_COMMANDS = {KIND_CODEX_CLI: "codex", KIND_CLAUDE_CLI: "claude"}
DEFAULT_ANTHROPIC_ENDPOINT = "https://api.anthropic.com"
DEFAULT_TIMEOUT_SECONDS = 600

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_NAME_LENGTH = 60


def default_key_env(name: str) -> str:
    """The environment variable a provider named `name` keeps its API key in."""
    slug = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
    return f"DIARRHIZER_LLM_KEY_{slug or 'PROVIDER'}"


# [SEMANTIC-BEGIN] CONFIG:SUMMARY_PROVIDERS
# @purpose: Where and how a summary is generated: a subscription CLI or an API with endpoint, model and key
# @description: A Provider is validated per kind. CLI kinds need only an optional command (default codex /
#   claude), model and extra arguments. API kinds need a model and an endpoint (the Anthropic one defaults
#   to api.anthropic.com); the key is never stored here - api_key_env names the environment variable it is
#   read from, which keeps the JSON file free of secrets (an OpenAI-compatible local server may need no key,
#   so an empty api_key_env sends no Authorization header). ProviderStore keeps the providers and the default
#   name in a JSON file; there are no built-ins, an empty store is a normal state the callers explain.
# @inputs: provider fields (validated on construction), JSON store path (DIARRHIZER_SUMMARY_PROVIDERS_FILE)
# @outputs: Provider instances, summary_providers.json
# @sideEffects: ProviderStore.load() reads and save() atomically writes the JSON store; Provider.api_key() reads os.environ
# @errors: ValueError for invalid fields, unknown names or an unreadable store file
# @see: ADAPTER:LLM, SUMMARY:JOB, CONFIG:SUMMARY_TEMPLATES, CONFIG:ENV, GUI:SUMMARY_SETTINGS
@dataclass(frozen=True)
class Provider:
    """One configured model provider.

    Attributes:
        name: Shown in the GUI and used on the CLI; unique, single line
        kind: One of KINDS
        command: CLI kinds: the executable (default: codex / claude); name on PATH or a full path
        endpoint: API kinds: base URL, e.g. https://api.openai.com/v1 (the full .../chat/completions URL also works)
        model: Model id; required for API kinds, optional for CLI kinds (the tool's own default)
        api_key_env: API kinds: environment variable holding the API key
        temperature: API kinds: sampling temperature, None = the service's default
        max_tokens: API kinds: cap on the reply length, None = the service's default
        timeout_seconds: How long to wait for the reply
        extra_args: CLI kinds: arguments appended to the command line
    """

    name: str
    kind: str
    command: str = ""
    endpoint: str = ""
    model: str = ""
    api_key_env: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # JSON hands back lists; a tuple keeps the frozen dataclass hashable.
        object.__setattr__(self, "extra_args", tuple(str(arg) for arg in self.extra_args))
        for attr in ("command", "endpoint", "model", "api_key_env"):
            value = getattr(self, attr)
            if not isinstance(value, str):
                raise ValueError(f"Provider '{self.name}': {attr} must be text, got {value!r}")
            object.__setattr__(self, attr, value.strip())
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Provider name must not be empty")
        if self.name != self.name.strip() or "\n" in self.name or len(self.name) > _MAX_NAME_LENGTH:
            raise ValueError(
                f"Provider name must be one line of up to {_MAX_NAME_LENGTH} characters "
                f"with no leading/trailing spaces: {self.name!r}"
            )
        prefix = f"Provider '{self.name}'"
        if self.kind not in KINDS:
            raise ValueError(f"{prefix}: kind must be one of {', '.join(KINDS)}, got {self.kind!r}")
        if self.api_key_env and not _ENV_NAME_RE.match(self.api_key_env):
            raise ValueError(
                f"{prefix}: api_key_env must be an environment variable name, got {self.api_key_env!r}"
            )
        if not _is_int(self.timeout_seconds) or not 1 <= self.timeout_seconds <= 24 * 3600:
            raise ValueError(f"{prefix}: timeout_seconds must be 1-86400, got {self.timeout_seconds!r}")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError(f"{prefix}: temperature must be a number from 0 to 2, got {self.temperature!r}")
        if self.max_tokens is not None and (not _is_int(self.max_tokens) or self.max_tokens < 1):
            raise ValueError(f"{prefix}: max_tokens must be a positive integer, got {self.max_tokens!r}")

        if self.kind in CLI_KINDS:
            return
        if not self.model:
            raise ValueError(f"{prefix}: an API provider needs a model name")
        if not self.endpoint:
            raise ValueError(f"{prefix}: an API provider needs an endpoint (base URL)")
        if not re.match(r"^https?://\S+$", self.endpoint):
            raise ValueError(f"{prefix}: endpoint must be an http(s) URL, got {self.endpoint!r}")
        if self.kind == KIND_ANTHROPIC_API and not self.api_key_env:
            raise ValueError(f"{prefix}: the Anthropic API needs api_key_env (the variable holding the key)")

    @property
    def is_cli(self) -> bool:
        return self.kind in CLI_KINDS

    @property
    def executable(self) -> str:
        """CLI kinds: the command to run."""
        return self.command or DEFAULT_COMMANDS.get(self.kind, "")

    def api_key(self) -> str:
        """The key from the environment, "" if there is none (or the provider needs none)."""
        return os.environ.get(self.api_key_env, "").strip() if self.api_key_env else ""

    def to_dict(self, include_name: bool = True) -> dict:
        data: dict[str, Any] = {"kind": self.kind}
        if self.is_cli:
            optional = {"command": self.command, "model": self.model, "extra_args": list(self.extra_args)}
        else:
            optional = {
                "endpoint": self.endpoint,
                "model": self.model,
                "api_key_env": self.api_key_env,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        data.update({key: value for key, value in optional.items() if value not in ("", None, [])})
        data["timeout_seconds"] = self.timeout_seconds
        return {"name": self.name, **data} if include_name else data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provider":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            # A typo in a hand-edited file ("modle") must not silently fall back to a default.
            raise ValueError(f"Provider '{data.get('name')}': unknown field(s) {', '.join(unknown)}")
        return cls(**dict(data))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def store_path() -> Path:
    """Where providers live: DIARRHIZER_SUMMARY_PROVIDERS_FILE, else <repo root>/summary_providers.json."""
    return store_file.resolve_path(ENV_PROVIDERS_FILE, PROVIDERS_FILENAME)


@dataclass
class ProviderStore:
    """The configured providers and the default provider name."""

    path: Path
    default_name: str = ""
    providers: dict[str, Provider] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ProviderStore":
        """Read the store; a missing file is an empty store, a malformed one raises ValueError."""
        path = store_path() if path is None else Path(path)
        default_name, entries = store_file.read_entries(path, "providers", "summary providers")
        providers = {}
        for name, entry in entries.items():
            try:
                providers[name] = Provider.from_dict({**entry, "name": name})
            except TypeError as e:
                raise ValueError(f"{path}: provider '{name}': {e}") from e
        return cls(path=path, default_name=default_name, providers=providers)

    def names(self) -> list[str]:
        return list(self.providers)

    def get(self, name: str) -> Provider:
        if name not in self.providers:
            raise ValueError(
                f"Unknown summary provider '{name}'. "
                + (f"Available: {', '.join(self.providers)}" if self.providers else self._empty_hint())
            )
        return self.providers[name]

    def default(self) -> Provider | None:
        """The default provider; the first one if the default name is unset or gone; None if there are none."""
        if self.default_name in self.providers:
            return self.providers[self.default_name]
        return next(iter(self.providers.values()), None)

    def resolve(self, name: str | None) -> Provider:
        """The named provider, or the default one for None.

        Raises:
            ValueError: unknown name, or no providers configured at all
        """
        if name is not None:
            return self.get(name)
        provider = self.default()
        if provider is None:
            raise ValueError(f"No summary provider is configured. {self._empty_hint()}")
        return provider

    def put(self, provider: Provider) -> None:
        """Add or replace a provider (in memory - call save() to persist)."""
        self.providers[provider.name] = provider
        if not self.default_name:
            self.default_name = provider.name

    def remove(self, name: str) -> None:
        self.providers.pop(name, None)
        if self.default_name == name:
            self.default_name = next(iter(self.providers), "")

    def set_default(self, name: str) -> None:
        self.get(name)
        self.default_name = name

    def save(self) -> None:
        write_json_atomic(
            self.path,
            {
                "default": self.default_name,
                "providers": {
                    name: provider.to_dict(include_name=False) for name, provider in self.providers.items()
                },
            },
        )

    def _empty_hint(self) -> str:
        return f"Add one in the GUI (Саммаризация → Провайдеры) or in {self.path}"
# [SEMANTIC-END] CONFIG:SUMMARY_PROVIDERS
