"""Tests for the summary configuration: prompt templates (CONFIG:SUMMARY_TEMPLATES)
and model providers (CONFIG:SUMMARY_PROVIDERS) and the JSON stores behind them.
"""

import json

import pytest

from diarrhizer.env_file import REPO_ROOT
from diarrhizer.summary.providers import (
    DEFAULT_COMMANDS,
    DEFAULT_TIMEOUT_SECONDS,
    ENV_PROVIDERS_FILE,
    KIND_ANTHROPIC_API,
    KIND_CLAUDE_CLI,
    KIND_CODEX_CLI,
    KIND_OPENAI_API,
    KINDS,
    Provider,
    ProviderStore,
    default_key_env,
)
from diarrhizer.summary.templates import (
    BUILTIN_TEMPLATES,
    DEFAULT_TEMPLATE_NAME,
    ENV_TEMPLATES_FILE,
    PromptTemplate,
    TemplateStore,
)


def _api(**overrides) -> Provider:
    fields = dict(
        name="my-api", kind=KIND_OPENAI_API, endpoint="https://api.example.com/v1", model="m1",
    )
    return Provider(**{**fields, **overrides})


# --- PromptTemplate ------------------------------------------------------------


def test_template_names_may_contain_spaces_and_cyrillic():
    template = PromptTemplate(name="Итоги недели", prompt="Подведи итоги.")
    assert template.to_dict() == {"name": "Итоги недели", "description": "", "prompt": "Подведи итоги."}


@pytest.mark.parametrize("name", ["", "   ", " padded", "two\nlines", "x" * 81])
def test_template_rejects_a_bad_name(name):
    with pytest.raises(ValueError, match="name"):
        PromptTemplate(name=name, prompt="p")


def test_template_needs_a_prompt():
    with pytest.raises(ValueError, match="prompt"):
        PromptTemplate(name="t", prompt="  \n ")


def test_template_from_dict_rejects_a_typo_instead_of_dropping_the_prompt():
    with pytest.raises(ValueError, match="promt"):
        PromptTemplate.from_dict({"name": "t", "promt": "x"})


# --- TemplateStore -------------------------------------------------------------


def test_missing_template_file_leaves_only_the_builtins(summary_stores):
    store = TemplateStore.load()
    assert store.path == summary_stores[0]
    assert store.names() == list(BUILTIN_TEMPLATES)
    assert store.default().name == DEFAULT_TEMPLATE_NAME
    assert store.resolve(None) is store.default()


def test_template_store_round_trips_and_keeps_builtins_first(summary_stores):
    store = TemplateStore.load()
    store.put(PromptTemplate(name="Мой", prompt="Сделай список.", description="d"))
    store.set_default("Мой")
    store.save()

    again = TemplateStore.load()
    assert again.names() == [*BUILTIN_TEMPLATES, "Мой"]
    assert again.default().prompt == "Сделай список."
    assert again.get("Мой").description == "d"
    # Cyrillic is stored readably, not as \u escapes.
    assert "Сделай список." in summary_stores[0].read_text(encoding="utf-8")


def test_builtin_templates_cannot_be_overwritten_or_deleted():
    store = TemplateStore.load()
    with pytest.raises(ValueError, match="built-in"):
        store.put(PromptTemplate(name=DEFAULT_TEMPLATE_NAME, prompt="x"))
    with pytest.raises(ValueError, match="built-in"):
        store.remove(DEFAULT_TEMPLATE_NAME)


def test_deleting_the_default_template_falls_back_to_the_builtin_default():
    store = TemplateStore.load()
    store.put(PromptTemplate(name="Мой", prompt="x"))
    store.set_default("Мой")
    store.remove("Мой")
    assert store.default().name == DEFAULT_TEMPLATE_NAME


def test_a_default_naming_a_vanished_template_is_not_fatal(summary_stores):
    summary_stores[0].write_text(json.dumps({"default": "gone", "templates": {}}), encoding="utf-8")
    assert TemplateStore.load().default().name == DEFAULT_TEMPLATE_NAME


def test_unknown_template_lists_the_available_ones():
    with pytest.raises(ValueError, match="Available: Краткое резюме"):
        TemplateStore.load().get("nope")


@pytest.mark.parametrize(
    "content, message",
    [
        ("{not json", "Cannot read"),
        ('["a"]', "must be an object"),
        ('{"templates": []}', "must be an object"),
        ('{"templates": {"a": "str"}}', "must be an object"),
        (json.dumps({"templates": {DEFAULT_TEMPLATE_NAME: {"prompt": "x"}}}), "built-in"),
        ('{"templates": {"a": {"prompt": "x", "bogus": 1}}}', "bogus"),
        ('{"templates": {"a": {"description": "no prompt"}}}', "prompt"),
    ],
)
def test_a_malformed_template_file_raises_and_names_the_file(summary_stores, content, message):
    summary_stores[0].write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        TemplateStore.load()


def test_template_store_path_follows_the_env_override(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere.json"
    monkeypatch.setenv(ENV_TEMPLATES_FILE, str(other))
    assert TemplateStore.load().path == other


# --- Provider ------------------------------------------------------------------


def test_cli_providers_need_nothing_but_a_kind():
    provider = Provider(name="ChatGPT", kind=KIND_CODEX_CLI)
    assert provider.is_cli
    assert provider.executable == DEFAULT_COMMANDS[KIND_CODEX_CLI] == "codex"
    assert Provider(name="c", kind=KIND_CLAUDE_CLI, command=r"C:\tools\claude.exe").executable == r"C:\tools\claude.exe"
    assert provider.timeout_seconds == DEFAULT_TIMEOUT_SECONDS


def test_api_providers_need_a_model_and_an_http_endpoint():
    with pytest.raises(ValueError, match="model"):
        _api(model="")
    with pytest.raises(ValueError, match="endpoint"):
        _api(endpoint="")
    with pytest.raises(ValueError, match="http"):
        _api(endpoint="ftp://example.com")


def test_the_anthropic_api_needs_a_key_variable_but_an_openai_server_may_not():
    with pytest.raises(ValueError, match="api_key_env"):
        Provider(name="a", kind=KIND_ANTHROPIC_API, endpoint="https://api.anthropic.com", model="m")
    local = _api(endpoint="http://localhost:11434/v1", api_key_env="")  # Ollama needs no key
    assert local.api_key() == ""


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "gemini"},
        {"api_key_env": "has space"},
        {"api_key_env": "1STARTS_WITH_DIGIT"},
        {"timeout_seconds": 0},
        {"timeout_seconds": True},
        {"temperature": 3},
        {"temperature": "hot"},
        {"max_tokens": 0},
        {"name": " padded"},
        {"name": ""},
    ],
)
def test_provider_rejects_bad_values(overrides):
    with pytest.raises(ValueError):
        _api(**overrides)


def test_prices_are_optional_and_turn_token_counts_into_a_cost_estimate():
    priced = _api(price_input=0.15, price_output=0.60)
    assert priced.estimate_cost(2_000_000, 500_000) == pytest.approx(2 * 0.15 + 0.5 * 0.60)
    assert _api().estimate_cost(1000, 1000) is None  # no prices, no guess
    assert _api(price_input=1.0).estimate_cost(1_000_000, 5_000_000) == pytest.approx(1.0)  # output priced at 0
    free = _api(price_input=0, price_output=0)  # a local model: a real price of zero, not "unknown"
    assert free.estimate_cost(10**9, 10**9) == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"price_input": -1},
        {"price_input": "cheap"},
        {"price_input": True},
        {"price_output": 1.0},  # an output price without an input price
        {"price_input": 1.0, "price_output": -0.1},
    ],
)
def test_bad_prices_are_rejected(overrides):
    with pytest.raises(ValueError, match="price"):
        _api(**overrides)


def test_prices_are_stored_for_api_providers_only():
    priced = _api(price_input=0.15, price_output=0.60)
    assert priced.to_dict(include_name=False)["price_input"] == 0.15
    assert Provider.from_dict(priced.to_dict()) == priced
    assert "price_input" not in _api().to_dict()  # unset prices are not written
    assert "price_input" not in Provider(name="c", kind=KIND_CODEX_CLI).to_dict()  # a subscription reports its own


def test_the_quotes_of_explorers_copy_as_path_are_not_part_of_the_command():
    pasted = Provider(name="c", kind=KIND_CODEX_CLI, command='  "C:\\Program Files\\OpenAI\\codex.exe" ')
    assert pasted.command == "C:\\Program Files\\OpenAI\\codex.exe"
    assert pasted.executable == "C:\\Program Files\\OpenAI\\codex.exe"
    assert pasted.to_dict()["command"] == "C:\\Program Files\\OpenAI\\codex.exe"  # stored without them
    assert Provider(name="c", kind=KIND_CODEX_CLI, command='""').executable == "codex"  # nothing left: the default


def test_provider_reads_its_key_from_the_environment(monkeypatch):
    provider = _api(api_key_env="MY_KEY")
    assert provider.api_key() == ""
    monkeypatch.setenv("MY_KEY", "  sk-test \n")
    assert provider.api_key() == "sk-test"


def test_provider_dict_holds_only_what_matters_for_its_kind_and_never_a_key():
    cli = Provider(name="c", kind=KIND_CLAUDE_CLI, model="sonnet", extra_args=("--effort", "low"))
    assert cli.to_dict(include_name=False) == {
        "kind": "claude-cli", "model": "sonnet", "extra_args": ["--effort", "low"],
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
    }
    api = _api(api_key_env="K", temperature=0.0, max_tokens=500)
    assert api.to_dict(include_name=False) == {
        "kind": "openai-api", "endpoint": "https://api.example.com/v1", "model": "m1",
        "api_key_env": "K", "temperature": 0.0, "max_tokens": 500,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
    }
    assert Provider.from_dict(api.to_dict()) == api
    assert Provider.from_dict(cli.to_dict()) == cli


def test_provider_from_dict_rejects_a_typo():
    with pytest.raises(ValueError, match="modle"):
        Provider.from_dict({"name": "a", "kind": KIND_OPENAI_API, "modle": "x"})


def test_default_key_env_is_a_valid_ascii_variable_name():
    assert default_key_env("OpenRouter") == "DIARRHIZER_LLM_KEY_OPENROUTER"
    assert default_key_env("my api (2)") == "DIARRHIZER_LLM_KEY_MY_API_2"
    assert default_key_env("Свой сервер") == "DIARRHIZER_LLM_KEY_PROVIDER"
    _api(api_key_env=default_key_env("Свой сервер"))  # accepted by validation


# --- ProviderStore -------------------------------------------------------------


def test_missing_provider_file_is_an_empty_store_that_explains_itself(summary_stores):
    store = ProviderStore.load()
    assert store.names() == []
    assert store.default() is None
    with pytest.raises(ValueError, match="No summary provider is configured") as excinfo:
        store.resolve(None)
    assert str(summary_stores[1]) in str(excinfo.value)


def test_the_first_provider_becomes_the_default_and_survives_a_round_trip(summary_stores):
    store = ProviderStore.load()
    store.put(Provider(name="ChatGPT", kind=KIND_CODEX_CLI))
    store.put(_api(api_key_env="MY_KEY"))
    store.save()

    again = ProviderStore.load()
    assert again.names() == ["ChatGPT", "my-api"]
    assert again.default().name == "ChatGPT"
    assert again.resolve("my-api") == _api(api_key_env="MY_KEY")
    assert "MY_KEY" in summary_stores[1].read_text(encoding="utf-8")


def test_removing_the_default_provider_promotes_another():
    store = ProviderStore.load()
    store.put(Provider(name="a", kind=KIND_CODEX_CLI))
    store.put(Provider(name="b", kind=KIND_CLAUDE_CLI))
    store.remove("a")
    assert store.default().name == "b"
    store.remove("b")
    assert store.default() is None and store.default_name == ""


def test_set_default_rejects_an_unknown_provider():
    store = ProviderStore.load()
    store.put(Provider(name="a", kind=KIND_CODEX_CLI))
    with pytest.raises(ValueError, match="Unknown summary provider 'zzz'. Available: a"):
        store.set_default("zzz")


def test_a_default_naming_a_deleted_provider_falls_back_to_the_first(summary_stores):
    summary_stores[1].write_text(
        json.dumps({"default": "gone", "providers": {"a": {"kind": "codex-cli"}}}), encoding="utf-8"
    )
    assert ProviderStore.load().default().name == "a"


@pytest.mark.parametrize(
    "content, message",
    [
        ("{not json", "Cannot read"),
        ('{"providers": []}', "must be an object"),
        ('{"providers": {"a": {"kind": "gemini"}}}', "kind"),
        ('{"providers": {"a": {"kind": "openai-api", "model": "m"}}}', "endpoint"),
        ('{"providers": {"a": {"kind": "codex-cli", "modle": "m"}}}', "modle"),
    ],
)
def test_a_malformed_provider_file_raises_and_names_the_file(summary_stores, content, message):
    summary_stores[1].write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        ProviderStore.load()


def test_provider_store_path_follows_the_env_override(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere.json"
    monkeypatch.setenv(ENV_PROVIDERS_FILE, str(other))
    assert ProviderStore.load().path == other


def test_example_providers_file_is_a_valid_store_of_every_kind():
    """summary_providers.example.json is what the README tells people to copy."""
    store = ProviderStore.load(REPO_ROOT / "summary_providers.example.json")

    assert store.default().name == "ChatGPT"
    assert {provider.kind for provider in store.providers.values()} == set(KINDS)
    # The example documents the variable name only; it must never hold a key.
    assert all(not provider.api_key() for provider in store.providers.values())
    assert "sk-" not in (REPO_ROOT / "summary_providers.example.json").read_text(encoding="utf-8")


def test_personal_summary_files_are_git_ignored_but_the_example_is_not():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "summary_providers.json" in ignored and "summary_templates.json" in ignored
    assert "summary_providers.example.json" not in ignored
    assert ".env" in ignored  # where the API keys go
