"""Tests for adapters.llm_accounts (ADAPTER:LLM_ACCOUNTS): signing in and out, the account, the model
list and the plan limits of the subscription providers - against stand-in executables that speak
`codex app-server` (JSON-RPC lines) and `claude` (auth commands, stream-json control protocol) - and
the model list of API providers against a local server. No real account is touched.
"""

import json
import subprocess
import sys
import time

import pytest
from vendor_stub import make_vendor

from diarrhizer.adapters import llm, llm_accounts
from diarrhizer.adapters.llm import LLMError
from diarrhizer.adapters.llm_accounts import LoginCancelled
from diarrhizer.summary.provider_info import Account, ModelOption
from diarrhizer.summary.providers import (
    KIND_ANTHROPIC_API,
    KIND_CLAUDE_CLI,
    KIND_CODEX_CLI,
    KIND_OPENAI_API,
    Provider,
)



@pytest.fixture
def vendor(tmp_path, monkeypatch):
    """Stand-in `codex` / `claude` executables (tests/vendor_stub.py)."""
    return make_vendor(tmp_path, monkeypatch)


def _codex(vendor, **overrides) -> Provider:
    return Provider(**{**dict(name="ChatGPT", kind=KIND_CODEX_CLI, command=vendor.make("codex"),
                              timeout_seconds=30), **overrides})


def _claude(vendor, **overrides) -> Provider:
    return Provider(**{**dict(name="Claude", kind=KIND_CLAUDE_CLI, command=vendor.make("claude"),
                              timeout_seconds=30), **overrides})


# --- Codex: account -------------------------------------------------------------------------


def test_codex_account_is_read_from_the_app_server(vendor):
    assert llm_accounts.account_status(_codex(vendor)) == Account(signed_in=False)

    vendor.sign_in()
    account = llm_accounts.account_status(_codex(vendor))

    assert account == Account(signed_in=True, email="me@example.com", plan="plus", method="ChatGPT")


def test_codex_signed_in_with_an_api_key_is_not_called_a_subscription(vendor):
    assert llm_accounts._codex_account({"account": {"type": "apiKey"}}) == Account(signed_in=True, method="API key")


def test_the_client_introduces_itself_and_refuses_what_the_server_asks_of_it(vendor):
    llm_accounts.account_status(_codex(vendor))

    (init,) = vendor.received("initialize")
    assert init["params"]["clientInfo"]["name"] == "diarrhizer"
    assert vendor.received("initialized")  # the handshake is completed before any other request
    refusals = [r["received"] for r in vendor.log() if r.get("received", {}).get("id") == 900]
    assert refusals and "error" in refusals[0]  # a request from the server is answered, not left hanging


def test_codex_started_with_the_subscription_environment_only(vendor, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-would-switch-the-billing")
    llm_accounts.account_status(_codex(vendor))
    (started,) = [r for r in vendor.log() if "started" in r]
    assert "OPENAI_API_KEY" not in started["env_keys"]


def test_a_crashing_tool_is_reported_with_what_it_said(vendor, monkeypatch):
    monkeypatch.setenv("FAKE_CRASH", "1")
    with pytest.raises(LLMError, match="cannot open the credentials store"):
        llm_accounts.account_status(_codex(vendor))


def test_a_missing_tool_is_explained(tmp_path):
    provider = Provider(name="ChatGPT", kind=KIND_CODEX_CLI, command=str(tmp_path / "no-such-codex"))
    with pytest.raises(LLMError, match="not found"):
        llm_accounts.account_status(provider)


# --- Codex: sign-in -----------------------------------------------------------------------------


def test_codex_login_hands_over_the_sign_in_address_and_returns_the_account(vendor):
    events = []

    account = llm_accounts.login(_codex(vendor), on_event=lambda name, payload: events.append((name, payload)))

    assert events == [("url", "https://auth.example/authorize?state=1")]  # the GUI opens this in the browser
    assert account == Account(signed_in=True, email="me@example.com", plan="plus", method="ChatGPT")
    assert vendor.received("account/login/start")[0]["params"] == {"type": "chatgpt"}
    assert vendor.signed_in()


def test_codex_login_failure_carries_the_reason(vendor, monkeypatch):
    monkeypatch.setenv("FAKE_LOGIN", "fail")
    with pytest.raises(LLMError, match="access denied"):
        llm_accounts.login(_codex(vendor))
    assert not vendor.signed_in()


def test_codex_login_can_be_cancelled_and_the_pending_sign_in_is_withdrawn(vendor, monkeypatch):
    monkeypatch.setenv("FAKE_LOGIN", "hang")
    asked = []

    def should_cancel():
        asked.append(1)
        return len(asked) >= 2  # the user gave up a moment after the address was shown

    with pytest.raises(LoginCancelled):
        llm_accounts.login(_codex(vendor), should_cancel=should_cancel)

    (cancel,) = vendor.received("account/login/cancel")
    assert cancel["params"] == {"loginId": "L1"}  # so the tool's local callback server is released
    assert not vendor.signed_in()


def test_codex_login_gives_up_after_the_timeout(vendor, monkeypatch):
    monkeypatch.setenv("FAKE_LOGIN", "hang")
    monkeypatch.setattr(llm_accounts, "LOGIN_TIMEOUT_SECONDS", 1)
    with pytest.raises(LLMError, match="not finished within 1 s"):
        llm_accounts.login(_codex(vendor))
    assert vendor.received("account/login/cancel")


def test_codex_logout(vendor):
    vendor.sign_in()
    assert llm_accounts.logout(_codex(vendor)) == Account(signed_in=False)
    assert not vendor.signed_in()


# --- Codex: models and limits ---------------------------------------------------------------------


def test_codex_models_come_from_the_provider_page_by_page_without_the_hidden_ones(vendor):
    models = llm_accounts.list_models(_codex(vendor))

    assert models == [
        ModelOption(id="gpt-6-astra", name="GPT-6-Astra", description="Frontier", default=True),
        ModelOption(id="gpt-6-sol", name="GPT-6-Sol", description="Previous generation"),
    ]
    assert [r["params"] for r in vendor.received("model/list")] == [{}, {"cursor": "page2"}]


def test_codex_limits_list_every_window_the_credits_and_whether_usage_is_allowed(vendor):
    limits = llm_accounts.read_limits(_codex(vendor))

    by_label = [(w.label, w.window_minutes, w.used_percent, w.resets_at) for w in limits.windows]
    assert by_label == [
        ("", 10080, 18.0, 1793293782.0),  # the main limit has no name of its own
        ("", 300, 40.0, 1793200000.0),
        ("GPT-Spark", 300, 40.0, 1793200000.0),  # an extra limit carries its name
    ]
    assert limits.credits == "none" and limits.allowed is True


@pytest.mark.parametrize(
    "credits, expected",
    [
        ({"hasCredits": True, "unlimited": False, "balance": "12.50"}, "12.50"),
        ({"hasCredits": True, "unlimited": True, "balance": None}, "unlimited"),
        ({"hasCredits": False, "unlimited": False, "balance": "0"}, "none"),
    ],
)
def test_codex_credits_are_described_without_inventing_a_balance(credits, expected):
    limits = llm_accounts._codex_limits({"rateLimits": {"limitId": "codex", "credits": credits}})
    assert limits.credits == expected and limits.windows == ()


# --- Claude ---------------------------------------------------------------------------------------


def test_claude_account_is_read_from_auth_status_even_when_it_exits_non_zero(vendor):
    assert llm_accounts.account_status(_claude(vendor)) == Account(signed_in=False)  # the fake exits 1 here

    vendor.sign_in()
    account = llm_accounts.account_status(_claude(vendor))

    assert account == Account(signed_in=True, email="me@example.com", plan="max", method="claude.ai",
                              organization="Example Org")


def test_claude_signed_in_through_a_cloud_is_labelled_with_it_not_as_a_claude_account(monkeypatch):
    """auth status says which service the sign-in is for; only "firstParty" is a Claude account."""
    status = {"loggedIn": True, "authMethod": "cloud", "apiProvider": "bedrock"}
    monkeypatch.setattr(llm_accounts, "_run_claude", lambda provider, args: (json.dumps(status), ""))

    account = llm_accounts.account_status(Provider(name="c", kind=KIND_CLAUDE_CLI))

    assert account.signed_in and account.method == "bedrock"


def test_unreadable_auth_status_output_is_reported_not_swallowed(monkeypatch):
    monkeypatch.setattr(llm_accounts, "_run_claude", lambda provider, args: ("Segmentation fault", ""))
    with pytest.raises(LLMError, match="unexpected output of `claude auth status`: Segmentation fault"):
        llm_accounts.account_status(Provider(name="c", kind=KIND_CLAUDE_CLI))


def test_claude_login_runs_the_vendors_own_flow_and_reports_the_account(vendor):
    events = []

    account = llm_accounts.login(_claude(vendor), on_event=lambda name, payload: events.append(name))

    assert events == ["waiting"]  # the CLI opens the browser itself: the GUI only has to wait
    assert account.signed_in and account.plan == "max"
    (run,) = [r for r in vendor.log() if r.get("args", [])[:2] == ["auth", "login"]]
    assert run["args"] == ["auth", "login", "--claudeai"]  # the subscription, not the API console


def test_claude_login_failure_shows_the_tools_own_words(vendor, monkeypatch):
    monkeypatch.setenv("FAKE_LOGIN", "fail")
    with pytest.raises(LLMError, match="Login failed: Request failed with status code 400"):
        llm_accounts.login(_claude(vendor))


def test_cancelling_a_claude_login_kills_the_tool(vendor, monkeypatch):
    monkeypatch.setenv("FAKE_LOGIN", "hang")
    started = time.monotonic()
    ticks = []

    def should_cancel():
        ticks.append(1)
        return len(ticks) >= 3

    with pytest.raises(LoginCancelled):
        llm_accounts.login(_claude(vendor), should_cancel=should_cancel)

    assert time.monotonic() - started < 30  # not the tool's two-minute wait
    if sys.platform == "win32":
        pid = [r["started"] for r in vendor.log() if r.get("args", [])[:2] == ["auth", "login"]][0]
        time.sleep(0.5)
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        assert str(pid) not in listing, "the sign-in process outlived the cancelled sign-in"


def test_claude_logout_then_status(vendor):
    vendor.sign_in()
    assert llm_accounts.logout(_claude(vendor)) == Account(signed_in=False)
    assert not vendor.signed_in()


def test_claude_models_come_from_the_control_protocol_with_the_default_as_an_empty_id(vendor):
    models = llm_accounts.list_models(_claude(vendor))

    assert models == [
        ModelOption(id="", name="Default (recommended)", default=True, resolved="claude-opus-5[1m]",
                    description="Use the default model (currently Opus 5) - $5/$25 per Mtok"),
        ModelOption(id="sonnet", name="Sonnet", resolved="claude-sonnet-5", description="Sonnet 5 - $2/$10 per Mtok"),
    ]
    (run,) = [r for r in vendor.log() if "--input-format" in r.get("args", [])]
    assert "--safe-mode" in run["args"] and "--no-session-persistence" in run["args"]


def test_claude_models_are_retried_without_isolation_flags_the_installed_version_lacks(vendor, monkeypatch):
    monkeypatch.setenv("FAKE_REJECT_ISOLATION", "1")

    models = llm_accounts.list_models(_claude(vendor))

    assert [m.id for m in models] == ["", "sonnet"]
    runs = [r for r in vendor.log() if "--input-format" in r.get("args", [])]
    assert "--safe-mode" in runs[0]["args"] and "--safe-mode" not in runs[1]["args"]


def test_claude_has_no_limits_query_so_a_one_word_request_is_made(vendor):
    limits = llm_accounts.read_limits(_claude(vendor))

    (window,) = limits.windows
    assert (window.status, window.window_minutes, window.used_percent) == ("allowed", 300, pytest.approx(40))
    (run,) = [r for r in vendor.log() if r.get("args", [])[:1] == ["-p"]]
    assert "--output-format" in run["args"]


# --- API providers -----------------------------------------------------------------------------------


def test_openai_models_are_listed_sorted_from_the_base_url_even_if_the_endpoint_is_the_full_one(api, monkeypatch):
    monkeypatch.setenv("GW_KEY", "sk-secret")
    api.body = {"object": "list", "data": [{"id": "gpt-z"}, {"id": "gpt-a"}, {"id": "gpt-m", "object": "model"}]}
    provider = Provider(name="gw", kind=KIND_OPENAI_API, endpoint=f"{api.url}/v1/chat/completions", model="gpt-a",
                        api_key_env="GW_KEY", timeout_seconds=10)

    models = llm_accounts.list_models(provider)

    assert [m.id for m in models] == ["gpt-a", "gpt-m", "gpt-z"]
    (request,) = api.requests
    assert (request["method"], request["path"]) == ("GET", "/v1/models")
    assert request["headers"]["authorization"] == "Bearer sk-secret"


def test_anthropic_models_keep_the_providers_order_and_names(api, monkeypatch):
    monkeypatch.setenv("ANT_KEY", "ant-secret")
    api.body = {"data": [{"id": "claude-new", "display_name": "Claude New"}, {"id": "claude-old", "display_name": "Claude Old"}]}
    provider = Provider(name="ant", kind=KIND_ANTHROPIC_API, endpoint=api.url, model="claude-new",
                        api_key_env="ANT_KEY", timeout_seconds=10)

    models = llm_accounts.list_models(provider)

    assert models == [ModelOption(id="claude-new", name="Claude New"), ModelOption(id="claude-old", name="Claude Old")]
    (request,) = api.requests
    assert request["path"] == "/v1/models?limit=1000" and request["headers"]["x-api-key"] == "ant-secret"


def test_a_server_that_refuses_the_key_gives_a_readable_error(api, monkeypatch):
    monkeypatch.setenv("GW_KEY", "wrong")
    api.status = 401
    api.body = {"error": {"message": "Incorrect API key provided"}}
    provider = Provider(name="gw", kind=KIND_OPENAI_API, endpoint=f"{api.url}/v1", model="m", api_key_env="GW_KEY",
                        timeout_seconds=10)
    with pytest.raises(LLMError, match="HTTP 401.*Incorrect API key"):
        llm_accounts.list_models(provider)


def test_an_api_provider_has_no_account_and_no_plan_limits(api):
    provider = Provider(name="gw", kind=KIND_OPENAI_API, endpoint=f"{api.url}/v1", model="m")
    for action in (llm_accounts.account_status, llm_accounts.login, llm_accounts.logout):
        with pytest.raises(LLMError, match="API provider"):
            action(provider)
    assert llm_accounts.read_limits(provider).windows == ()
    assert "no plan limits" in llm_accounts.read_limits(provider).note
    assert llm.CHECK_PROMPT  # the one-word request Claude's limits are read with
