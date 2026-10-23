"""A stand-in for the vendors' tools (`codex app-server`, `claude`), shared by the tests that need one.

Used by test_llm_accounts.py (in-process) and test_summary_gui.py (in a really spawned child process).
"""

import json
import sys

FAKE_VENDOR = '''
import json, os, sys, time

tool, args = sys.argv[1], sys.argv[2:]
STATE, LOG = os.environ["FAKE_STATE"], os.environ["FAKE_LOG"]
LOGIN = os.environ.get("FAKE_LOGIN", "success")  # success | fail | hang
NL = chr(10)


def load():
    try:
        return json.load(open(STATE))
    except OSError:
        return {"account": None}


def save(state):
    json.dump(state, open(STATE, "w"))


def log(**record):
    open(LOG, "a").write(json.dumps(record) + NL)


def out(obj):
    print(json.dumps(obj), flush=True)


log(started=os.getpid(), tool=tool, args=args, env_keys=sorted(os.environ))
if os.environ.get("FAKE_CRASH"):
    sys.stderr.write("fatal: cannot open the credentials store" + NL)
    sys.exit(3)
if os.environ.get("FAKE_REJECT_ISOLATION") and "--safe-mode" in args:
    sys.stderr.write("error: unknown option --safe-mode" + NL)
    sys.exit(1)

CHATGPT = {"type": "chatgpt", "email": "me@example.com", "planType": "plus"}
WINDOW = {"usedPercent": 18, "windowDurationMins": 10080, "resetsAt": 1793293782}
FIVE_HOURS = {"usedPercent": 40, "windowDurationMins": 300, "resetsAt": 1793200000}
LIMITS = {"limitId": "codex", "limitName": None, "primary": WINDOW, "secondary": FIVE_HOURS,
          "credits": {"hasCredits": False, "unlimited": False, "balance": "0"}}

if tool == "codex":
    state = load()
    for line in sys.stdin:
        msg = json.loads(line)
        log(received=msg)
        if "method" not in msg:
            continue  # the answer to a request this tool made
        method, mid = msg["method"], msg.get("id")
        params = msg.get("params") or {}
        if method == "initialize":
            out({"id": 900, "method": "attestation/generate", "params": {}})  # a server request: refuse it
            out({"id": mid, "result": {"userAgent": "fake", "codexHome": "x"}})
        elif method == "initialized":
            pass
        elif method == "account/read":
            out({"id": mid, "result": {"account": state["account"], "requiresOpenaiAuth": True}})
        elif method == "account/login/start":
            out({"id": mid, "result": {"type": "chatgpt", "loginId": "L1",
                                       "authUrl": "https://auth.example/authorize?state=1"}})
            if LOGIN == "success":
                state["account"] = CHATGPT
                save(state)
                out({"method": "account/login/completed", "params": {"loginId": "L1", "success": True, "error": None}})
            elif LOGIN == "fail":
                out({"method": "account/login/completed",
                     "params": {"loginId": "L1", "success": False, "error": "access denied"}})
        elif method == "account/login/cancel":
            out({"id": mid, "result": {"status": "canceled"}})
        elif method == "account/logout":
            state["account"] = None
            save(state)
            out({"id": mid, "result": {}})
        elif method == "model/list":
            if not params.get("cursor"):
                out({"id": mid, "result": {"data": [
                    {"id": "gpt-6-astra", "model": "gpt-6-astra", "displayName": "GPT-6-Astra",
                     "description": "Frontier", "hidden": False, "isDefault": True},
                    {"id": "internal", "model": "internal", "displayName": "Internal", "hidden": True,
                     "isDefault": False}], "nextCursor": "page2"}})
            else:
                out({"id": mid, "result": {"data": [
                    {"id": "gpt-6-sol", "model": "gpt-6-sol", "displayName": "GPT-6-Sol",
                     "description": "Previous generation", "hidden": False, "isDefault": False}],
                    "nextCursor": None}})
        elif method == "account/rateLimits/read":
            spark = {"limitId": "codex_spark", "limitName": "GPT-Spark", "primary": FIVE_HOURS, "secondary": None,
                     "credits": None}
            out({"id": mid, "result": {"ordinaryUsageAllowed": True, "rateLimits": LIMITS,
                                       "rateLimitsByLimitId": {"codex": LIMITS, "codex_spark": spark}}})
        else:
            out({"id": mid, "error": {"code": -32601, "message": "unknown method " + str(method)}})
    sys.exit(0)

# --- claude ---
if args[:2] == ["auth", "status"]:
    if load()["account"]:
        out({"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "email": "me@example.com",
             "subscriptionType": "max", "orgName": "Example Org"})
        sys.exit(0)
    out({"loggedIn": False, "authMethod": "none", "apiProvider": "firstParty"})
    sys.exit(1)  # nobody signed in: JSON on stdout, non-zero exit
if args[:2] == ["auth", "login"]:
    print("Opening browser to sign in...", flush=True)
    print("If the browser didn't open, visit: https://claude.example/oauth/authorize?x=1", flush=True)
    if LOGIN == "success":
        time.sleep(0.5)
        state = load()
        state["account"] = {"signed": True}
        save(state)
        print("Login successful.", flush=True)
        sys.exit(0)
    if LOGIN == "fail":
        print("Login failed: Request failed with status code 400", flush=True)
        sys.exit(1)
    time.sleep(120)
if args[:2] == ["auth", "logout"]:
    state = load()
    state["account"] = None
    save(state)
    print("Logged out.", flush=True)
    sys.exit(0)
if "--input-format" in args:  # the stream-json control protocol
    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("type") == "control_request" and msg["request"]["subtype"] == "initialize":
            out({"type": "control_response", "response": {"subtype": "success", "request_id": msg["request_id"],
                 "response": {"commands": [], "account": {"tokenSource": "none"}, "models": [
                     {"value": "default", "resolvedModel": "claude-opus-5[1m]", "displayName": "Default (recommended)",
                      "description": "Use the default model (currently Opus 5) - $5/$25 per Mtok"},
                     {"value": "sonnet", "resolvedModel": "claude-sonnet-5", "displayName": "Sonnet",
                      "description": "Sonnet 5 - $2/$10 per Mtok"}]}}})
    sys.exit(0)
sys.stdin.read()  # a request: one word back, with the rate-limit status the way Claude Code reports it
out({"type": "system", "subtype": "init", "model": "claude-sonnet-5"})
out({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "utilization": 0.4,
                                                     "resetsAt": 1793200000, "rateLimitType": "five_hour"}})
out({"type": "result", "subtype": "success", "is_error": False, "result": "ok", "total_cost_usd": 0.001,
     "usage": {"input_tokens": 5, "output_tokens": 1}})
'''


def make_vendor(tmp_path, monkeypatch):
    """Stand-in `codex` / `claude` executables. `vendor.make(name)` -> path; `vendor.log()` -> what they
    received and ran; `vendor.signed_in()` / `vendor.sign_in()` read and set the shared account state."""

    script = tmp_path / "fake_vendor.py"
    script.write_text(FAKE_VENDOR, encoding="utf-8")
    state_path, log_path = tmp_path / "state.json", tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_STATE", str(state_path))
    monkeypatch.setenv("FAKE_LOG", str(log_path))

    class Vendor:
        @staticmethod
        def make(name: str) -> str:
            if sys.platform == "win32":
                launcher = tmp_path / f"{name}.cmd"
                launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" {name} %*\r\n')
            else:
                launcher = tmp_path / name
                launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" {name} "$@"\n')
                launcher.chmod(0o755)
            return str(launcher)

        @staticmethod
        def log() -> list:
            if not log_path.exists():
                return []
            return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        @staticmethod
        def signed_in() -> bool:
            return state_path.exists() and bool(json.loads(state_path.read_text())["account"])

        @staticmethod
        def sign_in() -> None:
            state_path.write_text(json.dumps({"account": {"type": "chatgpt", "email": "me@example.com",
                                                          "planType": "plus"}}))

        @staticmethod
        def received(method: str) -> list:
            return [r["received"] for r in Vendor.log() if r.get("received", {}).get("method") == method]

    return Vendor
