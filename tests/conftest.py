"""Shared fixtures for the GUI tests."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class FakeApi:
    """A local stand-in for an HTTP API: records every request, answers each with the configured status/body."""

    def __init__(self) -> None:
        self.requests: list = []
        self.status = 200
        self.body: object = {}
        self.delay = 0.0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self, body):
                outer.requests.append(
                    {"method": self.command, "path": self.path,
                     "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body}
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

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self._answer(json.loads(self.rfile.read(length)))

            def do_GET(self):
                self._answer(None)

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
    """A FakeApi; a system proxy must not swallow requests to it."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    server = FakeApi()
    yield server
    server.close()


@pytest.fixture(scope="session")
def qapp():
    """One offscreen QApplication for every Qt test in the session (widgets
    need a QApplication, not just a QCoreApplication, and Qt allows only one)."""
    pytest.importorskip("PySide6", reason="GUI extra not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from diarrhizer_gui.child_process import kill_all

    app = QApplication.instance() or QApplication([])
    yield app
    # Child processes are not daemons: a test that failed mid-way must not
    # leave one running (or the interpreter would wait for it at exit).
    kill_all()


@pytest.fixture(autouse=True)
def summary_stores(tmp_path, monkeypatch):
    """Never read or write the developer's real <repo>/summary_templates.json
    and summary_providers.json. Returns (templates path, providers path)."""
    from diarrhizer.summary.providers import ENV_PROVIDERS_FILE
    from diarrhizer.summary.templates import ENV_TEMPLATES_FILE

    templates = tmp_path / "summary_templates.json"
    providers = tmp_path / "summary_providers.json"
    monkeypatch.setenv(ENV_TEMPLATES_FILE, str(templates))
    monkeypatch.setenv(ENV_PROVIDERS_FILE, str(providers))
    return templates, providers


@pytest.fixture
def real_plan_limits():
    """Opt-in for the tests of service._plan_limits itself (see no_plan_limits_lookup)."""


@pytest.fixture(autouse=True)
def no_plan_limits_lookup(request, monkeypatch):
    """summarize_job asks Codex's app-server for the plan limits once a request is done. A test that fakes
    the model must not start the real `codex` for that (it would read the developer's actual account), so
    the lookup is off unless the test asks for `real_plan_limits`."""
    if "real_plan_limits" in request.fixturenames:
        return
    from diarrhizer.summary import service

    monkeypatch.setattr(service, "_plan_limits", lambda provider: None)


@pytest.fixture
def ini_settings(tmp_path, monkeypatch):
    """Point every screen's QSettings("Diarrhizer", "DiarrhizerGUI") at a
    throwaway INI file, so tests never touch the real settings store."""
    pytest.importorskip("PySide6", reason="GUI extra not installed")
    from PySide6.QtCore import QSettings

    path = str(tmp_path / "gui-settings.ini")

    def factory(*_args, **_kwargs):
        return QSettings(path, QSettings.Format.IniFormat)

    return path, factory
