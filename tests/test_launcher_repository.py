"""Repository snapshots, retry behavior and environment preservation without external network."""

import io
import json
import threading
import zipfile
from urllib.error import URLError

import pytest

from diarrhizer_launcher import repository
from diarrhizer_launcher.config import Settings, State
from diarrhizer_launcher.setup import SetupWorker

SHA = "a" * 40


def archive(extra=None):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as bundle:
        for name, value in {
            "src/diarrhizer/__init__.py": '__version__ = "2.10"',
            "src/diarrhizer_gui/app.py": "", "pyproject.toml": "", "README.md": "", "LICENSE": "",
            "requirements/constraints-stable.txt": "", "requirements/base.txt": "",
            "requirements/cpu.txt": "", "requirements/cuda-cu121.txt": "",
            "recording.wav": "private", ".env": "do not extract", **(extra or {}),
        }.items():
            bundle.writestr("repo-" + SHA + "/" + name, value)
    return content.getvalue()


def responses(monkeypatch, payload):
    urls = []

    def open_url(url):
        urls.append(url)
        stream = io.BytesIO(json.dumps({"sha": SHA}).encode() if "api.github" in url else payload)
        stream.headers = {}
        return stream

    monkeypatch.setattr(repository, "_open", open_url)
    return urls


def test_download_pins_commit_caches_source_and_excludes_data(tmp_path, monkeypatch):
    urls = responses(monkeypatch, archive())
    folder, revision = repository.latest_source(tmp_path, lambda _: None, threading.Event())
    assert revision == SHA
    assert urls[-1].endswith(SHA)
    assert (folder / "src/diarrhizer/__init__.py").read_text() == '__version__ = "2.10"'
    assert not (folder / ".env").exists()
    assert not (folder / "recording.wav").exists()
    cached, _ = repository.latest_source(tmp_path, lambda _: None, threading.Event())
    assert cached == folder
    assert len(urls) == 3  # Recheck main; reuse the immutable source archive.


@pytest.mark.parametrize("extra", [{"../escape.py": "bad"}, {"src/../../escape.py": "bad"}])
def test_archive_traversal_is_rejected_without_publishing_source(tmp_path, monkeypatch, extra):
    responses(monkeypatch, archive(extra))
    with pytest.raises(ValueError, match="Небезопасный"):
        repository.latest_source(tmp_path, lambda _: None, threading.Event())
    assert not (tmp_path / "sources" / SHA).exists()
    assert not list(tmp_path.rglob("escape.py"))


def test_network_failure_preserves_ready_environment_and_does_not_install(tmp_path, monkeypatch):
    folder = tmp_path / "venv"
    folder.mkdir()
    marker = folder / "keep.txt"
    marker.write_text("existing application")
    state = State(Settings(str(folder), str(tmp_path / "models"), str(tmp_path / "out"),
                           str(tmp_path / "config")), ready=True)
    state_file = tmp_path / "launcher/settings.json"
    state.save(state_file)
    monkeypatch.setattr(repository, "_open", lambda _: (_ for _ in ()).throw(URLError("offline")))
    worker = SetupWorker(state, state_file, tmp_path / "bundle", lambda _: None, remote=True)
    worker.command = lambda *_, **__: pytest.fail("No installation on network failure")
    worker.run(application_only=True)
    assert state.ready and State.load(state_file).ready
    assert state.failed_step == "source"
    assert marker.read_text() == "existing application"


def test_downloaded_source_replaces_bundled_install_source(tmp_path, monkeypatch):
    responses(monkeypatch, archive())
    state = State(Settings(str(tmp_path / "venv"), str(tmp_path / "models"), str(tmp_path / "out"),
                           str(tmp_path / "config")), ready=True)
    worker = SetupWorker(state, tmp_path / "launcher/settings.json", tmp_path / "old-bundle", lambda _: None, remote=True)
    used = []
    worker._dependencies = lambda: used.append(worker.source)
    worker._diagnostics = lambda: None
    worker._entrypoint = lambda: None
    worker.run(application_only=True)
    assert used == [worker.home / "sources" / SHA]
    assert state.ready
    assert json.loads((worker.home / "source.json").read_text())["revision"] == SHA
