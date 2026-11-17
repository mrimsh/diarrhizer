"""Release trust, numeric versions, staging and UI update boundaries."""

import ctypes
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from diarrhizer_launcher import self_update


def response(version="2.10"):
    return {"tag_name": "v" + version, "draft": False, "prerelease": False, "assets": [{
        "name": "Diarrhizer.exe", "digest": "sha256:" + "a" * 64, "state": "uploaded", "size": 100,
        "browser_download_url": f"https://github.com/mrimsh/diarrhizer/releases/download/v{version}/Diarrhizer.exe",
    }]}


def serve(monkeypatch, data):
    monkeypatch.setattr(self_update, "_open", lambda _: io.BytesIO(json.dumps(data).encode()))


def test_versions_are_compared_numerically_and_never_downgraded(monkeypatch):
    serve(monkeypatch, response())
    assert self_update.latest_launcher("2.9").version == "2.10"
    assert self_update.latest_launcher("2.10") is None
    assert self_update.latest_launcher("3.0") is None


@pytest.mark.parametrize("key,value", [("digest", None), ("digest", "sha256:bad"),
                                     ("size", 0), ("size", 300 * 1024 * 1024),
                                     ("browser_download_url", "https://example.org/Diarrhizer.exe")])
def test_invalid_asset_never_becomes_executable_download(monkeypatch, key, value):
    data = response()
    data["assets"][0][key] = value
    serve(monkeypatch, data)
    with pytest.raises(ValueError):
        self_update.latest_launcher("2.9")


@pytest.mark.parametrize("field", ["draft", "prerelease"])
def test_only_stable_releases_are_accepted(monkeypatch, field):
    data = response()
    data[field] = True
    serve(monkeypatch, data)
    with pytest.raises(ValueError, match="стабильный"):
        self_update.latest_launcher("2.9")


def test_missing_asset_is_reported(monkeypatch):
    data = response()
    data["assets"] = []
    serve(monkeypatch, data)
    with pytest.raises(ValueError, match="Diarrhizer.exe"):
        self_update.latest_launcher("2.9")


@pytest.fixture
def staged(tmp_path, monkeypatch):
    target = tmp_path / "каталог % $ ' пробел" / "Diarrhizer.exe"
    target.parent.mkdir()
    target.write_bytes(b"old launcher")
    state = tmp_path / "data" / "settings.json"
    state.parent.mkdir()
    state.write_bytes(b"personal settings")
    payload = b"verified new launcher"
    release = self_update.Release("2.10", "https://example.invalid", hashlib.sha256(payload).hexdigest(), len(payload))

    def download(url, path, digest, emit):
        path.write_bytes(payload)
        return path

    monkeypatch.setattr(self_update, "download", download)
    monkeypatch.setattr(self_update, "executable_version", lambda _: "2.10")
    return target, state, release


def test_prepare_and_decline_do_not_replace_running_exe_or_settings(staged):
    target, state, release = staged
    prepared = self_update.prepare_update(release, target, state, lambda _: None, threading.Event())
    plan = json.loads(prepared.plan.read_text(encoding="utf-8"))
    assert plan["target"] == str(target)
    assert plan["state"] == str(state)
    assert prepared.staged.parent == target.parent
    assert target.read_bytes() == b"old launcher"
    assert state.read_bytes() == b"personal settings"
    self_update.discard_update(prepared)
    assert not prepared.staged.exists()
    assert target.read_bytes() == b"old launcher"


def test_mismatched_version_leaves_original_intact(staged, monkeypatch):
    target, state, release = staged
    monkeypatch.setattr(self_update, "executable_version", lambda _: "2.9")
    with pytest.raises(ValueError, match="версия"):
        self_update.prepare_update(release, target, state, lambda _: None, threading.Event())
    assert target.read_bytes() == b"old launcher"
    assert not list(target.parent.glob("*.update-*.exe"))


def test_candidate_tampering_prevents_helper_execution(staged):
    target, state, release = staged
    prepared = self_update.prepare_update(release, target, state, lambda _: None, threading.Event())
    prepared.staged.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="изменился"):
        self_update.start_replacement(prepared)
    assert target.read_bytes() == b"old launcher"


def test_cancellation_does_not_replace_target(staged):
    target, state, release = staged
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="остановлено"):
        self_update.prepare_update(release, target, state, lambda _: None, cancel)
    assert target.read_bytes() == b"old launcher"


def test_invalid_pe_is_not_executed(tmp_path):
    path = tmp_path / "fake.exe"
    path.write_bytes(b"not an exe")
    with pytest.raises(ValueError, match="Windows exe"):
        self_update.executable_version(path)


@pytest.fixture
def compiler():
    executable = Path(os.environ.get("SystemRoot", "C:/Windows")) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    if os.name != "nt" or not executable.is_file():
        pytest.skip("Windows .NET Framework compiler required")
    return executable


@pytest.mark.parametrize("acknowledge", [True, False])
def test_real_windows_helper_waits_replaces_and_rolls_back(tmp_path, monkeypatch, compiler, acknowledge):
    """Only test-owned binaries/processes: real File.Replace, restart arguments and rollback."""
    home = tmp_path / "test data % $ ' кириллица"
    home.mkdir()

    def compile_exe(path, version, ack):
        source = path.with_suffix(".cs")
        source.write_text('''using System; using System.IO; using System.Reflection;
[assembly: AssemblyVersion("''' + version + '''.0.0")]
[assembly: AssemblyFileVersion("''' + version + '''.0.0")]
class App { static void Main(string[] args) {
    string state="", ack="";
    for(int i=0;i+1<args.Length;i++) {
        if(args[i]=="--state") state=args[i+1];
        if(args[i]=="--update-ack") ack=args[i+1];
    }
    File.AppendAllText(Assembly.GetExecutingAssembly().Location+".started", state+"\\n");
    ''' + ('if(ack!="") File.WriteAllText(ack,"ready");' if ack else '') + '''
} }''', encoding="utf-8")
        subprocess.run([str(compiler), "/nologo", "/target:exe", "/platform:x64", "/out:" + str(path), str(source)],
                       check=True, capture_output=True, creationflags=self_update.CREATE_NO_WINDOW)

    target, candidate = home / "Diarrhizer.exe", home / "candidate.exe"
    compile_exe(target, "2.9", False)
    compile_exe(candidate, "2.10", acknowledge)
    assert self_update.executable_version(candidate) == "2.10"
    original = target.read_bytes()
    state = home / "data" / "settings.json"
    state.parent.mkdir()
    state.write_bytes(b"personal configuration")
    release = self_update.Release("2.10", "unused", hashlib.sha256(candidate.read_bytes()).hexdigest(), candidate.stat().st_size)

    def download(_url, path, _digest, _emit):
        path.write_bytes(candidate.read_bytes())
        return path

    monkeypatch.setattr(self_update, "download", download)
    prepared = self_update.prepare_update(release, target, state, lambda _: None, threading.Event())
    base = getattr(sys, "_base_executable", sys.executable)
    parent = subprocess.Popen([base, "-I", "-c", "import time; time.sleep(60)"], creationflags=self_update.CREATE_NO_WINDOW)
    helper = None
    locked = None
    try:
        plan = json.loads(prepared.plan.read_text(encoding="utf-8"))
        plan["parent_pid"] = parent.pid
        prepared.plan.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        try:
            helper = self_update.start_replacement(prepared)
        except RuntimeError:
            pytest.fail(Path(plan["log"]).read_text(encoding="utf-8", errors="replace"))
        assert target.read_bytes() == original  # Helper has not changed a running parent's file.
        if acknowledge:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
                                          ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
            kernel.CreateFileW.restype = ctypes.c_void_p
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            locked = kernel.CreateFileW(str(target), 0x80000000, 1, None, 3, 128, None)
            assert locked != ctypes.c_void_p(-1).value
        parent.terminate()
        parent.wait(timeout=10)
        if locked:
            time.sleep(0.6)
            assert not Path(plan["result"]).exists(), Path(plan["log"]).read_text(encoding="utf-8")
            assert target.read_bytes() == original
            kernel.CloseHandle(locked)
            locked = None
        code = helper.wait(timeout=50)
        result = json.loads(Path(plan["result"]).read_text(encoding="utf-8"))
        assert result["status"] == ("updated" if acknowledge else "failed"), Path(plan["log"]).read_text(encoding="utf-8")
        assert code == (0 if acknowledge else 1)
        assert state.read_bytes() == b"personal configuration"
        if acknowledge:
            assert self_update.executable_version(target) == "2.10"
            assert Path(plan["backup"]).read_bytes() == original
        else:
            assert target.read_bytes() == original
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not target.with_suffix(".exe.started").exists():
            time.sleep(0.1)
        assert str(state) in target.with_suffix(".exe.started").read_text(encoding="utf-8-sig")
    finally:
        if locked:
            kernel.CloseHandle(locked)
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=10)
        if helper and helper.poll() is None:
            helper.terminate()
            helper.wait(timeout=10)
