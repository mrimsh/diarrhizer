"""Cross-process writes, crash recovery and complete LLM budget regressions (B1–B4)."""

import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from diarrhizer import artifacts
from diarrhizer.adapters import llm
from diarrhizer.artifacts import job_write_lock, write_bundle, bundle_valid, recover_job
from diarrhizer.pipeline.runner import new_job_dir
from diarrhizer.summary import service
from diarrhizer.summary.context_budget import ContextTooLong, check_request
from diarrhizer.summary.providers import KIND_OPENAI_API, KIND_ANTHROPIC_API, Provider
from diarrhizer.summary.templates import PromptTemplate


def _hold_lock(path, ready):
    with job_write_lock(path):
        ready.set()
        # The test terminates this process to exercise OS cleanup, not a finally block.
        import time
        time.sleep(30)


def _provider(**kwargs):
    return Provider(name="synthetic", kind=KIND_OPENAI_API, model="fake", endpoint="https://example.invalid/v1", **kwargs)


def _job(tmp_path):
    job = tmp_path / "job"
    (job / "export").mkdir(parents=True)
    (job / "export" / "result.md").write_text("Тестовый разговор.", encoding="utf-8")
    return job


def test_project_rename_rejects_active_descendant_writer(tmp_path):
    from diarrhizer.projects import reserve_job, rename_scope
    job = reserve_job(tmp_path, Path("call.wav"), project="Project", team="Team")
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(target=_hold_lock, args=(str(job), ready))
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(RuntimeError, match="busy"):
            rename_scope(job.parent.parent, "Renamed")
        assert job.exists()
    finally:
        process.terminate()
        process.join(10)
    moved = rename_scope(job.parent.parent, "Renamed")
    assert (moved / "Team" / job.name).exists()


def test_os_lock_rejects_second_writer_and_releases_after_process_termination(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(target=_hold_lock, args=(str(tmp_path), ready))
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(RuntimeError, match="busy"):
            with job_write_lock(tmp_path):
                pytest.fail("second writer entered")
    finally:
        process.terminate()
        process.join(10)
    with job_write_lock(tmp_path):
        with job_write_lock(tmp_path):
            pass  # Reentrant on this thread; old lock file is harmless.


@pytest.mark.parametrize("lock_target", ["source", "destination"])
def test_batch_move_refuses_busy_job_or_destination_without_partial_moves(tmp_path, lock_target):
    from diarrhizer.projects import ensure_scope, move_jobs, reserve_job
    a = reserve_job(tmp_path, Path("a.wav"), project="First")
    b = reserve_job(tmp_path, Path("b.wav"), project="Second")
    target = ensure_scope(tmp_path, "Target", "Team") if lock_target == "destination" else b
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    process = ctx.Process(target=_hold_lock, args=(str(target), ready))
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(RuntimeError, match="busy"):
            move_jobs(tmp_path, [a, b], project="Target", team="Team")
        assert a.exists() and b.exists()
        if lock_target == "source":
            assert not (tmp_path / "Target").exists()
    finally:
        process.terminate()
        process.join(10)


def test_job_names_are_reserved_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr("diarrhizer.pipeline.runner.generate_job_id", lambda _: "same")
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: new_job_dir(tmp_path, "same.wav"), range(16)))
    assert len(set(paths)) == 16 and all(p.is_dir() for p in paths)


def test_partial_bundle_is_not_accepted_and_recovers_without_regeneration(tmp_path, monkeypatch):
    root = tmp_path / "export"
    files = {"result.md": "old", "result.json": "{}"}
    write_bundle(root, files, "manifest.json")
    original = artifacts.write_text_atomic
    def interrupted(path, content):
        if Path(path).suffix == ".json":
            raise OSError("synthetic full disk")
        original(path, content)
    monkeypatch.setattr(artifacts, "write_text_atomic", interrupted)
    with pytest.raises(OSError, match="full disk"):
        write_bundle(root, {"result.md": "new", "result.json": '{"new":true}'}, "manifest.json")
    assert not bundle_valid(root, "manifest.json")
    monkeypatch.setattr(artifacts, "write_text_atomic", original)
    assert recover_job(tmp_path) == {"export": 1, "summary": 0}
    assert bundle_valid(root, "manifest.json")
    assert (root / "result.md").read_text() == "new"
    assert json.loads((root / "result.json").read_text()) == {"new": True}
    assert recover_job(tmp_path) == {"export": 0, "summary": 0}


@pytest.mark.parametrize("filename", ["../outside", "C:outside", "sub/file", "sub\\file", ".hidden"])
def test_bundle_rejects_paths_outside_its_directory(tmp_path, filename):
    with pytest.raises(ValueError, match="filename"):
        write_bundle(tmp_path, {filename: "never"}, "manifest.json")


def test_corrupt_journal_is_preserved_for_manual_recovery(tmp_path):
    folder = tmp_path / "summary"
    folder.mkdir()
    journal = folder / ".x.manifest.json.pending.json"
    journal.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        recover_job(tmp_path)
    assert journal.read_text() == "{bad"


def test_bundle_hashes_exact_utf8_bytes_including_windows_line_endings(tmp_path):
    content = "Первая строка\r\nВторая\nТретья"
    write_bundle(tmp_path, {"reply.md": content}, "reply.manifest.json")
    assert (tmp_path / "reply.md").read_bytes() == content.encode("utf-8")
    assert bundle_valid(tmp_path, "reply.manifest.json")


def test_summary_profile_requires_a_provider_even_when_full_profile_does_not(summary_stores):
    from diarrhizer.diagnostics import doctor
    assert doctor.check_summarization()[1]
    assert not doctor.check_summarization(required=True)[1]


def test_changed_manifest_file_is_rejected_but_legacy_export_is_readable(tmp_path):
    job = _job(tmp_path)
    assert service.read_transcript(job) == "Тестовый разговор."
    write_bundle(job / "export", {"result.md": "generated"}, "manifest.json")
    (job / "export" / "result.md").write_text("edited", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or changed"):
        service.read_transcript(job)


def test_budget_boundary_counts_rendered_input_framing_and_response():
    provider = _provider(max_tokens=50)
    check = check_request(["а" * 300, "b" * 30], provider)
    assert (check.input_tokens, check.overhead_tokens, check.reserved_output_tokens) == (110, 64, 50)
    assert not check_request(["а" * 300, "b" * 30], provider, check.estimated_tokens).exceeds
    assert check_request(["а" * 300, "b" * 30], provider, check.estimated_tokens - 1).exceeds
    assert check_request(["x" * 10000], provider).limit is None


def test_anthropic_reserve_matches_the_adapter_default():
    provider = Provider(name="a", kind=KIND_ANTHROPIC_API, model="fake", endpoint="https://api.anthropic.com", api_key_env="TEST_ANTHROPIC_KEY")
    assert check_request(["x"], provider).reserved_output_tokens == llm.ANTHROPIC_DEFAULT_MAX_TOKENS


def test_summary_checks_template_note_and_response_before_network_or_writes(tmp_path, monkeypatch):
    job = _job(tmp_path)
    template = PromptTemplate("test", "x" * 400 + " {{transcript}} {{note}}")
    monkeypatch.setattr(llm, "complete", lambda *args: pytest.fail("network must not run"))
    with pytest.raises(ContextTooLong) as error:
        service.summarize_job(job, template, _provider(context_tokens=200, max_tokens=50), note="я" * 300)
    assert error.value.check.reserved_output_tokens == 50
    assert error.value.check.input_tokens > 200
    assert not (job / "summary").exists()


def test_recovered_summary_does_not_pay_for_a_second_response(tmp_path, monkeypatch):
    job = _job(tmp_path)
    calls = []
    monkeypatch.setattr(llm, "complete", lambda *args: calls.append(1) or llm.LLMReply("paid reply"))
    original = artifacts.write_text_atomic
    monkeypatch.setattr(artifacts, "write_text_atomic", lambda *args: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        service.summarize_job(job, PromptTemplate("test", "{{transcript}}"), _provider())
    monkeypatch.setattr(artifacts, "write_text_atomic", original)
    with pytest.raises(ValueError, match="Recovered"):
        service.summarize_job(job, PromptTemplate("test", "{{transcript}}"), _provider())
    assert calls == [1]
    (saved,) = service.list_summaries(job)
    assert Path(saved["path"]).read_text().strip() == "paid reply"
    assert saved["schema_version"] == 1 and saved["source_sha256"]


def test_doctor_profiles_skip_irrelevant_stack(monkeypatch):
    from diarrhizer.diagnostics import doctor
    monkeypatch.setattr(doctor, "check_python_version", lambda: ("Python version", True, "test"))
    monkeypatch.setattr(doctor, "check_torch", lambda: pytest.fail("ML imported"))
    assert doctor.run_all_checks("export") == [("Python version", True, "test")]
    assert doctor.is_blocking_failure("Summarization", False, "summary")
    assert not doctor.is_blocking_failure("Summarization", False, "full")
    assert not doctor.is_blocking_failure("CUDA", False, "asr")


def test_cpu_diagnostics_do_not_advertise_cuda_to_the_gui(monkeypatch):
    import sys
    from types import SimpleNamespace
    from diarrhizer.diagnostics import doctor
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda=None)))
    name, available, message = doctor.check_cuda()
    assert name == "CUDA" and not available and "--device cpu" in message


def test_doctor_continues_after_dll_failure(monkeypatch):
    from diarrhizer.diagnostics import doctor
    def broken():
        raise OSError("DLL load failed")
    monkeypatch.setattr(doctor, "ALL_CHECKS", [broken, lambda: ("next", True, "checked")])
    assert doctor.run_all_checks() == [("broken", False, "OSError: DLL load failed"), ("next", True, "checked")]


def test_installed_config_uses_profile_and_override_not_site_packages(tmp_path, monkeypatch):
    from diarrhizer import env_file
    from diarrhizer.summary.store_file import resolve_path
    monkeypatch.setattr(env_file, "_SOURCE_CHECKOUT", False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("DIARRHIZER_CONFIG_DIR", raising=False)
    assert env_file.config_dir() == tmp_path / "Diarrhizer"
    root = tmp_path / "configured"
    monkeypatch.setenv("DIARRHIZER_CONFIG_DIR", str(root))
    monkeypatch.delenv("TEST_STORE_PATH", raising=False)
    assert resolve_path("TEST_STORE_PATH", "providers.json") == root / "providers.json"
    monkeypatch.setenv("TEST_STORE_PATH", str(tmp_path / "separate.json"))
    assert resolve_path("TEST_STORE_PATH", "providers.json") == tmp_path / "separate.json"
    env_file.write_env_file(root / ".env", {"TEST_CONFIG_VALUE": "synthetic"})
    monkeypatch.delenv("TEST_CONFIG_VALUE", raising=False)
    monkeypatch.chdir(tmp_path)
    env_file.load_project_env()
    assert os.environ["TEST_CONFIG_VALUE"] == "synthetic"


def test_history_treats_unfinished_export_as_resumable(tmp_path):
    from diarrhizer_gui.job_scan import stage_done, scan_jobs
    job = _job(tmp_path)
    write_bundle(job / "export", {"result.md": "ok"}, "manifest.json")
    assert stage_done(job)["export"]
    (job / "export" / ".manifest.json.pending.json").write_text("{}")
    assert not stage_done(job)["export"]
    with job_write_lock(job):
        assert len(scan_jobs(tmp_path)) == 1  # hidden lock folder is not a job


def test_fresh_job_preflight_cannot_reuse_a_colliding_job_cache(tmp_path, monkeypatch):
    from diarrhizer.pipeline.runner import run_pipeline
    monkeypatch.setattr("diarrhizer.pipeline.runner.generate_job_id", lambda _: "same")
    media = tmp_path / "input.wav"
    media.write_bytes(b"synthetic")
    existing = tmp_path / "out" / "same"
    existing.mkdir(parents=True)
    (existing / "cached").write_text("old")
    class Stage:
        NAME = "convert"
        def preflight(self, job):
            assert not (job.job_dir / "cached").exists()
            raise RuntimeError("expected preflight rejection")
    with pytest.raises(RuntimeError, match="expected preflight"):
        run_pipeline(input_path=media, out_dir=existing.parent, stages=[Stage()])
    assert [p for p in existing.parent.iterdir() if p.name != artifacts.LOCK_DIR] == [existing]
