"""Corpus boundary/integration checks use local synthetic bytes and fake inference, never downloads."""
import hashlib
import importlib
import io
import json
import wave
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "tools"))
    return tuple(importlib.import_module(name) for name in ("prepare_der_corpus", "benchmark_diarization"))


@pytest.fixture
def sources(tmp_path):
    audio_dir, fixtures = tmp_path / "audio", tmp_path / "fixtures"
    audio_dir.mkdir()
    fixtures.mkdir()
    audio = audio_dir / "sample.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 64000)
    reference = fixtures / "sample.rttm"
    reference.write_bytes(b"SPEAKER sample 1 0 3 <NA> <NA> A <NA> <NA>\nSPEAKER sample 1 2 2 <NA> <NA> B <NA> <NA>\n")
    lock = {"schema_version": 1, "audio_archive": "https://example.invalid/corpus.zip", "cases": [
        {"id": "sample", "duration_seconds": 4, "audio_bytes": audio.stat().st_size,
         "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
         "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}]}
    (fixtures / "corpus-lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return audio_dir, fixtures, lock


def test_preparation_preserves_reference_and_full_file_uem(modules, sources, tmp_path):
    prepare, suite = modules
    audio_dir, fixtures, _ = sources
    root = tmp_path / "corpus"
    prepare.prepare(audio_dir, root, fixtures)
    cases = suite.load_cases(root / "corpus.json")
    assert len(cases) == 1 and cases[0]["uem_timeline"].duration() == 4
    assert cases[0]["reference_annotation"].get_overlap().duration() == 1
    assert (root / "sample.rttm").read_bytes() == (fixtures / "sample.rttm").read_bytes()


@pytest.mark.parametrize("damage", ["audio", "reference"])
def test_preparation_rejects_changed_source_before_publication(modules, sources, tmp_path, damage):
    prepare, _ = modules
    audio_dir, fixtures, _ = sources
    path = audio_dir / "sample.wav" if damage == "audio" else fixtures / "sample.rttm"
    path.write_bytes(path.read_bytes() + b"changed")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare.prepare(audio_dir, output, fixtures)
    assert not output.exists()


@pytest.mark.parametrize("damage", ["audio", "reference", "uem", "escape", "duration", "unknown", "outside", "empty_uem"])
def test_invalid_manifest_is_rejected_before_inference(modules, sources, tmp_path, damage):
    prepare, suite = modules
    audio_dir, fixtures, _ = sources
    root = tmp_path / "corpus"
    manifest = prepare.prepare(audio_dir, root, fixtures)
    case = manifest["cases"][0]
    if damage in ("audio", "reference", "uem"):
        case[damage + "_sha256"] = "0" * 64
    elif damage == "escape":
        case["reference"] = "../escape.rttm"
    elif damage == "duration":
        case["duration_seconds"] = 9
    elif damage in ("outside", "empty_uem"):
        (root / case["uem"]).write_text("sample 1 0 5\n" if damage == "outside" else "")
        case["uem_sha256"] = prepare.digest(root / case["uem"])
    (root / "corpus.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        suite.load_cases(root / "corpus.json", ["unknown"] if damage == "unknown" else None)


def test_suite_uses_raw_diarization_without_asr_or_oracle_counts(modules, sources, tmp_path, monkeypatch):
    prepare, suite = modules
    audio_dir, fixtures, _ = sources
    root, job = tmp_path / "corpus", tmp_path / "job"
    prepare.prepare(audio_dir, root, fixtures)
    (job / "diar").mkdir(parents=True)
    (job / "diar/diarization.json").write_text(json.dumps({"stage": "diarize", "segments": [
        {"start": 0, "end": 3, "speaker": "X"}, {"start": 2, "end": 4, "speaker": "Y"}]}))
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return {"job_dir": str(job)}
    monkeypatch.setattr(suite, "run_pipeline", run)
    monkeypatch.setattr(suite, "build_stages", lambda names: names)
    result = suite.run_case(suite.load_cases(root / "corpus.json")[0], tmp_path / "out")
    assert result["der"]["der"] == 0
    assert calls[0]["stages"] == ("convert", "diarize")
    assert calls[0]["min_speakers"] == 1 and calls[0]["max_speakers"] == 10


def test_suite_retains_failed_repeat_and_reports_incomplete_coverage(modules, sources, tmp_path, monkeypatch):
    prepare, suite = modules
    audio_dir, fixtures, _ = sources
    root = tmp_path / "corpus"
    prepare.prepare(audio_dir, root, fixtures)
    cases = suite.load_cases(root / "corpus.json")
    score = suite.score_der(cases[0]["reference_annotation"], cases[0]["reference_annotation"], cases[0]["uem_timeline"])
    calls = []
    def run(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic failure")
        return {"der": score, "wall_seconds": 1}
    monkeypatch.setattr(suite, "run_case", run)
    monkeypatch.setattr(suite, "load_project_env", lambda: None)
    report = tmp_path / "report.json"
    assert suite.main([str(root / "corpus.json"), "--out", str(tmp_path / "out"), "--report", str(report), "--repeat", "2"]) == 1
    data = json.loads(report.read_text())
    assert data["coverage"] == {"planned": 2, "completed": 1, "failed": 1}
    assert data["aggregate"]["recordings"] == 1 and data["aggregate"]["der"] == 0
    assert "synthetic failure" in data["runs"][1]["error"]


def test_range_download_and_hash_rejection_without_network(modules, sources, tmp_path, monkeypatch):
    prepare, _ = modules
    audio_dir, _, lock = sources
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("audio/sample.wav", (audio_dir / "sample.wav").read_bytes())
    payload = buffer.getvalue()
    class Response(io.BytesIO):
        def __init__(self, content, status, headers):
            super().__init__(content)
            self.status, self.headers = status, headers
    def open_url(request, timeout):
        if request.get_method() == "HEAD":
            return Response(b"", 200, {"Content-Length": str(len(payload))})
        start, end = map(int, request.get_header("Range")[6:].split("-"))
        return Response(payload[start:end + 1], 206, {"Content-Range": f"bytes {start}-{end}/{len(payload)}"})
    monkeypatch.setattr(prepare.urllib.request, "urlopen", open_url)
    output = tmp_path / "download"
    prepare.download_audio(lock, output)
    assert (output / "sample.wav").read_bytes() == (audio_dir / "sample.wav").read_bytes()
    lock["cases"][0]["audio_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned bytes"):
        prepare.download_audio(lock, output)
    assert (output / "sample.wav").read_bytes() == (audio_dir / "sample.wav").read_bytes()


def test_range_download_refuses_full_archive_response(modules, monkeypatch):
    prepare, _ = modules
    class Response:
        status = 200
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, *args):
            pytest.fail("Must reject before reading a full archive")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="bounded byte ranges"):
        prepare.read_range("https://example.invalid/large.zip", 0, 100)


def test_vendored_reference_hashes_and_real_overlap_are_stable(modules):
    prepare, _ = modules
    metrics = importlib.import_module("diarization_metrics")
    lock = json.loads((prepare.FIXTURES / "corpus-lock.json").read_text())
    overlaps = []
    for case in lock["cases"]:
        reference = prepare.FIXTURES / (case["id"] + ".rttm")
        assert prepare.digest(reference) == case["reference_sha256"]
        overlaps.append(metrics.read_rttm(reference, case["id"]).get_overlap().duration())
    assert overlaps == pytest.approx([0, 0, 59])
