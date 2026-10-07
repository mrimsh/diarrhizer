"""Tests for `diarrhizer run` with several inputs (CLI:RUN batch mode),
INPUT:EXPAND_MEDIA and runner.new_job_dir().

run_pipeline is replaced by a recorder: these tests are about which jobs the
CLI starts and how it reports them, not about the stages themselves.
"""

import sys
from pathlib import Path

import pytest

from diarrhizer import cli
from diarrhizer.media import expand_media_paths, is_media_file
from diarrhizer.pipeline import runner


@pytest.fixture
def calls(monkeypatch):
    """Record run_pipeline calls; a file named 'bad*' fails like a broken input."""
    recorded = []

    def fake_run_pipeline(**kwargs):
        recorded.append(kwargs)
        input_path = Path(kwargs["input_path"]) if kwargs["input_path"] else None
        if input_path is not None and input_path.name.startswith("bad"):
            raise RuntimeError(f"cannot decode {input_path.name}")
        job_dir = kwargs["job_dir"] or Path(kwargs["out_dir"]) / "job"
        return {"job_id": Path(job_dir).name, "job_dir": str(job_dir), "stages": []}

    monkeypatch.setattr(runner, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(cli, "load_project_env", lambda: None)
    monkeypatch.setattr(cli, "_configure_logging", lambda: None)
    return recorded


def _main(monkeypatch, *args) -> int:
    monkeypatch.setattr(sys, "argv", ["diarrhizer", "run", *map(str, args)])
    return cli.main()


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_folder_runs_every_media_file_as_its_own_job(tmp_path, monkeypatch, calls, capsys):
    folder = tmp_path / "calls"
    _touch(folder / "b.m4a")
    _touch(folder / "a.mp3")
    _touch(folder / "notes.txt")
    out = tmp_path / "out"

    assert _main(monkeypatch, folder, "--out", out) == 0

    assert [Path(c["input_path"]).name for c in calls] == ["a.mp3", "b.m4a"]
    job_dirs = [Path(c["job_dir"]) for c in calls]
    assert all(d.parent == out for d in job_dirs)
    assert len(set(job_dirs)) == 2
    assert "Batch finished: 2 succeeded, 0 failed" in capsys.readouterr().out


def test_a_failed_file_does_not_stop_the_batch(tmp_path, monkeypatch, calls, capsys):
    first = _touch(tmp_path / "bad.wav")
    second = _touch(tmp_path / "good.wav")

    assert _main(monkeypatch, first, second, "--out", tmp_path / "out") == 1

    assert len(calls) == 2
    captured = capsys.readouterr()
    assert "cannot decode bad.wav" in captured.err
    assert "Batch finished: 1 succeeded, 1 failed" in captured.out


def test_single_input_keeps_the_plain_behaviour(tmp_path, monkeypatch, calls, capsys):
    media = _touch(tmp_path / "call.wav")

    assert _main(monkeypatch, media, "--out", tmp_path / "out") == 0

    assert len(calls) == 1
    assert calls[0]["job_dir"] is None  # run_pipeline names the job itself
    assert "Pipeline completed successfully!" in capsys.readouterr().out


def test_job_dir_with_several_inputs_is_rejected(tmp_path, monkeypatch, calls):
    a = _touch(tmp_path / "a.wav")
    b = _touch(tmp_path / "b.wav")

    assert _main(monkeypatch, a, b, "--job-dir", tmp_path / "job") == 1
    assert calls == []


def test_folder_without_media_is_an_error(tmp_path, monkeypatch, calls, capsys):
    folder = tmp_path / "empty"
    _touch(folder / "readme.txt")

    assert _main(monkeypatch, folder) == 1
    assert calls == []
    assert "No media files found" in capsys.readouterr().err


def test_expand_keeps_explicit_files_and_missing_paths_and_dedupes(tmp_path):
    explicit = _touch(tmp_path / "recording.unknownext")
    missing = tmp_path / "missing.mp3"
    folder = tmp_path / "dir"
    inside = _touch(folder / "x.opus")

    result = expand_media_paths([explicit, missing, folder, inside])

    assert result == [explicit, missing, inside]


def test_call_recorder_formats_count_as_media():
    for name in ("a.amr", "b.3gp", "c.opus", "d.OGG", "e.webm"):
        assert is_media_file(name)
    assert not is_media_file("notes.txt")


def test_new_job_dir_never_reuses_an_existing_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "generate_job_id", lambda input_path: "call_20260101_000000")
    first = runner.new_job_dir(tmp_path, "call.wav")
    assert first == tmp_path / "call_20260101_000000"
    first.mkdir()
    second = runner.new_job_dir(tmp_path, "call.wav")
    assert second == tmp_path / "call_20260101_000000_2"
    second.mkdir()
    assert runner.new_job_dir(tmp_path, "call.wav") == tmp_path / "call_20260101_000000_3"


class _ConvertStub:
    NAME = "convert"


def test_missing_input_with_job_dir_leaves_no_folder_behind(tmp_path):
    job_dir = tmp_path / "out" / "job"
    with pytest.raises(FileNotFoundError):
        runner.run_pipeline(
            input_path=tmp_path / "missing.wav",
            out_dir=tmp_path / "out",
            stages=[_ConvertStub()],
            job_dir=job_dir,
        )
    assert not job_dir.exists()
