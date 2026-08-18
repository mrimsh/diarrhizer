"""Tests for diarrhizer.pipeline.stages.export: the Exporter registry mechanism.

Markdown/JSON rendering itself is covered by test_markdown_export.py (and by
export_to_json's own contract); these tests focus on ExportStage's control
flow - that run() and get_output_paths()/get_artifact_paths()/is_cache_valid()
all derive from the EXPORTERS registry instead of hardcoding per-format
paths, so adding a format is just appending an Exporter entry.
"""

import json

from diarrhizer.pipeline.runner import JobContext, PipelineConfig
from diarrhizer.pipeline.stages.export import Exporter, ExportStage


def _job(tmp_path, segments=None) -> JobContext:
    job_dir = tmp_path / "job"
    segments_dir = job_dir / "merged"
    segments_dir.mkdir(parents=True)
    if segments is None:
        segments = [{"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "hello"}]
    (segments_dir / "segments.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8"
    )
    config = PipelineConfig(job_id="job", input_file="input.wav", language="en", device="cpu")
    return JobContext(input_path=tmp_path / "input.wav", job_dir=job_dir, config=config)


# --- run() writes every registered format ---------------------------------

def test_run_writes_one_file_per_registered_exporter(tmp_path):
    job = _job(tmp_path)
    result = ExportStage().run(job)

    for exporter in ExportStage.EXPORTERS:
        output_file = job.job_dir / exporter.output_path
        assert output_file.exists()
        assert result["outputs"][exporter.name] == str(output_file)

    md = (job.job_dir / "export" / "result.md").read_text(encoding="utf-8")
    assert "hello" in md
    data = json.loads((job.job_dir / "export" / "result.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["text"] == "hello"


# --- get_output_paths/get_artifact_paths derive from EXPORTERS -----------

def test_get_output_paths_matches_registered_exporters(tmp_path):
    job_dir = tmp_path / "job"
    stage = ExportStage()
    outputs = stage.get_output_paths(job_dir)

    assert set(outputs) == {exporter.name for exporter in ExportStage.EXPORTERS}
    for exporter in ExportStage.EXPORTERS:
        assert outputs[exporter.name] == job_dir / exporter.output_path


def test_get_artifact_paths_includes_segments_and_every_output(tmp_path):
    job_dir = tmp_path / "job"
    stage = ExportStage()
    artifacts = stage.get_artifact_paths(job_dir)

    assert artifacts["segments"] == job_dir / stage.INPUT_SEGMENTS
    for name, path in stage.get_output_paths(job_dir).items():
        assert artifacts[name] == path


# --- extensibility: registering a new format needs no stage changes ------

def _word_count_export(segments, config, input_path) -> str:
    return str(sum(len(seg.get("text", "").split()) for seg in segments))


def test_registering_a_new_exporter_requires_no_stage_changes(tmp_path, monkeypatch):
    """A brand new format is picked up by run()/get_output_paths() purely by
    being appended to EXPORTERS - neither method references it by name, so
    this is what adding SRT/VTT/HTML/DOCX would look like.
    """
    extra = Exporter("wordcount", _word_count_export, "export/result.wordcount.txt")
    monkeypatch.setattr(ExportStage, "EXPORTERS", ExportStage.EXPORTERS + (extra,))

    job = _job(tmp_path, segments=[
        {"start": 0, "end": 1, "speaker_id": "Speaker_00", "text": "three word segment"}
    ])
    stage = ExportStage()
    result = stage.run(job)

    wordcount_file = job.job_dir / "export" / "result.wordcount.txt"
    assert wordcount_file.read_text(encoding="utf-8") == "3"
    assert result["outputs"]["wordcount"] == str(wordcount_file)
    assert "wordcount" in stage.get_output_paths(job.job_dir)
    # Original formats keep working unchanged alongside the new one.
    assert (job.job_dir / "export" / "result.md").exists()
    assert (job.job_dir / "export" / "result.json").exists()


# --- is_cache_valid: registered formats are one atomic group -------------

def test_is_cache_valid_false_if_any_single_format_output_missing(tmp_path, monkeypatch):
    extra = Exporter("wordcount", _word_count_export, "export/result.wordcount.txt")
    monkeypatch.setattr(ExportStage, "EXPORTERS", ExportStage.EXPORTERS + (extra,))

    job = _job(tmp_path)
    stage = ExportStage()
    stage.run(job)
    assert stage.is_cache_valid(job.job_dir) is True

    # Delete only the new format's output - the whole stage should still be
    # considered stale even though markdown/json are untouched and fresh,
    # because registered formats are cached as one atomic group.
    (job.job_dir / "export" / "result.wordcount.txt").unlink()
    assert stage.is_cache_valid(job.job_dir) is False
