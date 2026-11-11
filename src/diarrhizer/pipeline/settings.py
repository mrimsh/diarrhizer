"""Lightweight, shared readers for resuming CLI and GUI jobs."""

import json
import logging
import math
from pathlib import Path

from diarrhizer.audio_formats import AudioFormat
from diarrhizer.export.speakers import normalize_speaker_id
from diarrhizer.utils import write_json_atomic
from diarrhizer.projects import export_path

FULL_STAGES = ("convert", "transcribe", "diarize", "merge", "export")
RECORDED_RUN_KEYS = (
    "language", "device", "min_speakers", "max_speakers", "asr_model", "asr_compute_type",
    "asr_beam_size", "asr_temperature", "asr_condition_on_previous_text", "asr_initial_prompt",
    "asr_vad_filter", "asr_vad_min_silence_ms", "audio_profile", "min_turn_duration",
)


# [SEMANTIC-BEGIN] CONFIG:RESUME
# @purpose: Read the same recorded settings and speaker mapping for CLI and GUI resume
# @description: Explicit overrides are applied by the caller after these recorded values; omitted
#   legacy fields retain pipeline defaults. Corrupt critical settings fail clearly; optional derived
#   artifacts are read best-effort so they can still be rebuilt.
#   The full saved audio format survives removal/renaming of a personal profile. Speaker names
#   prefer speakers.json (including an explicit empty map), then the last export's mapping.
#   Portable jobs resolve their named export through job.json; legacy result.json remains supported.
# @inputs: job directory, optional already-read metadata
# @outputs: recorded run kwargs and stage names
# @sideEffects: Reads job JSON only; never changes a queued job
# @errors: ValueError for malformed JSON/schema, OSError for inaccessible files
# @see: CONFIG:PIPELINE, PIPELINE:EXECUTE, GUI:JOB_PLANS
def read_object(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        value = json.loads(text)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _optional_object(path: Path) -> dict:
    """A broken derived artifact must not prevent rebuilding that artifact."""
    try:
        return read_object(path)
    except (ValueError, OSError, UnicodeError) as exc:
        logging.getLogger(__name__).warning("Ignoring optional recorded settings: %s", exc)
        return {}


def read_job_settings(job_dir: Path, meta: dict | None = None, *, include_speakers: bool = True) -> dict:
    job_dir = Path(job_dir)
    meta = read_object(job_dir / "meta" / "run.json") if meta is None else meta
    config = meta.get("pipeline_config", {})
    if not isinstance(config, dict):
        raise ValueError(f"{job_dir}/meta/run.json: pipeline_config must be an object")
    settings = {key: config[key] for key in RECORDED_RUN_KEYS if key in config}
    integers = {"min_speakers", "max_speakers", "asr_beam_size", "asr_vad_min_silence_ms"}
    numbers = {"asr_temperature", "min_turn_duration"}
    booleans = {"asr_condition_on_previous_text", "asr_vad_filter"}
    nullable = {"asr_compute_type", "asr_initial_prompt"}
    for key, value in settings.items():
        if key in integers:
            valid = type(value) is int and value >= 0
        elif key in numbers:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        elif key in booleans:
            valid = isinstance(value, bool)
        else:
            valid = isinstance(value, str) or (key in nullable and value is None)
        if not valid:
            raise ValueError(f"{job_dir}/meta/run.json: invalid {key}")
    if "min_turn_duration" not in settings:
        merged = _optional_object(job_dir / "merged" / "segments.json")
        merge_meta = merged.get("metadata") or {}
        duration = merge_meta.get("min_turn_duration") if isinstance(merge_meta, dict) else None
        if type(duration) in (int, float) and math.isfinite(duration) and duration >= 0:
            settings["min_turn_duration"] = duration
    conversion = meta.get("config", {})
    if not isinstance(conversion, dict):
        raise ValueError(f"{job_dir}/meta/run.json: config must be an object")
    fmt = conversion.get("audio_format")
    if isinstance(fmt, dict):
        try:
            settings["audio_format"] = AudioFormat.from_dict(fmt)
        except TypeError as exc:
            raise ValueError(f"{job_dir}/meta/run.json: invalid audio_format: {exc}") from exc
    elif config.get("audio_format"):
        if not isinstance(config["audio_format"], str):
            raise ValueError(f"{job_dir}/meta/run.json: audio_format must be a profile name")
        settings["audio_format"] = config["audio_format"]
    if not include_speakers:
        return settings
    names_path = job_dir / "speakers.json"
    if names_path.exists():
        names = read_object(names_path)
    else:
        exported = _optional_object(export_path(job_dir, "json"))
        export_meta = exported.get("metadata") or {}
        names = export_meta.get("speakers", {}) if isinstance(export_meta, dict) else {}
    if not isinstance(names, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in names.items()):
        raise ValueError(f"{job_dir}: speaker names must be a string-to-string mapping")
    settings["speakers"] = {normalize_speaker_id(k): v.strip() for k, v in names.items()
                            if v.strip() and v.strip() != normalize_speaker_id(k)}
    return settings


def recorded_stage_names(job_dir: Path | None) -> tuple[str, ...]:
    if job_dir is None:
        return FULL_STAGES
    meta = read_object(Path(job_dir) / "meta" / "run.json")
    config = meta.get("pipeline_config", {})
    if not isinstance(config, dict):
        raise ValueError("pipeline_config must be an object")
    names = config.get("stages", list(FULL_STAGES))
    if (not isinstance(names, list) or not names or any(name not in FULL_STAGES for name in names)
            or names != [name for name in FULL_STAGES if name in names]):
        raise ValueError("Recorded stages must be unique known stages in pipeline order")
    return tuple(names)


# [SEMANTIC-BEGIN] READ:MERGED_SEGMENTS
# @purpose: Read saved results for GUI display and export, including legacy reversed intervals
# @description: Finite non-negative end < start collapses to start in memory, preserving text,
#   speakers and other fields with an alignment_warning containing the original bounds.
#   Word times on a recovered segment become unknown; the source file is never rewritten.
#   Null/non-numeric/non-finite/negative times and malformed schemas remain errors.
# @inputs: path to merged/segments.json
# @outputs: validated segment list with warnings on recovered intervals
# @sideEffects: Reads JSON and logs recovery warnings; no filesystem writes
# @errors: ValueError for malformed data (timestamp errors include index/bounds), OSError
# @see: CONFIG:RESUME, GUI:RESULT_SCREEN, STAGE:EXPORT, MERGE:ASSIGN_SPEAKERS
def read_segments(path: Path) -> list[dict]:
    """Validate saved segments, recovering legacy reversed intervals only in memory."""
    segments = read_object(path).get("segments", [])
    if not isinstance(segments, list):
        raise ValueError(f"{path}: segments must be a list")
    for index, seg in enumerate(segments):
        if not isinstance(seg, dict) or not isinstance(seg.get("text", ""), str):
            raise ValueError(f"{path}: each segment must be an object with text")
        start, end = seg.get("start", 0), seg.get("end", 0)
        if not (type(start) in (int, float) and type(end) in (int, float)
                and math.isfinite(start) and math.isfinite(end) and start >= 0 and end >= 0):
            raise ValueError(
                f"{path}: invalid segment timestamps (segment {index}, start={start!r}, end={end!r})"
            )
        if not isinstance(seg.get("speaker_id", "Speaker_00"), str):
            raise ValueError(f"{path}: speaker_id must be text")
        if end < start:
            warning = (f"Reversed merged segment bounds (start={start!r}, end={end!r}); "
                       "end set to start, timing uncertain; original text preserved")
            previous = seg.get("alignment_warning")
            if isinstance(previous, str) and previous:
                warning = f"{previous}; {warning}"
            seg = {**seg, "end": start, "alignment_warning": warning}
            if isinstance(seg.get("words"), list):
                seg["words"] = [{**word, "start": None, "end": None} if isinstance(word, dict) else word
                                for word in seg["words"]]
            segments[index] = seg
            logging.getLogger(__name__).warning("%s: segment %d: %s", path, index, warning)
    return segments
# [SEMANTIC-END] READ:MERGED_SEGMENTS
# [SEMANTIC-END] CONFIG:RESUME


# [SEMANTIC-BEGIN] JOB:RESET_SPEAKER_NAMES
# @purpose: Preserve old names immediately before actual diarization recomputation
# @inputs: job directory
# @outputs: speakers.before-rediarize.json if a nonempty mapping existed; empty speakers.json
# @sideEffects: Moves the old map and writes an empty map so interrupted runs cannot revive names
#   from a stale export. Retries preserve the nonempty backup. Never called while merely planning a run.
# @errors: OSError stops the run instead of silently losing the backup
# @see: PIPELINE:EXECUTE, GUI:JOB_PLANS
def set_aside_speaker_names(job_dir: Path) -> None:
    source = job_dir / "speakers.json"
    if source.exists() and read_object(source):
        source.replace(job_dir / "speakers.before-rediarize.json")
    write_json_atomic(source, {})
# [SEMANTIC-END] JOB:RESET_SPEAKER_NAMES


# [SEMANTIC-BEGIN] PIPELINE:BUILD_STAGES
# @purpose: Construct the CLI/GUI stage list lazily without importing ML in the GUI
# @inputs: ordered stage names
# @outputs: stage instances
# @sideEffects: Imports stage adapters
# @errors: ValueError for unknown stage names
# @see: PIPELINE:EXECUTE, CONFIG:RESUME, GUI:CHILD_TASKS
def build_stages(names=FULL_STAGES) -> list:
    from diarrhizer.pipeline.stages.convert import ConvertStage
    from diarrhizer.pipeline.stages.transcribe import TranscribeStage
    from diarrhizer.pipeline.stages.diarize import DiarizeStage
    from diarrhizer.pipeline.stages.merge import MergeStage
    from diarrhizer.pipeline.stages.export import ExportStage
    classes = {s.NAME: s for s in (ConvertStage, TranscribeStage, DiarizeStage, MergeStage, ExportStage)}
    if any(name not in classes for name in names):
        raise ValueError(f"Unknown pipeline stages: {names}")
    return [classes[name]() for name in names]
# [SEMANTIC-END] PIPELINE:BUILD_STAGES
