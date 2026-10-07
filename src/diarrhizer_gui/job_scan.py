"""Scans an `out/` directory tree for Diarrhizer job folders.

Reads only the documented on-disk artifact contract (docs/architecture.md,
"Artifacts (results directory)") - not the heavier stage/adapter modules
(those pull in torch/whisperx at import time), so this stays cheap to import
and keeps working even on a machine without the ML stack installed.

Also builds the run_pipeline() arguments to resume an interrupted job or to
re-run diarization on a finished one, from what the job recorded in
meta/run.json - so neither needs the user to re-enter the job's settings.
"""

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from diarrhizer.audio_formats import AudioFormat, find_job_audio

# (stage name, artifact path relative to job_dir) - mirrors the layout
# documented in docs/architecture.md, not the stage classes themselves.
STAGE_ARTIFACTS = [
    ("convert", "audio/normalized.wav"),
    ("transcribe", "asr/transcript.json"),
    ("diarize", "diar/diarization.json"),
    ("merge", "merged/segments.json"),
    ("export", "export/result.md"),
]

FULL_STAGES = [name for name, _ in STAGE_ARTIFACTS]
ASR_ONLY_STAGES = [name for name in FULL_STAGES if name != "diarize"]

# meta/run.json pipeline_config keys that are run_pipeline() parameters of the
# same name - what a resumed job must reuse to behave like the original run.
RECORDED_RUN_KEYS = (
    "language",
    "device",
    "min_speakers",
    "max_speakers",
    "asr_model",
    "asr_compute_type",
    "asr_beam_size",
    "asr_temperature",
    "asr_condition_on_previous_text",
    "asr_initial_prompt",
    "asr_vad_filter",
    "asr_vad_min_silence_ms",
    "audio_profile",
    "min_turn_duration",
)

# generate_job_id()'s `_YYYYMMDD_HHMMSS`, plus the `_N` new_job_dir() adds on a clash.
_JOB_ID_DATE = re.compile(r"_(\d{8})_(\d{6})(?:_\d+)?$")


@dataclass(frozen=True)
class JobSummary:
    job_id: str
    job_dir: Path
    input_name: str
    date_label: str
    sort_key: datetime
    language: str
    speakers: str
    asr_model: str
    stage_done: dict
    result_path: Optional[Path]
    mode: str = "full"
    resumable: bool = False

    @property
    def complete(self) -> bool:
        return self.result_path is not None


def _parse_job_date(job_id: str, job_dir: Path) -> datetime:
    """Parse the `_YYYYMMDD_HHMMSS` suffix generate_job_id() always appends.

    Falls back to the folder's mtime if the id doesn't parse (e.g. a job
    folder created by something other than this tool).
    """
    match = _JOB_ID_DATE.search(job_id)
    if match:
        try:
            return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(job_dir.stat().st_mtime)
    except OSError:
        return datetime.min


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_meta(job_dir: Path) -> dict:
    return _read_json(job_dir / "meta" / "run.json")


def stage_done(job_dir: Path) -> dict:
    """{stage name: its main artifact exists}."""
    done = {name: (job_dir / rel_path).exists() for name, rel_path in STAGE_ARTIFACTS}
    # An audio format with keep_wav=false drops normalized.wav once the job
    # completes and keeps audio/archive.<ext> instead - still converted.
    done["convert"] = find_job_audio(job_dir) is not None
    return done


def job_mode(meta: dict) -> str:
    """"asr_only" if the job was run without the diarize stage, else "full".

    Jobs from before the stage list was recorded count as "full".
    """
    stages = (meta.get("pipeline_config") or {}).get("stages")
    if isinstance(stages, list) and stages and "diarize" not in stages:
        return "asr_only"
    return "full"


def delete_job(out_dir: Path, job_dir: Path) -> None:
    """Permanently remove a job folder.

    Refuses anything that is not a direct child directory of out_dir, so a
    stale or tampered row can never delete outside the results root.
    """
    if job_dir.parent.resolve() != out_dir.resolve() or not job_dir.is_dir():
        raise ValueError(f"Not a job folder under {out_dir}: {job_dir}")
    shutil.rmtree(job_dir)


def scan_jobs(out_dir: Path) -> list:
    """List job folders under out_dir, newest first.

    A job with no meta/run.json yet (e.g. convert hasn't finished) still
    shows up with "-" placeholders rather than being skipped, so an
    interrupted job stays visible in the history.
    """
    if not out_dir.exists():
        return []

    summaries = []
    for job_dir in out_dir.iterdir():
        if not job_dir.is_dir():
            continue

        meta = _read_meta(job_dir)
        pipeline_config = meta.get("pipeline_config", {})

        input_path = meta.get("input_path")
        input_name = Path(input_path).name if input_path else "—"

        date = _parse_job_date(job_dir.name, job_dir)

        min_speakers = pipeline_config.get("min_speakers")
        max_speakers = pipeline_config.get("max_speakers")
        speakers = (
            f"{min_speakers}–{max_speakers}"
            if min_speakers is not None and max_speakers is not None
            else "—"
        )

        result_path = job_dir / "export" / "result.md"
        complete = result_path.exists()

        summaries.append(
            JobSummary(
                job_id=job_dir.name,
                job_dir=job_dir,
                input_name=input_name,
                date_label=date.strftime("%Y-%m-%d %H:%M"),
                sort_key=date,
                language=pipeline_config.get("language", "—"),
                speakers=speakers,
                asr_model=pipeline_config.get("asr_model", "—"),
                stage_done=stage_done(job_dir),
                result_path=result_path if complete else None,
                mode=job_mode(meta),
                # Without a recorded input the convert stage never finished -
                # there is nothing to resume from; start a new job instead.
                resumable=not complete and bool(input_path),
            )
        )

    summaries.sort(key=lambda s: s.sort_key, reverse=True)
    return summaries


def _recorded_run_kwargs(job_dir: Path, meta: dict) -> dict:
    """run_pipeline() kwargs that reproduce the job's recorded settings."""
    config = meta.get("pipeline_config") or {}
    kwargs = {
        "job_dir": job_dir,
        "out_dir": job_dir.parent,
        "input_path": meta.get("input_path"),
    }
    for key in RECORDED_RUN_KEYS:
        if key in config:
            kwargs[key] = config[key]
    if "min_turn_duration" not in kwargs:
        # Jobs from before it was recorded in run.json: the merge stage wrote it.
        merged = _read_json(job_dir / "merged" / "segments.json")
        value = (merged.get("metadata") or {}).get("min_turn_duration")
        if value is not None:
            kwargs["min_turn_duration"] = value
    # The full recorded format, not its name: a user profile may have been
    # renamed or deleted since, and the job should keep what it was run with.
    recorded_format = (meta.get("config") or {}).get("audio_format")
    if isinstance(recorded_format, dict):
        try:
            kwargs["audio_format"] = AudioFormat.from_dict(recorded_format)
        except (TypeError, ValueError):
            pass
    return kwargs


# [SEMANTIC-BEGIN] GUI:JOB_PLANS
# @purpose: Build run_pipeline() arguments to continue or partly re-run an existing job
# @description: resume_plan() finishes an interrupted job: it starts at the first stage (of the job's
#   recorded stage list - with or without diarize) whose artifact is missing, so finished stages are
#   skipped outright, not even cache-checked (a crash during diarization never re-runs ASR).
#   rediarize_plan() re-runs diarize -> merge -> export with a new speaker range, keeping the
#   transcript; it also adds speakers to a "recognition only" job. Both reuse the recorded
#   settings (RECORDED_RUN_KEYS, the full audio format) so the job behaves like the original run.
# @inputs: job_dir (meta/run.json, artifacts present on disk)
# @outputs: (mode, kwargs) for PipelineWorker, or None when there is nothing to do
# @sideEffects: Reads job files only
# @errors: None (unreadable files count as missing)
# @see: PIPELINE:RUNNER, STAGE:CONVERT, GUI:PIPELINE_WORKER
def resume_plan(job_dir: Path) -> Optional[tuple]:
    """(mode, kwargs) that finish an interrupted job, or None if it is
    complete or never got past convert (nothing recorded to resume from)."""
    meta = _read_meta(job_dir)
    if not meta.get("input_path"):
        return None
    mode = job_mode(meta)
    order = FULL_STAGES if mode == "full" else ASR_ONLY_STAGES
    done = stage_done(job_dir)
    missing = [name for name in order if not done[name]]
    if not missing:
        return None
    kwargs = _recorded_run_kwargs(job_dir, meta)
    kwargs["from_stage"] = missing[0]
    return mode, kwargs


def rediarize_plan(job_dir: Path, min_speakers: int, max_speakers: int) -> Optional[tuple]:
    """(mode, kwargs) that re-run diarization with a new speaker range and
    rebuild merge/export, or None if the job has no transcript to work from."""
    meta = _read_meta(job_dir)
    if not meta.get("input_path") or not (job_dir / "asr" / "transcript.json").exists():
        return None
    kwargs = _recorded_run_kwargs(job_dir, meta)
    kwargs.update(
        from_stage="diarize",
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        # Old names belong to the old speaker ids; new diarization renumbers.
        speakers=None,
    )
    return "full", kwargs
# [SEMANTIC-END] GUI:JOB_PLANS
