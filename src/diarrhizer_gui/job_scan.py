"""Scans an `out/` directory tree for Diarrhizer job folders.

Reads only the documented on-disk artifact contract (docs/architecture.md,
"Artifacts (results directory)") - not the heavier stage/adapter modules
(those pull in torch/whisperx at import time), so this stays cheap to import
and keeps working even on a machine without the ML stack installed.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# (stage name, artifact path relative to job_dir) - mirrors the layout
# documented in docs/architecture.md, not the stage classes themselves.
STAGE_ARTIFACTS = [
    ("convert", "audio/normalized.wav"),
    ("transcribe", "asr/transcript.json"),
    ("diarize", "diar/diarization.json"),
    ("merge", "merged/segments.json"),
    ("export", "export/result.md"),
]


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


def _parse_job_date(job_id: str, job_dir: Path) -> datetime:
    """Parse the `_YYYYMMDD_HHMMSS` suffix generate_job_id() always appends.

    Falls back to the folder's mtime if the id doesn't parse (e.g. a job
    folder created by something other than this tool).
    """
    parts = job_id.rsplit("_", 2)
    if len(parts) == 3:
        _, date_part, time_part = parts
        try:
            return datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(job_dir.stat().st_mtime)
    except OSError:
        return datetime.min


def _read_meta(job_dir: Path) -> dict:
    meta_path = job_dir / "meta" / "run.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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

        stage_done = {
            name: (job_dir / rel_path).exists() for name, rel_path in STAGE_ARTIFACTS
        }

        result_path = job_dir / "export" / "result.md"

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
                stage_done=stage_done,
                result_path=result_path if result_path.exists() else None,
            )
        )

    summaries.sort(key=lambda s: s.sort_key, reverse=True)
    return summaries
