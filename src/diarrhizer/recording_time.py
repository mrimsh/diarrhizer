"""Local recording timestamps, independent of processing time; no media/model dependencies."""

import re
from datetime import datetime
from pathlib import Path


# [SEMANTIC-BEGIN] JOB:RECORDING_TIME
# @purpose: Validate user recording time and offer conservative filename-derived suggestions
# @description: Recognizes year-first separated/compact dates and dotted day-first dates, optionally
#   followed by separated/compact time. Invalid dates and multiple distinct timestamps yield no suggestion.
#   Suggestions never write files or override explicit user values. All values are naive local wall time.
# @inputs: local ISO timestamp/datetime, input filename, or existing job directory
# @outputs: validated datetime or optional proposal; legacy job date fallback without modifying metadata
# @sideEffects: Existing-job fallback reads JSON/stat only
# @errors: ValueError for invalid explicit local timestamps
# @see: LIBRARY:JOB_IDENTITY, LIBRARY:MOVE_JOBS, GUI:RECORDING_TIME
def local_datetime(value: str | datetime) -> datetime:
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Укажите дату и время в формате YYYY-MM-DD HH:MM:SS") from exc
    if result.tzinfo is not None:
        raise ValueError("Укажите местные дату и время без часового пояса")
    return result.replace(microsecond=0)


def suggest_recording_time(source: str | Path) -> datetime | None:
    name = Path(source).stem
    patterns = (
        (r"(?<!\d)([12]\d{3})[-_.](\d{2})[-_.](\d{2})", False),
        (r"(?<!\d)([12]\d{3})(\d{2})(\d{2})", False),
        (r"(?<!\d)(\d{2})\.(\d{2})\.([12]\d{3})", True),
    )
    found = set()
    invalid = False
    for pattern, day_first in patterns:
        for match in re.finditer(pattern + r"(?!\d)", name):
            a, b, c = map(int, match.groups())
            year, month, day = (c, b, a) if day_first else (a, b, c)
            tail = name[match.end():]
            time = re.match(r"[ Tt_.-]+(\d{2})[-_: .](\d{2})(?:[-_: .](\d{2}))?(?!\d)", tail)
            if time is None:
                time = re.match(r"[ Tt_.-]+(\d{2})(\d{2})(\d{2})(?!\d)", tail)
            hour, minute, second = (int(v or 0) for v in time.groups()) if time else (0, 0, 0)
            try:
                found.add(datetime(year, month, day, hour, minute, second))
            except ValueError:
                invalid = True
    return next(iter(found)) if len(found) == 1 and not invalid else None


def job_recording_time(job: Path) -> datetime:
    from diarrhizer.projects import JOB_FILE, read_config
    identity = read_config(job / JOB_FILE)
    meta = read_config(job / "meta" / "run.json")
    value = identity.get("recorded_at") or identity.get("created_at") or meta.get("start_time")
    if value:
        return local_datetime(value)
    match = re.search(r"_(\d{8})_(\d{6})(?:_\d+)?$", job.name)
    return (datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S") if match
            else datetime.fromtimestamp(job.stat().st_mtime).replace(microsecond=0))
# [SEMANTIC-END] JOB:RECORDING_TIME
