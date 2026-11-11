"""Local recording timestamps, independent of processing time; no media/model dependencies."""

import re
from datetime import datetime
from pathlib import Path

# [SEMANTIC-BEGIN] JOB:RECORDING_TIME
# @purpose: Validate user recording time and offer conservative filename-derived suggestions
# @description: Case-sensitive atomic date/time tokens are shared by names and filename parsing (MM month,
#   mm minute). Explicit input formats escape literal separators; YY means 2000..2099. Auto detection
#   recognizes year-first separated/compact and dotted day-first dates with optional time.
#   Invalid dates and multiple distinct timestamps yield no suggestion.
#   Suggestions never write files or override explicit user values. All values are naive local wall time.
# @inputs: local ISO timestamp/datetime, input filename with optional date/time format, or job directory
# @outputs: validated datetime or optional proposal; legacy job date fallback without modifying metadata
# @sideEffects: Existing-job fallback reads JSON/stat only
# @errors: ValueError for invalid explicit local timestamps or input formats lacking a complete date
# @see: LIBRARY:JOB_IDENTITY, LIBRARY:MOVE_JOBS, GUI:RECORDING_TIME
DATE_TIME_FORMATS = {"YYYY": "%Y", "YY": "%y", "MM": "%m", "DD": "%d",
                     "HH": "%H", "mm": "%M", "SS": "%S"}
DATE_TIME_TOKEN = re.compile("|".join(DATE_TIME_FORMATS))


def local_datetime(value: str | datetime) -> datetime:
    try:
        result = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Укажите дату и время в формате YYYY-MM-DD HH:MM:SS") from exc
    if result.tzinfo is not None:
        raise ValueError("Укажите местные дату и время без часового пояса")
    return result.replace(microsecond=0)


def _input_pattern(template):
    parts, fields, end = [], [], 0
    for match in DATE_TIME_TOKEN.finditer(template):
        token = match.group()
        group = f"field{len(fields)}"
        parts.extend((re.escape(template[end:match.start()]),
                      rf"(?P<{group}>\d{{{4 if token == 'YYYY' else 2}}})"))
        fields.append((group, token))
        end = match.end()
    parts.append(re.escape(template[end:]))
    tokens = {token for _, token in fields}
    if not ({"MM", "DD"} <= tokens and tokens & {"YYYY", "YY"}):
        raise ValueError("Формат даты должен содержать YYYY или YY, MM (месяц) и DD (день); mm — минуты")
    return re.compile(r"(?<!\d)" + "".join(parts) + r"(?!\d)"), fields


def _timestamp(values):
    parts = {}
    for token, text in values:
        value = int(text)
        if token in parts and parts[token] != value:
            raise ValueError("Противоречивые значения даты/времени")
        parts[token] = value
    year = parts["YYYY"] if "YYYY" in parts else 2000 + parts["YY"]
    if "YY" in parts and year % 100 != parts["YY"]:
        raise ValueError("Противоречивые значения года")
    return datetime(year, parts["MM"], parts["DD"], parts.get("HH", 0),
                    parts.get("mm", 0), parts.get("SS", 0))


def suggest_recording_time(source: str | Path, template: str | None = None) -> datetime | None:
    name = Path(source).stem
    templates = (template,) if template else ("YYYY-MM-DD", "YYYY_MM_DD", "YYYY.MM.DD", "YYYYMMDD", "DD.MM.YYYY")
    found = set()
    invalid = False
    for date_template in templates:
        pattern, fields = _input_pattern(date_template)
        for match in pattern.finditer(name):
            values = [(token, match.group(group)) for group, token in fields]
            if not template:
                tail = name[match.end():]
                time = re.match(r"[ Tt_.-]+(\d{2})[-_: .](\d{2})(?:[-_: .](\d{2}))?(?!\d)", tail)
                if time is None:
                    time = re.match(r"[ Tt_.-]+(\d{2})(\d{2})(\d{2})(?!\d)", tail)
                if time:
                    values.extend(zip(("HH", "mm", "SS"), (v or "00" for v in time.groups())))
            try:
                found.add(_timestamp(values))
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
