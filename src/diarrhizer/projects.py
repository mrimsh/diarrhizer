"""Portable project/team settings, naming and job identity (stdlib only)."""

import json
import re
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from diarrhizer.utils import write_json_atomic, write_text_atomic

PROJECT_FILE = ".diarrhizer-project.json"
TEAM_FILE = ".diarrhizer-team.json"
ROOT_FILE = ".diarrhizer-library.json"
JOB_FILE = "job.json"
DEFAULT_TEMPLATE = "PROJECT-YYYY_MM_DD-TEAM"
STORAGE_MODES = ("original", "template", "discard")
DEFAULT_STORAGE = {"original": "discard", "working": "original", "archive": "original"}
_TOKEN = re.compile(r"\{\{\s*(project|team|date|custom|source)\s*\}\}|PROJECT|TEAM|YYYY_MM_DD|CUSTOM|SOURCE", re.IGNORECASE)
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.IGNORECASE)


# [SEMANTIC-BEGIN] LIBRARY:SCOPES
# @purpose: Keep portable projects, teams, defaults and explicit participant lists beside their jobs
# @description: Marker files distinguish scopes from legacy jobs. Project participants belong to its
#   root team; named teams have independent lists. Names are single Windows-safe path components.
#   Renames hold the shared project lock and move the scope directory, preserving historical job titles.
# @inputs: results root, project/team names, settings and explicit participant additions/removals
# @outputs: scope paths and JSON settings; renamed directory on explicit request
# @sideEffects: Reads/writes JSON and creates or renames directories; no external services
# @errors: ValueError for malformed settings/unsafe names/collisions; OSError; RuntimeError if busy
# @see: LIBRARY:JOB_IDENTITY, ARTIFACTS:WRITE_LOCK
def read_config(file: Path) -> dict:
    try:
        data = json.loads(Path(file).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Cannot read {file}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{file}: expected an object")
    return data


def valid_name(name: str) -> str:
    if (not isinstance(name, str) or not name or name != name.strip() or len(name) > 120
            or name.endswith(".") or name in (".", "..") or _RESERVED.match(name)
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name.startswith(".")):
        raise ValueError(f"Недопустимое имя папки: {name!r}")
    return name


def _ordinary_dir(directory: Path) -> bool:
    return (directory.is_dir() and not directory.is_symlink()
            and not getattr(directory.stat(), "st_file_attributes", 0) & 0x400)


def scope_path(root: Path, project: str = "", team: str = "") -> Path:
    if team and not project:
        raise ValueError("Команда должна принадлежать проекту")
    directory = Path(root)
    for name in (project, team):
        if name:
            directory /= valid_name(name)
            if directory.exists() and not _ordinary_dir(directory):
                raise ValueError(f"Not an ordinary directory: {directory}")
    return directory


def scope_file(directory: Path) -> Path:
    for marker in (TEAM_FILE, PROJECT_FILE):
        if (directory / marker).is_file():
            return directory / marker
    return directory / ROOT_FILE


def list_scopes(directory: Path, *, teams: bool = False) -> list[str]:
    directory = Path(directory)
    if not directory.exists():
        return []
    marker = TEAM_FILE if teams else PROJECT_FILE
    return sorted((p.name for p in directory.iterdir() if _ordinary_dir(p) and (p / marker).is_file()),
                  key=str.casefold)


def ensure_scope(root: Path, project: str = "", team: str = "") -> Path:
    scope_path(root, project, team)  # validate the entire path before writing
    directory = Path(root)
    for name, marker in ((project, PROJECT_FILE), (team, TEAM_FILE)):
        if not name:
            continue
        directory /= name
        if directory.exists() and not (directory / marker).is_file():
            raise ValueError(f"Папка уже существует и не является проектом/командой: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / marker).exists():
            write_json_atomic(directory / marker, {"name": name, "participants": [],
                                                  **({"default_team": ""} if marker == PROJECT_FILE else {})})
    return directory


def save_scope(directory: Path, settings: dict) -> None:
    directory = Path(directory)
    data = {**read_config(scope_file(directory)), **settings}
    people = data.get("participants", [])
    if not isinstance(people, list) or not all(isinstance(p, str) and p.strip() for p in people):
        raise ValueError("Участники должны быть списком непустых имён")
    data["participants"] = list(dict.fromkeys(p.strip() for p in people))
    if data.get("default_team"):
        valid_name(data["default_team"])
    if "naming_template" in data:
        render_name(data["naming_template"], project="Проект", team="Команда", custom="Значение", source="Запись")
    if "storage" in data:
        data["storage"] = validate_storage(data["storage"])
    if scope_file(directory).name != ROOT_FILE:
        data["name"] = directory.name
    write_json_atomic(scope_file(directory), data)


def participants(directory: Path) -> list[str]:
    data = read_config(scope_file(Path(directory))).get("participants", [])
    if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
        raise ValueError(f"{directory}: invalid participants")
    return sorted(set(data), key=str.casefold)


def change_participant(directory: Path, name: str, *, remove: bool = False) -> None:
    name = name.strip()
    if not name:
        raise ValueError("Введите имя участника")
    # Use the same scope lock as settings/renames to avoid losing another edit.
    from diarrhizer.artifacts import job_write_lock
    with job_write_lock(Path(directory)):
        people = participants(directory)
        people = [p for p in people if p != name] if remove else list(dict.fromkeys([*people, name]))
        save_scope(directory, {"participants": people})


def job_context(job_dir: Path) -> tuple[Path | None, Path | None]:
    parent = Path(job_dir).parent
    if (parent / TEAM_FILE).is_file() and (parent.parent / PROJECT_FILE).is_file():
        return parent.parent, parent
    if (parent / PROJECT_FILE).is_file():
        return parent, None
    return None, None


def iter_job_dirs(root: Path):
    root = Path(root)
    if not root.exists():
        return
    for child in root.iterdir():
        if child.name.startswith(".") or not _ordinary_dir(child):
            continue
        if (child / PROJECT_FILE).is_file() or (child / TEAM_FILE).is_file():
            yield from iter_job_dirs(child)
        else:
            yield child


def rename_scope(directory: Path, name: str) -> Path:
    from diarrhizer.artifacts import job_write_lock
    directory = Path(directory)
    valid_name(name)
    marker = scope_file(directory)
    if marker.name == ROOT_FILE:
        raise ValueError("Выберите проект или команду")
    target = directory.with_name(name)
    if target == directory:
        return directory
    if target.exists():
        raise ValueError(f"Папка уже существует: {target}")
    # job_write_lock includes the outer project lock, which all descendant writers share.
    with job_write_lock(directory):
        settings = read_config(marker)
        directory.rename(target)
        try:
            write_json_atomic(target / marker.name, {**settings, "name": name})
            if marker.name == TEAM_FILE:
                project = read_config(target.parent / PROJECT_FILE)
                if project.get("default_team") == directory.name:
                    write_json_atomic(target.parent / PROJECT_FILE, {**project, "default_team": name})
        except Exception:
            target.rename(directory)
            write_json_atomic(directory / marker.name, settings)
            raise
    return target
# [SEMANTIC-END] LIBRARY:SCOPES


# [SEMANTIC-BEGIN] LIBRARY:MOVE_JOBS
# @purpose: Move a selected batch of jobs into a project/team without overwriting recordings or exports
# @description: Only discovered ordinary job directories under the library root are accepted. All source
#   locks are acquired before creating scopes; both destination and source job locks protect renames.
#   Folder collisions receive numeric suffixes; job identity, files, names and team rosters stay intact.
#   Recorded paths inside moved jobs are relocated. Failures roll moved folders and metadata back;
#   an incomplete rollback reports its remaining relocations so the GUI can keep live references correct.
# @inputs: library root, selected job paths, explicit project/team (empty means its parent root)
# @outputs: old-to-new directory mapping for jobs that changed location
# @sideEffects: Creates destination scopes and lock files; renames job folders; updates internal run paths
# @errors: ValueError for unsafe targets, OSError/RuntimeError for busy or failed moves; JobMoveError on incomplete rollback
# @see: LIBRARY:SCOPES, ARTIFACTS:WRITE_LOCK, GUI:MOVE_JOBS
class JobMoveError(RuntimeError):
    def __init__(self, message: str, moved: dict[Path, Path]):
        super().__init__(message)
        self.moved = moved


def _move_destination(root: Path, project: str, team: str) -> Path:
    destination = scope_path(root, project, team)
    directory = root
    for name, marker in ((project, PROJECT_FILE), (team, TEAM_FILE)):
        if name:
            directory /= name
            if directory.exists() and not (directory / marker).is_file():
                raise ValueError(f"Папка уже существует и не является проектом/командой: {directory}")
    return destination


def _relocate_recorded_path(value, moves: dict[Path, Path]):
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            for previous, current in moves.items():
                if path.is_relative_to(previous):
                    return str(current / path.relative_to(previous))
    return value


def _move_metadata(job: Path, moves: dict[Path, Path]) -> tuple[bytes, dict] | None:
    file = job / "meta" / "run.json"
    if not file.is_file():
        return None
    data = read_config(file)
    updated = dict(data)
    for key in ("input_path", "output_path"):
        if key in updated:
            updated[key] = _relocate_recorded_path(updated[key], moves)
    for key in ("output_paths", "archive_paths"):
        if isinstance(updated.get(key), list):
            updated[key] = [_relocate_recorded_path(p, moves) for p in updated[key]]
    if isinstance(updated.get("source_fingerprint"), dict):
        fingerprint = dict(updated["source_fingerprint"])
        if "path" in fingerprint:
            fingerprint["path"] = _relocate_recorded_path(fingerprint["path"], moves)
        updated["source_fingerprint"] = fingerprint
    return (file.read_bytes(), updated) if updated != data else None


def move_jobs(root: Path, jobs, *, project: str = "", team: str = "") -> dict[Path, Path]:
    from diarrhizer.artifacts import job_write_lock
    root = Path(root).resolve()
    destination = _move_destination(root, project, team)
    sources = []
    available = {p.resolve() for p in iter_job_dirs(root)}
    for job in jobs:
        job = Path(job)
        if not _ordinary_dir(job) or job.resolve() not in available:
            raise ValueError(f"Не является заданием в {root}: {job}")
        source = job.resolve()
        if not source.is_relative_to(root) or source == root:
            raise ValueError(f"Задание находится вне папки результатов: {job}")
        if source not in sources:
            sources.append(source)
    moves = {}
    for source in sources:
        if source.parent == destination:
            continue
        target = destination / source.name
        index = 2
        while target.exists() or target in moves.values():
            suffix = f"_{index}"
            target = destination / f"{source.name[:120 - len(suffix)]}{suffix}"
            index += 1
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"Папка назначения находится вне результатов: {target}")
        moves[source] = target
    if not moves:
        return {}
    with ExitStack() as locks:
        # Hold source locks even for no-op members: a busy selection is never partly moved.
        for source in sorted(sources):
            locks.enter_context(job_write_lock(source))
            if not _ordinary_dir(source):
                raise ValueError(f"Задание изменилось: {source}")
        # Locking a not-yet-created project creates only a sibling lock file.
        if project:
            locks.enter_context(job_write_lock(root / project))
        metadata = {source: _move_metadata(source, moves) for source in sources}
        ensure_scope(root, project, team)
        for target in sorted(moves.values()):
            locks.enter_context(job_write_lock(target))
        completed = {}
        rewritten = []
        try:
            for source, target in moves.items():
                if target.exists():
                    raise FileExistsError(f"Папка назначения уже существует: {target}")
                source.rename(target)
                completed[source] = target
            for source in sources:
                target = moves.get(source, source)
                if metadata[source] is not None:
                    _, updated = metadata[source]
                    rewritten.append(source)
                    write_json_atomic(target / "meta" / "run.json", updated)
        except Exception as exc:
            failures = []
            for source in reversed(rewritten):
                try:
                    # Restore the exact original record before returning its directory.
                    write_text_atomic(moves.get(source, source) / "meta" / "run.json", metadata[source][0].decode("utf-8"))
                except OSError as rollback_error:
                    failures.append(str(rollback_error))
            for source, target in reversed(list(completed.items())):
                try:
                    if source.exists():
                        raise FileExistsError(f"Не удалось вернуть задание: {source}")
                    target.rename(source)
                    del completed[source]
                except OSError as rollback_error:
                    failures.append(str(rollback_error))
            if failures:
                raise JobMoveError(f"Перенос прерван: {exc}. Не удалось полностью отменить изменения: "
                                   + "; ".join(failures), completed) from exc
            raise
    return moves
# [SEMANTIC-END] LIBRARY:MOVE_JOBS


# [SEMANTIC-BEGIN] LIBRARY:JOB_IDENTITY
# @purpose: Plan names without writes and reserve portable jobs with independently retained source files
# @description: Templates accept PROJECT/TEAM/YYYY_MM_DD/CUSTOM/SOURCE or {{project/team/date/custom/source}}.
#   Missing optional fields collapse adjacent separators. Job titles are fixed at creation; collisions
#   receive numeric suffixes. job.json stores relative paths so moving/renaming scopes needs no rewrites.
# @inputs: root, source path, project/team, template/custom text, per-type storage modes
# @outputs: job directory, immutable identity/settings, export and retained-media paths
# @sideEffects: reserve_job creates scope and job folders; other helpers only read or transform
# @errors: ValueError for invalid templates/storage/names; OSError for disk failures
# @see: LIBRARY:SCOPES, PIPELINE:EXECUTE, STAGE:CONVERT, STAGE:EXPORT
def render_name(template: str = DEFAULT_TEMPLATE, *, project: str = "", team: str = "",
                custom: str = "", source: str = "", date: str | None = None) -> str:
    values = {"project": project, "team": team, "custom": custom, "source": source,
              "date": date or datetime.now().strftime("%Y_%m_%d")}
    def substitute(match):
        key = (match.group(1) or match.group()).lower()
        return values["date" if key == "yyyy_mm_dd" else key]
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Шаблон названия не может быть пустым")
    name = _TOKEN.sub(substitute, template)
    if "{{" in name or "}}" in name:
        raise ValueError("Неизвестная подстановка в шаблоне")
    name = re.sub(r"[-_ ]{2,}", "-", name).strip("-_ ")
    return valid_name(name)


def validate_storage(storage: dict | None) -> dict:
    if storage is not None and not isinstance(storage, dict):
        raise ValueError("Настройки хранения должны быть объектом")
    data = {**DEFAULT_STORAGE, **(storage or {})}
    if set(data) != set(DEFAULT_STORAGE) or any(mode not in STORAGE_MODES for mode in data.values()):
        raise ValueError("Хранение: original, working, archive = original/template/discard")
    return data


def inherited_settings(root: Path, project: str = "", team: str = "") -> dict:
    result = {"naming_template": DEFAULT_TEMPLATE, "storage": dict(DEFAULT_STORAGE), "default_team": ""}
    paths = [Path(root) / ROOT_FILE]
    if project:
        paths.append(scope_path(root, project) / PROJECT_FILE)
    if team:
        paths.append(scope_path(root, project, team) / TEAM_FILE)
    for file in paths:
        data = read_config(file)
        for key in ("naming_template", "default_team"):
            if key in data:
                result[key] = data[key]
        modes = data.get("storage", {})
        if not isinstance(modes, dict):
            raise ValueError(f"{file}: storage must be an object")
        result["storage"].update(modes)
    result["storage"] = validate_storage(result["storage"])
    return result


def reserve_job(root: Path, source: Path, *, project: str = "", team: str | None = None,
                naming_template: str | None = None, custom: str = "", storage: dict | None = None) -> Path:
    defaults = inherited_settings(root, project)
    if storage is not None and not isinstance(storage, dict):
        raise ValueError("Настройки хранения должны быть объектом")
    team = defaults["default_team"] if team is None else team
    defaults = inherited_settings(root, project, team)
    template = defaults["naming_template"] if naming_template is None else naming_template
    modes = validate_storage({**defaults["storage"], **(storage or {})})
    source = Path(source)
    created = datetime.now()
    name = render_name(template, project=project, team=team, custom=custom, source=source.stem,
                       date=created.strftime("%Y_%m_%d"))
    parent = ensure_scope(root, project, team)
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / name
    index = 2
    while True:
        try:
            candidate.mkdir()
            break
        except FileExistsError:
            suffix = f"_{index}"
            candidate = parent / f"{name[:120 - len(suffix)]}{suffix}"
            index += 1
    try:
        write_json_atomic(candidate / JOB_FILE, {"title": candidate.name, "created_at": created.isoformat(),
                          "naming_template": template, "custom": custom, "source_name": source.name,
                          "storage": modes})
    except Exception:
        candidate.rmdir()
        raise
    return candidate


def export_path(job_dir: Path, extension: str) -> Path:
    job_dir = Path(job_dir)
    identity = read_config(job_dir / JOB_FILE)
    stem = valid_name(identity.get("title")) if identity else "result"
    return job_dir / "export" / f"{stem}.{extension.lstrip('.')}"


def media_path(job_dir: Path, kind: str, extension: str, suffix: str = "") -> Path:
    identity = read_config(Path(job_dir) / JOB_FILE)
    if not identity:
        stem = "normalized" if kind == "working" else "archive"
        return Path(job_dir) / "audio" / f"{stem}{suffix}.{extension}"
    mode = validate_storage(identity.get("storage"))[kind]
    if not isinstance(identity.get("source_name"), str):
        raise ValueError(f"{job_dir}: source_name must be text")
    stem = identity.get("title") if mode == "template" else Path(identity["source_name"]).stem
    valid_name(stem)
    return Path(job_dir) / ("original" if kind == "original" else f"audio/{kind}") / f"{stem}{suffix}.{extension}"


def resolve_input(job_dir: Path, source: str | Path) -> Path:
    source = Path(source)
    if not source.is_file():
        identity = read_config(Path(job_dir) / JOB_FILE)
        if identity and validate_storage(identity.get("storage"))["original"] != "discard":
            if not isinstance(identity.get("source_name"), str):
                raise ValueError(f"{job_dir}: source_name must be text")
            retained = media_path(job_dir, "original", Path(identity["source_name"]).suffix.lstrip("."))
            if retained.is_file():
                return retained
    return source
# [SEMANTIC-END] LIBRARY:JOB_IDENTITY
