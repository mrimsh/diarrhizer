"""Wildcards of a prompt template: {{name}} marks the place where a value goes into the prompt.

The author of a template decides where the transcript, the user's note and the job's facts (name, date,
duration, speakers, language) stand; the values come from one registry, WILDCARDS. Adding a wildcard means
adding one entry there (name, description, a function that returns the value): validation, the
`summarize --list-wildcards` listing and the error message that names the valid ones all read the registry.
(The README and docs/architecture.md describe the wildcards in prose; a test checks that they name every
registered one.)

Syntax: `{{name}}`, spaces inside the braces allowed. Single braces are never touched (prompts carry JSON),
nor is `{{` followed by anything that is not a name. A literal `{{` is written `\\{{`; a lone `}}` needs no
escaping.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path, PureWindowsPath
from typing import Callable

from diarrhizer.export.speakers import resolve_speaker_name
from diarrhizer.export.turns import format_timestamp
from diarrhizer.utils import read_json

# `\{{` (an escaped literal "{{", no group) or `{{ name }}` (group 1 = name; spaces and tabs only, so a
# wildcard never spans lines). A wildcard in the template is found once, left to right, and a value is never
# scanned again: a transcript that happens to contain "{{note}}" stays as it is.
_TOKEN = re.compile(r"\\\{\{|\{\{[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*\}\}")
# generate_job_id() ends every job folder name with _YYYYMMDD_HHMMSS; new_job_dir() adds _2, _3... on a clash.
_JOB_STAMP = re.compile(r"_(\d{8})_\d{6}(?:_\d+)?$")  # group 1 = the date


class PromptContext:
    """What wildcard values are taken from: this run's transcript and note, and the job's own files.

    A job file is read when a wildcard needs it, once. A file that is missing, unreadable or shaped
    differently than expected gives an empty answer - a wildcard never fails for lack of data.
    """

    def __init__(self, transcript: str, note: str = "", job_dir: str | Path | None = None):
        self.transcript = transcript
        self.note = (note or "").strip()
        self.job_dir = Path(job_dir) if job_dir is not None else None

    def read_job_json(self, relative_path: str) -> dict:
        data = read_json(self.job_dir / relative_path) if self.job_dir is not None else None
        return data if isinstance(data, dict) else {}

    @cached_property
    def run_meta(self) -> dict:
        """meta/run.json: the input file and the settings the job was started with."""
        return self.read_job_json("meta/run.json")

    @cached_property
    def segments(self) -> list[dict]:
        """merged/segments.json's phrases (start, end, speaker_id, text)."""
        segments = self.read_job_json("merged/segments.json").get("segments")
        return [seg for seg in segments if isinstance(seg, dict)] if isinstance(segments, list) else []

    @cached_property
    def speaker_names(self) -> dict[str, str]:
        """{speaker id: name} the user saved in speakers.json (blank names left out)."""
        return {
            str(key): value.strip()
            for key, value in self.read_job_json("speakers.json").items()
            if isinstance(value, str) and value.strip()
        }


@dataclass(frozen=True)
class Wildcard:
    name: str
    description: str  # English, for the CLI and the docs
    value: Callable[[PromptContext], str]
    description_ru: str = ""  # one short Russian line for the GUI's list (the GUI falls back to `description`)


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _transcript(ctx: PromptContext) -> str:
    # Fenced, so a model cannot mistake the speakers' words for instructions, and a prompt can refer to it by name.
    return f"<transcript>\n{ctx.transcript}\n</transcript>"


def _note(ctx: PromptContext) -> str:
    return ctx.note


def _job_name(ctx: PromptContext) -> str:
    input_path = _text(ctx.run_meta.get("input_path"))
    if input_path:
        return PureWindowsPath(input_path).stem  # recorded as the OS wrote it; either kind of slash works
    # No record: the folder is named <input stem>_<stamp>.
    return _JOB_STAMP.sub("", ctx.job_dir.name) if ctx.job_dir is not None else ""


def _date(ctx: PromptContext) -> str:
    # The pipeline does not record when the call itself took place, only when the job was made.
    if ctx.job_dir is not None:
        stamp = _JOB_STAMP.search(ctx.job_dir.name)
        if stamp:
            try:
                return datetime.strptime(stamp.group(1), "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                pass
    started = _text(ctx.run_meta.get("start_time"))
    try:
        return datetime.fromisoformat(started).strftime("%Y-%m-%d") if started else ""
    except ValueError:
        return ""


def _duration(ctx: PromptContext) -> str:
    ends = [seg["end"] for seg in ctx.segments if isinstance(seg.get("end"), (int, float))]
    return format_timestamp(max(ends)) if ends else ""


def _speakers(ctx: PromptContext) -> str:
    names: list[str] = []
    for seg in ctx.segments:
        name = resolve_speaker_name(str(seg.get("speaker_id") or "Speaker_00"), ctx.speaker_names)
        if name not in names:  # two diarization ids named as one person are one speaker
            names.append(name)
    if not names:  # no phrases on disk: the names the user saved are all there is
        names = list(dict.fromkeys(ctx.speaker_names.values()))
    return ", ".join(names)


def _language(ctx: PromptContext) -> str:
    pipeline = ctx.run_meta.get("pipeline_config")
    language = _text(pipeline.get("language")) if isinstance(pipeline, dict) else ""
    if language and language.lower() != "auto":
        return language
    # "auto": what Whisper detected, as the transcribe stage recorded it.
    return _text(ctx.read_job_json("asr/transcript.json").get("language"))


# [SEMANTIC-BEGIN] CONFIG:SUMMARY_WILDCARDS
# @purpose: The one list of wildcards a prompt template may use
# @description: name -> Wildcard(name, description, value function, a short Russian line for the GUI's list;
#   find_wildcards() reads the references out of a prompt against it). The value functions take a PromptContext
#   and return a string; job facts come from what the job already records (meta/run.json,
#   merged/segments.json, speakers.json, asr/transcript.json, the folder name) and are "" when that is
#   missing. To add a wildcard, add an entry here and name it in README.md and docs/architecture.md.
# @inputs: PromptContext
# @outputs: str per wildcard
# @sideEffects: Reads job files (read-only)
# @errors: None (missing data gives "")
# @see: SUMMARY:WILDCARDS, CONFIG:SUMMARY_TEMPLATES, CLI:SUMMARIZE, GUI:SUMMARY_SETTINGS
WILDCARDS: dict[str, Wildcard] = {
    wildcard.name: wildcard
    for wildcard in (
        Wildcard(
            "transcript",
            "The job's transcript (export/result.md, speaker names as saved) inside <transcript>...</transcript>. "
            "A template without it gets the transcript at its end.",
            _transcript,
            "расшифровка разговора; без неё в шаблоне она добавляется в конец запроса",
        ),
        Wildcard(
            "note",
            "Your own text for this run (--note / --note-file); empty when none was given.",
            _note,
            "ваша заметка к этому запуску; пустая, если не задана",
        ),
        Wildcard(
            "job_name",
            "Name of the input file without its extension (read from the job's folder name if the job does "
            "not record it).",
            _job_name,
            "имя входного файла без расширения",
        ),
        Wildcard(
            "date",
            "Date of the job as YYYY-MM-DD, from its folder name: when the recording was processed "
            "(the call's own date is not recorded).",
            _date,
            "дата задания ГГГГ-ММ-ДД (когда запись обработана)",
        ),
        Wildcard(
            "duration",
            "Length of the recording as HH:MM:SS, up to the end of the last phrase.",
            _duration,
            "длительность записи ЧЧ:ММ:СС",
        ),
        Wildcard(
            "speakers",
            "The speakers, comma-separated, under the names given in the job (as renamed in speakers.json).",
            _speakers,
            "спикеры через запятую, с заданными именами",
        ),
        Wildcard(
            "language",
            "Language of the recording: the one set for the job, else the one Whisper detected.",
            _language,
            "язык записи",
        ),
    )
}
# [SEMANTIC-END] CONFIG:SUMMARY_WILDCARDS


class UnknownWildcardError(ValueError):
    """A template names a wildcard that is not in WILDCARDS."""

    def __init__(self, name: str, template_name: str = "", line: int | None = None):
        self.name = name
        self.available = list(WILDCARDS)
        where = f" in line {line}" if line else ""
        where += f" of the template '{template_name}'" if template_name else ""
        super().__init__(
            f"Unknown wildcard {{{{{name}}}}}{where}. Available: "
            + ", ".join("{{" + known + "}}" for known in self.available)
            + ". To write a literal {{ in a prompt, put a backslash before it: \\{{"
        )


@dataclass(frozen=True)
class WildcardRef:
    """One {{name}} in a prompt: where it stands (character offsets; `end` is past the closing braces)."""

    name: str
    start: int
    end: int
    line: int  # 1-based
    known: bool  # is it in WILDCARDS


def find_wildcards(template_prompt: str) -> list[WildcardRef]:
    """Every {{name}} in the prompt in order of appearance, the unknown ones too (an editor wants them all)."""
    return [
        WildcardRef(
            match.group(1), match.start(), match.end(),
            template_prompt.count("\n", 0, match.start()) + 1, match.group(1) in WILDCARDS,
        )
        for match in _TOKEN.finditer(template_prompt)
        if match.group(1) is not None  # None = an escaped "{{"
    ]


def used_wildcards(template_prompt: str, template_name: str = "") -> list[str]:
    """The wildcards a prompt uses, each once, in order of first appearance.

    Raises:
        UnknownWildcardError: the prompt names a wildcard that does not exist (the first one)
    """
    used: list[str] = []
    for ref in find_wildcards(template_prompt):
        if not ref.known:
            raise UnknownWildcardError(ref.name, template_name, ref.line)
        if ref.name not in used:
            used.append(ref.name)
    return used


@dataclass
class RenderedPrompt:
    text: str
    wildcards_used: list[str] = field(default_factory=list)  # the ones the template names, not the implicit ones
    warnings: list[str] = field(default_factory=list)


# [SEMANTIC-BEGIN] SUMMARY:WILDCARDS
# @purpose: Turn a template's prompt into the request text: fill in {{wildcards}}, keep the transcript and the note
# @description: Every {{name}} is replaced by its value from WILDCARDS (the same wildcard may stand several
#   times; a value is not scanned for wildcards again; `\{{` gives a literal "{{"; single braces are
#   untouched). A name that is not registered fails before anything is sent (UnknownWildcardError, naming
#   it and the valid ones). Nothing the caller supplied is dropped silently: a template without
#   {{transcript}} gets the fenced transcript at its end, as templates always did; a note given to a
#   template without {{note}} is added after the template's text - before the appended transcript, or at
#   the very end when the template placed the transcript itself - and a warning says so.
# @inputs: template prompt, PromptContext (transcript, note, job folder), template name (for messages)
# @outputs: RenderedPrompt (text, wildcards named by the template, warnings)
# @sideEffects: Reads job files (read-only), through PromptContext
# @errors: UnknownWildcardError (a ValueError)
# @see: CONFIG:SUMMARY_WILDCARDS, SUMMARY:JOB, CONFIG:SUMMARY_TEMPLATES
def render_prompt(template_prompt: str, ctx: PromptContext, template_name: str = "") -> RenderedPrompt:
    used = used_wildcards(template_prompt, template_name)
    values = {name: WILDCARDS[name].value(ctx) for name in used}

    def substitute(match: re.Match) -> str:
        return "{{" if match.group(1) is None else values[match.group(1)]

    parts = [_TOKEN.sub(substitute, template_prompt).strip()]
    warnings: list[str] = []
    if ctx.note and "note" not in used:
        label = f"Template '{template_name}'" if template_name else "The template"
        where = "at the end of the prompt" if "transcript" in used else "before the transcript"
        warnings.append(f"{label} has no {{{{note}}}}: the note was added {where}")
        parts.append(ctx.note)
    if "transcript" not in used:
        parts.append(_transcript(ctx))
    return RenderedPrompt("\n\n".join(part for part in parts if part) + "\n", used, warnings)
# [SEMANTIC-END] SUMMARY:WILDCARDS


@dataclass(frozen=True)
class PromptBlock:
    text: str
    transcript: bool = False  # this block is the fenced transcript


@dataclass
class RenderedBlocks:
    blocks: list[PromptBlock]
    wildcards_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_PLACE_OF_TRANSCRIPT = "\x00"  # stands in for the transcript while the prompt is rendered, then is cut out


# [SEMANTIC-BEGIN] SUMMARY:WILDCARD_BLOCKS
# @purpose: The rendered prompt as blocks, the transcript being a block of its own
# @description: The same rendering as render_prompt (same rules, same warnings), cut at the transcript: the text
#   before it, the fenced transcript, the text after it (a template that places {{transcript}} several times
#   gives several transcript blocks). A dialogue sends the blocks as one system message and marks the transcript
#   block as a cache breakpoint; run together they are the render_prompt text, without its final newline.
# @inputs: template prompt, PromptContext, template name
# @outputs: RenderedBlocks(blocks, wildcards_used, warnings)
# @sideEffects: Reads job files (read-only), through PromptContext
# @errors: UnknownWildcardError (a ValueError)
# @see: SUMMARY:WILDCARDS, SUMMARY:CHAT
def render_prompt_blocks(template_prompt: str, ctx: PromptContext, template_name: str = "") -> RenderedBlocks:
    probe = PromptContext(_PLACE_OF_TRANSCRIPT, ctx.note, ctx.job_dir)
    rendered = render_prompt(template_prompt, probe, template_name)
    fenced = _transcript(probe)
    pieces = rendered.text.rstrip("\n").split(fenced)
    blocks: list[PromptBlock] = []
    for i, piece in enumerate(pieces):
        if piece:
            blocks.append(PromptBlock(piece))
        if i < len(pieces) - 1:
            blocks.append(PromptBlock(_transcript(ctx), transcript=True))
    return RenderedBlocks(blocks, rendered.wildcards_used, rendered.warnings)
# [SEMANTIC-END] SUMMARY:WILDCARD_BLOCKS
