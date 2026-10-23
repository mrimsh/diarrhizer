"""A dialogue with a model about one finished job: the transcript is its context, each turn a question and an answer.

A summary is the special case of one question. Dialogues live in the job's chat/ folder (beside summary/), one
JSON file each, any number per job. What a turn sends is always the same prefix - the template's instruction
with the transcript in it, kept in the file as it was built when the dialogue was created - followed by the
turns so far and the new question, so a provider's prompt cache can serve the prefix on every turn.
"""

import copy
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from diarrhizer.adapters import llm
from diarrhizer.adapters.llm import Block, Message
from diarrhizer.summary import service, wildcards
from diarrhizer.export.turns import format_timestamp
from diarrhizer.summary.providers import KIND_CODEX_CLI, Provider
from diarrhizer.summary.templates import PromptTemplate
from diarrhizer.utils import read_json, write_json_atomic, write_text_atomic

logger = logging.getLogger("diarrhizer.summary")

CHAT_DIR = "chat"
# The question of a dialogue's first turn when none is given: the template's instruction is already in the prefix.
FIRST_QUESTION = "Выполни инструкцию из шаблона."
_ID_RE = re.compile(r"^[\w-]+$")
_TITLE_CHARS = 60
# The size of a text in tokens is estimated, not counted: characters / CHARS_PER_TOKEN, rounded up. Cyrillic runs
# about 2-3 characters to a token and English about 4; 3 is the compromise, a little high for English, which is the
# safe side for a limit check. The estimate is for comparing with a limit, nothing else.
CHARS_PER_TOKEN = 3.0
_FENCE_OPEN, _FENCE_CLOSE = "<transcript>\n", "\n</transcript>"
_ENTRY_START = re.compile(r"^\[(\d+):(\d\d):(\d\d)")  # a transcript line begins "[HH:MM:SS → HH:MM:SS] **Name:** ..."


def estimate_tokens(chars: int) -> int:
    """About how many tokens `chars` characters of text are (see CHARS_PER_TOKEN)."""
    return math.ceil(chars / CHARS_PER_TOKEN)


@dataclass
class ContextOption:
    """One way out of a context that is too long, for the caller to offer."""

    kind: str  # "window" | "drop_turns" | "switch_model"
    label: str  # English, for the CLI; an interface words it itself from kind and data
    data: dict = field(default_factory=dict)  # window: {"start", "end", "kept", "of"}; drop_turns: {"count"}


@dataclass
class ContextCheck:
    estimated_tokens: int
    limit: Optional[int]  # None: no limit known, nothing is checked
    options: list = field(default_factory=list)  # of ContextOption, when the context is too long

    @property
    def exceeds(self) -> bool:
        return self.limit is not None and self.estimated_tokens > self.limit


class ContextTooLong(ValueError):
    """The request would go over the model's context limit; nothing was sent. `options` are ways to shrink it."""

    def __init__(self, check: ContextCheck, provider_name: str):
        self.check = check
        super().__init__(
            f"The context is about {check.estimated_tokens:,} tokens, over the {check.limit:,} set for "
            f"'{provider_name}'. Options: " + "; ".join(option.label for option in check.options)
        )


def _provider_dict(provider: Provider) -> dict:
    return {"name": provider.name, "kind": provider.kind, "model": provider.model}


@dataclass
class Turn:
    """One question and its answer, with what the request used."""

    question: str
    answer: str
    asked_at: str
    provider: dict  # {name, kind, model} of the provider that answered - it may differ from turn to turn
    model: str = ""  # the model that answered, when the provider says
    duration_seconds: float = 0.0
    truncated: bool = False
    usage: Optional[dict] = None  # Usage.to_dict(): tokens, with the prompt cache's parts where reported
    cost_usd: Optional[float] = None
    cost_note: str = ""
    context_mode: str = ""  # llm.CONTEXT_SESSION / llm.CONTEXT_FULL_HISTORY: what was sent
    context_chars: Optional[int] = None
    cache_available: Optional[bool] = None
    estimated_tokens: Optional[int] = None  # the request's size before sending, by estimate_tokens()
    context_limit: Optional[int] = None  # the limit it was checked against (None = not checked)
    truncation: list = field(default_factory=list)  # what this request left out, if anything - never silent

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "Turn":
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        return cls(**known)


@dataclass
class Conversation:
    id: str
    title: str
    template: str  # the template's name; its prompt is in `prefix`, as it was
    provider: dict  # the provider of the last turn (of the creation, before the first)
    note: str  # fixed when the dialogue was created
    created_at: str
    updated_at: str
    prefix: list = field(default_factory=list)  # [{"text", "cache"}]: the system message, never rebuilt
    turns: list = field(default_factory=list)  # of Turn
    warnings: list = field(default_factory=list)
    session_id: str = ""  # a CLI session that holds the turns so far
    session_kind: str = ""  # the provider kind that session belongs to
    session_turns: int = 0  # how many turns that session has seen: it is resumed only if it saw all of them
    omitted_turns: int = 0  # the oldest turns that are no longer sent (they stay stored and shown)
    window: Optional[dict] = None  # {"start", "end", "kept", "of"}: the transcript in the prefix was cut to a time window

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        data["turns"] = [turn.to_dict() for turn in self.turns]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        known = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        known["turns"] = [Turn.from_dict(turn) for turn in data.get("turns") or []]
        return cls(**known)

    def system_message(self) -> Message:
        return Message("system", [Block(block["text"], bool(block.get("cache"))) for block in self.prefix])

    def messages(self, question: str) -> list[Message]:
        """The request: the prefix, the turns so far (but the omitted oldest ones), then the new question."""
        messages = [self.system_message()]
        for turn in self.turns[self.omitted_turns:]:
            messages += [Message("user", turn.question), Message("assistant", turn.answer)]
        messages.append(Message("user", question))
        return messages


def _path(job_dir: Path, conversation_id: str) -> Path:
    if not _ID_RE.match(conversation_id):
        raise ValueError(f"Not a dialogue id: {conversation_id!r}")
    return job_dir / CHAT_DIR / f"{conversation_id}.json"


def _save(job_dir: Path, conversation: Conversation) -> None:
    write_json_atomic(_path(job_dir, conversation.id), conversation.to_dict())


def _note_precedes_transcript(prompt: str, note: str) -> bool:
    """Is the note in the prefix ahead of the transcript - where the cache of one dialogue cannot be shared."""
    if not note.strip():
        return False
    first = {}
    for ref in wildcards.find_wildcards(prompt):
        first.setdefault(ref.name, ref.start)
    if "note" in first and "transcript" in first:
        return first["note"] < first["transcript"]
    return "note" not in first and "transcript" not in first  # the note is appended before the appended transcript


# [SEMANTIC-BEGIN] SUMMARY:CHAT
# @purpose: A multi-turn dialogue with a model about a job's transcript, stored with the job
# @description: create_conversation() builds the prefix once - the template's prompt with its wildcards filled
#   in (SUMMARY:WILDCARD_BLOCKS), the transcript a block of its own marked as a cache breakpoint - and keeps it,
#   with the note (fixed at creation), in chat/<id>.json; ask() sends prefix + earlier turns + the new question
#   through llm.complete_chat and appends the turn (question, answer, time, provider and model, usage with the
#   cache parts, cost, what was sent: session or whole history). The prefix is never rebuilt, so it is
#   byte-for-byte the same on every turn - even if the template or result.md changes afterwards - which is what
#   a provider's prompt cache needs; a note that stands before the transcript makes the prefix differ between
#   dialogues, and says so in a warning. Provider and model may change from turn to turn (each turn records its
#   own). A Codex CLI session is kept and resumed only while it has seen every turn; otherwise the whole
#   history is sent and a new session starts. A failed request adds nothing. Files are written atomically.
#   Before sending, the request's size is estimated (estimate_tokens) and compared with the provider's
#   context_tokens (none set = not checked); over it ask() raises ContextTooLong with the ways out - a transcript
#   window, fewer old turns, another model - and sends nothing; taking one is asking again with window= /
#   drop_turns=. What is cut stays recorded (the dialogue's window / omitted_turns, every turn's truncation).
#   list/load/delete work on the chat/ folder; export_markdown() renders a dialogue as Markdown.
# @inputs: job_dir with export/result.md, PromptTemplate, Provider, optional note, questions
# @outputs: chat/<id>.json in the job folder; Conversation / Turn objects; Markdown text
# @sideEffects: Sends the transcript to the provider (as a summary does); writes and deletes files in chat/
# @errors: FileNotFoundError/ValueError (no or empty result.md, unknown dialogue, unknown wildcard, bad id), adapters.llm.LLMError
# @see: SUMMARY:JOB, SUMMARY:WILDCARD_BLOCKS, ADAPTER:LLM, CLI:CHAT
def create_conversation(
    job_dir: str | Path, template: PromptTemplate, provider: Provider, note: str | None = None, title: str = ""
) -> Conversation:
    job_dir = Path(job_dir)
    transcript = service.read_transcript(job_dir)
    note = (note or "").strip()
    rendered = wildcards.render_prompt_blocks(
        template.prompt, wildcards.PromptContext(transcript, note, job_dir), template.name
    )
    last_transcript = max(i for i, block in enumerate(rendered.blocks) if block.transcript)
    prefix = [
        {"text": block.text, "cache": i == last_transcript} for i, block in enumerate(rendered.blocks)
    ]
    warnings = list(rendered.warnings)
    if _note_precedes_transcript(template.prompt, note):
        warnings.append(
            "The note stands before the transcript: the prompt cache works only within this dialogue "
            "(another dialogue with another note has another prefix)"
        )
    for warning in warnings:
        logger.warning("[chat] Warning: %s", warning)
    now = datetime.now()
    conversation = Conversation(
        id=f"{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}",
        title=title.strip(),
        template=template.name,
        provider=_provider_dict(provider),
        note=note,
        created_at=now.isoformat(timespec="seconds"),
        updated_at=now.isoformat(timespec="milliseconds"),  # finer than a turn's time: the order of "last used"
        prefix=prefix,
        warnings=warnings,
    )
    _save(job_dir, conversation)
    logger.info("[chat] %s: dialogue %s, template '%s', transcript %d chars", job_dir.name, conversation.id,
                template.name, len(transcript))
    return conversation


def ask(
    job_dir: str | Path,
    conversation_id: str,
    question: str,
    provider: Provider,
    keep_cli_session: bool = True,
    context_limit: int | None = None,
    drop_turns: int = 0,
    window: tuple[float, float] | None = None,
) -> Turn:
    """Ask the next question of `provider`; returns the new turn (it is saved with the dialogue).

    The provider (or its model) may differ from the earlier turns': each turn records its own. An empty question
    is allowed only as the first one (FIRST_QUESTION: carry out the template's instruction).

    The request's size is estimated first. Over the limit (`context_limit`, else the provider's context_tokens;
    none = no check) nothing is sent: ContextTooLong carries the options. Taking one is asking again with
    `drop_turns` (leave that many more of the oldest turns out from now on) or `window` ((start, end) seconds:
    keep only that part of the transcript from now on), or with another provider. What is left out is recorded
    in the turn's `truncation` and stays in the dialogue's omitted_turns / window.

    Raises:
        ContextTooLong: the context is over the limit
        FileNotFoundError/ValueError: no such dialogue, an empty question, a bad window
        adapters.llm.LLMError: the provider failed (no turn is added)
    """
    job_dir = Path(job_dir)
    conversation, question, check = _prepare(
        load_conversation(job_dir, conversation_id), question, provider, context_limit, drop_turns, window
    )
    if check.exceeds:
        raise ContextTooLong(check, provider.name)
    messages = conversation.messages(question)
    # A session holds the whole conversation, so it cannot serve a dialogue that has had something cut out of it:
    # such a dialogue is sent whole each time, without keeping sessions.
    cut = bool(conversation.omitted_turns or conversation.window)
    sessions = keep_cli_session and provider.kind == KIND_CODEX_CLI and not cut
    resumable = (
        sessions and conversation.session_id and conversation.session_kind == provider.kind
        and conversation.session_turns == len(conversation.turns)
    )
    started = time.monotonic()
    reply = llm.complete_chat(
        provider, messages, keep_session=sessions, session_id=conversation.session_id if resumable else ""
    )
    duration = time.monotonic() - started

    turn = Turn(
        question=question,
        answer=reply.text,
        asked_at=datetime.now().isoformat(timespec="seconds"),
        provider=_provider_dict(provider),
        model=reply.model or provider.model,
        duration_seconds=round(duration, 1),
        truncated=reply.truncated,
        usage=reply.usage.to_dict() if reply.usage else None,
        cost_usd=reply.cost_usd,
        cost_note=reply.cost_note,
        context_mode=reply.context_mode,
        context_chars=reply.context_chars,
        cache_available=reply.cache_available,
        estimated_tokens=check.estimated_tokens,
        context_limit=check.limit,
        truncation=truncation_notes(conversation),
    )
    if sessions and reply.session_id:
        conversation.session_id, conversation.session_kind = reply.session_id, provider.kind
        # A resumed session saw one more turn; a new one was sent the whole history and saw them all.
        conversation.session_turns = (
            conversation.session_turns + 1 if reply.context_mode == llm.CONTEXT_SESSION else len(conversation.turns) + 1
        )
    conversation.turns.append(turn)
    conversation.provider = _provider_dict(provider)
    conversation.updated_at = datetime.now().isoformat(timespec="milliseconds")
    if not conversation.title:
        conversation.title = question[:_TITLE_CHARS]
    _save(job_dir, conversation)
    return turn


def _entries(text: str) -> tuple[list[str], list[tuple[float, str]]]:
    """A transcript as (lines before the first timestamp, [(start seconds, the entry's lines)]).

    An entry is a line that begins with a [HH:MM:SS timestamp and the lines after it that do not.
    """
    head: list[str] = []
    entries: list[tuple[float, list[str]]] = []
    for line in text.split("\n"):
        stamp = _ENTRY_START.match(line)
        if stamp:
            hours, minutes, seconds = (int(part) for part in stamp.groups())
            entries.append((hours * 3600 + minutes * 60 + seconds, [line]))
        elif entries:
            entries[-1][1].append(line)
        else:
            head.append(line)
    return head, [(start, "\n".join(lines)) for start, lines in entries]


def _transcript_texts(conversation: Conversation) -> list[tuple[int, str]]:
    """[(block index, transcript text)] of the prefix's transcript blocks (the text between the fence tags)."""
    found = []
    for i, block in enumerate(conversation.prefix):
        text = block["text"]
        if text.startswith(_FENCE_OPEN) and text.endswith(_FENCE_CLOSE):
            found.append((i, text[len(_FENCE_OPEN):-len(_FENCE_CLOSE)]))
    return found


def _apply_window(conversation: Conversation, start: float, end: float) -> None:
    """Cut the transcript in the (in-memory) prefix to the entries that begin in [start, end)."""
    if not 0 <= start < end:
        raise ValueError(f"A transcript window needs 0 <= start < end, got {start}..{end}")
    kept = total = 0
    for i, text in _transcript_texts(conversation):
        head, entries = _entries(text)
        if not entries:
            raise ValueError("The transcript has no timestamps, so it cannot be cut to a window")
        chosen = [lines for stamp, lines in entries if start <= stamp < end]
        if not chosen:
            raise ValueError(f"No transcript lines begin between {format_timestamp(start)} and {format_timestamp(end)}")
        kept, total = len(chosen), len(entries)
        body = "\n".join([*head, *chosen])
        conversation.prefix[i] = {**conversation.prefix[i], "text": f"{_FENCE_OPEN}{body}{_FENCE_CLOSE}"}
    conversation.window = {"start": start, "end": end, "kept": kept, "of": total}


def truncation_notes(conversation: Conversation) -> list[dict]:
    """What is left out of a request to this dialogue: the omitted turns and the cut-off transcript, if any."""
    notes = []
    if conversation.omitted_turns:
        notes.append({"kind": "omitted_turns", "count": conversation.omitted_turns, "of": len(conversation.turns)})
    if conversation.window:
        notes.append({"kind": "window", **conversation.window})
    return notes


def _size(conversation: Conversation, question: str) -> int:
    return estimate_tokens(sum(len(message.text) for message in conversation.messages(question)))


def _options(conversation: Conversation, question: str, limit: int) -> list[ContextOption]:
    """The ways to bring a request under `limit`: a transcript window, fewer old turns, another model."""
    options = []
    sent = conversation.turns[conversation.omitted_turns:]
    # Fewer old turns: the fewest of the oldest that brings it under the limit (the transcript stays).
    over = _size(conversation, question) - limit
    saved = 0
    for count, turn in enumerate(sent, 1):
        saved += estimate_tokens(len(turn.question) + len(turn.answer))
        if saved >= over:
            options.append(ContextOption(
                "drop_turns", f"leave out the {count} oldest turn(s) (the transcript stays)", {"count": count}))
            break
    # A transcript window: its beginning, as much as fits next to everything else.
    for _i, text in _transcript_texts(conversation)[:1]:
        head, entries = _entries(text)
        if len(entries) < 2:
            break
        room = estimate_tokens(len(text)) - over
        used = estimate_tokens(len("\n".join(head)))
        kept = 0
        for _stamp, lines in entries:
            used += estimate_tokens(len(lines) + 1)
            if used > room:
                break
            kept += 1
        if kept >= 1 and kept < len(entries):
            end = entries[kept][0]
            minutes = round(end / 60)
            span = service.format_window(minutes) if minutes else f"{int(end)} s"
            options.append(ContextOption(
                "window",
                f"cut the transcript to its first {span} ({format_timestamp(0)}-{format_timestamp(end)}, "
                f"{kept} of {len(entries)} lines)",
                {"start": 0.0, "end": float(end), "kept": kept, "of": len(entries)},
            ))
    options.append(ContextOption("switch_model", "use a model with a larger context", {}))
    return options


def _prepare(
    conversation: Conversation, question: str, provider: Provider, context_limit: int | None, drop_turns: int,
    window: "tuple[float, float] | None",
) -> "tuple[Conversation, str, ContextCheck]":
    """The dialogue as it would be sent (a copy: nothing is changed or saved), the question, and the size check."""
    question = question.strip()
    if not question:
        if conversation.turns:
            raise ValueError("The question is empty")
        question = FIRST_QUESTION
    conversation = copy.deepcopy(conversation)
    if window is not None:
        _apply_window(conversation, *window)
    if drop_turns:
        if not 0 < drop_turns <= len(conversation.turns) - conversation.omitted_turns:
            raise ValueError(f"Cannot leave out {drop_turns} turn(s): there are "
                             f"{len(conversation.turns) - conversation.omitted_turns} sent")
        conversation.omitted_turns += drop_turns
    limit = context_limit if context_limit is not None else provider.context_tokens
    estimated = _size(conversation, question)
    check = ContextCheck(estimated, limit)
    if check.exceeds:
        check.options = _options(conversation, question, limit)
    return conversation, question, check


def check_context(
    job_dir: str | Path, conversation_id: str, question: str, provider: Provider,
    context_limit: int | None = None, drop_turns: int = 0, window: "tuple[float, float] | None" = None,
) -> ContextCheck:
    """The size of the next request and, if it is over the limit, the options - without sending anything.

    Takes the same arguments as ask(), so an interface can check what it is about to send.
    """
    _conversation, _question, check = _prepare(
        load_conversation(job_dir, conversation_id), question, provider, context_limit, drop_turns, window
    )
    return check


def conversation_totals(conversation: Conversation) -> dict:
    """What the whole dialogue used: turns, input / output tokens, cache parts and cost (where reported)."""
    totals = {"turns": len(conversation.turns), "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
              "cache_write_tokens": 0, "cost_usd": None}
    for turn in conversation.turns:
        usage = turn.usage or {}
        for key in ("input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens"):
            totals[key] += usage.get(key) or 0
        if turn.cost_usd is not None:
            totals["cost_usd"] = (totals["cost_usd"] or 0.0) + turn.cost_usd
    return totals


def quick_question(template: PromptTemplate, job_dir: str | Path, note: str = "") -> str:
    """A template as a question for an open dialogue: its instruction with the wildcards filled in, no transcript."""
    ctx = wildcards.PromptContext("", note, Path(job_dir))
    return wildcards.render_instruction(template.prompt, ctx, template.name)


def load_conversation(job_dir: str | Path, conversation_id: str) -> Conversation:
    """The dialogue, from its file.

    Raises:
        FileNotFoundError: no such dialogue in the job
        ValueError: a bad id, or a file that is not a dialogue
    """
    path = _path(Path(job_dir), conversation_id)
    if not path.exists():
        raise FileNotFoundError(f"Dialogue not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("prefix"), list):
        raise ValueError(f"{path} is not a readable dialogue")
    return Conversation.from_dict(data)


def list_conversations(job_dir: str | Path) -> list[Conversation]:
    """The job's dialogues, the most recently used first; unreadable files are left out."""
    conversations = []
    for path in (Path(job_dir) / CHAT_DIR).glob("*.json"):
        try:
            conversations.append(load_conversation(job_dir, path.stem))
        except (ValueError, TypeError, FileNotFoundError):
            continue
    return sorted(conversations, key=lambda c: (c.updated_at, c.id), reverse=True)


def delete_conversation(job_dir: str | Path, conversation_id: str) -> None:
    """Remove a dialogue's file (and its exported Markdown, if there is one).

    Raises:
        FileNotFoundError: no such dialogue
    """
    path = _path(Path(job_dir), conversation_id)
    if not path.exists():
        raise FileNotFoundError(f"Dialogue not found: {path}")
    path.unlink()
    path.with_suffix(".md").unlink(missing_ok=True)
# [SEMANTIC-END] SUMMARY:CHAT


def turn_stats(turn: Turn) -> list[str]:
    """What a turn used, one line each: tokens (with the cache parts), cost, and what was sent."""
    lines = service.describe_usage({"usage": turn.usage, "cost_usd": turn.cost_usd, "cost_note": turn.cost_note})
    if turn.context_mode:
        sent = f", {turn.context_chars:,} chars" if turn.context_chars is not None else ""
        lines.append(f"Context: {turn.context_mode}{sent}")
    return lines


def export_markdown(conversation: Conversation) -> str:
    """The dialogue as Markdown: a header, then each question, its answer and what the turn used."""
    provider = conversation.provider
    lines = [
        f"# {conversation.title or conversation.id}",
        "",
        f"- Template: {conversation.template}",
        f"- Provider: {provider.get('name', '?')} ({provider.get('model') or 'default model'})",
        f"- Started: {conversation.created_at}",
    ]
    if conversation.note:
        lines += ["", "> " + conversation.note.replace("\n", "\n> ")]
    for number, turn in enumerate(conversation.turns, 1):
        who = turn.provider.get("name", "?")
        lines += ["", f"## {number}. {turn.question}", "", f"*{turn.asked_at} · {who} · {turn.model or '?'}*", "",
                  turn.answer]
        stats = turn_stats(turn)
        if stats:
            lines += ["", "<sub>" + " · ".join(stats) + "</sub>"]
    return "\n".join(lines) + "\n"


def export_conversation(job_dir: str | Path, conversation_id: str, path: str | Path | None = None) -> Path:
    """Write the dialogue as Markdown (default: chat/<id>.md) and return the file."""
    job_dir = Path(job_dir)
    target = Path(path) if path else _path(job_dir, conversation_id).with_suffix(".md")
    write_text_atomic(target, export_markdown(load_conversation(job_dir, conversation_id)))
    return target
