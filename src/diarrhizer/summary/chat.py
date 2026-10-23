"""A dialogue with a model about one finished job: the transcript is its context, each turn a question and an answer.

A summary is the special case of one question. Dialogues live in the job's chat/ folder (beside summary/), one
JSON file each, any number per job. What a turn sends is always the same prefix - the template's instruction
with the transcript in it, kept in the file as it was built when the dialogue was created - followed by the
turns so far and the new question, so a provider's prompt cache can serve the prefix on every turn.
"""

import logging
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
from diarrhizer.summary.providers import KIND_CODEX_CLI, Provider
from diarrhizer.summary.templates import PromptTemplate
from diarrhizer.utils import read_json, write_json_atomic, write_text_atomic

logger = logging.getLogger("diarrhizer.summary")

CHAT_DIR = "chat"
# The question of a dialogue's first turn when none is given: the template's instruction is already in the prefix.
FIRST_QUESTION = "Выполни инструкцию из шаблона."
_ID_RE = re.compile(r"^[\w-]+$")
_TITLE_CHARS = 60


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
        """The request: the prefix, every turn so far, then the new question."""
        messages = [self.system_message()]
        for turn in self.turns:
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
        updated_at=now.isoformat(timespec="seconds"),
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
) -> Turn:
    """Ask the next question of `provider`; returns the new turn (it is saved with the dialogue).

    The provider (or its model) may differ from the earlier turns': each turn records its own. An empty question
    is allowed only as the first one (FIRST_QUESTION: carry out the template's instruction).
    """
    job_dir = Path(job_dir)
    conversation = load_conversation(job_dir, conversation_id)
    question = question.strip()
    if not question:
        if conversation.turns:
            raise ValueError("The question is empty")
        question = FIRST_QUESTION

    messages = conversation.messages(question)
    sessions = keep_cli_session and provider.kind == KIND_CODEX_CLI
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
    )
    if sessions and reply.session_id:
        conversation.session_id, conversation.session_kind = reply.session_id, provider.kind
        # A resumed session saw one more turn; a new one was sent the whole history and saw them all.
        conversation.session_turns = (
            conversation.session_turns + 1 if reply.context_mode == llm.CONTEXT_SESSION else len(conversation.turns) + 1
        )
    conversation.turns.append(turn)
    conversation.provider = _provider_dict(provider)
    conversation.updated_at = turn.asked_at
    if not conversation.title:
        conversation.title = question[:_TITLE_CHARS]
    _save(job_dir, conversation)
    return turn


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
