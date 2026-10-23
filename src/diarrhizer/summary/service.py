"""Summarize a finished job: its export/result.md, a template and a provider.

A summary is an on-demand add-on to a job, not a pipeline stage: it costs
money or subscription quota, may run several times per job with different
templates, and never invalidates the job's cached stages. Each run writes
summary/<timestamp>_<template>.md (the model's reply, verbatim) and a
.json next to it with what produced it.
"""

import logging
import re
import time
from datetime import datetime
from pathlib import Path

from diarrhizer.adapters import llm, llm_accounts
from diarrhizer.summary import wildcards
from diarrhizer.summary.provider_info import Limits
from diarrhizer.summary.providers import KIND_CODEX_CLI, Provider
from diarrhizer.summary.templates import PromptTemplate
from diarrhizer.utils import read_json, write_json_atomic, write_text_atomic

logger = logging.getLogger("diarrhizer.summary")

SUMMARY_DIR = "summary"
TRANSCRIPT_PATH = "export/result.md"
NOTE_TITLE_CHARS = 40  # how much of a note summary_title() shows

# What provider checks send: a request that costs next to nothing and proves the sign-in/key works.
CHECK_PROMPT = llm.CHECK_PROMPT


def resolve_job_dir(path: str | Path) -> Path:
    """The job folder for `path`: the folder itself, or its export/result.md (or another export file).

    Raises:
        FileNotFoundError: no such path
        ValueError: a file that is not inside a job's export folder
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Job folder not found: {path}")
    if path.is_dir():
        return path
    if path.parent.name != "export":
        raise ValueError(f"Expected a job folder or a file in its export/ folder, got {path}")
    return path.parent.parent


def read_transcript(job_dir: str | Path) -> str:
    """The text of export/result.md.

    Raises:
        FileNotFoundError: the job has not been exported yet
        ValueError: the transcript is empty
    """
    path = Path(job_dir) / TRANSCRIPT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - the job has no result yet (finish it, or re-run its export stage)"
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    return text


def read_note_file(path: str | Path) -> str:
    """The text of a note file. UTF-8 (with or without a BOM) or UTF-16 with a BOM - what PowerShell's `>` writes.

    Raises:
        FileNotFoundError: no such file
        ValueError: the file is not text in one of those encodings
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Note file not found: {path}")
    raw = path.read_bytes()
    try:
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16")
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{path} is not UTF-8 (or UTF-16 with a BOM) text") from None


def build_prompt(template_prompt: str, transcript: str, note: str = "", job_dir: str | Path | None = None) -> str:
    """The request text for a template: its {{wildcards}} filled in, the transcript attached if it is not placed.

    See wildcards.render_prompt (SUMMARY:WILDCARDS); this is its text only. Without wildcards and a note the
    prompt is the template followed by the transcript fenced in <transcript> tags, as it always was.
    """
    return wildcards.render_prompt(template_prompt, wildcards.PromptContext(transcript, note, job_dir)).text


def preview_prompt(
    job_dir: str | Path, template_prompt: str, note: str | None = None, template_name: str = ""
) -> wildcards.RenderedPrompt:
    """The request summarize_job would send for this prompt and note, built and not sent: its text and warnings.

    Takes the prompt as text, so an editor can preview a template that is not saved yet.

    Raises:
        FileNotFoundError/ValueError: no or empty result.md; UnknownWildcardError for an unknown wildcard
    """
    job_dir = Path(job_dir)
    context = wildcards.PromptContext(read_transcript(job_dir), note or "", job_dir)
    return wildcards.render_prompt(template_prompt, context, template_name)


# [SEMANTIC-BEGIN] SUMMARY:JOB
# @purpose: Produce and store a summary of one job's transcript with a chosen template and provider
# @description: Reads export/result.md as it is (speaker names included), builds the request from the template's
#   prompt (SUMMARY:WILDCARDS: {{transcript}}, {{note}} and the job's facts are filled in where the template
#   puts them; the transcript is appended when it does not; an optional note of the user for this run goes
#   in the same way, with a warning - logged and returned - when the template has no {{note}}), asks the
#   provider (ADAPTER:LLM) and writes the reply to
#   summary/<YYYYmmdd_HHMMSS>_<template>.md plus a .json with the template name and prompt (as written, with
#   its {{wildcards}}), the user's fields ({"note": ...}), the wildcards the template names, provider
#   name/kind/model, timing, transcript size and whether the reply was cut off at the length limit - and what
#   the request used, as far as the provider says: the model that answered, token counts, a cost (with a note
#   on where the figure comes from: Claude Code's estimate, the provider's own, or the prices set on the
#   Provider) and the plan's limits (from the reply for Claude; read from Codex right after the request, a
#   failure to read them is only logged). Nothing is written when the provider fails, and earlier summaries
#   are never overwritten (a name
#   clash within the same second gets a _2, _3 suffix). The transcript's size is not checked: a model
#   whose context is too small answers with an error that is passed on as is.
#   list_summaries() reads the .json files back, newest first, for the GUI and the CLI; a summary stored
#   before the user's fields and wildcards were recorded lists with them empty, and summary_title() adds the
#   start of the note to a summary's line. preview_prompt() builds the request text for a prompt (saved or not)
#   and a note without sending anything.
# @inputs: job_dir, PromptTemplate, Provider, optional note
# @outputs: dict with path (the .md), meta_path, text, warnings and the stored metadata (incl. user_fields, wildcards_used, model, usage, cost_usd, cost_note, limits)
# @sideEffects: Sends the transcript to the provider (and, for Codex, asks its app-server for the limits); writes summary/*.md and summary/*.json atomically
# @errors: FileNotFoundError/ValueError (no or empty result.md; UnknownWildcardError for a template that names an unknown wildcard), adapters.llm.LLMError (provider failed)
# @see: ADAPTER:LLM, ADAPTER:LLM_ACCOUNTS, CONFIG:PROVIDER_INFO, CONFIG:SUMMARY_TEMPLATES, CONFIG:SUMMARY_PROVIDERS, SUMMARY:WILDCARDS, STAGE:EXPORT, CLI:SUMMARIZE, GUI:SUMMARY_DIALOG
def summarize_job(
    job_dir: str | Path, template: PromptTemplate, provider: Provider, note: str | None = None
) -> dict:
    job_dir = Path(job_dir)
    transcript = read_transcript(job_dir)
    rendered = wildcards.render_prompt(
        template.prompt, wildcards.PromptContext(transcript, note or "", job_dir), template.name
    )
    prompt = rendered.text
    logger.info(
        "[summary] %s: template '%s', provider '%s' (%s), transcript %d chars",
        job_dir.name, template.name, provider.name, provider.kind, len(transcript),
    )
    for warning in rendered.warnings:
        logger.warning("[summary] Warning: %s", warning)

    started = time.monotonic()
    reply = llm.complete(provider, prompt)
    duration = time.monotonic() - started
    limits = reply.limits or _plan_limits(provider)

    now = datetime.now()
    meta = {
        "created_at": now.isoformat(timespec="seconds"),
        "template": template.name,
        "prompt": template.prompt,
        "user_fields": {"note": (note or "").strip()},
        "wildcards_used": rendered.wildcards_used,
        "provider": {"name": provider.name, "kind": provider.kind, "model": provider.model},
        "source": TRANSCRIPT_PATH,
        "transcript_chars": len(transcript),
        "duration_seconds": round(duration, 1),
        "truncated": reply.truncated,
        "model": reply.model or provider.model,
        "usage": reply.usage.to_dict() if reply.usage else None,
        "cost_usd": reply.cost_usd,
        "cost_note": reply.cost_note,
        "limits": limits.to_dict() if limits else None,
    }
    md_path = _new_summary_path(job_dir, now, template.name)
    write_text_atomic(md_path, reply.text + "\n")
    write_json_atomic(md_path.with_suffix(".json"), meta)
    logger.info("[summary] Saved %s (%.1fs)", md_path, duration)
    return {
        **meta,
        "path": str(md_path),
        "meta_path": str(md_path.with_suffix(".json")),
        "text": reply.text,
        "warnings": rendered.warnings,
    }
# [SEMANTIC-END] SUMMARY:JOB


def check_provider(provider: Provider) -> dict:
    """Send CHECK_PROMPT to the provider; returns the start of its reply and what the request used.

    The dict has "reply" (the first 200 characters) and, as far as the provider says, "model", "usage",
    "cost_usd", "cost_note" and "limits" (to_dict() shapes; None when unknown).

    Raises:
        adapters.llm.LLMError: the provider is not reachable, not signed in, or has a wrong key
    """
    reply = llm.complete(provider, CHECK_PROMPT)
    text = reply.text.strip()
    return {
        "reply": text if len(text) <= 200 else text[:200] + "…",
        "model": reply.model or provider.model,
        "usage": reply.usage.to_dict() if reply.usage else None,
        "cost_usd": reply.cost_usd,
        "cost_note": reply.cost_note,
        "limits": reply.limits.to_dict() if reply.limits else None,
    }


def _plan_limits(provider: Provider) -> Limits | None:
    """The plan's limits right after a request, for the providers that can report them on demand (Codex).

    Best effort: the summary is already made, so a failure here is logged, not raised. Claude's limits come
    with its reply instead (asking would be a second request); API providers have none.
    """
    if provider.kind != KIND_CODEX_CLI:
        return None
    try:
        return llm_accounts.read_limits(provider)
    except llm.LLMError as e:
        logger.warning("[summary] Could not read the plan limits: %s", e)
        return None


def list_summaries(job_dir: str | Path) -> list[dict]:
    """The job's stored summaries, newest first.

    Each entry is the stored metadata plus "path" (the .md) and "meta_path".
    A summary stored before "user_fields" and "wildcards_used" were recorded has them empty.
    Summaries whose .md or .json is missing or unreadable are left out.
    """
    folder = Path(job_dir) / SUMMARY_DIR
    summaries = []
    for meta_path in sorted(folder.glob("*.json"), reverse=True):
        meta = read_json(meta_path)
        md_path = meta_path.with_suffix(".md")
        if isinstance(meta, dict) and md_path.exists():
            summaries.append({
                "user_fields": {}, "wildcards_used": [], **meta, "path": str(md_path), "meta_path": str(meta_path),
            })
    return summaries


def summary_note(summary: dict) -> str:
    """The note a stored summary was made with; "" for none (older summaries, or a damaged field)."""
    fields = summary.get("user_fields")
    note = fields.get("note") if isinstance(fields, dict) else ""
    return note.strip() if isinstance(note, str) else ""


def summary_title(summary: dict) -> str:
    """One line for a list of summaries: when, which template, which provider - and the start of the note, if any."""
    try:
        when = datetime.fromisoformat(summary["created_at"]).strftime("%d.%m.%Y %H:%M")
    except (KeyError, ValueError):
        when = "?"
    provider = summary.get("provider") or {}
    title = f"{when} · {summary.get('template', '?')} · {provider.get('name', '?')}"
    note = " ".join(summary_note(summary).split())  # one line
    if note:
        title += f" · «{note[:NOTE_TITLE_CHARS]}{'…' if len(note) > NOTE_TITLE_CHARS else ''}»"
    return title


def format_window(minutes: int | None) -> str:
    """A usage window's length for people: 300 -> "5 h", 10080 -> "7 d", 45 -> "45 min"."""
    if not minutes:
        return ""
    if minutes % 1440 == 0:
        return f"{minutes // 1440} d"
    if minutes % 60 == 0:
        return f"{minutes // 60} h"
    return f"{minutes} min"


def describe_usage(meta: dict) -> list[str]:
    """What a request used, one line each (tokens, cost, limits), from a summary's metadata or a check result.

    Only what the provider reported is described: a field that is absent gives no line.
    """
    lines = []
    usage = meta.get("usage")
    if isinstance(usage, dict) and (usage.get("input_tokens") or usage.get("output_tokens")):
        cached = f" ({usage['cached_tokens']:,} cached)" if usage.get("cached_tokens") else ""
        lines.append(f"Tokens: {usage.get('input_tokens', 0):,} in{cached}, {usage.get('output_tokens', 0):,} out")
    cost = meta.get("cost_usd")
    if isinstance(cost, (int, float)):
        note = f" ({meta['cost_note']})" if meta.get("cost_note") else ""
        lines.append(f"Cost: about ${cost:.4f}{note}")
    limits = meta.get("limits")
    if isinstance(limits, dict):
        for window in limits.get("windows") or []:
            name = " ".join(part for part in (window.get("label"), format_window(window.get("window_minutes"))) if part)
            used = window.get("used_percent")
            resets = window.get("resets_at")
            parts = [f"{used:.0f}% used"] if isinstance(used, (int, float)) else [window.get("status") or "no figure"]
            if isinstance(resets, (int, float)):
                parts.append("resets " + datetime.fromtimestamp(resets).strftime("%d.%m %H:%M"))
            lines.append(f"Limit{' ' + name if name else ''}: " + ", ".join(parts))
        if limits.get("credits"):
            lines.append(f"Credits: {limits['credits']}")
        if limits.get("allowed") is False:
            lines.append("The plan's usage is exhausted")
        if limits.get("note"):
            lines.append(str(limits["note"]))
    return lines


def _new_summary_path(job_dir: Path, now: datetime, template_name: str) -> Path:
    slug = re.sub(r"[^\w-]+", "-", template_name).strip("-")[:40] or "summary"
    stem = f"{now:%Y%m%d_%H%M%S}_{slug}"
    path = job_dir / SUMMARY_DIR / f"{stem}.md"
    counter = 2
    while path.exists() or path.with_suffix(".json").exists():
        path = job_dir / SUMMARY_DIR / f"{stem}_{counter}.md"
        counter += 1
    return path
