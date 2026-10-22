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

from diarrhizer.adapters import llm
from diarrhizer.summary.providers import Provider
from diarrhizer.summary.templates import PromptTemplate
from diarrhizer.utils import read_json, write_json_atomic, write_text_atomic

logger = logging.getLogger("diarrhizer.summary")

SUMMARY_DIR = "summary"
TRANSCRIPT_PATH = "export/result.md"

# What provider checks send: a request that costs next to nothing and proves the sign-in/key works.
CHECK_PROMPT = "Ответь одним словом: ok"


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


def build_prompt(template_prompt: str, transcript: str) -> str:
    """The template's prompt with the transcript attached after it.

    The transcript is fenced in <transcript> tags so a model cannot mistake the
    speakers' words for instructions, and the prompt can refer to it by name.
    """
    return f"{template_prompt.strip()}\n\n<transcript>\n{transcript}\n</transcript>\n"


# [SEMANTIC-BEGIN] SUMMARY:JOB
# @purpose: Produce and store a summary of one job's transcript with a chosen template and provider
# @description: Reads export/result.md as it is (speaker names included), attaches it to the template's
#   prompt (build_prompt), asks the provider (ADAPTER:LLM) and writes the reply to
#   summary/<YYYYmmdd_HHMMSS>_<template>.md plus a .json with the template name and prompt, provider
#   name/kind/model, timing, transcript size and whether the reply was cut off at the length limit.
#   Nothing is written when the provider fails, and earlier summaries are never overwritten (a name
#   clash within the same second gets a _2, _3 suffix). The transcript's size is not checked: a model
#   whose context is too small answers with an error that is passed on as is.
#   list_summaries() reads the .json files back, newest first, for the GUI and the CLI.
# @inputs: job_dir, PromptTemplate, Provider
# @outputs: dict with path (the .md), meta_path, text and the stored metadata
# @sideEffects: Sends the transcript to the provider; writes summary/*.md and summary/*.json atomically
# @errors: FileNotFoundError/ValueError (no or empty result.md), adapters.llm.LLMError (provider failed)
# @see: ADAPTER:LLM, CONFIG:SUMMARY_TEMPLATES, CONFIG:SUMMARY_PROVIDERS, STAGE:EXPORT, CLI:SUMMARIZE, GUI:SUMMARY_DIALOG
def summarize_job(job_dir: str | Path, template: PromptTemplate, provider: Provider) -> dict:
    job_dir = Path(job_dir)
    transcript = read_transcript(job_dir)
    prompt = build_prompt(template.prompt, transcript)
    logger.info(
        "[summary] %s: template '%s', provider '%s' (%s), transcript %d chars",
        job_dir.name, template.name, provider.name, provider.kind, len(transcript),
    )

    started = time.monotonic()
    reply = llm.complete(provider, prompt)
    duration = time.monotonic() - started

    now = datetime.now()
    meta = {
        "created_at": now.isoformat(timespec="seconds"),
        "template": template.name,
        "prompt": template.prompt,
        "provider": {"name": provider.name, "kind": provider.kind, "model": provider.model},
        "source": TRANSCRIPT_PATH,
        "transcript_chars": len(transcript),
        "duration_seconds": round(duration, 1),
        "truncated": reply.truncated,
    }
    md_path = _new_summary_path(job_dir, now, template.name)
    write_text_atomic(md_path, reply.text + "\n")
    write_json_atomic(md_path.with_suffix(".json"), meta)
    logger.info("[summary] Saved %s (%.1fs)", md_path, duration)
    return {**meta, "path": str(md_path), "meta_path": str(md_path.with_suffix(".json")), "text": reply.text}
# [SEMANTIC-END] SUMMARY:JOB


def check_provider(provider: Provider) -> str:
    """Send CHECK_PROMPT to the provider; returns the start of its reply.

    Raises:
        adapters.llm.LLMError: the provider is not reachable, not signed in, or has a wrong key
    """
    reply = llm.complete(provider, CHECK_PROMPT)
    text = reply.text.strip()
    return text if len(text) <= 200 else text[:200] + "…"


def list_summaries(job_dir: str | Path) -> list[dict]:
    """The job's stored summaries, newest first.

    Each entry is the stored metadata plus "path" (the .md) and "meta_path".
    Summaries whose .md or .json is missing or unreadable are left out.
    """
    folder = Path(job_dir) / SUMMARY_DIR
    summaries = []
    for meta_path in sorted(folder.glob("*.json"), reverse=True):
        meta = read_json(meta_path)
        md_path = meta_path.with_suffix(".md")
        if isinstance(meta, dict) and md_path.exists():
            summaries.append({**meta, "path": str(md_path), "meta_path": str(meta_path)})
    return summaries


def summary_title(summary: dict) -> str:
    """One line for a list of summaries: when, which template, which provider."""
    try:
        when = datetime.fromisoformat(summary["created_at"]).strftime("%d.%m.%Y %H:%M")
    except (KeyError, ValueError):
        when = "?"
    provider = summary.get("provider") or {}
    return f"{when} · {summary.get('template', '?')} · {provider.get('name', '?')}"


def _new_summary_path(job_dir: Path, now: datetime, template_name: str) -> Path:
    slug = re.sub(r"[^\w-]+", "-", template_name).strip("-")[:40] or "summary"
    stem = f"{now:%Y%m%d_%H%M%S}_{slug}"
    path = job_dir / SUMMARY_DIR / f"{stem}.md"
    counter = 2
    while path.exists() or path.with_suffix(".json").exists():
        path = job_dir / SUMMARY_DIR / f"{stem}_{counter}.md"
        counter += 1
    return path
