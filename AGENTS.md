# Agent Guide (AGENTS.md)

This repository contains **Diarrhizer** — a personal, local (on‑prem) Windows tool for processing call recordings:
**media file → FFmpeg → WhisperX (ASR + alignment) → diarization (pyannote via WhisperX) → exports (MD/TXT/JSON)**.

The goal is a clean, extensible core shared by the CLI and the existing PySide6 GUI, without unnecessary complexity.

---

## Non-goals (keep it simple)

- Do not add heavy frameworks or “enterprise” patterns (DI containers, complex plugin systems, etc.).
- Prefer small, composable modules and straightforward code.
- Avoid introducing new dependencies unless they clearly pay for themselves.

---

## Quick commands (expected)

> Exact commands may evolve, but keep these entry points stable.

- Environment diagnostics:
  - `python -m diarrhizer doctor`
- Process a file:
  - `python -m diarrhizer run "<path>" --out "./out" --min-speakers 2 --max-speakers 6 --lang ru --device cuda`
- Summarize a finished job (templates + providers are shared with the GUI):
  - `python -m diarrhizer summarize "<job folder>" --template "<name>" --provider "<name>" --print`
  - `python -m diarrhizer summarize --list-wildcards` (the `{{wildcards}}` a template may use; `--note "<text>"` fills `{{note}}`)

---

## Repository map (where to look)

- `src/diarrhizer/cli.py` — CLI entry points (public API surface)
- `src/diarrhizer/pipeline/runner.py` — pipeline orchestration (stage runner)
- `src/diarrhizer/pipeline/settings.py` — shared recorded settings for CLI/GUI resume; explicit overrides win
- `src/diarrhizer_gui/` — desktop screens, queue and child-process workers; planning a run must not mutate job artifacts
- `src/diarrhizer/pipeline/stages/*` — individual stages (convert/transcribe/diarize/merge/export)
- `src/diarrhizer/adapters/*` — wrappers around FFmpeg / WhisperX / diarization / language models (`llm.py`: one request; `llm_accounts.py`: sign-in, models and limits of a subscription provider - through the vendors' own tools, never with tokens of ours)
- `src/diarrhizer/summary/*` — summaries of a finished job: prompt templates and their `{{wildcards}}` (`wildcards.py`), providers, the service (an on-demand add-on, not a pipeline stage)
- `src/diarrhizer/diagnostics/doctor.py` — environment checks (Windows is sensitive)
- `docs/architecture.md` — pipeline and artifacts overview
- `docs/realtime.md` and `prototypes/realtime/` — research on live-call capture, streaming ASR and live summaries; research material, not part of the product (nothing in `src/` imports it)

---

## Windows constraints (practical)

- Python + `torch/torchaudio/torchvision` versions must be compatible (CPU vs CUDA builds must match).
- FFmpeg must be available in `PATH` (also matters for decoding stacks used by diarization on Windows).
- Diarization requires a Hugging Face token (use env vars / `.env`, never commit secrets).

---

## Semantic tags policy (must-follow)

This repo uses **semantic tag blocks** to keep architectural intent stable and to reduce style drift.

- The normative specification lives in: **`SEMANTIC_TAGS.md`** (single source of truth).
- Whenever you modify or add:
  - public API (CLI commands, exported functions/classes),
  - boundary layers (adapters, diagnostics),
  - pipeline stages and non-trivial transformations,
  you must add/update semantic tag blocks.

If a refactor changes behavior, update the semantic tags and the docs that describe the behavior (README / docs/*).

---

## Security / secrets

Never paste or commit:
- Hugging Face tokens,
- `.env`,
- private recordings,
- model caches.

If a task requires a token, instruct the user to set `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` locally.

---

## Version policy (must-follow)

- Use the composite application version `MAJOR.MINOR`, for example `2.0`, `2.1`, `2.2`.
- Every new commit must increment `MINOR` by exactly one relative to its first parent,
  including documentation-only, test-only, tooling and merge commits. Keep `MAJOR` unchanged
  unless the user explicitly requests a new major version; then reset `MINOR` to zero.
  Minor numbers do not roll over at 9: `2.9` is followed by `2.10`.
- Include the version bump in the same commit as the changes, never in a separate follow-up
  commit. When amending a commit, keep one increment relative to its parent, not one per amend.
- The single source of truth is `src/diarrhizer/__init__.py::__version__`. Show the full
  `MAJOR.MINOR` value in CLI, GUI, launcher and built package/executable metadata; import it
  rather than hardcoding interface versions. Never derive the version from Git at runtime.
- Update current-version documentation and the changelog with the bump. Preserve historical
  release notes and published tags/assets; a source commit does not itself publish a release.
- Before committing, verify `python -m diarrhizer --version`, refresh editable distribution
  metadata as described in `docs/development.md`, and check that the staged version has advanced
  from the parent. Do not create a commit or publish solely because this policy exists.

---

## How to work on changes

When implementing a change:
1) Understand the requested behavior and keep scope minimal.
2) Update semantic tags (if applicable).
3) Implement (prefer the existing pipeline/stage/adapters structure).
4) Add/adjust basic checks (at least `doctor` expectations and a minimal smoke run if possible).
5) Update docs if user-facing behavior changed.
6) Before a commit, apply the version policy above and include the bump in that commit.

