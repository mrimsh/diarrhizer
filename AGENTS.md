# Agent Guide (AGENTS.md)

This repository contains **Diarrhizer** — a personal, local (on‑prem) Windows tool for processing call recordings:
**media file → FFmpeg → WhisperX (ASR + alignment) → diarization (pyannote via WhisperX) → exports (MD/TXT/JSON)**.

The goal is a clean, extensible core (CLI now, GUI later) without unnecessary complexity.

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

---

## Repository map (where to look)

- `src/diarrhizer/cli.py` — CLI entry points (public API surface)
- `src/diarrhizer/pipeline/runner.py` — pipeline orchestration (stage runner)
- `src/diarrhizer/pipeline/stages/*` — individual stages (convert/transcribe/diarize/merge/export)
- `src/diarrhizer/adapters/*` — wrappers around FFmpeg / WhisperX / diarization
- `src/diarrhizer/diagnostics/doctor.py` — environment checks (Windows is sensitive)
- `docs/architecture.md` — pipeline and artifacts overview

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

## How to work on changes

When implementing a change:
1) Understand the requested behavior and keep scope minimal.
2) Update semantic tags (if applicable).
3) Implement (prefer the existing pipeline/stage/adapters structure).
4) Add/adjust basic checks (at least `doctor` expectations and a minimal smoke run if possible).
5) Update docs if user-facing behavior changed.

