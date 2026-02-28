# Project Rules (Diarrhizer)

## Goals
- Provide a **local (on‑prem) Windows tool** for call recordings:
  FFmpeg → WhisperX (ASR + alignment) → diarization (pyannote via WhisperX) → exports.
- Keep the core **simple and extensible** (CLI first, GUI later).

## Keep it minimal (anti-overengineering)
- Prefer small modules, simple functions, and clear boundaries.
- Avoid new frameworks and complex patterns unless there is a strong reason.
- Do not introduce “plugin systems” or heavy abstractions prematurely.

## Respect the architecture
- Pipeline logic lives in `src/diarrhizer/pipeline/`.
- External integrations live in `src/diarrhizer/adapters/`.
- Exports live in `src/diarrhizer/export/`.
- Diagnostics live in `src/diarrhizer/diagnostics/`.

## Windows practicality
- Assume Windows is sensitive to dependency mismatches (Torch / CUDA / FFmpeg).
- Prefer user-friendly error messages and a reliable `doctor` command.

## Semantic tags (mandatory in key places)
Follow `SEMANTIC_TAGS.md`. If you touch public API, boundary layers, or non-trivial logic,
add/update semantic tag blocks.

