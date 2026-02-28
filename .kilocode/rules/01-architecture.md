# Architecture Rules

## Pipeline staging
- Treat each stage as a unit with explicit inputs/outputs (artifacts on disk).
- Stages should be:
  - deterministic (given the same inputs/config),
  - idempotent (safe to re-run),
  - cache-aware (skip if valid outputs already exist).

## Adapters boundary
- Keep FFmpeg/WhisperX/diarization calls inside `adapters/`.
- The rest of the code should depend on **our** interfaces/models, not on third-party APIs directly.

## Artifacts and caching
- Prefer saving intermediate artifacts (wav, transcript, diarization, merged segments).
- Do not store large model files or recordings inside the repo.

## Error handling
- Fail fast with actionable messages:
  - “FFmpeg not found”, “HF token missing”, “CUDA not available”, dependency conflicts, etc.

