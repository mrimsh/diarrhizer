# Bootstrap (local dev)

You are working in the Diarrhizer repository.

## 1) Read the project context
- Read `AGENTS.md`
- Read `docs/architecture.md`
- Read `SEMANTIC_TAGS.md`

## 2) Basic environment checks (Windows-first)
- Ensure a Python venv is active (`.venv` recommended).
- Confirm FFmpeg is available in PATH.
- If diarization is involved, confirm `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` is set.

## 3) Run diagnostics
Run:
- `python -m diarrhizer doctor`

If diagnostics fail, propose the minimal fix (do not add new tooling unless necessary).

## 4) Smoke run (if possible)
If you have a small sample file available:
- run `python -m diarrhizer run "<file>" --out "./out" --device cpu`
Keep the run short and do not require GPU unless needed.

