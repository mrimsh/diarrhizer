# Docs and Repo Hygiene

## Documentation updates
If you change user-facing behavior, update the docs in the same change:
- `README.md` (usage / commands / outputs)
- `docs/architecture.md` if pipeline or artifact layout changes

## Secrets
- Never commit tokens or secrets.
- Use `.env` locally; keep `.env.example` updated.

## Large files
- Do not add large media files or model weights to the repo.
- Prefer keeping outputs under `out/` (ignored by git and by Kilo via `.kilocodeignore`).

