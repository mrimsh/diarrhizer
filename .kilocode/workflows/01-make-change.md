# Make a Change (minimal, consistent)

Use this workflow for *any* implementation task.

## 1) Clarify the target behavior
- Restate what must change in one paragraph.
- Identify which module(s) should change (pipeline stage / adapter / export / CLI).
- Keep scope minimal.

## 2) Semantic tags first (when applicable)
- If the change touches public API, stages, adapters, diagnostics, or non-trivial logic:
  - add/update semantic blocks per `SEMANTIC_TAGS.md`
  - keep anchors stable across refactors

## 3) Implement
- Prefer small diffs.
- Keep external library calls inside `adapters/`.
- Keep stage logic inside `pipeline/stages/`.
- Preserve artifact layout unless intentionally changed.

## 4) Check
- Run `python -m diarrhizer doctor` (or update it if needed).
- If tests exist, run them.
- If outputs/contracts changed, update docs.

## 5) Update docs
- Update `README.md` and/or `docs/*` if user-facing behavior changed.

