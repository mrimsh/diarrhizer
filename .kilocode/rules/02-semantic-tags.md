# Semantic Tags Rules

## Single source of truth
- The semantic tagging specification is **`SEMANTIC_TAGS.md`**.
- Do not invent new tag formats ad-hoc. If a change is needed, update `SEMANTIC_TAGS.md` first.

## Where tags are required
Add/update semantic blocks for:
- public API (CLI commands, exported functions/classes)
- boundary layers (adapters, diagnostics)
- stages and non-trivial transformations (merge logic, formatting logic)
- functions with important side effects or invariants

## Mandatory structure
- Every tagged node must have:
  - `# [SEMANTIC-BEGIN] <ANCHOR>` block before the node
  - `# [SEMANTIC-END] <ANCHOR>` marker after the node
- Anchors must be unique across the repo and stable across refactors.

## Refactors
Follow the anchor change protocol from `SEMANTIC_TAGS.md`:
- renames/moves do NOT change anchors
- split/merge uses history/see rules

## Tag quality bar
- Keep `@purpose` one line.
- Use `@description` only for non-obvious details.
- Do not duplicate type hints in `@inputs/@outputs`.
- Always list side effects and notable error modes.

