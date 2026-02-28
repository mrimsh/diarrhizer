# Semantic Tags Specification

This document defines the **semantic tagging contract** for this repository.
The goal is to keep **architecture and intent stable** across refactors and AI-assisted edits.

> Principle: the code remains the source of truth, but semantic tags provide a *stable navigation and intent layer*
> that should survive renames, moves, and refactors.

---

## 1) When tags are required

Add semantic tag blocks for:

- **Public API**:
  - CLI commands and their handlers
  - exported functions/classes intended to be used from other modules
- **Boundary layers**:
  - adapters (FFmpeg / WhisperX / diarization)
  - diagnostics (`doctor`), file system boundaries
- **Places with invariants or side effects**:
  - file I/O, network access, GPU usage, caching
- **Non-trivial transformations**:
  - merging diarization with ASR words/segments
  - algorithms or tricky parsing/formatting

Tags are optional for small private helpers.

---

## 2) Block format (Python)

A semantic block has:
1) a **BEGIN** header *before* the node (function/class/module section)
2) tag lines
3) an **END** marker *after* the node

### BEGIN block (before the node)

```python
# [SEMANTIC-BEGIN] PIPELINE:RUN
# @purpose: Run the processing pipeline for a single job.
# @description: Orchestrates stages and reuses cached artifacts when possible.
# @inputs: input_path, config, out_dir
# @outputs: artifacts on disk + exported result files
# @sideEffects: filesystem I/O, GPU usage, model downloads (first run)
# @errors: RuntimeError, FileNotFoundError
# @see: STAGE:CONVERT, STAGE:TRANSCRIBE, STAGE:DIARIZE, STAGE:EXPORT
def run_pipeline(...):
    ...
```

### END marker (after the node)

```python
# [SEMANTIC-END] PIPELINE:RUN
```

**Rules:**
- The END marker must repeat the same anchor used in BEGIN.
- Place the END marker at the **same indentation level** as the tagged node definition.
- Keep tag values short. Prefer one line per tag.

---

## 3) Minimal tag set

Use these tags when applicable:

- `@anchor` (implicit in the BEGIN header, see below)
- `@purpose` — one-line “why it exists”
- `@description` — the *non-obvious* “how/why” (skip obvious restatements)
- `@inputs` / `@outputs` — brief description (do not duplicate type hints)
- `@sideEffects` — filesystem/network/GPU/cache/global state
- `@errors` — notable exceptions / error conditions / return codes
- `@see` — 2–5 related anchors (neighbors in the flow)

Optional (use sparingly):
- `@history` — notes like `merged-from: ...` when anchors are merged

---

## 4) Anchor rules

Anchors are **stable IDs** that must survive refactors.

- Anchors must be **unique across the repo**.
- Anchors must **not** depend on filenames or function/class names.
- Format: `AREA:ACTION` (uppercase recommended)

Examples:
- `PIPELINE:RUN`
- `STAGE:DIARIZE`
- `ADAPTER:FFMPEG`
- `EXPORT:MARKDOWN`
- `DIAGNOSTICS:DOCTOR`

**Uniqueness check:** before introducing a new anchor, search the repo for it.

---

## 5) Anchor change protocol (stability rules)

- **Rename a function/class** → do **not** change the anchor
- **Move code to another file** → do **not** change the anchor
- **Split a function**:
  - keep the old anchor on the part that preserves the original meaning
  - assign new anchors to the new extracted parts
- **Merge functions**:
  - choose one anchor as the primary
  - add old anchors to `@history` or `@see` (e.g. `merged-from: ...`)

Only change an anchor if the semantic meaning truly changed.

---

## 6) What to do during edits

If you change behavior or side effects:
- update the semantic tags *first* (purpose/sideEffects/errors/see)
- then update docs if user-facing behavior changed

