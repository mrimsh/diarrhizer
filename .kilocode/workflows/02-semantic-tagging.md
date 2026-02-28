# Semantic Tagging (how to apply)

Follow `SEMANTIC_TAGS.md` exactly.

## Decide if tags are required
Tags are required for:
- public API
- adapters / diagnostics
- pipeline stages
- non-trivial transformations or side effects

## Choose the anchor
- Format: `AREA:ACTION`
- Must be unique and stable
- Do NOT derive from filenames or function names

## Apply the block
1) Add the BEGIN block immediately above the node:
   - `# [SEMANTIC-BEGIN] <ANCHOR>`
   - tag lines (`@purpose`, `@inputs`, ...)

2) Add the END marker immediately after the node body:
   - `# [SEMANTIC-END] <ANCHOR>`

## Refactor rules
- renames/moves do NOT change anchors
- splits keep old anchor on the “main meaning”
- merges keep one primary anchor, record others in `@history` or `@see`

