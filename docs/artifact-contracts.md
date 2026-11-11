# Artifact versions, writes and recovery

New pipeline JSON (`meta/run.json`, ASR, diarization, merge and exported JSON), summary metadata and
chat files have `schema_version: 1`. It identifies this additive envelope, independently of a stage's
`metadata.cache_version` used to invalidate algorithm caches. Existing unversioned jobs remain readable;
ASR cache version is `3` (validated loading), merge is `2`. Only stages in the requested range check caches.
This is not a migration to a new transcript shape.

When opening or re-exporting old merged results, the shared segment reader recovers finite,
non-negative reversed intervals (`end < start`) in memory by setting `end = start`.
All text, speaker IDs and extra fields remain available; uncertain word times on that segment
become `null`. An `alignment_warning` records the original bounds, the GUI displays a warning,
and regenerated JSON exports include it in `metadata.warnings`. Opening never rewrites the
merged source or its exports. Explicit re-export writes corrected export bounds while its
`source_sha256` continues to identify the original merged file. Null, non-numeric, non-finite
or negative bounds still fail with the segment index and offending values.

`provenance` records Python and installed package versions without importing ML libraries. In exported
JSON it is under `metadata`; in other artifacts it is top-level. ASR retains model name, requested and
effective settings, alignment status and warnings. `metadata.alignment_model` records the HF alignment
revision and loading audit, or a torchaudio bundle name with `revision: null`. ASR/diarization model
revisions are not yet recorded; package versions alone do not guarantee identical weights. Exports aggregate ASR/merge warnings and
record `metadata.source_sha256` for the merged JSON. Summary metadata records source and rendered-request
hashes, template text, warnings and the complete context estimate. Chat snapshots the transcript hash and
records the budget per turn. These are diagnostics, not signatures or evidence of ASR accuracy.

## One writer per job

Pipeline execution, summaries, chat mutations, speaker-name saving and job deletion acquire the same
nonblocking OS file lock. A competing operation fails with `Job is busy … Retry after it finishes`;
different jobs can run independently. The lock spans a model call to prevent a second request from
overwriting state read before the first request completed. Read-only previews do not acquire it.

Tiny lock files live beside jobs in `.diarrhizer-locks/`; history ignores that directory. The OS releases
the lock on process exit/crash. Leftover lock files are harmless and must not be deleted while writers
are running. Fresh job names are reserved with atomic directory creation, including collision suffixes.
An empty reservation is removed when preflight fails; no existing job is reused implicitly.

## Recoverable bundles

Export renders all formats before changing any output. It then writes `.manifest.json.pending.json`,
containing the complete new texts, replaces the outputs, publishes `manifest.json` with their SHA-256
hashes and removes the journal. A summary uses the same sequence with `<stem>.manifest.json` for its
Markdown/metadata pair. All individual writes use a temporary file, flush/fsync and replace.

Pending or hash-mismatched bundles are not accepted as complete by cache checks, summary readers or
history. Legacy exports/summaries without manifests remain readable. Hand-editing a manifested result
makes it invalid; retain such edits separately before rebuilding exports. The source ASR is authoritative.

After an interrupted write:

```powershell
python -m diarrhizer recover "out/<job>"
```

This completes export/summary journals locally, without loading models or making provider calls.
Pipeline/export also recovers an export journal on retry. A summary retry recovers its saved response
and stops with a message to open it; it does **not** request another paid response automatically.
Malformed journals are preserved and reported. Back them up for inspection; do not delete all job data.
For changed exports without a pending journal, rebuild via `run --job-dir "out/<job>" --from-stage export`.

This is recoverable publication, not one atomic filesystem transaction across all files. An external
reader that ignores the manifest can see a partial group. Recovery needs a successfully written journal;
if the disk fails before that point, a received model response may be lost. Chat is a single atomically
replaced JSON file under the job lock; it has no paid-response recovery journal yet. `.env` replacement is
atomic but simultaneous settings edits across application instances are not merged. Power-loss durability
depends on the filesystem/storage device. None of these changes silently rewrites old jobs on opening.
