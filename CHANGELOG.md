# Changelog

## 2.6 — 2026-11-11

- Use independent, case-sensitive date/time wildcards `YYYY`, `YY`, `MM`, `DD`, `HH`, `mm`, `SS`
  with configurable order and literal separators in job and file names; `MM` is month, `mm` is minute.
- Parse input filenames with the same date/time tokens through an optional format field;
  preview suggestions before applying them, retaining automatic detection and manual date/time editing.
- Preserve complete custom date/time templates when editing existing jobs and keep the date/time
  controls readable in the rename dialog. Existing fixed date/time aliases remain supported.

## 2.5 — 2026-11-11

- Store recording date/time separately from processing time; propose conservative timestamps from
  input filenames with per-file manual editing, filename reset and a Now button.
- Edit an existing job's recording date/time and name in place with a shared preview and rollback;
  preserve processing timestamps, transcript contents and external originals.
- Support recording-time tokens `HH_MM_SS` / `{{time}}`, explicit CLI `--recorded-at`, and recording-time
  history sorting. Keep the default naming template unchanged and preserve legacy-job compatibility.

## 2.4 — 2026-11-11

- Download current GitHub `main` source for launcher setup, repair and application updates, pinned
  to an immutable commit with a local source cache; no Git installation or launcher rebuild per app version.
- Show downloaded version/revision; preserve the working environment on download failure instead of
  silently reinstalling an older bundled application. Launcher and downloaded application versions may differ.

## 2.3 — Unreleased

- Open and re-export old results containing reversed segment intervals without losing transcript
  text or speaker names. Recover bounds only in memory and show an alignment warning; source
  job artifacts remain unchanged on opening.

## 2.2 — Unreleased

- Fix launcher updates after `git pull`: an exe in the checkout uses current local sources;
  show the source path/version, verify the installed version, and support application-only updates
  without reinstalling the CPU/CUDA stack.

## 2.1 — Unreleased

- Require a minor version increment in every new commit, including documentation, tests and merges,
  with the version bump in the same commit as the changes (see AGENTS.md).
- Advance the shared application version to 2.1; CLI, GUI, launcher and newly built distribution
  metadata use the complete recorded `MAJOR.MINOR` value.

## 2.0 — 2026-11-06

### Unified application release

- Shared version **2.0** for the package, CLI, desktop GUI, Windows launcher and provider client identity.
- CLI `--version` and versioned help; the GUI shows the version in its window title and sidebar,
  and exposes it through Qt application metadata. The launcher and its Windows file/product
  properties display the same version, with MRIMSH in company/copyright metadata.
- Application authorship is **MRIMSH** in documentation, package metadata and interfaces.
- Standalone Windows setup/launch/recovery wizard, including opening and editing an existing environment.
- Projects and teams, configurable job names and media retention, and reusable participant lists.
- The stabilization, reliability and quality work below is included in this release.

### Direction for 2.x

Increasing autonomy in call logging and creation of useful artifacts. Planned work includes a separate
agent that joins calls and runs capture, transcription, diarization and summarization, with optional
skills to post messages to chats and prepare/execute task-tracker API requests. This is a development
direction, not functionality shipped in 2.0. See the [roadmap](docs/roadmap.md).

### Quality C1: RAM/VRAM and performance (2026-11-04)

- Shared ASR/DER sampling of full native process-tree RSS/CPU, Windows private commit, system RAM
  and device-wide NVIDIA VRAM/utilization with identity, baseline, sampling coverage and explicit scope.
- Wall time, throughput, weighted RTF, memory peaks and repeated-case median/p95; failed runs retain
  telemetry and unavailable counters remain missing. PyTorch-only memory remains separately labelled.
- Optional `bench` dependencies, supported constraints, Windows CI and native/stub regressions.
- Recorded CPU/CUDA examples with ambient load and sampling limits; no accuracy/cache changes.

### Quality C1: annotated DER (2026-11-04)

- Full duration-based DER using explicit RTTM/UEM, Hungarian speaker mapping and overlap included
  by default; reports false alarms, misses, confusion and additive overlap/non-overlap breakdowns.
- Re-scoring saved raw hypotheses, explicit collar/overlap policies and weighted corpus aggregation.
- Pinned VoxConverse dev mini references with attribution/hashes, bounded audio download and a fresh
  convert+diarize benchmark without reference speaker counts or ASR dependence.
- Regression coverage for analytical metrics, simultaneous speakers, corpus integrity and failed runs;
  documented measured results and limitations. No runtime dependencies or product cache changes.

### Quality C1 continuation (2026-11-04)

- Validated legacy alignment weight-norm loading, exact tensor checks and model/revision provenance;
  genuine checkpoint mismatches retain ASR text with a failed status. ASR cache version is now 3.
- Skipped alignment model loading for empty ASR (`not_needed`); retained meaningful VAD version warnings.
- Expanded to ten synthetic RU/EN cases with independent SAPI word onsets, deterministic transforms
  and hashes; added repeated multi-model suites, conditional speaker scoring and explicit coverage.
- Measured CPU/CUDA ASR and two-speaker diarization; documented checkpoint evidence and quality limits.
- Added local checkpoint/metric/corpus regressions and extended semantic checks to Python tools.

### Stabilization A1–A5 (2026-11-03)

- Separate API key variables, shared-key diagnostics and safe provider renaming.
- Applied ASR beam/prompt options, effective settings and alignment warnings.
- Preserved text with missing/invalid word times, including SRT.
- Shared CLI/GUI resume settings, source fingerprints and versioned ASR/merge caches.
- Protected queued speaker changes, dirty name edits and JSON/filesystem error handling.

### Reliability B1–B4 (2026-11-03)

- Supported-stack install checks, Windows CI, doctor profiles and separate research environments.
- Documentation/link checks and MIT license (MRIMSH).
- Shared summary/chat context estimate including framing and reserved response space.
- Versioned artifacts and package provenance; per-job OS locks and atomic job-name reservation.
- Recoverable export/summary bundles with manifests and a local `recover` command.
- Installed wheels use a user configuration directory; editable checkout paths remain compatible.
- Pinned the validated PySide6 6.11.2 baseline and fixed the empty-result close/save prompt.
- Synthetic Russian fixture generation and initial CPU/CUDA WER/CER/RTF measurements.

## 0.1.0

Initial local application: CLI and desktop GUI, pipeline, batch queue, playback, speaker names,
summary templates/providers and persisted dialogues. This was the initial package baseline,
before the unified CLI/GUI release 2.0.
