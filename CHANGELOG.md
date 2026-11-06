# Changelog

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
