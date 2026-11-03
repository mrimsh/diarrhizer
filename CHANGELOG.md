# Changelog

## Unreleased

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
summary templates/providers and persisted dialogues. The package version remains 0.1.0 until
a release is deliberately tagged; unreleased changes are installed from this checkout.
