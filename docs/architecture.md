# Architecture

Goal: a simple, extensible architecture for local call recording processing on Windows:  
**FFmpeg → WhisperX (ASR + alignment) → diarization (pyannote via WhisperX) → merge → export (MD/TXT/JSON).**

This document is intentionally concise. It defines the "skeleton" so the project can evolve (caching, new exports, GUI) without rewriting the core.

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| CLI entry point | ✅ Implemented | [`src/diarrhizer/cli.py`](src/diarrhizer/cli.py) |
| `doctor` command | ✅ Implemented | [`src/diarrhizer/diagnostics/doctor.py`](src/diarrhizer/diagnostics/doctor.py) |
| `run` command | ⚠️ Stub | [`src/diarrhizer/cli.py`](src/diarrhizer/cli.py) (lines 73-77) |
| Pipeline runner | ⏳ Planned | [`src/diarrhizer/pipeline/runner.py`](src/diarrhizer/pipeline/runner.py) |
| Pipeline stages | ⏳ Planned | [`src/diarrhizer/pipeline/stages/`](src/diarrhizer/pipeline/stages/) |
| Adapters | ⏳ Planned | [`src/diarrhizer/adapters/`](src/diarrhizer/adapters/) |
| Export modules | ⏳ Planned | [`src/diarrhizer/export/`](src/diarrhizer/export/) |

---

## 1. Core Principles

### 1) Stage-based pipeline

Each processing step is isolated and has clearly defined inputs and outputs (artifacts).

### 2) Disk artifacts as contract and cache

Results of heavy stages (conversion, ASR, diarization) are stored and reused.  
This ensures reproducibility and speeds up iterations.

### 3) Adapters as a boundary layer to external libraries

Calls to FFmpeg, WhisperX, and pyannote are centralized in `adapters/` rather than scattered across the codebase.  
This reduces coupling and simplifies updates or replacements.

---

## 2. Data Flow

Planned high-level flow:

```
[Input media file]
      |
      v
(1) Convert (FFmpeg)  
    -> artifacts/audio/normalized.wav
      |
      v
(2) Transcribe (WhisperX ASR + alignment)  
    -> artifacts/asr/transcript.json
      |
      v
(3) Diarize (pyannote via WhisperX)  
    -> artifacts/diar/diarization.json
      |
      v
(4) Merge (speaker ↔ words/segments)  
    -> artifacts/merged/segments.json
      |
      v
(5) Export  
    -> artifacts/export/result.md + result.txt + result.json
```

Supported inputs: various audio/video formats (`.mp3`, `.wav`, `.m4a`, `.mp4`, `.mkv`, `.webm`).  
Normalization via FFmpeg is mandatory.

---

## 3. Key Entities

### Job

"One processing run for one file."

At minimum, it contains:

- `input_path`
- `out_dir`
- `config` (language, device cpu/cuda, min/max speakers, etc.)

---

### Stage

A pipeline stage with a contract:

- `inputs`: expected artifacts
- `outputs`: produced artifacts
- `run(job, artifacts)`

A stage must be:

- **deterministic** (given identical inputs and config),
- **idempotent** (safe to re-run),
- **cache-aware** (if output exists and is valid, the stage may be skipped).

---

### Artifacts (results directory)

A disk structure associated with a job.

Planned layout:

```
out/
  <job_id>/
    meta/                 # metadata (config, versions, timestamps)
    audio/
      normalized.wav
    asr/
      transcript.json
    diar/
      diarization.json
    merged/
      segments.json
    export/
      result.md
      result.txt
      result.json
```

`job_id` can be generated as:

- filename + timestamp, or
- hash (input + config) — if reproducibility/deduplication is important.

---

## 4. Data Format (Overview)

### ASR output

Contains segments and/or words with timestamps (WhisperX alignment result).

### Diarization output

Contains speech intervals labeled with speaker identifiers (`Speaker_00...`).  
Real names are applied through a separate mapping layer.

### Merged segments

The "stitched" result: text + timestamps + speaker at segment (and/or word) level.

**Important:** Speaker names are identifiers only.  
Name mapping is a separate layer on top.

---

## 5. Environment Diagnostics (`doctor`)

The Windows dependency stack is sensitive:

- torch / torchaudio / torchvision version alignment (CPU vs CUDA builds),
- FFmpeg availability in `PATH`,
- Hugging Face token (gated models),
- potential torchcodec ↔ FFmpeg / torch compatibility issues.

Therefore, a `doctor` command is provided to verify these conditions before running heavy processing.

### Doctor Checks (Implemented)

The [`doctor`](src/diarrhizer/diagnostics/doctor.py) command performs 5 checks:

1. **Python version** — verifies Python 3.11+
2. **FFmpeg** — checks availability in PATH using `shutil.which("ffmpeg")`
3. **PyTorch/Torchaudio** — verifies installation and reports version and CUDA status
4. **CUDA** — checks GPU availability via `torch.cuda.is_available()`
5. **Hugging Face token** — verifies `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` environment variable is set

Run with:
```powershell
python -m diarrhizer doctor
```

---

## 6. CLI Commands

### `doctor`

```powershell
python -m diarrhizer doctor
```

Runs environment diagnostics. See [Section 5](#5-environment-diagnostics-doctor) for details.

### `run` (Stub)

```powershell
python -m diarrhizer run "<path>" --out "./out" --min-speakers 2 --max-speakers 6 --lang ru --device cuda
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input` | positional | — | Path to input media file (required) |
| `--out` | string | `./out` | Output directory |
| `--min-speakers` | int | `1` | Minimum number of speakers |
| `--max-speakers` | int | `10` | Maximum number of speakers |
| `--lang` | string | `"auto"` | Language code or `"auto"` for detection |
| `--device` | choice | `"cuda"` | Device: `cuda` or `cpu` |

> **Note:** Currently a stub. Pipeline stages are not yet implemented.

---

## 7. Extensibility (Future Growth)

The architecture is designed to allow extensions without breaking the core:

### New stages

- volume normalization, noise reduction
- text post-processing (punctuation/formatting)
- summaries / action items (optional)

### New export formats

- HTML, DOCX, SRT/VTT
- integration with note-taking systems

### GUI

- The GUI layer must call the same `pipeline runner` and operate on the same artifacts.
- The GUI must not contain processing logic — only UX.

---

## 8. Implicit Quality Requirements

- Every stage must log:
  - input/output artifact paths,
  - key parameters,
  - execution duration.
- Errors must be actionable and explanatory:
  - "FFmpeg not found"
  - "HF token missing"
  - "CUDA not available"
  - dependency conflicts, etc.

---

## 9. Project Structure (Planned)

```
src/diarrhizer/
├── cli.py                  # CLI entry point (doctor, run)
├── diagnostics/
│   └── doctor.py           # Environment diagnostics
├── pipeline/
│   ├── runner.py           # Pipeline orchestration
│   └── stages/             # Individual processing stages
│       ├── convert.py      # FFmpeg normalization
│       ├── transcribe.py   # WhisperX ASR + alignment
│       ├── diarize.py      # pyannote diarization
│       ├── merge.py        # Speaker + words merge
│       └── export.py       # MD/TXT/JSON export
├── adapters/               # External library wrappers
│   ├── ffmpeg.py
│   ├── whisperx.py
│   └── diarization.py
└── export/                 # Export formatters
    ├── markdown.py
    ├── text.py
    └── json.py
```
