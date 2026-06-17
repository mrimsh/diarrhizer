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
| `run` command | ✅ Implemented | [`src/diarrhizer/cli.py`](src/diarrhizer/cli.py) (lines 77-107) |
| Pipeline runner | ✅ Implemented | [`src/diarrhizer/pipeline/runner.py`](src/diarrhizer/pipeline/runner.py) |
| Convert stage | ✅ Implemented | [`src/diarrhizer/pipeline/stages/convert.py`](src/diarrhizer/pipeline/stages/convert.py) |
| Transcribe stage | ✅ Implemented | [`src/diarrhizer/pipeline/stages/transcribe.py`](src/diarrhizer/pipeline/stages/transcribe.py) |
| WhisperX adapter | ✅ Implemented | [`src/diarrhizer/adapters/whisperx.py`](src/diarrhizer/adapters/whisperx.py) |
| FFmpeg adapter | ✅ Implemented | [`src/diarrhizer/adapters/ffmpeg.py`](src/diarrhizer/adapters/ffmpeg.py) |
| Diarize stage | ✅ Implemented | [`src/diarrhizer/pipeline/stages/diarize.py`](src/diarrhizer/pipeline/stages/diarize.py) |
| Merge stage | ✅ Implemented | [`src/diarrhizer/pipeline/stages/merge.py`](src/diarrhizer/pipeline/stages/merge.py) |
| Export modules | ✅ Implemented | [`src/diarrhizer/export/`](src/diarrhizer/export/) |

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
    -> artifacts/export/result.md + result.json
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
    meta/                 # metadata (config, versions, timestamps, ASR params)
    audio/
      normalized.wav      # or normalized_left.wav/normalized_right.wav for split-stereo
    asr/
      transcript.json     # includes ASR config in metadata
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

```json
{
  "stage": "merge",
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "speaker_id": "Speaker_00",
      "text": "Hello world",
      "words": [
        {"start": 0.0, "end": 0.5, "word": "Hello", "speaker_id": "Speaker_00"},
        {"start": 0.5, "end": 1.0, "word": "world", "speaker_id": "Speaker_00"}
      ]
    }
  ],
  "num_segments": 1,
  "metadata": {
    "asr_params": {
      "model": "base",
      "compute_type": "float16",
      "beam_size": 5,
      "temperature": 0.0
    },
    "audio_profile": "raw"
  }
}
```

**Algorithm:** For each transcript segment, the speaker with maximum time overlap is chosen. Word-level speakers are assigned similarly if word timestamps are available.

**Edge cases:**
- No diarization data: defaults to "Speaker_00"
- Gaps in diarization: uses closest segment by time
- Overlapping speakers: chooses speaker with most overlap

**Important:** Speaker names are identifiers only.  
Name mapping is a separate layer on top.

---

## 5. Audio Profiles

Audio profiles apply FFmpeg filters during conversion:

| Profile | Filters Applied | Use Case |
|---------|-----------------|----------|
| `raw` | None | Default behavior, no preprocessing |
| `voice-call` | Bandpass filter (300Hz-7kHz) + mild EQ boost at 3kHz | Phone call recordings, VoIP |
| `denoise-light` | afftdn noise reduction | Noisy recordings with background noise |
| `split-stereo` | Separates L/R channels | Multi-channel recordings, interview separation |

---

## 6. ASR Parameter Persistence

All ASR parameters are saved in `asr/transcript.json` under `metadata` to enable experiment comparison:

```json
{
  "metadata": {
    "model": "large-v3",
    "compute_type": "float16",
    "beam_size": 5,
    "temperature": 0.0,
    "condition_on_previous_text": true,
    "vad_filter": true,
    "vad_min_silence_ms": 1000
  }
}
```

---

## 7. Environment Diagnostics (`doctor`)

The Windows dependency stack is sensitive:

- torch / torchaudio version alignment (CPU vs CUDA builds),
- **cuDNN major version** — WhisperX 3.3.1 requires ctranslate2<4.5.0 (cuDNN 8), while torch >=2.4.0+cu124 ships cuDNN 9. This mismatch causes `Could not locate cudnn_ops_infer64_8.dll` at transcribe time even though `torch.cuda.is_available()` returns `True`. See [troubleshooting](troubleshooting.md#cuDNN-dll-mismatch-ctranslate2-vs-torch-cuda).
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

### Troubleshooting

#### Diarization Issues

If diarization fails with audio decoding errors (torchcodec/FFmpeg compatibility issues on Windows), the adapter will automatically attempt a **fallback approach**:

1. First, it tries the default WhisperX audio loading
2. If that fails with a decoder/audio/FFmpeg/codec error, it falls back to using `torchaudio` to load the audio file directly
3. The fallback loads the waveform into memory, converts to mono if needed, resamples to 16kHz, and passes the preloaded waveform to the diarization model

This fallback is logged with a warning message. If diarization still fails after the fallback attempt, an error is raised with details about the failure.

To avoid this issue, ensure:
- FFmpeg is properly installed and in PATH
- torch and torchaudio versions are compatible
- Consider using `--device cpu` if GPU drivers are causing issues

---

## 8. CLI Commands

### `doctor`

```powershell
python -m diarrhizer doctor
```

Runs environment diagnostics. See [Section 7](#7-environment-diagnostics-doctor) for details.

### `run`

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
| `--asr-model` | string | `"base"` | WhisperX model (or HF repo like koekaverna/faster-whisper-podlodka-turbo) |
| `--asr-compute-type` | string | auto | Compute type: `float16`, `int8_float16`, `int8` |
| `--asr-beam-size` | int | `5` | Decoding beam size for quality/speed trade-off |
| `--asr-temperature` | float | `0.0` | Decoding temperature for stability |
| `--asr-condition-on-previous-text` | bool | `true` | Prevent repeats and hallucinations |
| `--asr-initial-prompt-file` | string | — | Path to glossary/prompt file for terminology |
| `--asr-hotwords-file` | string | — | Path to hotwords file (not yet implemented) |
| `--asr-vad-filter` | bool | `true` | Enable VAD filtering |
| `--asr-vad-min-silence-ms` | int | `1000` | VAD minimum silence in milliseconds |
| `--audio-profile` | choice | `"raw"` | Audio preprocessing: `raw`, `voice-call`, `denoise-light`, `split-stereo` |
| `--force-stage` | choice | — | Force recompute specific stage |

> **Note:** Pipeline runs: convert → transcribe → diarize → merge → export.

---

## 9. Extensibility (Future Growth)

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

## 10. Implicit Quality Requirements

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

## 11. Project Structure (Planned)

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
│       ├── diarize.py      # Speaker diarization (pyannote)
│       ├── merge.py        # Merge ASR with speaker labels
│       └── export.py       # Export to MD/JSON
├── adapters/               # External library wrappers
│   ├── ffmpeg.py
│   └── whisperx.py
└── export/                 # Export formatters
    ├── markdown_export.py  # Markdown exporter
    └── json_export.py     # JSON exporter
```
