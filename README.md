# Diarrhizer

A local (on-prem) Windows tool for processing call recordings:
input audio/video file → FFmpeg → WhisperX (ASR + alignment) → speaker diarization (pyannote via WhisperX) → export (`.md/.txt/.json`).

---

## Current Status

| Component | Status |
|-----------|--------|
| `doctor` command | ✅ Implemented |
| `run` command | ✅ Implemented |
| Pipeline stages | ✅ Implemented |
| Adapters (FFmpeg/WhisperX/diarization) | ✅ Implemented |
| Speaker name mapping | ✅ Implemented |

---

## MVP Features (Planned)

* Supported input formats: `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mkv`, `.webm`, etc.
* Audio normalization via FFmpeg (typically: WAV mono 16 kHz)
* WhisperX transcription with word-level timestamps (alignment)
* Speaker diarization (`Speaker_00`, `Speaker_01`, …) with text-to-speaker assignment
* Export:
  * Human-readable `.md/.txt` (segments + timestamps + speaker)
  * Structured `.json` (for further processing without recomputing ASR/diarization)

### Important About Speakers

Diarization returns **speaker identifiers** (`Speaker_00...`), not real names.
You can provide real names using the `--speakers` option (see below).

### Speaker Name Mapping

You can map diarization IDs to real names using a JSON file:

```json
{
    "Speaker_00": "Ivan",
    "Speaker_01": "Maria",
    "Speaker_02": "John"
}
```

Then run with `--speakers <path_to_json>`. The mapping will be applied at export time:
- `result.md` will show real names instead of Speaker_XX
- `result.json` will include both `speaker_id` (original) and `speaker_name` (mapped)

---

## Requirements

* Windows 10/11
* Python 3.11+ (recommended for stable Torch stack on Windows)
* FFmpeg available in `PATH` (also important for decoding via torchcodec/pyannote on Windows)
* torchcodec (included in CUDA lock files; for fast audio decoding via pyannote)
* For diarization: Hugging Face token + acceptance of gated model terms
* (Optional) NVIDIA GPU + CUDA-compatible Torch wheels

> Internet access is usually required for the initial model download; models are cached locally afterward.

---

## Installation (Draft)

### 1) Create a virtual environment

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Using `.venv` is the recommended way to isolate dependencies.

---

### 2) Install dependencies

> Important: On Windows with GPU, the `torch/torchaudio/torchvision` stack must be strictly aligned by version and build type (CPU or `+cuXXX`). This is the main compatibility bottleneck.

**CPU option (simpler):**

```powershell
pip install -r requirements/lock-win-cpu.txt
pip install -e .
```

**CUDA option (example, depends on your setup):**

```powershell
pip install -r requirements/lock-win-cuda-cu128.txt
pip install -e .
```

---

### 3) Set the HF token (for diarization)

The token is **not stored in code** and must never be committed.
Use either a Windows environment variable or a local `.env` file (ignored by git).

Example `.env`:

```env
HF_TOKEN=hf_xxx
```

---

## Reproducible setup on another Windows machine

Once you have a working environment, capture the exact dependency versions for portability:

### Generate a lock file

```powershell
# Activate your working venv first
.\.venv\Scripts\Activate.ps1

# Generate lock file (CPU profile example)
pip freeze > requirements/lock-win-cpu.txt
```

For CUDA, use a descriptive name:

```powershell
pip freeze > requirements/lock-win-cuda-cu128.txt
```

### Recreate the environment on another machine

1. Copy the project (without `.venv`)
2. Copy the lock file
3. On the target machine:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements/lock-win-cpu.txt
pip install -e .
```

This guarantees the same versions are installed. The lock file records **all** packages (including transitive dependencies), so it's fully reproducible.

> **Note:** Lock files are machine-specific for Torch (CPU vs CUDA builds). Generate one for each profile you use.

---

## Quick Start

### Environment Check

```powershell
python -m diarrhizer doctor
```

The `doctor` command performs 5 diagnostic checks:

1. **Python version** — verifies Python 3.11+
2. **FFmpeg** — checks availability in PATH
3. **PyTorch/Torchaudio** — verifies installation and reports version
4. **CUDA** — checks GPU availability
5. **Hugging Face token** — verifies `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is set

---

### Run Processing

Basic usage:

```powershell
python -m diarrhizer run "D:\records\meeting.mp4" --out ".\out" --min-speakers 2 --max-speakers 6 --lang ru --device cuda
```

With speaker name mapping:

```powershell
python -m diarrhizer run "D:\records\meeting.mp4" --out ".\out" --speakers ".\speakers.json"
```

All options:

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Path to input media file | (required) |
| `--out` | Output directory | `./out` |
| `--min-speakers` | Minimum number of speakers | 1 |
| `--max-speakers` | Maximum number of speakers | 10 |
| `--lang` | Language code or `auto` | `auto` |
| `--device` | Device to use (`cuda` or `cpu`) | `cuda` |
| `--force` | Force recompute all stages | false |
| `--force-stage` | Force recompute specific stage | none |
| `--speakers` | Path to JSON speaker mapping file | none |

---

## Planned Pipeline

The processing pipeline consists of 5 stages:

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

---

## Output Structure (Planned Artifacts)

Each run will create a job-specific folder inside `--out`, containing artifacts by stage:

* `audio/` — normalized WAV
* `asr/` — WhisperX transcript (timestamps/words)
* `diar/` — diarization result
* `merged/` — merged segments (text + speaker)
* `export/` — final `.md`, `.txt`, `.json`

This allows:

* Reusing cached intermediate results (avoid recomputing heavy stages)
* Re-exporting in new formats without rerunning ASR/diarization

---

## Troubleshooting (Essentials)

* `pip check` — check for dependency conflicts
* `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`
* `where ffmpeg` and `ffmpeg -version`
* If pyannote/torchcodec reports FFmpeg or compatibility issues:

  * Verify FFmpeg installation (shared DLL build)
  * Ensure torchcodec ↔ torch versions are compatible
  * A fallback may be used: loading waveform into memory to bypass internal decoding

See `docs/troubleshooting.md` (to be expanded during development).

---

## Development

* CLI entry point: [`src/diarrhizer/cli.py`](src/diarrhizer/cli.py)
* Pipeline runner: [`src/diarrhizer/pipeline/runner.py`](src/diarrhizer/pipeline/runner.py) (to be implemented)
* Pipeline stages: [`src/diarrhizer/pipeline/stages/`](src/diarrhizer/pipeline/stages/) (to be implemented)
* External integrations (adapters): [`src/diarrhizer/adapters/`](src/diarrhizer/adapters/) (to be implemented)
* Diagnostics: [`src/diarrhizer/diagnostics/doctor.py`](src/diarrhizer/diagnostics/doctor.py)

---

## Roadmap

1. Implement pipeline stages (convert, transcribe, diarize, merge, export)
2. Implement adapters (FFmpeg, WhisperX, diarization)
3. Stable CLI + caching + environment diagnostics
4. UX improvements: speaker name mapping, protocol formatting
5. GUI + additional features

---

## License

Personal project.
