# Diarrhizer

A local (on-prem) Windows tool for processing call recordings:
input audio/video file → speech recognition (WhisperX) → speaker diarization (pyannote via WhisperX) → export of a structured protocol (`.md/.txt`) and machine-readable output (`.json`).

The project is intended for personal use and iterative development (future plans include a GUI, additional formats, and post-processing features).

---

## MVP Features

* Supported input formats: `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mkv`, `.webm`, etc.
* Audio normalization via FFmpeg (typically: WAV mono 16 kHz)
* WhisperX transcription with word-level timestamps (alignment)
* Speaker diarization (`Speaker_00`, `Speaker_01`, …) with text-to-speaker assignment
* Export:

  * Human-readable `.md/.txt` (segments + timestamps + speaker)
  * Structured `.json` (for further processing without recomputing ASR/diarization)

### Important About Speakers

Diarization returns **speaker identifiers** (`Speaker_00...`), not real names.
Real names are expected to be provided via **manual mapping** (a mapping file).
Voice enrollment or automatic speaker identification may be added later.

---

## Requirements

* Windows 10/11
* Python 3.11/3.12 (recommended for stable Torch stack on Windows)
* FFmpeg available in `PATH` (also important for decoding via torchcodec/pyannote on Windows)
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
pip install -r requirements/base.txt
pip install -r requirements/cpu.txt
pip install -e .
```

**CUDA option (example, depends on your setup):**

```powershell
pip install -r requirements/base.txt
pip install -r requirements/cuda-cu128.txt
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

## Quick Start

### Environment Check

```powershell
python -m diarrhizer doctor
```

Diagnostics should verify:

* Python and Torch versions, CUDA availability
* FFmpeg availability
* Presence of HF token
* (If applicable) torchcodec import

---

### Run Processing

```powershell
python -m diarrhizer run "D:\records\meeting.mp4" --out ".\out" --min-speakers 2 --max-speakers 6 --lang ru --device cuda
```

---

## Output Structure (Artifacts Directory)

Each run creates a job-specific folder inside `--out`, containing artifacts by stage:

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

* CLI entry point: `src/diarrhizer/cli.py`
* Pipeline stages: `src/diarrhizer/pipeline/stages/`
* External integrations (adapters): `src/diarrhizer/adapters/`

---

## Roadmap (Very Brief)

1. Stable CLI + caching + environment diagnostics
2. UX improvements: speaker name mapping, protocol formatting
3. GUI + additional features

---

## License

Personal project.
