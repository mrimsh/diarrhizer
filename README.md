# Diarrhizer

By MRIMSH, L. L. MRIMSH and all other LLMs

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
* `setuptools<81` — WhisperX 3.3.1 depends on `pkg_resources` at runtime; setuptools 81+ removed it. This is pinned automatically via `constraints-stable.txt`.
* For diarization: Hugging Face token + acceptance of gated model terms
* (Optional) NVIDIA GPU + CUDA-compatible Torch wheels

> **GPU / cuDNN note:** WhisperX 3.3.1 requires `ctranslate2<4.5.0`, which depends on **cuDNN 8**. PyTorch `>=2.4.0+cu124` ships **cuDNN 9**, causing `Could not locate cudnn_ops_infer64_8.dll` at transcribe time. The stable GPU path uses **torch 2.3.1+cu121** (cuDNN 8). See [troubleshooting](docs/troubleshooting.md) for diagnostics.

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

> **Important:** On Windows with GPU, the `torch/torchaudio` stack must be strictly aligned by version **and** cuDNN major version. `torch.cuda.is_available() == True` does **not** guarantee that WhisperX / CTranslate2 can actually use the GPU — cuDNN 8 vs 9 DLL mismatch is a silent failure that only appears during transcription.

**CPU option (simpler, recommended first):**

```powershell
pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cpu.txt
pip install -e .
```

**CUDA option (NVIDIA GPU, CUDA 12.1 cuDNN 8 — stable supported path):**

```powershell
pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cuda-cu121.txt
pip install -e .
```

This installs `torch==2.3.1+cu121` which bundles cuDNN 8 — compatible with `ctranslate2==4.4.0` (required by WhisperX 3.3.1).

> **Why cu121 and not cu124?** WhisperX 3.3.1 pins `ctranslate2<4.5.0`, which requires cuDNN 8. PyTorch cu124 wheels ship cuDNN 9, causing `Could not locate cudnn_ops_infer64_8.dll`. Until WhisperX lifts the ctranslate2 upper bound, the cu121 path is the only stable GPU option.

**Verify your GPU setup after install:**

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
pip show whisperx faster-whisper ctranslate2 pyannote.audio torch torchaudio
dir .venv\Lib\site-packages\torch\lib\cudnn*.dll
```

You should see `cudnn_ops_infer64_8.dll` (cuDNN 8) in the last command. If you see only `cudnn*64_9.dll` (cuDNN 9), the setup is broken — see [troubleshooting](docs/troubleshooting.md).

---

### 3) Set the HF token (for diarization)

The token is **not stored in code** and must never be committed.
Use either a Windows environment variable or a local `.env` file (ignored by git).
Both `python -m diarrhizer` and the GUI load `.env` automatically (process env wins, then cwd `.env`, then the repo-root `.env`).

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

The `doctor` command performs these diagnostic checks:

1. **Python version** — verifies Python 3.11+
2. **FFmpeg** — checks availability in PATH (or `DIARRHIZER_FFMPEG_PATH`)
3. **PyTorch/Torchaudio** — verifies installation and reports version
4. **CUDA** — checks GPU availability
5. **cuDNN** — checks cuDNN 8 DLLs needed by WhisperX/CTranslate2 (skipped on CPU-only)
6. **torchcodec** — optional fast decoder
7. **Critical imports** — whisperx / speechbrain / pyannote / transformers
8. **Hugging Face token** — verifies `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is set (including from a local `.env`)

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
| `input` | Path to input media file | (required, unless `--job-dir` resumes a job that already recorded it) |
| `--out` | Output directory | `./out` (ignored when `--job-dir` is given) |
| `--job-dir` | Resume an existing job directory instead of starting a new job | none |
| `--min-speakers` | Minimum number of speakers | 1 |
| `--max-speakers` | Maximum number of speakers | 10 |
| `--min-turn-duration` | Shortest speaker turn (seconds) the `merge` stage splits out on its own; `0` splits on every word-level speaker change | 0.4 |
| `--lang` | Language code or `auto` | `auto` |
| `--device` | Device to use (`cuda` or `cpu`) | `cuda` |
| `--asr-model` | WhisperX/Whisper model size or HF repo (e.g. `base`, `small`, `medium`, `large-v3`) — bigger models are more accurate but slower and need more VRAM | `large-v3` |
| `--force` | Force recompute all stages | false |
| `--force-stage` | Force recompute specific stage | none |
| `--from-stage` | Start the pipeline at this stage, skipping earlier ones (their outputs must already exist on disk) | none (starts at `convert`) |
| `--to-stage` | Stop the pipeline after this stage, skipping later ones | none (runs through `export`) |
| `--speakers` | Path to JSON speaker mapping file | none |
| `--audio-profile` | Audio preprocessing profile: `raw`, `voice-call`, `denoise-light`, `split-stereo` | `raw` |
| `--audio-format` | How the job keeps its audio on disk: `wav`, `flac`, `mp3-q5`, `opus-24k` or a profile from `audio_formats.json` | the default profile in `audio_formats.json`, else `wav` |
| `--keep-wav` | Keep the working `audio/normalized.wav` after the job completes (overrides the format's setting) | as the format says |

---

### Audio Profiles

`--audio-profile` controls FFmpeg preprocessing during the `convert` stage. Every
profile writes the same `audio/normalized.wav` that the rest of the pipeline
reads, so switching profiles never changes which stages run:

* `raw` — no filtering.
* `voice-call` — bandpass filter (300Hz-7kHz) + mild EQ boost, for phone/VoIP recordings.
* `denoise-light` — light noise reduction (`afftdn`) for noisy recordings.
* `split-stereo` — in addition to the usual mono `normalized.wav`, also writes
  `normalized_left.wav`/`normalized_right.wav` (separated L/R channels) as
  extra artifacts. These per-channel files are not currently consumed by
  transcribe/diarize/merge - diarization still runs on the mono mix as usual.

See [`docs/architecture.md`](docs/architecture.md#5-audio-profiles) for details.

---

### Audio Storage Formats (disk space vs. quality)

Transcription and diarization always run on a lossless working file,
`audio/normalized.wav` (PCM 16 kHz mono, ~115 MB per hour of audio).
`--audio-format` decides what the job **keeps** on disk:

| Format | Archive copy | Working WAV after the job | Size per hour |
|--------|--------------|---------------------------|---------------|
| `wav` (default) | none | kept (it is the only file) | ~115 MB |
| `flac` | `audio/archive.flac`, lossless | deleted | ~50-75 MB |
| `mp3-q5` | `audio/archive.mp3` (`-c:a libmp3lame -q:a 5`) | deleted | ~15-20 MB |
| `opus-24k` | `audio/archive.opus` (`-c:a libopus -b:a 24k`) | deleted | ~11 MB |

Sizes were measured at 16 kHz mono on a speech sample; VBR and lossless sizes vary with the recording.

```powershell
# Keep only a small MP3 of the call:
python -m diarrhizer run "D:\records\call.m4a" --audio-format mp3-q5

# Keep the MP3 and the WAV:
python -m diarrhizer run "D:\records\call.m4a" --audio-format mp3-q5 --keep-wav true
```

* The archive is encoded from the original input, always with `-vn` (no video),
  and gets the same `--audio-profile` filters as the WAV.
* The WAV is deleted only once the whole job has completed (a `--to-stage` run
  that stops early keeps it for the stages still to come), and only if the
  archive exists. Later re-runs of transcribe/diarize decode the archive.
* Changing the format of an existing job (`--job-dir … --audio-format opus-24k`)
  re-encodes only the archive; ASR and diarization stay cached. Files left
  behind by the previous format are removed. The exception: turning
  `--keep-wav true` back on after the WAV was deleted recreates it, and
  transcribe/diarize then run again.

**Profiles.** Your own profiles and the default one live in `audio_formats.json`
in the repo root (gitignored like `.env`; set `DIARRHIZER_AUDIO_FORMATS_FILE` to
keep it elsewhere). The GUI edits it under Settings → «Хранение аудио», and the
CLI uses the same default. To write one by hand, start from
[`audio_formats.example.json`](audio_formats.example.json). Fields: `codec`
(ffmpeg `-c:a`), `extension`, `quality` (`-q:a`) or `bitrate` (`-b:a`, e.g.
`"32k"`), `sample_rate`, `channels`, `extra_args` (a list of extra ffmpeg output
options), `keep_wav`, `description`. Built-in names cannot be redefined.
`python -m diarrhizer doctor` checks that the file parses and that your FFmpeg
build has the encoder the default profile needs.

There is no AAC/M4A preset on purpose: diarization's torchaudio fallback cannot
decode it on Windows, and the archive becomes the pipeline's input once the WAV
is gone.

---

### Resuming a Job / Re-running Only Part of the Pipeline

`--job-dir` points the CLI at an existing job folder instead of starting a new one. If that job already went through `convert`, its `meta/run.json` has the original input path recorded, so `input` can be omitted entirely. Combined with `--from-stage`/`--to-stage`, this lets you re-run just a slice of the pipeline without recomputing the expensive stages (ASR, diarization):

```powershell
# Only re-export (e.g. after editing speakers.json) - no ASR/diarization recompute:
python -m diarrhizer run --job-dir ".\out\meeting_20260101_120000" --from-stage export --speakers ".\speakers.json"

# Resume a job that stopped partway through, starting at merge:
python -m diarrhizer run --job-dir ".\out\meeting_20260101_120000" --from-stage merge
```

Stages outside the `--from-stage`/`--to-stage` range are skipped entirely (not even checked); stages inside it still use the normal cache (only recomputed if their inputs changed or `--force`/`--force-stage` was passed). If the first stage in range doesn't have the artifacts it needs on disk yet, the command fails with a clear error naming the missing file instead of a confusing crash.

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

* `audio/` — normalized WAV and/or its archive copy (see [Audio Storage Formats](#audio-storage-formats-disk-space-vs-quality))
* `asr/` — WhisperX transcript (timestamps/words)
* `diar/` — diarization result
* `merged/` — merged segments (text + speaker)
* `export/` — final `.md`, `.txt`, `.json`

This allows:

* Reusing cached intermediate results (avoid recomputing heavy stages)
* Re-exporting in new formats without rerunning ASR/diarization, via `--job-dir <job folder> --from-stage export` (see [Resuming a Job](#resuming-a-job--re-running-only-part-of-the-pipeline))

---

## Troubleshooting (Essentials)

* `python -m diarrhizer doctor` — run full environment diagnostics
* `pip check` — check for dependency conflicts
* `python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"` — verify torch/CUDA
* `pip show whisperx faster-whisper ctranslate2 pyannote.audio torch torchaudio` — check installed versions
* `dir .venv\Lib\site-packages\torch\lib\cudnn*.dll` — verify cuDNN 8 DLLs present (GPU setups)
* `where ffmpeg` and `ffmpeg -version`
* `Could not locate cudnn_ops_infer64_8.dll` — cuDNN mismatch, see [troubleshooting](docs/troubleshooting.md)
* `Import error: No module named 'pkg_resources'` — install `setuptools<81` (see `docs/troubleshooting.md`)
* **Fallback to CPU for verification:** if GPU fails with cryptic DLL errors, try `--device cpu` to confirm the pipeline works, then fix the GPU stack

See `docs/troubleshooting.md` for detailed diagnostics.

---

## Development

* CLI entry point: [`src/diarrhizer/cli.py`](src/diarrhizer/cli.py)
* Pipeline runner: [`src/diarrhizer/pipeline/runner.py`](src/diarrhizer/pipeline/runner.py)
* Pipeline stages: [`src/diarrhizer/pipeline/stages/`](src/diarrhizer/pipeline/stages/)
* External integrations (adapters): [`src/diarrhizer/adapters/`](src/diarrhizer/adapters/)
* Diagnostics: [`src/diarrhizer/diagnostics/doctor.py`](src/diarrhizer/diagnostics/doctor.py)

---

## Roadmap

1. Implement pipeline stages (convert, transcribe, diarize, merge, export)
2. Implement adapters (FFmpeg, WhisperX, diarization)
3. Stable CLI + caching + environment diagnostics
4. UX improvements: speaker name mapping, protocol formatting
5. GUI + additional features

---

## Author

By MRIMSH, L. L. MRIMSH and all other LLMs

## License

Personal project.
