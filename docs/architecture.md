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
    -> artifacts/audio/normalized.wav (+ archive.<ext> for non-wav storage formats)
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
      run.json            # written by convert; records the original input_path
    audio/
      normalized.wav      # always written, regardless of audio_profile; deleted after the job
                          # completes when the storage format has keep_wav=false
      normalized_left.wav  # split-stereo only: extra left-channel artifact (not consumed downstream)
      normalized_right.wav # split-stereo only: extra right-channel artifact (not consumed downstream)
      archive.<ext>       # non-wav storage formats only: archive copy (+ archive_left/right.<ext>)
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

`meta/run.json`'s recorded `input_path` is also what `run --job-dir <dir>` reads to make the `input` argument optional when resuming a job (see [Section 8](#8-cli-commands)).

---

## 4. Data Format (Overview)

### ASR output

Contains segments and/or words with timestamps (WhisperX alignment result).

### Diarization output

Contains speech intervals labeled with speaker identifiers (`Speaker_00...`).  
Real names are applied through a separate mapping layer.

### Merged segments

The "stitched" result: text + timestamps + speaker at segment (and/or word) level.

Every merged segment has exactly one speaker: `speaker_id` always matches the
`speaker_id` of each entry in its `words`. Whisper cuts segments on pauses and
punctuation rather than on who is talking, so an ASR segment can span a speaker
change; the merge stage splits such a segment at the boundary between
consecutive same-speaker word runs. Merged segments can therefore outnumber the
ASR segments they came from. Word-level speaker flips shorter than
`min_turn_duration` (`--min-turn-duration`, default `0.4`s) that sit between two
runs of the same other speaker are treated as diarization jitter and folded back
in, so a stray word can't fragment a sentence. Raise it when noisy diarization
shreds sentences; `0` splits on every word-level speaker change. The value is
recorded in `metadata.min_turn_duration`, and changing it invalidates the
merge stage's cache.

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
      "model": "large-v3",
      "compute_type": "float16",
      "beam_size": 5,
      "temperature": 0.0
    },
    "audio_profile": "raw"
  }
}
```

**Algorithm:** Each word is assigned the diarization speaker with maximum time
overlap; the segment is then split into consecutive same-speaker word runs, one
output segment per run. A segment with no word timestamps is matched as a whole
and emitted unsplit.

**Edge cases:**
- No diarization data: defaults to "Speaker_00"
- Gaps in diarization: uses closest segment by time
- Overlapping speakers: chooses speaker with most overlap

**Important:** Speaker names are identifiers only.  
Name mapping is a separate layer on top.

---

## 5. Audio Profiles

Audio profiles apply FFmpeg filters during conversion. Every profile writes the
standard mono `audio/normalized.wav` that transcribe/diarize/merge read
unconditionally, so the pipeline shape never depends on `audio_profile`.

| Profile | Filters Applied | Use Case |
|---------|-----------------|----------|
| `raw` | None | Default behavior, no preprocessing |
| `voice-call` | Bandpass filter (300Hz-7kHz) + mild EQ boost at 3kHz | Phone call recordings, VoIP |
| `denoise-light` | afftdn noise reduction | Noisy recordings with background noise |
| `split-stereo` | Writes the standard mono downmix to `normalized.wav` (used by the rest of the pipeline) **and** additionally splits L/R channels into `normalized_left.wav`/`normalized_right.wav` | Keeping raw per-channel audio available for manual inspection; the channels themselves are not currently fed into transcribe/diarize/merge - diarization still runs as usual on the mono mix |

### Audio storage formats

Profiles decide *what the pipeline processes*; storage formats
(`src/diarrhizer/audio_formats.py`, `--audio-format`) decide *what the job keeps*.
Processing always uses the lossless working WAV. A format other than `wav` makes
convert also write an archive copy, `audio/archive.<ext>` (plus
`archive_left/right.<ext>` for `split-stereo`). Each copy is encoded from the
original input with the format's codec, `-q:a` or `-b:a`, sample rate, channel
count and extra args, with `-vn`, and gets the same profile filters as the WAV.
Formats with `keep_wav: false` drop the working WAV(s) once the job completes.

Lifecycle and cache rules:

* **Deletion happens in a finalize hook.** The runner calls each stage's
  optional `finalize(job)` once a run reaches the last stage.
  `ConvertStage.finalize` deletes a WAV only if its archive exists. A run cut
  short with `--to-stage` keeps the WAV for the stages still to come.
* **Downstream decoding.** Transcribe and diarize decode `find_job_audio()`: the
  WAV while it exists, otherwise the archive. Their cache staleness stays keyed
  on the WAV path, and a missing input is never "newer", so dropping the WAV
  never invalidates ASR or diarization.
* **Convert's cache** follows what `meta/run.json` recorded in
  `config.audio_format`: the archive is an expected output, and the WAV only
  while the format keeps it. A different encoding or `keep_wav` makes convert
  re-run. Jobs from before formats existed count as `wav`.
* **Changing only the storage** re-encodes the archive without touching the WAV,
  as long as the source file and `audio_profile` are unchanged. A WAV that
  exists is reused as is (same mtime). A WAV that was already dropped stays
  dropped. Either way ASR and diarization stay cached. A newer source, a
  profile change, `--force`, or restoring a dropped WAV with `keep_wav: true`
  rewrites the WAV, and the later stages re-run.
* **Cleanup of old files.** Once the new files are written, convert deletes
  `normalized*.wav` / `archive*.*` files that the new outputs no longer
  include, e.g. `archive.mp3` after switching to Opus.

User profiles and the default name are stored in `audio_formats.json` at the
repo root (`DIARRHIZER_AUDIO_FORMATS_FILE` overrides the path). CLI and GUI share
this file. Built-in names (`wav`, `flac`, `mp3-q5`, `opus-24k`) are read-only.
There is no AAC preset because diarization's torchaudio fallback (libsndfile)
cannot decode M4A on Windows. FLAC is pinned to `-sample_fmt s16`: from a
float-decoded source (mp3/aac), ffmpeg otherwise writes 24-bit FLAC, which
ended up larger than the WAV.

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

The [`doctor`](src/diarrhizer/diagnostics/doctor.py) command performs these checks:

1. **Python version** — verifies Python 3.11+
2. **FFmpeg** — checks availability via `DIARRHIZER_FFMPEG_PATH` or PATH
3. **PyTorch/Torchaudio** — verifies installation and reports version and CUDA status
4. **CUDA** — checks GPU availability via `torch.cuda.is_available()`
5. **cuDNN** — looks for cuDNN 8 DLLs required by WhisperX/CTranslate2 (skipped on CPU-only)
6. **torchcodec** — optional fast decoder
7. **Critical imports** — whisperx / speechbrain / pyannote / transformers
8. **Hugging Face token** — verifies `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (CLI and GUI load a local `.env`)

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
| `input` | positional | — | Path to input media file (required, unless `--job-dir` resumes a job that already recorded it in `meta/run.json`) |
| `--out` | string | `./out` | Output directory (ignored when `--job-dir` is given) |
| `--job-dir` | string | — | Resume an existing job directory instead of starting a new one, e.g. `out/meeting_20260101_120000` |
| `--min-speakers` | int | `1` | Minimum number of speakers |
| `--max-speakers` | int | `10` | Maximum number of speakers |
| `--min-turn-duration` | float | `0.4` | Shortest speaker turn (s) split out on its own |
| `--lang` | string | `"auto"` | Language code or `"auto"` for detection |
| `--device` | choice | `"cuda"` | Device: `cuda` or `cpu` |
| `--asr-model` | string | `"large-v3"` | WhisperX model (or HF repo); use a smaller size (e.g. `base`) for faster/cheaper runs |
| `--asr-compute-type` | string | auto | Compute type: `float16`, `int8_float16`, `int8` |
| `--asr-beam-size` | int | `5` | Decoding beam size |
| `--asr-temperature` | float | `0.0` | Decoding temperature |
| `--asr-condition-on-previous-text` | string | `"true"` | previous text conditioning (true/false) |
| `--asr-initial-prompt-file` | string | — | Path to prompt/glossary file |
| `--asr-hotwords-file` | string | — | Path to hotwords file (not yet implemented) |
| `--asr-vad-filter` | string | `"true"` | Enable VAD filtering (true/false) |
| `--asr-vad-min-silence-ms` | int | `1000` | VAD minimum silence (ms) |
| `--audio-profile` | choice | `"raw"` | Audio preprocessing |
| `--audio-format` | string | default profile in `audio_formats.json`, else `"wav"` | Storage format of the job's audio (see [Audio storage formats](#audio-storage-formats)) |
| `--keep-wav` | string | — (format's own setting) | Keep the working WAV after the job completes (true/false) |
| `--force-stage` | choice | — | Force recompute specific stage |
| `--from-stage` | choice | — | Start at this stage, skipping earlier ones entirely (their outputs must already be on disk) |
| `--to-stage` | choice | — | Stop after this stage, skipping later ones entirely |

> **Note:** Pipeline runs: convert → transcribe → diarize → merge → export.
> `--from-stage`/`--to-stage` select a sub-range of that order (choices: `convert`, `transcribe`, `diarize`, `merge`, `export`); stages outside the range are skipped entirely (not even cache-checked), while stages inside it keep the normal cache behavior instead of being unconditionally recomputed. Combine with `--job-dir` to resume a job partway through - e.g. `--job-dir out/meeting_20260101_120000 --from-stage export` re-exports without recomputing ASR/diarization.

**Example:**

```powershell
python -m diarrhizer run "meeting.mp4" `
  --out ".\out" `
  --min-speakers 2 --max-speakers 3 `
  --lang ru `
  --device cuda `
  --asr-model "koekaverna/faster-whisper-podlodka-turbo" `
  --asr-compute-type float16 `
  --asr-beam-size 5 `
  --asr-condition-on-previous-text false `
  --asr-initial-prompt-file ".\glossaries\default_ru.txt" `
  --audio-profile voice-call
```

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
