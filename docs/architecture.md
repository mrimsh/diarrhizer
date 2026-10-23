# Architecture

Goal: a simple, extensible architecture for local call recording processing on Windows:  
**FFmpeg → WhisperX (ASR + alignment) → diarization (pyannote via WhisperX) → merge → export (MD/TXT/SRT/JSON).**

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
| Desktop GUI | ✅ Implemented | [`src/diarrhizer_gui/`](src/diarrhizer_gui/) — see [GUI](#gui) |
| Summaries (templates, providers, `summarize` command) | ✅ Implemented | [`src/diarrhizer/summary/`](src/diarrhizer/summary/), [`src/diarrhizer/adapters/llm.py`](src/diarrhizer/adapters/llm.py) — see [Summaries](#summaries) |

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
      run.json            # written by convert; records the original input_path and the job's
                          # full settings (pipeline_config: ASR params incl. the initial prompt,
                          # speaker range, min_turn_duration, audio profile/format, stage list);
                          # a later stage that runs with other settings (--from-stage, GUI "restart
                          # stage") updates its own keys (runner.STAGE_SETTINGS) after it ran
    speakers.json         # GUI only: speaker names {Speaker_XX: name}
    audio/
      normalized.wav      # always written, regardless of audio_profile; deleted after the job
                          # completes when the storage format has keep_wav=false
      normalized_left.wav  # split-stereo only: extra left-channel artifact (not consumed downstream)
      normalized_right.wav # split-stereo only: extra right-channel artifact (not consumed downstream)
      archive.<ext>       # non-wav storage formats only: archive copy (+ archive_left/right.<ext>)
    asr/
      transcript.json     # includes ASR config in metadata
      transcript.txt      # readable paragraphs, no speakers/timestamps (built from the same segments)
    diar/
      diarization.json
    merged/
      segments.json
    export/
      result.md           # speaker turns + speaking-time summary
      result.txt          # speaker turns, no markup
      result.srt          # subtitles, "Name: text" per cue
      result.json         # every segment and word with timings
    summary/              # on-demand, not a pipeline stage (see Summaries); never cached or invalidated
      <YYYYmmdd_HHMMSS>_<template>.md    # the model's reply, verbatim
      <YYYYmmdd_HHMMSS>_<template>.json  # template + prompt, provider (name/kind/model), duration,
                                         # transcript size, whether the reply was cut off
```

`job_id` can be generated as:

- filename + timestamp, or
- hash (input + config) — if reproducibility/deduplication is important.

`meta/run.json`'s recorded `input_path` is also what `run --job-dir <dir>` reads to make the `input` argument optional when resuming a job (see [Section 8](#8-cli-commands)). The GUI's "Продолжить" and "Перезапустить этап" (History) and "Пересчитать спикеров" (Result) rebuild the whole run from `pipeline_config` (`diarrhizer_gui/job_scan.py`, `GUI:JOB_PLANS`). Restarting a stage is `--job-dir` + `--from-stage X` + `--force-stage X`: the stage's own outputs are dropped, and the stages that consumed them (convert → transcribe, diarize; transcribe, diarize → merge; merge → export) recompute through the mtime-based cache. The History dialog (`GUI:RESTART_DIALOG`) may change the settings of exactly those recomputed stages (`runner.STAGE_SETTINGS`, filtered by `job_scan.restart_settings`); `restart_plan(..., overrides)` rejects any other setting.

Job folders are named `<input stem>_<YYYYMMDD_HHMMSS>`; callers that start several jobs in a row (GUI queue, CLI batch) use `runner.new_job_dir()`, which appends `_2`, `_3`, … rather than reuse an existing folder.

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

1. **Python version** — verifies Python 3.11 or 3.12 (whisperx 3.3.1 declares `Requires-Python <3.13`, torch 2.3.1 has no wheels past 3.12)
2. **FFmpeg** — checks availability via `DIARRHIZER_FFMPEG_PATH` or PATH
3. **PyTorch/Torchaudio** — verifies installation and reports version and CUDA status
4. **CUDA** — checks GPU availability via `torch.cuda.is_available()`
5. **cuDNN** — looks for cuDNN 8 DLLs required by WhisperX/CTranslate2 (skipped on CPU-only)
6. **torchcodec** — optional fast decoder; a failure is a `WARN` and does not fail the run (`OPTIONAL_CHECKS`)
7. **Critical imports** — whisperx / speechbrain / pyannote / transformers
8. **Hugging Face token** — verifies `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (CLI and GUI load a local `.env`)
9. **Summarization** — in `OPTIONAL_CHECKS` (an add-on: a problem is a `WARN`): `summary_templates.json` / `summary_providers.json` parse, every CLI provider's command is on `PATH`, every API provider's key variable is set. Whether a CLI is signed in is not checked - that would cost a model request (the GUI's "Проверить" does it on demand).

Besides `doctor`, the pipeline itself checks the token up front: `DiarizeStage.preflight()` runs before
any stage, so a run that will have to diarize without a token fails in seconds instead of after the
transcription. Cached diarization (e.g. `--from-stage export`) needs no token.

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
| `input` | positional, 0..n | — | Media file(s) and/or folder(s) (a folder adds the media files directly inside it). Several inputs run as a batch: one job per file, a failed file does not stop the rest, exit code 1 if any failed. Optional when `--job-dir` resumes a job that already recorded its input (at most one input with `--job-dir`) |
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

### `summarize`

```powershell
python -m diarrhizer summarize "<job folder or its export/result.md>" --template "Протокол встречи" --provider ChatGPT --print
python -m diarrhizer summarize --list
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `job` | positional, optional | — | Job folder, or its `export/result.md`. Required unless `--list` |
| `--template` | string | the default template | Template name (`--list` shows them) |
| `--provider` | string | the default provider | Provider name from `summary_providers.json`; with none configured the error says where to add one |
| `--note` | string | — | Your text for this run: it goes where the template has `{{note}}`; a template without it gets the note before the transcript, with a warning. Not together with `--note-file` |
| `--note-file` | path | — | The same, read from a text file (UTF-8, or UTF-16 with a BOM) |
| `--print` | flag | off | Also print the summary (it is always stored under the job's `summary/`) |
| `--list` | flag | off | List templates and providers and exit |
| `--list-wildcards` | flag | off | List the `{{wildcards}}` a template may use, with descriptions, and exit (no job needed) |

Exit code 1 for a missing job/result or note file, an unknown template, provider or wildcard, or a provider
failure (2 for `--note` together with `--note-file`). See [Summaries](#summaries) below.

---

## Summaries

A summary of a finished job's transcript, written by a language model. It is an **on-demand add-on, not a
pipeline stage**: it costs money or subscription quota, may run several times per job with different
templates, and has no cache to invalidate — so it lives beside the stages (`summary/` in the job folder),
not in them.

```
export/result.md ──► SUMMARY:JOB ──► ADAPTER:LLM ──► provider (CLI or HTTPS API)
(read as it is,      render_prompt:                      │
 names included)     {{wildcards}} filled in,            ▼
                     transcript / note appended   summary/<stamp>_<template>.md + .json
                     where the template has no place
```

- **Templates** (`summary/templates.py`, `CONFIG:SUMMARY_TEMPLATES`): a named prompt. Built-ins are read-only;
  user templates and the default name are stored in `summary_templates.json` (same shape of store as
  `audio_formats.json`: missing file = empty, malformed = `ValueError` naming the file).
- **Wildcards** (`summary/wildcards.py`, `CONFIG:SUMMARY_WILDCARDS` / `SUMMARY:WILDCARDS`): a template says where
  a value goes by writing `{{name}}`. The one registry `WILDCARDS` (name → description → function of a
  `PromptContext`) defines them; validation, the `summarize --list-wildcards` listing and the error that names
  the valid ones all read it, so a new wildcard is one entry there (plus a line in the README table).

  | Wildcard | Value | Source |
  |----------|-------|--------|
  | `{{transcript}}` | the transcript in `<transcript>…</transcript>` | `export/result.md` |
  | `{{note}}` | the user's text for this run; `""` when none | `summarize_job(note=…)` |
  | `{{job_name}}` | input file name without extension | `meta/run.json` `input_path`, else the folder name |
  | `{{date}}` | `YYYY-MM-DD` of the job | folder name `<stem>_<YYYYMMDD>_<HHMMSS>`, else `meta/run.json` `start_time` |
  | `{{duration}}` | `HH:MM:SS` up to the last phrase | `merged/segments.json` (largest `end`) |
  | `{{speakers}}` | speakers in order of appearance, comma-separated | `merged/segments.json` + the names in `speakers.json` (an unnamed speaker keeps `Speaker_NN`) |
  | `{{language}}` | language code | `meta/run.json` `pipeline_config.language`; for `auto` the detected one from `asr/transcript.json` |

  Syntax: `{{name}}`, spaces inside the braces allowed; single braces and `{{` + a non-name (JSON) are never
  touched; `\{{` is an escaped literal `{{`. One left-to-right pass, so a value is never scanned again. Job
  facts come from what the job already records and are `""` when it is missing or malformed - never an error.
  Rules: no `{{transcript}}` → the fenced transcript is appended (templates written before wildcards are
  unchanged); a note for a template without `{{note}}` is added after the template's text (before the appended
  transcript, else at the end) and reported as a warning (logged, and returned by `summarize_job`); an unknown
  name raises `UnknownWildcardError` (a `ValueError`) naming it, its line and the valid ones, before anything is
  sent. `{{date}}` is when the job was made: the pipeline does not record when the call took place.
- **Providers** (`summary/providers.py`, `CONFIG:SUMMARY_PROVIDERS`): `codex-cli` and `claude-cli` (the vendors'
  CLIs, i.e. the user's ChatGPT / Claude subscription), `openai-api` (any OpenAI-compatible
  chat/completions endpoint) and `anthropic-api`. Stored in `summary_providers.json`, which never contains a
  secret: an API provider names an environment variable (`api_key_env`); the GUI keeps the key in the
  repo-root `.env`, like the HF token. An API provider may carry prices per million tokens (cost estimate).
- **What a provider reports** (`summary/provider_info.py`, `CONFIG:PROVIDER_INFO`): `Account`, `ModelOption`,
  `Usage`, `Limits`/`LimitWindow` - plain dataclasses with `to_dict()`/`from_dict()` (they cross the GUI's
  process queue and are stored in a summary's `.json`). A field the provider did not report stays empty; nothing
  is defaulted on its behalf.
- **Request adapter** (`adapters/llm.py`, `ADAPTER:LLM`): sends one prompt and returns an `LLMReply` with the
  text and what the request used. CLIs get the prompt on stdin (no argv length or quoting limits) and run in an
  empty temporary folder, with the API-key variables that would switch the billing removed and with flags that
  skip the user's own tool setup (retried without them if the installed version rejects a flag); a timeout kills
  the whole process tree. Codex runs with `--json` (tokens from `turn.completed`), Claude with
  `--output-format stream-json` (the result event: reply, tokens, an estimated `total_cost_usd`; the last
  `rate_limit_event`: status, utilization, reset). APIs are one non-streaming `urllib` POST; usage comes from the
  reply, the cost from the provider (OpenRouter) or the provider's prices. The key is read at call time and
  never put into an error message. No SDK dependency.
- **Account adapter** (`adapters/llm_accounts.py`, `ADAPTER:LLM_ACCOUNTS`): signing in and out, the model list
  and the plan's limits of a subscription provider - through the vendors' own interfaces, never through tokens
  of ours. OpenAI documents `codex app-server` (JSON-RPC lines on stdio) as the way to embed Codex with ChatGPT
  sign-in: `account/read`, `account/login/start` (returns OpenAI's sign-in address; completion arrives as a
  notification), `account/login/cancel`, `account/logout`, `model/list`, `account/rateLimits/read`. Anthropic
  does not allow other applications to offer a Claude.ai sign-in or to collect, store or relay its credentials
  (*"sign-in to a Claude account must complete through Anthropic's own flow"*), so for Claude the adapter only
  starts the unmodified `claude auth login --claudeai` (it opens the browser itself), reads `claude auth status
  --json`, and gets the model list from the stream-json control protocol's `initialize` request (answered
  without a model request). Claude has no limits query: `read_limits` sends a one-word request and returns the
  status that came with the reply. API providers: `GET <endpoint>/models`.
- **Service** (`summary/service.py`, `SUMMARY:JOB`): reads `export/result.md`, builds the prompt
  (`summarize_job(…, note=None)`; the template's prompt is stored as written, with the user's fields
  `{"note": …}` and the `wildcards_used`; entries of older summaries list those two empty), asks the
  provider, and stores the reply atomically only after it arrived (a failed or cancelled request leaves
  nothing behind). Earlier summaries are never overwritten. The summary's `.json` also records the model that
  answered, the usage, the cost (with a note on where the figure comes from - an estimate is not a bill) and the
  limits; for Codex the limits are read right after the request (best effort, a failure is only logged).
- **Not done on purpose:** splitting a transcript that exceeds a model's context window (the provider's
  error is passed on as is), streaming, and chat-style follow-ups.

GUI: the **Саммаризация** screen (`screens/summary_settings_screen.py`, `GUI:SUMMARY_SETTINGS`) edits both
stores. A subscription provider there has an account box (`screens/account_panel.py`, `GUI:ACCOUNT_PANEL`):
the signed-in account and one button that signs in, cancels a sign-in or signs out, plus the plan's limits;
the model field is a list the provider fills (`screens/model_combo.py`) that still accepts a typed id. The
Result screen's **Саммаризация…** opens `screens/summary_dialog.py` (`GUI:SUMMARY_DIALOG`), which shows what
each request used. Everything that talks to a provider - a summary, a check, a sign-in, a model query - runs in
a child process (`llm_worker.py` + `child_tasks.run_summary_task` / `run_provider_check_task` /
`run_account_task`; a sign-in reports the address to open as an event), so it can be cancelled together with
whatever the CLI started. The Russian wording of usage, cost and limits is `provider_text.py`.

---

## 9. Extensibility (Future Growth)

The architecture is designed to allow extensions without breaking the core:

### New stages

- volume normalization, noise reduction
- text post-processing (punctuation/formatting)
- action items extraction as a stage (summaries exist, but as an on-demand add-on — see [Summaries](#summaries))

### New export formats

- HTML, DOCX, VTT (SRT exists - see `ExportStage.EXPORTERS`)
- integration with note-taking systems

### GUI

- The GUI layer must call the same `pipeline runner` and operate on the same artifacts.
- The GUI must not contain processing logic — only UX.

How the GUI runs things (`src/diarrhizer_gui/`):

- **Child processes.** Every pipeline job runs in a spawned child process
  (`child_process.py` + `child_tasks.py`, wrapped by `pipeline_worker.py`). Progress comes back as
  the runner's log records over a multiprocessing queue. The GUI process never imports
  torch/whisperx: "Остановить" kills the job at once (stages write atomically, so finished stages
  survive and the job can be resumed), a native CUDA/cuDNN crash is reported instead of closing the
  window, and all memory/VRAM is returned after each job.
- **Environment probe.** The doctor checks also run in a child process, once at startup
  (`environment.py`); New Job/Settings/Models read CUDA availability from it, and the Doctor screen
  and its sidebar badge show the results. The window opens immediately instead of waiting for the
  ML imports.
- **Queue.** New Job emits one run per file; History ("Продолжить", "Перезапустить этап") and Result
  ("Пересчитать спикеров") emit runs for existing job folders. `main_window.py` executes them one at a time
  (`job_queue.py`); the Monitor screen ("Выполнение") is a sidebar item showing the batch.
- **In-process work** is limited to cheap, pure steps: scanning job folders and re-exporting with
  new speaker names (`ExportStage` only, milliseconds).
- **Model calls** (a summary, a provider check) and provider **account actions** (sign-in, models, limits) also
  run in a child process (`llm_worker.py`): a request takes seconds to minutes, a sign-in waits for the user,
  and cancelling must kill the subscription CLI's own processes too (which frees its local sign-in port).

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

## 11. Project Structure

```
src/diarrhizer/
├── cli.py                  # CLI entry point (doctor, run, summarize)
├── diagnostics/
│   └── doctor.py           # Environment diagnostics
├── summary/                # Summaries of a finished job (stdlib only)
│   ├── templates.py        # Prompt templates + their JSON store
│   ├── providers.py        # Model providers + their JSON store
│   ├── provider_info.py    # What a provider reports: account, models, usage, limits
│   ├── store_file.py       # Reading the two JSON stores
│   └── service.py          # result.md + template + provider -> summary/ files
├── pipeline/
│   ├── runner.py           # Pipeline orchestration
│   └── stages/             # Individual processing stages
│       ├── convert.py      # FFmpeg normalization
│       ├── transcribe.py   # WhisperX ASR + alignment
│       ├── diarize.py      # Speaker diarization (pyannote)
│       ├── merge.py        # Merge ASR with speaker labels
│       └── export.py       # Export to MD/TXT/SRT/JSON
├── adapters/               # External library wrappers
│   ├── ffmpeg.py
│   ├── llm.py              # One request to a summary provider: codex/claude CLIs, OpenAI/Anthropic HTTP APIs
│   ├── llm_accounts.py     # Sign-in, account, model list and limits of a provider (vendors' own interfaces)
│   └── whisperx.py
├── media.py                # Media extensions, folder expansion for batch input
└── export/                 # Export formatters
    ├── turns.py            # Speaker turns + speaking time (md/txt/GUI)
    ├── markdown_export.py  # Markdown exporter
    ├── text_export.py      # Plain-text exporter
    ├── srt_export.py       # SRT subtitles
    └── json_export.py      # JSON exporter

src/diarrhizer_gui/
├── app.py                  # Entry point (python -m diarrhizer_gui.app)
├── main_window.py          # Navigation + job queue execution
├── job_queue.py            # Queue bookkeeping (pure Python)
├── pipeline_worker.py      # One job in a child process -> Qt signals
├── child_process.py        # Spawned process watched by a QTimer
├── llm_worker.py           # One model call / account action in a child process -> Qt signals
├── child_tasks.py          # Child entry points (pipeline, doctor, summary, provider check, account) - Qt-free
├── provider_text.py        # Russian wording of accounts, usage, cost and limits
├── environment.py          # Background doctor checks shared by the screens
├── job_scan.py             # History scan, resume / restart-stage / re-diarize plans
└── screens/                # New Job, Monitor, History (+ restart_dialog), Result, Models, Doctor, Settings,
                            # Summarization (+ account_panel, model_combo, summary_dialog)
```
