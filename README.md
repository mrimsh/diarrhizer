# Diarrhizer

By MRIMSH, L. L. MRIMSH and all other LLMs

A local (on-prem) Windows tool for processing call recordings:
input audio/video file → FFmpeg → WhisperX (ASR + alignment) → speaker diarization (pyannote via WhisperX) → export (`.md/.txt/.srt/.json`).

Two front ends share one pipeline: a desktop GUI (`python -m diarrhizer_gui.app`, see [GUI](#gui))
and a CLI (`python -m diarrhizer run …`).

---

## Current Status

| Component | Status |
|-----------|--------|
| `doctor` command | ✅ Implemented |
| `run` command | ✅ Implemented |
| Pipeline stages | ✅ Implemented |
| Adapters (FFmpeg/WhisperX/diarization) | ✅ Implemented |
| Speaker name mapping | ✅ Implemented |
| Batch processing (several files / folders) | ✅ Implemented |
| Desktop GUI (PySide6) | ✅ Implemented |
| Summaries (prompt templates + model providers) | ✅ Implemented |

---

## Features

* Input: anything FFmpeg decodes — `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`/`.oga`/`.opus`
  (messenger voice notes), `.amr`/`.3gp` (phone call recorders), `.wma`, video (`.mp4`, `.mkv`,
  `.webm`, `.mov`, `.avi`)
* One file, several files, or whole folders per run (processed one after another)
* Audio normalization via FFmpeg (WAV mono 16 kHz) with optional preprocessing profiles
* WhisperX transcription with word-level timestamps (alignment)
* Speaker diarization (`Speaker_00`, `Speaker_01`, …) with text-to-speaker assignment
* Export:
  * Human-readable `.md/.txt`: one line per speaker turn (consecutive phrases of one person are
    joined), plus a speaking-time summary per speaker in `.md`
  * `.srt` subtitles (`Name: text` cues, long phrases split at word boundaries)
  * Structured `.json` (every segment and word with timings, for further processing)
* Caching and resuming: finished stages are never recomputed; an interrupted job continues from
  the first missing stage
* Summaries of a finished job by a language model, from your own prompt templates, through a
  ChatGPT or Claude subscription or any API with a key (see [Summaries](#summaries))

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
- `result.md`/`result.txt`/`result.srt` will show real names instead of Speaker_XX
- `result.json` will include both `speaker_id` (original) and `speaker_name` (mapped)
- Mapping two ids to the same name merges them: diarization sometimes splits one person in two

In the GUI you don't need the file: name the speakers on the result screen (with a "▶" button to
hear each one) and the exports are rewritten immediately.

---

## Requirements

* Windows 10/11
* Python **3.11 or 3.12** (3.12 recommended). **Not 3.13+**: the pinned WhisperX 3.3.1 declares `Requires-Python <3.13` and torch 2.3.1 has no wheels for newer Pythons, so `pip` fails with `No matching distribution found for whisperx==3.3.1`. Check `python --version` — a fresh Windows install often has 3.13 as the default
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
py -0p                      # lists installed Pythons - you need a 3.12 (or 3.11) entry
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version            # must say 3.12.x (or 3.11.x), not 3.13
python -m pip install -U pip
```

Using `.venv` is the recommended way to isolate dependencies. If `py -0p` has no 3.12, install it first
(`winget install Python.Python.3.12`), or let [uv](https://docs.astral.sh/uv/) fetch it:
`uv venv --python 3.12 --seed .venv`. An old `.venv` created with another Python (or whose Python was
uninstalled) must be deleted and recreated - a venv cannot be switched to another Python version.

---

### 2) Install dependencies

> **Important:** On Windows with GPU, the `torch/torchaudio` stack must be strictly aligned by version **and** cuDNN major version. `torch.cuda.is_available() == True` does **not** guarantee that WhisperX / CTranslate2 can actually use the GPU — cuDNN 8 vs 9 DLL mismatch is a silent failure that only appears during transcription.

**CPU option (simpler, recommended first):**

```powershell
pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cpu.txt
pip install -e .[gui]
```

**CUDA option (NVIDIA GPU, CUDA 12.1 cuDNN 8 — stable supported path):**

```powershell
pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cuda-cu121.txt
pip install -e .[gui]
```

(`[gui]` adds PySide6 for the desktop GUI; plain `pip install -e .` is enough for the CLI.)

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

In the GUI, paste it under **Настройки → HF-токен** instead (it is written to the same `.env`).
Without a token you can still use "recognition only" (GUI) / `--to-stage transcribe` (CLI); a run
that needs diarization and has no token fails within seconds, before transcription starts.

---

## GUI

```powershell
python -m diarrhizer_gui.app
```

(or `diarrhizer-gui` after `pip install -e .[gui]`). The interface is in Russian.

* **Новое задание** — add files or whole folders (buttons or drag & drop onto the window), pick the
  mode and parameters, run. Every file becomes its own job; they run one after another. The
  parameters of the last run are remembered.
* **Выполнение** — the running batch: stage progress, live log, queue, **Остановить** (stops at
  once; finished stages are kept). Reachable from the sidebar at any time; the sidebar, window title
  and a taskbar flash show progress and completion.
* **История заданий** — all jobs in the results folder with stage indicators, a filter, and
  **Продолжить** for unfinished jobs: it resumes from the first missing stage with the job's own
  settings (e.g. after a stop or a failed diarization, without re-transcribing).
  **Перезапустить этап** (any job) recomputes the chosen stage from scratch, plus the stages built
  on its output (restarting transcription also redoes merge and export; restarting diarization or
  conversion resets the saved speaker names, which are kept in `speakers.before-rediarize.json`).
  A stage is available once every earlier stage has its output. A dialog lists what will be redone
  (the old results of those stages are deleted) and shows the settings of those stages as the job
  ran with them - ASR model, language, compute type, prompt, speaker range, turn length, audio
  profile and storage format, device - so they can be changed before the restart (e.g. a smaller
  model for transcription). Changed settings are written back to the job, so History and later
  restarts show what the results were really made with.
* **Результат** — the transcript by speaker turns, one colour per speaker; click a timestamp to play
  the recording from there, **▶** next to a speaker plays a sample of their voice; name speakers
  and save (exports are rewritten immediately); search (Ctrl+F); copy the whole transcript;
  **Пересчитать спикеров** re-runs diarization with another speaker count (or adds speakers to a
  recognition-only job) while keeping the transcript; **Саммаризация…** makes a summary of the
  transcript (see below).
* **Саммаризация** — prompt templates and model providers for summaries (see
  [Summaries](#summaries)).
* **Модели**, **Доктор**, **Настройки** — model cache, environment checks (run in the background;
  the sidebar shows a ⚠ badge if something required is missing), defaults, HF token, FFmpeg path,
  audio storage profiles.

Processing runs in a separate process: the window stays responsive, a crash inside CUDA/cuDNN
cannot take the GUI down, and memory (including VRAM) is fully released after every job.

---

## Quick Start

### Environment Check

```powershell
python -m diarrhizer doctor
```

The `doctor` command performs these diagnostic checks:

1. **Python version** — verifies Python 3.11 or 3.12 (3.13+ cannot install the pinned WhisperX/torch)
2. **FFmpeg** — checks availability in PATH (or `DIARRHIZER_FFMPEG_PATH`)
3. **PyTorch/Torchaudio** — verifies installation and reports version
4. **CUDA** — checks GPU availability
5. **cuDNN** — checks cuDNN 8 DLLs needed by WhisperX/CTranslate2 (skipped on CPU-only)
6. **torchcodec** — optional fast decoder (reported as `WARN`, does not fail the run)
7. **Critical imports** — whisperx / speechbrain / pyannote / transformers
8. **Hugging Face token** — verifies `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` is set (including from a local `.env`)
9. **Summarization** — optional (a problem is a `WARN`): the template/provider files parse, every
   configured CLI provider is on `PATH`, every API provider's key is set

---

### Run Processing

Basic usage:

```powershell
python -m diarrhizer run "D:\records\meeting.mp4" --out ".\out" --min-speakers 2 --max-speakers 6 --lang ru --device cuda
```

Several files, or every recording in a folder (one job per file, run one after another; a failed
file is reported and the rest continue, the exit code is 1 if any failed):

```powershell
python -m diarrhizer run "D:\records\call1.m4a" "D:\records\call2.amr" --lang ru
python -m diarrhizer run "D:\records\2026-10" --out ".\out" --lang ru
```

With speaker name mapping:

```powershell
python -m diarrhizer run "D:\records\meeting.mp4" --out ".\out" --speakers ".\speakers.json"
```

All options:

| Option | Description | Default |
|--------|-------------|---------|
| `input` | Media file(s) and/or folder(s); a folder adds the media files directly inside it | (required, unless `--job-dir` resumes a job that already recorded it; `--job-dir` takes at most one) |
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

### Summaries

A finished job's transcript (`export/result.md`, with the speaker names as saved) can be summarized by
a language model. A summary is an on-demand add-on, not a pipeline stage: you ask for it, as often as
you like and with different templates, and it never touches the job's cached stages. Each one is stored
in the job folder as `summary/<timestamp>_<template>.md` (the model's reply, verbatim) plus a `.json`
next to it with what produced it (template, prompt, provider, model, duration).

**Templates.** A template is a named prompt; the transcript is attached to it automatically (inside
`<transcript>` tags) when the request is sent, so the prompt only says what to do. Three built-in
templates (short summary, meeting minutes, call review) are read-only; yours live in
`summary_templates.json`.

**Providers.** Where the request goes. Add as many as you like and mark one as the default:

| Kind | How it works | You need |
|------|--------------|----------|
| ChatGPT subscription | runs the official Codex CLI (`codex exec`) | `codex login` once, in a terminal |
| Claude subscription | runs Claude Code (`claude -p`, tools off) | `claude auth login` once, in a terminal |
| OpenAI-compatible API | `POST <endpoint>/chat/completions` — OpenAI, OpenRouter, DeepSeek, Groq, and local Ollama, LM Studio, vLLM | endpoint, model, API key (not needed for a local server) |
| Anthropic API | `POST <endpoint>/v1/messages` | model, API key |

* Subscription providers use the sign-in you already made in those tools: Diarrhizer never sees or stores
  the account password or token. The tool is started in an empty temporary folder, without your personal
  tool setup (hooks, plugins, MCP servers; for Codex also its `config.toml`, whose model or reasoning
  effort may not suit `exec`) and without `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `CODEX_API_KEY` in its
  environment, so a key kept for an API provider cannot silently switch the billing from the subscription.
  A model and extra command-line arguments (e.g. `-c model_reasoning_effort="low"` for Codex) can be set
  per provider.
* An API key never goes into `summary_providers.json`: the provider names an environment variable
  (`api_key_env`) and the key lives in `.env` (git-ignored) — the GUI writes it there for you, like the
  Hugging Face token.
* Providers are stored in `summary_providers.json` — see `summary_providers.example.json`.
  `DIARRHIZER_SUMMARY_PROVIDERS_FILE` / `DIARRHIZER_SUMMARY_TEMPLATES_FILE` move the two files.

**GUI.** The **Саммаризация** screen has two tabs, "Шаблоны промптов" and "Провайдеры" (**Проверить**
sends a one-word test request with the form's current values). On a job's **Результат** screen,
**Саммаризация…** opens a dialog: pick a template and a provider, **Создать саммари**; the dialog says
exactly where the text will be sent, shows the summary as Markdown, keeps the job's earlier summaries and
can copy or open them. The request runs in a separate process and can be cancelled.

**CLI.**

```powershell
python -m diarrhizer summarize --list
python -m diarrhizer summarize ".\out\meeting_20260101_120000" --template "Протокол встречи" --provider ChatGPT --print
```

`--template` / `--provider` default to the ones marked as default; the argument may also be the job's
`export/result.md`.

Things to know:

* The transcript text leaves your machine (to OpenAI, Anthropic or the endpoint you configured) — the
  tool is otherwise fully local. A local OpenAI-compatible server (Ollama, LM Studio) keeps it on-prem.
* The whole transcript goes in one request. A very long recording needs a model with a large enough
  context window; when it is too small, the provider's own error is shown. A reply cut off at the
  provider's length limit is flagged (raise `max_tokens` for API providers).
* A subscription request counts against that subscription's usage limits.

---

## Pipeline

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
    -> artifacts/export/result.md + result.txt + result.srt + result.json
```

---

## Output Structure

Each run creates a job-specific folder inside `--out`, containing artifacts by stage:

* `audio/` — normalized WAV and/or its archive copy (see [Audio Storage Formats](#audio-storage-formats-disk-space-vs-quality))
* `asr/` — WhisperX transcript (timestamps/words)
* `diar/` — diarization result
* `merged/` — merged segments (text + speaker)
* `export/` — final `.md`, `.txt`, `.srt`, `.json`
* `meta/run.json` — the input path and the job's full settings (used to resume it)
* `speakers.json` — speaker names saved from the GUI
* `summary/` — summaries made on request (see [Summaries](#summaries)): `<timestamp>_<template>.md` + `.json`

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

* Tests: `python -m pytest tests -q` (GUI tests run offscreen and never touch your saved settings)
* CLI entry point: [`src/diarrhizer/cli.py`](src/diarrhizer/cli.py)
* GUI: [`src/diarrhizer_gui/`](src/diarrhizer_gui/) — screens in `screens/`, the job queue and child-process runner in `main_window.py`, `job_queue.py`, `pipeline_worker.py`, `child_process.py`
* Pipeline runner: [`src/diarrhizer/pipeline/runner.py`](src/diarrhizer/pipeline/runner.py)
* Pipeline stages: [`src/diarrhizer/pipeline/stages/`](src/diarrhizer/pipeline/stages/)
* External integrations (adapters): [`src/diarrhizer/adapters/`](src/diarrhizer/adapters/) — including `llm.py`, the summary providers
* Summaries: [`src/diarrhizer/summary/`](src/diarrhizer/summary/) — templates, providers, and the service that ties a job to them
* Diagnostics: [`src/diarrhizer/diagnostics/doctor.py`](src/diarrhizer/diagnostics/doctor.py)

---

## Roadmap

1. Implement pipeline stages (convert, transcribe, diarize, merge, export)
2. Implement adapters (FFmpeg, WhisperX, diarization)
3. Stable CLI + caching + environment diagnostics
4. UX improvements: speaker name mapping, protocol formatting
5. GUI + additional features (batch queue, resume, playback, speaker naming)
6. Summaries from prompt templates (subscription CLIs and API providers)
7. Next ideas: VTT/DOCX export, per-turn text editing in the GUI, splitting very long transcripts for small-context models

---

## Author

By MRIMSH, L. L. MRIMSH and all other LLMs

## License

Personal project.
