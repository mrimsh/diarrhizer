# Output Schema

This document describes the JSON schema for exported results from the Diarrhizer pipeline.

---

## Overview

The export stage produces four output files:
- `export/result.md` — Human-readable Markdown transcript: a speaking-time summary, then one line per speaker turn
- `export/result.txt` — The same turns without markup or metadata header
- `export/result.srt` — Subtitles, one cue per segment (long segments split), `Name: text`
- `export/result.json` — Structured machine-readable JSON: every segment and word with timings

All of them are rendered from the same merged segments. `result.json` keeps them as they are;
`result.md`/`result.txt` join consecutive segments of the same speaker into *turns* (see below).
New exports also publish `export/manifest.json` with hashes after all four files are written.
See [artifact contracts and recovery](artifact-contracts.md) for pending journals and legacy compatibility.

---

## JSON Schema (`result.json`)

### Top-Level Structure

```json
{
  "schema_version": 1,
  "metadata": {
    "input_file": "string",
    "created_at": "ISO 8601 datetime string",
    "config": {
      "language": "string",
      "device": "string",
      "min_speakers": "integer",
      "max_speakers": "integer"
    },
    "speakers": {"Speaker_00": "display name"},
    "provenance": {"python": "version", "packages": {"whisperx": "version"}},
    "source_sha256": "SHA-256 of merged JSON text",
    "warnings": []
  },
  "segments": [
    {
      "start": "float (seconds)",
      "end": "float (seconds)",
      "speaker_id": "string",
      "speaker_name": "string",
      "text": "string",
      "words": [
        {
          "start": "float (seconds) or null",
          "end": "float (seconds) or null",
          "word": "string",
          "speaker_id": "string",
          "speaker_name": "string"
        }
      ]
    }
  ]
}
```

---

## Field Descriptions

### `metadata`

| Field | Type | Description |
|-------|------|-------------|
| `input_file` | string | Path to the original input media file |
| `created_at` | string | ISO 8601 timestamp when export was generated |
| `config` | object | Summary of pipeline configuration used |
| `speakers` | object | Normalized speaker ID → display name mapping; empty when unnamed |
| `provenance` | object | Python and installed package versions; no model revision guarantee |
| `source_sha256` | string | SHA-256 of the UTF-8 merged source JSON text |
| `warnings` | array of strings | Application warnings from ASR and merge, including alignment fallback |

#### `config` object

| Field | Type | Description |
|-------|------|-------------|
| `language` | string | Language code or "auto" for detection |
| `device` | string | Device used ("cuda" or "cpu") |
| `min_speakers` | integer | Minimum number of speakers configured |
| `max_speakers` | integer | Maximum number of speakers configured |

### `segments`

Array of transcribed segments, each containing:

| Field | Type | Description |
|-------|------|-------------|
| `start` | float | Segment start time in seconds |
| `end` | float | Segment end time in seconds |
| `speaker_id` | string | Speaker identifier (e.g., "Speaker_00") |
| `speaker_name` | string | Resolved display name, or the speaker ID if unnamed |
| `text` | string | Transcribed text for the segment |
| `words` | array (optional) | Word-level details if available |
| `alignment_warning` | string (optional) | Why the segment was retained unsplit to preserve text |

#### `words` array (optional)

Each word object contains:

| Field | Type | Description |
|-------|------|-------------|
| `start` | float or null | Word start time in seconds; null means alignment is unavailable |
| `end` | float or null | Word end time in seconds; null means alignment is unavailable |
| `word` | string | The transcribed word |
| `speaker_id` | string | Speaker identifier for this word |
| `speaker_name` | string | Resolved display name, or the speaker ID if unnamed |

The segment's `text` is authoritative. An incomplete word list is diagnostic data, not a replacement
for that text. Unknown times are never fabricated as zero. In this case exports retain the full
segment and SRT uses its segment timestamps. `segment_index` is an ownership field in the intermediate
ASR word records; it is not required in the exported words after merging/splitting.

---

## Example

Illustrative source hash (zeros); real exports compute it from merged JSON.

```json
{
  "metadata": {
    "input_file": "C:/recordings/call_20240115.wav",
    "created_at": "2024-01-15T14:30:00.123456",
    "config": {
      "language": "en",
      "device": "cuda",
      "min_speakers": 2,
      "max_speakers": 6
    },
    "speakers": {
      "Speaker_00": "Alice",
      "Speaker_01": "Bob"
    },
    "provenance": {
      "python": "3.12.13",
      "packages": {
        "whisperx": "3.3.1"
      }
    },
    "source_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "warnings": []
  },
  "segments": [
    {
      "start": 0.0,
      "end": 5.5,
      "speaker_id": "Speaker_00",
      "text": "Hello, how are you today?",
      "words": [
        {
          "start": 0.0,
          "end": 0.8,
          "word": "Hello,",
          "speaker_id": "Speaker_00",
          "speaker_name": "Alice"
        },
        {
          "start": 0.8,
          "end": 1.2,
          "word": "how",
          "speaker_id": "Speaker_00",
          "speaker_name": "Alice"
        },
        {
          "start": 1.2,
          "end": 1.6,
          "word": "are",
          "speaker_id": "Speaker_00",
          "speaker_name": "Alice"
        },
        {
          "start": 1.6,
          "end": 2.2,
          "word": "you",
          "speaker_id": "Speaker_00",
          "speaker_name": "Alice"
        },
        {
          "start": 2.2,
          "end": 3.0,
          "word": "today?",
          "speaker_id": "Speaker_00",
          "speaker_name": "Alice"
        }
      ],
      "speaker_name": "Alice"
    },
    {
      "start": 5.5,
      "end": 10.2,
      "speaker_id": "Speaker_01",
      "text": "I'm doing great, thanks for asking.",
      "words": [
        {
          "start": 5.5,
          "end": 6.0,
          "word": "I'm",
          "speaker_id": "Speaker_01",
          "speaker_name": "Bob"
        },
        {
          "start": 6.0,
          "end": 6.8,
          "word": "doing",
          "speaker_id": "Speaker_01",
          "speaker_name": "Bob"
        },
        {
          "start": 6.8,
          "end": 7.5,
          "word": "great,",
          "speaker_id": "Speaker_01",
          "speaker_name": "Bob"
        },
        {
          "start": 7.5,
          "end": 8.5,
          "word": "thanks",
          "speaker_id": "Speaker_01",
          "speaker_name": "Bob"
        },
        {
          "start": 8.5,
          "end": 9.2,
          "word": "for",
          "speaker_id": "Speaker_01",
          "speaker_name": "Bob"
        },
        {
          "start": 9.2,
          "end": 10.2,
          "word": "asking.",
          "speaker_id": "Speaker_01",
          "speaker_name": "Bob"
        }
      ],
      "speaker_name": "Bob"
    }
  ],
  "schema_version": 1
}
```

---

## Markdown Format (`result.md`)

The Markdown output follows this format:

```markdown
# Transcription

**Input:** C:/recordings/call_20240115.wav
**Generated:** 2024-01-15T14:30:00.123456
**Language:** en
**Device:** cuda

**Speakers:**

- Speaker_01 — 00:00:42 (61%)
- Speaker_00 — 00:00:27 (39%)

[00:00:00 → 00:00:09] **Speaker_00:** Hello, how are you today? Did you get my e-mail?

[00:00:09 → 00:00:15] **Speaker_01:** I'm doing great, thanks for asking.
```

### Format Rules

- Timecodes are in `[HH:MM:SS]` format
- Speaker labels are bolded
- The **Speakers** summary lists each speaker's total speaking time and share, largest first
- One line per **speaker turn**: consecutive segments whose speakers resolve to the same name are
  joined. A new turn starts when the speaker changes, after a pause of 3 s or more, or — past
  ~600 characters — at the next sentence end (so a monologue or a recognition-only job still
  reads as paragraphs). Mapping two ids to the same name merges their turns
- Word-level timings are **not** repeated here; they are available in full (with per-word
  `speaker_id`) in `result.json`

---

## Subtitles (`result.srt`)

```
1
00:00:00,000 --> 00:00:04,200
Ivan: Hello, how are you today?

2
00:00:04,500 --> 00:00:09,000
Ivan: Did you get my e-mail?
```

- One cue per merged segment, prefixed with the speaker's display name
- A segment longer than 7 s or 84 characters is split into several cues at word boundaries
  (preferring sentence ends), using the word timings
- Segments with empty text are skipped

---

## Notes

- Speaker identifiers (e.g., `Speaker_00`, `Speaker_01`) are assigned by the diarization model
- Real name mapping is a separate layer on top of this output
- Timestamps are in seconds (float values)
- Each segment has exactly one speaker: `speaker_id` always matches the `speaker_id`
  of every entry in its `words`. An ASR segment that spans a speaker change is split
  into one segment per speaker at merge time, so segments here can outnumber the
  segments WhisperX produced. `--min-turn-duration` (default `0.4`s) sets how short a
  turn may be before it is folded back into its neighbours as diarization jitter
- The `words` array is optional — it is only present when WhisperX provides word-level alignment data

---

## Related Documentation

The intermediate `asr/transcript.json` has a separate `metadata.alignment_model` object with `name`,
`backend`, `revision` and `loading`. HF loading records `weight_norm_renamed`, `all_keys_matched` and
`renamed_tensors_verified`; torchaudio records its bundle loader without an HF revision. It is `null`
when no model was loaded. `alignment_status: not_needed` denotes empty ASR; `failed` preserves source
text and records the failure in warnings. Cache version `3` rebuilds older ASR when that stage is run.
These fields do not change the exported segment/word shape or schema version `1`.

- [Architecture](architecture.md) — Pipeline overview
- [Merge Stage](../src/diarrhizer/pipeline/stages/merge.py) — How segments are merged
