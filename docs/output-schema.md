# Output Schema

This document describes the JSON schema for exported results from the Diarrhizer pipeline.

---

## Overview

The export stage produces two output files:
- `export/result.md` — Human-readable Markdown transcript
- `export/result.json` — Structured machine-readable JSON

Both exports contain the same underlying data: merged segments with speaker labels and timestamps.

---

## JSON Schema (`result.json`)

### Top-Level Structure

```json
{
  "metadata": {
    "input_file": "string",
    "created_at": "ISO 8601 datetime string",
    "config": {
      "language": "string",
      "device": "string",
      "min_speakers": "integer",
      "max_speakers": "integer"
    }
  },
  "segments": [
    {
      "start": "float (seconds)",
      "end": "float (seconds)",
      "speaker_id": "string",
      "text": "string",
      "words": [
        {
          "start": "float (seconds)",
          "end": "float (seconds)",
          "word": "string",
          "speaker_id": "string"
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
| `text` | string | Transcribed text for the segment |
| `words` | array (optional) | Word-level details if available |

#### `words` array (optional)

Each word object contains:

| Field | Type | Description |
|-------|------|-------------|
| `start` | float | Word start time in seconds |
| `end` | float | Word end time in seconds |
| `word` | string | The transcribed word |
| `speaker_id` | string | Speaker identifier for this word |

---

## Example

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
    }
  },
  "segments": [
    {
      "start": 0.0,
      "end": 5.5,
      "speaker_id": "Speaker_00",
      "text": "Hello, how are you today?",
      "words": [
        {"start": 0.0, "end": 0.8, "word": "Hello,", "speaker_id": "Speaker_00"},
        {"start": 0.8, "end": 1.2, "word": "how", "speaker_id": "Speaker_00"},
        {"start": 1.2, "end": 1.6, "word": "are", "speaker_id": "Speaker_00"},
        {"start": 1.6, "end": 2.2, "word": "you", "speaker_id": "Speaker_00"},
        {"start": 2.2, "end": 3.0, "word": "today?", "speaker_id": "Speaker_00"}
      ]
    },
    {
      "start": 5.5,
      "end": 10.2,
      "speaker_id": "Speaker_01",
      "text": "I'm doing great, thanks for asking.",
      "words": [
        {"start": 5.5, "end": 6.0, "word": "I'm", "speaker_id": "Speaker_01"},
        {"start": 6.0, "end": 6.8, "word": "doing", "speaker_id": "Speaker_01"},
        {"start": 6.8, "end": 7.5, "word": "great,", "speaker_id": "Speaker_01"},
        {"start": 7.5, "end": 8.5, "word": "thanks", "speaker_id": "Speaker_01"},
        {"start": 8.5, "end": 9.2, "word": "for", "speaker_id": "Speaker_01"},
        {"start": 9.2, "end": 10.2, "word": "asking.", "speaker_id": "Speaker_01"}
      ]
    }
  ]
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

[00:00:00 → 00:00:05] **Speaker_00:** Hello, how are you today?
    - [00:00:00] Hello,
    - [00:00:00] how
    - [00:00:01] are
    - [00:00:01] you
    - [00:00:02] today?

[00:00:05 → 00:00:10] **Speaker_01:** I'm doing great, thanks for asking.
    - [00:00:05] I'm
    - [00:00:06] doing
    ...
```

### Format Rules

- Timecodes are in `[HH:MM:SS]` format
- Speaker labels are bolded
- Word-level details are indented with dashes (only shown if word timestamps are available)

---

## Notes

- Speaker identifiers (e.g., `Speaker_00`, `Speaker_01`) are assigned by the diarization model
- Real name mapping is a separate layer on top of this output
- Timestamps are in seconds (float values)
- The `words` array is optional — it is only present when WhisperX provides word-level alignment data

---

## Related Documentation

- [Architecture](architecture.md) — Pipeline overview
- [Merge Stage](../src/diarrhizer/pipeline/stages/merge.py) — How segments are merged
