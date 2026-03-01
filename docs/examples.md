# Examples

This document provides usage examples for Diarrhizer.

---

## Speaker Name Mapping

Diarization outputs speaker identifiers like `Speaker_00`, `Speaker_01`, etc. You can map these to real names using a JSON file.

### Creating a Speaker Mapping File

Create a JSON file (e.g., `speakers.json`):

```json
{
    "Speaker_00": "Ivan",
    "Speaker_01": "Maria",
    "Speaker_02": "John"
}
```

### Running with Speaker Mapping

```powershell
python -m diarrhizer run "D:\records\meeting.mp4" --out "./out" --speakers "./speakers.json"
```

### Output Examples

#### Markdown (result.md)

Without mapping:
```
[00:00:00 → 00:00:05] **Speaker_00:** Hello everyone
[00:00:05 → 00:00:10] **Speaker_01:** Hi there
```

With mapping:
```
[00:00:00 → 00:00:05] **Ivan:** Hello everyone
[00:00:05 → 00:00:10] **Maria:** Hi there
```

#### JSON (result.json)

Each segment includes both `speaker_id` (original) and `speaker_name` (mapped):

```json
{
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker_id": "Speaker_00",
            "speaker_name": "Ivan",
            "text": "Hello everyone"
        },
        {
            "start": 5.0,
            "end": 10.0,
            "speaker_id": "Speaker_01",
            "speaker_name": "Maria",
            "text": "Hi there"
        }
    ]
}
```

### Notes

- The mapping is applied **at export time only**. Internal diarization IDs remain unchanged.
- If a speaker ID is not in the mapping, the original ID is used as the display name.
- You can re-export with a different mapping without re-running the pipeline (the merged segments are cached).

---

## Basic Processing

Process a recording with default settings:

```powershell
python -m diarrhizer run "D:\records\call.mp3" --out "./out"
```

## Specifying Speaker Count

If you know the approximate number of speakers:

```powershell
python -m diarrhizer run "D:\records\meeting.mp4" --out "./out" --min-speakers 2 --max-speakers 4
```

## Using CPU

If CUDA is not available:

```powershell
python -m diarrhizer run "D:\records\call.mp3" --out "./out" --device cpu
```

## Language Settings

Specify the language for better accuracy:

```powershell
python -m diarrhizer run "D:\records\russian_call.mp3" --out "./out" --lang ru
```

## Force Re-processing

To re-run all stages, overwriting cached outputs:

```powershell
python -m diarrhizer run "D:\records\call.mp3" --out "./out" --force
```

To re-run only a specific stage:

```powershell
python -m diarrhizer run "D:\records\call.mp3" --out "./out" --force-stage export
```
