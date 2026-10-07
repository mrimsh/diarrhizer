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

Consecutive phrases of one speaker are joined into one turn. If diarization split one person
into two ids, map both to the same name and their phrases merge too:

```json
{"Speaker_00": "Ivan", "Speaker_02": "Ivan", "Speaker_01": "Maria"}
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

## Batch Processing

Several files, or a whole folder of recordings — one job per file, processed one after another:

```powershell
python -m diarrhizer run "D:\records\call1.m4a" "D:\records\call2.amr" --out "./out" --lang ru
python -m diarrhizer run "D:\records\2026-10" --out "./out" --lang ru --min-speakers 2 --max-speakers 2
```

A failed file is reported and the batch moves on; a summary is printed at the end and the exit
code is 1 if any file failed. In the GUI, add several files (or drop a folder onto the window) on
"Новое задание".

## Transcribe Now, Diarize Later

Without an HF token, stop after transcription; once you have the token, continue the same job:

```powershell
python -m diarrhizer run "D:\records\call.mp3" --out "./out" --to-stage transcribe
python -m diarrhizer run --job-dir ".\out\call_20261001_101500" --from-stage diarize
```

In the GUI: run in "Только распознавание" mode, then "Разделить по спикерам…" on the result screen.

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
