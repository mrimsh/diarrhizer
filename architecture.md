# Architecture

Goal: a simple, extensible architecture for local call recording processing on Windows:  
**FFmpeg → WhisperX (ASR + alignment) → diarization (pyannote via WhisperX) → protocol and JSON export.**

This document is intentionally concise. It defines the “skeleton” so the project can evolve (caching, new exports, GUI) without rewriting the core.

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

High-level flow:

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
(3) Diarize (WhisperX diarization / pyannote)  
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

Supported inputs: various audio/video formats.  
Normalization via FFmpeg is mandatory.

---

## 3. Key Entities

### Job

“One processing run for one file.”

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

Recommended layout:

```
out/
  <job_id>/
    meta/                 # metadata (config, versions, timestamps)
    audio/
      normalized.wav
    asr/
      transcript.json
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
    

---

## 4. Data Format (Overview)

### ASR output

Contains segments and/or words with timestamps (WhisperX alignment result).

### Diarization output

Contains speech intervals labeled with speaker identifiers (`Speaker_00...`).  
Real names are applied through a separate mapping layer.

### Merged segments

The “stitched” result: text + timestamps + speaker at segment (and/or word) level.

**Important:** In MVP, speaker names are identifiers only.  
Name mapping is a separate layer on top.

---

## 5. Environment Diagnostics (`doctor`)

The Windows dependency stack is sensitive:

- torch / torchaudio / torchvision version alignment (CPU vs CUDA builds),
    
- FFmpeg availability in `PATH`,
    
- Hugging Face token (gated models),
    
- potential torchcodec ↔ FFmpeg / torch compatibility issues.
    

Therefore, a `doctor` command is provided to verify these conditions before running heavy processing.

---

## 6. Extensibility (Future Growth)

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

## 7. Implicit Quality Requirements

- Every stage must log:
    
    - input/output artifact paths,
        
    - key parameters,
        
    - execution duration.
        
- Errors must be actionable and explanatory:
    
    - “FFmpeg not found”
        
    - “HF token missing”
        
    - “CUDA not available”
        
    - dependency conflicts, etc.
        

---

If you'd like, next we can tighten this document slightly for production readiness (e.g., add a small “Non-goals” section or a short “Failure model” section) — without overengineering it.