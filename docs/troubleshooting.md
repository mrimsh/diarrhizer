# Troubleshooting Guide

This guide covers common issues you may encounter when running Diarrhizer and how to resolve them.

---

## Quick Diagnostic

Always start by running the diagnostics:

```cmd
python -m diarrhizer doctor
```

This will verify Python version, FFmpeg, torch/torchaudio, CUDA, torchcodec, critical imports, and HF token.

---

## Dependency Compatibility Issues

### Symptom: LazyModule / SpeechBrain Import Errors

**Error message:**
```
Transcription failed: Failed to load WhisperX model 'base':
Lazy import of LazyModule(package=None, target=speechbrain.inference, loaded=False) failed
```

Or:
```
Model was trained with torch 1.10.0+cu102, yours is 2.10.0+cu128. Bad things might happen...
Model was trained with pyannote.audio 0.0.1, yours is 3.4.0. Bad things might happen...
```

**Root Cause:** The WhisperX/pyannote/speechbrain stack has complex dependency requirements. Using the latest versions (especially torch 2.10+, numpy 2.x, transformers 5.x) can cause compatibility issues.

**Solution:**

1. **Use the stable constraints file:**
   ```cmd
   pip install -c requirements/constraints-stable.txt -r requirements/base.txt
   ```

2. **If issues persist, recreate your environment:**
   ```cmd
   # Deactivate and remove existing venv
   deactivate
   rmdir /s .venv
   
   # Create fresh environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -U pip
   pip install -c requirements/constraints-stable.txt -r requirements/base.txt
   pip install -e .
   ```

**Key constraints in `requirements/constraints-stable.txt`:**
- `numpy<2.0` - Many ML packages still adapting to NumPy 2.0
- `huggingface-hub<1.0` - Avoids breaking changes in model loading
- `transformers<5.0` - API changes in 5.x break some integrations
- `torch>=2.0.0,<2.5.0` - More stable with WhisperX models

---

## FFmpeg Missing

**Symptom:** `check_ffmpeg` fails in `doctor` output.

**Solution:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Add FFmpeg `bin` directory to your system PATH
3. Restart your terminal/IDE

**Verify:** Run `ffmpeg -version` in a new terminal.

---

## HF Token Missing / Gated Model Access

**Symptom:** `check_hf_token` fails, or you get authentication errors when loading diarization models.

**Solution:**
1. Get a Hugging Face token from https://huggingface.co/settings/tokens
2. Set the environment variable:
   ```cmd
   set HF_TOKEN=your_token_here
   ```
   Or add to `.env` file in project root:
   ```
   HF_TOKEN=your_token_here
   ```

**Note:** The token needs access to `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.1` models.

---

## torch Stack Mismatch (CPU vs CUDA)

**Symptom:** 
- `doctor` shows torch with "(CPU only)" but you have a GPU
- CUDA available but no devices found
- Slow processing despite having GPU

**Solution:**
Reinstall torch with the correct CUDA version. Example for CUDA 12.8:

```cmd
pip uninstall torch torchaudio torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only (no GPU):
```cmd
pip install torch torchaudio torchvision
```

**Verify:** Run `python -c "import torch; print(torch.cuda.is_available())"`

---

## torchcodec / FFmpeg Decoding Issues

**Symptom:** 
- `doctor` shows torchcodec as "Not installed"
- Audio decoding errors or warnings
- Slow waveform loading

**Solution (optional):**
torchcodec provides fast audio decoding. Install it with:

```cmd
pip install torchcodec
```

**Fallback:** If torchcodec fails to install, Diarrhizer will automatically use waveform preload (slower but functional). No action needed.

**Note:** torchcodec requires FFmpeg libraries. On Windows, you may need Visual C++ redistributables.

---

## Word Alignment Failures

**Symptom:**
```
WARNING:root:Word alignment failed: load_align_model() got an unexpected keyword argument 'language'
```

**Solution:** This is usually caused by WhisperX API changes. The fix has been applied in Diarrhizer v0.1.1+. If you see this error, ensure you're using the latest version:
```cmd
pip install -U diarrhizer
```

---

## Compute Type Errors

**Symptom:**
```
ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation.
```

**Solution:** This happens when using CPU with float16 compute type. Ensure you're using the correct device flag:
- For GPU: `--device cuda` (uses float16)
- For CPU: `--device cpu` (uses int8)

---

## Known Issues with Version Combinations

| Version | Status | Notes |
|---------|--------|-------|
| torch 2.10 + whisperx 3.7 | ⚠️ Works with warnings | Shows model version mismatch warnings |
| numpy 2.x | ⚠️ May work | Some packages still adapting |
| transformers 5.x | ⚠️ May work | API changes in 5.x |
| huggingface-hub 1.5+ | ⚠️ May work | Lazy loading changes |
| pyannote.audio 3.4 | ✅ Works | Current default |
| speechbrain 1.0.3 | ✅ Works | Current default |
