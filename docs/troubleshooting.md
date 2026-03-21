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

## `pkg_resources` Missing (setuptools Compatibility)

**Symptom:**
```
Import error: No module named 'pkg_resources'
```
or WhisperX fails at startup with a `ModuleNotFoundError` pointing to `pkg_resources`.

**Root Cause:** WhisperX 3.3.1 uses the `pkg_resources` module from `setuptools` at runtime. On Python 3.12+, a fresh virtual environment may not include `setuptools`, or a newer `setuptools` (81+) may have removed `pkg_resources` in favor of `importlib.metadata`.

**Fix:**

1. **Reinstall using the constraints file** (recommended — the pin is already included):
   ```powershell
   pip install -c requirements/constraints-stable.txt -r requirements/base.txt
   ```

2. **Or install setuptools directly:**
   ```powershell
   pip install "setuptools<81"
   ```

**Why `setuptools<81`?** Versions 81+ dropped `pkg_resources`. WhisperX 3.3.1 still depends on it, so a compatible `setuptools` version must be present in the environment. The constraint file pins `setuptools<81` to ensure this.

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
- `setuptools<81` - WhisperX needs `pkg_resources`, removed in setuptools 81+
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
Reinstall torch with the correct CUDA version. The stable supported path uses CUDA 12.1 (cuDNN 8):

```powershell
pip uninstall torch torchaudio
pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cuda-cu121.txt
```

For CPU-only (no GPU):
```powershell
pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cpu.txt
```

**Verify:** Run `python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"`

**Important:** `torch.cuda.is_available() == True` does **not** guarantee that WhisperX / CTranslate2 can use the GPU. The cuDNN DLL mismatch is a separate, silent failure. See the next section.

---

## cuDNN DLL Mismatch (CTranslate2 vs Torch CUDA)

This is the most common GPU failure on Windows with newer PyTorch.

### Symptom

During the **transcribe** stage (WhisperX / faster-whisper / CTranslate2):

```
Could not locate cudnn_ops_infer64_8.dll. Please make sure it is in your library path!
```

Or during **diarization**:

```
Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}
Invalid handle. Cannot load symbol cudnnCreateConvolutionDescriptor
```

The `doctor` command may pass all checks (torch imports fine, CUDA reports available) — the error only occurs when WhisperX actually calls CTranslate2.

### Root Cause

**cuDNN version conflict** between PyTorch and CTranslate2:

| Package | cuDNN version bundled |
|---------|----------------------|
| `torch>=2.4.0+cu124` | cuDNN **9** |
| `torch<2.4.0` or `torch+cu121` | cuDNN **8** |
| `ctranslate2<4.5.0` (WhisperX 3.3.1) | expects cuDNN **8** |
| `ctranslate2>=4.5.0` | expects cuDNN **9** |

**WhisperX 3.3.1 explicitly requires `ctranslate2<4.5.0`**, which needs cuDNN 8.
But `torch>=2.4.0+cu124` bundles cuDNN 9 DLLs into `site-packages\torch\lib\`.
CTranslate2 4.4.0 searches for `cudnn_ops_infer64_8.dll` (cuDNN 8), doesn't find it, and crashes.

This is why `torch.cuda.is_available() == True` is not sufficient — torch itself works fine with cuDNN 9, but CTranslate2 doesn't.

### Verification Commands

Run these to diagnose:

```powershell
# 1. Check torch version and CUDA build
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available())"

# 2. Check all relevant package versions
pip show whisperx faster-whisper ctranslate2 pyannote.audio torch torchaudio

# 3. Check which cuDNN DLLs are actually in the torch package
dir .venv\Lib\site-packages\torch\lib\cudnn*.dll
```

**What to look for:**
- If step 3 shows `cudnn_ops_infer64_8.dll` → cuDNN 8, should work
- If step 3 shows only `cudnn*64_9.dll` files → cuDNN 9, **will fail** with CTranslate2 4.4.0
- If `ctranslate2` shows `4.4.0` and `torch` shows `2.4.x+cu124` → **mismatch confirmed**

### Fix Path

**Option A — Use the project's pinned setup (recommended):**

Recreate your venv with the correct pins:

```powershell
deactivate
rmdir /s .venv

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

# For GPU (CUDA 12.1, cuDNN 8):
pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cuda-cu121.txt
pip install -e .

# Or for CPU:
pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cpu.txt
pip install -e .
```

**Option B — Manual fix (if you can't recreate the venv):**

```powershell
pip uninstall torch torchaudio
pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**Option C — Use CPU for verification:**

If you need to confirm the pipeline works before fixing the GPU stack:

```powershell
python -m diarrhizer run "path\to\file.mp4" --out ".\out" --device cpu
```

### Why You Must Recreate the Venv

Pip's dependency resolver does not retroactively downgrade packages when you change constraint files. If torch 2.4.1+cu124 is already installed, `pip install torch==2.3.1+cu121` may leave fragments. A clean venv ensures all DLLs are consistent.

### Compatibility Matrix

| torch | CUDA | cuDNN | ctranslate2 | Status |
|-------|------|-------|-------------|--------|
| 2.3.1+cu121 | 12.1 | 8 | 4.4.0 | **Stable (recommended)** |
| 2.4.1+cu124 | 12.4 | 9 | 4.4.0 | Broken — cuDNN mismatch |
| 2.4.1+cu124 | 12.4 | 9 | >=4.5.0 | Would work but WhisperX 3.3.1 forbids it |
| 2.3.1+cpu | — | — | 4.4.0 | Stable (CPU only) |

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
| torch 2.3.1+cu121 + ctranslate2 4.4.0 | **Stable** | Recommended GPU path (cuDNN 8) |
| torch 2.4.1+cu124 + ctranslate2 4.4.0 | **Broken** | cuDNN 9 vs 8 mismatch → `cudnn_ops_infer64_8.dll` error |
| torch 2.4.x+cu124 + ctranslate2 >=4.5.0 | Would work | Blocked by WhisperX 3.3.1 `ctranslate2<4.5.0` |
| torch 2.10 + whisperx 3.7 | Works with warnings | Shows model version mismatch warnings |
| numpy 2.x | May work | Some packages still adapting |
| transformers 5.x | May work | API changes in 5.x |
| huggingface-hub 1.5+ | May work | Lazy loading changes |
| pyannote.audio 3.3.2 | **Stable** | Current default |
| speechbrain 1.0.x | **Stable** | Current default |
