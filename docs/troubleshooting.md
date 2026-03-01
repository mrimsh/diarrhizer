# Troubleshooting Guide

This guide covers common issues you may encounter when running Diarrhizer and how to resolve them.

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
Reinstall torch with the correct CUDA version. Example for CUDA 12.1:

```cmd
pip uninstall torch torchaudio torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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

## Running Diagnostics

To check your environment:

```cmd
python -m diarrhizer doctor
```

This will verify Python version, FFmpeg, torch/torchaudio, CUDA, torchcodec, and HF token.
