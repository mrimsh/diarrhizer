"""Environment diagnostics module for Diarrhizer."""

import os
import sys
from pathlib import Path
from typing import List, Tuple

from diarrhizer.adapters.ffmpeg import ENV_FFMPEG_PATH, resolve_ffmpeg_path


# [SEMANTIC-BEGIN] DIAGNOSTICS:DOCTOR
# @purpose: Run environment diagnostics to verify Diarrhizer dependencies
# @description: Checks for Python version, FFmpeg, torch/torchaudio, CUDA, cuDNN, torchcodec, HF token, and critical ML imports
# @sideEffects: Reads environment variables, imports optional modules, scans torch/lib for cuDNN DLLs
# @errors: Prints warnings for missing dependencies; returns False if any check failed
# @see: CLI:ENTRY, ADAPTER:FFMPEG, CONFIG:ENV
def run_doctor_checks() -> bool:
    """Run all diagnostic checks and print results.

    Returns:
        True if every check passed.
    """
    print("=" * 50)
    print("Diarrhizer Environment Diagnostics")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_ffmpeg,
        check_torch,
        check_cuda,
        check_cudnn,
        check_torchcodec,
        check_critical_imports,
        check_hf_token,
    ]
    
    results: List[Tuple[str, bool, str]] = []
    for check in checks:
        name, passed, message = check()
        results.append((name, passed, message))
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if message:
            print(f"       {message}")
    
    print("=" * 50)
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    print(f"Results: {passed_count}/{total_count} checks passed")
    
    # Print additional help for failures
    failed_imports = [name for name, passed, _ in results if not passed and name.startswith("Import:")]
    if failed_imports:
        print("\n" + "=" * 50)
        print("RECOMMENDED ACTION:")
        print("  Rebuild environment with stable constraints:")
        print("    pip install -c requirements/constraints-stable.txt -r requirements/base.txt")
        print("  Or reinstall PyTorch with a compatible version.")
    print("=" * 50)
    return passed_count == total_count


def check_python_version() -> Tuple[str, bool, str]:
    """Check Python version is 3.11+."""
    version = sys.version_info
    required_major = 3
    required_minor = 11
    
    passed = version.major >= required_major and version.minor >= required_minor
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if passed:
        message = f"Python {version_str} (OK)"
    else:
        message = f"Python {version_str} (need 3.11+)"
    
    return ("Python version", passed, message)


def check_ffmpeg() -> Tuple[str, bool, str]:
    """Check FFmpeg is resolvable via DIARRHIZER_FFMPEG_PATH or PATH.

    Mirrors FFmpegAdapter's resolution order (minus the constructor's
    explicit ffmpeg_path argument, which doctor has no way to know about).
    """
    try:
        ffmpeg_path = resolve_ffmpeg_path()
    except FileNotFoundError as e:
        return ("FFmpeg", False, str(e))

    if ffmpeg_path:
        source = f"via {ENV_FFMPEG_PATH}" if os.environ.get(ENV_FFMPEG_PATH) else "via PATH"
        return ("FFmpeg", True, f"Found at: {ffmpeg_path} ({source})")
    else:
        return (
            "FFmpeg",
            False,
            f"Not found. Install FFmpeg and add to PATH, or set {ENV_FFMPEG_PATH}.",
        )


def check_torch() -> Tuple[str, bool, str]:
    """Check torch and torchaudio are installed with version info."""
    try:
        import torch
        import torchaudio
        torch_version = torch.__version__
        torchaudio_version = torchaudio.__version__
        has_cuda = torch.cuda.is_available()
        
        info = f"torch {torch_version}, torchaudio {torchaudio_version}"
        if has_cuda:
            # Check for CPU-only build mismatch
            cuda_version = torch.version.cuda
            info += f" (CUDA {cuda_version})"
            
            # Heuristic: if CUDA available but device count is 0, likely CPU-only build
            if torch.cuda.device_count() == 0:
                info += " - WARNING: CUDA available but no devices. CPU-only build?"
        else:
            info += " (CPU only)"
        
        return ("torch/torchaudio", True, info)
    except ImportError as e:
        return ("torch/torchaudio", False, f"Not installed: {e}")


def check_cuda() -> Tuple[str, bool, str]:
    """Check CUDA availability and device count."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ("CUDA", False, "CUDA not available (torch compiled without CUDA)")
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return ("CUDA", False, "CUDA available but no devices found")
        
        device_name = torch.cuda.get_device_name(0)
        return ("CUDA", True, f"{device_count} device(s): {device_name}")
    except ImportError:
        return ("CUDA", False, "torch not installed, cannot check CUDA")


def check_cudnn() -> Tuple[str, bool, str]:
    """Check that the cuDNN major version matches what WhisperX/CTranslate2 needs.

    WhisperX 3.3.1 pins ctranslate2<4.5.0, which loads cuDNN 8
    (cudnn_ops_infer64_8.dll). torch>=2.4.0+cu124 ships cuDNN 9 only, so
    torch.cuda.is_available() can be True while transcription still crashes.
    """
    try:
        import torch
    except ImportError:
        return ("cuDNN", False, "torch not installed, cannot check cuDNN")

    if not torch.cuda.is_available():
        return ("cuDNN", True, "Skipped (CPU-only torch; cuDNN only needed for CUDA)")

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    cudnn8 = list(torch_lib.glob("cudnn_ops_infer64_8.dll")) + list(
        torch_lib.glob("libcudnn_ops_infer.so.8*")
    )
    cudnn9 = list(torch_lib.glob("cudnn*64_9.dll")) + list(torch_lib.glob("libcudnn*.so.9*"))

    if cudnn8:
        return ("cuDNN", True, f"cuDNN 8 found ({cudnn8[0].name}) — compatible with ctranslate2<4.5")
    if cudnn9:
        return (
            "cuDNN",
            False,
            "cuDNN 9 found, but WhisperX/ctranslate2<4.5 needs cuDNN 8 "
            "(cudnn_ops_infer64_8.dll). See docs/troubleshooting.md",
        )
    return (
        "cuDNN",
        False,
        f"No cuDNN DLLs found in {torch_lib}. GPU transcription will likely fail.",
    )


def check_hf_token() -> Tuple[str, bool, str]:
    """Check Hugging Face token is present."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    if token:
        return ("HF Token", True, "Found in environment")
    else:
        return ("HF Token", False, "Not found. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN")


def check_torchcodec() -> Tuple[str, bool, str]:
    """Check torchcodec is available for audio decoding."""
    try:
        import torchcodec
        return ("torchcodec", True, "Available (fast decoding)")
    except ImportError:
        msg = "Not installed. Will fallback to waveform preload (slower)."
        return ("torchcodec", False, msg)


def check_critical_imports() -> Tuple[str, bool, str]:
    """Check critical ML library imports work correctly.
    
    Tests imports for whisperx, speechbrain, and pyannote.audio.
    These are the most fragile dependencies that commonly cause issues.
    """
    issues = []
    
    # Test whisperx import
    try:
        import whisperx
    except ImportError as e:
        issues.append(f"whisperx: {e}")
    except Exception as e:
        issues.append(f"whisperx: {type(e).__name__}: {e}")
    
    # Test speechbrain import (base package)
    try:
        import speechbrain
    except ImportError as e:
        issues.append(f"speechbrain: {e}")
    
    # Test speechbrain.inference import (critical for model loading)
    try:
        import speechbrain.inference
    except Exception as e:
        issues.append(f"speechbrain.inference: {type(e).__name__}: {e}")
    
    # Test pyannote.audio import
    try:
        import pyannote.audio
    except ImportError as e:
        issues.append(f"pyannote.audio: {e}")
    except Exception as e:
        issues.append(f"pyannote.audio: {type(e).__name__}: {e}")
    
    # Test transformers import
    try:
        import transformers
    except ImportError as e:
        issues.append(f"transformers: {e}")
    
    if issues:
        msg = "; ".join(issues)
        return ("Critical imports", False, msg)
    else:
        return ("Critical imports", True, "All critical imports OK")


if __name__ == "__main__":
    run_doctor_checks()
# [SEMANTIC-END] DIAGNOSTICS:DOCTOR
