"""Environment diagnostics module for Diarrhizer."""

import os
import sys
import shutil
from typing import List, Tuple


# [SEMANTIC-BEGIN] DIAGNOSTICS:DOCTOR
# @purpose: Run environment diagnostics to verify Diarrhizer dependencies
# @description: Checks for Python version, FFmpeg, torch/torchaudio, CUDA, and HF token
# @sideEffects: Reads environment variables, imports optional modules
# @errors: Prints warnings for missing dependencies
# @see: CLI:ENTRY
def run_doctor_checks() -> None:
    """Run all diagnostic checks and print results."""
    print("=" * 50)
    print("Diarrhizer Environment Diagnostics")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_ffmpeg,
        check_torch,
        check_cuda,
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
    print("=" * 50)


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
    """Check FFmpeg is available in PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    
    if ffmpeg_path:
        return ("FFmpeg", True, f"Found at: {ffmpeg_path}")
    else:
        return ("FFmpeg", False, "Not found in PATH. Install FFmpeg and add to PATH.")


def check_torch() -> Tuple[str, bool, str]:
    """Check torch and torchaudio are installed."""
    try:
        import torch
        import torchaudio
        torch_version = torch.__version__
        has_cuda = torch.cuda.is_available()
        
        info = f"torch {torch_version}"
        if has_cuda:
            info += " (with CUDA)"
        else:
            info += " (CPU only)"
        
        return ("torch/torchaudio", True, info)
    except ImportError as e:
        return ("torch/torchaudio", False, f"Not installed: {e}")


def check_cuda() -> Tuple[str, bool, str]:
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return ("CUDA", True, f"{device_count} device(s): {device_name}")
        else:
            return ("CUDA", False, "CUDA available but no devices found")
    except ImportError:
        return ("CUDA", False, "torch not installed, cannot check CUDA")


def check_hf_token() -> Tuple[str, bool, str]:
    """Check Hugging Face token is present."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    if token:
        return ("HF Token", True, "Found in environment")
    else:
        return ("HF Token", False, "Not found. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN")


if __name__ == "__main__":
    run_doctor_checks()
# [SEMANTIC-END] DIAGNOSTICS:DOCTOR
