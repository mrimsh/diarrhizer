"""Local Hugging Face Hub model cache inspection and management.

Covers the models Diarrhizer downloads lazily via WhisperX/pyannote (ASR,
alignment, diarization) - all of which land in the standard HF Hub cache
(see adapters/whisperx.py). This module lets a caller (CLI or future GUI)
inventory that cache and pre-trigger downloads instead of discovering them
as a stall on first pipeline run.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import CacheNotFound, scan_cache_dir


# [SEMANTIC-BEGIN] DIAGNOSTICS:MODEL_CACHE
# @purpose: Inspect and manage the local Hugging Face Hub cache used by ASR/alignment/diarization models
# @description: Wraps huggingface_hub.scan_cache_dir()/delete_revisions() so callers never touch the
#   cache's blob/symlink layout directly. Scanning functions (list_cached_models, is_model_cached) stay
#   import-light - no torch/whisperx pulled in at module level, mirroring how cli.py defers those imports
#   so `doctor` stays usable on a broken install. warm_up_* functions defer their whisperx import to call
#   time for the same reason, but do end up loading torch/whisperx/pyannote once invoked.
# @inputs: model repo ids, optional cache_dir override (defaults to HF_HUB_CACHE/HF_HOME like huggingface_hub)
# @outputs: CachedModelInfo list, bool, bytes freed by clear_cache
# @sideEffects: filesystem reads (scan), filesystem deletes (clear_cache), network + model downloads (warm_up_*)
# @errors: a missing cache directory is not an error - scanning functions treat huggingface_hub.CacheNotFound as "no cache yet" and return empty/False
# @see: DIAGNOSTICS:DOCTOR, ADAPTER:WHISPERX_ASR, ADAPTER:WHISPERX_DIARIZE

CacheDirArg = Optional[Union[str, Path]]


def _resolve_cache_dir(cache_dir: CacheDirArg = None) -> Path:
    """Resolve the HF Hub cache directory, honoring HF_HUB_CACHE/HF_HOME.

    Reads environment variables at call time rather than relying on
    huggingface_hub's own module-level constants, which are computed once at
    import time and won't pick up an HF_HOME set afterwards (e.g. by a test
    via monkeypatch, or by a GUI reconfiguring the cache location at runtime).
    Mirrors huggingface_hub's own precedence: an explicit cache_dir wins,
    then HF_HUB_CACHE/HUGGINGFACE_HUB_CACHE, then HF_HOME/hub, then the
    platform default.
    """
    if cache_dir is not None:
        return Path(cache_dir)

    hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(os.path.expandvars(os.path.expanduser(hub_cache)))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(os.path.expandvars(os.path.expanduser(hf_home))) / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class CachedModelInfo:
    """One model repo entry found in the local HF Hub cache."""

    repo_id: str
    size_on_disk: int
    last_used: datetime


def list_cached_models(cache_dir: CacheDirArg = None) -> List[CachedModelInfo]:
    """List models present in the local HF Hub cache.

    Args:
        cache_dir: Cache directory to scan. Defaults to the resolved
            HF_HUB_CACHE/HF_HOME location (see _resolve_cache_dir).

    Returns:
        CachedModelInfo entries sorted by repo_id, or an empty list if the
        cache directory doesn't exist yet (nothing downloaded so far).
    """
    resolved = _resolve_cache_dir(cache_dir)
    try:
        cache_info = scan_cache_dir(resolved)
    except CacheNotFound:
        return []

    return sorted(
        (
            CachedModelInfo(
                repo_id=repo.repo_id,
                size_on_disk=repo.size_on_disk,
                last_used=datetime.fromtimestamp(repo.last_accessed),
            )
            for repo in cache_info.repos
            if repo.repo_type == "model"
        ),
        key=lambda m: m.repo_id,
    )


def is_model_cached(model_id: str, cache_dir: CacheDirArg = None) -> bool:
    """Check whether a specific model repo is already in the local HF Hub cache."""
    return any(m.repo_id == model_id for m in list_cached_models(cache_dir=cache_dir))


def _require_cuda_if_requested(device: str) -> None:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available. "
            "Please install PyTorch with CUDA support or use device='cpu'"
        )


def warm_up_asr_model(
    model: str,
    device: str = "cpu",
    compute_type: Optional[str] = None,
) -> None:
    """Pre-download and load a WhisperX ASR model so later pipeline runs don't stall on it.

    This is a thin call to whisperx.load_model() - it exists so a future GUI
    can trigger the (possibly multi-GB) model download explicitly, instead of
    it happening silently the first time a job runs.

    Progress reporting limitation: WhisperX's loader (faster_whisper's
    download_model()) calls huggingface_hub.snapshot_download() internally
    without exposing its `tqdm_class` parameter, and neither whisperx.load_model
    nor faster_whisper.utils.download_model accept a progress callback of their
    own. There is therefore no clean way to hook per-file download progress from
    here. huggingface_hub still prints its own tqdm progress bars to stderr
    (unless HF_HUB_DISABLE_PROGRESS_BARS is set) - a caller wanting visible
    progress today has to surface that stream, not a callback from this function.

    Args:
        model: WhisperX model size (tiny, base, small, medium, large) or HF repo id
        device: "cuda" or "cpu"
        compute_type: Compute type (float16, int8_float16, int8). Auto-based on device if None.

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    _require_cuda_if_requested(device)

    import whisperx

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    whisperx.load_model(model, device=device, compute_type=compute_type)


def warm_up_diarization_model(hf_token: str, device: str = "cpu") -> None:
    """Pre-download and load the pyannote diarization pipeline via WhisperX.

    Same rationale and progress-reporting limitation as warm_up_asr_model:
    whisperx.diarize.DiarizationPipeline wraps pyannote.audio.Pipeline.from_pretrained(),
    which does not expose a tqdm_class/progress callback either, so download
    progress can only be observed via huggingface_hub's own stderr tqdm bars.

    Args:
        hf_token: Hugging Face access token with access to the gated pyannote model
        device: "cuda" or "cpu"

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    _require_cuda_if_requested(device)

    from whisperx.diarize import DiarizationPipeline

    DiarizationPipeline(use_auth_token=hf_token, device=device)


def clear_cache(model_id: Optional[str] = None, cache_dir: CacheDirArg = None) -> int:
    """Delete a cached model, or the entire HF Hub cache, via huggingface_hub's cache manager.

    Uses HFCacheInfo.delete_revisions() rather than deleting files directly,
    since the HF cache uses a blob/symlink layout that's easy to corrupt with
    a naive rmtree.

    Args:
        model_id: repo id to delete (e.g. "Systran/faster-whisper-large-v3").
            If None, every cached model is deleted.
        cache_dir: Cache directory to operate on. Defaults like list_cached_models.

    Returns:
        Number of bytes freed. 0 if there was nothing to delete.
    """
    resolved = _resolve_cache_dir(cache_dir)
    try:
        cache_info = scan_cache_dir(resolved)
    except CacheNotFound:
        return 0

    if model_id is None:
        revisions = [rev.commit_hash for repo in cache_info.repos for rev in repo.revisions]
    else:
        revisions = [
            rev.commit_hash
            for repo in cache_info.repos
            if repo.repo_id == model_id
            for rev in repo.revisions
        ]

    if not revisions:
        return 0

    strategy = cache_info.delete_revisions(*revisions)
    freed = strategy.expected_freed_size
    strategy.execute()
    return freed


# [SEMANTIC-END] DIAGNOSTICS:MODEL_CACHE
