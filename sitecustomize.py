"""Site customization for Diarrhizer - applies patches before any imports.

This file is automatically imported by Python at the start of any script.
It applies critical compatibility patches before any other modules load.

This is required for torchaudio 2.10.0+ compatibility with packages like
speechbrain and pyannote.audio that expect older torchaudio APIs.
"""

import sys
import os

# Force pyannote.audio to use soundfile backend instead of torchcodec
# This MUST be set before pyannote.audio is imported
os.environ["PYANNOTE_AUDIO_BACKEND"] = "soundfile"
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"


def _patch_pyannote_audio():
    """Configure pyannote.audio to use soundfile backend instead of torchcodec.
    
    pyannote.audio 3.x prefers torchcodec but it may not be available.
    This patches it to use soundfile/scipy instead.
    """
    # Set environment variable to prefer soundfile backend
    os.environ["PYANNOTE_AUDIO_BACKEND"] = "soundfile"
    os.environ["TORCHAUDIO_BACKEND"] = "soundfile"


def _patch_huggingface_hub():
    """Patch huggingface_hub for compatibility with older parameter names.
    
    Newer versions of huggingface_hub removed use_auth_token in favor of token.
    But pyannote.audio and other libraries still use the old parameter name.
    """
    try:
        import huggingface_hub
    except ImportError:
        return  # Not installed
    
    # Skip if already patched
    if hasattr(huggingface_hub, '_diarrhizer_patched'):
        return
    
    # Save original function
    _original_hf_download = huggingface_hub.hf_hub_download
    _original_hf_hub_download = huggingface_hub.hf_hub_download
    
    def _patched_hf_hub_download(*args, **kwargs):
        # Convert use_auth_token to token if present
        if 'use_auth_token' in kwargs:
            kwargs['token'] = kwargs.pop('use_auth_token')
        return _original_hf_hub_download(*args, **kwargs)
    
    # Also patch snapshot_download and any other functions that might use use_auth_token
    huggingface_hub.hf_hub_download = _patched_hf_hub_download
    
    # Patch snapshot_download too
    if hasattr(huggingface_hub, 'snapshot_download'):
        _original_snapshot = huggingface_hub.snapshot_download
        def _patched_snapshot(*args, **kwargs):
            if 'use_auth_token' in kwargs:
                kwargs['token'] = kwargs.pop('use_auth_token')
            return _original_snapshot(*args, **kwargs)
        huggingface_hub.snapshot_download = _patched_snapshot
    
    # Patch from_pretrained methods that might use use_auth_token
    for attr_name in dir(huggingface_hub):
        if attr_name.startswith('_'):
            continue
        attr = getattr(huggingface_hub, attr_name)
        if callable(attr):
            try:
                import inspect
                sig = inspect.signature(attr)
                if 'use_auth_token' in sig.parameters:
                    def make_patched(orig):
                        def patched(*args, **kwargs):
                            if 'use_auth_token' in kwargs:
                                kwargs['token'] = kwargs.pop('use_auth_token')
                            return orig(*args, **kwargs)
                        return patched
                    setattr(huggingface_hub, attr_name, make_patched(attr))
            except (ValueError, TypeError):
                pass
    
    # Mark as patched
    huggingface_hub._diarrhizer_patched = True


def _patch_torchaudio_compat():
    """Patch torchaudio for backward compatibility with packages expecting older APIs."""
    # Only patch if torchaudio is available but missing the expected attributes
    try:
        import torchaudio
    except ImportError:
        return  # torchaudio not installed, nothing to patch

    # Patch AudioMetaData if missing (needed by pyannote.audio)
    if not hasattr(torchaudio, "AudioMetaData"):
        # Create a compatible AudioMetaData class for pyannote.audio
        class AudioMetaData:
            def __init__(self, num_frames, num_channels, sample_rate, duration):
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.sample_rate = sample_rate
                self.duration = duration

            def __repr__(self):
                return (
                    f"AudioMetaData(num_frames={self.num_frames}, "
                    f"num_channels={self.num_channels}, "
                    f"sample_rate={self.sample_rate}, "
                    f"duration={self.duration})"
                )

        # Also need info property that pyannote.audio uses
        class AudioMetaDataWithInfo(AudioMetaData):
            @property
            def info(self):
                return self

        torchaudio.AudioMetaData = AudioMetaDataWithInfo

    # Patch list_audio_backends if missing (needed by speechbrain)
    # torchaudio 2.10.0 removed this function, but older packages still expect it
    if not hasattr(torchaudio, "list_audio_backends"):

        def list_audio_backends():
            return ["soundfile", "ffmpeg"]

        torchaudio.list_audio_backends = list_audio_backends


def _patch_torch_load():
    """Patch torch.load for backward compatibility with older serialization."""
    try:
        import torch
    except ImportError:
        return  # torch not installed

    # Skip if already patched
    if hasattr(torch, "_diarrhizer_patched"):
        return

    # Save original load function
    _original_torch_load = torch.load

    def _patched_torch_load(
        f,
        map_location=None,
        pickle_module=None,
        weights_only=False,
        **kwargs,
    ):
        """Patched torch.load that defaults to weights_only=False for compatibility."""
        return _original_torch_load(
            f,
            map_location=map_location,
            pickle_module=pickle_module,
            weights_only=False,  # Always force False for compatibility
            **kwargs,
        )

    torch.load = _patched_torch_load

    # Also patch lightning_fabric if available
    try:
        import lightning_fabric.utilities.cloud_io as cloud_io

        _original_lf_load = cloud_io._load

        def _patched_lf_load(f, map_location=None, **kwargs):
            # Remove weights_only if present or set to True to force False
            kwargs.pop("weights_only", None)
            kwargs["weights_only"] = False
            return _original_lf_load(f, map_location=map_location, **kwargs)

        cloud_io._load = _patched_lf_load
    except (ImportError, AttributeError):
        pass

    # Mark as patched to avoid double-patching
    torch._diarrhizer_patched = True


# Apply patches at module import time
_patch_pyannote_audio()
_patch_huggingface_hub()
_patch_torchaudio_compat()
_patch_torch_load()
