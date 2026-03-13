"""Entry point for python -m diarrhizer."""

import sys
import warnings
warnings.filterwarnings("ignore")

# CRITICAL: Patch torchaudio BEFORE any other imports
def _apply_early_patches():
    """Apply all compatibility patches at the earliest possible point."""
    import os
    
    # Force pyannote.audio to use soundfile backend instead of torchcodec
    # This MUST be set before pyannote.audio is imported
    os.environ["PYANNOTE_AUDIO_BACKEND"] = "soundfile"
    os.environ["TORCHAUDIO_BACKEND"] = "soundfile"
    
    # Patch torchaudio for backward compatibility
    try:
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            def list_audio_backends():
                return ['soundfile', 'ffmpeg']
            torchaudio.list_audio_backends = list_audio_backends
        if not hasattr(torchaudio, 'AudioMetaData'):
            class AudioMetaData:
                def __init__(self, num_frames, num_channels, sample_rate, duration):
                    self.num_frames = num_frames
                    self.num_channels = num_channels
                    self.sample_rate = sample_rate
                    self.duration = duration
                def __repr__(self):
                    return f"AudioMetaData(num_frames={self.num_frames}, num_channels={self.num_channels}, sample_rate={self.sample_rate}, duration={self.duration})"
            class AudioMetaDataWithInfo(AudioMetaData):
                @property
                def info(self):
                    return self
            torchaudio.AudioMetaData = AudioMetaDataWithInfo
    except ImportError:
        pass
    
    # Patch huggingface_hub for deprecated use_auth_token parameter
    try:
        import huggingface_hub
        if not hasattr(huggingface_hub, '_diarrhizer_patched'):
            _orig_download = huggingface_hub.hf_hub_download
            def _patched_download(*args, **kwargs):
                if 'use_auth_token' in kwargs:
                    kwargs['token'] = kwargs.pop('use_auth_token')
                return _orig_download(*args, **kwargs)
            huggingface_hub.hf_hub_download = _patched_download
            huggingface_hub._diarrhizer_patched = True
    except ImportError:
        pass

# Apply patches immediately
_apply_early_patches()

# Workaround for torchaudio 2.10.0+ incompatibility with pyannote.audio
# torchaudio 2.10.0 removed/changed several attributes that pyannote.audio still expects
# This shim provides backward compatibility
def _patch_torchaudio():
    import torchaudio
    if not hasattr(torchaudio, 'AudioMetaData'):
        # Create a compatible AudioMetaData class for pyannote.audio
        class AudioMetaData:
            def __init__(self, num_frames, num_channels, sample_rate, duration):
                self.num_frames = num_frames
                self.num_channels = num_channels
                self.sample_rate = sample_rate
                self.duration = duration
                
            def __repr__(self):
                return (f"AudioMetaData(num_frames={self.num_frames}, "
                        f"num_channels={self.num_channels}, "
                        f"sample_rate={self.sample_rate}, "
                        f"duration={self.duration})")
        
        # Also need info property that pyannote.audio uses
        class AudioMetaDataWithInfo(AudioMetaData):
            @property
            def info(self):
                return self
        
        torchaudio.AudioMetaData = AudioMetaDataWithInfo
    
    # Also patch list_audio_backends if missing
    if not hasattr(torchaudio, 'list_audio_backends'):
        def list_audio_backends():
            return ['soundfile', 'ffmpeg']
        torchaudio.list_audio_backends = list_audio_backends


# Workaround for PyTorch 2.6+ weights_only security feature
# Replace torch.load to always use weights_only=False
def _patch_torch():
    import torch
    
    # Save original load function
    _original_torch_load = torch.load
    
    def _patched_torch_load(f, map_location=None, pickle_module=None, 
                           weights_only=False, **kwargs):
        """Patched torch.load that defaults to weights_only=False for compatibility"""
        return _original_torch_load(
            f, 
            map_location=map_location, 
            pickle_module=pickle_module, 
            weights_only=False,  # Always force False
            **kwargs
        )
    
    torch.load = _patched_torch_load
    
    # Also patch lightning_fabric if available
    try:
        import lightning_fabric.utilities.cloud_io as cloud_io
        _original_lf_load = cloud_io._load
        
        def _patched_lf_load(f, map_location=None, **kwargs):
            # Remove weights_only if present or set to True to force False
            kwargs.pop('weights_only', None)
            kwargs['weights_only'] = False
            return _original_lf_load(f, map_location=map_location, **kwargs)
        
        cloud_io._load = _patched_lf_load
    except (ImportError, AttributeError):
        pass

_patch_torchaudio()
_patch_torch()

from diarrhizer.cli import main

if __name__ == "__main__":
    sys.exit(main())
