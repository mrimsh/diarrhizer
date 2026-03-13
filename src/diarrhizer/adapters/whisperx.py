"""WhisperX adapter for ASR, alignment, and diarization."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch

# [SEMANTIC-BEGIN] ADAPTER:WHISPERX_ASR
# @purpose: Wrap WhisperX for ASR transcription and word-level alignment
# @description: Provides a clean interface to WhisperX for transcribing audio with timestamps
# @inputs: audio_path, language, device
# @outputs: Transcript dictionary with segments and word-level timestamps
# @sideEffects: Loads WhisperX model into memory (GPU/CPU), creates output artifacts
# @errors: RuntimeError if whisperx/torch is missing, model download fails, or transcription fails
# @see: STAGE:TRANSCRIBE
class WhisperXAdapter:
    """Adapter for WhisperX ASR operations."""

    # Default model - small and fast, good for most use cases
    DEFAULT_MODEL = "base"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "cuda",
        language: Optional[str] = None,
    ) -> None:
        """Initialize the WhisperX adapter.

        Args:
            model: WhisperX model size (tiny, base, small, medium, large)
            device: Device to use ("cuda" or "cpu")
            language: Language code (e.g., "en", "ru"). If None, auto-detect.

        Raises:
            RuntimeError: If required dependencies are missing or CUDA requested but unavailable
        """
        self._model = model
        self._device = self._validate_device(device)
        self._language = language
        self._whisperx: Any = None
        self._model_loaded = False

    def _validate_device(self, device: str) -> str:
        """Validate and return the device to use.

        Args:
            device: Requested device ("cuda" or "cpu")

        Returns:
            Valid device string

        Raises:
            RuntimeError: If CUDA is requested but not available
        """
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                "Please install PyTorch with CUDA support or use --device cpu"
            )
        return device

    def _load_whisperx(self) -> None:
        """Load WhisperX and download model if needed."""
        if self._model_loaded:
            return

        try:
            import whisperx
        except ImportError as e:
            raise RuntimeError(
                "WhisperX not installed. Please install it with:\n"
                "pip install whisperx\n"
                f"Import error: {e}"
            ) from e

        # Determine compute type based on device
        compute_type = "float16" if self._device == "cuda" else "int8"

        # Load model
        try:
            self._whisperx = whisperx
            self._whisper_model = whisperx.load_model(
                self._model,
                device=self._device,
                compute_type=compute_type,
            )
            self._model_loaded = True
        except Exception as e:
            error_msg = str(e)
            # Check for specific known errors and provide helpful messages
            if "LazyModule" in error_msg or "speechbrain" in error_msg:
                raise RuntimeError(
                    f"Failed to load WhisperX model '{self._model}': {e}\n\n"
                    "This error typically indicates a dependency compatibility issue.\n"
                    "Common causes:\n"
                    "  - Incompatible versions of speechbrain, huggingface-hub, or torch\n"
                    "  - Using torch 2.10+ with WhisperX models trained on older torch\n"
                    "  - numpy 2.x incompatibility with speechbrain\n"
                    "Try:\n"
                    "  1. Run: python -m diarrhizer doctor\n"
                    "  2. Rebuild environment with: pip install -c requirements/constraints-stable.txt -r requirements/base.txt"
                ) from e
            elif "float16" in error_msg.lower():
                raise RuntimeError(
                    f"Failed to load WhisperX model '{self._model}': {e}\n\n"
                    "This error indicates the compute type is incompatible with your device.\n"
                    "For CPU, use --device cpu or ensure int8 compute type is available."
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to load WhisperX model '{self._model}': {e}"
                ) from e

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
    ) -> dict:
        """Transcribe audio file with word-level alignment.

        Args:
            audio_path: Path to audio file (WAV mono 16kHz preferred)
            language: Language code. Uses adapter default if not provided.

        Returns:
            Dictionary containing:
            - text: Full transcribed text
            - segments: List of segments with start/end times and text
            - words: List of words with start/end times
            - language: Detected or specified language

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If transcription fails
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use provided language or fall back to adapter default
        lang = language or self._language

        # Load WhisperX if not already loaded
        self._load_whisperx()

        try:
            # Load audio
            audio = self._whisperx.load_audio(str(audio_path))

            # Transcribe
            result = self._whisper_model.transcribe(
                audio,
                language=lang,
                # Batch size for faster processing on GPU
                batch_size=16 if self._device == "cuda" else 1,
            )

            # Get detected language if auto-detected
            detected_language = result.get("language", lang or "unknown")

            # Align words if language detected
            alignment_result = result
            if detected_language and detected_language != "unknown":
                try:
                    # Load alignment model
                    # Note: whisperx 3.7+ API changed - language_code is positional, device is required
                    align_model, metadata = self._whisperx.load_align_model(
                        detected_language,
                        self._device
                    )
                    # Align words
                    alignment_result = self._whisperx.align(
                        result["segments"],
                        align_model,
                        metadata,
                        audio,
                        device=self._device,
                    )
                except Exception as e:
                    # Alignment is best-effort, log but continue
                    logging.warning(f"Word alignment failed: {e}")

            return {
                "text": alignment_result.get("text", ""),
                "segments": alignment_result.get("segments", []),
                "words": alignment_result.get("words", []),
                "language": detected_language,
            }

        except Exception as e:
            raise RuntimeError(
                f"WhisperX transcription failed: {e}"
            ) from e

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    @property
    def model(self) -> str:
        """Get the model being used."""
        return self._model


# [SEMANTIC-END] ADAPTER:WHISPERX_ASR


# Module-level convenience function
def transcribe_audio(
    audio_path: str | Path,
    language: Optional[str] = None,
    device: str = "cuda",
    model: str = "base",
) -> dict:
    """Convenience function to transcribe audio using WhisperX.

    Args:
        audio_path: Path to audio file
        language: Language code or None for auto-detect
        device: Device to use ("cuda" or "cpu")
        model: WhisperX model size

    Returns:
        Transcript dictionary with text, segments, and words
    """
    adapter = WhisperXAdapter(model=model, device=device, language=language)
    return adapter.transcribe(audio_path, language=language)


# [SEMANTIC-BEGIN] ADAPTER:WHISPERX_DIARIZE
# @purpose: Wrap WhisperX for speaker diarization using pyannote
# @description: Provides diarization capability via WhisperX's integration with pyannote
# @inputs: audio_path, min_speakers, max_speakers, device
# @outputs: Diarization result with speaker segments and timestamps
# @sideEffects: Loads pyannote model, accesses HF_TOKEN for gated models
# @errors: RuntimeError if HF_TOKEN missing, model fails, audio decoding fails
# @see: STAGE:DIARIZE
class WhisperXDiarizeAdapter:
    """Adapter for WhisperX diarization operations."""

    def __init__(
        self,
        device: str = "cuda",
        min_speakers: int = 1,
        max_speakers: int = 10,
    ) -> None:
        """Initialize the WhisperX diarization adapter.

        Args:
            device: Device to use ("cuda" or "cpu")
            min_speakers: Minimum number of expected speakers (currently unused)
            max_speakers: Maximum number of expected speakers (currently unused)

        Raises:
            RuntimeError: If HF_TOKEN is not set or CUDA requested but unavailable
        """
        # TODO: Implement min_speakers/max_speakers filtering in diarize()
        # Currently WhisperX/pyannote determines speaker count automatically.
        # These parameters could be used to post-filter results or constrain the model.
        self._device = self._validate_device(device)
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._hf_token: str | None = None
        self._diarize_model: Any = None

    def _validate_device(self, device: str) -> str:
        """Validate and return the device to use."""
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. "
                "Please install PyTorch with CUDA support or use --device cpu"
            )
        return device

    def _check_hf_token(self) -> str:
        """Check and return HuggingFace token.

        Returns:
            HF token string

        Raises:
            RuntimeError: If token is not set
        """
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            raise RuntimeError(
                "HuggingFace token is required for diarization.\n"
                "Please set HF_TOKEN or HUGGINGFACE_HUB_TOKEN environment variable.\n"
                "You can get a token from: https://huggingface.co/settings/tokens"
            )
        return token

    def _load_diarization_model(self) -> None:
        """Load WhisperX diarization model."""
        if self._diarize_model is not None:
            return

        try:
            import whisperx
        except ImportError as e:
            raise RuntimeError(
                "WhisperX not installed. Please install it with:\n"
                "pip install whisperx\n"
                f"Import error: {e}"
            ) from e

        # Check HF token first
        self._hf_token = self._check_hf_token()

        try:
            # Load diarization model using pyannote.audio directly
            # Note: whisperx 3.7+ changed the API - we use pyannote.audio.Pipeline directly
            from pyannote.audio import Pipeline
            import os
            
            # Set HF token in environment for pyannote.audio
            # Both variables are checked by huggingface_hub
            os.environ["HF_TOKEN"] = self._hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = self._hf_token
            
            # Load the pyannote diarization pipeline
            # Using version 3.1 which is stable
            # Note: pyannote.audio uses HF_TOKEN from environment, no need to pass explicitly
            self._diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            
            # Move to device if CUDA
            if self._device == "cuda":
                self._diarize_model.to(torch.device("cuda"))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load diarization model: {e}"
            ) from e

    def _load_audio_fallback(self, audio_path: str | Path) -> Any:
        """Fallback audio loading using scipy.

        This is used when the default WhisperX audio loading fails
        (e.g., due to torchcodec/FFmpeg compatibility issues on Windows).

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary format that pyannote.pipeline accepts
        """
        import scipy.io.wavfile as wav
        import numpy as np

        logging.warning(
            "Using fallback audio loading via scipy. "
            "Diarization will continue with preloaded waveform."
        )

        # Load audio using scipy
        sample_rate, waveform = wav.read(str(audio_path))
        
        # Convert to float32 if needed
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32) / 32768.0
        
        # Ensure mono
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy.signal import resample
            num_samples = int(len(waveform) * 16000 / sample_rate)
            waveform = resample(waveform, num_samples)
            sample_rate = 16000
        
        # Return as dict that pyannote.pipeline can use
        import torch
        return {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sample_rate}

    def diarize(
        self,
        audio_path: str | Path,
        use_fallback: bool = False,
    ) -> dict:
        """Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file (WAV mono 16kHz preferred)
            use_fallback: If True, use torchaudio fallback for loading audio

        Returns:
            Dictionary containing:
            - segments: List of segments with start/end times and speaker labels
            - num_speakers: Detected/estimated number of speakers

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If diarization fails
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load model if not already loaded
        self._load_diarization_model()

        try:
            # Run diarization using the pyannote pipeline
            if use_fallback:
                # Use scipy to load audio and pass to pipeline as dict
                audio_dict = self._load_audio_fallback(audio_path)
                result = self._diarize_model(audio_dict)
            else:
                # Use default pyannote.audio loading (requires torchcodec or FFmpeg)
                result = self._diarize_model(str(audio_path))

            # Convert to our format
            segments = []
            for segment in result.itertracks(yield_label=True):
                # segment is (start, end, speaker)
                segments.append({
                    "start": float(segment[0].start),
                    "end": float(segment[0].end),
                    "speaker": str(segment[2]),
                })

            # Get unique speakers
            speakers = set(s["speaker"] for s in segments)
            num_speakers = len(speakers)

            return {
                "segments": segments,
                "num_speakers": num_speakers,
                "speakers": sorted(list(speakers)),
            }

        except RuntimeError:
            # Re-raise RuntimeErrors (like missing token)
            raise
        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's an audio decoding issue
            if (
                not use_fallback
                and (
                    "decoder" in error_msg
                    or "audio" in error_msg
                    or "ffmpeg" in error_msg
                    or "codec" in error_msg
                    or "load_audio" in error_msg
                    or "torchcodec" in error_msg
                    or "torchaudio" in error_msg
                )
            ):
                logging.warning(
                    f"Diarization failed with default audio loading: {e}\n"
                    "Attempting fallback approach with torchaudio..."
                )
                # Try fallback
                return self.diarize(audio_path, use_fallback=True)

            raise RuntimeError(
                f"Diarization failed: {e}"
            ) from e

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device


# [SEMANTIC-END] ADAPTER:WHISPERX_DIARIZE


# Module-level convenience function
def diarize_audio(
    audio_path: str | Path,
    min_speakers: int = 1,
    max_speakers: int = 10,
    device: str = "cuda",
) -> dict:
    """Convenience function to diarize audio using WhisperX.

    Args:
        audio_path: Path to audio file
        min_speakers: Minimum number of expected speakers
        max_speakers: Maximum number of expected speakers
        device: Device to use ("cuda" or "cpu")

    Returns:
        Diarization dictionary with segments and speaker info
    """
    adapter = WhisperXDiarizeAdapter(
        device=device,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return adapter.diarize(audio_path)
