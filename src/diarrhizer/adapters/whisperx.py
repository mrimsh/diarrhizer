"""WhisperX adapter for ASR and word-level alignment."""

import logging
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
                    align_model, metadata = self._whisperx.load_align_model(
                        language=detected_language
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
