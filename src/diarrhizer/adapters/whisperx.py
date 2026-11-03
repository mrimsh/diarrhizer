"""WhisperX adapter for ASR, alignment, and diarization."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import gc

import torch

from diarrhizer.adapters.transcript_result import extract_transcript_fields
from diarrhizer.export.speakers import normalize_speaker_id


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# [SEMANTIC-BEGIN] ADAPTER:WHISPERX_ASR
# @purpose: Wrap WhisperX for ASR transcription and word-level alignment
# @description: Provides a clean interface to WhisperX for transcribing audio with timestamps.
#   extract_transcript_fields() normalizes align()/transcribe() payloads (word_segments or
#   per-segment words, plus concatenated text) into our text/segments/words shape.
#   Beam size and initial prompt go to load_model(asr_options). Unsupported batched-backend
#   temperature/previous-text/VAD controls produce warnings and explicit effective settings.
#   Alignment failure and partial timing are returned as quality metadata, never silently hidden.
# @inputs: audio_path, language, device, compute_type, beam_size, temperature, initial_prompt, vad_params
# @outputs: Transcript dictionary with segments, word-level timestamps and quality metadata
# @sideEffects: Loads WhisperX model into memory (GPU/CPU); unload() releases it and empties the CUDA cache
# @errors: RuntimeError if whisperx/torch is missing, model download fails, or transcription fails
# @see: STAGE:TRANSCRIBE, CONFIG:INITIAL_PROMPT
class WhisperXAdapter:
    """Adapter for WhisperX ASR operations."""

    # Default model - largest official Whisper checkpoint, best accuracy
    DEFAULT_MODEL = "large-v3"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "cuda",
        language: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        temperature: float = 0.0,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        vad_min_silence_ms: int = 1000,
    ) -> None:
        """Initialize the WhisperX adapter.

        Args:
            model: WhisperX model size (tiny, base, small, medium, large) or HF repo
            device: Device to use ("cuda" or "cpu")
            language: Language code (e.g., "en", "ru"). If None, auto-detect.
            compute_type: Compute type (float16, int8_float16, int8). Auto-based on device if None.
            beam_size: Decoding beam size
            temperature: Decoding temperature
            condition_on_previous_text: Condition on previous text for stable decoding
            initial_prompt: Initial prompt string for terminology guidance
            vad_filter: Enable VAD filtering
            vad_min_silence_ms: VAD minimum silence in milliseconds

        Raises:
            RuntimeError: If required dependencies are missing or CUDA requested but unavailable
        """
        self._model = model
        self._device = self._validate_device(device)
        self._language = language
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._temperature = temperature
        self._condition_on_previous_text = condition_on_previous_text
        self._initial_prompt = initial_prompt
        self._vad_filter = vad_filter
        self._vad_min_silence_ms = vad_min_silence_ms
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

    def _load_whisperx(self, language, compute_type, beam_size, initial_prompt) -> None:
        """Load WhisperX and download model if needed."""
        compute_type = compute_type or ("float16" if self._device == "cuda" else "int8")
        settings = (language, compute_type, beam_size, initial_prompt)
        if self._model_loaded and self._loaded_settings == settings:
            return
        if self._model_loaded:
            self.unload()

        try:
            import whisperx
        except ImportError as e:
            raise RuntimeError(
                "WhisperX not installed. Please install it with:\n"
                "pip install whisperx\n"
                f"Import error: {e}"
            ) from e

        # Load model
        try:
            self._whisperx = whisperx
            self._whisper_model = whisperx.load_model(
                self._model,
                device=self._device,
                compute_type=compute_type,
                language=language,
                asr_options={"beam_size": beam_size, "initial_prompt": initial_prompt},
            )
            self._loaded_settings = settings
            self._model_loaded = True
        except Exception as e:
            error_msg = str(e)
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
            elif "float16" in error_msg.lower() or "int8" in error_msg.lower():
                raise RuntimeError(
                    f"Failed to load WhisperX model '{self._model}': {e}\n\n"
                    "This error indicates the compute type is incompatible with your device.\n"
                    "For CPU, try --asr-compute-type int8 or int8_float16."
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to load WhisperX model '{self._model}': {e}"
                ) from e

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: Optional[int] = None,
        temperature: Optional[float] = None,
        condition_on_previous_text: Optional[bool] = None,
        initial_prompt: Optional[str] = None,
        vad_filter: Optional[bool] = None,
        vad_min_silence_ms: Optional[int] = None,
    ) -> dict:
        """Transcribe audio file with word-level alignment.

        Args:
            audio_path: Path to audio file (WAV mono 16kHz preferred)
            language: Language code. Uses adapter default if not provided.
            compute_type: Override compute type for this transcription
            beam_size: Override beam size for this transcription
            temperature: Override temperature for this transcription
            condition_on_previous_text: Override condition flag for this transcription
            initial_prompt: Override initial prompt for this transcription
            vad_filter: Override VAD filter flag for this transcription
            vad_min_silence_ms: Override VAD min silence for this transcription

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

        lang = language or self._language
        current_compute_type = compute_type or self._compute_type
        current_beam_size = beam_size if beam_size is not None else self._beam_size
        current_temperature = temperature if temperature is not None else self._temperature
        current_condition = condition_on_previous_text if condition_on_previous_text is not None else self._condition_on_previous_text
        current_initial_prompt = self._initial_prompt if initial_prompt is None else initial_prompt
        current_vad_filter = vad_filter if vad_filter is not None else self._vad_filter

        warnings = []
        if current_temperature != 0:
            warnings.append("WhisperX batched ASR does not support temperature; effective value is 0.")
        if current_condition:
            warnings.append("WhisperX batched ASR does not condition on previous text; effective value is false.")
        if not current_vad_filter:
            warnings.append("WhisperX requires VAD; vad_filter=false is unsupported, VAD remains enabled.")
        warnings.append("WhisperX does not support vad_min_silence_ms; its own VAD thresholds are used.")
        effective = {
            "compute_type": current_compute_type or ("float16" if self._device == "cuda" else "int8"),
            "beam_size": current_beam_size, "initial_prompt": current_initial_prompt,
            "temperature": 0.0, "condition_on_previous_text": False,
            "vad_filter": True, "vad_min_silence_ms": None,
        }
        for warning in warnings:
            logging.getLogger(__name__).warning(warning)
        self._load_whisperx(lang, current_compute_type, current_beam_size, current_initial_prompt)

        try:
            audio = self._whisperx.load_audio(str(audio_path))

            # WhisperX 3.3.1 consumes decoding options at load_model, not here.
            result = self._whisper_model.transcribe(audio, language=lang)

            # Get detected language if auto-detected
            detected_language = result.get("language", lang or "unknown")

            # Align words if language detected
            alignment_result = result
            alignment_status = "unavailable"
            align_model = None
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
                    alignment_status = "aligned"
                except Exception as e:
                    # Alignment is best-effort, log but continue
                    logging.warning(f"Word alignment failed: {e}")
                    alignment_status = "failed"
                    warnings.append(f"Word alignment failed: {e}")
                finally:
                    del align_model

            extracted = extract_transcript_fields(alignment_result)
            original = extract_transcript_fields(result)
            if "".join(extracted["text"].split()) != "".join(original["text"].split()):
                extracted = original
                alignment_status = "partial"
                warnings.append("Word alignment changed transcript text; original ASR segments are preserved.")
            untimed = sum(w["start"] is None or w["end"] is None for w in extracted["words"])
            if untimed:
                alignment_status = "partial"
                warnings.append(f"{untimed} word(s) have no valid alignment timestamps; text is preserved.")
            elif alignment_status == "aligned" and extracted["text"] and not extracted["words"]:
                alignment_status = "partial"
                warnings.append("Word alignment returned no words; original segment text is preserved.")

            return {
                "text": extracted["text"],
                "segments": extracted["segments"],
                "words": extracted["words"],
                "language": detected_language,
                "effective_settings": effective,
                "alignment_status": alignment_status,
                "warnings": warnings,
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

    def unload(self) -> None:
        """Drop the loaded ASR model and free GPU memory if possible."""
        self._whisper_model = None
        self._whisperx = None
        self._model_loaded = False
        _release_cuda()


# [SEMANTIC-END] ADAPTER:WHISPERX_ASR


# Module-level convenience function
def transcribe_audio(
    audio_path: str | Path,
    language: Optional[str] = None,
    device: str = "cuda",
    model: str = "large-v3",
    compute_type: Optional[str] = None,
    beam_size: int = 5,
    temperature: float = 0.0,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    vad_filter: bool = True,
    vad_min_silence_ms: int = 1000,
) -> dict:
    """Convenience function to transcribe audio using WhisperX.

    Args:
        audio_path: Path to audio file
        language: Language code or None for auto-detect
        device: Device to use ("cuda" or "cpu")
        model: WhisperX model size
        compute_type: Compute type (float16, int8_float16, int8)
        beam_size: Decoding beam size
        temperature: Decoding temperature
        condition_on_previous_text: Condition on previous text for stable decoding
        initial_prompt: Initial prompt string for terminology guidance
        vad_filter: Enable VAD filtering
        vad_min_silence_ms: VAD minimum silence in milliseconds

    Returns:
        Transcript dictionary with text, segments, and words
    """
    adapter = WhisperXAdapter(
        model=model,
        device=device,
        language=language,
        compute_type=compute_type,
        beam_size=beam_size,
        temperature=temperature,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
        vad_min_silence_ms=vad_min_silence_ms,
    )
    return adapter.transcribe(audio_path, language=language)


# [SEMANTIC-BEGIN] ADAPTER:WHISPERX_DIARIZE
# @purpose: Wrap WhisperX for speaker diarization using pyannote
# @description: Provides diarization capability via WhisperX's integration with pyannote
# @inputs: audio_path, min_speakers, max_speakers, device
# @outputs: Diarization result with speaker segments and timestamps
# @sideEffects: Loads pyannote model, accesses HF_TOKEN for gated models; unload() releases GPU memory
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
            min_speakers: Minimum expected speakers, passed to pyannote as a constraint
            max_speakers: Maximum expected speakers, passed to pyannote as a constraint

        Raises:
            RuntimeError: If HF_TOKEN is not set or CUDA requested but unavailable
        """
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
        """Load WhisperX diarization model (pyannote pipeline)."""
        if self._diarize_model is not None:
            return

        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError as e:
            raise RuntimeError(
                "WhisperX not installed. Please install it with:\n"
                "pip install whisperx\n"
                f"Import error: {e}"
            ) from e

        # Check HF token first
        self._hf_token = self._check_hf_token()

        try:
            # DiarizationPipeline wraps pyannote.audio Pipeline.
            # The token is passed via use_auth_token (not huggingface_token).
            self._diarize_model = DiarizationPipeline(
                use_auth_token=self._hf_token,
                device=self._device,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load diarization model: {e}"
            ) from e

    def _load_audio_fallback(self, audio_path: str | Path) -> Any:
        """Fallback audio loading using torchaudio.

        This is used when the default WhisperX audio loading fails
        (e.g., due to torchcodec/FFmpeg compatibility issues on Windows).

        Args:
            audio_path: Path to audio file

        Returns:
            Audio array compatible with diarization
        """
        import torchaudio

        logging.warning(
            "Using fallback audio loading via torchaudio. "
            "If this succeeds, diarization will continue with preloaded waveform."
        )

        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if needed (diarization expects 16kHz)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

        # Return as numpy array (WhisperX expects this format)
        return waveform.squeeze().numpy()

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
            import whisperx

            # Try default loading first
            if use_fallback:
                logging.info("Attempting diarization with fallback audio loading...")
                audio = self._load_audio_fallback(audio_path)
            else:
                # Default WhisperX audio loading
                audio = whisperx.load_audio(str(audio_path))

            # Run diarization (DiarizationPipeline returns a DataFrame)
            result = self._diarize_model(
                audio,
                min_speakers=self._min_speakers,
                max_speakers=self._max_speakers,
            )

            # Convert DataFrame to our format (columns: segment, label, speaker, start, end)
            segments = []
            for _, row in result.iterrows():
                segments.append({
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "speaker": normalize_speaker_id(str(row["speaker"])),
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

    def unload(self) -> None:
        """Drop the loaded diarization model and free GPU memory if possible."""
        self._diarize_model = None
        _release_cuda()


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
