"""Transcribe stage for WhisperX ASR and alignment."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from diarrhizer.adapters.whisperx import WhisperXAdapter

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import JobContext


# [SEMANTIC-BEGIN] STAGE:TRANSCRIBE
# @purpose: Transcribe audio using WhisperX with word-level alignment
# @description: Consumes normalized WAV from convert stage and produces transcript with timestamps
# @inputs: artifacts/audio/normalized.wav
# @outputs: artifacts/asr/transcript.json
# @sideEffects: Loads WhisperX model, writes transcript JSON to disk
# @errors: RuntimeError, FileNotFoundError
# @see: ADAPTER:WHISPERX_ASR, STAGE:CONVERT, PIPELINE:RUNNER
class TranscribeStage:
    """Stage for transcribing audio using WhisperX."""

    # Stage name for identification
    NAME = "transcribe"

    # Output paths relative to job directory
    ASR_DIR = "asr"
    TRANSCRIPT_JSON = "asr/transcript.json"

    # Input artifact path (from convert stage)
    INPUT_WAV = "audio/normalized.wav"

    def __init__(self, model: str = "base", device: str = "cuda") -> None:
        """Initialize the transcribe stage.

        Args:
            model: WhisperX model size (tiny, base, small, medium, large)
            device: Device to use ("cuda" or "cpu")
        """
        self._whisperx_adapter: WhisperXAdapter | None = None
        self._model = model
        self._device = device
        self._language: str | None = None
        # ASR parameters
        self._compute_type: Optional[str] = None
        self._beam_size: int = 5
        self._temperature: float = 0.0
        self._condition_on_previous_text: bool = True
        self._initial_prompt: Optional[str] = None
        self._vad_filter: bool = True
        self._vad_min_silence_ms: int = 1000

    def configure(
        self,
        language: str,
        device: str,
        model: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: Optional[int] = None,
        temperature: Optional[float] = None,
        condition_on_previous_text: Optional[bool] = None,
        initial_prompt: Optional[str] = None,
        vad_filter: Optional[bool] = None,
        vad_min_silence_ms: Optional[int] = None,
    ) -> None:
        """Configure transcription parameters.

        Args:
            language: Language code or "auto" for detection
            device: Device to use ("cuda" or "cpu")
            model: WhisperX model override
            compute_type: Compute type override
            beam_size: Decoding beam size
            temperature: Decoding temperature
            condition_on_previous_text: Condition on previous text
            initial_prompt: Initial prompt string
            vad_filter: Enable VAD filtering
            vad_min_silence_ms: VAD minimum silence in milliseconds
        """
        self._device = device
        self._language = None if language == "auto" else language
        if model is not None:
            self._model = model
        if compute_type is not None:
            self._compute_type = compute_type
        if beam_size is not None:
            self._beam_size = beam_size
        if temperature is not None:
            self._temperature = temperature
        if condition_on_previous_text is not None:
            self._condition_on_previous_text = condition_on_previous_text
        if initial_prompt is not None:
            self._initial_prompt = initial_prompt
        if vad_filter is not None:
            self._vad_filter = vad_filter
        if vad_min_silence_ms is not None:
            self._vad_min_silence_ms = vad_min_silence_ms
        self._whisperx_adapter = None

    @property
    def whisperx_adapter(self) -> WhisperXAdapter:
        """Get or create WhisperX adapter (lazy initialization)."""
        if self._whisperx_adapter is None:
            self._whisperx_adapter = WhisperXAdapter(
                model=self._model,
                device=self._device,
                language=self._language,
                compute_type=self._compute_type,
                beam_size=self._beam_size,
                temperature=self._temperature,
                condition_on_previous_text=self._condition_on_previous_text,
                initial_prompt=self._initial_prompt,
                vad_filter=self._vad_filter,
                vad_min_silence_ms=self._vad_min_silence_ms,
            )
        return self._whisperx_adapter

    def run(self, job: "JobContext") -> dict:
        """Run the transcribe stage.

        Args:
            job: Job context containing input path and configuration

        Returns:
            Dictionary with stage output paths and metadata
        """
        job_dir = job.job_dir
        config = job.config

        # Get input audio path
        audio_input = job_dir / self.INPUT_WAV

        # Build output paths
        transcript_output = job_dir / self.TRANSCRIPT_JSON

        print(f"[{self.NAME}] Transcribing: {audio_input}")

        # Check if input exists
        if not audio_input.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_input}. "
                "Please run the convert stage first."
            )

        # Check if output already exists (idempotency)
        if transcript_output.exists():
            print(f"[{self.NAME}] Skipping - output already exists")
            return {
                "stage": self.NAME,
                "status": "skipped",
                "output_path": str(transcript_output),
            }

        # Ensure output directory exists
        transcript_output.parent.mkdir(parents=True, exist_ok=True)

        # Configure adapter with job-specific settings
        language = config.get("language", "auto")
        device = config.get("device", "cuda")

        # Apply configuration (resets adapter if settings changed)
        self.configure(
            language=language,
            device=device,
            model=config.get("asr_model"),
            compute_type=config.get("asr_compute_type"),
            beam_size=config.get("asr_beam_size"),
            temperature=config.get("asr_temperature"),
            condition_on_previous_text=config.get("asr_condition_on_previous_text"),
            initial_prompt=config.get("asr_initial_prompt"),
            vad_filter=config.get("asr_vad_filter"),
            vad_min_silence_ms=config.get("asr_vad_min_silence_ms"),
        )

        # Run transcription
        start_time = datetime.now()

        try:
            # Handle "auto" language detection
            lang_param = None if language == "auto" else language
            result = self.whisperx_adapter.transcribe(
                audio_path=str(audio_input),
                language=lang_param,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Transcription failed: {e}"
            ) from e

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Prepare transcript data with full ASR configuration
        transcript_data = {
            "stage": self.NAME,
            "model": self._model,
            "compute_type": self._compute_type,
            "language": result.get("language", language),
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "words": result.get("words", []),
            "metadata": {
                "input_audio": str(audio_input),
                "output_path": str(transcript_output),
                "device": device,
                "language_setting": language,
                "beam_size": self._beam_size,
                "temperature": self._temperature,
                "condition_on_previous_text": self._condition_on_previous_text,
                "vad_filter": self._vad_filter,
                "vad_min_silence_ms": self._vad_min_silence_ms,
                "initial_prompt": self._initial_prompt[:100] + "..." if self._initial_prompt and len(self._initial_prompt) > 100 else self._initial_prompt,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
            },
        }

        # Write transcript to JSON
        with open(transcript_output, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        print(f"[{self.NAME}] Completed in {duration:.2f}s")
        print(f"[{self.NAME}] Language: {result.get('language', 'unknown')}")
        print(f"[{self.NAME}] Output: {transcript_output}")
        if result.get("text"):
            print(f"[{self.NAME}] Text preview: {result['text'][:200]}...")

        return {
            "stage": self.NAME,
            "status": "completed",
            "output_path": str(transcript_output),
            "language": result.get("language", "unknown"),
            "duration_seconds": duration,
        }

    def get_artifact_paths(self, job_dir: Path) -> dict:
        """Get the expected artifact paths for this stage.

        Args:
            job_dir: Job directory path

        Returns:
            Dictionary of artifact name to path
        """
        return {
            "input_audio": job_dir / self.INPUT_WAV,
            "transcript": job_dir / self.TRANSCRIPT_JSON,
        }

    def is_cache_valid(self, job_dir: Path) -> bool:
        """Check if stage output exists and is valid.

        Args:
            job_dir: Job directory path

        Returns:
            True if output exists and is valid
        """
        artifacts = self.get_artifact_paths(job_dir)
        return artifacts["transcript"].exists()


# [SEMANTIC-END] STAGE:TRANSCRIBE