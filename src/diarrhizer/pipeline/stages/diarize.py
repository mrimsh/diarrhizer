"""Diarize stage for speaker diarization."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from diarrhizer.adapters.whisperx import WhisperXDiarizeAdapter

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import JobContext


# [SEMANTIC-BEGIN] STAGE:DIARIZE
# @purpose: Perform speaker diarization using WhisperX/pyannote
# @description: Consumes normalized WAV from convert stage and produces speaker segments
# @inputs: artifacts/audio/normalized.wav
# @outputs: artifacts/diar/diarization.json
# @sideEffects: Loads pyannote model, writes diarization JSON to disk
# @errors: RuntimeError if HF_TOKEN missing, FileNotFoundError
# @see: ADAPTER:WHISPERX_DIARIZE, STAGE:CONVERT, PIPELINE:RUNNER
class DiarizeStage:
    """Stage for performing speaker diarization."""

    # Stage name for identification
    NAME = "diarize"

    # Output paths relative to job directory
    DIAR_DIR = "diar"
    DIARIZATION_JSON = "diar/diarization.json"

    # Input artifact path (from convert stage)
    INPUT_WAV = "audio/normalized.wav"

    def __init__(
        self,
        device: str = "cuda",
        min_speakers: int = 1,
        max_speakers: int = 10,
    ) -> None:
        """Initialize the diarize stage.

        Args:
            device: Device to use ("cuda" or "cpu")
            min_speakers: Minimum number of expected speakers
            max_speakers: Maximum number of expected speakers
        """
        self._diarize_adapter: WhisperXDiarizeAdapter | None = None
        self._device = device
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers

    def configure(self, device: str, min_speakers: int, max_speakers: int) -> None:
        """Configure diarization parameters.

        Args:
            device: Device to use ("cuda" or "cpu")
            min_speakers: Minimum number of expected speakers
            max_speakers: Maximum number of expected speakers
        """
        self._device = device
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        # Reset adapter to use new settings
        self._diarize_adapter = None

    @property
    def diarize_adapter(self) -> WhisperXDiarizeAdapter:
        """Get or create diarize adapter (lazy initialization)."""
        if self._diarize_adapter is None:
            self._diarize_adapter = WhisperXDiarizeAdapter(
                device=self._device,
                min_speakers=self._min_speakers,
                max_speakers=self._max_speakers,
            )
        return self._diarize_adapter

    def run(self, job: "JobContext") -> dict:
        """Run the diarize stage.

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
        diar_output = job_dir / self.DIARIZATION_JSON

        print(f"[{self.NAME}] Diarizing: {audio_input}")

        # Check if input exists
        if not audio_input.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_input}. "
                "Please run the convert stage first."
            )

        # Check if output already exists (idempotency)
        if diar_output.exists():
            print(f"[{self.NAME}] Skipping - output already exists")
            return {
                "stage": self.NAME,
                "status": "skipped",
                "output_path": str(diar_output),
            }

        # Ensure output directory exists
        diar_output.parent.mkdir(parents=True, exist_ok=True)

        # Configure adapter with job-specific settings
        device = config.get("device", "cuda")
        min_speakers = config.get("min_speakers", 1)
        max_speakers = config.get("max_speakers", 10)

        # Apply configuration (resets adapter if settings changed)
        self.configure(
            device=device,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # Run diarization
        start_time = datetime.now()

        try:
            result = self.diarize_adapter.diarize(
                audio_path=str(audio_input),
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Diarization failed: {e}"
            ) from e

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Prepare diarization data
        diar_data = {
            "stage": self.NAME,
            "segments": result.get("segments", []),
            "num_speakers": result.get("num_speakers", 0),
            "speakers": result.get("speakers", []),
            "metadata": {
                "input_audio": str(audio_input),
                "output_path": str(diar_output),
                "device": device,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
            },
        }

        # Write diarization to JSON
        with open(diar_output, "w", encoding="utf-8") as f:
            json.dump(diar_data, f, indent=2, ensure_ascii=False)

        print(f"[{self.NAME}] Completed in {duration:.2f}s")
        print(f"[{self.NAME}] Speakers detected: {result.get('num_speakers', 0)}")
        print(f"[{self.NAME}] Output: {diar_output}")

        return {
            "stage": self.NAME,
            "status": "completed",
            "output_path": str(diar_output),
            "num_speakers": result.get("num_speakers", 0),
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
            "diarization": job_dir / self.DIARIZATION_JSON,
        }

    def is_cache_valid(self, job_dir: Path) -> bool:
        """Check if stage output exists and is valid.

        Args:
            job_dir: Job directory path

        Returns:
            True if output exists and is valid
        """
        artifacts = self.get_artifact_paths(job_dir)
        return artifacts["diarization"].exists()


# [SEMANTIC-END] STAGE:DIARIZE
