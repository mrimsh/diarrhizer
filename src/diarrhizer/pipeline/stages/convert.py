"""Convert stage for audio normalization."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union, List

from diarrhizer.adapters.ffmpeg import FFmpegAdapter

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import JobContext


# [SEMANTIC-BEGIN] STAGE:CONVERT
# @purpose: Normalize input media to WAV mono 16kHz with optional audio profiles
# @description: Uses FFmpeg adapter to convert input audio/video to a standardized format with preprocessing
# @inputs: job.input_path, config.audio_profile
# @outputs: artifacts/audio/normalized.wav, meta/run.json
# @sideEffects: Creates output directory structure, writes audio file to disk
# @errors: RuntimeError, FileNotFoundError
# @see: ADAPTER:FFMPEG, PIPELINE:RUNNER
class ConvertStage:
    """Stage for converting input media to normalized WAV format."""

    # Stage name for identification
    NAME = "convert"

    # Output paths relative to job directory
    AUDIO_DIR = "audio"
    META_DIR = "meta"
    NORMALIZED_WAV = "audio/normalized.wav"
    NORMALIZED_LEFT_WAV = "audio/normalized_left.wav"
    NORMALIZED_RIGHT_WAV = "audio/normalized_right.wav"
    META_RUN_JSON = "meta/run.json"

    def __init__(self) -> None:
        """Initialize the convert stage."""
        self._ffmpeg_adapter: FFmpegAdapter | None = None

    @property
    def ffmpeg_adapter(self) -> FFmpegAdapter:
        """Get or create FFmpeg adapter (lazy initialization)."""
        if self._ffmpeg_adapter is None:
            self._ffmpeg_adapter = FFmpegAdapter()
        return self._ffmpeg_adapter

    def run(self, job: "JobContext") -> dict:
        """Run the convert stage.

        Args:
            job: Job context containing input path and configuration

        Returns:
            Dictionary with stage output paths and metadata
        """
        input_path = job.input_path
        job_dir = job.job_dir
        config = job.config

        print(f"[{self.NAME}] Converting: {input_path}")

        # Get audio profile from config
        audio_profile = config.get("audio_profile", "raw")

        # Build output paths
        audio_output = job_dir / self.NORMALIZED_WAV
        meta_output = job_dir / self.META_RUN_JSON

        # Check if output already exists (idempotency)
        # For split-stereo, check if both channel files exist
        if audio_profile == FFmpegAdapter.PROFILE_SPLIT_STEREO:
            left_output = job_dir / self.NORMALIZED_LEFT_WAV
            right_output = job_dir / self.NORMALIZED_RIGHT_WAV
            if left_output.exists() and right_output.exists() and meta_output.exists():
                print(f"[{self.NAME}] Skipping - output already exists")
                return {
                    "stage": self.NAME,
                    "status": "skipped",
                    "output_path": str(left_output),
                    "output_paths": [str(left_output), str(right_output)],
                }
        else:
            if audio_output.exists() and meta_output.exists():
                print(f"[{self.NAME}] Skipping - output already exists")
                return {
                    "stage": self.NAME,
                    "status": "skipped",
                    "output_path": str(audio_output),
                }

        # Ensure output directories exist
        audio_output.parent.mkdir(parents=True, exist_ok=True)
        meta_output.parent.mkdir(parents=True, exist_ok=True)

        # Run FFmpeg conversion
        start_time = datetime.now()
        result_path = self.ffmpeg_adapter.convert_to_wav(
            input_path=str(input_path),
            output_path=str(audio_output),
            audio_profile=audio_profile,
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Determine output paths for metadata
        if isinstance(result_path, list):
            output_paths = result_path
            main_output = result_path[0]
        else:
            output_paths = [result_path]
            main_output = result_path

        # Write metadata
        meta_info = {
            "stage": self.NAME,
            "input_path": str(input_path),
            "output_path": str(main_output),
            "output_paths": [str(p) for p in output_paths],
            "config": {
                "sample_rate": self.ffmpeg_adapter.TARGET_SAMPLE_RATE,
                "channels": self.ffmpeg_adapter.TARGET_CHANNELS,
                "format": "wav",
                "audio_profile": audio_profile,
            },
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "pipeline_config": {
                "min_speakers": config.get("min_speakers"),
                "max_speakers": config.get("max_speakers"),
                "language": config.get("language"),
                "device": config.get("device"),
                "asr_model": config.get("asr_model"),
                "asr_compute_type": config.get("asr_compute_type"),
                "asr_beam_size": config.get("asr_beam_size"),
                "asr_temperature": config.get("asr_temperature"),
                "audio_profile": audio_profile,
            },
        }

        with open(meta_output, "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2, ensure_ascii=False)

        print(f"[{self.NAME}] Completed in {duration:.2f}s")
        print(f"[{self.NAME}] Profile: {audio_profile}")
        if isinstance(result_path, list):
            print(f"[{self.NAME}] Outputs: {', '.join(str(p) for p in result_path)}")
        else:
            print(f"[{self.NAME}] Output: {result_path}")

        return {
            "stage": self.NAME,
            "status": "completed",
            "output_path": str(main_output),
            "output_paths": [str(p) for p in output_paths],
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
            "audio": job_dir / self.NORMALIZED_WAV,
            "meta": job_dir / self.META_RUN_JSON,
        }

    def is_cache_valid(self, job_dir: Path) -> bool:
        """Check if stage output exists and is valid.

        Args:
            job_dir: Job directory path

        Returns:
            True if output exists and is valid
        """
        artifacts = self.get_artifact_paths(job_dir)
        return artifacts["audio"].exists() and artifacts["meta"].exists()


# [SEMANTIC-END] STAGE:CONVERT