"""Convert stage for audio normalization."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union, List

from diarrhizer.adapters.ffmpeg import FFmpegAdapter
from diarrhizer.pipeline.cache import is_stale
from diarrhizer.utils import write_json_atomic

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import JobContext


# [SEMANTIC-BEGIN] STAGE:CONVERT
# @purpose: Normalize input media to WAV mono 16kHz with optional audio profiles
# @description: Uses FFmpeg adapter to convert input audio/video to a standardized format with preprocessing.
#   Every profile, including split-stereo, always produces audio/normalized.wav - the file every downstream
#   stage reads unconditionally - so pipeline shape never depends on audio_profile. split-stereo additionally
#   produces audio/normalized_left.wav / normalized_right.wav as extra artifacts (not consumed downstream).
#   get_output_paths()/is_cache_valid() only expect those two extras once a prior run's meta/run.json recorded
#   audio_profile="split-stereo" (see _read_recorded_audio_profile) - on a first run, before meta/run.json
#   exists, only normalized.wav+meta are checked, which is still correct since a missing normalized.wav alone
#   is enough to mark the stage stale.
# @inputs: job.input_path, config.audio_profile
# @outputs: artifacts/audio/normalized.wav, meta/run.json, and for split-stereo also
#   artifacts/audio/normalized_left.wav / normalized_right.wav
# @sideEffects: Creates output directory structure, writes audio file(s) to disk
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
        audio_profile = config.audio_profile

        # Build output paths
        audio_output = job_dir / self.NORMALIZED_WAV
        meta_output = job_dir / self.META_RUN_JSON

        # Whether to (re)run is decided by the pipeline runner (is_cache_valid
        # + force flags), not here - run() always does the work when called.

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
                "min_speakers": config.min_speakers,
                "max_speakers": config.max_speakers,
                "language": config.language,
                "device": config.device,
                "asr_model": config.asr_model,
                "asr_compute_type": config.asr_compute_type,
                "asr_beam_size": config.asr_beam_size,
                "asr_temperature": config.asr_temperature,
                "audio_profile": audio_profile,
            },
        }

        write_json_atomic(meta_output, meta_info)

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

    def _read_recorded_audio_profile(self, job_dir: Path) -> str | None:
        """Recover the audio_profile a prior convert run recorded in
        meta/run.json, so cache checks know whether to expect split-stereo's
        extra normalized_left.wav/normalized_right.wav artifacts. Returns
        None if there's no prior run to read (e.g. before convert ever ran).
        """
        meta_path = job_dir / self.META_RUN_JSON
        if not meta_path.exists():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data.get("config", {}).get("audio_profile")

    def get_artifact_paths(self, job_dir: Path) -> dict:
        """Get the expected artifact paths for this stage.

        For split-stereo, also includes the per-channel extras once a prior
        run has recorded that profile in meta/run.json (see
        _read_recorded_audio_profile) - not just the primary normalized.wav
        every profile shares.

        Args:
            job_dir: Job directory path

        Returns:
            Dictionary of artifact name to path
        """
        paths = {
            "audio": job_dir / self.NORMALIZED_WAV,
            "meta": job_dir / self.META_RUN_JSON,
        }
        if self._read_recorded_audio_profile(job_dir) == FFmpegAdapter.PROFILE_SPLIT_STEREO:
            paths["audio_left"] = job_dir / self.NORMALIZED_LEFT_WAV
            paths["audio_right"] = job_dir / self.NORMALIZED_RIGHT_WAV
        return paths

    def get_output_paths(self, job_dir: Path) -> dict:
        """Get only the artifact paths this stage produces.

        Convert is the first stage: it has no job-directory inputs of its
        own (it reads the original external input file), so its outputs are
        the same as its full artifact set.

        Args:
            job_dir: Job directory path

        Returns:
            Dictionary of output artifact name to path
        """
        return self.get_artifact_paths(job_dir)

    def is_cache_valid(self, job_dir: Path) -> bool:
        """Check if stage output exists and is up to date.

        Args:
            job_dir: Job directory path

        Returns:
            True if output exists and is valid
        """
        outputs = list(self.get_output_paths(job_dir).values())
        return not is_stale(outputs=outputs, inputs=[])


# [SEMANTIC-END] STAGE:CONVERT