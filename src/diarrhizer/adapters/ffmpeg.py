"""FFmpeg adapter for audio normalization."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List


# [SEMANTIC-BEGIN] ADAPTER:FFMPEG
# @purpose: Wrap FFmpeg calls for audio normalization and format conversion
# @description: Provides a clean interface to FFmpeg for converting media files to WAV with optional profiles
# @inputs: input_path (str or Path), output_path (str or Path), audio_profile
# @outputs: Path to converted audio file(s)
# @sideEffects: Executes FFmpeg subprocess, creates output file(s) on disk
# @errors: RuntimeError if FFmpeg is not found or conversion fails
# @see: STAGE:CONVERT
class FFmpegAdapter:
    """Adapter for FFmpeg audio conversion operations."""

    # Target audio format: WAV, mono, 16kHz
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1  # Mono

    # Audio profile filter presets
    PROFILE_RAW = "raw"
    PROFILE_VOICE_CALL = "voice-call"
    PROFILE_DENOISE_LIGHT = "denoise-light"
    PROFILE_SPLIT_STEREO = "split-stereo"

    def __init__(self) -> None:
        """Initialize the FFmpeg adapter and verify FFmpeg availability."""
        self._ffmpeg_path: Optional[str] = None
        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is available in PATH."""
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "FFmpeg not found in PATH. Please install FFmpeg and add it to your system PATH. "
                "See: https://ffmpeg.org/download.html"
            )
        self._ffmpeg_path = ffmpeg_path

    @property
    def ffmpeg_path(self) -> str:
        """Get the path to FFmpeg executable."""
        return self._ffmpeg_path or ""

    def convert_to_wav(
        self,
        input_path: str | Path,
        output_path: str | Path,
        audio_profile: str = PROFILE_RAW,
    ) -> Path | List[Path]:
        """Convert input media file to WAV with optional processing profile.

        Args:
            input_path: Path to input media file (any format FFmpeg supports)
            output_path: Path to output WAV file (for raw/denoise/voice-call) or base path (for split-stereo)
            audio_profile: Audio preprocessing profile (raw, voice-call, denoise-light, split-stereo)

        Returns:
            Path to the converted audio file, or list of paths for split-stereo

        Raises:
            RuntimeError: If FFmpeg conversion fails
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if audio_profile == self.PROFILE_SPLIT_STEREO:
            return self._convert_split_stereo(input_path, output_path)
        else:
            return self._convert_single_channel(input_path, output_path, audio_profile)

    def _convert_single_channel(
        self,
        input_path: Path,
        output_path: Path,
        audio_profile: str,
    ) -> Path:
        """Convert to single channel WAV with optional audio filters.

        Args:
            input_path: Path to input media file
            output_path: Path to output WAV file
            audio_profile: Audio processing profile

        Returns:
            Path to the converted audio file
        """
        # Build base command
        cmd = [
            self._ffmpeg_path,
            "-y",  # Overwrite output file if exists
            "-i", str(input_path),
            "-ac", str(self.TARGET_CHANNELS),  # Mono
            "-ar", str(self.TARGET_SAMPLE_RATE),  # 16kHz
            "-acodec", "pcm_s16le",  # 16-bit PCM
        ]

        # Add profile-specific audio filters
        afilters = []
        if audio_profile == self.PROFILE_VOICE_CALL:
            afilters.append("lowpass=7000,highpass=200,equalizer=f=3000:width_type=q:w=1:g=3")
        elif audio_profile == self.PROFILE_DENOISE_LIGHT:
            afilters.append("afftdn=nr=12:nt=auto")

        if afilters:
            cmd.extend(["-af", ",".join(afilters)])

        cmd.append(str(output_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour max for large files
            )
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg conversion failed: {e.stderr}"
            ) from e

    def _convert_split_stereo(self, input_path: Path, output_path: Path) -> List[Path]:
        """Split stereo audio into separate left and right channel files.

        Args:
            input_path: Path to input media file
            output_path: Base output path (left/right suffixes will be added)

        Returns:
            List of paths to converted audio files [left, right]
        """
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent

        left_path = parent / f"{stem}_left{suffix}"
        right_path = parent / f"{stem}_right{suffix}"

        # Extract left channel
        cmd_left = [
            self._ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-map_channel", "0.0.0",  # Left channel
            "-ar", str(self.TARGET_SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            str(left_path),
        ]

        # Extract right channel
        cmd_right = [
            self._ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-map_channel", "0.0.1",  # Right channel
            "-ar", str(self.TARGET_SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            str(right_path),
        ]

        try:
            subprocess.run(cmd_left, capture_output=True, text=True, check=True, timeout=3600)
            subprocess.run(cmd_right, capture_output=True, text=True, check=True, timeout=3600)
            return [left_path, right_path]
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg split-stereo conversion failed: {e.stderr}"
            ) from e

    def get_audio_info(self, input_path: str | Path) -> dict:
        """Get audio information from a media file.

        Args:
            input_path: Path to input media file

        Returns:
            Dictionary with audio properties (duration, sample_rate, channels, codec, etc.)
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        cmd = [
            self._ffmpeg_path,
            "-i", str(input_path),
            "-f", "null",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            stderr = result.stderr

            # Parse audio info from FFmpeg output
            info: dict = {
                "path": str(input_path),
                "exists": True,
            }

            # Parse duration (format: Duration: HH:MM:SS.ms)
            duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', stderr)
            if duration_match:
                hours, minutes, seconds, centiseconds = duration_match.groups()
                total_seconds = (
                    int(hours) * 3600
                    + int(minutes) * 60
                    + int(seconds)
                    + int(centiseconds) / 100
                )
                info["duration_seconds"] = round(total_seconds, 2)

            # Parse audio stream info (format: Stream #0:0: Audio: codec_name, ...)
            audio_stream_match = re.search(r'Stream.*Audio: (\w+)', stderr)
            if audio_stream_match:
                info["codec"] = audio_stream_match.group(1)

            # Parse sample rate (format: ... Hz, ...)
            sample_rate_match = re.search(r'(\d+) Hz', stderr)
            if sample_rate_match:
                info["sample_rate"] = int(sample_rate_match.group(1))

            # Parse channel layout (format: stereo, 5.1(side), etc.)
            channel_match = re.search(r'(stereo|mono|(\d+(\.\d+)? ch))', stderr)
            if channel_match:
                info["channels"] = channel_match.group(1)

            return info
        except Exception as e:
            return {"path": str(input_path), "exists": True, "error": str(e)}


# [SEMANTIC-END] ADAPTER:FFMPEG


# Module-level convenience function
def convert_audio(
    input_path: str | Path,
    output_path: str | Path,
    audio_profile: str = "raw",
) -> Path | List[Path]:
    """Convenience function to convert audio using FFmpeg.

    Args:
        input_path: Path to input media file
        output_path: Path to output WAV file
        audio_profile: Audio preprocessing profile

    Returns:
        Path to the converted audio file or list of paths for split-stereo
    """
    adapter = FFmpegAdapter()
    return adapter.convert_to_wav(input_path, output_path, audio_profile)