"""FFmpeg adapter for audio normalization."""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional


# [SEMANTIC-BEGIN] ADAPTER:FFMPEG
# @purpose: Wrap FFmpeg calls for audio normalization and format conversion
# @description: Provides a clean interface to FFmpeg for converting media files to WAV mono 16kHz
# @inputs: input_path (str or Path), output_path (str or Path)
# @outputs: Path to converted audio file
# @sideEffects: Executes FFmpeg subprocess, creates output file on disk
# @errors: RuntimeError if FFmpeg is not found or conversion fails
# @see: STAGE:CONVERT
class FFmpegAdapter:
    """Adapter for FFmpeg audio conversion operations."""

    # Target audio format: WAV, mono, 16kHz
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1  # Mono

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
    ) -> Path:
        """Convert input media file to WAV mono 16kHz.

        Args:
            input_path: Path to input media file (any format FFmpeg supports)
            output_path: Path to output WAV file

        Returns:
            Path to the converted audio file

        Raises:
            RuntimeError: If FFmpeg conversion fails
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build FFmpeg command
        cmd = [
            self._ffmpeg_path,
            "-y",  # Overwrite output file if exists
            "-i", str(input_path),
            "-ac", str(self.TARGET_CHANNELS),  # Mono
            "-ar", str(self.TARGET_SAMPLE_RATE),  # 16kHz
            "-acodec", "pcm_s16le",  # 16-bit PCM
            str(output_path),
        ]

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
) -> Path:
    """Convenience function to convert audio using FFmpeg.

    Args:
        input_path: Path to input media file
        output_path: Path to output WAV file

    Returns:
        Path to the converted audio file
    """
    adapter = FFmpegAdapter()
    return adapter.convert_to_wav(input_path, output_path)
