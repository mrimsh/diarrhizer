"""Which files count as media inputs, and expanding folders into them."""

from pathlib import Path
from typing import Iterable

# Everything FFmpeg decodes that call recorders, messengers and meeting tools
# commonly produce: phone recorders (.amr/.3gp/.m4a), WhatsApp/Telegram voice
# (.opus/.ogg/.oga), meeting recordings (.mp4/.mkv/.webm/.mov).
MEDIA_EXTENSIONS = (
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma",
    ".amr", ".3gp", ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi",
)


def is_media_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


# [SEMANTIC-BEGIN] INPUT:EXPAND_MEDIA
# @purpose: Turn user-supplied paths (files and folders) into the list of media files to process
# @description: Files are kept as given, whatever their extension - the user picked them explicitly and
#   FFmpeg decides what it can read. A folder contributes the media files directly inside it
#   (MEDIA_EXTENSIONS, not recursive), sorted by name. Missing paths are kept so the pipeline reports
#   them as "not found" per file instead of silently dropping them. Duplicates are removed, order kept.
# @inputs: paths (str/Path, files or folders)
# @outputs: list[Path]
# @sideEffects: Filesystem stat/listdir only
# @errors: None
# @see: CLI:RUN, PIPELINE:RUNNER
def expand_media_paths(paths: Iterable[str | Path]) -> list[Path]:
    result: list[Path] = []
    seen: set = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(
                (p for p in path.iterdir() if p.is_file() and is_media_file(p)),
                key=lambda p: p.name.lower(),
            )
        else:
            candidates = [path]
        for candidate in candidates:
            key = str(candidate.resolve()).lower()
            if key not in seen:
                seen.add(key)
                result.append(candidate)
    return result
# [SEMANTIC-END] INPUT:EXPAND_MEDIA
