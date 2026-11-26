"""Manual transcript snapshots and edits to the merge stage's segment dictionaries."""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from diarrhizer.artifacts import bundle_valid, digest, job_write_lock
from diarrhizer.export.speakers import normalize_speaker_id
from diarrhizer.utils import read_json, read_object, write_json_atomic

SOURCE = Path("merged/segments.json")
MANUAL = Path("edits/transcript.json")


# [SEMANTIC-BEGIN] READ:MERGED_SEGMENTS
# @purpose: Read saved results for GUI display and export, including legacy reversed intervals
# @description: Finite non-negative end < start collapses to start in memory, preserving text,
#   speakers and other fields with an alignment_warning containing the original bounds.
#   Word times on a recovered segment become unknown; the source file is never rewritten.
#   Manual snapshots use the same validator with explicit bounds and no legacy recovery.
# @inputs: path to merged/segments.json
# @outputs: validated segment list with warnings on recovered intervals
# @sideEffects: Reads JSON and logs recovery warnings; no filesystem writes
# @errors: ValueError for malformed data (timestamp errors include index/bounds), OSError
# @see: CONFIG:RESUME, GUI:RESULT_SCREEN, STAGE:EXPORT, MERGE:ASSIGN_SPEAKERS
def read_segments(path: Path) -> list[dict]:
    """Validate saved segments, recovering legacy reversed intervals only in memory."""
    return validate_segments(read_object(path).get("segments", []), path)


def validate_segments(segments, path: Path, *, manual: bool = False) -> list[dict]:
    if not isinstance(segments, list):
        raise ValueError(f"{path}: segments must be a list")
    for index, seg in enumerate(segments):
        if not isinstance(seg, dict) or not isinstance(seg.get("text", ""), str):
            raise ValueError(f"{path}: each segment must be an object with text")
        default = None if manual else 0
        start, end = seg.get("start", default), seg.get("end", default)
        if not (type(start) in (int, float) and type(end) in (int, float)
                and math.isfinite(start) and math.isfinite(end) and start >= 0 and end >= 0):
            raise ValueError(
                f"{path}: invalid segment timestamps (segment {index}, start={start!r}, end={end!r})"
            )
        if not isinstance(seg.get("speaker_id", "Speaker_00"), str):
            raise ValueError(f"{path}: speaker_id must be text")
        if end < start:
            if manual:
                raise ValueError(f"{path}: segment {index}: end {end} precedes start {start}")
            warning = (f"Reversed merged segment bounds (start={start!r}, end={end!r}); "
                       "end set to start, timing uncertain; original text preserved")
            previous = seg.get("alignment_warning")
            if isinstance(previous, str) and previous:
                warning = f"{previous}; {warning}"
            seg = {**seg, "end": start, "alignment_warning": warning}
            if isinstance(seg.get("words"), list):
                seg["words"] = [{**word, "start": None, "end": None} if isinstance(word, dict) else word
                                for word in seg["words"]]
            segments[index] = seg
            logging.getLogger(__name__).warning("%s: segment %d: %s", path, index, warning)
    return segments
# [SEMANTIC-END] READ:MERGED_SEGMENTS


def _hash(value) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


# [SEMANTIC-BEGIN] READ:CURRENT_TRANSCRIPT
# @purpose: Select one complete segment list for results, export and prompt data.
# @description: Bind a snapshot to segment content independently of run timestamps and JSON formatting.
#   A changed source requires an explicit choice before any consumer can use the transcript.
# @sideEffects: Reads merged and optional manual JSON. Never modifies the machine result.
# @errors: ValueError for corrupt data or an unresolved conflict.
# @see: READ:MERGED_SEGMENTS, TRANSCRIPT:REVISION, STAGE:EXPORT
@dataclass
class Transcript:
    source: list[dict]
    source_sha256: str
    manual: dict | None

    @property
    def conflict(self) -> bool:
        return self.manual is not None and self.source_sha256 != self.manual.get(
            "accepted_source_sha256", self.manual["base_sha256"]
        )

    @property
    def segments(self) -> list[dict]:
        if self.conflict:
            raise ValueError("Исходные реплики изменились. Выберите ручную или новую версию на экране результатов.")
        return self.manual["segments"] if self.manual is not None else self.source

    @property
    def current_sha256(self) -> str:
        return _hash(self.segments) if self.manual is not None else self.source_sha256


def load_transcript(job: Path, *, merged: dict | None = None) -> Transcript:
    path = job / SOURCE
    if not path.exists():
        raise FileNotFoundError(f"Segments not found: {path}")
    if merged is None:
        source = read_segments(path)
    elif isinstance(merged, dict):
        source = validate_segments(merged.get("segments", []), path)
    else:
        raise ValueError(f"{path}: expected a JSON object")
    path = job / MANUAL
    manual = read_object(path) if path.exists() else None
    if manual is not None:
        if manual.get("schema_version") != 1 or not isinstance(manual.get("base_sha256"), str):
            raise ValueError(f"{path}: invalid manual transcript")
        validate_segments(manual.get("segments"), path, manual=True)
    return Transcript(source, _hash(source), manual)


def exports_current(job: Path, transcript: Transcript | None = None) -> bool:
    """Validate the published bundle and its input. Legacy Markdown-only jobs remain readable."""
    folder = job / "export"
    if not bundle_valid(folder, "manifest.json"):
        return False
    manifest = read_json(folder / "manifest.json") or {}
    metadata = manifest.get("metadata", {})
    identity = metadata.get("transcript_sha256") if isinstance(metadata, dict) else None
    if not identity or (transcript is None and not (job / SOURCE).exists()):
        # A failed first manual export may leave a legacy bundle, even after the user archives the edit.
        manual_path = job / MANUAL
        return not manual_path.exists() and not any(manual_path.parent.glob("transcript-*.json"))
    if transcript is None:
        transcript = load_transcript(job)
    return (not transcript.conflict and transcript.current_sha256 == identity
            and metadata.get("merged_segments_sha256", transcript.source_sha256) == transcript.source_sha256)
# [SEMANTIC-END] READ:CURRENT_TRANSCRIPT


# [SEMANTIC-BEGIN] TRANSCRIPT:REVISION
# @purpose: Persist or explicitly resolve a complete manual revision without overwriting upstream files.
# @description: Check the opened revision under the job lock. If merge changed during editing,
#   keep the draft bound to the opened source so the next read exposes the conflict.
#   Selecting the new source archives the manual file. Keeping edits acknowledges the current source hash.
# @sideEffects: Atomic JSON write or rename inside edits/. The caller holds the lock through export.
# @errors: ValueError for invalid intervals; RuntimeError if another editor saved first.
# @see: READ:CURRENT_TRANSCRIPT, ARTIFACTS:WRITE_LOCK, GUI:RESULT_SCREEN
def save_revision(job: Path, opened: Transcript, segments: list[dict]) -> Transcript:
    with job_write_lock(job):
        current = load_transcript(job)
        if current.manual != opened.manual or opened.conflict:
            raise RuntimeError("Ручная версия изменилась. Откройте результат заново перед сохранением.")
        record = {**(opened.manual or {"schema_version": 1, "base_sha256": opened.source_sha256}),
                  "segments": validate_segments(segments, job / MANUAL, manual=True)}
        if record != current.manual:
            write_json_atomic(job / MANUAL, record)
        return Transcript(current.source, current.source_sha256, record)


def resolve_conflict(job: Path, opened: Transcript, *, keep_manual: bool) -> Transcript:
    with job_write_lock(job):
        current = load_transcript(job)
        if not current.conflict or current != opened:
            raise RuntimeError("Версии изменились. Откройте результат заново перед выбором.")
        path = job / MANUAL
        if keep_manual:
            current.manual = {**current.manual, "accepted_source_sha256": current.source_sha256}
            write_json_atomic(path, current.manual)
        else:
            path.replace(path.with_name(f"transcript-{_hash(current.manual)}.json"))
            current.manual = None
        return current
# [SEMANTIC-END] TRANSCRIPT:REVISION


# [SEMANTIC-BEGIN] TRANSCRIPT:EDIT
# @purpose: Edit individual segments while preserving unaffected segments and real overlapping speech.
# @description: Text or boundary changes invalidate word alignments. A speaker-only change keeps word
#   times and updates their IDs. Splits retain an explicit paragraph break in readable exports.
# @inputs: segment dictionaries, cursor offset and time, or two adjacent segments and a speaker
# @outputs: replacement dictionaries; input segments are never mutated
# @errors: ValueError for an empty split, invalid cut time or missing speaker choice
# @see: READ:MERGED_SEGMENTS, EXPORT:SPEAKER_TURNS, GUI:TRANSCRIPT_EDITOR
def speaker_id(segment: dict) -> str:
    return normalize_speaker_id(segment.get("speaker_id") or "Speaker_00")


def change_text(segment: dict, text: str) -> dict:
    updated = {**segment, "text": text}
    updated.pop("words", None)
    return updated


def change_speaker(segment: dict, speaker: str) -> dict:
    updated = {**segment, "speaker_id": speaker}
    if segment.get("words"):
        updated["words"] = [{**word, "speaker_id": speaker} for word in segment["words"]]
    return updated


def split_segment(segment: dict, cursor: int, at: float) -> tuple[dict, dict]:
    text = segment.get("text", "")
    left, right = text[:cursor].rstrip(), text[cursor:].lstrip()
    if not 0 < cursor < len(text) or not left.strip() or not right.strip():
        raise ValueError("Поставьте курсор между двумя частями текста реплики.")
    if not segment["start"] < at < segment["end"]:
        raise ValueError("Момент разреза должен быть строго внутри реплики.")
    return ({**change_text(segment, left), "end": at},
            {**change_text(segment, right), "start": at, "manual_break_before": True})


def merge_segments(first: dict, second: dict, speaker: str | None = None) -> dict:
    first, second = sorted((first, second), key=lambda seg: seg["start"])
    if speaker is None and speaker_id(first) != speaker_id(second):
        raise ValueError("Выберите спикера объединённой реплики.")
    text = " ".join(part for seg in (first, second) if (part := seg.get("text", "").strip()))
    merged = {**change_text(first, text), "end": max(first["end"], second["end"]),
              "speaker_id": speaker or speaker_id(first)}
    for key in ("channel", "source_track"):
        if first.get(key) != second.get(key):
            merged.pop(key, None)
    if second.get("alignment_warning"):
        merged["alignment_warning"] = " / ".join(dict.fromkeys(
            seg["alignment_warning"] for seg in (first, second) if seg.get("alignment_warning")
        ))
    return merged
# [SEMANTIC-END] TRANSCRIPT:EDIT
