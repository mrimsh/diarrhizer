"""Merge stage for combining ASR transcripts with speaker diarization."""

import bisect
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from diarrhizer.pipeline.cache import is_stale
from diarrhizer.utils import write_json_atomic

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import JobContext


# [SEMANTIC-BEGIN] STAGE:MERGE
# @purpose: Merge ASR transcripts with speaker diarization to create speaker-annotated segments
# @description: Consumes transcript.json and, if present, diarization.json, produces segments.json
#   with speaker labels. A missing diarization.json is treated as "no diarization data" (empty
#   diar_segments) rather than an error - assign_speakers already defaults every segment to
#   Speaker_00 in that case - so ASR-only pipelines (no diarize stage) still produce readable
#   merged/exported output instead of failing.
# @inputs: artifacts/asr/transcript.json (required), artifacts/diar/diarization.json (optional)
# @outputs: artifacts/merged/segments.json
# @sideEffects: Reads JSON files, writes merged segments to disk
# @errors: FileNotFoundError if transcript.json is missing
# @see: STAGE:TRANSCRIBE, STAGE:DIARIZE, MERGE:ASSIGN_SPEAKERS
class MergeStage:
    """Stage for merging ASR transcripts with speaker diarization."""

    # Stage name for identification
    NAME = "merge"

    # Output paths relative to job directory
    MERGE_DIR = "merged"
    SEGMENTS_JSON = "merged/segments.json"

    # Input artifact paths
    INPUT_TRANSCRIPT = "asr/transcript.json"
    INPUT_DIARIZATION = "diar/diarization.json"

    def run(self, job: "JobContext") -> dict:
        """Run the merge stage.

        Args:
            job: Job context containing input path and configuration

        Returns:
            Dictionary with stage output paths and metadata
        """
        job_dir = job.job_dir

        # Build input paths
        transcript_input = job_dir / self.INPUT_TRANSCRIPT
        diar_input = job_dir / self.INPUT_DIARIZATION

        # Build output path
        segments_output = job_dir / self.SEGMENTS_JSON

        print(f"[{self.NAME}] Merging transcripts with diarization")

        # Check if inputs exist
        if not transcript_input.exists():
            raise FileNotFoundError(
                f"Transcript not found: {transcript_input}. "
                "Please run the transcribe stage first."
            )

        # Ensure output directory exists
        segments_output.parent.mkdir(parents=True, exist_ok=True)

        # Load input artifacts
        with open(transcript_input, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        # Diarization is optional: if it hasn't been run, treat it as no
        # diarization data (assign_speakers already defaults every segment to
        # Speaker_00 in that case) instead of failing the whole pipeline.
        if diar_input.exists():
            with open(diar_input, "r", encoding="utf-8") as f:
                diar_data = json.load(f)
            diar_segments = diar_data.get("segments", [])
        else:
            print(f"[{self.NAME}] No diarization found at {diar_input}; defaulting to Speaker_00")
            diar_segments = []

        # Extract segments and words from transcript
        transcript_segments = transcript_data.get("segments", [])
        transcript_words = transcript_data.get("words", [])

        start_time = datetime.now()

        # Perform merge
        merged_segments = assign_speakers(
            transcript_segments=transcript_segments,
            transcript_words=transcript_words,
            diar_segments=diar_segments,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Prepare output data
        output_data = {
            "stage": self.NAME,
            "segments": merged_segments,
            "num_segments": len(merged_segments),
            "metadata": {
                "input_transcript": str(transcript_input),
                "input_diarization": str(diar_input),
                "output_path": str(segments_output),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
            },
        }

        # Write merged segments to JSON
        write_json_atomic(segments_output, output_data)

        print(f"[{self.NAME}] Completed in {duration:.2f}s")
        print(f"[{self.NAME}] Segments: {len(merged_segments)}")
        print(f"[{self.NAME}] Output: {segments_output}")

        return {
            "stage": self.NAME,
            "status": "completed",
            "output_path": str(segments_output),
            "num_segments": len(merged_segments),
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
            "transcript": job_dir / self.INPUT_TRANSCRIPT,
            "diarization": job_dir / self.INPUT_DIARIZATION,
            "segments": job_dir / self.SEGMENTS_JSON,
        }

    def get_output_paths(self, job_dir: Path) -> dict:
        """Get only the artifact paths this stage produces (not its inputs).

        Args:
            job_dir: Job directory path

        Returns:
            Dictionary of output artifact name to path
        """
        return {"segments": job_dir / self.SEGMENTS_JSON}

    def is_cache_valid(self, job_dir: Path) -> bool:
        """Check if stage output exists and is up to date relative to its inputs.

        Args:
            job_dir: Job directory path

        Returns:
            True if output exists and is valid
        """
        artifacts = self.get_artifact_paths(job_dir)
        return not is_stale(
            outputs=list(self.get_output_paths(job_dir).values()),
            inputs=[artifacts["transcript"], artifacts["diarization"]],
        )


# [SEMANTIC-END] STAGE:MERGE


# [SEMANTIC-BEGIN] MERGE:ASSIGN_SPEAKERS
# @purpose: Assign speaker labels to ASR segments based on overlap with diarization
# @description: For each segment/word (processed in chronological order), finds the
#   speaker with max overlap via a sweep over sorted, time-ordered queries
#   (see _DiarizationSweep) instead of rescanning all diarization segments per
#   query - matters for long calls with many words/diarization segments
# @inputs: transcript_segments, transcript_words, diar_segments
# @outputs: List of merged segments with speaker_id
# @sideEffects: None (pure function)
# @errors: None
# @see: STAGE:MERGE
def assign_speakers(
    transcript_segments: list[dict],
    transcript_words: list[dict],
    diar_segments: list[dict],
) -> list[dict]:
    """Assign speaker labels to transcript segments based on diarization overlap.

    Algorithm:
    1. For each transcript segment, find the diarization segment with maximum time overlap
    2. Assign that speaker to the entire segment
    3. For each word within a segment, find the speaker with maximum overlap

    Assumptions:
    - Diarization segments may overlap with each other (pyannote allows overlapping speakers)
    - If no overlap exists, use the closest diarization segment by time
    - If no diarization data exists, default to "Speaker_00"
    - Word-level timestamps are optional - only include if available in transcript

    Edge cases:
    - Empty transcript: return empty list
    - Empty diarization: assign all to "Speaker_00"
    - Gaps in diarization: assign based on closest segment
    - Overlapping speakers in diarization: choose speaker with most overlap

    Args:
        transcript_segments: List of ASR segments with start/end/text
        transcript_words: List of words with start/end/word (optional)
        diar_segments: List of diarization segments with start/end/speaker

    Returns:
        List of merged segments with speaker_id and optional word-level data
    """
    # Handle empty inputs
    if not transcript_segments:
        return []

    # Default speaker if no diarization
    default_speaker = "Speaker_00"

    # If no diarization, assign all to default speaker
    if not diar_segments:
        return [
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker_id": default_speaker,
                "text": seg.get("text", ""),
            }
            for seg in transcript_segments
        ]

    # Build word index for faster lookup
    # Words are grouped by which transcript segment they belong to.
    #
    # Both lists are chronologically ordered and transcript segments don't
    # overlap each other, so a single forward-moving pointer into
    # transcript_segments suffices - O(segments + words) total instead of
    # rescanning all segments for every word.
    word_segment_map: dict[int, list[dict]] = {}
    if transcript_words:
        seg_idx = 0
        num_segments = len(transcript_segments)
        for word in transcript_words:
            word_start = word.get("start", 0)
            # Advance past segments that ended before this (or any later,
            # since words are chronological) word could start.
            while (
                seg_idx < num_segments
                and transcript_segments[seg_idx].get("end", 0) <= word_start
            ):
                seg_idx += 1
            if seg_idx >= num_segments:
                break
            # Word belongs to segment if it starts within the segment
            # (inclusive start, exclusive end); otherwise it falls in a gap
            # between segments and is left unmatched.
            if transcript_segments[seg_idx].get("start", 0) <= word_start:
                word_segment_map.setdefault(seg_idx, []).append(word)

    # Process each transcript segment. Segments, and each segment's words, are
    # visited in chronological order, so a single _DiarizationSweep instance
    # can answer every query in amortized O(1) instead of rescanning all of
    # diar_segments per segment/word (see _DiarizationSweep for why this is
    # safe even though diarization segments may overlap each other).
    speaker_lookup = _DiarizationSweep(diar_segments)
    merged_segments = []

    for seg_idx, seg in enumerate(transcript_segments):
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg_text = seg.get("text", "")

        speaker_id = speaker_lookup.find(seg_start, seg_end)

        # Build merged segment
        merged_seg = {
            "start": seg_start,
            "end": seg_end,
            "speaker_id": speaker_id,
            "text": seg_text,
        }

        # Add word-level data if available for this segment
        if seg_idx in word_segment_map:
            words = word_segment_map[seg_idx]
            merged_words = []

            for word in words:
                word_start = word.get("start", 0)
                word_end = word.get("end", 0)
                word_text = word.get("word", "")

                word_speaker = speaker_lookup.find(word_start, word_end)

                merged_words.append({
                    "start": word_start,
                    "end": word_end,
                    "word": word_text,
                    "speaker_id": word_speaker,
                })

            merged_seg["words"] = merged_words

        merged_segments.append(merged_seg)

    return merged_segments


def _find_overlapping_speaker(
    start: float,
    end: float,
    diar_segments: list[dict],
) -> str:
    """Find the speaker with maximum overlap for a given time range.

    Args:
        start: Start time in seconds
        end: End time in seconds
        diar_segments: List of diarization segments

    Returns:
        Speaker ID with maximum overlap, or default if no overlap found
    """
    default_speaker = "Speaker_00"

    if not diar_segments:
        return default_speaker

    max_overlap = 0.0
    best_speaker = default_speaker

    for diar_seg in diar_segments:
        diar_start = diar_seg.get("start", 0)
        diar_end = diar_seg.get("end", 0)
        speaker = diar_seg.get("speaker", default_speaker)

        # Calculate overlap
        overlap_start = max(start, diar_start)
        overlap_end = min(end, diar_end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = speaker

    # If no overlap found, find closest segment by time
    if max_overlap == 0:
        min_distance = float("inf")
        for diar_seg in diar_segments:
            diar_start = diar_seg.get("start", 0)
            diar_end = diar_seg.get("end", 0)
            speaker = diar_seg.get("speaker", default_speaker)

            # Calculate distance from our segment to this diar segment
            if diar_end < start:
                distance = start - diar_end
            elif diar_start > end:
                distance = diar_start - end
            else:
                distance = 0  # Overlaps

            if distance < min_distance:
                min_distance = distance
                best_speaker = speaker

    return best_speaker


class _DiarizationSweep:
    """Faster replacement for repeated _find_overlapping_speaker calls against
    the same diar_segments, when queries are issued in chronological
    (non-decreasing start time) order - true for how assign_speakers walks
    segments, then each segment's words.

    Internally sorts diar_segments by start once. The *left* edge of the
    candidate window is tracked incrementally across calls in amortized O(1):
    once a segment's end falls at/before some query's start, it can never
    overlap that or any later query (queries only move forward), so it's
    retired for good. The *right* edge cannot be tracked the same way -
    queries move forward in start time but not necessarily in end time (an
    earlier query can have a later end than a later query with a smaller
    span), so it's recomputed with a binary search every call. That keeps
    each call O(log diar_segments) plus the size of the (usually tiny)
    candidate window, instead of the O(diar_segments) full rescan
    _find_overlapping_speaker does.

    When nothing in the window overlaps, the closest neighbor is used
    instead: the best already-ended segment seen so far (tracked
    incrementally as the left edge advances) versus the next segment to
    start (found via the same binary search used for the window's right
    edge) - equivalent to _find_overlapping_speaker's fallback, which
    considers every segment, because sorting by start means neither
    candidate can be beaten by a segment the window/search didn't consider.
    """

    DEFAULT_SPEAKER = "Speaker_00"

    def __init__(self, diar_segments: list[dict]) -> None:
        # Keep each segment's original position so overlap/gap ties can be
        # broken the same way _find_overlapping_speaker breaks them - first
        # match in the caller's original order wins - even though this class
        # scans in a different (start-sorted) order. Real diarization output
        # commonly ties on overlap: any query fully inside two or more
        # concurrently-overlapping speakers' segments overlaps all of them by
        # exactly the query's own duration.
        indexed = sorted(enumerate(diar_segments), key=lambda pair: pair[1].get("start", 0))
        self._segments = [seg for _, seg in indexed]
        self._orig_index = [idx for idx, _ in indexed]
        self._starts = [seg.get("start", 0) for seg in self._segments]
        self._left = 0
        self._before_end: float | None = None
        self._before_speaker: str | None = None
        self._before_orig_index: int = -1

    def find(self, start: float, end: float) -> str:
        """Find the speaker with maximum overlap for [start, end).

        Args:
            start: Query start time in seconds (must be >= every previous
                call's start within this sweep instance)
            end: Query end time in seconds

        Returns:
            Speaker ID with maximum overlap, or the closest one by time if
            nothing overlaps, or the default speaker if there's no
            diarization data at all.
        """
        segments = self._segments
        n = len(segments)
        if n == 0:
            return self.DEFAULT_SPEAKER

        # Retire segments that ended at/before this query's start - they can
        # never overlap this or any later query. Remember the best (latest
        # ending) one retired so far in case we need a "closest before" fallback.
        while self._left < n and segments[self._left].get("end", 0) <= start:
            seg_end = segments[self._left].get("end", 0)
            orig_idx = self._orig_index[self._left]
            if (
                self._before_end is None
                or seg_end > self._before_end
                or (seg_end == self._before_end and orig_idx < self._before_orig_index)
            ):
                self._before_end = seg_end
                self._before_speaker = segments[self._left].get("speaker", self.DEFAULT_SPEAKER)
                self._before_orig_index = orig_idx
            self._left += 1

        # Segments sorted by start form a contiguous run with start < end
        # starting at _left; binary-search its upper bound fresh each call
        # (see class docstring for why this can't be tracked incrementally).
        right = bisect.bisect_left(self._starts, end, lo=self._left)

        best_speaker = self.DEFAULT_SPEAKER
        max_overlap = 0.0
        best_orig_index = -1
        for i in range(self._left, right):
            seg = segments[i]
            overlap = max(0.0, min(end, seg.get("end", 0)) - max(start, seg.get("start", 0)))
            orig_idx = self._orig_index[i]
            if overlap > max_overlap or (overlap == max_overlap and overlap > 0 and orig_idx < best_orig_index):
                max_overlap = overlap
                best_speaker = seg.get("speaker", self.DEFAULT_SPEAKER)
                best_orig_index = orig_idx

        if max_overlap > 0:
            return best_speaker

        # No overlap anywhere: fall back to whichever neighbor is closer in
        # time - the last segment that already ended, or the next one to start.
        before_gap = start - self._before_end if self._before_end is not None else float("inf")
        after_gap = segments[right].get("start", 0) - end if right < n else float("inf")

        if before_gap <= after_gap and self._before_speaker is not None:
            return self._before_speaker
        if right < n:
            return segments[right].get("speaker", self.DEFAULT_SPEAKER)
        return self.DEFAULT_SPEAKER


# [SEMANTIC-END] MERGE:ASSIGN_SPEAKERS
