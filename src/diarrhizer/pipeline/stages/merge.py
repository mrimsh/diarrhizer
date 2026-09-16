"""Merge stage for combining ASR transcripts with speaker diarization."""

import bisect
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from diarrhizer.export.speakers import normalize_speaker_id
from diarrhizer.pipeline.cache import is_stale
from diarrhizer.utils import write_json_atomic

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import JobContext

logger = logging.getLogger(__name__)


# [SEMANTIC-BEGIN] STAGE:MERGE
# @purpose: Merge ASR transcripts with speaker diarization to create speaker-annotated segments
# @description: Consumes transcript.json and, if present, diarization.json, produces segments.json
#   with speaker labels. A missing diarization.json is treated as "no diarization data" (empty
#   diar_segments) rather than an error - assign_speakers already defaults every segment to
#   Speaker_00 in that case - so ASR-only pipelines (no diarize stage) still produce readable
#   merged/exported output instead of failing. Output segments are one-speaker-each and so may
#   outnumber the ASR segments they came from, since an ASR segment spanning a speaker change
#   is split at that boundary (MERGE:SPEAKER_TURNS).
# @inputs: artifacts/asr/transcript.json (required), artifacts/diar/diarization.json (optional)
# @outputs: artifacts/merged/segments.json
# @sideEffects: Reads JSON files, writes merged segments to disk,
#   logs progress via logging (INFO, extra={"stage": "merge"})
# @errors: FileNotFoundError if transcript.json is missing
# @see: STAGE:TRANSCRIBE, STAGE:DIARIZE, MERGE:ASSIGN_SPEAKERS, MERGE:SPEAKER_TURNS
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

        logger.info(f"[{self.NAME}] Merging transcripts with diarization", extra={"stage": self.NAME})

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
            logger.info(
                f"[{self.NAME}] No diarization found at {diar_input}; defaulting to Speaker_00",
                extra={"stage": self.NAME},
            )
            diar_segments = []

        # Extract segments and words from transcript
        transcript_segments = transcript_data.get("segments", [])
        transcript_words = transcript_data.get("words", [])

        min_turn_duration = job.config.min_turn_duration

        start_time = datetime.now()

        # Perform merge
        merged_segments = assign_speakers(
            transcript_segments=transcript_segments,
            transcript_words=transcript_words,
            diar_segments=diar_segments,
            min_turn_duration=min_turn_duration,
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
                "min_turn_duration": min_turn_duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
            },
        }

        # Write merged segments to JSON
        write_json_atomic(segments_output, output_data)

        logger.info(f"[{self.NAME}] Completed in {duration:.2f}s", extra={"stage": self.NAME})
        logger.info(
            f"[{self.NAME}] Segments: {len(merged_segments)} "
            f"(from {len(transcript_segments)} ASR segments)",
            extra={"stage": self.NAME},
        )
        logger.info(f"[{self.NAME}] Output: {segments_output}", extra={"stage": self.NAME})

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


# Default for PipelineConfig.min_turn_duration (--min-turn-duration): word runs
# shorter than this that sit between two runs of the same other speaker are
# treated as diarization jitter, not a real turn, and absorbed into their
# neighbours (see _absorb_jitter_runs). Kept deliberately small: a short
# backchannel ("да", "ага") is a genuine turn and must survive, so only blips
# below a fraction of a second are smoothed away by default. Raise it when
# diarization is noisy and shreds sentences; set 0 to split on every flip.
DEFAULT_MIN_TURN_DURATION = 0.4


# [SEMANTIC-BEGIN] MERGE:ASSIGN_SPEAKERS
# @purpose: Assign speaker labels to ASR segments based on overlap with diarization
# @description: For each segment/word (processed in chronological order), finds the
#   speaker with max overlap via a sweep over sorted, time-ordered queries
#   (see _DiarizationSweep) instead of rescanning all diarization segments per
#   query - matters for long calls with many words/diarization segments.
#   An ASR segment can straddle a speaker change (Whisper segments on pauses and
#   punctuation, not on who is talking), which used to produce one segment
#   labelled with the majority speaker while some of its words carried a
#   different speaker_id. Such segments are now split into one output segment
#   per speaker run (MERGE:SPEAKER_TURNS), so every emitted segment is
#   homogeneous: seg["speaker_id"] always equals the speaker_id of every word
#   in seg["words"]. Output segments can therefore outnumber input ASR segments.
# @inputs: transcript_segments, transcript_words, diar_segments
# @outputs: List of merged segments with speaker_id, one speaker per segment
# @sideEffects: None (pure function)
# @errors: None
# @see: STAGE:MERGE, MERGE:SPEAKER_TURNS
def assign_speakers(
    transcript_segments: list[dict],
    transcript_words: list[dict],
    diar_segments: list[dict],
    min_turn_duration: float = DEFAULT_MIN_TURN_DURATION,
) -> list[dict]:
    """Assign speaker labels to transcript segments based on diarization overlap.

    Algorithm:
    1. For each word within a transcript segment, find the diarization speaker
       with maximum time overlap
    2. Split the segment into consecutive runs of same-speaker words, emitting
       one output segment per run (see _split_into_speaker_turns)
    3. For a segment with no word timestamps, fall back to matching the whole
       segment against diarization and emit it unsplit

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
        min_turn_duration: Shortest word run, in seconds, that may stand as its
            own speaker turn; shorter runs between two runs of the same other
            speaker are absorbed as diarization jitter. 0 splits on every flip.

    Returns:
        List of merged segments with speaker_id and optional word-level data.
        May be longer than transcript_segments when a segment spans a speaker
        change.
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

        words = word_segment_map.get(seg_idx)

        # No word timestamps for this segment: nothing to split on, so match
        # the segment as a whole against diarization as before.
        if not words:
            merged_segments.append({
                "start": seg_start,
                "end": seg_end,
                "speaker_id": speaker_lookup.find(seg_start, seg_end),
                "text": seg_text,
            })
            continue

        merged_words = []
        for word in words:
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)
            merged_words.append({
                "start": word_start,
                "end": word_end,
                "word": word.get("word", ""),
                "speaker_id": speaker_lookup.find(word_start, word_end),
            })

        merged_segments.extend(
            _split_into_speaker_turns(
                seg_start, seg_end, seg_text, merged_words, min_turn_duration
            )
        )

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
        speaker = normalize_speaker_id(diar_seg.get("speaker", default_speaker))

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
            speaker = normalize_speaker_id(diar_seg.get("speaker", default_speaker))

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
                self._before_speaker = normalize_speaker_id(
                    segments[self._left].get("speaker", self.DEFAULT_SPEAKER)
                )
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
                best_speaker = normalize_speaker_id(seg.get("speaker", self.DEFAULT_SPEAKER))
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
            return normalize_speaker_id(segments[right].get("speaker", self.DEFAULT_SPEAKER))
        return self.DEFAULT_SPEAKER


# [SEMANTIC-END] MERGE:ASSIGN_SPEAKERS


# [SEMANTIC-BEGIN] MERGE:SPEAKER_TURNS
# @purpose: Split one ASR segment into one output segment per speaker turn
# @description: Whisper cuts segments on pauses and punctuation, not on who is
#   talking, so a single ASR segment can span a speaker change. Left alone it
#   gets one speaker label (whoever overlaps it most) while its words carry
#   two, which reads as a misattribution: the losing speaker's words appear
#   under the winner's name. This splits the segment at the boundaries between
#   consecutive same-speaker word runs.
#   Text: the original ASR text is kept verbatim when there's nothing to split.
#   Only when a segment really is split is text rebuilt per run by joining word
#   tokens, since the ASR text can't be sliced reliably.
#   Time: the original segment's start and end are preserved on the first and
#   last run so a split never shrinks the transcript's coverage; interior
#   boundaries use word timings, which leaves honest gaps at pauses between turns.
# @inputs: segment start/end/text, its merged words (each with speaker_id)
# @outputs: List of segments, each with a single speaker_id matching all its words
# @sideEffects: None (relabels only the word dicts it was handed, which
#   assign_speakers builds fresh per segment)
# @errors: None
# @see: MERGE:ASSIGN_SPEAKERS, STAGE:MERGE
def _split_into_speaker_turns(
    seg_start: float,
    seg_end: float,
    seg_text: str,
    merged_words: list[dict],
    min_turn_duration: float = DEFAULT_MIN_TURN_DURATION,
) -> list[dict]:
    """Split a segment into one segment per consecutive same-speaker word run.

    Args:
        seg_start: Original segment start time in seconds
        seg_end: Original segment end time in seconds
        seg_text: Original ASR text for the whole segment
        merged_words: The segment's words, each already carrying a speaker_id
        min_turn_duration: Jitter threshold in seconds (see _absorb_jitter_runs)

    Returns:
        One segment per speaker turn, in chronological order. A segment whose
        words all share one speaker comes back as a single segment with its
        original text and timings untouched.
    """
    runs = _absorb_jitter_runs(_speaker_runs(merged_words), min_turn_duration)

    # Homogeneous segment: keep the ASR text and timings exactly as they were.
    # The speaker comes from the words rather than from a separate whole-segment
    # overlap query, so the segment label can't disagree with its own words.
    if len(runs) == 1:
        return [{
            "start": seg_start,
            "end": seg_end,
            "speaker_id": runs[0][0]["speaker_id"],
            "text": seg_text,
            "words": merged_words,
        }]

    last = len(runs) - 1
    turns = []
    for i, run in enumerate(runs):
        start = seg_start if i == 0 else run[0]["start"]
        # Words can extend past the ASR segment's own end; keep whichever is later.
        end = max(seg_end, run[-1]["end"]) if i == last else run[-1]["end"]
        turns.append({
            "start": start,
            "end": max(end, start),
            "speaker_id": run[0]["speaker_id"],
            "text": " ".join(w["word"].strip() for w in run if w["word"].strip()),
            "words": run,
        })
    return turns


def _speaker_runs(merged_words: list[dict]) -> list[list[dict]]:
    """Group words into maximal runs of consecutive same-speaker words."""
    runs: list[list[dict]] = []
    for word in merged_words:
        if runs and runs[-1][0]["speaker_id"] == word["speaker_id"]:
            runs[-1].append(word)
        else:
            runs.append([word])
    return runs


def _absorb_jitter_runs(
    runs: list[list[dict]],
    min_turn_duration: float = DEFAULT_MIN_TURN_DURATION,
) -> list[list[dict]]:
    """Fold runs shorter than min_turn_duration that are sandwiched between two
    runs of the same other speaker back into that speaker.

    A word near a diarization boundary can pick up the wrong speaker on max
    overlap, and splitting on every such blip would shred a sentence into
    fragments. Only interior runs are considered, and only when both neighbours
    agree on who is really talking - a run at either edge of the segment is
    never absorbed, because there's no evidence on both sides that it's jitter
    rather than the start or end of a genuine turn. Absorbed words are
    relabelled to the surrounding speaker so seg["words"] can't contradict the
    segment they end up in.
    """
    changed = True
    while changed and len(runs) >= 3:
        changed = False
        for i in range(1, len(runs) - 1):
            previous, run, following = runs[i - 1], runs[i], runs[i + 1]
            speaker = previous[0]["speaker_id"]
            if speaker != following[0]["speaker_id"]:
                continue
            if run[-1]["end"] - run[0]["start"] >= min_turn_duration:
                continue
            for word in run:
                word["speaker_id"] = speaker
            runs[i - 1:i + 2] = [previous + run + following]
            changed = True
            break
    return runs
# [SEMANTIC-END] MERGE:SPEAKER_TURNS
