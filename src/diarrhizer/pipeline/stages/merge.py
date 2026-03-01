"""Merge stage for combining ASR transcripts with speaker diarization."""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diarrhizer.pipeline.runner import JobContext


# [SEMANTIC-BEGIN] STAGE:MERGE
# @purpose: Merge ASR transcripts with speaker diarization to create speaker-annotated segments
# @description: Consumes transcript.json and diarization.json, produces segments.json with speaker labels
# @inputs: artifacts/asr/transcript.json, artifacts/diar/diarization.json
# @outputs: artifacts/merged/segments.json
# @sideEffects: Reads JSON files, writes merged segments to disk
# @errors: FileNotFoundError if input artifacts missing
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

        if not diar_input.exists():
            raise FileNotFoundError(
                f"Diarization not found: {diar_input}. "
                "Please run the diarize stage first."
            )

        # Check if output already exists (idempotency)
        if segments_output.exists():
            print(f"[{self.NAME}] Skipping - output already exists")
            return {
                "stage": self.NAME,
                "status": "skipped",
                "output_path": str(segments_output),
            }

        # Ensure output directory exists
        segments_output.parent.mkdir(parents=True, exist_ok=True)

        # Load input artifacts
        with open(transcript_input, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        with open(diar_input, "r", encoding="utf-8") as f:
            diar_data = json.load(f)

        # Extract segments and words from transcript
        transcript_segments = transcript_data.get("segments", [])
        transcript_words = transcript_data.get("words", [])

        # Extract diarization segments
        diar_segments = diar_data.get("segments", [])

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
        with open(segments_output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

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

    def is_cache_valid(self, job_dir: Path) -> bool:
        """Check if stage output exists and is valid.

        Args:
            job_dir: Job directory path

        Returns:
            True if output exists and is valid
        """
        artifacts = self.get_artifact_paths(job_dir)
        return artifacts["segments"].exists()


# [SEMANTIC-END] STAGE:MERGE


# [SEMANTIC-BEGIN] MERGE:ASSIGN_SPEAKERS
# @purpose: Assign speaker labels to ASR segments based on overlap with diarization
# @description: Simple overlap-based algorithm - for each segment/word, find speaker with max overlap
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
    # Words are grouped by which transcript segment they belong to
    word_segment_map: dict[int, list[dict]] = {}
    if transcript_words:
        # Map each word to its containing segment
        for word_idx, word in enumerate(transcript_words):
            word_start = word.get("start", 0)
            # Find the containing segment (word must start within the segment)
            for seg_idx, seg in enumerate(transcript_segments):
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                # Word belongs to segment if it starts within the segment
                # Use inclusive start, exclusive end for cleaner boundary handling
                if seg_start <= word_start < seg_end:
                    if seg_idx not in word_segment_map:
                        word_segment_map[seg_idx] = []
                    word_segment_map[seg_idx].append(word)
                    break

    # Process each transcript segment
    merged_segments = []

    for seg_idx, seg in enumerate(transcript_segments):
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg_text = seg.get("text", "")

        # Find speaker with maximum overlap
        speaker_id = _find_overlapping_speaker(
            seg_start, seg_end, diar_segments
        )

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

                # Find speaker for this word
                word_speaker = _find_overlapping_speaker(
                    word_start, word_end, diar_segments
                )

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


# [SEMANTIC-END] MERGE:ASSIGN_SPEAKERS
