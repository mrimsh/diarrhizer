"""CLI entry point for Diarrhizer."""

import argparse
import json
import sys
from pathlib import Path

from diarrhizer.diagnostics.doctor import run_doctor_checks
from diarrhizer.pipeline.runner import run_pipeline
from diarrhizer.pipeline.stages.convert import ConvertStage
from diarrhizer.pipeline.stages.transcribe import TranscribeStage
from diarrhizer.pipeline.stages.diarize import DiarizeStage
from diarrhizer.pipeline.stages.merge import MergeStage
from diarrhizer.pipeline.stages.export import ExportStage


# [SEMANTIC-BEGIN] CLI:ENTRY
# @purpose: CLI entry point for Diarrhizer commands
# @description: Provides doctor and run commands for diagnostics and processing
# @sideEffects: Parses args, runs diagnostics or pipeline
# @errors: Exits with code 1 on invalid arguments
# @see: DIAGNOSTICS:DOCTOR
def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="diarrhizer",
        description="Local Windows tool for processing call recordings"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run environment diagnostics"
    )
    
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Process a media file"
    )
    run_parser.add_argument(
        "input",
        help="Path to input media file"
    )
    run_parser.add_argument(
        "--out",
        default="./out",
        help="Output directory (default: ./out)"
    )
    run_parser.add_argument(
        "--min-speakers",
        type=int,
        default=1,
        help="Minimum number of speakers (default: 1)"
    )
    run_parser.add_argument(
        "--max-speakers",
        type=int,
        default=10,
        help="Maximum number of speakers (default: 10)"
    )
    run_parser.add_argument(
        "--lang",
        default="auto",
        help="Language code or 'auto' for detection (default: auto)"
    )
    run_parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute all stages, overwriting existing outputs"
    )
    run_parser.add_argument(
        "--force-stage",
        type=str,
        choices=["convert", "transcribe", "diarize", "merge", "export"],
        help="Force recompute a specific stage (convert, transcribe, diarize, merge, export)"
    )
    run_parser.add_argument(
        "--speakers",
        type=str,
        default=None,
        help="Path to JSON file with speaker name mapping (e.g., {\"Speaker_00\": \"Ivan\", \"Speaker_01\": \"Maria\"})"
    )
    
    args = parser.parse_args()
    
    if args.command == "doctor":
        run_doctor_checks()
        return 0
    elif args.command == "run":
        # [SEMANTIC-BEGIN] CLI:RUN
        # @purpose: Run the processing pipeline for a media file
        # @description: Orchestrates FFmpeg conversion, ASR, diarization, and export
        # @inputs: args.input, args.out, args.min_speakers, args.max_speakers, args.lang, args.device, args.force, args.force_stage, args.speakers
        # @outputs: Artifacts in out/ directory
        # @sideEffects: Creates job directory, writes artifacts to disk
        # @errors: Exits with code 1 on failure
        # @see: PIPELINE:RUNNER, STAGE:CONVERT, STAGE:TRANSCRIBE, STAGE:DIARIZE, STAGE:MERGE, STAGE:EXPORT

        # [SEMANTIC-BEGIN] CONFIG:SPEAKERS_MAP
        # @purpose: Load speaker name mapping from JSON file
        # @description: Reads a JSON file that maps diarization IDs to display names
        # @inputs: args.speakers (file path)
        # @outputs: speakers dict or None
        # @sideEffects: File I/O, error handling for missing/invalid files
        # @errors: FileNotFoundError, json.JSONDecodeError
        # @see: CLI:RUN, EXPORT:MARKDOWN, EXPORT:JSON
        speakers = None
        if args.speakers:
            speakers_path = Path(args.speakers)
            if not speakers_path.exists():
                print(f"Error: Speakers file not found: {speakers_path}", file=sys.stderr)
                return 1
            with open(speakers_path, "r", encoding="utf-8") as f:
                speakers = json.load(f)
            # Validate speakers structure
            if not isinstance(speakers, dict):
                print(f"Error: Speakers file must contain a JSON object (dictionary)", file=sys.stderr)
                return 1
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in speakers.items()):
                print(f"Error: Speakers mapping must have string keys and string values", file=sys.stderr)
                return 1
            print(f"[CLI] Loaded speaker mapping: {speakers}")
        # [SEMANTIC-END] CONFIG:SPEAKERS_MAP

        try:
            # Wire pipeline: convert -> transcribe -> diarize -> merge -> export
            result = run_pipeline(
                input_path=args.input,
                out_dir=args.out,
                stages=[ConvertStage(), TranscribeStage(), DiarizeStage(), MergeStage(), ExportStage()],
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                language=args.lang,
                device=args.device,
                force=args.force,
                force_stage=args.force_stage,
                speakers=speakers,
            )
            print(f"\nPipeline completed successfully!")
            print(f"Job ID: {result['job_id']}")
            print(f"Output: {result['job_dir']}")
            return 0
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        # [SEMANTIC-END] CLI:RUN
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
# [SEMANTIC-END] CLI:ENTRY
