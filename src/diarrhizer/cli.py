"""CLI entry point for Diarrhizer."""

import argparse
import sys

from diarrhizer.diagnostics.doctor import run_doctor_checks


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
    
    args = parser.parse_args()
    
    if args.command == "doctor":
        run_doctor_checks()
        return 0
    elif args.command == "run":
        print("Processing not implemented yet")
        print(f"Input: {args.input}")
        print(f"Output: {args.out}")
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
# [SEMANTIC-END] CLI:ENTRY
