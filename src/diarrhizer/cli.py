"""CLI entry point for Diarrhizer."""

import argparse
import json
import logging
import sys
from pathlib import Path

from diarrhizer.audio_formats import BUILTIN_FORMATS, store_path
from diarrhizer.diagnostics.doctor import run_doctor_checks
from diarrhizer.env_file import load_project_env

# Pipeline stage names in execution order. Shared by --force-stage,
# --from-stage, and --to-stage so the three choices lists can't drift apart.
STAGE_NAMES = ["convert", "transcribe", "diarize", "merge", "export"]


def _configure_logging() -> None:
    """Attach a plain StreamHandler to the "diarrhizer" package logger so
    pipeline/stage progress (logged via logging, not print - see
    PIPELINE:RUNNER, STAGE:CONVERT et al.) reaches the console looking
    exactly like the print()-based output it replaced: the formatter emits
    only the message text, since each progress call already spells out its
    own "[stage] ..." / "Stage x: ..." wording. Idempotent so calling main()
    more than once (e.g. in tests) doesn't attach duplicate handlers.
    """
    package_logger = logging.getLogger("diarrhizer")
    if package_logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False


# [SEMANTIC-BEGIN] CLI:SUMMARIZE
# @purpose: Summarize a finished job's transcript with a prompt template and a model provider
# @description: Takes the job folder (or its export/result.md), attaches export/result.md to the chosen
#   template's prompt and asks the chosen provider; both default to the ones marked as default in
#   summary_templates.json / summary_providers.json (shared with the GUI). The reply is stored in the job's
#   summary/ folder; --print also writes it to stdout, --list shows the available templates and providers.
# @inputs: args.job, args.template, args.provider, args.print, args.list
# @outputs: summary/<timestamp>_<template>.md (+ .json) in the job folder; its path on stdout
# @sideEffects: Sends the transcript to the provider (subscription CLI or HTTPS API); writes files
# @errors: Exits with code 1 on a missing job/result, unknown template or provider, or a provider failure
# @see: SUMMARY:JOB, CONFIG:SUMMARY_TEMPLATES, CONFIG:SUMMARY_PROVIDERS, ADAPTER:LLM
def _run_summarize(args: argparse.Namespace) -> int:
    # Imported here so `doctor` and `run` never load the summary modules.
    from diarrhizer.adapters.llm import LLMError
    from diarrhizer.summary import service
    from diarrhizer.summary.providers import ProviderStore
    from diarrhizer.summary.templates import TemplateStore

    try:
        templates = TemplateStore.load()
        providers = ProviderStore.load()
        if args.list:
            print("Templates ('*' = default):")
            for name, template in templates.templates().items():
                kind = "built-in" if templates.is_builtin(name) else "yours"
                mark = "*" if template.name == templates.default().name else " "
                print(f" {mark} {name} ({kind}){': ' + template.description if template.description else ''}")
            print(f"\nProviders ('*' = default), from {providers.path}:")
            default = providers.default()
            for name, provider in providers.providers.items():
                mark = "*" if default is not None and name == default.name else " "
                model = f", model {provider.model}" if provider.model else ""
                print(f" {mark} {name} ({provider.kind}{model})")
            if not providers.providers:
                print("   none - add one in the GUI (Саммаризация → Провайдеры) or edit that file")
            return 0

        if not args.job:
            print("Error: pass the job folder (or its export/result.md) to summarize", file=sys.stderr)
            return 1
        job_dir = service.resolve_job_dir(args.job)
        template = templates.resolve(args.template)
        provider = providers.resolve(args.provider)
        print(f"Summarizing {job_dir.name} with template '{template.name}' via '{provider.name}'...")
        result = service.summarize_job(job_dir, template, provider)
    except (FileNotFoundError, ValueError, LLMError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Summary saved: {result['path']}")
    if result["truncated"]:
        print("Warning: the provider cut the reply off at its length limit", file=sys.stderr)
    if args.print:
        try:
            print("\n" + result["text"])
        except UnicodeEncodeError:  # a legacy console code page cannot show the text
            sys.stdout.buffer.write(("\n" + result["text"] + "\n").encode("utf-8"))
    return 0
# [SEMANTIC-END] CLI:SUMMARIZE


# [SEMANTIC-BEGIN] CLI:ENTRY
# @purpose: CLI entry point for Diarrhizer commands
# @description: Provides doctor, run and summarize commands for diagnostics, processing and summaries
# @sideEffects: Loads project .env, parses args, runs diagnostics or pipeline, configures a StreamHandler
#   (stdout, "%(message)s") on the "diarrhizer" package logger so stage/pipeline
#   progress logged via logging (see PIPELINE:RUNNER) prints like the old print()s did
# @errors: Exits with code 1 on invalid arguments
# @see: DIAGNOSTICS:DOCTOR, PIPELINE:RUNNER
def main() -> int:
    """Main CLI entry point."""
    load_project_env()
    _configure_logging()

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
    #
    # NOTE on defaults: the literals below intentionally duplicate
    # PipelineConfig's field defaults (src/diarrhizer/pipeline/runner.py).
    # They can't just reference PipelineConfig directly: argparse needs
    # concrete values for its own --help text, and two of them
    # (--asr-condition-on-previous-text, --asr-vad-filter) are parsed here as
    # true/false strings while PipelineConfig stores real bools. Keep these in
    # sync with PipelineConfig by hand when either changes.
    run_parser = subparsers.add_parser(
        "run",
        help="Process a media file"
    )
    run_parser.add_argument(
        "input",
        nargs="*",
        default=[],
        help="Media file(s) and/or folder(s) to process; a folder adds the media files directly "
             "inside it. Several inputs run one after another as separate jobs, and a failed file "
             "does not stop the rest. Optional if --job-dir resumes a job that already recorded its input"
    )
    run_parser.add_argument(
        "--out",
        default="./out",
        help="Output directory (default: ./out; ignored when --job-dir is given)"
    )
    run_parser.add_argument(
        "--job-dir",
        type=str,
        default=None,
        help="Resume an existing job directory instead of starting a new job "
             "(e.g. out/meeting_20260101_120000)"
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
        "--min-turn-duration",
        type=float,
        default=0.4,
        help="Shortest speaker turn in seconds the merge stage splits out on "
             "its own; shorter word runs between two runs of the same other "
             "speaker are folded back in as diarization jitter. Raise it if "
             "noisy diarization shreds sentences, use 0 to split on every "
             "word-level speaker change (default: 0.4)"
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
        choices=STAGE_NAMES,
        help="Force recompute a specific stage (convert, transcribe, diarize, merge, export)"
    )
    run_parser.add_argument(
        "--from-stage",
        type=str,
        choices=STAGE_NAMES,
        default=None,
        help="Start the pipeline at this stage, skipping earlier stages entirely "
             "(their outputs must already exist on disk, e.g. via --job-dir)"
    )
    run_parser.add_argument(
        "--to-stage",
        type=str,
        choices=STAGE_NAMES,
        default=None,
        help="Stop the pipeline after this stage, skipping later stages entirely "
             "(e.g. --from-stage export --to-stage export to only re-export)"
    )
    run_parser.add_argument(
        "--speakers",
        type=str,
        default=None,
        help="Path to JSON file with speaker name mapping (e.g., {\"Speaker_00\": \"Ivan\", \"Speaker_01\": \"Maria\"})"
    )
    run_parser.add_argument(
        "--asr-model",
        type=str,
        default="large-v3",
        help="WhisperX model (default: large-v3; use a smaller size like 'base' for faster/cheaper "
             "runs, or an HF repo like koekaverna/faster-whisper-podlodka-turbo)"
    )
    run_parser.add_argument(
        "--asr-compute-type",
        type=str,
        default=None,
        choices=["float16", "int8_float16", "int8"],
        help="Compute type for WhisperX (default: auto based on device)"
    )
    run_parser.add_argument(
        "--asr-beam-size",
        type=int,
        default=5,
        help="Decoding beam size (default: 5)"
    )
    run_parser.add_argument(
        "--asr-temperature",
        type=float,
        default=0.0,
        help="Decoding temperature (default: 0.0)"
    )
    run_parser.add_argument(
        "--asr-condition-on-previous-text",
        type=str,
        default="true",
        choices=["true", "false", "1", "0", "yes", "no"],
        help="Condition on previous text for stable decoding (default: true)"
    )
    run_parser.add_argument(
        "--asr-initial-prompt-file",
        type=str,
        default=None,
        help="Path to file containing initial prompt/glossary for ASR"
    )
    run_parser.add_argument(
        "--asr-hotwords-file",
        type=str,
        default=None,
        help="Path to hotwords file (not yet implemented)"
    )
    run_parser.add_argument(
        "--asr-vad-filter",
        type=str,
        default="true",
        choices=["true", "false", "1", "0", "yes", "no"],
        help="Enable VAD filtering (default: true)"
    )
    run_parser.add_argument(
        "--asr-vad-min-silence-ms",
        type=int,
        default=1000,
        help="VAD minimum silence in milliseconds (default: 1000)"
    )
    run_parser.add_argument(
        "--audio-profile",
        type=str,
        default="raw",
        choices=["raw", "voice-call", "denoise-light", "split-stereo"],
        help="Audio preprocessing profile (default: raw)"
    )
    # Not a PipelineConfig literal like the flags above: the default is
    # whatever profile the user marked as default in audio_formats.json
    # (resolved in run_pipeline from None), so it is shared with the GUI.
    run_parser.add_argument(
        "--audio-format",
        type=str,
        default=None,
        help="How the job keeps its audio on disk: a built-in format "
             f"({', '.join(BUILTIN_FORMATS)}) or a profile from {store_path().name}. "
             "Non-wav formats write audio/archive.<ext> next to the working WAV "
             "(default: the default profile in that file, else wav)"
    )
    run_parser.add_argument(
        "--keep-wav",
        type=str,
        default=None,
        choices=["true", "false", "1", "0", "yes", "no"],
        help="Keep the working audio/normalized.wav after the job completes, overriding "
             "the audio format's own setting (ignored for the wav format, which only has that file)"
    )

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize a finished job's transcript (export/result.md) with a language model"
    )
    summarize_parser.add_argument(
        "job",
        nargs="?",
        default=None,
        help="Job folder (e.g. out/meeting_20260101_120000) or its export/result.md"
    )
    summarize_parser.add_argument(
        "--template",
        default=None,
        help="Prompt template name (default: the default template in summary_templates.json, "
             "else the first built-in one); --list shows the names"
    )
    summarize_parser.add_argument(
        "--provider",
        default=None,
        help="Provider name from summary_providers.json (default: its default provider); "
             "providers are added in the GUI (Саммаризация → Провайдеры)"
    )
    summarize_parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the summary to stdout (it is always saved in the job's summary/ folder)"
    )
    summarize_parser.add_argument(
        "--list",
        action="store_true",
        help="List the available templates and providers and exit"
    )

    args = parser.parse_args()

    if args.command == "doctor":
        return 0 if run_doctor_checks() else 1
    elif args.command == "summarize":
        return _run_summarize(args)
    elif args.command == "run":
        # [SEMANTIC-BEGIN] CLI:RUN
        # @purpose: Run the processing pipeline for one or several media files
        # @description: Orchestrates FFmpeg conversion, ASR, diarization, and export. --job-dir
        #   resumes an existing job directory (args.input becomes optional if that job already
        #   recorded it in meta/run.json); --from-stage/--to-stage select a stage range to run,
        #   skipping stages outside it entirely - e.g. --from-stage export re-exports without
        #   recomputing ASR/diarization. Several inputs (or a folder, see INPUT:EXPAND_MEDIA) run as a
        #   batch: one job per file, each in its own new_job_dir(), a failed file is reported and the
        #   batch moves on, and a summary is printed at the end.
        # @inputs: args.input, args.out, args.job_dir, args.min_speakers, args.max_speakers, args.lang, args.device, args.force, args.force_stage, args.from_stage, args.to_stage, args.speakers, args.min_turn_duration, ASR params, audio_profile, audio_format, keep_wav
        # @outputs: Artifacts in out/ directory
        # @sideEffects: Creates job directories, writes artifacts to disk
        # @errors: Exits with code 1 on failure (in a batch: if any file failed)
        # @see: PIPELINE:RUNNER, INPUT:EXPAND_MEDIA, STAGE:CONVERT, STAGE:TRANSCRIBE, STAGE:DIARIZE, STAGE:MERGE, STAGE:EXPORT, CONFIG:PIPELINE, CONFIG:AUDIO_FORMATS

        from diarrhizer.media import expand_media_paths

        inputs = expand_media_paths(args.input)
        if args.input and not inputs:
            print(f"Error: No media files found in: {', '.join(args.input)}", file=sys.stderr)
            return 1
        if args.job_dir and len(inputs) > 1:
            print("Error: --job-dir resumes a single job; pass at most one input with it", file=sys.stderr)
            return 1

        # Imported lazily (not at module level) so that `doctor` never pulls in
        # torch/whisperx as a side effect - it needs to stay import-safe to diagnose
        # a broken install.
        from diarrhizer.pipeline.runner import new_job_dir, run_pipeline
        from diarrhizer.pipeline.stages.convert import ConvertStage
        from diarrhizer.pipeline.stages.transcribe import TranscribeStage
        from diarrhizer.pipeline.stages.diarize import DiarizeStage
        from diarrhizer.pipeline.stages.merge import MergeStage
        from diarrhizer.pipeline.stages.export import ExportStage

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
            if not isinstance(speakers, dict):
                print(f"Error: Speakers file must contain a JSON object (dictionary)", file=sys.stderr)
                return 1
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in speakers.items()):
                print(f"Error: Speakers mapping must have string keys and string values", file=sys.stderr)
                return 1
            print(f"[CLI] Loaded speaker mapping: {speakers}")
        # [SEMANTIC-END] CONFIG:SPEAKERS_MAP

        # Helper to convert string bool to actual bool
        def _parse_bool(val: str) -> bool:
            return val.lower() in ("true", "1", "yes")

        # [SEMANTIC-BEGIN] CONFIG:INITIAL_PROMPT
        # @purpose: Load initial prompt/glossary for ASR from file
        # @description: Reads a text file containing initial prompt for terminology guidance
        # @inputs: args.asr_initial_prompt_file (file path)
        # @outputs: initial_prompt string or None
        # @sideEffects: File I/O
        # @errors: FileNotFoundError
        # @see: CLI:RUN, STAGE:TRANSCRIBE, ADAPTER:WHISPERX_ASR
        initial_prompt = None
        if args.asr_initial_prompt_file:
            prompt_path = Path(args.asr_initial_prompt_file)
            if not prompt_path.exists():
                print(f"Error: Initial prompt file not found: {prompt_path}", file=sys.stderr)
                return 1
            with open(prompt_path, "r", encoding="utf-8") as f:
                initial_prompt = f.read().strip()
            print(f"[CLI] Loaded initial prompt: {len(initial_prompt)} chars")
        # [SEMANTIC-END] CONFIG:INITIAL_PROMPT

        def _run(input_path, job_dir) -> dict:
            return run_pipeline(
                input_path=input_path,
                out_dir=args.out,
                stages=[ConvertStage(), TranscribeStage(), DiarizeStage(), MergeStage(), ExportStage()],
                job_dir=job_dir,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                language=args.lang,
                device=args.device,
                force=args.force,
                force_stage=args.force_stage,
                from_stage=args.from_stage,
                to_stage=args.to_stage,
                speakers=speakers,
                asr_model=args.asr_model,
                asr_compute_type=args.asr_compute_type,
                asr_beam_size=args.asr_beam_size,
                asr_temperature=args.asr_temperature,
                asr_condition_on_previous_text=_parse_bool(args.asr_condition_on_previous_text),
                asr_initial_prompt=initial_prompt,
                asr_hotwords_file=args.asr_hotwords_file,
                asr_vad_filter=_parse_bool(args.asr_vad_filter),
                asr_vad_min_silence_ms=args.asr_vad_min_silence_ms,
                audio_profile=args.audio_profile,
                audio_format=args.audio_format,
                keep_wav=None if args.keep_wav is None else _parse_bool(args.keep_wav),
                min_turn_duration=args.min_turn_duration,
            )

        def _error_text(e: Exception) -> str:
            # Expected failures carry a readable message already; anything else
            # (corrupt cached JSON, malformed inputs, ...) gets its type name so
            # it exits cleanly instead of dumping a raw traceback.
            if isinstance(e, (FileNotFoundError, RuntimeError, ValueError)):
                return str(e)
            return f"{type(e).__name__}: {e}"

        if len(inputs) <= 1:
            try:
                result = _run(inputs[0] if inputs else None, args.job_dir)
            except Exception as e:
                print(f"Error: {_error_text(e)}", file=sys.stderr)
                return 1
            print(f"\nPipeline completed successfully!")
            print(f"Job ID: {result['job_id']}")
            print(f"Output: {result['job_dir']}")
            return 0

        summary = []
        for index, input_path in enumerate(inputs, 1):
            print(f"\n[{index}/{len(inputs)}] {input_path}")
            try:
                result = _run(input_path, new_job_dir(args.out, input_path))
                summary.append((True, input_path, result["job_dir"]))
            except Exception as e:
                print(f"Error: {_error_text(e)}", file=sys.stderr)
                summary.append((False, input_path, _error_text(e)))

        failed = sum(1 for ok, _, _ in summary if not ok)
        print("\n" + "=" * 50)
        print(f"Batch finished: {len(summary) - failed} succeeded, {failed} failed")
        for ok, input_path, detail in summary:
            print(f"  {'OK  ' if ok else 'FAIL'}  {input_path.name}: {detail}")
        print("=" * 50)
        return 1 if failed else 0
        # [SEMANTIC-END] CLI:RUN
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
# [SEMANTIC-END] CLI:ENTRY