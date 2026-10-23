"""CLI entry point for Diarrhizer."""

import argparse
import json
import logging
import sys
import textwrap
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
#   summary_templates.json / summary_providers.json (shared with the GUI). The template's {{wildcards}}
#   (SUMMARY:WILDCARDS) are filled in; --note / --note-file (mutually exclusive) give the text for {{note}}
#   and --list-wildcards shows the wildcards (names and descriptions) and exits. The reply is stored in the
#   job's summary/ folder; --print also writes it to stdout, --list shows the available templates and providers.
# @inputs: args.job, args.template, args.provider, args.note, args.note_file, args.print, args.list, args.list_wildcards
# @outputs: summary/<timestamp>_<template>.md (+ .json) in the job folder; its path on stdout
# @sideEffects: Sends the transcript to the provider (subscription CLI or HTTPS API); writes files
# @errors: Exits with code 1 on a missing job/result or note file, an unknown template, provider or wildcard, or a provider failure; code 2 for --note together with --note-file
# @see: SUMMARY:JOB, SUMMARY:WILDCARDS, CONFIG:SUMMARY_TEMPLATES, CONFIG:SUMMARY_PROVIDERS, ADAPTER:LLM
def _run_summarize(args: argparse.Namespace) -> int:
    # Imported here so `doctor` and `run` never load the summary modules.
    from diarrhizer.adapters.llm import LLMError
    from diarrhizer.summary import service
    from diarrhizer.summary.providers import ProviderStore
    from diarrhizer.summary.templates import TemplateStore
    from diarrhizer.summary.wildcards import WILDCARDS

    if args.list_wildcards:
        print("Wildcards for prompt templates: write {{name}} where the value should stand;")
        print("a literal {{ is written \\{{ (single braces are never touched).\n")
        for wildcard in WILDCARDS.values():
            print(f"  {{{{{wildcard.name}}}}}")
            print(textwrap.fill(wildcard.description, width=100, initial_indent="      ", subsequent_indent="      "))
        return 0

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
                print("   none - add one in the GUI (Работа с языковой моделью → Провайдеры) or edit that file")
            return 0

        if not args.job:
            print("Error: pass the job folder (or its export/result.md) to summarize", file=sys.stderr)
            return 1
        job_dir = service.resolve_job_dir(args.job)
        template = templates.resolve(args.template)
        provider = providers.resolve(args.provider)
        note = service.read_note_file(args.note_file) if args.note_file else (args.note or "")
        with_note = f" and a note ({len(note.strip())} chars)" if note.strip() else ""
        print(f"Summarizing {job_dir.name} with template '{template.name}'{with_note} via '{provider.name}'...")
        result = service.summarize_job(job_dir, template, provider, note=note)  # warnings: logged by the service
    except (FileNotFoundError, ValueError, LLMError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Summary saved: {result['path']}")
    for line in service.describe_usage(result):  # what the request used, as far as the provider says
        print(line)
    if result["truncated"]:
        print("Warning: the provider cut the reply off at its length limit", file=sys.stderr)
    if args.print:
        try:
            print("\n" + result["text"])
        except UnicodeEncodeError:  # a legacy console code page cannot show the text
            sys.stdout.buffer.write(("\n" + result["text"] + "\n").encode("utf-8"))
    return 0
# [SEMANTIC-END] CLI:SUMMARIZE


def _seconds(text: str) -> float:
    """HH:MM:SS, MM:SS or plain seconds -> seconds; ValueError says what is wrong."""
    try:
        parts = [float(part) for part in text.strip().split(":")]
    except ValueError:
        raise ValueError(f"Not a time: {text!r} (use HH:MM:SS, MM:SS or seconds)") from None
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Not a time: {text!r} (use HH:MM:SS, MM:SS or seconds)")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


# [SEMANTIC-BEGIN] CLI:CHAT
# @purpose: Ask a model questions about a finished job's transcript, in a dialogue that is kept with the job
# @description: `chat <job> "<question>"` starts a new dialogue (the template - default or --template - gives the
#   instruction, --note/--note-file the note, fixed for the whole dialogue) and asks its first question, or, with
#   --conversation <id> / --last, asks the next question of an existing one (--provider / --model may differ from
#   the earlier turns'). The answer is stored in the job's chat/ folder; --print also prints it, and what the turn
#   used (tokens with the cache parts, cost, session or whole history sent) is always printed. A request whose
#   estimated size is over the provider's context_tokens is not sent: the error lists the options, and --drop-turns /
#   --window / another provider take one. --list shows the
#   job's dialogues, --export <path|-> writes one as Markdown, --delete <id> removes one. summarize is unchanged.
# @inputs: args.job, args.question, args.conversation, args.last, args.template, args.provider, args.model, args.note,
#   args.note_file, args.title, args.no_session, args.print, args.list, args.export, args.delete
# @outputs: chat/<id>.json (and .md on export) in the job folder; the dialogue id and statistics on stdout
# @sideEffects: Sends the transcript and the dialogue to the provider; writes files
# @errors: Exits with code 1 on a missing job/result/dialogue, unknown template, provider or wildcard, a missing question, or a provider failure
# @see: SUMMARY:CHAT, CLI:SUMMARIZE, ADAPTER:LLM
def _run_chat(args: argparse.Namespace) -> int:
    import dataclasses

    from diarrhizer.adapters.llm import LLMError
    from diarrhizer.summary import chat, service
    from diarrhizer.summary.providers import ProviderStore
    from diarrhizer.summary.templates import TemplateStore

    try:
        job_dir = service.resolve_job_dir(args.job)
        if args.list:
            conversations = chat.list_conversations(job_dir)
            for c in conversations:
                print(f"{c.id}  {c.updated_at}  {len(c.turns)} turn(s)  {c.template}  {c.title}")
            if not conversations:
                print("No dialogues for this job yet.")
            return 0
        if args.delete:
            chat.delete_conversation(job_dir, args.delete)
            print(f"Deleted dialogue {args.delete}")
            return 0

        conversation = None
        if args.conversation:
            conversation = chat.load_conversation(job_dir, args.conversation)
        elif args.last:
            found = chat.list_conversations(job_dir)
            if not found:
                raise ValueError("This job has no dialogues yet")
            conversation = found[0]
        question = (args.question or "").strip()

        if args.export and not question:
            if conversation is None:
                raise ValueError("--export needs --conversation <id> or --last")
            if args.export == "-":
                sys.stdout.buffer.write(chat.export_markdown(conversation).encode("utf-8"))
            else:
                print(f"Exported: {chat.export_conversation(job_dir, conversation.id, args.export)}")
            return 0
        if conversation is not None and not question:
            raise ValueError("Pass the question to ask")

        providers = ProviderStore.load()
        if conversation is None:
            template = TemplateStore.load().resolve(args.template)
            provider = providers.resolve(args.provider)
            note = service.read_note_file(args.note_file) if args.note_file else (args.note or "")
            conversation = chat.create_conversation(job_dir, template, provider, note=note, title=args.title or "")
            print(f"New dialogue {conversation.id} (template '{template.name}')")
        else:
            provider = providers.resolve(args.provider or conversation.provider.get("name"))
        if args.model:
            provider = dataclasses.replace(provider, model=args.model)
        window = None
        if args.window:
            first, _, last = args.window.partition("-")
            window = (_seconds(first), _seconds(last))
        print(f"Asking '{provider.name}'...")
        turn = chat.ask(job_dir, conversation.id, question, provider, keep_cli_session=not args.no_session,
                        drop_turns=args.drop_turns or 0, window=window)
    except chat.ContextTooLong as e:  # a ValueError: says what to do instead of sending blindly
        print(f"Error: {e}\nNothing was sent. Repeat with --drop-turns N, --window START-END or another --provider/--model.",
              file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError, LLMError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Dialogue {conversation.id}, turn {len(conversation.turns) + 1}")
    for line in chat.turn_stats(turn):
        print(line)
    if turn.truncated:
        print("Warning: the provider cut the reply off at its length limit", file=sys.stderr)
    if args.print:
        sys.stdout.buffer.write(("\n" + turn.answer + "\n").encode("utf-8"))
    if args.export:
        if args.export == "-":
            sys.stdout.buffer.write(chat.export_markdown(chat.load_conversation(job_dir, conversation.id)).encode("utf-8"))
        else:
            print(f"Exported: {chat.export_conversation(job_dir, conversation.id, args.export)}")
    return 0
# [SEMANTIC-END] CLI:CHAT


# [SEMANTIC-BEGIN] CLI:ENTRY
# @purpose: CLI entry point for Diarrhizer commands
# @description: Provides doctor, run, summarize and chat commands for diagnostics, processing, summaries and dialogues
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
             "providers are added in the GUI (Работа с языковой моделью → Провайдеры)"
    )
    note_group = summarize_parser.add_mutually_exclusive_group()
    note_group.add_argument(
        "--note",
        default=None,
        metavar="TEXT",
        help="Your own text for this run: it goes where the template has {{note}}; a template without "
             "{{note}} gets it before the transcript, with a warning (--list-wildcards describes {{note}})"
    )
    note_group.add_argument(
        "--note-file",
        default=None,
        metavar="PATH",
        help="Read the note from a text file (UTF-8, or UTF-16 with a BOM) instead of --note"
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
    summarize_parser.add_argument(
        "--list-wildcards",
        action="store_true",
        help="List the {{wildcards}} a prompt template may contain, with descriptions, and exit"
    )

    # Chat command
    chat_parser = subparsers.add_parser(
        "chat",
        help="Ask a language model questions about a finished job's transcript, in a dialogue kept with the job"
    )
    chat_parser.add_argument("job", help="Job folder (e.g. out/meeting_20260101_120000) or its export/result.md")
    chat_parser.add_argument(
        "question", nargs="?", default=None,
        help="The question. A new dialogue may start without one: the template's instruction is carried out"
    )
    chat_which = chat_parser.add_mutually_exclusive_group()
    chat_which.add_argument("--conversation", default=None, metavar="ID",
                            help="Continue this dialogue (ids: --list); without it a new one is started")
    chat_which.add_argument("--last", action="store_true", help="Continue the job's most recently used dialogue")
    chat_parser.add_argument("--template", default=None,
                             help="New dialogue: the prompt template that gives the instruction (default as for summarize)")
    chat_parser.add_argument("--provider", default=None,
                             help="Provider name (default: the dialogue's last one, for a new dialogue the default)")
    chat_parser.add_argument("--model", default=None, help="Use this model for this turn instead of the provider's own")
    chat_note = chat_parser.add_mutually_exclusive_group()
    chat_note.add_argument("--note", default=None, metavar="TEXT",
                           help="New dialogue: the text for {{note}}, fixed for the whole dialogue")
    chat_note.add_argument("--note-file", default=None, metavar="PATH", help="The same, read from a text file")
    chat_parser.add_argument("--title", default=None, help="New dialogue: its title (default: the first question)")
    chat_parser.add_argument("--no-session", action="store_true",
                             help="Do not keep a Codex CLI session: send the whole dialogue every turn")
    chat_parser.add_argument("--drop-turns", type=int, default=None, metavar="N",
                             help="Leave this many more of the oldest turns out of the request from now on "
                                  "(the answer to a 'context too long' error)")
    chat_parser.add_argument("--window", default=None, metavar="START-END",
                             help="Keep only this part of the transcript from now on, e.g. 00:00:00-00:45:00 "
                                  "(HH:MM:SS, MM:SS or seconds)")
    chat_parser.add_argument("--print", action="store_true", help="Also print the answer (it is always stored)")
    chat_parser.add_argument("--list", action="store_true", help="List the job's dialogues and exit")
    chat_parser.add_argument("--export", default=None, metavar="PATH",
                             help="Write the dialogue as Markdown to PATH ('-' = stdout), after the question if one is asked")
    chat_parser.add_argument("--delete", default=None, metavar="ID", help="Delete a dialogue and exit")

    args = parser.parse_args()

    if args.command == "doctor":
        return 0 if run_doctor_checks() else 1
    elif args.command == "summarize":
        return _run_summarize(args)
    elif args.command == "chat":
        return _run_chat(args)
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