# Diarrhizer 2.0: direction for 2.x

Author: MRIMSH.

Version 2.0 establishes a shared release for the CLI, desktop GUI and Windows launcher.
The development vector for 2.x is an increasingly autonomous tool for logging calls and
creating useful artifacts from them.

## Current foundation

The application processes supplied recordings through FFmpeg, WhisperX and speaker diarization,
exports transcripts, and offers on-demand summaries and dialogues with language models. Projects,
teams, participant names and job storage organize the resulting call history. The CLI and GUI share
the same local core, settings and artifacts.

## Planned call agent

One direction is a separate agent that connects to calls and manages the full process: capture,
transcription, diarization, summarization and saving the results into the call history. Useful outputs
may include meeting notes, decisions, action items and drafts of follow-up messages or tasks.

The agent should reuse the processing and summary services through a small orchestration layer,
so that improvements remain available to both the CLI and GUI. Existing local file processing stays
a supported entry point. Live capture, streaming recognition and live summaries are currently
[research](realtime.md) under `prototypes/realtime/`, separate from production code.

## Optional skills

The agent could be extended with small, composable skills for specific integrations, for example:

- Prepare and send a message to a call's chat with a summary, decisions or next steps.
- Form a request to a task-tracker API from agreed action items and execute it to create or update tasks.

These actions should use explicitly configured destinations and permissions, keep credentials local,
and record their results with the call. Skills are a proposed extension point; no heavy plugin
framework or new integration dependency is required by this release.

## Supporting work

Reliable capture, recovery after interruptions, reproducible environments, measured transcription and
diarization quality, and editable transcripts support increasing autonomy. Implementations and their
limits should be checked before a planned capability is marked as available.

The call agent and integration skills are development goals, not shipped 2.0 features or a committed
delivery schedule. See the [release notes](../CHANGELOG.md), [architecture](architecture.md) and
[development checks](development.md) for the current application.
