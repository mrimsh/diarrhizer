# Multi-turn dialogue through the subscription CLIs

Research note (Codex CLI 0.155.1 and Claude Code 2.1.278, on the developer's Windows machine). The question: can a
conversation be continued through `codex exec` / `claude -p` with the context kept, instead of sending the whole
history every time?

## Decision

| CLI | Decision | Why |
|-----|----------|-----|
| Codex | **Resume the session.** `complete_chat(..., keep_session=True)` starts a session that is written to disk and returns its id (`LLMReply.session_id`); `complete_chat(..., session_id=id)` resumes it and sends only the messages after the last assistant message. | Worked when run, in the isolated mode the adapter uses (below). |
| Claude Code | **Whole history every time**, reply marked `full-history`, `cache_available=False`. | The CLI is not signed in on this machine, so nothing could be run; the documented behaviour has a catch (below) that would need a live check first. |

Every `LLMReply` says what happened: `context_mode` is `cli-session` (only the new message was sent) or
`full-history`, `context_chars` is the size of the message text in the request, `cache_available` is `True` for a
resumed session, `False` for a CLI sent the whole history, `None` when not known (API providers report it in `usage`).
If a Codex session to resume is gone (`no rollout found`), the adapter sends the whole conversation in a new session
and says `full-history` with the new id; any other failure is raised as before. One-shot `complete()` is unchanged:
`--ephemeral`, nothing written to the user's Codex history.

## Confirmed by running (Codex, invented text, real account)

- `codex exec --json` prints `{"type":"thread.started","thread_id":"<uuid>"}` first: that is the id.
- `codex exec resume [OPTIONS] <SESSION_ID> -` continues it (prompt on stdin with `-`) and prints the same
  `thread_id`. The model remembered an invented code word from the earlier turn, also when resumed **from a different
  working directory** (the adapter uses a fresh temporary folder per request) and with `--ignore-user-config`.
- `--ephemeral` and resume exclude each other (an ephemeral session is not saved), so a kept session starts without
  `--ephemeral`. `exec resume` does not accept `--sandbox` or `--color` (`unexpected argument`); `--json`,
  `-o/--output-last-message`, `--model`, `--ignore-user-config`, `--skip-git-repo-check` work.
- An unknown id fails with exit code 1 and `Error: thread/resume: ... no rollout found for thread id ... (code -32600)`.
- Metrics: `turn.completed.usage` has `input_tokens` (the whole context of that turn, cached part included),
  `cached_input_tokens`, **`cache_write_input_tokens`** (new - the adapter reads it), `output_tokens`. In a run of
  three turns: 19 187 → 39 102 → 59 047 input tokens, of which 12 288 → 31 360 → 51 072 were cached. So a resumed turn
  is still billed as the whole context, mostly as cached input; the saving is cache hits and not having to send the
  history ourselves, not a smaller token count.
- The adapter, end to end (`complete_chat` three times): first turn `full-history` + id, second `cli-session`
  (26 characters sent, 31 360 of 38 545 input tokens cached), a made-up id fell back to `full-history` in a new session.

## Taken from the documentation (`claude --help`), not run

- `claude -p --resume <session-id>` continues a session; `--session-id <uuid>` lets the caller pick the id (so it
  would not have to be parsed); `--fork-session` copies instead. The stream-json `result` event carries `session_id`
  (documented, not seen here). `claude -p --output-format stream-json` already reports `cache_read_input_tokens` and
  `cache_creation_input_tokens`, which the adapter parses.
- The adapter passes `--no-session-persistence`, which the help says makes sessions unresumable. Using sessions means
  dropping it for kept sessions.
- Claude Code keeps sessions per project directory (`-c` is "the most recent conversation in the current directory"),
  and the adapter runs every request in a new temporary directory. A kept Claude session would need a **stable**
  working directory per conversation. Whether `--resume <id>` finds a session from another directory is not known.
- `--safe-mode` (the isolation flag) with resume: not known.

To enable Claude: sign in (`claude auth login`), then check on a short invented text that `--resume` works from the
chosen directory with `--safe-mode`, and that quota use and `usage` look as expected.

## Not verified / things to know

- **Which sandbox a resumed Codex session runs under** was not tested (`resume` takes no `--sandbox`; the first turn
  was `read-only`). The prompts are plain text questions; do not rely on it for tool use.
- **How a resumed turn counts against the subscription limits** was not measured; only the token counts above.
- **Privacy:** a kept Codex session is written to Codex's own history on disk (`~/.codex/sessions`), unlike the
  ephemeral one-shot requests. Nothing in the app deletes it yet.
- Codex messages and flags are those of version 0.155.1; the session-lost test matches `no rollout found` /
  `thread/resume failed`, a different wording in a future version would surface as an ordinary error.
