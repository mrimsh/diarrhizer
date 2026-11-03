# Context budget for summaries and chat

CLI, GUI preview and service refusal use `summary/context_budget.py` (`LLM:CONTEXT_BUDGET`). The estimate is:

`ceil(total rendered characters / 3) + 32 + 16 × message count + response reserve`

Rendered input includes template, note, transcript, retained history and the new question. The response
reserve is `Provider.max_tokens` when set, otherwise 8192 for Anthropic API (matching the adapter's request
default) or an advisory 1024 for other backends. This advisory reserve does not impose a server output limit.
Provider CLI tools may add their own instructions/tool definitions; their unknown overhead is **not** measured
by this formula. Character ratios also vary with language. This is a consistent estimate, not tokenization
or a guarantee that the provider will accept a request. Its own context error can still occur.

If total estimate exceeds a configured `context_tokens`, nothing is sent. Equality is allowed. Without
a configured window, preview shows the estimate and “предел не задан”; it does not invent a model limit.
The full breakdown is recorded in summary metadata and each chat turn's `context_budget`.

```powershell
python -m diarrhizer summarize "out/<job>" --template "<name>" --provider "<name>" --preview --note "<text>"
```

Preview prints the full prompt and budget, performs no provider call and creates no summary. It exits 1
when the configured limit is exceeded, 0 otherwise. GUI preview displays the same breakdown; Create checks
it before launching a worker, and the service checks again before sending.

For an oversized summary, shorten the template/note or select a larger model/window. There is no automatic
chunking or silent truncation. Chat can additionally offer explicit removal of old turns or a transcript
time window; each offered reduction is checked with the response reserve and recorded if chosen.
