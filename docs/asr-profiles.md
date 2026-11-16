# ASR profiles, glossary and VAD

The settings below affect WhisperX 3.3.1 recognition, before alignment and diarization. They do not
run an LLM or change a finished transcript. Existing jobs default to `default`, no glossary, VAD
onset `0.5`, offset `0.363`, maximum chunk duration `30` seconds.

## Russian-call profile

`--asr-profile ru-call` adds this fixed punctuation/style sample:

> Обсуждаем рабочие вопросы на русском языке. Что уже готово? Проверим задачи, сроки и результаты.
> Да, договорились. Нет, этот вопрос ещё нужно уточнить.

It leaves language and model selection to the user. A predominantly Russian recording can be tried
with `--lang ru`; compare `auto` for mixed-language calls rather than assuming a universal winner.
Whisper hints bias recognition and can also bias errors; the sample does not establish a measured
quality improvement on your recordings.

The effective prompt combines, in order:

1. The profile sample (`default` adds nothing).
2. The existing custom prompt (`--asr-initial-prompt-file` / **Дополнительный prompt**).
3. `Термины и имена: ...` from `--asr-glossary-file` / **Глоссарий созвона**.

Glossary files use UTF-8 (with or without BOM). Use one relevant term per line or separate terms by
commas; blank lines are ignored when composing the prompt. For example:

```text
Анна Соколова
Diarrhizer
WhisperX, FFmpeg, .NET
```

The glossary is placed last because the pinned decoder keeps the tail of a prompt that exceeds
its budget. We measure tokens using the loaded model's tokenizer and report the count/limit in
`effective_settings.initial_prompt_tokens` / `initial_prompt_token_limit`. On standard Whisper
the limit is 223 tokens. Overflow generates a warning; shorten the inputs to keep all hints.
There is no approximate character limit or extra tokenizer dependency in the GUI.

## Real VAD controls

| CLI / Python runner setting | Default | Validation | Backend destination |
| --- | --- | --- | --- |
| `--asr-vad-onset` / `asr_vad_onset` | `0.5` | finite, `0 < onset < 1` | `load_model(vad_options.vad_onset)` |
| `--asr-vad-offset` / `asr_vad_offset` | `0.363` | finite, `0 < offset <= onset`, `offset < 1` | `load_model(vad_options.vad_offset)` |
| `--asr-chunk-size` / `asr_chunk_size` | `30` | whole seconds, `1..30` | `transcribe(chunk_size=...)` |

Onset controls activation and offset controls the end of detected speech. Lower thresholds may
recover quiet speech and admit more noise. Shorter chunks change the available acoustic context and
segmentation; they are not an unconditional quality improvement. Keep the defaults as a baseline.

Temperature, previous-text conditioning, VAD disabling and `--asr-vad-min-silence-ms` retain their
legacy compatibility behavior and warnings; the last setting is not an alias for onset/offset.
The new controls are in New Job's advanced section and the restart-stage dialog, with cross-field
validation. Model loading is refreshed when thresholds change; changing only chunk size does not
reload the model inside an adapter session.

## Reproducible jobs and resume

```powershell
python -m diarrhizer run "call.mp4" --lang ru --asr-profile ru-call --asr-glossary-file "terms.txt" --asr-vad-onset 0.5 --asr-vad-offset 0.363 --asr-chunk-size 30
python -m diarrhizer run --job-dir "out/job" --from-stage transcribe --asr-vad-onset 0.45 --asr-vad-offset 0.30
```

Python callers use `asr_profile`, `asr_glossary` (text, not a path), `asr_initial_prompt`, and the
three VAD/chunk settings on `run_pipeline`. Omitted settings restore the recorded values; explicit
values override them. `asr_glossary=None` or an explicitly supplied empty glossary file clears it.
`--asr-profile default` removes the built-in sample while keeping the recorded custom prompt/glossary.

`meta/run.json.pipeline_config` stores input texts and controls, so later file edits/removal cannot
change a resumed or queued job. The ASR artifact records profile, effective-prompt and glossary hashes,
requested controls, effective settings and warnings. Cache version 4 forces older ASR artifacts to be
recomputed once when transcription is selected. Subsequent changes to profile, glossary, custom
prompt, onset, offset or chunk size invalidate ASR; opening/exporting an existing result alone does
not rerun recognition.

## Comparing quality

Validation on 2026-11-14 included CLI/GUI/adapter/cache/resume regressions, repository-contract
checks and an offline CPU/int8 `tiny` smoke on the generated `ru_clean` speech unit from
`tools/fixtures/corpus-spec.json`. The real CLI run used `ru-call`, a short glossary, onset `0.45`,
offset `0.30` and chunk duration `15`; alignment completed as `aligned`, prompt usage was 67/223
tokens, and the recorded/effective settings matched. This verifies integration, not call accuracy.

Use the same non-private synthetic fixtures for regression/smoke checks and a separate permitted,
manually transcribed call corpus for quality decisions. Keep an untouched evaluation subset.
Compare baseline vs profile, then glossary, then one VAD adjustment at a time. Record WER/CER,
names/numbers/negations, short answers, punctuation, insertions on silence and timing coverage.
The existing `tools/benchmark_asr.py` scorer removes punctuation for WER/CER, so punctuation requires
a separate check. Its runner does not yet expose these tuning options; use normal CLI/Python runs
with separate output jobs for experiments. Passing contract tests or a smoke is not a quality score.
