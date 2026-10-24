# prototypes/realtime/streaming_asr

Step B2: streaming transcription and online diarization measured on our own Russian test materials.
Findings: [`docs/realtime.md`](../../../docs/realtime.md), section "Потоковая транскрибация". Nothing here is imported from `src/`.

Materials, scripts of the speech, references, stream runs and models stay in `../_local/` (git-ignored). Libraries: see
[`../requirements.txt`](../requirements.txt) (silero, silero-vad, jiwer, num2words, diart installed with `--no-deps`).
`whisper_streaming` is not a pip package: `git clone --depth 1 https://github.com/ufal/whisper_streaming ../_local/whisper_streaming`.

| file | what it does |
|---|---|
| `make_materials.py` | Silero TTS: plain-text scripts in `../_local/texts/*.txt` -> 16 kHz mix, 48 kHz copy, per-speaker stems, turn timings, reference text; `--noisy` adds noise / babble / band limit |
| `concat_materials.py` | glue materials into a longer one (32 and 60 minute recordings) |
| `stream_lib.py` | real-time feeder (100 ms chunks, wall clock = audio clock), event log, CPU/GPU/RSS monitor |
| `variant_pseudo.py` | variant A: Silero VAD cuts utterances, faster-whisper decodes each one (`--adaptive`: shorter end-of-speech wait for long utterances) |
| `variant_ufal.py` | variant B: whisper_streaming, LocalAgreement-2 on faster-whisper |
| `eval_stream.py` | WER/CER vs the Diarrhizer pipeline transcript and vs the script text; phrase latency (mean/median/p95) from word alignment |
| `gpu_load.py` | synthetic third-party GPU load (CUDA matmul at a duty cycle, optional NVENC encode = screen recording) |
| `gpu_who.py` | which processes used the GPU during a run (`typeperf` GPU Engine counters) |
| `diar_tools.py`, `diar_eval.py`, `diar_diart.py` | DER of the offline pipeline, online diart and the channel split ("me"/"others") |

Typical run (from this directory, `PYTHONUTF8=1`, interpreter `..\..\..\.venvcuda\Scripts\python.exe`):

```
python make_materials.py                      # then: make_materials.py overlap --noisy, dialog --noisy, concat_materials.py ...
python variant_pseudo.py ..\_local\materials\dialog.wav --adaptive
python variant_ufal.py   ..\_local\materials\dialog.wav
python eval_stream.py ..\_local\stream_runs\dialog.pseudo_large.json <pipeline result.json> ..\_local\materials\dialog.ref.txt
```

Latency numbers are valid only at `--speed 1.0` (the default) and with an otherwise quiet GPU: a run is repeated if `gpu_who.py`
shows another process using the GPU.
