# Synthetic ASR baseline

The initial fixture is generated locally from [a short Russian reference](../tools/fixtures/ru-synthetic.txt)
using Windows System.Speech, Microsoft Irina Desktop, rate 0. It contains no personal recording. Run the
generator with **Windows PowerShell 5.1** (`powershell.exe`, not `pwsh`); the selected voice must be installed.

```powershell
powershell.exe -NoProfile -File tools/make_synthetic_fixture.ps1 -OutputDirectory "$env:TEMP\diarrhizer-baseline"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python tools/benchmark_asr.py "$env:TEMP\diarrhizer-baseline\ru-synthetic.wav" --reference tools/fixtures/ru-synthetic.txt --out "$env:TEMP\diarrhizer-baseline\out" --report "$env:TEMP\diarrhizer-baseline\baseline.json" --device cuda --model tiny
```

Models must already be cached for offline execution. CPU is supported with `--device cpu` (int8 rather
than float16). Each measurement creates a fresh ASR-only job with WAV retained. The report includes audio
and reference hashes, package versions, effective ASR settings, warnings, WER/CER, words missing timestamps,
text preservation through export and wall time / audio duration (RTF). Wall time includes model loading,
conversion, alignment and export; it is not steady-state streaming throughput. Avoid concurrent heavy work
when comparing timings. The voice/Windows/model versions can change the fixture or result; compare hashes.

Scoring uses Unicode NFKC/casefold, replaces punctuation with spaces and compares letters/numbers.
WER is word Levenshtein edits / reference words; CER excludes spaces. Values can exceed 1 due to insertions.
There is no number expansion, transliteration or `ё`→`е` mapping. Empty references are rejected.

`pytorch_peak_allocated_bytes` measures **only the PyTorch CUDA allocator**. It excludes CTranslate2,
the CUDA context and other allocations; it is not total peak VRAM. CPU memory and total VRAM remain
unmeasured. No speaker metric/DER is produced for this single-voice ASR-only fixture.

This is an initial reproducible measurement, not a quality gate for calls. Next: English, numbers/names,
silence/noise, overlapping speakers with time-labelled references, channel-separated recordings, several
model sizes and repeated runs; then establish WER/CER/DER thresholds and full RAM/VRAM sampling.

## Initial measurements, 2026-11-03

Windows 10 build 19045, Ryzen 7 5700X, RTX 3090, Python 3.12.13, WhisperX 3.3.1,
faster-whisper 1.1.0, CTranslate2 4.4.0, pyannote.audio 3.3.2, transformers 4.39.3.
Both used `tiny`, Russian, beam 5 and offline cached models. The source is 11.153 s, 15 reference words,
99 reference characters; generated audio SHA-256:
`52eea71a2eab20bbec09fd11cbd75f1461ab65cb5a5579ab8abf5a144fa8c64c`.

| Environment | WER | CER | Wall time | RTF | PyTorch peak allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing CUDA, torch 2.3.1+cu121, float16 | 0 / 15 = 0% | 0 / 99 = 0% | 55.288 s | 4.957 | 1,479,882,752 bytes |
| Clean CPU, torch 2.3.1+cpu, int8 | 2 / 15 = 13.33% | 1 / 99 = 1.01% | 11.486 s | 1.030 | Not measured |

Both preserved normalized ASR text in exports and reported no words with missing timestamps.
These are **one run per environment**, with different precision and cold/warm OS caches; they do not
establish that CPU is faster than GPU or that one model configuration has superior general quality.
Lightning versions were 2.6.5 (existing CUDA) and 2.6.6 (clean CPU).

The Russian alignment checkpoint reported unused `weight_g/weight_v` and newly initialized
`parametrizations.weight.original*` during both runs. WhisperX's VAD checkpoint also carries older
torch/pyannote training metadata. Neither warning was fixed or suppressed in this cycle. Successful
exports and zero missing times do **not** establish alignment accuracy; validating compatible checkpoint
loading and timestamp quality is a prerequisite for speaker-quality/DER conclusions. Application warnings
also report unsupported previous-text conditioning and VAD silence duration; effective settings record
those values as `false` and `null` respectively.
