# Quality baseline and alignment loading

## C1 continuation, 2026-11-04

### Checkpoint warnings: cause, fix and evidence

On the supported torch 2.3.1 / Transformers 4.39.3 stack, the Russian checkpoint's legacy
`weight_g/weight_v` names reach Transformers' missing/unexpected-key inventory before PyTorch's
compatibility hook remaps them to `parametrizations.weight.original0/1`. The reported random
initialization was a **false warning for this checkpoint**. This order-of-loading issue is also described
in [Transformers #26796](https://github.com/huggingface/transformers/issues/26796); PyTorch documents
the legacy state-dict/name mapping in [weight_norm](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.weight_norm.html).

`ADAPTER:ALIGNMENT_LOAD` now canonicalizes complete, unambiguous pairs before the key inventory,
rejects real missing/unexpected/mismatched keys, and verifies converted tensors exactly. It uses a local
CTC subclass for the pinned loader, without global patches, warning filters, checkpoint rewrites or new
dependencies. Legacy sharded checkpoints are explicitly rejected; modern checkpoints need no rename.
Requalify this private loader hook when changing Transformers. A failure preserves source ASR text and
`alignment_status: failed`, including when raw words have no times. Empty ASR skips loading with
`not_needed`. Model name/backend, resolved HF revision and load audit are persisted in ASR metadata;
the processor and HF weights load from the same resolved revision. Torchaudio bundles keep their strict
loader and have `revision: null`. ASR cache version is `3`; merge remains `2` and schema remains `1`.

The real offline audit on RTX 3090 compared the original and canonical loaders for
`jonatasgrosman/wav2vec2-large-xlsr-53-russian`, revision
`2329100508896c6d9b157019803ab5601e6f3406`: both converted tensors matched exactly, and the maximum
logit difference on a deterministic one-second ramp was **0.0**. Unit regressions independently compare
every tensor and logits on tiny locally created CTC checkpoints. This removes misleading diagnostics
and rejects genuine loading failures; it does not establish an accuracy improvement over the old loader.

The packaged WhisperX VAD checksum was verified against its own download constant:
`0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea`.
Strict loading restored **all 54 tensors exactly**, with no extra keys. Its training metadata still says
torch `1.10.0+cu102`, pyannote.audio `0.0.1`, Lightning `1.5.4`. Those version warnings and Lightning's
in-memory migration notice remain visible. Tensor equality does not prove equivalent behavior across
training/runtime versions; no automatic library downgrade or on-disk checkpoint migration is performed.

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python tools/audit_alignment.py --device cuda --report "$env:TEMP\alignment-audit.json"
```

The audit deliberately invokes the original loader once, so its false warnings appear in this comparison.
All referenced models must be cached for offline use. Normal product runs may download missing models.

### Expanded synthetic corpus

The checked-in [specification](../tools/fixtures/corpus-spec.json) contains eight short speech units and
ten cases: clean RU/EN, numbers/names in both languages, a two-second pause, seeded 12 dB noise,
quiet stereo, alternating voices, overlapping voices and eight seconds of silence. No private recordings
are included. Speech uses **Microsoft Irina Desktop, Microsoft Zira Desktop and Microsoft David Desktop**,
rate 0; missing voices cause an explicit failure. System.Speech `SpeakProgress.AudioPosition` supplies
independent word-onset markers. Generated units are PCM16/16 kHz. Noise SNR uses whole-wave RMS,
including silence; quiet stereo applies -18 dB and halves the right channel. This is a downmix test,
not a channel-separated conversation test. The overlap case starts the second voice at 2.5 seconds.

```powershell
# Run the speech generator in Windows PowerShell 5.1; subsequent Python uses the product environment.
powershell.exe -NoProfile -File tools/make_quality_corpus.ps1 -OutputDirectory "$env:TEMP\diarrhizer-corpus"
python tools/build_quality_corpus.py "$env:TEMP\diarrhizer-corpus"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python tools/benchmark_suite.py "$env:TEMP\diarrhizer-corpus\corpus.json" --out "$env:TEMP\quality-cpu" --report "$env:TEMP\quality-cpu.json" --device cpu
python tools/benchmark_suite.py "$env:TEMP\diarrhizer-corpus\corpus.json" --out "$env:TEMP\quality-cuda" --report "$env:TEMP\quality-cuda.json" --device cuda
python tools/benchmark_suite.py "$env:TEMP\diarrhizer-corpus\corpus.json" --out "$env:TEMP\quality-repeat" --report "$env:TEMP\quality-repeat.json" --device cuda --models tiny small --repeat 3 --only ru_clean en_clean
# Requires locally configured HF_TOKEN/HUGGINGFACE_HUB_TOKEN and cached diarization models.
python tools/benchmark_suite.py "$env:TEMP\diarrhizer-corpus\corpus.json" --out "$env:TEMP\quality-diar" --report "$env:TEMP\quality-diar.json" --device cuda --models tiny small --only en_turns en_overlap --diarize
```

The manifest records source audio/marker hashes, voice, platform, NumPy and specification hash. The
runner verifies selected audio hashes and paths before inference; reports record the manifest hash and
each model/device's effective settings. Keep generated WAVs/jobs/reports outside Git. Regeneration on
another Windows/voice version may differ; compare hashes before comparing scores. All measured cases
below use the same generated bytes on CPU and CUDA.

Each repetition creates a fresh job; models and OS files can remain cached. Wall time includes conversion,
model loading, alignment and export, plus diarization when selected. Run sequentially without competing
inference. Results are saved after each case; a failed case is retained and causes a nonzero exit, as does
text loss or failed alignment. Partial alignment is reported, not promoted to a passing accuracy gate.

Metrics and limits:

- WER/CER use NFKC/casefold, punctuation-to-space and Levenshtein edits; CER excludes spaces. There is
  no number expansion or `ё`→`е` mapping. Aggregate rates divide summed edits by summed reference
  counts, excluding failed/unscored cases. Silence reports inserted words without a zero-denominator WER.
- Word-onset MAE/p95 use exact lexical matches on a minimum-edit alignment, with deterministic
  diagonal/deletion/insertion tie order. Lexical and timed coverage use **all reference words** as the
  denominator. Invalid/missing times are excluded from MAE and reduce coverage. SAPI markers are
  synthetic timing references, not human-verified acoustic boundaries; word ends are not evaluated.
- Conditional speaker error (WDER) optimally permutes predicted labels on matched words; missing labels
  count as errors. It is **not DER** and excludes unrecognized words, so always read coverage and WER.
  Overlap has no single reference text order and is deliberately unscored for text/timing/WDER. Utterance
  envelopes include silence and cannot substitute for speech-activity annotations needed by DER.
- PyTorch peak allocation excludes CTranslate2/context and is not total VRAM. Full RAM/VRAM sampling,
  human speech, channel-separated calls, overlap-aware scoring/DER and release thresholds remain open.

### Measured results

Windows 10 build 19045, Ryzen 7 5700X, RTX 3090, Python 3.12.13, WhisperX 3.3.1, faster-whisper 1.1.0,
CTranslate2 4.4.0, Transformers 4.39.3, pyannote.audio 3.3.2. CPU uses torch 2.3.1+cpu/int8/Lightning 2.6.6;
CUDA uses torch 2.3.1+cu121/float16/Lightning 2.6.5. Both use beam 5 and cached models.

| Case | Reference words | tiny CPU WER | tiny CUDA WER | Timed coverage CPU / CUDA | Onset MAE CPU / CUDA, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| ru_clean | 15 | 0% | 0% | 100% / 100% | 67 / 67 |
| en_clean | 20 | 0% | 0% | 100% / 100% | 49 / 49 |
| ru_numbers_names | 15 | 20.00% | 26.67% | 46.7% / 40.0% | 50 / 47 |
| en_numbers_names | 17 | 29.41% | 29.41% | 47.1% / 47.1% | 51 / 51 |
| ru_pause | 10 | 0% | 0% | 100% / 100% | 79 / 79 |
| ru_noise_12db | 15 | 0% | 26.67% | 100% / 73.3% | 60 / 60 |
| ru_quiet_stereo | 15 | 0% | 0% | 100% / 100% | 65 / 65 |
| en_turns | 25 | 0% | 48.00% | 100% / 52.0% | 60 / 48 |
| en_overlap | unscored | — | — | — | — |
| silence | 0 | 0 inserted words | 0 inserted words | — | — |

All 20 jobs completed and preserved normalized ASR text in exports. Numbers/names have `partial`
alignment in both profiles (one of 14 hypothesis words has no times); silence has `not_needed`.
No checkpoint-loading failures occurred. Clean RU/EN onset p95 was 104/78 ms in both profiles.
Across the eight scored cases (132 words, 767 characters), CPU WER is **8/132 = 6.06%**, CER
**43/767 = 5.61%**; CUDA WER is **25/132 = 18.94%**, CER **102/767 = 13.30%**. Across all ten cases
(103.055 seconds of audio), wall time was CPU **39.794 s**, RTF **0.386**, CUDA **33.112 s**, RTF **0.321**.
These short synthetic, differently quantized runs do not establish general CPU/GPU superiority.

Three additional CUDA repetitions of clean RU and EN for each of `tiny` and `small` all had WER 0%
(12 jobs). Wall-time ranges show process/cache effects: tiny RU 1.898–8.712 s, EN 2.033–2.230 s;
small RU 2.761–3.425 s, EN 2.551–3.219 s. No steady-state latency conclusion is drawn.

Four further CUDA jobs ran actual diarization for alternating/overlapping voices with `tiny` and `small`.
On `en_turns`, tiny had WER 48%, conditional speaker error 0% on **13/25 matched words from only one
reference speaker**; small had WER 0%, conditional speaker error 0% on **25/25 words across both
speakers**, onset MAE 60 ms. The tiny score is not evidence of successful whole-call speaker attribution.
Overlap jobs completed but remain unscored. Pyannote emitted a short-sequence `std(): degrees of freedom`
warning during diarization; it remains visible and requires longer/reference-annotated audio evaluation.

Audio SHA-256 identifiers for these measurements:

```text
ru_clean          19731022ec61e886b64bb38a9bf509a4a6a52c31c7a7b6c9ded77839a2799c3f
en_clean          d8a1fd6da1c6580a0b5fdec4072585d73494c9f8ed778f4539db3e38f9b5cc35
ru_numbers_names  3cf31c7c9a7680db8f7266a87d6d9ac217a37230cc523b0e25ab46457ed848f8
en_numbers_names  d2787af6029fc0a0601e34c8c153f589195d6ee18b4df4a3268c107fc45efe87
ru_pause          b0ec5cd87be6f1fd534ac8d60acecc32e27af7106174ce4e18aaff7fb22f9125
ru_noise_12db     3dab42e9ed9c3821d5f83ae2dae98c10b16394fa6f0297148eb3bf2756e1f7ae
ru_quiet_stereo   159500ef4397d95e26f8831e5fe737e0837db0d7c220ede99aad6f2dbad3c3a8
en_turns          6c8b0e7296e942d0e4a59e5b82330c5a42aab0aa610250ee6e1cc9ab94e255f0
en_overlap        ca73db1e9e205e7aac6a52f5e591bcad449e59fd5222cf95fc3b959190ac4f59
silence           190415015d40eef0cd797e1b59ffe00846b01873050b12d44b3a216de3faa2a9
```

## Historical single-voice baseline (B1–B4)

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

This was the initial reproducible measurement, not a quality gate for calls. The C1 continuation above
extends coverage and resolves the loading question; remaining quality gates are stated there.

### Initial measurements, 2026-11-03

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
torch/pyannote training metadata. Neither warning was resolved during B1–B4; see the C1 findings above. Successful
exports and zero missing times do **not** establish alignment accuracy; validating compatible checkpoint
loading and timestamp quality is a prerequisite for speaker-quality/DER conclusions. Application warnings
also report unsupported previous-text conditioning and VAD silence duration; effective settings record
those values as `false` and `null` respectively.
