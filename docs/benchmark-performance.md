# Benchmark RAM, VRAM and performance (C1)

ASR (`benchmark_asr.py`, `benchmark_suite.py`) and DER (`benchmark_diarization.py`) now share
`tools/benchmark_resources.py`. Each run stores `performance` alongside its accuracy metrics, with
OS/native memory, observed CPU time, whole-device GPU memory, wall time and sampling coverage.
The old `pytorch_peak_allocated_bytes` remains a separately labelled PyTorch-only ASR diagnostic.

## Install and run

Use an existing supported CPU or CUDA product environment. Telemetry is an optional `bench` extra:
`psutil 7.2.2` provides process-tree/OS counters, and `nvidia-ml-py 13.615.71` provides NVIDIA's NVML
bindings. They avoid hand-written OS wrappers and per-sample subprocesses. The core application does
not depend on this extra. The supported constraints and Windows CI include the validated versions.

```powershell
python -m pip install -c requirements/constraints-stable.txt -e '.[bench]'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
# Generate/download the corpora using the quality guides linked below first.
python tools/benchmark_suite.py "$env:TEMP\diarrhizer-c1-corpus\corpus.json" --out "$env:TEMP\perf-asr-jobs" --report "$env:TEMP\perf-asr.json" --device cuda --only ru_clean en_clean --repeat 3 --sample-interval 0.1
python tools/benchmark_diarization.py "$env:TEMP\diarrhizer-c1-der-corpus\corpus.json" --out "$env:TEMP\perf-der-jobs" --report "$env:TEMP\perf-der.json" --device cuda --only abjxc --repeat 2 --sample-interval 0.1
```

Run the same commands with the **CPU environment's interpreter**, `--device cpu` and separate
output/report paths. The single-file ASR tool accepts `--sample-interval` too. Valid values are 0.02–10
seconds. Corpus generation, references and scoring are documented in [ASR quality](quality-baseline.md)
and [DER quality](diarization-quality.md). Audio, model caches and generated reports stay outside Git.

## What is measured

| Field under `performance` | Meaning |
| --- | --- |
| `wall_seconds` | Timed pipeline call, including conversion, model load, ASR/alignment or diarization, and selected pipeline output writes |
| `rtf` | Wall seconds / audio seconds; smaller is faster |
| `audio_seconds_per_wall_second` | Audio seconds / wall seconds; larger is faster |
| `cpu.observed_tree_seconds` | Sum of observed user+system CPU deltas for the process and descendants |
| `cpu.average_cores` | Observed CPU seconds / wall seconds; can exceed 1 |
| `cpu.affinity_normalized_percent` | Average cores / available affinity CPUs × 100 |
| `ram.start_rss_bytes`, `end_rss_bytes`, `peak_rss_bytes` | Resident memory summed over concurrently observed processes; includes native libraries and allocator memory |
| `ram.peak_increase_bytes` | Sampled RSS peak minus initial RSS; not a measure of private ownership |
| `ram.peak_private_commit_bytes` | Windows private committed memory summed over the tree; includes nonresident pages, separately from RSS; null on other platforms |
| `ram.system_total_bytes`, `system_used_peak_bytes`, `system_available_min_bytes` | Whole-machine memory context, including unrelated programs |
| `gpu.devices[]` | Per physical GPU: UUID/index/name, total bytes, initial/final/peak occupied framebuffer bytes, peak increase, initial/sample-mean utilization and valid/failed sample counts |
| `sampling` | Requested delay, sample count, largest observed sample gap and largest read duration |
| `host`, `gpu.driver` | OS/CPU identity, CPU counts/affinity, telemetry library versions and GPU driver |
| `warnings` | Missing counters, partial samples or a GPU busy at baseline |

The [psutil counters](https://psutil.readthedocs.io/stable/) include native allocations outside Python.
RSS can count shared pages in several processes; it is not USS. Windows `memory_info().vms` maps to
private committed/pagefile usage, rather than Linux virtual address-space size. A process can commit
more memory than is resident; RSS can also include shared mapped libraries. Keep these quantities separate.
Process identity includes creation time, so PID reuse cannot merge unrelated CPU counters. CPU work from
an observed child is retained after it exits; a Windows venv launcher and its interpreter are both included.

VRAM uses [NVML `nvmlDeviceGetMemoryInfo`](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).
It covers the **whole physical device**, including allocations from CTranslate2 and CUDA contexts; all
applications contribute. It is not a per-process attribution, and subtracting the baseline does not remove
concurrent allocation changes. All physical NVIDIA devices are reported by UUID, independently of CUDA's
logical device ordering or `CUDA_VISIBLE_DEVICES`. On Windows/WDDM,
[NVIDIA documents per-process memory as unavailable](https://docs.nvidia.com/deploy/nvidia-smi/).
Shared GPU system memory is not added to dedicated VRAM; host RAM counters already provide its OS context.
AMD/iGPU telemetry is not implemented. Driver/permission/API failures produce missing data and warnings,
never fabricated zero usage. Unsupported utilization alone does not discard a valid memory reading.

Sampling captures a baseline before the timer, then runs in a background thread. The timer stops when the
pipeline returns/raises; a final sample is taken after the sampler is joined. Initialization, imports/stage
construction, final telemetry, result-file reading and benchmark accuracy scoring are excluded from wall
time. Model loading **inside** the pipeline remains included. Background observer overhead is included;
CPU counter endpoints also include a little boundary-query work. This is pipeline throughput, not kernel
latency or total CLI startup time. No CUDA initialization is needed for CPU resource measurement.

The sampler waits the requested interval after each read, so real gaps also include read/scheduling delay.
**Peaks are sampled lower bounds**, not guaranteed allocation maxima. Short spikes or children that exit
between samples can be missed. Lost process reads and actual gaps are reported. Whole-machine used RAM
and process-tree RSS have different scopes and must not be added together. A pure process-lifetime peak
would contaminate later runs in a persistent interpreter and is deliberately not used as a per-case peak.

Successful-call aggregates use total wall / total audio for RTF, maximum observed memory peaks across runs,
and per-case wall median/nearest-rank p95. They also expose missing measurement coverage; legacy reports
without `performance` remain readable. Failed pipeline exceptions retain their resource report in the
failed case but do not enter successful-call summaries. Collection/cleanup failures preserve the workload's
original result or error. Missing psutil is an actionable install error before pipeline execution.

For comparisons, use identical audio hashes, task/stage settings, model and precision, sample interval,
speaker bounds and CPU affinity. Run serially on a quiet host. Each repetition creates a fresh job, but
Python/native heaps and OS/model file caches can remain warm. Keep the first ordered repetition visible;
do not silently discard it or label later repetitions as a cold process. Two or three repetitions do not
establish a confidence interval. NVML utilization is the driver's own sampling window, not CUDA kernel time.

## Observed CPU/CUDA runs, 2026-11-04

Windows 10 build 19045, Ryzen 7 5700X / 16 logical CPUs, approximately 31.93 GiB system RAM,
RTX 3090 / 24 GiB, driver 610.88, Python 3.12.13, existing supported ML pins. CPU ASR uses tiny/int8,
CUDA ASR tiny/float16. Both use the same clean RU/EN WAV bytes and prior references. DER uses raw audio,
speaker bounds 1–10 and strict overlap-aware scoring on the complete 68.376-second `abjxc` recording.

These measurements were collected while **other applications were active**. GPU baseline usage ranged
from approximately 8 to 13 GiB; some utilization samples approached saturation. They demonstrate telemetry
under actual load and are **not a controlled CPU/GPU ranking**. Reported VRAM below includes those programs.
Use a quiet host before setting acceptance thresholds.

Twelve ASR runs, three ordered repetitions of each case/device, requested interval 0.1 s:

| Case / device | Wall seconds, repetitions 1 / 2 / 3 | Median, s | RSS peak across runs, GiB | Whole-GPU occupied peak, GiB |
| --- | --- | ---: | ---: | ---: |
| RU / CPU | 66.577 / 3.992 / 3.632 | 3.992 | 3.492 | 12.558 |
| EN / CPU | 2.813 / 2.351 / 2.439 | 2.439 | 2.451 | 8.322 |
| RU / CUDA | 84.145 / 5.112 / 4.396 | 5.112 | 3.709 | 14.274 |
| EN / CUDA | 6.388 / 3.033 / 4.585 | 4.585 | 2.638 | 13.462 |

Across six runs per device (65.925 audio seconds each), CPU wall time was 81.803 s / RTF 1.241,
CUDA 107.658 s / RTF 1.633. All twelve retained WER 0% and preserved exported text. Large first-run
latencies and competing load prevent a general speed conclusion. Sample counts were 18–592, largest
observed gaps 0.195–0.524 s. These earlier ASR reports contain RSS; the Windows private-commit field was
added subsequently and is validated in the DER measurements below.

Four DER runs, two repetitions per device on identical `abjxc` bytes:

| Device / repetition | Wall, s | RTF | Observed CPU, s | RSS peak, GiB | Private commit peak, GiB | Whole-GPU baseline → peak, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU / 1 | 48.841 | 0.7143 | 356.719 | 2.105 | 3.106 | 8.048 → 8.266 |
| CPU / 2 | 38.918 | 0.5692 | 322.625 | 2.118 | 3.119 | 8.019 → 8.020 |
| CUDA / 1 | 29.420 | 0.4303 | 21.938 | 1.316 | 4.639 | 8.081 → 10.296 |
| CUDA / 2 | 1.898 | 0.0278 | 9.219 | 1.366 | 4.767 | 8.355 → 10.296 |

DER stayed 0.6030% in all four jobs. Weighted RTF was CPU 0.642 / CUDA 0.229; this is one recording,
with different first-run overhead and ambient load. Samples ranged from 15 to 325, largest gaps from
0.199 to 0.419 s. No model settings or accuracy thresholds were adjusted based on these measurements.
Generated JSON reports retain byte counts, input hashes, counters and warnings for local inspection.
