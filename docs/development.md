# Development and validation

## Release version

The current source version is **2.3** by **MRIMSH**. `src/diarrhizer/__init__.py` defines
`__version__` for the CLI (`python -m diarrhizer --version`), GUI window/sidebar and Qt metadata,
Windows launcher and provider client identity. Setuptools reads that same value when building
package metadata; no processing libraries are imported to read the version. After changing it,
reinstall an editable package with `python -m pip install --no-deps --no-build-isolation -e .`
to refresh the installed distribution metadata used in artifact provenance.

The [version policy](../AGENTS.md#version-policy-must-follow) requires one minor increment in
every new commit, including documentation, tests and merges: `2.0 → 2.1 → 2.2` (and `2.9 → 2.10`).
Include the bump in that same commit, update current-version documentation and the changelog,
and show the full `MAJOR.MINOR` everywhere. Amending keeps one increment from the parent.
Only an explicit major-version change resets the minor component. Recorded versions work in
installed packages without Git; publishing a GitHub release remains a separate action.

The published 2.0 GitHub release uses the `chore(release): 2.0` marker commit for tag `v2.0`
and the [2.0 changelog entry](../CHANGELOG.md#20--2026-11-06) as release notes.
Build the wheel and standalone Windows launcher from that commit using the checks below and the
[launcher build instructions](launcher.md#build-and-validate).

The supported runtime is Python 3.11/3.12 with WhisperX 3.3.1, PySide6 6.11.2 and the checked constraints.
Use the same interpreter for installation and execution. A bare unconstrained dependency upgrade
is not a supported ML migration.

The [independent Windows launcher](launcher.md#build-and-validate) has a separate lightweight build
environment (`requirements/launcher-build.txt`). Its bundled Tk/Python does not depend on the processing
venv. Launcher tests use temporary configuration and synthetic downloads; they do not install a system
Python, access model caches or modify personal GUI settings. The standalone smoke opens/closes the UI
outside the checkout with Python removed from PATH. Clean-machine Python/FFmpeg bootstrap and CUDA
driver qualification remain separate integration checks.

PySide6 6.12.0 passed test assertions but exited with code 1 during GUI teardown in the clean Windows
environment. It is excluded from the supported profile pending investigation; this is an observed local
compatibility failure, not a claim that every 6.12 installation is broken.

## Clean product environment

```powershell
py -3.12 -m venv .venv-check
.venv-check\Scripts\python.exe -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cpu.txt -e '.[gui,dev,bench]'
.venv-check\Scripts\python.exe -m pip check
.venv-check\Scripts\python.exe tools/check_repository.py
.venv-check\Scripts\python.exe -m pytest -q
.venv-check\Scripts\python.exe -m diarrhizer doctor --profile export
```

Use `requirements/cuda-cu121.txt` instead of `requirements/cpu.txt` for the supported CUDA build.
Do not combine those files. `pip check` must pass in the product environment; prototype dependencies
must not be installed there. Tests use synthetic files, local fake providers and offscreen Qt.
They do not require model downloads, real credentials or a real recording.

Doctor profiles:

| Profile | Required capabilities |
| --- | --- |
| `full` (default) | Processing stack and HF token; summary/torchcodec/CUDA are optional |
| `asr` | FFmpeg and ML imports, without HF token or summary setup |
| `export` | Supported Python; no model, FFmpeg or credentials needed |
| `summary` | Supported Python and configured provider tools/key references; never calls a model |

CPU-only PyTorch is valid: doctor reports it and processing must use `--device cpu`.
Without a usable GPU, CUDA is a warning and the GUI keeps CUDA unavailable; CPU processing still works.
Individual failed imports/DLL checks
do not stop later diagnostics. An empty provider store fails the `summary` profile.

The Windows workflow checks both supported Pythons, installs the CPU stack with constraints,
runs `pip check`, tests, syntax/semantic/local-link checks and an installed-wheel CLI smoke from
outside the checkout. No CUDA inference runs on hosted CI. [Checkout](https://github.com/actions/checkout)
and [setup-python](https://github.com/actions/setup-python) are the official actions.
The repository checker registers documentary semantic anchors explicitly; `ARTIFACTS:LAYOUT`
refers to the artifact layout in [architecture.md](architecture.md). It does not validate external URLs.
Python syntax and semantic references are checked in both `src/` and `tools/`.

## Separate realtime research environment

```powershell
py -3.12 -m venv .venv-research
.venv-research\Scripts\python.exe -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cuda-cu121.txt
.venv-research\Scripts\python.exe -m pip install -r prototypes/realtime/requirements.txt
.venv-research\Scripts\python.exe -m pip check
```

Research tools can have incompatible pins. Experiments such as diart that need a different stack
belong in their own environment; `--no-deps` is not evidence that dependencies are compatible.
Do not copy the prototype's historical install commands into the product environment.
Existing `.venvcuda` installations are not automatically upgraded or cleaned by these changes.

## Before committing

Run the relevant regression tests, `tools/check_repository.py` and `git diff --check`.
Never add `.env`, model caches, recordings or generated run artifacts. Record the difference between
mock tests, a real offline ASR smoke and a quality benchmark; a successful smoke is not a WER/DER result.

## Installed wheel and configuration

Editable installs keep `.env`, `audio_formats.json`, `summary_templates.json` and `summary_providers.json`
in the checkout. A regular wheel uses `%LOCALAPPDATA%\Diarrhizer` on Windows (otherwise
`~/.config/diarrhizer`). `DIARRHIZER_CONFIG_DIR` overrides that directory; existing per-file environment
overrides still take priority. `.env` precedence is process environment, cwd `.env`, then config `.env`.
To reuse an existing checkout's settings with a wheel, explicitly set `DIARRHIZER_CONFIG_DIR` to that
checkout or copy the desired local files yourself. Secrets are never copied automatically.

## Synthetic ASR baseline

See [quality-baseline.md](quality-baseline.md) for generation, measurements and their limits.
The expanded corpus covers RU/EN, numbers/names, silence, noise, pauses, quiet stereo and two voices.
Unit regressions include real tiny local CTC checkpoints with exact tensor/logit comparisons; corpus
metrics and transforms use small synthetic arrays. No downloaded checkpoints are needed for pytest.
Requalify the private Transformers loader contract with these tests and the cached-checkpoint audit
before changing the supported ML pins.

## Annotated diarization baseline

See [diarization-quality.md](diarization-quality.md) for full DER, overlap breakdown, strict scoring
policy and a pinned public human-speech corpus. Analytic DER, RTTM/UEM, corpus validation and download
boundary tests run without network or models. Downloaded audio and model runs are explicit manual checks.
Reference RTTM uses LF via `.gitattributes` to preserve source hashes on Windows. No new dependencies
are needed: the scorer uses the existing pinned pyannote.metrics/core packages.

## Benchmark resource telemetry

The optional `bench` extra installs psutil/NVML bindings; clean development installs and Windows CI
include it for resource regressions. Tests cover deterministic counter accounting, unavailable GPU data,
exception preservation, summaries and a real child-process allocation, without CUDA or model downloads.
See [benchmark-performance.md](benchmark-performance.md) for full RSS/VRAM scopes, interval control,
observed CPU/CUDA runs and the limits of sampled peaks. Application installs do not need this extra.
