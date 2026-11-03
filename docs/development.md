# Development and validation

The supported runtime is Python 3.11/3.12 with WhisperX 3.3.1, PySide6 6.11.2 and the checked constraints.
Use the same interpreter for installation and execution. A bare unconstrained dependency upgrade
is not a supported ML migration.

PySide6 6.12.0 passed test assertions but exited with code 1 during GUI teardown in the clean Windows
environment. It is excluded from the supported profile pending investigation; this is an observed local
compatibility failure, not a claim that every 6.12 installation is broken.

## Clean product environment

```powershell
py -3.12 -m venv .venv-check
.venv-check\Scripts\python.exe -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cpu.txt -e '.[gui,dev]'
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
