# Troubleshooting Guide

This guide covers common issues you may encounter when running Diarrhizer and how to resolve them.

---

## Quick Diagnostic

Always start by running the diagnostics:

```cmd
python -m diarrhizer doctor
```

This will verify Python version, FFmpeg, torch/torchaudio, CUDA, torchcodec, critical imports, and HF token.

---

## `pip install` fails: `No matching distribution found for whisperx==3.3.1`

**Symptom:**
```
ERROR: Cannot install whisperx==3.3.1 because these package versions have conflicting dependencies.
Additionally, some packages in these conflicts have no matching distributions available for your environment:
    whisperx
ERROR: ResolutionImpossible
```
Running `pip install whisperx==3.3.1` on its own says more: `Ignored the following versions that require a
different python version: 3.3.1 Requires-Python >=3.9,<3.13`.

**Root Cause:** the virtual environment (or the global Python) is **Python 3.13 or newer**. WhisperX 3.3.1
declares `Requires-Python <3.13`, and the pinned `torch==2.3.1` has no wheels past Python 3.12, so pip
treats every pinned version as unavailable. Pip's message does not mention the Python version.
`python -m diarrhizer doctor` reports it as well ("Python … is too new").

**Second cause - the right Python, the wrong `pip`.** The same error appears when the venv *is* 3.12
(`python --version` says so) but `pip` is not the venv's: a venv created without pip (for example
`uv venv` without `--seed`) has no `Scripts\pip.exe`, so `pip` resolves to the next one on `PATH` -
typically the global Python 3.13. Check which one runs:

```powershell
pip --version               # path must be inside your venv; 'python 3.13' means the wrong pip
python -m pip --version     # 'No module named pip' = the venv has no pip
```

Fix: give the venv a pip and always call it as `python -m pip`:

```powershell
uv pip install --python .venv\Scripts\python.exe pip      # or recreate with: uv venv --python 3.12 --seed .venv
python -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cuda-cu121.txt
```

Or skip pip: `uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cuda-cu121.txt`
(`unsafe-best-match` lets uv pick the `+cu121` torch build from the PyTorch index, as pip does).
And if the venv already has everything, `uv pip install --dry-run ...` says `Would make no changes` -
then there is nothing to install: run `python -m diarrhizer doctor`.

**Fix for the first cause:** use Python 3.12 (3.11 also works). A venv cannot be switched to another Python - recreate it:

```powershell
py -0p                                  # is there a 3.12 entry?
deactivate                              # if a venv is active
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv                  # or: uv venv --python 3.12 --seed .venv
.\.venv\Scripts\Activate.ps1
python --version                        # 3.12.x
python -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cuda-cu121.txt
```

No 3.12 installed: `winget install Python.Python.3.12`, then reopen the terminal. A `.venv` that fails
with "No Python at '...\Python312\python.exe'" was made by a Python that has since been uninstalled -
the same recreate step applies.

---

## `pkg_resources` Missing (setuptools Compatibility)

**Symptom:**
```
Import error: No module named 'pkg_resources'
```
or WhisperX fails at startup with a `ModuleNotFoundError` pointing to `pkg_resources`.

**Root Cause:** WhisperX 3.3.1 uses the `pkg_resources` module from `setuptools` at runtime. On Python 3.12+, a fresh virtual environment may not include `setuptools`, or a newer `setuptools` (81+) may have removed `pkg_resources` in favor of `importlib.metadata`.

**Fix:**

1. **Reinstall using the constraints file** (recommended — the pin is already included):
   ```powershell
   python -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt
   ```

2. **Or install setuptools directly:**
   ```powershell
   pip install "setuptools<81"
   ```

**Why `setuptools<81`?** Versions 81+ dropped `pkg_resources`. WhisperX 3.3.1 still depends on it, so a compatible `setuptools` version must be present in the environment. The constraint file pins `setuptools<81` to ensure this.

---

## Dependency Compatibility Issues

### Symptom: LazyModule / SpeechBrain Import Errors

**Error message:**
```
Transcription failed: Failed to load WhisperX model 'base':
Lazy import of LazyModule(package=None, target=speechbrain.inference, loaded=False) failed
```

Or:
```
Model was trained with torch 1.10.0+cu102, yours is 2.10.0+cu128. Bad things might happen...
Model was trained with pyannote.audio 0.0.1, yours is 3.4.0. Bad things might happen...
```

**Root Cause:** The WhisperX/pyannote/speechbrain stack has complex dependency requirements. Using the latest versions (especially torch 2.10+, numpy 2.x, transformers 5.x) can cause compatibility issues.

**Solution:**

1. **Use the stable constraints file:**
   ```cmd
   python -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt
   ```

2. **If issues persist, recreate your environment:**
   ```cmd
   # Deactivate and remove existing venv
   deactivate
   rmdir /s .venv
   
   # Create fresh environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -U pip
   python -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt
   pip install -e .
   ```

**Key constraints in `requirements/constraints-stable.txt`:**
- `setuptools<81` - WhisperX needs `pkg_resources`, removed in setuptools 81+
- `numpy<2.0` - Many ML packages still adapting to NumPy 2.0
- `huggingface-hub<1.0` - Avoids breaking changes in model loading
- `transformers<5.0` - API changes in 5.x break some integrations
- `torch>=2.0.0,<2.5.0` - More stable with WhisperX models

---

## FFmpeg Missing

**Symptom:** `check_ffmpeg` fails in `doctor` output.

**Solution:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Add FFmpeg `bin` directory to your system PATH
3. Restart your terminal/IDE

**Verify:** Run `ffmpeg -version` in a new terminal.

---

## HF Token Missing / Gated Model Access

**Symptom:** `check_hf_token` fails, or you get authentication errors when loading diarization models.

**Solution:**
1. Get a Hugging Face token from https://huggingface.co/settings/tokens
2. Set the environment variable:
   ```cmd
   set HF_TOKEN=your_token_here
   ```
   Or add to `.env` file in project root:
   ```
   HF_TOKEN=your_token_here
   ```

**Note:** The token needs access to `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.1` models.

---

## torch Stack Mismatch (CPU vs CUDA)

**Symptom:** 
- `doctor` shows torch with "(CPU only)" but you have a GPU
- CUDA available but no devices found
- Slow processing despite having GPU

**Solution:**
Reinstall torch with the correct CUDA version. The stable supported path uses CUDA 12.1 (cuDNN 8):

```powershell
pip uninstall torch torchaudio
python -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cuda-cu121.txt
```

For CPU-only (no GPU):
```powershell
python -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cpu.txt
```

**Verify:** Run `python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"`

**Important:** `torch.cuda.is_available() == True` does **not** guarantee that WhisperX / CTranslate2 can use the GPU. The cuDNN DLL mismatch is a separate, silent failure. See the next section.

---

## cuDNN DLL Mismatch (CTranslate2 vs Torch CUDA)

This is the most common GPU failure on Windows with newer PyTorch.

### Symptom

During the **transcribe** stage (WhisperX / faster-whisper / CTranslate2):

```
Could not locate cudnn_ops_infer64_8.dll. Please make sure it is in your library path!
```

Or during **diarization**:

```
Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}
Invalid handle. Cannot load symbol cudnnCreateConvolutionDescriptor
```

The `doctor` command may pass all checks (torch imports fine, CUDA reports available) — the error only occurs when WhisperX actually calls CTranslate2.

### Root Cause

**cuDNN version conflict** between PyTorch and CTranslate2:

| Package | cuDNN version bundled |
|---------|----------------------|
| `torch>=2.4.0+cu124` | cuDNN **9** |
| `torch<2.4.0` or `torch+cu121` | cuDNN **8** |
| `ctranslate2<4.5.0` (WhisperX 3.3.1) | expects cuDNN **8** |
| `ctranslate2>=4.5.0` | expects cuDNN **9** |

**WhisperX 3.3.1 explicitly requires `ctranslate2<4.5.0`**, which needs cuDNN 8.
But `torch>=2.4.0+cu124` bundles cuDNN 9 DLLs into `site-packages\torch\lib\`.
CTranslate2 4.4.0 searches for `cudnn_ops_infer64_8.dll` (cuDNN 8), doesn't find it, and crashes.

This is why `torch.cuda.is_available() == True` is not sufficient — torch itself works fine with cuDNN 9, but CTranslate2 doesn't.

### Verification Commands

Run these to diagnose:

```powershell
# 1. Check torch version and CUDA build
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available())"

# 2. Check all relevant package versions
pip show whisperx faster-whisper ctranslate2 pyannote.audio torch torchaudio

# 3. Check which cuDNN DLLs are actually in the torch package
dir .venv\Lib\site-packages\torch\lib\cudnn*.dll
```

**What to look for:**
- If step 3 shows `cudnn_ops_infer64_8.dll` → cuDNN 8, should work
- If step 3 shows only `cudnn*64_9.dll` files → cuDNN 9, **will fail** with CTranslate2 4.4.0
- If `ctranslate2` shows `4.4.0` and `torch` shows `2.4.x+cu124` → **mismatch confirmed**

### Fix Path

**Option A — Use the project's pinned setup (recommended):**

Recreate your venv with the correct pins:

```powershell
deactivate
rmdir /s .venv

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

# For GPU (CUDA 12.1, cuDNN 8):
python -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cuda-cu121.txt
pip install -e .

# Or for CPU:
python -m pip install -c requirements/constraints-stable.txt -r requirements/base.txt -r requirements/cpu.txt
pip install -e .
```

**Option B — Manual fix (if you can't recreate the venv):**

```powershell
pip uninstall torch torchaudio
pip install torch==2.3.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**Option C — Use CPU for verification:**

If you need to confirm the pipeline works before fixing the GPU stack:

```powershell
python -m diarrhizer run "path\to\file.mp4" --out ".\out" --device cpu
```

### Why You Must Recreate the Venv

Pip's dependency resolver does not retroactively downgrade packages when you change constraint files. If torch 2.4.1+cu124 is already installed, `pip install torch==2.3.1+cu121` may leave fragments. A clean venv ensures all DLLs are consistent.

### Compatibility Matrix

| torch | CUDA | cuDNN | ctranslate2 | Status |
|-------|------|-------|-------------|--------|
| 2.3.1+cu121 | 12.1 | 8 | 4.4.0 | **Stable (recommended)** |
| 2.4.1+cu124 | 12.4 | 9 | 4.4.0 | Broken — cuDNN mismatch |
| 2.4.1+cu124 | 12.4 | 9 | >=4.5.0 | Would work but WhisperX 3.3.1 forbids it |
| 2.3.1+cpu | — | — | 4.4.0 | Stable (CPU only) |

---

## torchcodec / FFmpeg Decoding Issues

**Symptom:** 
- `doctor` shows torchcodec as "Not installed"
- Audio decoding errors or warnings
- Slow waveform loading

**Solution (optional):**
torchcodec provides fast audio decoding. Install it with:

```cmd
pip install torchcodec
```

**Fallback:** If torchcodec fails to install, Diarrhizer will automatically use waveform preload (slower but functional). No action needed.

**Note:** torchcodec requires FFmpeg libraries. On Windows, you may need Visual C++ redistributables.

---

## Word Alignment Failures

**Symptom:**
```
WARNING:root:Word alignment failed: load_align_model() got an unexpected keyword argument 'language'
```

**Solution:** This is usually caused by WhisperX API changes. The fix has been applied in Diarrhizer v0.1.1+. If you see this error, ensure you're using the latest version:
```cmd
pip install -U diarrhizer
```

---

## Compute Type Errors

**Symptom:**
```
ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation.
```

**Solution:** This happens when using CPU with float16 compute type. Ensure you're using the correct device flag:
- For GPU: `--device cuda` (uses float16)
- For CPU: `--device cpu` (uses int8)

---

## Known Issues with Version Combinations

| Version | Status | Notes |
|---------|--------|-------|
| torch 2.3.1+cu121 + ctranslate2 4.4.0 | **Stable** | Recommended GPU path (cuDNN 8) |
| torch 2.4.1+cu124 + ctranslate2 4.4.0 | **Broken** | cuDNN 9 vs 8 mismatch → `cudnn_ops_infer64_8.dll` error |
| torch 2.4.x+cu124 + ctranslate2 >=4.5.0 | Would work | Blocked by WhisperX 3.3.1 `ctranslate2<4.5.0` |
| torch 2.10 + whisperx 3.7 | Works with warnings | Shows model version mismatch warnings |
| numpy 2.x | May work | Some packages still adapting |
| transformers 5.x | May work | API changes in 5.x |
| huggingface-hub 1.5+ | May work | Lazy loading changes |
| pyannote.audio 3.3.2 | **Stable** | Current default |
| speechbrain 1.0.x | **Stable** | Current default |

---

## Summaries: sign-in, models and limits of ChatGPT / Claude providers

Where the **Саммаризация → Провайдеры** screen shows an error, this is what usually stands behind it.

| What you see | Why | What to do |
|--------------|-----|------------|
| `command 'codex' not found` / `command 'claude' not found` | The vendor's tool is not installed or not on `PATH` | See [the next section](#command-codex-not-found-although-codex-is-installed): find the file and either give its full path in the provider's **Команда** field or put its folder on `PATH` |
| **Войти через ChatGPT**: the page opens but the window does not update | Codex waits for the browser on `localhost:1455`; another process (a stuck `codex login`, another Codex window) may hold that port | Close the other Codex process, press **Отменить вход**, try again |
| **Войти через Claude**: no browser page | Claude Code could not open the browser | Press **Отменить вход**, run `claude auth login` in a terminal (it also lets you paste the code from the page), then **Проверить вход** |
| `The '<model>' model is not supported when using Codex with a ChatGPT account` | The chosen model is not available for the signed-in plan | Press **Список моделей** and pick one from the list, or choose the tool's default |
| Signed in "по API-ключу, а не по подписке" | The tool has an API key stored instead of a subscription sign-in, so requests are billed per token | **Выйти**, then sign in with the button |
| Claude: **Обновить лимиты** takes seconds and uses a little quota | Claude Code reports limits only with a reply, so a one-word request is sent | Expected; Codex's limits are a local query and cost nothing |
| Limits or cost missing from a summary | The provider did not report them (Codex reports no cost; an API provider reports cost only with prices set or from OpenRouter-style replies) | Nothing is invented; set prices on an API provider to get an estimate |

Signing out in the app signs the tool out for everything that uses it (the tool keeps one stored sign-in).
The app never sees or stores the tokens: for Claude it only starts Anthropic's own `claude auth login`,
because Anthropic does not allow other applications to offer a Claude.ai sign-in themselves.

---

## `command 'codex' not found` although Codex is installed

**Symptom** (the account box of a provider on **Саммаризация → Провайдеры**, or a failed summary):

```
Вход не выполнен: Provider 'ChatGPT': command 'codex' not found - it is not on PATH. ...
```

Everything below is the same for Claude: read `claude` instead of `codex`, **Войти через Claude** instead of
**Войти через ChatGPT**.

**Cause.** Diarrhizer starts Codex by name, as a terminal would, and finds it only through `PATH`. Installing
the program does not always put it there:

- the Codex **desktop app** (and editor extensions) keep their own `codex.exe` in a folder that is not on `PATH`;
- an **npm** install puts `codex.cmd` into npm's global folder, which is on `PATH` only if it was added;
- a terminal (or a Diarrhizer window) opened **before** the installation or before `PATH` was changed still
  has the old `PATH` - a running program never sees a later change;
- in PowerShell `codex` may work only as a *function or alias* from your profile; other programs cannot
  see those;
- Codex installed inside **WSL** (Linux) is not a Windows program: Diarrhizer cannot start it by name.

**Fix.**

**1. Is it really missing?** In a *new* PowerShell window:

```powershell
Get-Command codex -All | Format-Table CommandType, Source
```

- A row of type `Application` (`codex.exe` or `codex.cmd`): Codex *is* on `PATH` - the Diarrhizer window just
  predates it. Close Diarrhizer and start it again.
- Only `Function` / `Alias` rows, or nothing: other programs cannot start it. Go on.

**2. Find the real file.** Where it usually lives on Windows:

| Installed as | The file |
|--------------|----------|
| npm (`npm install -g @openai/codex`) | `%APPDATA%\npm\codex.cmd` (`npm config get prefix` prints the folder) |
| Codex desktop app | `%LOCALAPPDATA%\OpenAI\Codex\bin\<folder-name>\codex.exe` - the folder name changes with app updates |
| Claude Code, native installer | `%USERPROFILE%\.local\bin\claude.exe` |
| Claude Code, npm | `%APPDATA%\npm\claude.cmd` |

Or let PowerShell look through those places (a second or two):

```powershell
$places = "$env:APPDATA\npm", "$env:LOCALAPPDATA\OpenAI\Codex\bin", "$env:LOCALAPPDATA\Programs", "$env:USERPROFILE\.local\bin"
Get-ChildItem $places -Recurse -Depth 5 -Include codex.exe, codex.cmd, claude.exe, claude.cmd -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
```

If several files turn up, prefer `codex.cmd` of npm or the `codex.exe` of the app, not the `codex.exe` buried deep
inside `node_modules` (it works, but its folder changes with every npm update). Do **not** take the copies under
`%USERPROFILE%\.codex\` - those are the tool's internal helpers. Nothing found? It may be installed elsewhere:
search the folder you chose at installation (`Get-ChildItem "D:\Tools" -Recurse -Include codex.exe`) - a search
of the whole `%LOCALAPPDATA%` takes many minutes. Check the file you picked:

```powershell
& "C:\full\path\to\codex.exe" --version
```

It must print a version. If the file cannot be found anywhere, the command-line tool is not installed (some
apps do not include it): install it (`npm install -g @openai/codex`, needs Node.js) and repeat step 1.

**3. Tell Diarrhizer where it is** - either way works:

- **A. The full path in the provider (simplest).** **Саммаризация → Провайдеры** → the provider → field
  **Команда** → paste the full path → **Сохранить** → **Проверить вход** in the account box. Spaces in the path
  are fine, and so are the quotes that Explorer's "Copy as path" adds. Downside: the Codex desktop app's folder
  name changes with updates - after an update the path goes stale, repeat the step (or use B).
- **B. Put the folder on `PATH`** (every program then finds it; leave **Команда** empty). Only your own `PATH`
  changes, no administrator rights are needed. Press `Win+R`, run

  ```
  rundll32 sysdm.cpl,EditEnvironmentVariables
  ```

  In the **upper** list (variables of your account) select `Path` → **Изменить** → **Создать** → paste the
  folder that holds `codex.exe` / `codex.cmd` (the folder, not the file) → **OK** in every window.

  Then **close every terminal and Diarrhizer and start them again**, and check in a new terminal that
  `codex --version` works.

  If you prefer PowerShell, this appends the folder to your `PATH`:

  ```powershell
  $dir = "C:\full\path\to\the\folder"
  $old = [Environment]::GetEnvironmentVariable("Path", "User")
  [Environment]::SetEnvironmentVariable("Path", "$old;$dir", "User")
  ```

  It works, but it writes your `PATH` back as plain text: entries such as `%USERPROFILE%\bin` are replaced by
  the paths they stand for (the dialog above keeps them). Do not use `setx PATH ...`: it cuts the value at 1024
  characters and merges the machine-wide and your own lists.

If the account box then says **Вход не выполнен** (without "not found"), the tool is found - press the sign-in
button. A message that `'...' is a folder, not a program` means the path points at the folder: add the file name.
