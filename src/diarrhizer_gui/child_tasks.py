"""Work the GUI runs in a spawned child process (see child_process.py).

Qt-free on purpose: a spawned child starts a fresh interpreter and imports
this module to find its target, so everything here must be cheap to import.
The heavy imports (torch, whisperx, pyannote) happen inside the targets,
i.e. only in the child - the GUI process never loads the ML stack.

Each target's first argument is a multiprocessing queue; the child reports
through it with small picklable tuples:
    ("progress", stage, message)       - one pipeline log line
    ("finished", result_dict)          - run_pipeline() / a summary / a provider check returned
    ("failed", error_type, message)    - run_pipeline() / a summary / a provider check raised
    ("checks", [(name, passed, msg)])  - doctor results
    ("event", name, payload)           - something to show while a task runs (a sign-in address)
"""

import logging
import os
import sys


def build_stages(mode: str) -> list:
    """Build the stage list for a pipeline mode. Imports stage classes lazily."""
    from diarrhizer.pipeline.stages.convert import ConvertStage
    from diarrhizer.pipeline.stages.export import ExportStage
    from diarrhizer.pipeline.stages.merge import MergeStage
    from diarrhizer.pipeline.stages.transcribe import TranscribeStage

    if mode == "asr_only":
        return [ConvertStage(), TranscribeStage(), MergeStage(), ExportStage()]

    if mode == "full":
        from diarrhizer.pipeline.stages.diarize import DiarizeStage

        return [
            ConvertStage(),
            TranscribeStage(),
            DiarizeStage(),
            MergeStage(),
            ExportStage(),
        ]

    raise ValueError(f"Unknown pipeline mode: {mode!r}")


def _ensure_std_streams() -> None:
    """A child of pythonw.exe has no console: sys.stdout/stderr are None, and
    tqdm progress bars (model downloads, pyannote) crash writing to None."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


class _QueueLogHandler(logging.Handler):
    """Forwards "diarrhizer" log records (stage progress) to the parent."""

    def __init__(self, queue) -> None:
        super().__init__()
        self._queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put(("progress", getattr(record, "stage", "-"), record.getMessage()))


# [SEMANTIC-BEGIN] GUI:CHILD_TASKS
# @purpose: Entry points for the GUI's child processes: one pipeline job, the doctor checks, a summary or a provider check
# @description: run_pipeline_task() runs run_pipeline() with the given stage mode and kwargs and reports
#   progress/outcome over the queue; it catches the same exceptions as the CLI, so only a hard crash
#   (e.g. a native CUDA/cuDNN abort) ends the process without a final message - the parent reports that
#   as a crash. run_doctor_task() runs doctor.run_all_checks() in a fresh interpreter, so the GUI never
#   imports torch/whisperx itself and a re-check sees packages installed since the GUI started.
#   run_summary_task() / run_provider_check_task() run summary.service.summarize_job() / check_provider()
#   with a template and provider passed as dicts; being a process, the model call can be cancelled.
#   run_account_task() runs one account action of a subscription provider (status, sign in, sign out, model
#   list, limits - ADAPTER:LLM_ACCOUNTS); a sign-in reports the address to open as an ("event", ...) message.
# @inputs: queue, mode + run_pipeline() kwargs (picklable) | queue | job_dir + template/provider dicts | provider dict + env
# @outputs: tuples on the queue (see module docstring)
# @sideEffects: Everything run_pipeline does (files, GPU, model downloads); doctor's imports and probes;
#   a summary sends the transcript to the provider and writes the job's summary/ folder
# @errors: None raised - reported over the queue
# @see: GUI:CHILD_PROCESS, GUI:PIPELINE_WORKER, GUI:LLM_WORKER, PIPELINE:RUNNER, DIAGNOSTICS:DOCTOR, SUMMARY:JOB, ADAPTER:LLM_ACCOUNTS
def run_pipeline_task(queue, mode: str, kwargs: dict) -> None:
    _ensure_std_streams()
    from diarrhizer.pipeline.runner import run_pipeline

    logger = logging.getLogger("diarrhizer")
    logger.addHandler(_QueueLogHandler(queue))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        result = run_pipeline(stages=build_stages(mode), **kwargs)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        queue.put(("failed", type(exc).__name__, str(exc)))
        return
    except Exception as exc:  # matches cli.py's catch-all for unexpected failures
        queue.put(("failed", type(exc).__name__, f"{type(exc).__name__}: {exc}"))
        return
    queue.put(("finished", result))


def run_doctor_task(queue) -> None:
    _ensure_std_streams()
    from diarrhizer.diagnostics import doctor

    queue.put(("checks", doctor.run_all_checks()))


# run_summary_task / run_provider_check_task: a model call can take minutes, and a subscription CLI starts
# grandchildren of its own - a child process can be killed with its whole tree (GUI:CHILD_PROCESS), a thread
# stuck in a socket read cannot. They receive templates/providers as dicts (their to_dict()), so what the
# user sees on screen is what is used, saved or not.
def run_summary_task(queue, job_dir: str, template: dict, provider: dict) -> None:
    _ensure_std_streams()
    from diarrhizer.summary import service
    from diarrhizer.summary.providers import Provider
    from diarrhizer.summary.templates import PromptTemplate

    try:
        result = service.summarize_job(job_dir, PromptTemplate.from_dict(template), Provider.from_dict(provider))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:  # RuntimeError covers adapters.llm.LLMError
        queue.put(("failed", type(exc).__name__, str(exc)))
        return
    except Exception as exc:  # matches cli.py's catch-all for unexpected failures
        queue.put(("failed", type(exc).__name__, f"{type(exc).__name__}: {exc}"))
        return
    queue.put(("finished", result))


def run_provider_check_task(queue, provider: dict, env: dict) -> None:
    """Send a trivial prompt through the provider. `env` is added to the child's environment: an API key
    typed on screen but not saved yet."""
    _ensure_std_streams()
    os.environ.update(env)
    from diarrhizer.summary import service
    from diarrhizer.summary.providers import Provider

    try:
        result = service.check_provider(Provider.from_dict(provider))
    except (RuntimeError, ValueError) as exc:
        queue.put(("failed", type(exc).__name__, str(exc)))
        return
    except Exception as exc:
        queue.put(("failed", type(exc).__name__, f"{type(exc).__name__}: {exc}"))
        return
    queue.put(("finished", result))


# The account actions a provider's panel offers. All of them talk to the vendor's tool (or its API) and a
# sign-in waits for the user, so they run in a child process too: cancelling kills the tool, which also frees
# the local port its sign-in listens on. Results are plain dicts (to_dict() shapes) for the queue.
ACCOUNT_ACTIONS = ("status", "login", "logout", "models", "limits")


def run_account_task(queue, action: str, provider: dict, env: dict) -> None:
    """One of ACCOUNT_ACTIONS for a provider (a to_dict() of it); `env` is added to the environment first
    (an API key typed on screen but not saved yet). A sign-in also reports ("event", "url" | "waiting", text)."""
    _ensure_std_streams()
    os.environ.update(env)
    from diarrhizer.adapters import llm_accounts
    from diarrhizer.summary.providers import Provider

    try:
        target = Provider.from_dict(provider)
        if action == "status":
            result = {"account": llm_accounts.account_status(target).to_dict()}
        elif action == "login":
            result = {"account": llm_accounts.login(
                target, on_event=lambda name, payload: queue.put(("event", name, payload))
            ).to_dict()}
        elif action == "logout":
            result = {"account": llm_accounts.logout(target).to_dict()}
        elif action == "models":
            result = {"models": [model.to_dict() for model in llm_accounts.list_models(target)]}
        elif action == "limits":
            result = {"limits": llm_accounts.read_limits(target).to_dict()}
        else:
            raise ValueError(f"Unknown account action: {action!r}")
    except (RuntimeError, ValueError) as exc:  # RuntimeError covers adapters.llm.LLMError
        queue.put(("failed", type(exc).__name__, str(exc)))
        return
    except Exception as exc:
        queue.put(("failed", type(exc).__name__, f"{type(exc).__name__}: {exc}"))
        return
    queue.put(("finished", result))
# [SEMANTIC-END] GUI:CHILD_TASKS
