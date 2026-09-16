"""Cache-staleness helper shared by pipeline stages."""

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence


# [SEMANTIC-BEGIN] PIPELINE:CACHE
# @purpose: Decide whether a stage's cached output artifacts are still valid
# @description: Compares output mtimes against input mtimes so that recomputing
#   an upstream stage's output automatically invalidates every downstream
#   stage that consumed it, without the runner tracking stage dependencies.
#   config_mismatch() compares stored vs current stage config (legacy missing
#   keys are ignored); stages also pass the original input file / speakers map.
# @inputs: output artifact paths, input artifact paths, optional stored/expected config
# @outputs: bool (True = stale, must recompute)
# @sideEffects: None (pure function; only stats files)
# @errors: None
# @see: PIPELINE:RUNNER, STAGE:CONVERT, STAGE:TRANSCRIBE, STAGE:DIARIZE, STAGE:MERGE, STAGE:EXPORT
def is_stale(outputs: Sequence[Path], inputs: Sequence[Path]) -> bool:
    """Check whether cached output artifacts need to be recomputed.

    Output is considered stale if any output artifact is missing, or if any
    existing input artifact was modified more recently than the oldest output
    artifact. This lets recomputing an upstream stage (e.g. via --force-stage)
    automatically invalidate every downstream stage that consumed its output,
    without the runner having to track stage dependencies explicitly.

    Args:
        outputs: Paths this stage is expected to have produced
        inputs: Paths this stage consumed to produce its outputs

    Returns:
        True if the stage should be recomputed, False if the cache is valid
    """
    if not outputs or any(not p.exists() for p in outputs):
        return True

    existing_inputs = [p for p in inputs if p.exists()]
    if not existing_inputs:
        # Nothing to compare freshness against; a genuinely missing input is
        # surfaced by the stage's own run() with a clearer error message.
        return False

    oldest_output_mtime = min(p.stat().st_mtime for p in outputs)
    newest_input_mtime = max(p.stat().st_mtime for p in existing_inputs)
    return newest_input_mtime > oldest_output_mtime


def prompt_digest(prompt: str | None) -> str | None:
    """Stable hash of an ASR initial prompt, or None when unset."""
    if not prompt:
        return None
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def config_mismatch(stored: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> bool:
    """True when a stored artifact config disagrees with the current run.

    Keys present in `expected` but missing from `stored` are ignored so
    artifacts written before a field existed stay valid (mtime still rules).
    Keys present in both must be equal.
    """
    if stored is None:
        return False
    for key, value in expected.items():
        if key not in stored:
            continue
        if stored[key] != value:
            return True
    return False


# [SEMANTIC-END] PIPELINE:CACHE
