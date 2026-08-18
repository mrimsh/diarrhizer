"""Tests for diarrhizer.pipeline.cache.is_stale."""

import os
import time

from diarrhizer.pipeline.cache import is_stale


def touch(path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def set_mtime(path, offset_seconds):
    """Set a path's mtime to `offset_seconds` relative to now (can be negative)."""
    t = time.time() + offset_seconds
    os.utime(path, (t, t))


def test_missing_output_is_stale(tmp_path):
    output = tmp_path / "out.json"
    assert is_stale(outputs=[output], inputs=[]) is True


def test_no_inputs_output_exists_is_not_stale(tmp_path):
    output = touch(tmp_path / "out.json")
    assert is_stale(outputs=[output], inputs=[]) is False


def test_output_newer_than_input_is_not_stale(tmp_path):
    inp = touch(tmp_path / "in.json")
    out = touch(tmp_path / "out.json")
    set_mtime(inp, -10)
    set_mtime(out, -5)
    assert is_stale(outputs=[out], inputs=[inp]) is False


def test_input_newer_than_output_is_stale(tmp_path):
    inp = touch(tmp_path / "in.json")
    out = touch(tmp_path / "out.json")
    set_mtime(out, -10)
    set_mtime(inp, -5)
    assert is_stale(outputs=[out], inputs=[inp]) is True


def test_missing_input_does_not_force_recompute(tmp_path):
    # A declared input that doesn't exist on disk shouldn't be treated as
    # "infinitely new" - the stage's own run() surfaces a clearer error for a
    # genuinely missing input than a silent recompute would.
    out = touch(tmp_path / "out.json")
    missing_input = tmp_path / "does_not_exist.json"
    assert is_stale(outputs=[out], inputs=[missing_input]) is False


def test_multiple_outputs_uses_oldest(tmp_path):
    inp = touch(tmp_path / "in.json")
    out1 = touch(tmp_path / "out1.json")
    out2 = touch(tmp_path / "out2.json")
    set_mtime(inp, -5)
    set_mtime(out1, -10)  # older than input -> should make the pair stale
    set_mtime(out2, 0)
    assert is_stale(outputs=[out1, out2], inputs=[inp]) is True


def test_multiple_inputs_uses_newest(tmp_path):
    out = touch(tmp_path / "out.json")
    inp1 = touch(tmp_path / "in1.json")
    inp2 = touch(tmp_path / "in2.json")
    set_mtime(out, -5)
    set_mtime(inp1, -10)
    set_mtime(inp2, 0)  # newer than output -> stale
    assert is_stale(outputs=[out], inputs=[inp1, inp2]) is True


def test_one_missing_output_among_several_is_stale(tmp_path):
    out1 = touch(tmp_path / "out1.json")
    out2 = tmp_path / "out2.json"  # never created
    assert is_stale(outputs=[out1, out2], inputs=[]) is True


def test_equal_mtime_is_not_stale(tmp_path):
    inp = touch(tmp_path / "in.json")
    out = touch(tmp_path / "out.json")
    t = time.time()
    os.utime(inp, (t, t))
    os.utime(out, (t, t))
    assert is_stale(outputs=[out], inputs=[inp]) is False
