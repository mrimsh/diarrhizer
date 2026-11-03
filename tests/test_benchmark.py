"""Validate baseline metrics independently of model inference."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("benchmark_asr", Path(__file__).parents[1] / "tools" / "benchmark_asr.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_word_substitution_deletion_and_insertion():
    assert benchmark.score("a b c", "a x c d")["wer"] == pytest.approx(2 / 3)
    assert benchmark.score("a b c", "a c")["wer"] == pytest.approx(1 / 3)
    assert benchmark.score("a", "x y z")["wer"] == 3
    assert benchmark.score("a b", "")["wer"] == 1


def test_cyrillic_case_punctuation_and_cer():
    assert benchmark.score("Привет, МИР!", "привет мир")["cer"] == 0
    assert benchmark.score("кот", "кит")["cer"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        benchmark.score("...", "text")


def test_missing_times_are_reported_without_inventing_timestamps():
    assert benchmark.missing_times([]) is None
    assert benchmark.missing_times([{"start": 0, "end": 1}, {"start": None, "end": 2}]) == .5
    assert benchmark.missing_times([{"start": float("nan"), "end": 2}]) == 1
