"""Tests for diarrhizer.diagnostics.models (HF Hub cache inspection).

All tests run against a mocked/faked HF Hub cache scan result - no real
model download, no network access, no dependency on symlink-based cache
directories actually existing on disk.
"""

from pathlib import Path

import pytest
from huggingface_hub import CacheNotFound
from huggingface_hub.utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo

from diarrhizer.diagnostics import models


def make_revision(commit_hash: str, size: int = 100, last_modified: float = 1_700_000_000.0) -> CachedRevisionInfo:
    return CachedRevisionInfo(
        commit_hash=commit_hash,
        snapshot_path=Path("fake-snapshot"),
        size_on_disk=size,
        files=frozenset(),
        refs=frozenset(),
        last_modified=last_modified,
    )


def make_repo(
    repo_id: str,
    size: int = 1000,
    last_accessed: float = 1_700_000_100.0,
    last_modified: float = 1_700_000_000.0,
    repo_type: str = "model",
    revisions=None,
) -> CachedRepoInfo:
    if revisions is None:
        revisions = frozenset({make_revision(f"{repo_id}-rev1", size=size)})
    return CachedRepoInfo(
        repo_id=repo_id,
        repo_type=repo_type,
        repo_path=Path("fake-repo-path"),
        size_on_disk=size,
        nb_files=1,
        revisions=frozenset(revisions),
        last_accessed=last_accessed,
        last_modified=last_modified,
    )


def make_cache_info(repos) -> HFCacheInfo:
    return HFCacheInfo(
        size_on_disk=sum(r.size_on_disk for r in repos),
        repos=frozenset(repos),
        warnings=[],
    )


# --- _resolve_cache_dir -------------------------------------------------


def test_resolve_cache_dir_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "C:/should-not-be-used")
    assert models._resolve_cache_dir("C:/explicit/path") == Path("C:/explicit/path")


def test_resolve_cache_dir_uses_hf_hub_cache_env(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", "C:/from/hub/cache")
    assert models._resolve_cache_dir() == Path("C:/from/hub/cache")


def test_resolve_cache_dir_uses_hf_home_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", "C:/from/hf/home")
    assert models._resolve_cache_dir() == Path("C:/from/hf/home") / "hub"


def test_resolve_cache_dir_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert models._resolve_cache_dir() == Path.home() / ".cache" / "huggingface" / "hub"


# --- list_cached_models / is_model_cached -------------------------------


def test_list_cached_models_returns_sorted_entries(monkeypatch):
    cache_info = make_cache_info(
        [
            make_repo("Systran/faster-whisper-large-v3", size=3_000_000_000),
            make_repo("jonatasgrosman/wav2vec2-large-xlsr-53-russian", size=1_200_000_000),
        ]
    )
    monkeypatch.setattr(models, "scan_cache_dir", lambda cache_dir: cache_info)

    result = models.list_cached_models(cache_dir="C:/fake/hub")

    assert [m.repo_id for m in result] == [
        "Systran/faster-whisper-large-v3",
        "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    ]
    assert result[0].size_on_disk == 3_000_000_000


def test_list_cached_models_excludes_non_model_repo_types(monkeypatch):
    cache_info = make_cache_info(
        [
            make_repo("org/some-model", repo_type="model"),
            make_repo("org/some-dataset", repo_type="dataset"),
        ]
    )
    monkeypatch.setattr(models, "scan_cache_dir", lambda cache_dir: cache_info)

    result = models.list_cached_models(cache_dir="C:/fake/hub")

    assert [m.repo_id for m in result] == ["org/some-model"]


def test_list_cached_models_missing_cache_dir_returns_empty(monkeypatch):
    def raise_not_found(cache_dir):
        raise CacheNotFound("no cache here", cache_dir=Path(cache_dir))

    monkeypatch.setattr(models, "scan_cache_dir", raise_not_found)

    assert models.list_cached_models(cache_dir="C:/does/not/exist") == []


def test_is_model_cached_true_for_present_repo(monkeypatch):
    cache_info = make_cache_info([make_repo("Systran/faster-whisper-large-v3")])
    monkeypatch.setattr(models, "scan_cache_dir", lambda cache_dir: cache_info)

    assert models.is_model_cached("Systran/faster-whisper-large-v3", cache_dir="C:/fake/hub") is True


def test_is_model_cached_false_for_absent_repo(monkeypatch):
    cache_info = make_cache_info([make_repo("Systran/faster-whisper-large-v3")])
    monkeypatch.setattr(models, "scan_cache_dir", lambda cache_dir: cache_info)

    assert models.is_model_cached("some/other-model", cache_dir="C:/fake/hub") is False


def test_is_model_cached_false_when_cache_missing(monkeypatch):
    def raise_not_found(cache_dir):
        raise CacheNotFound("no cache here", cache_dir=Path(cache_dir))

    monkeypatch.setattr(models, "scan_cache_dir", raise_not_found)

    assert models.is_model_cached("anything", cache_dir="C:/does/not/exist") is False


# --- clear_cache ----------------------------------------------------------
#
# HFCacheInfo is a frozen dataclass, so its delete_revisions can't be
# monkeypatched on a real instance. clear_cache() only reads cache_info.repos
# and calls cache_info.delete_revisions(*revisions), so a minimal stand-in
# object exercises the same contract without fighting the frozen dataclass.


class FakeDeleteStrategy:
    def __init__(self, revisions):
        self.revisions = revisions
        self.expected_freed_size = len(revisions) * 100
        self.executed = False

    def execute(self):
        self.executed = True


class FakeCacheInfo:
    def __init__(self, repos, delete_revisions):
        self.repos = repos
        self.delete_revisions = delete_revisions


def test_clear_cache_deletes_single_model(monkeypatch):
    repo_a = make_repo("org/model-a", revisions=frozenset({make_revision("rev-a")}))
    repo_b = make_repo("org/model-b", revisions=frozenset({make_revision("rev-b")}))

    captured = {}

    def fake_delete_revisions(*revisions):
        captured["revisions"] = revisions
        strategy = FakeDeleteStrategy(revisions)
        captured["strategy"] = strategy
        return strategy

    fake_cache_info = FakeCacheInfo(repos=[repo_a, repo_b], delete_revisions=fake_delete_revisions)
    monkeypatch.setattr(models, "scan_cache_dir", lambda cache_dir: fake_cache_info)

    freed = models.clear_cache(model_id="org/model-a", cache_dir="C:/fake/hub")

    assert captured["revisions"] == ("rev-a",)
    assert captured["strategy"].executed is True
    assert freed == 100


def test_clear_cache_deletes_everything_when_model_id_none(monkeypatch):
    repo_a = make_repo("org/model-a", revisions=frozenset({make_revision("rev-a")}))
    repo_b = make_repo("org/model-b", revisions=frozenset({make_revision("rev-b")}))

    captured = {}

    def fake_delete_revisions(*revisions):
        captured["revisions"] = set(revisions)
        return FakeDeleteStrategy(revisions)

    fake_cache_info = FakeCacheInfo(repos=[repo_a, repo_b], delete_revisions=fake_delete_revisions)
    monkeypatch.setattr(models, "scan_cache_dir", lambda cache_dir: fake_cache_info)

    freed = models.clear_cache(model_id=None, cache_dir="C:/fake/hub")

    assert captured["revisions"] == {"rev-a", "rev-b"}
    assert freed == 200


def test_clear_cache_no_matching_model_does_nothing(monkeypatch):
    repo_a = make_repo("org/model-a", revisions=frozenset({make_revision("rev-a")}))

    called = {"delete_revisions": False}

    def fake_delete_revisions(*revisions):
        called["delete_revisions"] = True
        return FakeDeleteStrategy(revisions)

    fake_cache_info = FakeCacheInfo(repos=[repo_a], delete_revisions=fake_delete_revisions)
    monkeypatch.setattr(models, "scan_cache_dir", lambda cache_dir: fake_cache_info)

    freed = models.clear_cache(model_id="org/does-not-exist", cache_dir="C:/fake/hub")

    assert freed == 0
    assert called["delete_revisions"] is False


def test_clear_cache_missing_cache_dir_returns_zero(monkeypatch):
    def raise_not_found(cache_dir):
        raise CacheNotFound("no cache here", cache_dir=Path(cache_dir))

    monkeypatch.setattr(models, "scan_cache_dir", raise_not_found)

    assert models.clear_cache(model_id=None, cache_dir="C:/does/not/exist") == 0


# --- warm_up_* device validation (no network, no whisperx import needed) --


def test_warm_up_asr_model_rejects_cuda_when_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        models.warm_up_asr_model("large-v3", device="cuda")


def test_warm_up_diarization_model_rejects_cuda_when_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        models.warm_up_diarization_model("fake-token", device="cuda")
