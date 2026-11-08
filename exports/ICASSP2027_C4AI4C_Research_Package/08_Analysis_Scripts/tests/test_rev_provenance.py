"""Provenance: git cleanliness levels, fingerprints, privacy of the recorded payload."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from src.rev.provenance import (
    ProvenanceError,
    _parse_porcelain_z,
    assert_no_private_strings,
    capture_provenance,
    git_state,
    model_init_fingerprint,
    partition_fingerprint,
    torch_environment,
)
from tests.rev_toy_federation import TinyNet

ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
                   check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "cfg.yaml").write_text("a: 1\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_porcelain_parser_handles_renames_and_untracked() -> None:
    raw = b" M a.py\0R  new.py\0old.py\0?? b dir/c.py\0"
    assert _parse_porcelain_z(raw) == (["a.py", "new.py"], ["b dir/c.py"])


def test_clean_committed_tree(repo: Path) -> None:
    state = git_state(repo, used_files=["cfg.yaml"])
    assert len(state["commit"]) == 40 and state["working_tree_clean"] and state["execution_clean"]


def test_untracked_unused_file_is_dirty_tree_but_clean_execution(repo: Path) -> None:
    (repo / "notes.txt").write_text("x", encoding="utf-8")
    state = git_state(repo, used_files=["cfg.yaml"])
    assert not state["working_tree_clean"] and state["execution_clean"] and state["untracked_count"] == 1


def test_modified_tracked_file_or_used_untracked_config_is_not_execution_clean(repo: Path) -> None:
    (repo / "extra.yaml").write_text("b: 2\n", encoding="utf-8")
    state = git_state(repo, used_files=["extra.yaml"])
    assert not state["execution_clean"] and state["used_uncommitted_files"] == ["extra.yaml"]
    diff_before = git_state(repo)["diff_head_sha256"]
    (repo / "cfg.yaml").write_text("a: 2\n", encoding="utf-8")
    state = git_state(repo)
    assert state["tracked_modified"] == ["cfg.yaml"] and not state["execution_clean"]
    assert state["diff_head_sha256"] != diff_before


def test_executed_untracked_module_is_detected(repo: Path, monkeypatch) -> None:
    (repo / "pkg" / "new_mod.py").write_text("Y = 2\n", encoding="utf-8")
    fake = types.ModuleType("fake_new_mod")
    fake.__file__ = str(repo / "pkg" / "new_mod.py")
    monkeypatch.setitem(sys.modules, "fake_new_mod", fake)
    state = git_state(repo)
    assert state["executed_uncommitted_modules"] == ["pkg/new_mod.py"] and not state["execution_clean"]


def test_fingerprints() -> None:
    parts = {"client_001": [5, 3, 9], "client_000": [1, 2]}
    assert partition_fingerprint(parts) == partition_fingerprint(dict(reversed(list(parts.items()))))
    assert partition_fingerprint(parts) != partition_fingerprint({"client_001": [3, 5, 9], "client_000": [1, 2]})
    torch.manual_seed(1)
    a = model_init_fingerprint(TinyNet().state_dict())
    torch.manual_seed(1)
    b = model_init_fingerprint(TinyNet().state_dict())
    torch.manual_seed(2)
    c = model_init_fingerprint(TinyNet().state_dict())
    assert a == b != c


def test_private_strings_are_refused() -> None:
    assert_no_private_strings({"path": "results_rev/x", "list": ["configs/rev/a.yaml"]}, ROOT)
    for bad in ({"p": str(Path.home() / "x")}, {"p": "D:\\data\\x"}, {"p": "/home/someone/x"},
                {"p": f"{ROOT.as_posix()}/src"}, {str(ROOT): 1}):
        with pytest.raises(ProvenanceError):
            assert_no_private_strings(bad, ROOT)


def test_capture_provenance_on_this_repository_is_complete_and_private() -> None:
    payload = capture_provenance(ROOT, config_files=["configs/rev/R1_exp016_logged.rev.yaml"],
                                 parsed_config={"a": 1}, seed=123, query_gpu=False)
    assert payload["git"]["commit"] and isinstance(payload["git"]["execution_clean"], bool)
    assert payload["environment"]["python"] and payload["environment"]["packages_sha256"]
    assert any(p.startswith("torch==") for p in payload["environment"]["packages"])
    assert set(payload["config"]["files"]) == {"configs/rev/R1_exp016_logged.rev.yaml"}
    assert payload["seed"] == 123 and "dataloader_shuffle" in payload["rng_policy"]
    env = torch_environment(query_gpu=False)
    assert {"torch", "cuda_build", "cudnn", "deterministic_algorithms", "driver_version"} <= set(env)
