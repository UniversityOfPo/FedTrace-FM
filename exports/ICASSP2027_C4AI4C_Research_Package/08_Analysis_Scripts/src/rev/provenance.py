"""Provenance capture for reviewer-response runs.

Records what is needed to decide whether two runs are a matched pair and
whether a run is reproducible from committed code: git commit, working-tree
state and diff hash, which executed modules/configs were uncommitted,
interpreter and package versions, PyTorch/CUDA/cuDNN, GPU and driver,
determinism settings, configuration hashes, dataset-partition hash, model
initialization fingerprint, seed and RNG policy.

Privacy: only repository-relative paths are stored; no environment variables
except the single determinism setting ``CUBLAS_WORKSPACE_CONFIG``; no user,
host or home-directory names. :func:`assert_no_private_strings` enforces this
before anything is written.

Two cleanliness levels are reported, never conflated:
    working_tree_clean -- no tracked modification and no untracked file at all.
    execution_clean    -- no tracked modification, and no executed module or
                          used configuration file is untracked or modified.
"""

from __future__ import annotations

import getpass
import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from src.rev.hashing import sha256_bytes, sha256_file, sha256_json, state_content_sha256

PROVENANCE_SCHEMA = "fedtrace-rev-provenance-v1"

RNG_POLICY: Mapping[str, Any] = {
    "global_seeding": "src.utils.seed.set_seed(seed, deterministic=True) once, inside the NDSS main()",
    "client_selection": "random.Random(f'{seed}-{round_id}') in FederatedServer.select_clients",
    "dataloader_shuffle": "global torch RNG (DataLoader shuffle=True; worker base seeds from global RNG)",
    "ndss_attacks": "dedicated torch.Generator per (seed, round, client[, attack]) in src.attacks.traceability",
    "rev_attacks": "dedicated numpy Generator per (seed, client, attack) for poisoned-index selection; "
    "no random draw at sample time",
    "update_logger": "draws no random numbers",
}


class ProvenanceError(RuntimeError):
    pass


def _git(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, check=False)
    if result.returncode != 0:
        raise ProvenanceError(f"git {' '.join(args)} failed with exit code {result.returncode}")
    return result.stdout


def _parse_porcelain_z(raw: bytes) -> tuple[list[str], list[str]]:
    """Return (tracked_modified, untracked) relative paths from ``git status --porcelain=v1 -z``."""
    tracked, untracked = [], []
    entries = raw.decode("utf-8", errors="replace").split("\0")
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if not entry:
            continue
        code, path = entry[:2], entry[3:]
        if code == "??":
            untracked.append(path)
        else:
            tracked.append(path)
            if code[0] in "RC":  # rename/copy: the next field is the original path
                i += 1
    return sorted(tracked), sorted(untracked)


def git_state(repo_root: Path, used_files: Iterable[Path | str] = ()) -> dict[str, Any]:
    """Commit, cleanliness, diff hash and the uncommitted files actually used by this process."""
    repo_root = Path(repo_root).resolve()
    commit = _git(repo_root, "rev-parse", "HEAD").decode().strip()
    tracked, untracked = _parse_porcelain_z(
        _git(repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    )
    diff = _git(repo_root, "diff", "HEAD", "--binary")
    untracked_hashes = {p: sha256_file(repo_root / p) for p in untracked if (repo_root / p).is_file()}

    uncommitted = set(tracked) | set(untracked)
    executed = set(_loaded_repo_modules(repo_root))
    used = {_relative(repo_root, Path(p)) for p in used_files}
    executed_uncommitted = sorted(executed & uncommitted)
    used_uncommitted = sorted(used & uncommitted)
    return {
        "commit": commit,
        "working_tree_clean": not tracked and not untracked,
        "execution_clean": not tracked and not executed_uncommitted and not used_uncommitted,
        "tracked_modified": tracked,
        "untracked_count": len(untracked),
        "untracked_content_sha256": sha256_json(untracked_hashes),
        "diff_head_sha256": sha256_bytes(diff),
        "executed_uncommitted_modules": executed_uncommitted,
        "used_uncommitted_files": used_uncommitted,
    }


def _relative(repo_root: Path, path: Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve().relative_to(repo_root).as_posix()


def _loaded_repo_modules(repo_root: Path) -> list[str]:
    """Repository-relative source files of every module currently imported from the repository."""
    files = set()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        try:
            files.add(Path(file).resolve().relative_to(repo_root).as_posix())
        except ValueError:
            continue
    return sorted(f for f in files if not f.startswith(".venv/"))


def package_versions() -> dict[str, Any]:
    pins = sorted(
        {f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in importlib.metadata.distributions()}
    )
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "os": f"{platform.system()} {platform.release()}", "packages": pins, "packages_sha256": sha256_json(pins)}


def _nvidia_driver() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"driver_version": None, "reason": "nvidia-smi not found"}
    result = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=False, timeout=30)
    if result.returncode != 0:
        return {"driver_version": None, "reason": f"nvidia-smi exit code {result.returncode}"}
    return {"driver_version": result.stdout.strip().splitlines()[0] if result.stdout.strip() else None}


def torch_environment(query_gpu: bool = True) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    env: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available and query_gpu else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_seed_fixed": os.environ.get("PYTHONHASHSEED") is not None and sys.flags.hash_randomization == 0,
    }
    env.update(_nvidia_driver() if query_gpu else {"driver_version": None, "reason": "not queried"})
    return env


def config_fingerprint(repo_root: Path, files: Sequence[Path | str], parsed: Mapping[str, Any]) -> dict[str, Any]:
    """File hashes (by relative path) plus a canonical hash of the parsed configuration."""
    repo_root = Path(repo_root).resolve()
    return {
        "files": {_relative(repo_root, Path(f)): sha256_file(repo_root / _relative(repo_root, Path(f))) for f in files},
        "parsed_sha256": sha256_json(parsed),
    }


def partition_fingerprint(partitions: Mapping[str, Sequence[int]]) -> str:
    """Canonical hash of a client -> sample-index assignment (order within a client preserved)."""
    return sha256_json({str(cid): [int(i) for i in idx] for cid, idx in sorted(partitions.items())})


def model_init_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    return state_content_sha256(state)


def _private_strings(repo_root: Path | None) -> list[str]:
    candidates = [str(Path.home()), getpass.getuser(), socket.gethostname()]
    if repo_root is not None:
        resolved = Path(repo_root).resolve()
        candidates += [str(resolved), resolved.as_posix()]
    return [c for c in candidates if c and len(c) >= 3]


def _walk_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            yield str(key)
            yield from _walk_strings(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk_strings(item)


def assert_no_private_strings(obj: Any, repo_root: Path | None = None) -> None:
    """Raise if any string in ``obj`` contains an absolute path, home, user or host name."""
    needles = [s.lower() for s in _private_strings(repo_root)]
    for text in _walk_strings(obj):
        lowered = text.lower()
        if any(n in lowered for n in needles):
            raise ProvenanceError("provenance contains a private string (home/user/host/absolute path)")
        if len(text) > 2 and text[1] == ":" and text[2] in "\\/":
            raise ProvenanceError("provenance contains an absolute Windows path")
        if text.startswith(("/home/", "/Users/", "\\\\")):
            raise ProvenanceError("provenance contains an absolute path")


def capture_provenance(
    repo_root: Path,
    *,
    config_files: Sequence[Path | str],
    parsed_config: Mapping[str, Any],
    seed: int,
    query_gpu: bool = True,
) -> dict[str, Any]:
    """Pre-run provenance. Partition and initialization fingerprints are added after the run."""
    repo_root = Path(repo_root).resolve()
    payload = {
        "schema": PROVENANCE_SCHEMA,
        "git": git_state(repo_root, used_files=config_files),
        "environment": package_versions(),
        "torch": torch_environment(query_gpu=query_gpu),
        "config": config_fingerprint(repo_root, config_files, parsed_config),
        "seed": int(seed),
        "rng_policy": dict(RNG_POLICY),
    }
    assert_no_private_strings(payload, repo_root)
    return payload
