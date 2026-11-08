"""Canonical hashing of tensor states and JSON payloads, plus atomic file writes.

Tensor hashes are computed over each tensor's raw bytes (plus key, dtype and
shape), so they identify the exact numerical content independently of how a
file was serialized. No function here draws random numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_CHUNK = 1 << 22


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace variation, no NaN."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_json(payload: Any) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def state_content_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash of a state dict's exact content: sorted keys, dtype, shape and raw bytes.

    Two states hash equal iff they have the same keys and every tensor is
    bit-identical with the same dtype and shape. Reads tensors only.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        header = f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def state_nbytes(state: Mapping[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in state.values())


def states_bit_identical(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    if set(a) != set(b):
        return False
    return all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]) for k in a)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a sibling ``.partial`` file, fsync, then atomic rename."""
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def atomic_torch_save(state: Mapping[str, torch.Tensor], path: Path) -> None:
    """``torch.save`` a plain dict of tensors atomically (``.partial`` + fsync + rename)."""
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        torch.save({key: state[key] for key in sorted(state)}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def load_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    """Load a state saved by :func:`atomic_torch_save` (tensors only, no pickled code)."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)
