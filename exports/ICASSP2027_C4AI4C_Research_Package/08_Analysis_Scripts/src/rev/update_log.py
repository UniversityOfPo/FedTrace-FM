"""Optional, read-only fp32 logger of submitted client updates and global states.

What is logged (each tensor in its original dtype -- fp32 parameters stay
fp32, integer buffers stay integer; nothing is down-cast or compressed):

    submitted  -- the exact model state a client submitted to the server in a
                  round (after any update transform), i.e. exactly the tensor
                  dict wrapped into that round's ``ClientUpdate``.
    pre_transform -- optionally, for configured clients only, the state before
                  the update transform (the state produced by local training).
    global     -- the global model state before round 0 (``phase="initial"``)
                  and after every round (``phase="after_round"``).

The update delta used by the NDSS trace is ``submitted[k] - pre_round_global[k]``
over the trainable keys; both operands are logged bit-exactly, so the delta is
recomputable exactly (see :mod:`src.rev.replay`).

Guarantees (tested in ``tests/test_rev_update_log.py``):
    * tensors are only read: never modified in place, never moved to another device;
    * no random number is drawn (Python, NumPy, torch CPU/CUDA RNG states unchanged);
    * every file is written atomically (``.partial`` + fsync + rename) and recorded
      in an append-only JSONL manifest only after it is complete, with file and
      content SHA-256;
    * an existing, unsealed log is an interrupted run: it is never resumed or
      overwritten (:class:`LogInterruptedError`); the run must be rerun under a
      new run id;
    * a configurable byte budget and free-disk reserve are enforced before every
      write and, after the first state is seen, for the whole planned run.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.rev.hashing import (
    atomic_torch_save,
    atomic_write_bytes,
    canonical_json,
    sha256_file,
    state_content_sha256,
    state_nbytes,
)

LOG_SCHEMA = "fedtrace-rev-update-log-v1"
MANIFEST_NAME = "manifest.jsonl"
SEAL_NAME = "SEAL.json"
_GIB = 1024**3


class UpdateLogError(RuntimeError):
    pass


class LogInterruptedError(UpdateLogError):
    """An update log exists but was never sealed: the run that produced it was interrupted."""


class StorageLimitError(UpdateLogError):
    pass


@dataclass(frozen=True)
class UpdateLogConfig:
    """Logger settings.

    Attributes:
        root: Directory of this run's log (must not exist yet).
        run_id: Identifier recorded in the manifest header.
        pre_transform_clients: Clients whose pre-transform state is also logged.
        max_bytes: Byte budget for all tensor files of this log (None = no budget).
        min_free_bytes: Free-disk reserve that must remain after every write.
        expected_client_states: Planned number of submitted states for the run
            (e.g. clients x rounds); with ``expected_global_states`` it is checked
            against the budget and free disk as soon as the first state's size is known.
        expected_global_states: Planned number of global states (rounds + 1).
    """

    root: Path
    run_id: str
    pre_transform_clients: tuple[str, ...] = ()
    max_bytes: int | None = None
    min_free_bytes: int = 100 * _GIB
    expected_client_states: int | None = None
    expected_global_states: int | None = None


@dataclass
class _RoundState:
    round_id: int
    pre_round_global_sha256: str | None = None
    next_submission_index: int = 0


@dataclass
class UpdateLogger:
    config: UpdateLogConfig
    _bytes_written: int = 0
    _records: int = 0
    _round: _RoundState | None = None
    _sealed: bool = False
    _initial_logged: bool = False
    _plan_checked: bool = False
    _last_global_sha256: str | None = None
    failure: BaseException | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def open(cls, config: UpdateLogConfig) -> UpdateLogger:
        """Create a new log directory. Refuses an existing directory (interrupted or completed)."""
        root = Path(config.root)
        if root.exists():
            if (root / SEAL_NAME).exists():
                raise UpdateLogError(f"update log already completed and sealed: {root.name}; use a new run id")
            if any(root.iterdir()):
                raise LogInterruptedError(
                    f"update log {root.name} exists without a seal: the run was interrupted. "
                    "Mark it FAILED and rerun from scratch under a new run id (resume is not supported)."
                )
        root.mkdir(parents=True, exist_ok=True)
        (root / "states").mkdir(exist_ok=True)
        logger = cls(config=config)
        logger._append({"kind": "header", "schema": LOG_SCHEMA, "run_id": config.run_id,
                        "created_unix": int(time.time()),
                        "pre_transform_clients": list(config.pre_transform_clients)})
        return logger

    def seal(self, status: str = "COMPLETED", extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Write the seal (manifest hash and counts). Only a sealed log is verifiable."""
        if self._sealed:
            raise UpdateLogError("log already sealed")
        if self.failure is not None and status == "COMPLETED":
            raise UpdateLogError("cannot seal as COMPLETED: a logging failure occurred")
        root = Path(self.config.root)
        seal = {
            "schema": LOG_SCHEMA,
            "run_id": self.config.run_id,
            "status": status,
            "manifest_sha256": sha256_file(root / MANIFEST_NAME),
            "records": self._records,
            "tensor_bytes": self._bytes_written,
            "extra": dict(extra or {}),
        }
        atomic_write_bytes(root / SEAL_NAME, (json.dumps(seal, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        self._sealed = True
        return seal

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def begin_round(self, round_id: int) -> None:
        self._round = _RoundState(round_id=round_id)

    def log_client(
        self,
        client_id: str,
        round_id: int,
        submitted_state: Mapping[str, torch.Tensor],
        pre_round_global_state: Mapping[str, torch.Tensor],
        num_samples: int,
        pre_transform_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Log one client's submitted state (and pre-transform state if configured)."""
        if self._round is None or self._round.round_id != round_id:
            self.begin_round(round_id)
        state = self._round
        if state.pre_round_global_sha256 is None:
            state.pre_round_global_sha256 = state_content_sha256(pre_round_global_state)
            if not self._initial_logged:
                if round_id != 0:
                    raise UpdateLogError("logging must start at round 0 (resume is not supported)")
                self._log_global(-1, pre_round_global_state, phase="initial")
                self._initial_logged = True
            elif state.pre_round_global_sha256 != self._last_global_sha256:
                raise UpdateLogError(
                    f"round {round_id}: pre-round global state differs from the last logged global state"
                )
        index = state.next_submission_index
        state.next_submission_index += 1
        self._write_state(
            "submitted", submitted_state, round_id=round_id, client_id=client_id,
            extra={"submission_index": index, "num_samples": int(num_samples),
                   "pre_round_global_sha256": state.pre_round_global_sha256},
        )
        if pre_transform_state is not None and client_id in self.config.pre_transform_clients:
            self._write_state(
                "pre_transform", pre_transform_state, round_id=round_id, client_id=client_id,
                extra={"submission_index": index, "pre_round_global_sha256": state.pre_round_global_sha256,
                       "identical_to_submitted": pre_transform_state is submitted_state},
            )

    def log_round_end(
        self, round_id: int, global_state: Mapping[str, torch.Tensor], round_status: str,
        successful_client_ids: tuple[str, ...] = (),
    ) -> None:
        """Log the global state after a round (also for a failed round, where it is unchanged)."""
        if not self._initial_logged:
            # Only possible when no client submitted in round 0, i.e. the global state is unchanged.
            if round_id != 0 or round_status == "completed":
                raise UpdateLogError("no initial global state was logged before a completed round")
            self._log_global(-1, global_state, phase="initial")
            self._initial_logged = True
        self._log_global(round_id, global_state, phase="after_round", round_status=round_status,
                         successful_client_ids=list(successful_client_ids))
        self._round = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _log_global(self, round_id: int, state: Mapping[str, torch.Tensor], phase: str, **extra: Any) -> None:
        sha = self._write_state("global", state, round_id=round_id, client_id=None, extra={"phase": phase, **extra})
        self._last_global_sha256 = sha

    def _check_storage(self, nbytes: int) -> None:
        cfg = self.config
        if not self._plan_checked and (cfg.expected_client_states or cfg.expected_global_states):
            planned = nbytes * ((cfg.expected_client_states or 0) + (cfg.expected_global_states or 0))
            if cfg.max_bytes is not None and planned > cfg.max_bytes:
                raise StorageLimitError(
                    f"planned log size {planned / _GIB:.2f} GiB exceeds the budget {cfg.max_bytes / _GIB:.2f} GiB"
                )
            free = shutil.disk_usage(cfg.root).free
            if free - planned < cfg.min_free_bytes:
                raise StorageLimitError(
                    f"planned log size {planned / _GIB:.2f} GiB would leave less than the "
                    f"{cfg.min_free_bytes / _GIB:.0f} GiB free-disk reserve"
                )
            self._plan_checked = True
        if cfg.max_bytes is not None and self._bytes_written + nbytes > cfg.max_bytes:
            raise StorageLimitError(f"log byte budget exhausted ({cfg.max_bytes / _GIB:.2f} GiB)")
        if shutil.disk_usage(cfg.root).free - nbytes < cfg.min_free_bytes:
            raise StorageLimitError(f"write would leave less than the {cfg.min_free_bytes / _GIB:.0f} GiB reserve")

    def _write_state(
        self, kind: str, state: Mapping[str, torch.Tensor], *, round_id: int, client_id: str | None,
        extra: Mapping[str, Any],
    ) -> str:
        if self._sealed:
            raise UpdateLogError("log is sealed")
        try:
            nbytes = state_nbytes(state)
            self._check_storage(nbytes)
            name = f"r{round_id:04d}_{kind}" + (f"_{client_id}" if client_id is not None else "") + ".pt"
            if round_id < 0:
                name = f"init_{kind}.pt"
            relative = f"states/{name}"
            path = Path(self.config.root) / relative
            if path.exists():
                raise UpdateLogError(f"refusing to overwrite logged state {relative}")
            content_sha = state_content_sha256(state)
            atomic_torch_save(state, path)
            record = {
                "kind": kind, "round_id": round_id, "client_id": client_id, "file": relative,
                "file_sha256": sha256_file(path), "content_sha256": content_sha, "tensor_bytes": nbytes,
                "dtypes": sorted({str(t.dtype) for t in state.values()}), **extra,
            }
            self._append(record)
            self._bytes_written += nbytes
            return content_sha
        except BaseException as exc:
            if self.failure is None:
                self.failure = exc
            raise

    def _append(self, record: Mapping[str, Any]) -> None:
        line = canonical_json(dict(record)) + "\n"
        with (Path(self.config.root) / MANIFEST_NAME).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        if record.get("kind") != "header":
            self._records += 1


def read_manifest(root: Path) -> list[dict[str, Any]]:
    with (Path(root) / MANIFEST_NAME).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
