"""Offline verification and replay of an update log (:mod:`src.rev.update_log`).

* :func:`verify_update_log` -- seal, manifest hash, per-file and per-content
  SHA-256, no partial or unlisted files, contiguous rounds and submission
  indices, and the global-state chain (every submission's pre-round global
  state is the previous round's logged global state).
* :func:`replay_aggregation` -- re-aggregates each round's logged submitted
  states with the NDSS aggregation strategy, in submission order, and checks
  that the result is bit-identical to the logged post-round global state.
* :func:`recompute_trace_rows` -- recomputes the NDSS ``client_round_trace``
  rows from logged tensors with the unmodified ``build_trace_row`` /
  ``apply_naive_detector``.

Known limitation (documented, tested): the NDSS script forms each client's
peer list by iterating a Python ``set`` of client-id strings, whose order
depends on per-process string-hash randomisation (``PYTHONHASHSEED``). The
peer mean is an fp64 sum over that order, so ``peer_mean_cosine_similarity``
can differ in the last bits between processes. Replay uses sorted order.
FedTrace v3 does not use this column; ``update_l2_norm`` and
``self_prev_round_cosine`` do not depend on the order.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.attacks.traceability import AttackConfig, apply_naive_detector, build_trace_row
from src.federated.aggregation.base import AggregationStrategy
from src.federated.aggregation.fedavg import FedAvg
from src.federated.contracts import ClientUpdate
from src.rev.hashing import load_tensor_state, sha256_file, state_content_sha256
from src.rev.update_log import LOG_SCHEMA, MANIFEST_NAME, SEAL_NAME, read_manifest

ORDER_DEPENDENT_TRACE_FIELDS: tuple[str, ...] = ("peer_mean_cosine_similarity", "naive_threshold", "naive_flag",
                                                 "first_flagged_round")


@dataclass
class VerificationReport:
    ok: bool
    status: str | None
    records: int
    problems: list[str] = field(default_factory=list)


@dataclass
class RoundRecords:
    round_id: int
    submitted: list[dict[str, Any]] = field(default_factory=list)
    pre_transform: dict[str, dict[str, Any]] = field(default_factory=dict)
    global_before: dict[str, Any] | None = None
    global_after: dict[str, Any] | None = None


def verify_update_log(root: Path, check_content: bool = True) -> VerificationReport:
    root = Path(root)
    problems: list[str] = []
    seal_path = root / SEAL_NAME
    if not seal_path.exists():
        return VerificationReport(False, None, 0, ["not sealed: the run was interrupted or is still running"])
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("manifest_sha256") != sha256_file(root / MANIFEST_NAME):
        problems.append("manifest hash does not match the seal")
    records = read_manifest(root)
    if not records or records[0].get("kind") != "header" or records[0].get("schema") != LOG_SCHEMA:
        problems.append("missing or unknown manifest header")
    body = [r for r in records if r.get("kind") != "header"]
    if seal.get("records") != len(body):
        problems.append(f"seal counts {seal.get('records')} records, manifest has {len(body)}")

    partial = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.partial"))
    if partial:
        problems.append(f"interrupted writes present: {partial}")
    malformed = [r for r in body if not {"kind", "round_id", "file", "file_sha256", "content_sha256"} <= set(r)]
    if malformed:
        problems.append(f"{len(malformed)} malformed manifest record(s)")
        body = [r for r in body if r not in malformed]
    listed = {r["file"] for r in body}
    on_disk = {p.relative_to(root).as_posix() for p in (root / "states").glob("*") if p.is_file()}
    if on_disk - listed:
        problems.append(f"files not in the manifest: {sorted(on_disk - listed)}")

    for record in body:
        path = root / record["file"]
        if not path.exists():
            problems.append(f"missing file {record['file']}")
            continue
        if sha256_file(path) != record["file_sha256"]:
            problems.append(f"file hash mismatch {record['file']}")
            continue
        if check_content and state_content_sha256(load_tensor_state(path)) != record["content_sha256"]:
            problems.append(f"content hash mismatch {record['file']}")

    try:
        rounds = group_by_round(body)
    except (ValueError, KeyError) as exc:
        problems.append(f"manifest structure: {exc!r}")
        rounds = {}
    for round_id, rr in sorted(rounds.items()):
        if rr.global_after is None:
            problems.append(f"round {round_id}: no post-round global state")
        if rr.global_before is None:
            problems.append(f"round {round_id}: no pre-round global state in the chain")
        indices = [r["submission_index"] for r in rr.submitted]
        if indices != list(range(len(indices))):
            problems.append(f"round {round_id}: submission indices {indices} are not contiguous")
        for r in rr.submitted:
            if rr.global_before is not None and r["pre_round_global_sha256"] != rr.global_before["content_sha256"]:
                problems.append(f"round {round_id}: {r['client_id']} pre-round global is not the logged chain state")
        if rr.global_after is not None and rr.global_after.get("round_status") == "completed":
            accepted = set(rr.global_after.get("successful_client_ids", []))
            if accepted != {r["client_id"] for r in rr.submitted}:
                problems.append(f"round {round_id}: accepted clients differ from logged submissions")
    if rounds and sorted(rounds) != list(range(len(rounds))):
        problems.append(f"rounds are not contiguous from 0: {sorted(rounds)}")
    return VerificationReport(not problems and seal.get("status") == "COMPLETED", seal.get("status"), len(body),
                              problems)


def group_by_round(records: Iterable[Mapping[str, Any]]) -> dict[int, RoundRecords]:
    """Organise manifest records per round; ``global_before`` of round r is the global after r-1."""
    records = [dict(r) for r in records if r.get("kind") != "header"]
    rounds: dict[int, RoundRecords] = {}
    globals_by_round: dict[int, dict[str, Any]] = {}
    for r in records:
        if r["kind"] == "global":
            if r["round_id"] in globals_by_round:
                raise ValueError(f"duplicate global state for round {r['round_id']}")
            globals_by_round[r["round_id"]] = r
        elif r["kind"] in ("submitted", "pre_transform"):
            rr = rounds.setdefault(r["round_id"], RoundRecords(r["round_id"]))
            if r["kind"] == "submitted":
                rr.submitted.append(r)
            else:
                rr.pre_transform[r["client_id"]] = r
        else:
            raise ValueError(f"unknown record kind {r['kind']!r}")
    for round_id in [k for k in globals_by_round if k >= 0]:
        rounds.setdefault(round_id, RoundRecords(round_id))
    for round_id, rr in rounds.items():
        rr.submitted.sort(key=lambda r: r["submission_index"])
        rr.global_before = globals_by_round.get(round_id - 1)
        rr.global_after = globals_by_round.get(round_id)
    return rounds


def _load(root: Path, record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return load_tensor_state(Path(root) / record["file"])


def replay_aggregation(root: Path, strategy: AggregationStrategy | None = None) -> dict[int, bool]:
    """Round id -> whether re-aggregating the logged submissions reproduces the logged global bit-exactly."""
    strategy = strategy or FedAvg()
    outcome: dict[int, bool] = {}
    for round_id, rr in sorted(group_by_round(read_manifest(root)).items()):
        if rr.global_after is None or rr.global_before is None or rr.global_after.get("round_status") != "completed":
            continue
        updates = [
            ClientUpdate.create(client_id=r["client_id"], round_id=round_id, model_state=_load(root, r),
                                num_samples=r["num_samples"])
            for r in rr.submitted
        ]
        aggregated = strategy.aggregate(_load(root, rr.global_before), updates).aggregated_state
        outcome[round_id] = state_content_sha256(aggregated) == rr.global_after["content_sha256"]
    return outcome


def recompute_trace_rows(
    root: Path,
    client_ids: Sequence[str],
    trainable_keys: Iterable[str],
    attack_config: AttackConfig,
) -> list[dict[str, Any]]:
    """Recompute NDSS trace rows from logged tensors (peer order: sorted client ids)."""
    trainable_keys = frozenset(trainable_keys)
    previous: dict[str, dict[str, torch.Tensor]] = {}
    rows: list[dict[str, Any]] = []
    for round_id, rr in sorted(group_by_round(read_manifest(root)).items()):
        if rr.global_after is None:
            continue
        pre_global = _load(root, rr.global_before) if rr.global_before is not None else None
        accepted = set(rr.global_after.get("successful_client_ids", []))
        deltas: dict[str, dict[str, torch.Tensor]] = {}
        honest: dict[str, dict[str, torch.Tensor]] = {}
        for record in rr.submitted:
            cid = record["client_id"]
            submitted = _load(root, record)
            deltas[cid] = {k: submitted[k] - pre_global[k] for k in trainable_keys}
            pre = rr.pre_transform.get(cid)
            if pre is not None:
                pre_state = _load(root, pre)
                honest[cid] = {k: pre_state[k] - pre_global[k] for k in trainable_keys}
            else:
                honest[cid] = deltas[cid]
        selected = {r["client_id"] for r in rr.submitted}
        for cid in sorted(client_ids):
            successful = cid in accepted and cid in deltas
            rows.append(build_trace_row(
                round_id=round_id, client_id=cid, selected=cid in selected, successful=successful,
                attack_config=attack_config, client_delta=deltas.get(cid),
                peer_deltas=[deltas[p] for p in sorted(accepted) if p != cid and p in deltas],
                trainable_keys=trainable_keys, honest_delta=honest.get(cid), self_prev_delta=previous.get(cid),
            ))
            if cid in deltas:
                previous[cid] = deltas[cid]
    return apply_naive_detector(rows, attack_config)


def compare_trace_rows(
    expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]],
    ignore_fields: Iterable[str] = (),
) -> list[str]:
    """Exact (==) comparison of two trace-row lists; returns human-readable mismatches."""
    ignore = set(ignore_fields)
    if len(expected) != len(actual):
        return [f"row count {len(expected)} != {len(actual)}"]
    mismatches = []
    for i, (a, b) in enumerate(zip(expected, actual)):
        for key in sorted((set(a) | set(b)) - ignore):
            if a.get(key) != b.get(key):
                mismatches.append(f"row {i} ({a.get('round_id')}, {a.get('client_id')}) {key}: {a.get(key)!r} != {b.get(key)!r}")
    return mismatches
