"""Malicious-client attack simulation and traceability pilot infrastructure (Phase 5).

Implements only the minimum pieces pre-registered in
``research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`` for the
``EXP_014`` pilot: the ``gaussian_update_noise`` attack transform (Section
5), the cheap diagnostic update features (Section 10), ground-truth
client-round bookkeeping (Section 9), and the naive per-client
peer-agreement baseline detector (Section 11).

This module deliberately does **not** implement the FedTrace attribution
algorithm (the plan's Section 21 explicit boundary) — ``src/attribution/``
remains an empty placeholder. Everything here operates purely on
already-computed, in-memory model-update tensors (client-reported state
dicts and the pre-round global state dict); no dataset, dataloader, or
test-set object is read, imported, or referenced anywhere in this module.
"""

from __future__ import annotations

import csv
import hashlib
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_VALID_ATTACK_TYPES: tuple[str, ...] = ("none", "gaussian_update_noise")

TRACE_CSV_FIELDNAMES: tuple[str, ...] = (
    "round_id",
    "client_id",
    "selected",
    "attack_configured",
    "attack_active",
    "attack_type",
    "attack_strength_rho",
    "malicious_client_id",
    "attack_start_round",
    "update_l2_norm",
    "peer_mean_cosine_similarity",
    "naive_threshold",
    "naive_flag",
    "first_flagged_round",
)


@dataclass(frozen=True)
class AttackConfig:
    """Fully describes (or disables) the single first-order attack scenario for one run.

    Attributes:
        attack_type: ``"none"`` (the default; a strict no-op) or
            ``"gaussian_update_noise"``.
        malicious_client_id: The one client id perturbed when the attack is
            active. Required (non-empty) when ``attack_type != "none"``.
        attack_start_round: The first round (0-indexed, inclusive) at which
            the attack becomes active for ``malicious_client_id``. Required
            when ``attack_type != "none"``.
        attack_strength_rho: The noise-to-update L2-norm ratio (Section 5
            of the plan). Must be ``>= 0``; must be ``> 0`` when
            ``attack_type == "gaussian_update_noise"`` (a zero-strength
            attack is indistinguishable from no attack and is rejected as
            a likely configuration mistake, mirroring
            ``FederatedClientConfig.fedprox_mu``'s equivalent check).
    """

    attack_type: str = "none"
    malicious_client_id: str | None = None
    attack_start_round: int | None = None
    attack_strength_rho: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration values, raising ``ValueError`` with a clear message."""
        if self.attack_type not in _VALID_ATTACK_TYPES:
            raise ValueError(
                f"attack_type must be one of {_VALID_ATTACK_TYPES}, got '{self.attack_type}'."
            )
        if self.attack_strength_rho < 0:
            raise ValueError(
                f"attack_strength_rho must be zero or positive, got {self.attack_strength_rho}."
            )
        if self.attack_type != "none":
            if not self.malicious_client_id:
                raise ValueError(
                    "malicious_client_id must be a non-empty string when attack_type != 'none'."
                )
            if self.attack_start_round is None or self.attack_start_round < 0:
                raise ValueError(
                    "attack_start_round must be a non-negative integer when attack_type != "
                    f"'none', got {self.attack_start_round}."
                )
        if self.attack_type == "gaussian_update_noise" and self.attack_strength_rho <= 0:
            raise ValueError(
                "attack_strength_rho must be > 0 when attack_type='gaussian_update_noise', "
                f"got {self.attack_strength_rho}."
            )

    def is_active(self, round_id: int, client_id: str) -> bool:
        """Return whether the attack is active for ``client_id`` at ``round_id``.

        ``False`` whenever ``attack_type == "none"``, for every client
        other than ``malicious_client_id``, and for every round before
        ``attack_start_round`` -- the delayed-onset, single-client
        semantics required by the plan's Section 3/7.
        """
        return (
            self.attack_type != "none"
            and client_id == self.malicious_client_id
            and self.attack_start_round is not None
            and round_id >= self.attack_start_round
        )


def _derive_attack_seed(seed: int, round_id: int, client_id: str) -> int:
    """Deterministically derive a per-(seed, round, client) integer RNG seed.

    Mirrors the style already used by ``FederatedServer.select_clients``
    (``random.Random(f"{seed}-{round_id}")``): a plain string key hashed
    into an integer, so the same ``(seed, round_id, client_id)`` triple
    always derives the same value, and different triples derive
    (with overwhelming probability) different values. Masked to 63 bits so
    the result is always a valid ``torch.Generator.manual_seed`` argument.
    """
    digest = hashlib.sha256(f"{seed}-{round_id}-{client_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def _flatten_sorted(tensors: Mapping[str, torch.Tensor], keys: Iterable[str]) -> torch.Tensor:
    """Flatten and concatenate ``tensors[key]`` for every ``key`` in ``keys``, sorted.

    Sorting keys before concatenation guarantees the same flattened vector
    regardless of the input mapping's iteration order.
    """
    sorted_keys = sorted(keys)
    return torch.cat([tensors[key].detach().reshape(-1).double() for key in sorted_keys])


def flattened_l2_norm(delta: Mapping[str, torch.Tensor]) -> float:
    """Return the L2 norm of ``delta``, flattened and concatenated across all its keys."""
    if not delta:
        return 0.0
    return float(_flatten_sorted(delta, delta.keys()).norm(p=2))


def compute_update_l2_norm(
    model_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    trainable_keys: Iterable[str],
) -> float:
    """Return ``||model_state - global_state||_2`` over ``trainable_keys`` only."""
    keys = list(trainable_keys)
    delta = {key: model_state[key] - global_state[key] for key in keys}
    return flattened_l2_norm(delta)


def compute_peer_mean_cosine_similarity(
    client_delta: Mapping[str, torch.Tensor],
    peer_deltas: Sequence[Mapping[str, torch.Tensor]],
    trainable_keys: Iterable[str],
) -> float | None:
    """Cosine similarity between ``client_delta`` and the mean of ``peer_deltas``.

    Both ``client_delta`` and every element of ``peer_deltas`` are
    flattened (over ``trainable_keys``, sorted) into single vectors before
    comparison. Returns ``None`` (undefined, never fabricated as 0.0) if
    there are no peers, or if either the client's or the peer-mean vector
    has zero L2 norm (cosine similarity is undefined against a zero
    vector).
    """
    keys = list(trainable_keys)
    if not peer_deltas:
        return None
    client_vec = _flatten_sorted(client_delta, keys)
    peer_vecs = torch.stack([_flatten_sorted(peer_delta, keys) for peer_delta in peer_deltas])
    peer_mean_vec = peer_vecs.mean(dim=0)

    client_norm = client_vec.norm(p=2)
    peer_norm = peer_mean_vec.norm(p=2)
    if client_norm == 0 or peer_norm == 0:
        return None
    return float(torch.dot(client_vec, peer_mean_vec) / (client_norm * peer_norm))


def generate_gaussian_attack_noise(
    delta: Mapping[str, torch.Tensor],
    rho: float,
    seed: int,
    round_id: int,
    client_id: str,
) -> dict[str, torch.Tensor]:
    """Generate deterministic, norm-scaled Gaussian noise for the ``gaussian_update_noise`` attack.

    Per ``research_plans/PHASE5_Malicious_Client_Traceability_Plan.md``
    Section 5: draws raw i.i.d. standard-normal noise (same shapes as
    ``delta``) from a dedicated ``torch.Generator`` seeded deterministically
    from ``(seed, round_id, client_id)`` (see ``_derive_attack_seed``) --
    never the global RNG state -- then rescales the *entire* noise vector
    (flattened and concatenated across every tensor in ``delta``) so its L2
    norm equals exactly ``rho * ||delta||_2`` (up to floating-point
    tolerance from the rescaling division/multiplication).

    ``rho == 0`` returns exact zero tensors for every key (no RNG draw
    needed), guaranteeing the required no-op property: adding this result
    to an honest update leaves it byte-unchanged.

    Args:
        delta: The honest client's update relative to the global model
            (``honest_model_state[k] - global_state[k]`` for the keys this
            attack perturbs), keyed exactly as the tensors to be perturbed.
        rho: Noise-to-update L2-norm ratio; must be ``>= 0``.
        seed: The experiment's base seed.
        round_id: The communication round this noise is for.
        client_id: The client id this noise is for.

    Returns:
        A dict, same keys/shapes/dtypes as ``delta``, holding the noise to
        add to the honest update.

    Raises:
        ValueError: If ``rho < 0``.
        RuntimeError: If ``rho > 0`` and either ``delta`` or the drawn raw
            noise has zero L2 norm (degenerate; cannot rescale a zero
            vector to a nonzero target norm, or scale onto a zero target
            from a nonzero one).
    """
    if rho < 0:
        raise ValueError(f"rho must be >= 0, got {rho}.")
    if rho == 0:
        return {key: torch.zeros_like(tensor) for key, tensor in delta.items()}

    generator = torch.Generator(device="cpu").manual_seed(
        _derive_attack_seed(seed, round_id, client_id)
    )
    # Always draw in sorted-key order, never raw dict/mapping iteration order:
    # a caller may pass a set/frozenset-derived key ordering (e.g. a
    # trainable-parameter-name frozenset), whose iteration order depends on
    # Python's per-process hash-seed randomization and is therefore not
    # guaranteed stable across separate process invocations -- sorting here
    # is what actually makes cross-run reproducibility (Section 4 of the
    # plan) hold, not just same-process determinism.
    raw_noise = {
        key: torch.randn(delta[key].shape, generator=generator) for key in sorted(delta.keys())
    }

    delta_norm = flattened_l2_norm(delta)
    raw_noise_norm = flattened_l2_norm(raw_noise)
    if delta_norm == 0:
        raise RuntimeError(
            "Cannot scale Gaussian attack noise: the honest update has zero L2 norm."
        )
    if raw_noise_norm == 0:
        raise RuntimeError(
            "Cannot scale Gaussian attack noise: the drawn raw noise has zero L2 norm."
        )

    scale = (rho * delta_norm) / raw_noise_norm
    return {
        key: (tensor * scale).to(delta[key].dtype) for key, tensor in raw_noise.items()
    }


def apply_gaussian_update_noise(
    honest_model_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    trainable_keys: Iterable[str],
    rho: float,
    seed: int,
    round_id: int,
    client_id: str,
) -> dict[str, torch.Tensor]:
    """Return a perturbed copy of ``honest_model_state`` implementing the ``gaussian_update_noise`` attack.

    Only ``trainable_keys`` are perturbed (mirroring FedProx's own
    trainable-parameters-only scope, per the plan's Section 5) -- every
    other key (non-trainable buffers, e.g. BatchNorm running statistics or
    ``num_batches_tracked``) is copied through unchanged, exactly as
    honest local training produced it.

    ``delta = honest_model_state[k] - global_state[k]`` for ``k`` in
    ``trainable_keys``; the returned state's trainable keys are
    ``honest_model_state[k] + noise[k]`` where ``noise`` is generated by
    ``generate_gaussian_attack_noise``. With ``rho == 0`` this is
    byte-identical to ``honest_model_state`` (required no-op property).
    """
    keys = list(trainable_keys)
    delta = {key: honest_model_state[key] - global_state[key] for key in keys}
    noise = generate_gaussian_attack_noise(
        delta, rho=rho, seed=seed, round_id=round_id, client_id=client_id
    )

    perturbed = {key: value.detach().clone() for key, value in honest_model_state.items()}
    for key in keys:
        perturbed[key] = (honest_model_state[key] + noise[key]).detach().clone()
    return perturbed


def build_trace_row(
    round_id: int,
    client_id: str,
    selected: bool,
    successful: bool,
    attack_config: AttackConfig,
    client_delta: Mapping[str, torch.Tensor] | None,
    peer_deltas: Sequence[Mapping[str, torch.Tensor]],
    trainable_keys: Iterable[str],
) -> dict[str, Any]:
    """Build one ground-truth/diagnostic client-round trace row (plan Sections 9-10).

    ``update_l2_norm``/``peer_mean_cosine_similarity`` are only computed
    (non-``None``) when ``successful`` and ``client_delta is not None`` --
    a client that was not selected, or was selected but failed to submit
    an update, gets ``None`` for both diagnostic fields, never a fabricated
    value. The ``naive_threshold``/``naive_flag``/``first_flagged_round``
    fields are intentionally **not** set here -- they require the full
    run's pre-attack window and are added afterward by
    ``apply_naive_detector``.
    """
    update_l2_norm: float | None = None
    peer_cosine: float | None = None
    if successful and client_delta is not None:
        update_l2_norm = flattened_l2_norm(client_delta)
        peer_cosine = compute_peer_mean_cosine_similarity(client_delta, list(peer_deltas), trainable_keys)

    return {
        "round_id": round_id,
        "client_id": client_id,
        "selected": selected,
        "attack_configured": attack_config.attack_type != "none",
        "attack_active": attack_config.is_active(round_id, client_id),
        "attack_type": attack_config.attack_type,
        "attack_strength_rho": attack_config.attack_strength_rho,
        "malicious_client_id": attack_config.malicious_client_id,
        "attack_start_round": attack_config.attack_start_round,
        "update_l2_norm": update_l2_norm,
        "peer_mean_cosine_similarity": peer_cosine,
    }


def apply_naive_detector(
    rows: list[dict[str, Any]], attack_config: AttackConfig
) -> list[dict[str, Any]]:
    """Apply the Phase 5 naive per-client peer-agreement baseline detector.

    Per ``research_plans/PHASE5_Malicious_Client_Traceability_Plan.md``
    Section 11 -- **this is only a baseline detector, not FedTrace.** For
    each client independently: computes the sample mean/standard deviation
    of ``peer_mean_cosine_similarity`` over rows with
    ``round_id < attack_config.attack_start_round`` (the pre-attack window
    only -- rows at or after ``attack_start_round`` are never used to
    compute the threshold, regardless of their content). Rows at or after
    ``attack_start_round`` are flagged (``naive_flag=True``) when that
    client's ``peer_mean_cosine_similarity`` falls below
    ``mean_pre_attack - 3 * std_pre_attack``.

    Does not mutate ``rows``; returns a new list of shallow-copied dicts
    with ``naive_threshold`` (per client, constant across its rows),
    ``naive_flag`` (per row), and ``first_flagged_round`` (per client,
    constant across its rows; ``None`` if the client is never flagged)
    added.

    If ``attack_config.attack_start_round is None`` (no attack configured
    at all, i.e. ``attack_type == "none"``), the pre/post-attack split is
    undefined; every row gets ``naive_threshold=None``,
    ``naive_flag=False``, ``first_flagged_round=None`` -- the detector is
    inert. A client with fewer than 2 pre-attack observations (undefined
    sample standard deviation) similarly gets ``naive_threshold=None`` and
    is never flagged.
    """
    result_rows = [dict(row) for row in rows]
    attack_start_round = attack_config.attack_start_round

    if attack_start_round is None:
        for row in result_rows:
            row["naive_threshold"] = None
            row["naive_flag"] = False
            row["first_flagged_round"] = None
        return result_rows

    rows_by_client: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        rows_by_client.setdefault(row["client_id"], []).append(row)

    for client_rows in rows_by_client.values():
        pre_attack_values = [
            row["peer_mean_cosine_similarity"]
            for row in client_rows
            if row["round_id"] < attack_start_round
            and row["peer_mean_cosine_similarity"] is not None
        ]

        threshold: float | None = None
        if len(pre_attack_values) >= 2:
            mean_pre = statistics.mean(pre_attack_values)
            std_pre = statistics.stdev(pre_attack_values)
            threshold = mean_pre - 3.0 * std_pre

        first_flagged_round: int | None = None
        for row in sorted(client_rows, key=lambda r: r["round_id"]):
            row["naive_threshold"] = threshold
            flagged = (
                threshold is not None
                and row["round_id"] >= attack_start_round
                and row["peer_mean_cosine_similarity"] is not None
                and row["peer_mean_cosine_similarity"] < threshold
            )
            row["naive_flag"] = flagged
            if flagged and first_flagged_round is None:
                first_flagged_round = row["round_id"]

        for row in client_rows:
            row["first_flagged_round"] = first_flagged_round

    return result_rows


def save_client_round_trace_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Save the per-client-round ground-truth/diagnostic/detector trace as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRACE_CSV_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)
