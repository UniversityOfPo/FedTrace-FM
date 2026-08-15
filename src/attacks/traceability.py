"""Malicious-client attack simulation and traceability pilot infrastructure (Phase 5).

Implements only the minimum pieces pre-registered in
``research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`` for the
``EXP_014`` pilot: the ``gaussian_update_noise`` attack transform (Section
5), the cheap diagnostic update features (Section 10), ground-truth
client-round bookkeeping (Section 9), and the naive per-client
peer-agreement baseline detector (Section 11) -- plus, per
``research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md``, a
second attack transform, ``norm_preserving_directional`` (Section 4), and
the additional ground-truth/diagnostic trace fields it requires
(Sections 14-15).

This module deliberately does **not** implement the FedTrace attribution
algorithm (the plans' Section 21/20 explicit boundary) —
``src/attribution/`` remains an empty placeholder. Everything here
operates purely on already-computed, in-memory model-update tensors
(client-reported state dicts and the pre-round global state dict); no
dataset, dataloader, or test-set object is read, imported, or referenced
anywhere in this module.
"""

from __future__ import annotations

import csv
import hashlib
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_VALID_ATTACK_TYPES: tuple[str, ...] = (
    "none",
    "gaussian_update_noise",
    "norm_preserving_directional",
)

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
    "directional_target_cosine",
    "delta_attack_norm_ratio",
    "realized_cosine_to_honest",
    "self_prev_round_cosine",
)


@dataclass(frozen=True)
class AttackConfig:
    """Fully describes (or disables) the single first-order attack scenario for one run.

    Attributes:
        attack_type: ``"none"`` (the default; a strict no-op),
            ``"gaussian_update_noise"``, or ``"norm_preserving_directional"``.
        malicious_client_id: The one client id perturbed when the attack is
            active. Required (non-empty) when ``attack_type != "none"``.
        attack_start_round: The first round (0-indexed, inclusive) at which
            the attack becomes active for ``malicious_client_id``. Required
            when ``attack_type != "none"``.
        attack_strength_rho: The noise-to-update L2-norm ratio (Section 5
            of ``PHASE5_Malicious_Client_Traceability_Plan.md``). Must be
            ``>= 0``; must be ``> 0`` when
            ``attack_type == "gaussian_update_noise"`` (a zero-strength
            attack is indistinguishable from no attack and is rejected as
            a likely configuration mistake, mirroring
            ``FederatedClientConfig.fedprox_mu``'s equivalent check).
            Inert (unused, not required) for ``"norm_preserving_directional"``.
        directional_target_cosine: The target cosine similarity between the
            attacked update and the client's own honest update (Section 7
            of ``EXP_015_Norm_Preserving_Directional_Attack_Plan.md``), in
            ``[-1, 1]``. Required when
            ``attack_type == "norm_preserving_directional"``. Validated to
            lie in ``[-1, 1]`` whenever it is set at all (mirroring
            ``attack_strength_rho``'s always-checked ``>= 0`` bound), but
            otherwise inert (unused, not required) for every other
            ``attack_type``.
        attack_active_windows: Optional explicit set of inclusive
            ``(start, end)`` round intervals during which the attack is
            active for ``malicious_client_id``, for intermittent (rather
            than continuous-from-onset) malicious behavior (NDSS E3;
            see ``research_plans/`` for the corresponding experiment
            design, if present). Defaults to ``None`` -- a strict no-op
            relative to every config/call site predating this addition:
            when ``None``, ``is_active`` uses exactly the original
            continuous-from-``attack_start_round`` rule below, unchanged.
            When provided, must be non-empty, sorted, non-overlapping, and
            its first window's start must equal ``attack_start_round``
            (which remains the single source of truth for "the true attack
            onset" read by every existing evaluation/analysis script).
    """

    attack_type: str = "none"
    malicious_client_id: str | None = None
    attack_start_round: int | None = None
    attack_strength_rho: float = 0.0
    directional_target_cosine: float | None = None
    attack_active_windows: tuple[tuple[int, int], ...] | None = None

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
        if self.directional_target_cosine is not None and not (
            -1.0 <= self.directional_target_cosine <= 1.0
        ):
            raise ValueError(
                "directional_target_cosine must be in [-1, 1], got "
                f"{self.directional_target_cosine}."
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
        if (
            self.attack_type == "norm_preserving_directional"
            and self.directional_target_cosine is None
        ):
            raise ValueError(
                "directional_target_cosine is required when "
                "attack_type='norm_preserving_directional'."
            )
        if self.attack_active_windows is not None:
            if self.attack_type == "none":
                raise ValueError("attack_active_windows must not be set when attack_type == 'none'.")
            if len(self.attack_active_windows) == 0:
                raise ValueError(
                    "attack_active_windows must be non-empty when provided (use None, not an "
                    "empty tuple/list, to disable intermittent scheduling)."
                )
            prev_end: int | None = None
            for window in self.attack_active_windows:
                if len(window) != 2:
                    raise ValueError(
                        f"Each attack_active_windows entry must be a (start, end) pair, got {window!r}."
                    )
                start, end = window
                if start < 0 or end < start:
                    raise ValueError(
                        f"Invalid attack_active_windows entry {window!r}: require 0 <= start <= end."
                    )
                if prev_end is not None and start <= prev_end:
                    raise ValueError(
                        "attack_active_windows must be sorted and non-overlapping; window "
                        f"{window!r} starts at or before the previous window's end ({prev_end})."
                    )
                prev_end = end
            if self.attack_active_windows[0][0] != self.attack_start_round:
                raise ValueError(
                    "attack_active_windows[0][0] must equal attack_start_round (the single "
                    f"source of truth for the true attack onset): attack_start_round="
                    f"{self.attack_start_round}, first window starts at "
                    f"{self.attack_active_windows[0][0]}."
                )

    def is_active(self, round_id: int, client_id: str) -> bool:
        """Return whether the attack is active for ``client_id`` at ``round_id``.

        ``False`` whenever ``attack_type == "none"``, for every client
        other than ``malicious_client_id``, and for every round before
        ``attack_start_round`` -- the delayed-onset, single-client
        semantics required by the plan's Section 3/7.

        When ``attack_active_windows`` is ``None`` (the default, and the
        only possibility for every config predating this field), this is
        byte-for-byte the original continuous-from-onset rule:
        ``round_id >= attack_start_round``. When ``attack_active_windows``
        is set, it instead governs activity exactly: ``True`` only for
        ``round_id`` falling within one of the given inclusive
        ``[start, end]`` windows, ``False`` everywhere else (including
        rounds ``>= attack_start_round`` that fall in a gap between
        windows) -- intermittent, not continuous, malicious behavior.
        """
        if (
            self.attack_type == "none"
            or client_id != self.malicious_client_id
            or self.attack_start_round is None
        ):
            return False
        if self.attack_active_windows is not None:
            return any(start <= round_id <= end for start, end in self.attack_active_windows)
        return round_id >= self.attack_start_round


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


def _derive_directional_attack_seed(seed: int, round_id: int, client_id: str) -> int:
    """Deterministically derive a per-(seed, round, client, attack-type) integer RNG seed.

    Identical construction to ``_derive_attack_seed``, but with the
    ``norm_preserving_directional`` attack-type identifier folded into the
    hashed key (per
    ``research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md``
    Section 5: "seeded only from experiment seed, round_id, client_id,
    attack type"). This is a deliberately separate function -- not a new
    parameter on ``_derive_attack_seed`` -- so the Gaussian attack's own
    derived seeds (and therefore its noise draws, already frozen in
    ``EXP_014``'s results) are provably unaffected by this addition: the
    directional attack's random-direction draws are seeded independently
    of, and never coincide with, the Gaussian attack's noise draws for the
    same ``(seed, round_id, client_id)`` triple.
    """
    digest = hashlib.sha256(
        f"{seed}-{round_id}-{client_id}-norm_preserving_directional".encode("utf-8")
    ).digest()
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


def compute_cosine_similarity(
    vec_a: Mapping[str, torch.Tensor],
    vec_b: Mapping[str, torch.Tensor],
    trainable_keys: Iterable[str],
) -> float | None:
    """Cosine similarity between two update-shaped tensors, flattened over ``trainable_keys``.

    Used for the ``norm_preserving_directional`` attack's
    ``realized_cosine_to_honest`` verification field and for the
    ``self_prev_round_cosine`` temporal diagnostic (Sections 10/15 of
    ``research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md``)
    -- a general two-vector cosine helper, kept separate from
    ``compute_peer_mean_cosine_similarity`` (which compares one client
    against a *mean* of several peers) so that existing function is left
    completely unmodified.

    Returns ``None`` (undefined, never fabricated as 0.0) if either vector
    has zero L2 norm.
    """
    keys = list(trainable_keys)
    a = _flatten_sorted(vec_a, keys)
    b = _flatten_sorted(vec_b, keys)
    a_norm = a.norm(p=2)
    b_norm = b.norm(p=2)
    if a_norm == 0 or b_norm == 0:
        return None
    return float(torch.dot(a, b) / (a_norm * b_norm))


def compute_norm_ratio(
    submitted_delta: Mapping[str, torch.Tensor],
    honest_delta: Mapping[str, torch.Tensor],
) -> float | None:
    """Return ``||submitted_delta||_2 / ||honest_delta||_2``.

    Used for the ``norm_preserving_directional`` attack's
    ``delta_attack_norm_ratio`` verification field (expected ``~1.0`` when
    the attack is behaving correctly). Returns ``None`` (undefined) if
    ``honest_delta`` has zero L2 norm.
    """
    honest_norm = flattened_l2_norm(honest_delta)
    if honest_norm == 0:
        return None
    return flattened_l2_norm(submitted_delta) / honest_norm


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


def generate_directional_attack_perturbation(
    delta: Mapping[str, torch.Tensor],
    target_cosine: float,
    seed: int,
    round_id: int,
    client_id: str,
) -> dict[str, torch.Tensor]:
    """Generate the deterministic, norm-preserving ``delta_attack`` for one client-round.

    Per ``research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md``
    Section 4: constructs a unit vector ``direction_attack`` that is an
    exact rotation of ``delta``'s own unit direction (``delta_hat``) by
    ``theta = arccos(target_cosine)``, within the 2-plane spanned by
    ``delta_hat`` and a random, ``delta_hat``-orthogonal unit direction
    (``u_perp``), then rescales by ``||delta||_2``. Because
    ``direction_attack`` is an exact rotation of a unit vector within an
    orthonormal 2-plane basis, the result satisfies
    ``||delta_attack||_2 == ||delta||_2`` and
    ``cos(delta_attack, delta) == target_cosine`` **exactly** (up to
    floating-point rounding) -- not merely approximately, unlike
    ``generate_gaussian_attack_noise``'s empirical rescaling of
    independent noise.

    Handles two degenerate/boundary cases explicitly and deterministically:

    - ``||delta||_2 == 0``: there is no honest direction to rotate, and
      ``delta_attack`` must also have zero norm; returns exact zero
      tensors, drawing no randomness at all (mirrors
      ``generate_gaussian_attack_noise``'s ``rho == 0`` short-circuit).
    - ``target_cosine == 1.0`` or ``target_cosine == -1.0``: the random
      orthogonal component's coefficient (``sqrt(1 - target_cosine**2)``)
      is exactly zero, so ``u_perp`` never contributes to the result;
      these two cases are special-cased to skip the random-direction draw
      and Gram-Schmidt step entirely, both as a minor efficiency and to
      avoid depending on the (already near-zero-probability) orthogonal
      construction succeeding when its result would be multiplied by
      zero anyway.

    Args:
        delta: The honest client's update relative to the global model
            (``honest_model_state[k] - global_state[k]``), keyed exactly
            as the tensors to be perturbed.
        target_cosine: The intended ``cos(delta_attack, delta)``; must be
            in ``[-1, 1]``.
        seed: The experiment's base seed.
        round_id: The communication round this perturbation is for.
        client_id: The client id this perturbation is for.

    Returns:
        A dict, same keys/shapes/dtypes as ``delta``, holding
        ``delta_attack`` (**not** an additive perturbation to be summed
        with ``delta`` -- this replaces ``delta`` entirely; see
        ``apply_norm_preserving_directional_attack``).

    Raises:
        ValueError: If ``target_cosine`` is outside ``[-1, 1]``.
        RuntimeError: If the randomly-drawn direction lands numerically
            exactly parallel to ``delta_hat`` (a near-zero-probability
            degenerate case in a high-dimensional parameter space) and
            ``target_cosine`` is not ``+-1`` (where this case cannot
            arise, since the draw is skipped entirely).
    """
    if not (-1.0 <= target_cosine <= 1.0):
        raise ValueError(f"target_cosine must be in [-1, 1], got {target_cosine}.")

    delta_norm = flattened_l2_norm(delta)
    if delta_norm == 0:
        return {key: torch.zeros_like(tensor) for key, tensor in delta.items()}

    sorted_keys = sorted(delta.keys())
    delta_hat = {key: delta[key].detach().double() / delta_norm for key in sorted_keys}

    if target_cosine == 1.0:
        direction: dict[str, torch.Tensor] = delta_hat
    elif target_cosine == -1.0:
        direction = {key: -tensor for key, tensor in delta_hat.items()}
    else:
        generator = torch.Generator(device="cpu").manual_seed(
            _derive_directional_attack_seed(seed, round_id, client_id)
        )
        raw = {
            key: torch.randn(delta[key].shape, generator=generator) for key in sorted_keys
        }
        dot_r_deltahat = sum(
            (raw[key].double() * delta_hat[key]).sum() for key in sorted_keys
        )
        u_raw = {
            key: raw[key].double() - dot_r_deltahat * delta_hat[key] for key in sorted_keys
        }
        u_raw_norm = flattened_l2_norm(u_raw)
        if u_raw_norm == 0:
            raise RuntimeError(
                "Cannot construct directional attack perturbation: the randomly-drawn "
                "direction landed numerically exactly parallel to the honest update's own "
                "direction (a near-zero-probability degenerate case in this many dimensions)."
            )
        u_perp = {key: u_raw[key] / u_raw_norm for key in sorted_keys}

        s = math.sqrt(max(0.0, 1.0 - target_cosine * target_cosine))
        direction = {
            key: target_cosine * delta_hat[key] + s * u_perp[key] for key in sorted_keys
        }

    return {key: (delta_norm * direction[key]).to(delta[key].dtype) for key in sorted_keys}


def apply_norm_preserving_directional_attack(
    honest_model_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    trainable_keys: Iterable[str],
    target_cosine: float,
    seed: int,
    round_id: int,
    client_id: str,
) -> dict[str, torch.Tensor]:
    """Return a perturbed copy of ``honest_model_state`` implementing the ``norm_preserving_directional`` attack.

    Only ``trainable_keys`` are perturbed (identical scope to
    ``apply_gaussian_update_noise``) -- every other key (non-trainable
    buffers, e.g. BatchNorm running statistics or ``num_batches_tracked``)
    is copied through unchanged, exactly as honest local training produced
    it.

    Unlike ``apply_gaussian_update_noise`` (which *adds* noise to the
    honest delta), this attack *replaces* the honest delta entirely with
    ``delta_attack`` (``generate_directional_attack_perturbation``'s
    return value) -- the submitted state is therefore
    ``global_state[k] + delta_attack[k]``, not
    ``honest_model_state[k] + delta_attack[k]``.
    """
    keys = list(trainable_keys)
    delta = {key: honest_model_state[key] - global_state[key] for key in keys}
    delta_attack = generate_directional_attack_perturbation(
        delta, target_cosine=target_cosine, seed=seed, round_id=round_id, client_id=client_id
    )

    perturbed = {key: value.detach().clone() for key, value in honest_model_state.items()}
    for key in keys:
        perturbed[key] = (global_state[key] + delta_attack[key]).detach().clone()
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
    honest_delta: Mapping[str, torch.Tensor] | None = None,
    self_prev_delta: Mapping[str, torch.Tensor] | None = None,
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

    ``honest_delta``/``self_prev_delta`` are optional (default ``None``,
    a strict no-op relative to the pre-Phase-5.2 signature -- every
    existing caller/test that omits them continues to work unchanged),
    per ``research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md``:

    - ``honest_delta``: this client's *pre-perturbation* honest update for
      this round, if the caller has it available. Used only to compute
      ``delta_attack_norm_ratio``/``realized_cosine_to_honest`` -- and
      only when this row is an *active* ``norm_preserving_directional``
      attack round for this client; every other row gets ``None`` for
      both fields (a clearly documented inert value, never a fabricated
      ~1.0/~1.0 for rows where the attack does not apply).
    - ``self_prev_delta``: this same client's *own* submitted delta from
      its most recent prior successful round, if any (``None`` on a
      client's first observed round, or if the caller has no such
      memory). Used to compute ``self_prev_round_cosine`` -- available
      for any attack type/client, not only the directional attack, since
      it is a general temporal self-consistency diagnostic.
    """
    update_l2_norm: float | None = None
    peer_cosine: float | None = None
    if successful and client_delta is not None:
        update_l2_norm = flattened_l2_norm(client_delta)
        peer_cosine = compute_peer_mean_cosine_similarity(client_delta, list(peer_deltas), trainable_keys)

    directional_norm_ratio: float | None = None
    directional_realized_cosine: float | None = None
    is_directional_active_row = (
        attack_config.attack_type == "norm_preserving_directional"
        and attack_config.is_active(round_id, client_id)
    )
    if (
        successful
        and is_directional_active_row
        and client_delta is not None
        and honest_delta is not None
    ):
        directional_norm_ratio = compute_norm_ratio(client_delta, honest_delta)
        directional_realized_cosine = compute_cosine_similarity(
            client_delta, honest_delta, trainable_keys
        )

    self_cosine: float | None = None
    if successful and client_delta is not None and self_prev_delta is not None:
        self_cosine = compute_cosine_similarity(client_delta, self_prev_delta, trainable_keys)

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
        "directional_target_cosine": attack_config.directional_target_cosine,
        "delta_attack_norm_ratio": directional_norm_ratio,
        "realized_cosine_to_honest": directional_realized_cosine,
        "self_prev_round_cosine": self_cosine,
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
