"""Lightweight tests for the Phase 5 attack/trace pilot infrastructure.

Uses tiny synthetic tensors throughout, so the suite runs in seconds on
CPU with no dataset, dataloader, GPU, or test-set object ever constructed
-- directly exercising (and, for the last test, statically verifying) the
"no test-set information" requirement of
``research_plans/PHASE5_Malicious_Client_Traceability_Plan.md``.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from src.attacks import traceability
from src.attacks.traceability import (
    AttackConfig,
    apply_gaussian_update_noise,
    apply_naive_detector,
    apply_norm_preserving_directional_attack,
    build_trace_row,
    compute_cosine_similarity,
    compute_norm_ratio,
    compute_peer_mean_cosine_similarity,
    compute_update_l2_norm,
    flattened_l2_norm,
    generate_directional_attack_perturbation,
    generate_gaussian_attack_noise,
)

TRAINABLE_KEYS = ("layer.weight", "layer.bias")


def _tensors(*values: float, shape: tuple[int, ...] = (2,)) -> dict[str, torch.Tensor]:
    """Build a tiny two-key state dict, one constant value per key."""
    return {key: torch.full(shape, value) for key, value in zip(TRAINABLE_KEYS, values)}


# ----------------------------------------------------------------------
# AttackConfig validation and is_active()
# ----------------------------------------------------------------------


def test_attack_config_defaults_are_inert() -> None:
    """The default AttackConfig ('none') is never active, for any client/round."""
    config = AttackConfig()
    assert config.attack_type == "none"
    assert config.is_active(round_id=0, client_id="client_002") is False
    assert config.is_active(round_id=999, client_id="client_002") is False


def test_attack_config_rejects_invalid_attack_type() -> None:
    with pytest.raises(ValueError):
        AttackConfig(attack_type="not_a_type")


def test_attack_config_rejects_negative_rho() -> None:
    with pytest.raises(ValueError):
        AttackConfig(attack_strength_rho=-0.1)


def test_attack_config_requires_malicious_client_id_when_active() -> None:
    with pytest.raises(ValueError):
        AttackConfig(
            attack_type="gaussian_update_noise", attack_start_round=5, attack_strength_rho=1.0
        )


def test_attack_config_requires_attack_start_round_when_active() -> None:
    with pytest.raises(ValueError):
        AttackConfig(
            attack_type="gaussian_update_noise",
            malicious_client_id="client_002",
            attack_strength_rho=1.0,
        )


def test_attack_config_requires_positive_rho_for_gaussian_update_noise() -> None:
    with pytest.raises(ValueError):
        AttackConfig(
            attack_type="gaussian_update_noise",
            malicious_client_id="client_002",
            attack_start_round=20,
            attack_strength_rho=0.0,
        )


def test_attack_config_accepts_well_formed_gaussian_attack() -> None:
    config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_002",
        attack_start_round=20,
        attack_strength_rho=1.0,
    )
    assert config.attack_type == "gaussian_update_noise"


def test_is_active_inactive_before_start_round() -> None:
    """The attack is inactive for the malicious client on every round before attack_start_round."""
    config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_002",
        attack_start_round=20,
        attack_strength_rho=1.0,
    )
    for round_id in (0, 1, 19):
        assert config.is_active(round_id=round_id, client_id="client_002") is False
    for round_id in (20, 21, 49):
        assert config.is_active(round_id=round_id, client_id="client_002") is True


def test_is_active_only_true_for_malicious_client() -> None:
    """The attack never activates for any client other than malicious_client_id."""
    config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_002",
        attack_start_round=20,
        attack_strength_rho=1.0,
    )
    for other_client in ("client_000", "client_001", "client_003", "client_004"):
        assert config.is_active(round_id=49, client_id=other_client) is False


# ----------------------------------------------------------------------
# Deterministic, norm-scaled Gaussian noise
# ----------------------------------------------------------------------


def test_generate_gaussian_attack_noise_zero_rho_is_zero() -> None:
    """rho=0 always returns exact zero noise, with no RNG draw needed."""
    delta = _tensors(3.0, -2.0)
    noise = generate_gaussian_attack_noise(delta, rho=0.0, seed=42, round_id=20, client_id="client_002")
    for tensor in noise.values():
        assert torch.equal(tensor, torch.zeros_like(tensor))


def test_generate_gaussian_attack_noise_rejects_negative_rho() -> None:
    with pytest.raises(ValueError):
        generate_gaussian_attack_noise(
            _tensors(1.0, 1.0), rho=-1.0, seed=42, round_id=20, client_id="client_002"
        )


def test_generate_gaussian_attack_noise_matches_target_norm() -> None:
    """||noise||_2 == rho * ||delta||_2, within a small numerical tolerance."""
    delta = _tensors(3.0, -4.0, shape=(10,))  # ||delta||_2 = sqrt(10*9 + 10*16) = sqrt(250)
    delta_norm = flattened_l2_norm(delta)
    rho = 1.7
    noise = generate_gaussian_attack_noise(delta, rho=rho, seed=7, round_id=3, client_id="client_002")
    noise_norm = flattened_l2_norm(noise)
    assert noise_norm == pytest.approx(rho * delta_norm, rel=1e-5)


def test_generate_gaussian_attack_noise_is_deterministic() -> None:
    """Identical (delta, rho, seed, round_id, client_id) always produces identical noise."""
    delta = _tensors(1.0, 2.0, shape=(6,))
    first = generate_gaussian_attack_noise(delta, rho=1.0, seed=42, round_id=20, client_id="client_002")
    second = generate_gaussian_attack_noise(delta, rho=1.0, seed=42, round_id=20, client_id="client_002")
    for key in delta:
        assert torch.equal(first[key], second[key])


def test_generate_gaussian_attack_noise_differs_across_rounds() -> None:
    """A different round_id (all else equal) produces different noise."""
    delta = _tensors(1.0, 2.0, shape=(6,))
    noise_r20 = generate_gaussian_attack_noise(delta, rho=1.0, seed=42, round_id=20, client_id="client_002")
    noise_r21 = generate_gaussian_attack_noise(delta, rho=1.0, seed=42, round_id=21, client_id="client_002")
    assert any(not torch.equal(noise_r20[k], noise_r21[k]) for k in delta)


def test_generate_gaussian_attack_noise_differs_across_seeds() -> None:
    """A different experiment seed (all else equal) produces different noise."""
    delta = _tensors(1.0, 2.0, shape=(6,))
    noise_seed_a = generate_gaussian_attack_noise(delta, rho=1.0, seed=42, round_id=20, client_id="client_002")
    noise_seed_b = generate_gaussian_attack_noise(delta, rho=1.0, seed=123, round_id=20, client_id="client_002")
    assert any(not torch.equal(noise_seed_a[k], noise_seed_b[k]) for k in delta)


def test_generate_gaussian_attack_noise_differs_across_clients() -> None:
    """A different client_id (all else equal) produces different noise -- per-client independence."""
    delta = _tensors(1.0, 2.0, shape=(6,))
    noise_a = generate_gaussian_attack_noise(delta, rho=1.0, seed=42, round_id=20, client_id="client_001")
    noise_b = generate_gaussian_attack_noise(delta, rho=1.0, seed=42, round_id=20, client_id="client_002")
    assert any(not torch.equal(noise_a[k], noise_b[k]) for k in delta)


def test_generate_gaussian_attack_noise_rejects_zero_delta() -> None:
    """A zero honest-update delta cannot be rescaled to a nonzero target norm."""
    delta = {key: torch.zeros(4) for key in TRAINABLE_KEYS}
    with pytest.raises(RuntimeError):
        generate_gaussian_attack_noise(delta, rho=1.0, seed=42, round_id=20, client_id="client_002")


# ----------------------------------------------------------------------
# apply_gaussian_update_noise: only the malicious client's trainable keys change
# ----------------------------------------------------------------------


def test_apply_gaussian_update_noise_perturbs_only_trainable_keys() -> None:
    """Non-trainable buffer keys (e.g. BatchNorm stats) pass through completely unchanged."""
    honest_state = {
        "layer.weight": torch.full((4,), 2.0),
        "layer.bias": torch.full((4,), 1.0),
        "bn.running_mean": torch.full((4,), 0.5),
        "bn.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
    }
    global_state = {
        "layer.weight": torch.zeros(4),
        "layer.bias": torch.zeros(4),
        "bn.running_mean": torch.full((4,), 0.5),
        "bn.num_batches_tracked": torch.tensor(6, dtype=torch.int64),
    }
    perturbed = apply_gaussian_update_noise(
        honest_state, global_state, TRAINABLE_KEYS, rho=1.0, seed=42, round_id=20, client_id="client_002"
    )
    assert not torch.equal(perturbed["layer.weight"], honest_state["layer.weight"])
    assert torch.equal(perturbed["bn.running_mean"], honest_state["bn.running_mean"])
    assert torch.equal(perturbed["bn.num_batches_tracked"], honest_state["bn.num_batches_tracked"])


def test_apply_gaussian_update_noise_rho_zero_is_byte_identical_to_honest() -> None:
    """rho=0 recovers the honest update exactly (required no-op property)."""
    honest_state = _tensors(2.0, 1.0, shape=(4,))
    global_state = {key: torch.zeros(4) for key in TRAINABLE_KEYS}
    perturbed = apply_gaussian_update_noise(
        honest_state, global_state, TRAINABLE_KEYS, rho=0.0, seed=42, round_id=20, client_id="client_002"
    )
    for key in TRAINABLE_KEYS:
        assert torch.equal(perturbed[key], honest_state[key])


def test_apply_gaussian_update_noise_matches_target_norm_relative_to_delta() -> None:
    """The applied perturbation's own norm matches rho * ||honest delta||_2."""
    honest_state = _tensors(5.0, 5.0, shape=(8,))
    global_state = {key: torch.zeros(8) for key in TRAINABLE_KEYS}
    rho = 0.5
    perturbed = apply_gaussian_update_noise(
        honest_state, global_state, TRAINABLE_KEYS, rho=rho, seed=1, round_id=20, client_id="client_002"
    )
    applied_noise = {key: perturbed[key] - honest_state[key] for key in TRAINABLE_KEYS}
    honest_delta = {key: honest_state[key] - global_state[key] for key in TRAINABLE_KEYS}
    assert flattened_l2_norm(applied_noise) == pytest.approx(rho * flattened_l2_norm(honest_delta), rel=1e-5)


# ----------------------------------------------------------------------
# Diagnostic features: update L2 norm, peer-mean cosine similarity
# ----------------------------------------------------------------------


def test_compute_update_l2_norm_matches_hand_computation() -> None:
    model_state = {"a": torch.tensor([4.0, 0.0]), "b": torch.tensor([0.0, 3.0])}
    global_state = {"a": torch.zeros(2), "b": torch.zeros(2)}
    # ||[4, 0, 0, 3]||_2 = sqrt(16 + 9) = 5
    norm = compute_update_l2_norm(model_state, global_state, ["a", "b"])
    assert norm == pytest.approx(5.0)


def test_compute_peer_mean_cosine_similarity_identical_direction() -> None:
    """A client whose update is a positive scalar multiple of the peer mean has cosine similarity 1.0."""
    client_delta = {"a": torch.tensor([2.0, 0.0])}
    peer_deltas = [{"a": torch.tensor([1.0, 0.0])}, {"a": torch.tensor([1.0, 0.0])}]
    similarity = compute_peer_mean_cosine_similarity(client_delta, peer_deltas, ["a"])
    assert similarity == pytest.approx(1.0)


def test_compute_peer_mean_cosine_similarity_opposite_direction() -> None:
    """A client whose update points exactly opposite the peer mean has cosine similarity -1.0."""
    client_delta = {"a": torch.tensor([-1.0, 0.0])}
    peer_deltas = [{"a": torch.tensor([1.0, 0.0])}, {"a": torch.tensor([1.0, 0.0])}]
    similarity = compute_peer_mean_cosine_similarity(client_delta, peer_deltas, ["a"])
    assert similarity == pytest.approx(-1.0)


def test_compute_peer_mean_cosine_similarity_orthogonal() -> None:
    client_delta = {"a": torch.tensor([0.0, 1.0])}
    peer_deltas = [{"a": torch.tensor([1.0, 0.0])}]
    similarity = compute_peer_mean_cosine_similarity(client_delta, peer_deltas, ["a"])
    assert similarity == pytest.approx(0.0, abs=1e-6)


def test_compute_peer_mean_cosine_similarity_no_peers_returns_none() -> None:
    client_delta = {"a": torch.tensor([1.0, 0.0])}
    assert compute_peer_mean_cosine_similarity(client_delta, [], ["a"]) is None


def test_compute_peer_mean_cosine_similarity_zero_vector_returns_none() -> None:
    """Cosine similarity against/from a zero vector is undefined, not fabricated as 0.0."""
    client_delta = {"a": torch.tensor([0.0, 0.0])}
    peer_deltas = [{"a": torch.tensor([1.0, 0.0])}]
    assert compute_peer_mean_cosine_similarity(client_delta, peer_deltas, ["a"]) is None


# ----------------------------------------------------------------------
# build_trace_row: ground-truth logging correctness
# ----------------------------------------------------------------------


def test_build_trace_row_ground_truth_fields_before_attack_start() -> None:
    config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_002",
        attack_start_round=20,
        attack_strength_rho=1.0,
    )
    row = build_trace_row(
        round_id=5,
        client_id="client_002",
        selected=True,
        successful=True,
        attack_config=config,
        client_delta={"a": torch.tensor([1.0])},
        peer_deltas=[{"a": torch.tensor([1.0])}],
        trainable_keys=["a"],
    )
    assert row["round_id"] == 5
    assert row["client_id"] == "client_002"
    assert row["selected"] is True
    assert row["attack_configured"] is True
    assert row["attack_active"] is False  # round 5 < attack_start_round=20
    assert row["attack_type"] == "gaussian_update_noise"
    assert row["attack_strength_rho"] == pytest.approx(1.0)
    assert row["malicious_client_id"] == "client_002"
    assert row["attack_start_round"] == 20
    assert row["update_l2_norm"] is not None
    assert row["peer_mean_cosine_similarity"] is not None


def test_build_trace_row_attack_active_flag_at_and_after_start_round() -> None:
    config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_002",
        attack_start_round=20,
        attack_strength_rho=1.0,
    )
    for round_id, expected_active in ((19, False), (20, True), (21, True)):
        row = build_trace_row(
            round_id=round_id,
            client_id="client_002",
            selected=True,
            successful=True,
            attack_config=config,
            client_delta={"a": torch.tensor([1.0])},
            peer_deltas=[{"a": torch.tensor([1.0])}],
            trainable_keys=["a"],
        )
        assert row["attack_active"] is expected_active


def test_build_trace_row_unselected_client_has_no_diagnostics() -> None:
    config = AttackConfig()
    row = build_trace_row(
        round_id=0,
        client_id="client_003",
        selected=False,
        successful=False,
        attack_config=config,
        client_delta=None,
        peer_deltas=[],
        trainable_keys=["a"],
    )
    assert row["selected"] is False
    assert row["update_l2_norm"] is None
    assert row["peer_mean_cosine_similarity"] is None
    assert row["attack_configured"] is False
    assert row["attack_active"] is False


# ----------------------------------------------------------------------
# Naive per-client peer-agreement detector
# ----------------------------------------------------------------------


def _rows_for_client(client_id: str, pre_attack: list[float], post_attack: list[float]) -> list[dict]:
    """Build a client's rows: pre_attack values for rounds 0..N-1, post_attack for N..N+M-1."""
    rows = []
    for round_id, value in enumerate(pre_attack):
        rows.append(
            {"round_id": round_id, "client_id": client_id, "peer_mean_cosine_similarity": value}
        )
    start = len(pre_attack)
    for offset, value in enumerate(post_attack):
        rows.append(
            {"round_id": start + offset, "client_id": client_id, "peer_mean_cosine_similarity": value}
        )
    return rows


def test_naive_detector_flags_known_injected_outlier_only() -> None:
    """A synthetic outlier (client, round) is flagged; the honest synthetic client is not."""
    attack_start_round = 5
    # Honest client: tight pre-attack cluster around 0.9, stays there post-attack too.
    honest_rows = _rows_for_client("client_honest", [0.90, 0.91, 0.89, 0.90, 0.90], [0.90, 0.89, 0.91])
    # Malicious client: tight pre-attack cluster around 0.9, collapses post-attack.
    malicious_rows = _rows_for_client("client_malicious", [0.90, 0.91, 0.89, 0.90, 0.90], [0.10, 0.05, 0.02])

    config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_malicious",
        attack_start_round=attack_start_round,
        attack_strength_rho=1.0,
    )
    result_rows = apply_naive_detector(honest_rows + malicious_rows, config)

    honest_flags = [row["naive_flag"] for row in result_rows if row["client_id"] == "client_honest"]
    malicious_flags = [row["naive_flag"] for row in result_rows if row["client_id"] == "client_malicious"]
    malicious_post_flags = [
        row["naive_flag"]
        for row in result_rows
        if row["client_id"] == "client_malicious" and row["round_id"] >= attack_start_round
    ]

    assert not any(honest_flags)
    assert all(malicious_post_flags)
    assert any(malicious_flags)


def test_naive_detector_threshold_is_per_client() -> None:
    """Two clients with different pre-attack distributions get different thresholds."""
    attack_start_round = 4
    client_a_rows = _rows_for_client("client_a", [0.5, 0.5, 0.5, 0.5], [0.5])
    client_b_rows = _rows_for_client("client_b", [0.9, 0.9, 0.9, 0.9], [0.9])

    config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_a",
        attack_start_round=attack_start_round,
        attack_strength_rho=1.0,
    )
    result_rows = apply_naive_detector(client_a_rows + client_b_rows, config)

    threshold_a = {row["naive_threshold"] for row in result_rows if row["client_id"] == "client_a"}
    threshold_b = {row["naive_threshold"] for row in result_rows if row["client_id"] == "client_b"}
    assert len(threshold_a) == 1  # constant per client
    assert len(threshold_b) == 1
    assert threshold_a != threshold_b


def test_naive_detector_uses_only_pre_attack_window() -> None:
    """Post-attack-window values never influence the computed threshold."""
    attack_start_round = 3
    pre_attack = [0.80, 0.82, 0.81]  # tight, low variance
    # Two variants that differ ONLY in post-attack-window values.
    rows_a = _rows_for_client("client_x", pre_attack, [0.80, 0.81])
    rows_b = _rows_for_client("client_x", pre_attack, [-0.90, -0.95])  # wildly different post-attack

    config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_x",
        attack_start_round=attack_start_round,
        attack_strength_rho=1.0,
    )
    result_a = apply_naive_detector(rows_a, config)
    result_b = apply_naive_detector(rows_b, config)

    threshold_a = result_a[0]["naive_threshold"]
    threshold_b = result_b[0]["naive_threshold"]
    assert threshold_a == pytest.approx(threshold_b)


def test_naive_detector_first_flagged_round_is_earliest_flag() -> None:
    attack_start_round = 3
    rows = _rows_for_client("client_x", [0.9, 0.9, 0.9], [0.9, -0.9, -0.9])
    config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_x",
        attack_start_round=attack_start_round,
        attack_strength_rho=1.0,
    )
    result_rows = apply_naive_detector(rows, config)
    first_flagged = {row["first_flagged_round"] for row in result_rows}
    assert first_flagged == {4}  # round 3 (0.9) not flagged; round 4 (-0.9) first flagged


def test_naive_detector_inert_when_no_attack_configured() -> None:
    """attack_type='none' (attack_start_round=None) leaves every row unflagged, threshold=None."""
    rows = _rows_for_client("client_x", [0.9, 0.9], [0.1, -0.9])
    result_rows = apply_naive_detector(rows, AttackConfig())
    assert all(row["naive_flag"] is False for row in result_rows)
    assert all(row["naive_threshold"] is None for row in result_rows)
    assert all(row["first_flagged_round"] is None for row in result_rows)


def test_naive_detector_does_not_mutate_input_rows() -> None:
    rows = _rows_for_client("client_x", [0.9, 0.9], [0.1])
    original = [dict(row) for row in rows]
    apply_naive_detector(rows, AttackConfig())
    assert rows == original


# ----------------------------------------------------------------------
# Architectural guarantee: no dataset / test-set dependency anywhere
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# AttackConfig: norm_preserving_directional plumbing (EXP_015)
# ----------------------------------------------------------------------


def test_attack_config_defaults_directional_target_cosine_to_none() -> None:
    config = AttackConfig()
    assert config.directional_target_cosine is None


def test_attack_config_rejects_out_of_range_directional_target_cosine() -> None:
    with pytest.raises(ValueError):
        AttackConfig(directional_target_cosine=1.5)
    with pytest.raises(ValueError):
        AttackConfig(directional_target_cosine=-1.5)


def test_attack_config_requires_directional_target_cosine_for_directional_attack() -> None:
    with pytest.raises(ValueError):
        AttackConfig(
            attack_type="norm_preserving_directional",
            malicious_client_id="client_002",
            attack_start_round=20,
        )


def test_attack_config_accepts_well_formed_directional_attack() -> None:
    config = AttackConfig(
        attack_type="norm_preserving_directional",
        malicious_client_id="client_002",
        attack_start_round=20,
        directional_target_cosine=0.0,
    )
    assert config.attack_type == "norm_preserving_directional"
    assert config.directional_target_cosine == pytest.approx(0.0)


def test_directional_target_cosine_is_inert_for_gaussian_and_none() -> None:
    """A directional_target_cosine value is accepted (unused) under other attack types."""
    gaussian_config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_002",
        attack_start_round=20,
        attack_strength_rho=1.0,
        directional_target_cosine=0.5,
    )
    assert gaussian_config.directional_target_cosine == pytest.approx(0.5)

    none_config = AttackConfig(attack_type="none", directional_target_cosine=0.5)
    assert none_config.is_active(round_id=99, client_id="client_002") is False


def test_is_active_directional_attack_inactive_before_start_round() -> None:
    config = AttackConfig(
        attack_type="norm_preserving_directional",
        malicious_client_id="client_002",
        attack_start_round=20,
        directional_target_cosine=0.0,
    )
    for round_id in (0, 1, 19):
        assert config.is_active(round_id=round_id, client_id="client_002") is False
    for round_id in (20, 21, 49):
        assert config.is_active(round_id=round_id, client_id="client_002") is True


# ----------------------------------------------------------------------
# generate_directional_attack_perturbation: geometry, determinism, RNG isolation
# ----------------------------------------------------------------------


def test_directional_perturbation_rejects_out_of_range_target_cosine() -> None:
    delta = _tensors(1.0, 1.0, shape=(6,))
    with pytest.raises(ValueError):
        generate_directional_attack_perturbation(delta, target_cosine=1.1, seed=1, round_id=0, client_id="c")
    with pytest.raises(ValueError):
        generate_directional_attack_perturbation(delta, target_cosine=-1.1, seed=1, round_id=0, client_id="c")


def test_directional_perturbation_handles_zero_delta_safely() -> None:
    """A zero-norm honest delta produces an exact zero delta_attack, no RNG draw, no crash."""
    delta = {key: torch.zeros(5) for key in TRAINABLE_KEYS}
    result = generate_directional_attack_perturbation(
        delta, target_cosine=0.3, seed=1, round_id=0, client_id="client_002"
    )
    for tensor in result.values():
        assert torch.equal(tensor, torch.zeros_like(tensor))


def test_directional_perturbation_preserves_norm_within_tolerance() -> None:
    """||delta_attack||_2 / ||delta||_2 is within 1e-4 of 1.0, across several target_cosine values."""
    delta = _tensors(3.0, -4.0, shape=(50,))
    delta_norm = flattened_l2_norm(delta)
    for target_cosine in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0):
        perturbation = generate_directional_attack_perturbation(
            delta, target_cosine=target_cosine, seed=7, round_id=3, client_id="client_002"
        )
        ratio = flattened_l2_norm(perturbation) / delta_norm
        assert ratio == pytest.approx(1.0, abs=1e-4)


def test_directional_perturbation_realized_cosine_matches_target() -> None:
    """cos(delta_attack, delta) matches the configured target_cosine within 1e-3, across a grid."""
    delta = _tensors(2.0, 5.0, shape=(64,))
    for target_cosine in (-1.0, -0.7, -0.3, 0.0, 0.3, 0.7, 1.0):
        perturbation = generate_directional_attack_perturbation(
            delta, target_cosine=target_cosine, seed=11, round_id=5, client_id="client_002"
        )
        realized = compute_cosine_similarity(perturbation, delta, TRAINABLE_KEYS)
        assert realized == pytest.approx(target_cosine, abs=1e-3)


def test_directional_perturbation_target_cosine_one_reproduces_honest_update() -> None:
    """target_cosine=1.0 returns delta_attack exactly equal to delta (no RNG draw needed)."""
    delta = _tensors(1.5, -2.5, shape=(12,))
    perturbation = generate_directional_attack_perturbation(
        delta, target_cosine=1.0, seed=42, round_id=20, client_id="client_002"
    )
    for key in TRAINABLE_KEYS:
        assert torch.allclose(perturbation[key], delta[key], atol=1e-5)


def test_directional_perturbation_target_cosine_zero_is_orthogonal() -> None:
    delta = _tensors(4.0, 1.0, shape=(40,))
    perturbation = generate_directional_attack_perturbation(
        delta, target_cosine=0.0, seed=42, round_id=20, client_id="client_002"
    )
    realized = compute_cosine_similarity(perturbation, delta, TRAINABLE_KEYS)
    assert realized == pytest.approx(0.0, abs=1e-3)


def test_directional_perturbation_target_cosine_negative_one_is_exact_reversal() -> None:
    """target_cosine=-1.0 returns delta_attack exactly equal to -delta (no RNG draw needed)."""
    delta = _tensors(2.0, 3.0, shape=(10,))
    perturbation = generate_directional_attack_perturbation(
        delta, target_cosine=-1.0, seed=42, round_id=20, client_id="client_002"
    )
    for key in TRAINABLE_KEYS:
        assert torch.allclose(perturbation[key], -delta[key], atol=1e-5)


def test_directional_perturbation_is_deterministic() -> None:
    delta = _tensors(1.0, 2.0, shape=(30,))
    first = generate_directional_attack_perturbation(
        delta, target_cosine=0.4, seed=42, round_id=20, client_id="client_002"
    )
    second = generate_directional_attack_perturbation(
        delta, target_cosine=0.4, seed=42, round_id=20, client_id="client_002"
    )
    for key in TRAINABLE_KEYS:
        assert torch.equal(first[key], second[key])


def test_directional_perturbation_differs_across_rounds_and_seeds() -> None:
    delta = _tensors(1.0, 2.0, shape=(30,))
    base = generate_directional_attack_perturbation(
        delta, target_cosine=0.4, seed=42, round_id=20, client_id="client_002"
    )
    different_round = generate_directional_attack_perturbation(
        delta, target_cosine=0.4, seed=42, round_id=21, client_id="client_002"
    )
    different_seed = generate_directional_attack_perturbation(
        delta, target_cosine=0.4, seed=43, round_id=20, client_id="client_002"
    )
    assert any(not torch.equal(base[k], different_round[k]) for k in TRAINABLE_KEYS)
    assert any(not torch.equal(base[k], different_seed[k]) for k in TRAINABLE_KEYS)


def test_directional_perturbation_does_not_touch_global_rng_state() -> None:
    """Generating a directional perturbation leaves torch's global RNG state unchanged."""
    delta = _tensors(1.0, 2.0, shape=(30,))
    torch.manual_seed(123)
    state_before = torch.get_rng_state()
    generate_directional_attack_perturbation(
        delta, target_cosine=0.4, seed=42, round_id=20, client_id="client_002"
    )
    state_after = torch.get_rng_state()
    assert torch.equal(state_before, state_after)

    # A draw made immediately afterward must be identical to a draw made with no
    # intervening directional-attack call -- proof the global generator was never advanced.
    torch.manual_seed(123)
    draw_without_attack_call = torch.randn(4)
    torch.manual_seed(123)
    generate_directional_attack_perturbation(
        delta, target_cosine=0.4, seed=42, round_id=20, client_id="client_002"
    )
    draw_after_attack_call = torch.randn(4)
    assert torch.equal(draw_without_attack_call, draw_after_attack_call)


# ----------------------------------------------------------------------
# apply_norm_preserving_directional_attack: trainable-keys-only scope
# ----------------------------------------------------------------------


def test_apply_directional_attack_perturbs_only_trainable_keys() -> None:
    honest_state = {
        "layer.weight": torch.full((4,), 2.0),
        "layer.bias": torch.full((4,), 1.0),
        "bn.running_mean": torch.full((4,), 0.5),
        "bn.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
    }
    global_state = {
        "layer.weight": torch.zeros(4),
        "layer.bias": torch.zeros(4),
        "bn.running_mean": torch.full((4,), 0.5),
        "bn.num_batches_tracked": torch.tensor(6, dtype=torch.int64),
    }
    perturbed = apply_norm_preserving_directional_attack(
        honest_state, global_state, TRAINABLE_KEYS,
        target_cosine=0.0, seed=42, round_id=20, client_id="client_002",
    )
    assert not torch.allclose(perturbed["layer.weight"], honest_state["layer.weight"])
    assert torch.equal(perturbed["bn.running_mean"], honest_state["bn.running_mean"])
    assert torch.equal(perturbed["bn.num_batches_tracked"], honest_state["bn.num_batches_tracked"])


def test_apply_directional_attack_norm_preserved_end_to_end() -> None:
    honest_state = _tensors(3.0, 4.0, shape=(20,))
    global_state = {key: torch.zeros(20) for key in TRAINABLE_KEYS}
    honest_delta_norm = flattened_l2_norm(
        {key: honest_state[key] - global_state[key] for key in TRAINABLE_KEYS}
    )
    perturbed = apply_norm_preserving_directional_attack(
        honest_state, global_state, TRAINABLE_KEYS,
        target_cosine=0.5, seed=5, round_id=20, client_id="client_002",
    )
    submitted_delta = {key: perturbed[key] - global_state[key] for key in TRAINABLE_KEYS}
    assert flattened_l2_norm(submitted_delta) == pytest.approx(honest_delta_norm, rel=1e-4)


# ----------------------------------------------------------------------
# compute_cosine_similarity / compute_norm_ratio
# ----------------------------------------------------------------------


def test_compute_cosine_similarity_matches_hand_computation() -> None:
    a = {"a": torch.tensor([1.0, 0.0])}
    b = {"a": torch.tensor([0.0, 1.0])}
    assert compute_cosine_similarity(a, b, ["a"]) == pytest.approx(0.0, abs=1e-6)


def test_compute_cosine_similarity_zero_vector_returns_none() -> None:
    a = {"a": torch.tensor([0.0, 0.0])}
    b = {"a": torch.tensor([1.0, 0.0])}
    assert compute_cosine_similarity(a, b, ["a"]) is None


def test_compute_norm_ratio_matches_hand_computation() -> None:
    submitted = {"a": torch.tensor([3.0, 4.0])}  # norm 5
    honest = {"a": torch.tensor([0.0, 10.0])}  # norm 10
    assert compute_norm_ratio(submitted, honest) == pytest.approx(0.5)


def test_compute_norm_ratio_zero_honest_delta_returns_none() -> None:
    submitted = {"a": torch.tensor([1.0, 0.0])}
    honest = {"a": torch.tensor([0.0, 0.0])}
    assert compute_norm_ratio(submitted, honest) is None


# ----------------------------------------------------------------------
# build_trace_row: EXP_015 verification and temporal fields
# ----------------------------------------------------------------------


def test_build_trace_row_directional_fields_only_populated_on_active_directional_rows() -> None:
    config = AttackConfig(
        attack_type="norm_preserving_directional",
        malicious_client_id="client_002",
        attack_start_round=20,
        directional_target_cosine=0.0,
    )
    honest_delta = {"a": torch.tensor([1.0, 0.0])}
    submitted_delta = {"a": torch.tensor([0.0, 1.0])}  # orthogonal, same norm

    active_row = build_trace_row(
        round_id=25, client_id="client_002", selected=True, successful=True,
        attack_config=config, client_delta=submitted_delta, peer_deltas=[],
        trainable_keys=["a"], honest_delta=honest_delta,
    )
    assert active_row["directional_target_cosine"] == pytest.approx(0.0)
    assert active_row["delta_attack_norm_ratio"] == pytest.approx(1.0)
    assert active_row["realized_cosine_to_honest"] == pytest.approx(0.0, abs=1e-6)

    # Pre-attack row (round 5 < attack_start_round=20): inert, even though
    # honest_delta/client_delta are technically provided by the caller.
    inactive_row = build_trace_row(
        round_id=5, client_id="client_002", selected=True, successful=True,
        attack_config=config, client_delta=submitted_delta, peer_deltas=[],
        trainable_keys=["a"], honest_delta=honest_delta,
    )
    assert inactive_row["delta_attack_norm_ratio"] is None
    assert inactive_row["realized_cosine_to_honest"] is None

    # Honest client on an active round: also inert (not the malicious client).
    honest_row = build_trace_row(
        round_id=25, client_id="client_000", selected=True, successful=True,
        attack_config=config, client_delta=submitted_delta, peer_deltas=[],
        trainable_keys=["a"], honest_delta=honest_delta,
    )
    assert honest_row["delta_attack_norm_ratio"] is None
    assert honest_row["realized_cosine_to_honest"] is None


def test_build_trace_row_directional_target_cosine_logged_regardless_of_activity() -> None:
    """directional_target_cosine (the configured, constant value) is logged on every row."""
    config = AttackConfig(
        attack_type="norm_preserving_directional",
        malicious_client_id="client_002",
        attack_start_round=20,
        directional_target_cosine=0.5,
    )
    row = build_trace_row(
        round_id=0, client_id="client_000", selected=True, successful=True,
        attack_config=config, client_delta={"a": torch.tensor([1.0])}, peer_deltas=[],
        trainable_keys=["a"],
    )
    assert row["directional_target_cosine"] == pytest.approx(0.5)


def test_build_trace_row_self_prev_round_cosine_none_without_history() -> None:
    config = AttackConfig()
    row = build_trace_row(
        round_id=0, client_id="client_000", selected=True, successful=True,
        attack_config=config, client_delta={"a": torch.tensor([1.0, 0.0])}, peer_deltas=[],
        trainable_keys=["a"], self_prev_delta=None,
    )
    assert row["self_prev_round_cosine"] is None


def test_build_trace_row_self_prev_round_cosine_correct_with_history() -> None:
    config = AttackConfig()
    current = {"a": torch.tensor([1.0, 0.0])}
    previous = {"a": torch.tensor([0.0, 1.0])}
    row = build_trace_row(
        round_id=1, client_id="client_000", selected=True, successful=True,
        attack_config=config, client_delta=current, peer_deltas=[],
        trainable_keys=["a"], self_prev_delta=previous,
    )
    assert row["self_prev_round_cosine"] == pytest.approx(0.0, abs=1e-6)


def test_build_trace_row_self_prev_round_cosine_available_for_honest_clients_too() -> None:
    """self_prev_round_cosine is a general diagnostic, not restricted to the malicious client."""
    config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_002",
        attack_start_round=20,
        attack_strength_rho=1.0,
    )
    current = {"a": torch.tensor([2.0, 0.0])}
    previous = {"a": torch.tensor([1.0, 0.0])}
    row = build_trace_row(
        round_id=25, client_id="client_000", selected=True, successful=True,
        attack_config=config, client_delta=current, peer_deltas=[],
        trainable_keys=["a"], self_prev_delta=previous,
    )
    assert row["self_prev_round_cosine"] == pytest.approx(1.0, abs=1e-6)


def test_traceability_module_has_no_dataset_or_test_set_dependency() -> None:
    """The Phase 5 attack/trace module never references dataset/test-set concepts.

    Every test above already demonstrates this functionally (no dataset,
    dataloader, or test-set object is ever constructed in this file); this
    statically confirms it holds for the module's own source as well.
    """
    source = inspect.getsource(traceability)
    for forbidden in ("cifar100", "test_dataset", "test_loader", "Dataset", "DataLoader"):
        assert forbidden not in source
