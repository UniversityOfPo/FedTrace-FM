"""Lightweight tests for NDSS E3's intermittent attack-window scheduling.

Covers only the new ``attack_active_windows`` field on
``src.attacks.traceability.AttackConfig`` and its mirror in
``src.experiments.run_federated_cifar100.FederatedExperimentConfig`` /
``load_federated_experiment_config``. Uses no dataset, dataloader, GPU, or
training loop -- pure dataclass/validation-logic tests, matching the style
and scope of ``tests/test_attacks_traceability.py``. Not a broad
integration test: nothing here trains a model or launches a federated run.
"""

from __future__ import annotations

import pytest

from src.attacks.traceability import AttackConfig


def _gaussian_config(**overrides) -> dict:
    base = dict(
        attack_type="gaussian_update_noise",
        malicious_client_id="client_002",
        attack_start_round=20,
        attack_strength_rho=1.0,
    )
    base.update(overrides)
    return base


# ----------------------------------------------------------------------
# A. Legacy continuous behavior is unchanged when attack_active_windows is None
# ----------------------------------------------------------------------


def test_legacy_continuous_behavior_unaffected_by_new_field() -> None:
    """attack_active_windows defaults to None; is_active must match the
    original round_id >= attack_start_round rule exactly, for every round."""
    config = AttackConfig(**_gaussian_config())
    assert config.attack_active_windows is None
    for round_id in range(50):
        expected = round_id >= 20
        assert config.is_active(round_id, "client_002") is expected


def test_legacy_continuous_behavior_explicit_none_is_same_as_omitted() -> None:
    """Explicitly passing attack_active_windows=None must be indistinguishable
    from omitting the argument entirely (the dataclass default)."""
    implicit = AttackConfig(**_gaussian_config())
    explicit = AttackConfig(**_gaussian_config(), attack_active_windows=None)
    for round_id in range(50):
        assert implicit.is_active(round_id, "client_002") == explicit.is_active(round_id, "client_002")


# ----------------------------------------------------------------------
# B. Exact intermittent active-round set
# ----------------------------------------------------------------------


def test_intermittent_active_round_set_matches_exactly() -> None:
    windows = ((20, 24), (30, 34), (40, 44))
    config = AttackConfig(**_gaussian_config(attack_active_windows=windows))
    expected_active = set(range(20, 25)) | set(range(30, 35)) | set(range(40, 45))
    actual_active = {r for r in range(50) if config.is_active(r, "client_002")}
    assert actual_active == expected_active


def test_intermittent_gap_rounds_are_inactive() -> None:
    """Rounds strictly between windows -- including rounds >= attack_start_round
    that fall in a gap -- must be inactive, unlike the continuous rule."""
    windows = ((20, 24), (30, 34), (40, 44))
    config = AttackConfig(**_gaussian_config(attack_active_windows=windows))
    for round_id in list(range(25, 30)) + list(range(35, 40)) + list(range(45, 50)):
        assert config.is_active(round_id, "client_002") is False, f"round {round_id} must be inactive"


def test_intermittent_window_boundaries_are_inclusive() -> None:
    """No off-by-one: both endpoints of every window are active; the round
    immediately before/after a window is not."""
    windows = ((20, 24), (30, 34), (40, 44))
    config = AttackConfig(**_gaussian_config(attack_active_windows=windows))
    for start, end in windows:
        assert config.is_active(start, "client_002") is True
        assert config.is_active(end, "client_002") is True
        assert config.is_active(start - 1, "client_002") is False
        assert config.is_active(end + 1, "client_002") is False


# ----------------------------------------------------------------------
# C. Honest client never active, with or without windows
# ----------------------------------------------------------------------


def test_honest_client_never_active_with_windows() -> None:
    windows = ((20, 24), (30, 34), (40, 44))
    config = AttackConfig(**_gaussian_config(attack_active_windows=windows))
    for round_id in range(50):
        for other_client in ("client_000", "client_001", "client_003", "client_004"):
            assert config.is_active(round_id, other_client) is False


# ----------------------------------------------------------------------
# D. Invalid overlapping windows rejected
# ----------------------------------------------------------------------


def test_overlapping_windows_rejected() -> None:
    with pytest.raises(ValueError, match="sorted and non-overlapping"):
        AttackConfig(**_gaussian_config(attack_active_windows=((20, 25), (24, 30))))


def test_touching_windows_rejected_as_overlapping() -> None:
    """A window starting exactly where the previous one ends counts as
    overlapping (windows must be strictly separated by at least one gap
    round), per the same rule that rejects true overlaps."""
    with pytest.raises(ValueError, match="sorted and non-overlapping"):
        AttackConfig(**_gaussian_config(attack_active_windows=((20, 24), (24, 30))))


# ----------------------------------------------------------------------
# E. Invalid unsorted windows rejected
# ----------------------------------------------------------------------


def test_unsorted_windows_rejected() -> None:
    with pytest.raises(ValueError, match="sorted and non-overlapping"):
        AttackConfig(**_gaussian_config(attack_active_windows=((30, 34), (20, 24))))


def test_invalid_window_start_after_end_rejected() -> None:
    with pytest.raises(ValueError, match="require 0 <= start <= end"):
        AttackConfig(**_gaussian_config(attack_active_windows=((24, 20),)))


def test_empty_windows_tuple_rejected() -> None:
    """An empty tuple is rejected outright -- None (not an empty
    collection) is the only way to disable intermittent scheduling."""
    with pytest.raises(ValueError, match="non-empty"):
        AttackConfig(**_gaussian_config(attack_active_windows=()))


# ----------------------------------------------------------------------
# F. First window start must equal attack_start_round
# ----------------------------------------------------------------------


def test_first_window_start_must_equal_attack_start_round() -> None:
    with pytest.raises(ValueError, match="attack_active_windows\\[0\\]\\[0\\] must equal attack_start_round"):
        AttackConfig(**_gaussian_config(attack_start_round=20, attack_active_windows=((21, 24), (30, 34))))


def test_first_window_start_equal_to_attack_start_round_is_accepted() -> None:
    config = AttackConfig(**_gaussian_config(attack_start_round=20, attack_active_windows=((20, 24), (30, 34))))
    assert config.attack_active_windows == ((20, 24), (30, 34))


# ----------------------------------------------------------------------
# Additional: attack_active_windows must not be set for attack_type="none"
# (directly relevant to the same validation block; not a broad/integration test)
# ----------------------------------------------------------------------


def test_windows_rejected_when_attack_type_none() -> None:
    with pytest.raises(ValueError, match="attack_type == 'none'"):
        AttackConfig(attack_type="none", attack_active_windows=((0, 1),))
