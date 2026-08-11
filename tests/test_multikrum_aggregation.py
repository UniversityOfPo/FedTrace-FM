"""Tests for the Multi-Krum external Byzantine-robust aggregation baseline.

Mirrors ``tests/test_aggregation.py``'s style: tiny, hand-computable
synthetic client updates, no CIFAR-100 data, no training, no
communication-round orchestration. Every hand-computed value below is
derived from a single shared 5-client, 1-D toy example (see
``_FIVE_CLIENT_W_VALUES`` and the module-level comment deriving its
expected Krum scores) so the arithmetic can be checked by hand against
``research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md``
Section 2's exact score formula.

Toy example (n=5, f=1, neighbors_used=n-f-2=2, m=n-f=4), trainable key
"w", global_state["w"]=0.0 so each client's delta equals its own "w" value:

    client_a=0.0  client_b=1.0  client_c=2.0  client_d=3.0  client_e=100.0

Pairwise squared distances:
    a-b=1   a-c=4   a-d=9    a-e=10000
    b-c=1   b-d=4   b-e=9801
    c-d=1   c-e=9604
    d-e=9409

Krum scores (sum of each client's 2 smallest peer distances):
    a: 1+4=5        b: 1+1=2        c: 1+1=2        d: 1+4=5        e: 9409+9604=19013

Ranking (ascending score, ties broken by client_id): b(1), c(2), a(3),
d(4), e(5). Selected (m=4, lowest-scoring): {client_a, client_b,
client_c, client_d}. Rejected: {client_e} -- the true outlier.
"""

from __future__ import annotations

import torch
import pytest

from src.federated.aggregation.fedavg import FedAvg
from src.federated.aggregation.multikrum import MultiKrum
from src.federated.contracts import ClientUpdate

_FIVE_CLIENT_W_VALUES: dict[str, float] = {
    "client_a": 0.0,
    "client_b": 1.0,
    "client_c": 2.0,
    "client_d": 3.0,
    "client_e": 100.0,
}
_EXPECTED_SCORES: dict[str, float] = {
    "client_a": 5.0,
    "client_b": 2.0,
    "client_c": 2.0,
    "client_d": 5.0,
    "client_e": 19013.0,
}
_EXPECTED_RANKS: dict[str, int] = {
    "client_b": 1,
    "client_c": 2,
    "client_a": 3,
    "client_d": 4,
    "client_e": 5,
}
_EXPECTED_SELECTED = ("client_a", "client_b", "client_c", "client_d")
_EXPECTED_REJECTED = ("client_e",)


def _make_global_state(extra: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {"w": torch.tensor([0.0], dtype=torch.float32)}
    if extra:
        state.update(extra)
    return state


def _make_update(
    client_id: str,
    round_id: int,
    w_value: float,
    num_samples: int = 10,
    extra_state: dict[str, torch.Tensor] | None = None,
) -> ClientUpdate:
    state: dict[str, torch.Tensor] = {"w": torch.tensor([w_value], dtype=torch.float32)}
    if extra_state:
        state.update(extra_state)
    return ClientUpdate.create(
        client_id=client_id, round_id=round_id, model_state=state, num_samples=num_samples
    )


def _make_five_client_updates(round_id: int = 1, num_samples: dict[str, int] | None = None) -> list[ClientUpdate]:
    num_samples = num_samples or {cid: 10 for cid in _FIVE_CLIENT_W_VALUES}
    return [
        _make_update(client_id, round_id, w_value, num_samples=num_samples[client_id])
        for client_id, w_value in _FIVE_CLIENT_W_VALUES.items()
    ]


# ----------------------------------------------------------------------
# 1. Pairwise squared Euclidean distance correctness
# ----------------------------------------------------------------------


def test_pairwise_squared_distance_correctness() -> None:
    """`_pairwise_squared_distances` matches hand-computed values exactly."""
    vectors = {
        client_id: torch.tensor([value], dtype=torch.float64)
        for client_id, value in _FIVE_CLIENT_W_VALUES.items()
    }
    distances = MultiKrum._pairwise_squared_distances(vectors)

    assert distances["client_a"]["client_b"] == pytest.approx(1.0)
    assert distances["client_a"]["client_c"] == pytest.approx(4.0)
    assert distances["client_a"]["client_d"] == pytest.approx(9.0)
    assert distances["client_a"]["client_e"] == pytest.approx(10000.0)
    assert distances["client_b"]["client_c"] == pytest.approx(1.0)
    assert distances["client_c"]["client_d"] == pytest.approx(1.0)
    assert distances["client_d"]["client_e"] == pytest.approx(9409.0)
    # Symmetric, and no self-entry.
    assert distances["client_b"]["client_a"] == pytest.approx(distances["client_a"]["client_b"])
    assert "client_a" not in distances["client_a"]


# ----------------------------------------------------------------------
# 2. Krum-score correctness
# ----------------------------------------------------------------------


def test_krum_score_hand_computable() -> None:
    """`_krum_scores` with neighbors_used=2 matches the hand-derived scores exactly."""
    vectors = {
        client_id: torch.tensor([value], dtype=torch.float64)
        for client_id, value in _FIVE_CLIENT_W_VALUES.items()
    }
    distances = MultiKrum._pairwise_squared_distances(vectors)
    scores = MultiKrum._krum_scores(distances, neighbors_used=2)

    for client_id, expected in _EXPECTED_SCORES.items():
        assert scores[client_id] == pytest.approx(expected)


# ----------------------------------------------------------------------
# 3-4. n/f/m/neighbors_used derivation
# ----------------------------------------------------------------------


def test_n5_f1_neighbors_used_is_2() -> None:
    """At n=5, f=1: neighbors_used = n-f-2 = 2."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    result = strategy.aggregate(_make_global_state(), _make_five_client_updates())
    assert result.metadata["neighbors_used"] == 2


def test_m_equals_n_minus_f() -> None:
    """At n=5, f=1: m = n-f = 4."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    result = strategy.aggregate(_make_global_state(), _make_five_client_updates())
    assert result.metadata["m"] == 4
    assert result.metadata["f"] == 1
    assert result.metadata["num_clients"] == 5


# ----------------------------------------------------------------------
# 5-6. Feasibility validation
# ----------------------------------------------------------------------


def test_feasibility_boundary_n5_f1_valid() -> None:
    """n=5, f=1 sits exactly at n=2f+3 and is accepted, not rejected."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    result = strategy.aggregate(_make_global_state(), _make_five_client_updates())
    assert result.strategy_name == "multikrum"


def test_infeasible_n_less_than_2f_plus_3_raises() -> None:
    """n=4, f=1 violates n >= 2f+3 = 5 and raises, with no fallback to another strategy."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    updates = _make_five_client_updates()[:4]  # drop one client -> n=4
    with pytest.raises(ValueError):
        strategy.aggregate(_make_global_state(), updates)


def test_negative_f_rejected() -> None:
    """A negative f is rejected at construction time."""
    with pytest.raises(ValueError):
        MultiKrum(trainable_keys=frozenset({"w"}), f=-1)


def test_empty_trainable_keys_rejected() -> None:
    """An empty trainable_keys set is rejected at construction time."""
    with pytest.raises(ValueError):
        MultiKrum(trainable_keys=frozenset(), f=1)


# ----------------------------------------------------------------------
# 8-10. Deterministic tie-breaking, outlier rejection, expected selection
# ----------------------------------------------------------------------


def test_deterministic_tie_breaking() -> None:
    """Score ties (client_b/client_c at 2.0, client_a/client_d at 5.0) resolve by client_id."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    result = strategy.aggregate(_make_global_state(), _make_five_client_updates())

    diagnostics = result.metadata["multikrum_diagnostics"]
    for client_id, expected_rank in _EXPECTED_RANKS.items():
        assert diagnostics[client_id]["krum_rank"] == expected_rank
    # client_b < client_c lexically, and both tie at score 2.0.
    assert diagnostics["client_b"]["krum_rank"] < diagnostics["client_c"]["krum_rank"]
    # client_a < client_d lexically, and both tie at score 5.0.
    assert diagnostics["client_a"]["krum_rank"] < diagnostics["client_d"]["krum_rank"]


def test_outlier_receives_worst_score() -> None:
    """The obviously-outlying client_e (w=100.0 vs. 0..3 for everyone else) gets the worst rank."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    result = strategy.aggregate(_make_global_state(), _make_five_client_updates())

    diagnostics = result.metadata["multikrum_diagnostics"]
    assert diagnostics["client_e"]["krum_rank"] == 5
    assert diagnostics["client_e"]["krum_score"] == pytest.approx(19013.0)
    assert diagnostics["client_e"]["selected"] is False


def test_expected_clients_selected_in_toy_example() -> None:
    """The 4 near-clustered clients are selected; the outlier is rejected."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    result = strategy.aggregate(_make_global_state(), _make_five_client_updates())

    assert result.metadata["selected_client_ids"] == _EXPECTED_SELECTED
    assert result.metadata["rejected_client_ids"] == _EXPECTED_REJECTED
    assert set(result.participating_client_ids) == set(_EXPECTED_SELECTED)


# ----------------------------------------------------------------------
# 11-12. Unweighted averaging; sample counts have no influence
# ----------------------------------------------------------------------


def test_selected_trainable_parameters_averaged_unweighted() -> None:
    """Aggregated 'w' equals the plain (unweighted) mean of the 4 selected clients: (0+1+2+3)/4=1.5."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    result = strategy.aggregate(_make_global_state(), _make_five_client_updates())

    assert torch.allclose(result.aggregated_state["w"], torch.tensor([1.5]))


def test_sample_counts_do_not_influence_multikrum_output() -> None:
    """Wildly unequal num_samples across clients do not change the aggregated result at all."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    skewed_samples = {
        "client_a": 1,
        "client_b": 1_000_000,
        "client_c": 1,
        "client_d": 1,
        "client_e": 1,
    }
    updates = _make_five_client_updates(num_samples=skewed_samples)
    result = strategy.aggregate(_make_global_state(), updates)

    # Still exactly the unweighted mean of the selected 4 -- unaffected by
    # client_b's enormous sample count, unlike FedAvg.
    assert torch.allclose(result.aggregated_state["w"], torch.tensor([1.5]))
    assert result.total_weight == pytest.approx(4.0)


# ----------------------------------------------------------------------
# 13-14. Buffer exclusion from distance; buffer averaging over selected clients
# ----------------------------------------------------------------------

# A second toy example, designed so that including a non-trainable
# floating buffer in the distance computation (which would be a bug)
# would change the selection outcome relative to the correct,
# trainable-parameter-only computation:
#   - "w" (trainable): client_a=100.0 (true outlier), b=c=d=e=0.0 (honest cluster)
#   - "bn_stat" (non-trainable floating buffer): a=500.0, b=c=d=1.0, e=999999.0
# Correct (buffer-excluded) behavior rejects client_a (the trainable
# outlier) and selects {b, c, d, e} despite e's huge buffer value.

_BUFFER_W_VALUES: dict[str, float] = {
    "client_a": 100.0,
    "client_b": 0.0,
    "client_c": 0.0,
    "client_d": 0.0,
    "client_e": 0.0,
}
_BUFFER_BN_VALUES: dict[str, float] = {
    "client_a": 500.0,
    "client_b": 1.0,
    "client_c": 1.0,
    "client_d": 1.0,
    "client_e": 999999.0,
}


def _make_buffer_updates() -> list[ClientUpdate]:
    return [
        _make_update(
            client_id,
            round_id=1,
            w_value=_BUFFER_W_VALUES[client_id],
            extra_state={"bn_stat": torch.tensor([_BUFFER_BN_VALUES[client_id]], dtype=torch.float32)},
        )
        for client_id in _BUFFER_W_VALUES
    ]


def test_non_trainable_floating_buffer_excluded_from_distance() -> None:
    """Selection is driven only by 'w' (the trainable key); the huge 'bn_stat' outlier on
    client_e does not change who gets rejected -- the true 'w' outlier (client_a) is rejected."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    global_state = _make_global_state(extra={"bn_stat": torch.tensor([0.0], dtype=torch.float32)})
    result = strategy.aggregate(global_state, _make_buffer_updates())

    assert result.metadata["rejected_client_ids"] == ("client_a",)
    assert result.metadata["selected_client_ids"] == ("client_b", "client_c", "client_d", "client_e")


def test_selected_floating_buffers_averaged_correctly() -> None:
    """'bn_stat' is unweighted-averaged over the SAME selected clients as 'w', excluding
    the rejected client_a's buffer value entirely: (1+1+1+999999)/4 = 250000.5."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    global_state = _make_global_state(extra={"bn_stat": torch.tensor([0.0], dtype=torch.float32)})
    result = strategy.aggregate(global_state, _make_buffer_updates())

    assert torch.allclose(result.aggregated_state["bn_stat"], torch.tensor([250000.5]))


# ----------------------------------------------------------------------
# 15. Non-floating buffer handling matches FedAvg's convention exactly
# ----------------------------------------------------------------------


def test_non_floating_num_batches_tracked_uses_max_across_all_clients() -> None:
    """A '.num_batches_tracked' buffer is aggregated as the max across ALL n clients
    (not just the m selected), even when the max comes from a rejected client."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    nbt_values = {"client_a": 5, "client_b": 6, "client_c": 7, "client_d": 8, "client_e": 999}
    global_state = _make_global_state(
        extra={"bn.num_batches_tracked": torch.tensor(0, dtype=torch.int64)}
    )
    updates = [
        _make_update(
            client_id,
            round_id=1,
            w_value=value,
            extra_state={"bn.num_batches_tracked": torch.tensor(nbt_values[client_id], dtype=torch.int64)},
        )
        for client_id, value in _FIVE_CLIENT_W_VALUES.items()
    ]
    result = strategy.aggregate(global_state, updates)

    # client_e (rejected: the trainable outlier) still contributes the max.
    assert result.metadata["rejected_client_ids"] == ("client_e",)
    assert torch.equal(
        result.aggregated_state["bn.num_batches_tracked"], torch.tensor(999, dtype=torch.int64)
    )


def test_non_floating_other_buffer_requires_identical_values() -> None:
    """A plain non-floating buffer (not '.num_batches_tracked') requires exact agreement,
    exactly like FedAvg's `_require_identical`; disagreement raises rather than averaging."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    global_state = _make_global_state(extra={"counter": torch.tensor(0, dtype=torch.int64)})

    identical_updates = [
        _make_update(
            client_id, round_id=1, w_value=value, extra_state={"counter": torch.tensor(7, dtype=torch.int64)}
        )
        for client_id, value in _FIVE_CLIENT_W_VALUES.items()
    ]
    result = strategy.aggregate(global_state, identical_updates)
    assert torch.equal(result.aggregated_state["counter"], torch.tensor(7, dtype=torch.int64))

    disagreeing_updates = [
        _make_update(
            client_id,
            round_id=1,
            w_value=value,
            extra_state={"counter": torch.tensor(7 if client_id != "client_b" else 9, dtype=torch.int64)},
        )
        for client_id, value in _FIVE_CLIENT_W_VALUES.items()
    ]
    with pytest.raises(ValueError):
        strategy.aggregate(global_state, disagreeing_updates)


# ----------------------------------------------------------------------
# 16-17. Structural validation reuse (missing key / shape mismatch)
# ----------------------------------------------------------------------


def test_missing_state_key_rejected() -> None:
    """An update whose model_state is missing a key present in global_state is rejected."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    global_state = _make_global_state(extra={"bn_stat": torch.tensor([0.0])})
    updates = _make_five_client_updates()  # none of these include "bn_stat"
    with pytest.raises(ValueError):
        strategy.aggregate(global_state, updates)


def test_shape_mismatch_rejected() -> None:
    """An update with the right keys but a wrong tensor shape for 'w' is rejected."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    global_state = _make_global_state()
    updates = _make_five_client_updates()
    bad_update = ClientUpdate.create(
        client_id="client_a", round_id=1, model_state={"w": torch.zeros(3)}, num_samples=10
    )
    updates = [bad_update] + updates[1:]
    with pytest.raises(ValueError):
        strategy.aggregate(global_state, updates)


# ----------------------------------------------------------------------
# 18-19. Determinism and diagnostics completeness
# ----------------------------------------------------------------------


def test_deterministic_repeated_aggregation() -> None:
    """Aggregating the identical inputs twice produces bit-identical results."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    global_state = _make_global_state()
    updates = _make_five_client_updates()

    result_1 = strategy.aggregate(global_state, updates)
    result_2 = strategy.aggregate(global_state, updates)

    assert torch.equal(result_1.aggregated_state["w"], result_2.aggregated_state["w"])
    assert result_1.metadata["selected_client_ids"] == result_2.metadata["selected_client_ids"]
    assert result_1.metadata["rejected_client_ids"] == result_2.metadata["rejected_client_ids"]
    assert (
        result_1.metadata["multikrum_diagnostics"] == result_2.metadata["multikrum_diagnostics"]
    )


def test_diagnostics_contain_required_fields() -> None:
    """Metadata exposes n/f/m/neighbors_used and per-client score/rank/selected diagnostics."""
    strategy = MultiKrum(trainable_keys=frozenset({"w"}), f=1)
    result = strategy.aggregate(_make_global_state(), _make_five_client_updates())

    for key in (
        "num_clients", "f", "m", "neighbors_used", "weighting",
        "selected_client_ids", "rejected_client_ids", "multikrum_diagnostics",
    ):
        assert key in result.metadata

    diagnostics = result.metadata["multikrum_diagnostics"]
    assert set(diagnostics.keys()) == set(_FIVE_CLIENT_W_VALUES.keys())
    for client_id, entry in diagnostics.items():
        assert set(entry.keys()) == {"krum_score", "krum_rank", "selected"}
        assert isinstance(entry["krum_score"], float)
        assert isinstance(entry["krum_rank"], int)
        assert isinstance(entry["selected"], bool)


# ----------------------------------------------------------------------
# 20. FedAvg behavior is unaffected by MultiKrum's addition
# ----------------------------------------------------------------------


def test_fedavg_behavior_unchanged_by_multikrum_addition() -> None:
    """Importing/using MultiKrum alongside FedAvg does not alter FedAvg's own sample-weighted math."""
    global_state = {
        "linear.weight": torch.zeros(2, 2),
        "linear.bias": torch.zeros(2),
    }
    update_a = ClientUpdate.create(
        client_id="client_a", round_id=1, num_samples=10,
        model_state={"linear.weight": torch.full((2, 2), 2.0), "linear.bias": torch.zeros(2)},
    )
    update_b = ClientUpdate.create(
        client_id="client_b", round_id=1, num_samples=30,
        model_state={"linear.weight": torch.full((2, 2), 6.0), "linear.bias": torch.full((2,), 8.0)},
    )

    result = FedAvg().aggregate(global_state, [update_a, update_b])
    expected_weight = 0.25 * 2.0 + 0.75 * 6.0  # 5.0
    expected_bias = 0.25 * 0.0 + 0.75 * 8.0  # 6.0

    assert result.strategy_name == "fedavg"
    assert torch.allclose(result.aggregated_state["linear.weight"], torch.full((2, 2), expected_weight))
    assert torch.allclose(result.aggregated_state["linear.bias"], torch.full((2,), expected_bias))
