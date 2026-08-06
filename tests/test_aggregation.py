"""Lightweight tests for the pluggable aggregation interface and FedAvg (Sprint 3.3).

These use a tiny model and synthetic client updates so the suite runs
quickly on CPU. No CIFAR-100 data is loaded, no training is run, and no
communication rounds/client selection are exercised.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from torch import nn

from src.federated.aggregation.base import AggregationResult
from src.federated.aggregation.fedavg import FedAvg
from src.federated.contracts import ClientUpdate
from src.federated.server import FederatedServer, FederatedServerConfig


class _TinyModel(nn.Module):
    """A minimal model with a float parameter pair and a non-floating buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_global_state() -> dict[str, torch.Tensor]:
    """Build a reference global state dict (values are irrelevant to FedAvg)."""
    return _TinyModel().state_dict()


def _make_update(
    client_id: str,
    round_id: int,
    num_samples: int,
    weight_value: float,
    bias_value: float,
    counter_value: int = 5,
) -> ClientUpdate:
    """Build a valid ClientUpdate with constant-filled tensors for easy expected-value math."""
    return ClientUpdate.create(
        client_id=client_id,
        round_id=round_id,
        model_state={
            "linear.weight": torch.full((2, 2), weight_value, dtype=torch.float32),
            "linear.bias": torch.full((2,), bias_value, dtype=torch.float32),
            "counter": torch.tensor(counter_value, dtype=torch.int64),
        },
        num_samples=num_samples,
        train_metrics={"loss": 0.1},
    )


# ----------------------------------------------------------------------
# AggregationResult validation and immutability
# ----------------------------------------------------------------------


def test_aggregation_result_validation_and_immutability() -> None:
    """AggregationResult validates its fields and is immutable at every level."""
    tensor = torch.ones(2)
    result = AggregationResult(
        strategy_name="fedavg",
        round_id=0,
        aggregated_state={"w": tensor},
        participating_client_ids=("a", "b"),
        total_weight=10.0,
        metadata={"k": "v"},
    )
    assert result.aggregated_state["w"].device.type == "cpu"
    assert result.aggregated_state["w"].requires_grad is False

    with pytest.raises(TypeError):
        result.aggregated_state["w"] = torch.zeros(2)
    with pytest.raises(TypeError):
        result.metadata["k"] = "changed"
    with pytest.raises(FrozenInstanceError):
        result.round_id = 1

    base_kwargs = dict(
        strategy_name="fedavg",
        round_id=0,
        aggregated_state={"w": tensor},
        participating_client_ids=("a",),
        total_weight=1.0,
        metadata={},
    )
    with pytest.raises(ValueError):
        AggregationResult(**{**base_kwargs, "strategy_name": ""})
    with pytest.raises(ValueError):
        AggregationResult(**{**base_kwargs, "aggregated_state": {}})
    with pytest.raises(ValueError):
        AggregationResult(**{**base_kwargs, "participating_client_ids": ()})
    with pytest.raises(ValueError):
        AggregationResult(**{**base_kwargs, "participating_client_ids": ("a", "a")})
    with pytest.raises(ValueError):
        AggregationResult(**{**base_kwargs, "total_weight": 0.0})
    with pytest.raises(ValueError):
        AggregationResult(**{**base_kwargs, "total_weight": float("inf")})


# ----------------------------------------------------------------------
# FedAvg: basic aggregation math
# ----------------------------------------------------------------------


def test_empty_updates_rejected() -> None:
    """Aggregating zero updates raises."""
    with pytest.raises(ValueError):
        FedAvg().aggregate(_make_global_state(), [])


def test_single_client_aggregation_returns_client_state() -> None:
    """With one client, the aggregated state equals that client's own state."""
    global_state = _make_global_state()
    update = _make_update("client_a", round_id=1, num_samples=10, weight_value=3.5, bias_value=-1.0)

    result = FedAvg().aggregate(global_state, [update])

    assert result.strategy_name == "fedavg"
    assert result.round_id == 1
    assert result.participating_client_ids == ("client_a",)
    assert result.total_weight == 10.0
    assert torch.allclose(result.aggregated_state["linear.weight"], torch.full((2, 2), 3.5))
    assert torch.allclose(result.aggregated_state["linear.bias"], torch.full((2,), -1.0))
    assert torch.equal(result.aggregated_state["counter"], torch.tensor(5, dtype=torch.int64))


def test_two_client_equal_sample_fedavg() -> None:
    """Equal sample counts produce a plain (unweighted) average."""
    global_state = _make_global_state()
    update_a = _make_update("client_a", round_id=1, num_samples=10, weight_value=2.0, bias_value=0.0)
    update_b = _make_update("client_b", round_id=1, num_samples=10, weight_value=6.0, bias_value=4.0)

    result = FedAvg().aggregate(global_state, [update_a, update_b])

    assert result.total_weight == 20.0
    assert torch.allclose(result.aggregated_state["linear.weight"], torch.full((2, 2), 4.0))
    assert torch.allclose(result.aggregated_state["linear.bias"], torch.full((2,), 2.0))


def test_two_client_unequal_sample_weighted_fedavg() -> None:
    """Unequal sample counts weight the average toward the larger client."""
    global_state = _make_global_state()
    update_a = _make_update("client_a", round_id=1, num_samples=10, weight_value=2.0, bias_value=0.0)
    update_b = _make_update("client_b", round_id=1, num_samples=30, weight_value=6.0, bias_value=8.0)

    result = FedAvg().aggregate(global_state, [update_a, update_b])
    expected_weight = 0.25 * 2.0 + 0.75 * 6.0  # 5.0
    expected_bias = 0.25 * 0.0 + 0.75 * 8.0  # 6.0

    assert result.total_weight == 40.0
    assert torch.allclose(result.aggregated_state["linear.weight"], torch.full((2, 2), expected_weight))
    assert torch.allclose(result.aggregated_state["linear.bias"], torch.full((2,), expected_bias))


def test_three_client_weighted_fedavg() -> None:
    """Three clients with distinct sample counts aggregate to the correct weighted mean."""
    global_state = _make_global_state()
    updates = [
        _make_update("client_a", round_id=1, num_samples=10, weight_value=3.0, bias_value=0.0),
        _make_update("client_b", round_id=1, num_samples=20, weight_value=6.0, bias_value=0.0),
        _make_update("client_c", round_id=1, num_samples=30, weight_value=9.0, bias_value=0.0),
    ]

    result = FedAvg().aggregate(global_state, updates)
    expected = (10 * 3.0 + 20 * 6.0 + 30 * 9.0) / 60  # 7.0

    assert result.total_weight == 60.0
    assert torch.allclose(result.aggregated_state["linear.weight"], torch.full((2, 2), expected))


# ----------------------------------------------------------------------
# FedAvg: rejection paths
# ----------------------------------------------------------------------


def test_repeated_client_ids_rejected() -> None:
    """Two updates from the same client id are rejected."""
    global_state = _make_global_state()
    update_a1 = _make_update("client_a", round_id=1, num_samples=10, weight_value=1.0, bias_value=0.0)
    update_a2 = _make_update("client_a", round_id=1, num_samples=20, weight_value=2.0, bias_value=0.0)
    with pytest.raises(ValueError):
        FedAvg().aggregate(global_state, [update_a1, update_a2])


def test_mismatched_round_ids_rejected() -> None:
    """Updates that disagree on round_id are rejected."""
    global_state = _make_global_state()
    update_a = _make_update("client_a", round_id=1, num_samples=10, weight_value=1.0, bias_value=0.0)
    update_b = _make_update("client_b", round_id=2, num_samples=10, weight_value=1.0, bias_value=0.0)
    with pytest.raises(ValueError):
        FedAvg().aggregate(global_state, [update_a, update_b])


def test_mismatched_keys_rejected() -> None:
    """An update whose model_state keys don't match global_state is rejected."""
    global_state = _make_global_state()
    update = ClientUpdate.create(
        client_id="client_a",
        round_id=1,
        model_state={"linear.weight": torch.zeros(2, 2)},  # missing linear.bias, counter
        num_samples=10,
    )
    with pytest.raises(ValueError):
        FedAvg().aggregate(global_state, [update])


def test_mismatched_shapes_rejected() -> None:
    """An update with the right keys but a wrong tensor shape is rejected."""
    global_state = _make_global_state()
    update = ClientUpdate.create(
        client_id="client_a",
        round_id=1,
        model_state={
            "linear.weight": torch.zeros(3, 3),
            "linear.bias": torch.zeros(2),
            "counter": torch.tensor(0, dtype=torch.int64),
        },
        num_samples=10,
    )
    with pytest.raises(ValueError):
        FedAvg().aggregate(global_state, [update])


def test_incompatible_dtypes_rejected() -> None:
    """A non-floating tensor where the global model expects a floating tensor is rejected."""
    global_state = _make_global_state()
    update = ClientUpdate.create(
        client_id="client_a",
        round_id=1,
        model_state={
            "linear.weight": torch.zeros(2, 2, dtype=torch.int64),
            "linear.bias": torch.zeros(2),
            "counter": torch.tensor(0, dtype=torch.int64),
        },
        num_samples=10,
    )
    with pytest.raises(TypeError):
        FedAvg().aggregate(global_state, [update])


def test_integer_buffers_identical_values_accepted() -> None:
    """Non-floating tensors are passed through unchanged when all clients agree."""
    global_state = _make_global_state()
    update_a = _make_update("client_a", round_id=1, num_samples=10, weight_value=1.0, bias_value=0.0, counter_value=7)
    update_b = _make_update("client_b", round_id=1, num_samples=10, weight_value=1.0, bias_value=0.0, counter_value=7)

    result = FedAvg().aggregate(global_state, [update_a, update_b])
    assert torch.equal(result.aggregated_state["counter"], torch.tensor(7, dtype=torch.int64))


def test_integer_buffers_different_values_rejected() -> None:
    """Non-floating tensors that disagree across clients are rejected, not averaged."""
    global_state = _make_global_state()
    update_a = _make_update("client_a", round_id=1, num_samples=10, weight_value=1.0, bias_value=0.0, counter_value=7)
    update_b = _make_update("client_b", round_id=1, num_samples=10, weight_value=1.0, bias_value=0.0, counter_value=9)

    with pytest.raises(ValueError):
        FedAvg().aggregate(global_state, [update_a, update_b])


# ----------------------------------------------------------------------
# FedAvg: no mutation, safe outputs
# ----------------------------------------------------------------------


def test_global_state_not_mutated() -> None:
    """global_state is read-only from FedAvg's perspective."""
    global_state = _make_global_state()
    snapshot = {key: value.clone() for key, value in global_state.items()}

    update = _make_update("client_a", round_id=1, num_samples=10, weight_value=99.0, bias_value=99.0)
    FedAvg().aggregate(global_state, [update])

    for key, value in snapshot.items():
        assert torch.equal(global_state[key], value)


def test_client_update_inputs_not_mutated() -> None:
    """Aggregating does not modify any input ClientUpdate's tensors."""
    global_state = _make_global_state()
    update_a = _make_update("client_a", round_id=1, num_samples=10, weight_value=2.0, bias_value=0.0)
    update_b = _make_update("client_b", round_id=1, num_samples=30, weight_value=6.0, bias_value=8.0)
    snapshot_a = {key: value.clone() for key, value in update_a.model_state.items()}
    snapshot_b = {key: value.clone() for key, value in update_b.model_state.items()}

    FedAvg().aggregate(global_state, [update_a, update_b])

    for key, value in snapshot_a.items():
        assert torch.equal(update_a.model_state[key], value)
    for key, value in snapshot_b.items():
        assert torch.equal(update_b.model_state[key], value)


def test_output_tensors_are_cpu_cloned_detached() -> None:
    """Aggregated tensors are CPU, detached, and independent of the source update's tensors."""
    global_state = _make_global_state()
    update = _make_update("client_a", round_id=1, num_samples=10, weight_value=1.0, bias_value=0.0)

    result = FedAvg().aggregate(global_state, [update])

    for tensor in result.aggregated_state.values():
        assert tensor.device.type == "cpu"
        assert tensor.requires_grad is False

    result.aggregated_state["linear.weight"].add_(100.0)
    assert not torch.allclose(
        result.aggregated_state["linear.weight"], update.model_state["linear.weight"]
    )


def test_metadata_and_total_weight_correct() -> None:
    """Metadata reports the expected client count, weighting scheme, and sample total."""
    global_state = _make_global_state()
    update_a = _make_update("client_a", round_id=1, num_samples=10, weight_value=1.0, bias_value=0.0)
    update_b = _make_update("client_b", round_id=1, num_samples=30, weight_value=1.0, bias_value=0.0)

    result = FedAvg().aggregate(global_state, [update_a, update_b])

    assert result.metadata["num_clients"] == 2
    assert result.metadata["weighting"] == "num_samples"
    assert result.metadata["total_samples"] == 40
    assert result.total_weight == 40.0


# ----------------------------------------------------------------------
# Integration: composing with the existing (unmodified) FederatedServer
# ----------------------------------------------------------------------


def test_aggregation_result_applies_via_existing_load_global_weights(tmp_path: Path) -> None:
    """AggregationResult.aggregated_state composes with FederatedServer.load_global_weights
    as-is: computing an AggregationResult does not touch the server, and the global model
    only changes once that result is explicitly loaded. No new server method was needed."""
    config = FederatedServerConfig(
        server_id="server_0", seed=0, clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path
    )
    server = FederatedServer(config, model_fn=_TinyModel)
    before = server.get_global_weights()

    update_a = _make_update("client_a", round_id=1, num_samples=10, weight_value=2.0, bias_value=0.0)
    update_b = _make_update("client_b", round_id=1, num_samples=30, weight_value=6.0, bias_value=8.0)
    result = FedAvg().aggregate(server.get_global_weights(), [update_a, update_b])

    # Merely computing the result must not change the server's global model.
    unchanged = server.get_global_weights()
    for key in before:
        assert torch.equal(before[key], unchanged[key])

    server.load_global_weights(result.aggregated_state)

    after = server.get_global_weights()
    for key in result.aggregated_state:
        assert torch.equal(after[key], result.aggregated_state[key])
    assert not torch.equal(after["linear.weight"], before["linear.weight"])


# ----------------------------------------------------------------------
# FedAvg: num_batches_tracked handling for unequal client batch counts
# (Sprint 6.1 -- fixes the EXP_006 failure: "Non-floating tensor
# 'layer1.0.bn1.num_batches_tracked' differs across clients.")
# ----------------------------------------------------------------------


class _TinyModelWithBatchNorm(nn.Module):
    """A minimal model with a real BatchNorm1d layer.

    Its state_dict naturally includes a ``bn.num_batches_tracked`` buffer
    using PyTorch's own BatchNorm buffer-naming convention -- more
    faithful to the real EXP_006 failure than a hand-crafted key would be.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.linear(x))


def _make_bn_global_state() -> dict[str, torch.Tensor]:
    """Build a reference global state dict containing a real `.num_batches_tracked` buffer."""
    return _TinyModelWithBatchNorm().state_dict()


def _make_bn_update(
    client_id: str,
    round_id: int,
    num_samples: int,
    num_batches_tracked: int,
    weight_value: float = 1.0,
    bias_value: float = 0.0,
) -> ClientUpdate:
    """Build a valid ClientUpdate matching `_TinyModelWithBatchNorm`'s keys.

    ``num_batches_tracked`` is set independently of ``num_samples`` on
    purpose: it models a client that processed a different number of local
    batches than its sample count alone would suggest (e.g. a smaller
    Dirichlet partition, or a different local batch size), which is
    exactly the scenario the EXP_006 failure surfaced.
    """
    state = _TinyModelWithBatchNorm().state_dict()
    state["linear.weight"] = torch.full((2, 2), weight_value, dtype=torch.float32)
    state["linear.bias"] = torch.full((2,), bias_value, dtype=torch.float32)
    state["bn.num_batches_tracked"] = torch.tensor(num_batches_tracked, dtype=torch.int64)
    return ClientUpdate.create(
        client_id=client_id,
        round_id=round_id,
        model_state=state,
        num_samples=num_samples,
        train_metrics={"loss": 0.1},
    )


def test_differing_num_batches_tracked_aggregates_successfully() -> None:
    """Clients reporting different num_batches_tracked no longer raise ValueError."""
    global_state = _make_bn_global_state()
    update_a = _make_bn_update("client_a", round_id=1, num_samples=10, num_batches_tracked=12)
    update_b = _make_bn_update("client_b", round_id=1, num_samples=5, num_batches_tracked=27)

    result = FedAvg().aggregate(global_state, [update_a, update_b])

    assert "bn.num_batches_tracked" in result.aggregated_state


def test_num_batches_tracked_result_equals_maximum() -> None:
    """The aggregated num_batches_tracked equals the maximum value submitted, across 3 clients."""
    global_state = _make_bn_global_state()
    updates = [
        _make_bn_update("client_a", round_id=1, num_samples=10, num_batches_tracked=5),
        _make_bn_update("client_b", round_id=1, num_samples=10, num_batches_tracked=41),
        _make_bn_update("client_c", round_id=1, num_samples=10, num_batches_tracked=18),
    ]

    result = FedAvg().aggregate(global_state, updates)

    assert torch.equal(
        result.aggregated_state["bn.num_batches_tracked"], torch.tensor(41, dtype=torch.int64)
    )
    assert result.aggregated_state["bn.num_batches_tracked"].dtype == torch.int64


def test_equal_num_batches_tracked_still_works() -> None:
    """Equal num_batches_tracked values across clients still aggregate correctly (max of equals)."""
    global_state = _make_bn_global_state()
    update_a = _make_bn_update("client_a", round_id=1, num_samples=10, num_batches_tracked=9)
    update_b = _make_bn_update("client_b", round_id=1, num_samples=10, num_batches_tracked=9)

    result = FedAvg().aggregate(global_state, [update_a, update_b])

    assert torch.equal(
        result.aggregated_state["bn.num_batches_tracked"], torch.tensor(9, dtype=torch.int64)
    )


def test_key_merely_containing_num_batches_tracked_still_requires_equality() -> None:
    """A key that contains 'num_batches_tracked' as a substring, not a '.'-prefixed suffix,
    is NOT treated as a BatchNorm counter and still requires exact agreement."""
    global_state = {
        "linear.weight": torch.zeros(2, 2),
        "linear.bias": torch.zeros(2),
        "num_batches_tracked_legacy": torch.tensor(0, dtype=torch.int64),
    }

    def _update(client_id: str, value: int) -> ClientUpdate:
        return ClientUpdate.create(
            client_id=client_id,
            round_id=1,
            model_state={
                "linear.weight": torch.full((2, 2), 1.0),
                "linear.bias": torch.zeros(2),
                "num_batches_tracked_legacy": torch.tensor(value, dtype=torch.int64),
            },
            num_samples=10,
        )

    with pytest.raises(ValueError):
        FedAvg().aggregate(global_state, [_update("client_a", 3), _update("client_b", 9)])


def test_other_differing_integer_buffers_still_rejected() -> None:
    """A plain integer buffer unrelated to num_batches_tracked still requires exact agreement."""
    global_state = _make_global_state()
    update_a = _make_update(
        "client_a", round_id=1, num_samples=10, weight_value=1.0, bias_value=0.0, counter_value=7
    )
    update_b = _make_update(
        "client_b", round_id=1, num_samples=10, weight_value=1.0, bias_value=0.0, counter_value=9
    )

    with pytest.raises(ValueError):
        FedAvg().aggregate(global_state, [update_a, update_b])


def test_floating_tensor_weighted_averaging_unchanged_alongside_num_batches_tracked() -> None:
    """Sample-weighted averaging of floating tensors is unaffected by the num_batches_tracked change."""
    global_state = _make_bn_global_state()
    update_a = _make_bn_update(
        "client_a", round_id=1, num_samples=10, num_batches_tracked=4, weight_value=2.0, bias_value=0.0
    )
    update_b = _make_bn_update(
        "client_b", round_id=1, num_samples=30, num_batches_tracked=17, weight_value=6.0, bias_value=8.0
    )

    result = FedAvg().aggregate(global_state, [update_a, update_b])
    expected_weight = 0.25 * 2.0 + 0.75 * 6.0  # 5.0
    expected_bias = 0.25 * 0.0 + 0.75 * 8.0  # 6.0

    assert torch.allclose(result.aggregated_state["linear.weight"], torch.full((2, 2), expected_weight))
    assert torch.allclose(result.aggregated_state["linear.bias"], torch.full((2,), expected_bias))
    assert torch.equal(
        result.aggregated_state["bn.num_batches_tracked"], torch.tensor(17, dtype=torch.int64)
    )


def test_num_batches_tracked_aggregation_applies_via_existing_server_load_global_weights(
    tmp_path: Path,
) -> None:
    """A round with differing num_batches_tracked aggregates and loads through the unmodified
    FederatedServer.load_global_weights path -- no server changes were needed for this fix."""
    config = FederatedServerConfig(
        server_id="server_bn", seed=0, clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path
    )
    server = FederatedServer(config, model_fn=_TinyModelWithBatchNorm)

    update_a = _make_bn_update(
        "client_a", round_id=1, num_samples=10, num_batches_tracked=12, weight_value=2.0
    )
    update_b = _make_bn_update(
        "client_b", round_id=1, num_samples=30, num_batches_tracked=45, weight_value=6.0, bias_value=8.0
    )
    result = FedAvg().aggregate(server.get_global_weights(), [update_a, update_b])
    assert torch.equal(
        result.aggregated_state["bn.num_batches_tracked"], torch.tensor(45, dtype=torch.int64)
    )

    server.load_global_weights(result.aggregated_state)

    after = server.get_global_weights()
    for key in result.aggregated_state:
        assert torch.equal(after[key], result.aggregated_state[key])
    assert torch.equal(after["bn.num_batches_tracked"], torch.tensor(45, dtype=torch.int64))
