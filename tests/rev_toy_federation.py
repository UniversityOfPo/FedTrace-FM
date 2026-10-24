"""Tiny CPU federation for reviewer-response tests (no CIFAR-100, no GPU).

Uses the real, unmodified ``FederatedClient`` / ``FederatedServer`` / ``FedAvg`` /
``FederatedRunner`` with a small model that has BatchNorm (so float buffers and an
int64 ``num_batches_tracked`` counter are exercised) and a dataset whose
``__getitem__`` draws from the global torch RNG, like a data augmentation.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.attacks.traceability import AttackConfig, build_trace_row
from src.federated.aggregation.fedavg import FedAvg
from src.federated.client import FederatedClient, FederatedClientConfig
from src.federated.runner import FederatedRunner, FederatedRunnerConfig
from src.federated.server import FederatedServer, FederatedServerConfig

NUM_CLASSES = 4
FEATURES = 6


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(FEATURES, 8)
        self.bn = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.bn(self.fc1(x))))


class AugmentedDataset(Dataset):
    """Fixed data; ``__getitem__`` adds noise from the GLOBAL torch RNG (like augmentation)."""

    def __init__(self, num_samples: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(num_samples, FEATURES, generator=g)
        self.targets = [int(t) for t in torch.randint(0, NUM_CLASSES, (num_samples,), generator=g)]
        self.access_count = 0

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        self.access_count += 1
        return self.x[i] + 0.01 * torch.randn(FEATURES), self.targets[i]


class ImageLikeDataset(Dataset):
    """3x4x4 'images' for trigger tests; ``__getitem__`` draws from the global RNG."""

    def __init__(self, num_samples: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(num_samples, 3, 4, 4, generator=g)
        self.targets = [int(t) for t in torch.randint(0, NUM_CLASSES, (num_samples,), generator=g)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.x[i] + 0.001 * torch.rand(1), self.targets[i]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_snapshot() -> tuple[Any, ...]:
    return (torch.get_rng_state().clone(), np.random.get_state()[1].copy(), random.getstate())


def same_rng(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    return torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]


CLIENT_IDS = ("client_000", "client_001", "client_002")
SIZES = (24, 16, 20)


@dataclass
class FederationResult:
    globals_after: list[dict[str, torch.Tensor]] = field(default_factory=list)
    trace_rows: list[dict[str, Any]] = field(default_factory=list)
    history: list[Any] = field(default_factory=list)
    final_rng: tuple[Any, ...] | None = None
    runner: FederatedRunner | None = None


def build_and_run(
    tmp: Path,
    *,
    rounds: int = 4,
    seed: int = 7,
    runner_factory: Callable[..., FederatedRunner] = FederatedRunner,
    transform: Callable | None = None,
    dataset_wrapper: Callable[[str, Dataset], Dataset] | None = None,
    attack_config: AttackConfig | None = None,
) -> FederationResult:
    """Build a 3-client federation and run it; mirrors the NDSS script's trace bookkeeping."""
    seed_everything(seed)
    result = FederationResult()
    clients = {}
    for cid, n in zip(CLIENT_IDS, SIZES):
        dataset: Dataset = AugmentedDataset(n, seed=hash_seed(cid))
        if dataset_wrapper is not None:
            dataset = dataset_wrapper(cid, dataset)
        cfg = FederatedClientConfig(client_id=cid, local_epochs=1, batch_size=8, num_workers=0, device="cpu",
                                    optimizer="sgd", learning_rate=0.05, momentum=0.9, weight_decay=0.0,
                                    scheduler="none", mixed_precision=False, log_interval=0)
        client = FederatedClient(client_id=cid, model_fn=TinyNet, train_dataset=dataset, config=cfg)
        client.initialize_model()
        clients[cid] = client
    server = FederatedServer(FederatedServerConfig(server_id="toy", seed=seed, clients_per_round=3, minimum_updates=3,
                                                   checkpoint_dir=tmp / "server_ckpt"), model_fn=TinyNet)
    for cid, n in zip(CLIENT_IDS, SIZES):
        server.register_client(cid, num_samples=n)

    attack_config = attack_config or AttackConfig(attack_type="none")
    trainable = frozenset(name for name, _ in TinyNet().named_parameters())
    current: dict[str, dict[str, torch.Tensor]] = {}
    honest: dict[str, dict[str, torch.Tensor]] = {}
    previous: dict[str, dict[str, torch.Tensor]] = {}

    def ndss_like_transform(cid, round_id, state, pre):
        honest[cid] = {k: state[k] - pre[k] for k in trainable}
        submitted = transform(cid, round_id, state, pre) if transform is not None else state
        current[cid] = {k: submitted[k] - pre[k] for k in trainable}
        return submitted

    def round_callback(res):
        result.globals_after.append(runner.server.get_global_weights())
        accepted = set(res.successful_client_ids)
        for cid in sorted(clients):
            result.trace_rows.append(build_trace_row(
                round_id=res.round_id, client_id=cid, selected=cid in res.selected_client_ids,
                successful=cid in accepted and cid in current, attack_config=attack_config,
                client_delta=current.get(cid), peer_deltas=[current[p] for p in sorted(accepted) if p != cid],
                trainable_keys=trainable, honest_delta=honest.get(cid), self_prev_delta=previous.get(cid)))
            if cid in current:
                previous[cid] = current[cid]
        current.clear()
        honest.clear()

    runner = runner_factory(server=server, clients=clients, aggregation_strategy=FedAvg(),
                            config=FederatedRunnerConfig(total_rounds=rounds, fail_round_on_client_error=True,
                                                         continue_after_failed_round=False),
                            round_callback=round_callback, update_transform_callback=ndss_like_transform)
    runner.run()
    result.history = list(runner.get_history())
    result.final_rng = rng_snapshot()
    result.runner = runner
    return result


def hash_seed(text: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text))
