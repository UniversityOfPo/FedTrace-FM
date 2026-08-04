"""Federated learning components.

Sprint 3.1 scope: the local-computation ``FederatedClient`` abstraction.
Sprint 3.2 scope: the client-server update contract (``ClientUpdate``,
``RoundRecord``) and the ``FederatedServer`` abstraction, which owns the
global model and round bookkeeping but implements no aggregation strategy,
runs no communication rounds, and never trains a client.
Sprint 3.3 scope: the pluggable aggregation interface
(``AggregationStrategy``, ``AggregationResult``) and its first concrete
strategy, ``FedAvg``. Aggregation is pure (state in, result out) — it does
not select clients, run rounds, train clients, or apply its result to any
model; the caller decides whether to load an ``AggregationResult`` into a
model (e.g. via ``FederatedServer.load_global_weights``).

No attack, attribution, or provenance logic exists in this package yet.
"""

from src.federated.aggregation import AggregationResult, AggregationStrategy, FedAvg
from src.federated.client import FederatedClient, FederatedClientConfig
from src.federated.contracts import ClientUpdate, RoundRecord
from src.federated.server import FederatedServer, FederatedServerConfig

__all__ = [
    "FederatedClient",
    "FederatedClientConfig",
    "ClientUpdate",
    "RoundRecord",
    "FederatedServer",
    "FederatedServerConfig",
    "AggregationStrategy",
    "AggregationResult",
    "FedAvg",
]
