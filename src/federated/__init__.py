"""Federated learning components.

Sprint 3.1 scope: the local-computation ``FederatedClient`` abstraction.
Sprint 3.2 scope: the client-server update contract (``ClientUpdate``,
``RoundRecord``) and the ``FederatedServer`` abstraction, which owns the
global model and round bookkeeping but implements no aggregation strategy
(e.g. FedAvg), runs no communication rounds, and never trains a client.
No attack, attribution, or provenance logic exists in this package yet.
"""

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
]
