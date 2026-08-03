"""Federated learning components.

Sprint 3.1 scope: only the local-computation ``FederatedClient``
abstraction. No Server, aggregation (e.g. FedAvg), communication-round
orchestration, attack, or attribution logic exists in this package yet.
"""

from src.federated.client import FederatedClient, FederatedClientConfig

__all__ = ["FederatedClient", "FederatedClientConfig"]
