"""Pluggable aggregation strategies for federated learning (Sprint 3.3).

``AggregationStrategy`` (base.py) defines the plugin contract; ``FedAvg``
(fedavg.py) is the first concrete strategy. Neither this package nor its
contents select clients, run communication rounds, train clients, or
apply results to any model — aggregation here is pure: state in, a
validated ``AggregationResult`` out.
"""

from src.federated.aggregation.base import (
    AggregationResult,
    AggregationStrategy,
    validate_common_update_structure,
)
from src.federated.aggregation.fedavg import FedAvg

__all__ = [
    "AggregationResult",
    "AggregationStrategy",
    "FedAvg",
    "validate_common_update_structure",
]
