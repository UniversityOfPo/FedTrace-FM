"""Integration hooks: a ``FederatedRunner`` subclass and installation into the NDSS script.

No protected file is edited. The NDSS runner already exposes the two hook points
needed here (``update_transform_callback`` and ``round_callback``); this module
composes additional behaviour around whatever callbacks the caller passes:

    per client:  rev attack transform -> caller's transform (NDSS attack + trace
                 bookkeeping) -> update logger (records the exact submitted state)
    per round:   round-start hooks (e.g. poisoned datasets learn the round id)
                 -> unchanged ``FederatedRunner.run_round`` -> caller's round
                 callback -> global-state logging

:func:`ndss_hooks_installed` temporarily rebinds the names ``FederatedRunner``
(and, only for data-poisoning attacks, ``FederatedClient``) inside the
``src.experiments.run_federated_cifar100`` module object, so its unmodified
``main()`` builds hooked objects. The names are restored on exit.

When no hook is configured, :class:`HookedRunner` passes ``None`` callbacks
through unchanged, i.e. it is the original runner.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset

from src.federated.client import FederatedClient
from src.federated.runner import FederatedRoundResult, FederatedRunner
from src.rev.hashing import state_content_sha256
from src.rev.update_log import UpdateLogger

UpdateTransform = Callable[[str, int, Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]]
DatasetWrapper = Callable[[str, Dataset], Dataset]


@dataclass
class RevHooks:
    """What to compose around the runner.

    Attributes:
        update_logger: Optional :class:`UpdateLogger`.
        attack_transform: Optional update transform applied *before* the caller's
            transform (so NDSS trace bookkeeping sees the final submitted state).
        round_start_hooks: Called with the round id before each round.
        dataset_wrapper: Optional ``(client_id, dataset) -> dataset`` used to wrap
            client training datasets (data-poisoning attacks); identity for clients
            it does not target.
        init_global_sha256: Filled with the initial global-state content hash
            (model-initialization fingerprint) when the runner is constructed.
    """

    update_logger: UpdateLogger | None = None
    attack_transform: UpdateTransform | None = None
    round_start_hooks: Sequence[Callable[[int], None]] = ()
    dataset_wrapper: DatasetWrapper | None = None
    init_global_sha256: str | None = None
    runners: list[FederatedRunner] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.update_logger or self.attack_transform or self.round_start_hooks or self.dataset_wrapper)


class HookedRunner(FederatedRunner):
    """``FederatedRunner`` with :class:`RevHooks` composed around the caller's callbacks."""

    def __init__(self, *args: Any, rev_hooks: RevHooks | None = None, **kwargs: Any) -> None:
        hooks = rev_hooks or RevHooks()
        inner_transform = kwargs.pop("update_transform_callback", None)
        inner_round_callback = kwargs.pop("round_callback", None)
        self.rev_hooks = hooks

        transform = inner_transform
        if hooks.attack_transform is not None or hooks.update_logger is not None:
            transform = functools.partial(self._composed_transform, inner_transform)
        round_callback = inner_round_callback
        if hooks.update_logger is not None:
            round_callback = functools.partial(self._composed_round_callback, inner_round_callback)

        super().__init__(*args, update_transform_callback=transform, round_callback=round_callback, **kwargs)
        hooks.init_global_sha256 = state_content_sha256(self.server.get_global_weights())
        hooks.runners.append(self)

    def run_round(self, round_id: int) -> FederatedRoundResult:
        for hook in self.rev_hooks.round_start_hooks:
            hook(round_id)
        logger = self.rev_hooks.update_logger
        if logger is not None:
            logger.begin_round(round_id)
        result = super().run_round(round_id)
        if logger is not None and logger.failure is not None:
            # The runner records client-side exceptions as client failures; a logging
            # failure must instead abort the run, so it is re-raised here.
            raise RuntimeError(f"update logging failed in round {round_id}") from logger.failure
        return result

    def _composed_transform(
        self,
        inner: UpdateTransform | None,
        client_id: str,
        round_id: int,
        honest_state: Mapping[str, torch.Tensor],
        pre_round_global_state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        state = honest_state
        if self.rev_hooks.attack_transform is not None:
            state = self.rev_hooks.attack_transform(client_id, round_id, state, pre_round_global_state)
        if inner is not None:
            state = inner(client_id, round_id, state, pre_round_global_state)
        logger = self.rev_hooks.update_logger
        if logger is not None:
            logger.log_client(
                client_id, round_id, state, pre_round_global_state,
                num_samples=len(self.clients[client_id].train_dataset), pre_transform_state=honest_state,
            )
        return state

    def _composed_round_callback(
        self, inner: Callable[[FederatedRoundResult], None] | None, result: FederatedRoundResult
    ) -> None:
        if inner is not None:
            inner(result)
        if self.rev_hooks.update_logger.failure is not None:  # type: ignore[union-attr]
            return  # the log is already broken; run_round() re-raises the original failure
        self.rev_hooks.update_logger.log_round_end(  # type: ignore[union-attr]
            result.round_id, self.server.get_global_weights(), result.round_status,
            successful_client_ids=tuple(result.successful_client_ids),
        )


def make_client_factory(original: type[FederatedClient], wrapper: DatasetWrapper) -> Callable[..., FederatedClient]:
    """A drop-in for ``FederatedClient(...)`` that wraps ``train_dataset`` first."""

    def factory(*args: Any, **kwargs: Any) -> FederatedClient:
        if "train_dataset" not in kwargs or "client_id" not in kwargs:
            raise TypeError("the rev client factory requires keyword arguments client_id and train_dataset")
        kwargs["train_dataset"] = wrapper(kwargs["client_id"], kwargs["train_dataset"])
        return original(*args, **kwargs)

    return factory


@contextlib.contextmanager
def ndss_hooks_installed(module: Any, hooks: RevHooks) -> Iterator[None]:
    """Temporarily point ``module.FederatedRunner`` (and ``FederatedClient``) at hooked versions."""
    originals = {"FederatedRunner": module.FederatedRunner, "FederatedClient": module.FederatedClient}
    if originals["FederatedRunner"] is not FederatedRunner or originals["FederatedClient"] is not FederatedClient:
        raise RuntimeError("NDSS module names are already patched")
    module.FederatedRunner = functools.partial(HookedRunner, rev_hooks=hooks)
    if hooks.dataset_wrapper is not None:
        module.FederatedClient = make_client_factory(FederatedClient, hooks.dataset_wrapper)
    try:
        yield
    finally:
        module.FederatedRunner = originals["FederatedRunner"]
        module.FederatedClient = originals["FederatedClient"]
