"""Reviewer-response attack interfaces (CPU-testable; PROVISIONAL parameters).

Class B (data poisoning, measurable harmful objective):
    BackdoorPoisonedDataset -- BadNets-style patch trigger + relabel to a target
        class on a fixed fraction of the malicious client's non-target samples.
    LabelFlipDataset        -- targeted flip of a fixed fraction of source-class
        samples to a target class.
Class C prototype (detector-aware):
    NormConstrainedTransform -- applied to the malicious client's submitted update
        after (poisoned) local training: rescales the update so the client's own
        FedTrace ``z_m`` (update-norm z-score against its own calibration rounds)
        stays <= kappa. The attacker uses only information it has: its own past
        submitted updates. The temporal-residual signal ``z_t_res`` needs the
        same-round median of *other* clients' updates and is therefore NOT
        constrained; an omniscient variant is not implemented (configuring it
        raises). Nothing here guarantees evasion.

Isolation guarantees (tested):
    * poisoned-index selection uses a dedicated ``numpy.random.Generator`` seeded
      from (seed, client, attack); no global RNG is read or advanced, at
      construction or at sample time;
    * wrappers call ``base[i]`` exactly once per access, so the base dataset's own
      augmentation consumes exactly the same RNG as without the wrapper; length
      is unchanged, so DataLoader shuffling is unchanged;
    * before ``onset_round`` (and while the round is unknown) every sample is
      returned unchanged; the base dataset is never modified.

Labels for index selection must come from metadata (``labels`` argument, or a
``Subset`` over a dataset with ``.targets``), never by iterating the dataset,
which would run augmentations and consume RNG.
"""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from src.attacks.traceability import compute_update_l2_norm
from src.data.cifar100 import CIFAR100_MEAN, CIFAR100_STD

PROVISIONAL = True


def derive_attack_seed(seed: int, client_id: str, attack: str) -> int:
    digest = hashlib.sha256(f"rev-{seed}-{client_id}-{attack}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def labels_from_metadata(dataset: Dataset) -> list[int]:
    """Labels without touching samples: (nested) ``Subset`` over ``.targets``, or ``.targets`` / ``.tensors[1]``.

    NDSS client datasets are ``Subset(Subset(CIFAR100, split_indices), partition_indices)``; indices are mapped
    through every level.
    """
    if isinstance(dataset, Subset):
        parent = labels_from_metadata(dataset.dataset)
        return [parent[int(i)] for i in dataset.indices]
    if getattr(dataset, "targets", None) is not None:
        return [int(t) for t in dataset.targets]
    tensors = getattr(dataset, "tensors", None)
    if tensors is not None and len(tensors) >= 2:
        return [int(t) for t in tensors[1]]
    raise TypeError("cannot read labels from metadata; pass labels explicitly")


@dataclass(frozen=True)
class PatchTrigger:
    """A square patch written into the bottom-right corner of a CHW tensor (post-transform)."""

    size: int
    value: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("trigger size must be >= 1")

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 3 or image.shape[0] != len(self.value):
            raise ValueError(f"expected a CHW tensor with {len(self.value)} channels")
        out = image.clone()
        for c, v in enumerate(self.value):
            out[c, -self.size:, -self.size:] = v
        return out


def cifar100_white_patch(size: int = 3) -> PatchTrigger:
    """White pixels expressed in the normalised space of the NDSS CIFAR-100 pipeline."""
    return PatchTrigger(size=size, value=tuple((1.0 - m) / s for m, s in zip(CIFAR100_MEAN, CIFAR100_STD)))


def _relabel(original: Any, new_label: int) -> Any:
    if isinstance(original, torch.Tensor):
        return torch.tensor(new_label, dtype=original.dtype)
    return int(new_label)


class _RoundAwareDataset(Dataset):
    def __init__(self, base: Dataset, onset_round: int) -> None:
        if onset_round < 0:
            raise ValueError("onset_round must be >= 0")
        self.base = base
        self.onset_round = onset_round
        self.current_round: int | None = None

    def set_round(self, round_id: int) -> None:
        self.current_round = round_id

    @property
    def active(self) -> bool:
        return self.current_round is not None and self.current_round >= self.onset_round

    def __len__(self) -> int:
        return len(self.base)


def _select(candidates: Sequence[int], fraction: float, seed: int) -> frozenset[int]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    count = int(round(fraction * len(candidates)))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=count, replace=False) if count else []
    return frozenset(int(candidates[i]) for i in chosen)


class BackdoorPoisonedDataset(_RoundAwareDataset):
    """BadNets poisoning; with ``apply_trigger=False`` it is the matched relabel-only control: the same samples
    (same selection key) are relabelled to the target class but the image is returned unchanged (no trigger)."""

    def __init__(self, base: Dataset, *, target_class: int, poison_fraction: float, onset_round: int,
                 trigger: PatchTrigger, seed: int, client_id: str, labels: Sequence[int] | None = None,
                 apply_trigger: bool = True) -> None:
        super().__init__(base, onset_round)
        labels = list(labels) if labels is not None else labels_from_metadata(base)
        if len(labels) != len(base):
            raise ValueError("labels must align with the dataset")
        self.target_class = target_class
        self.trigger = trigger
        self.apply_trigger = apply_trigger
        candidates = [i for i, y in enumerate(labels) if y != target_class]
        self.poisoned_indices = _select(candidates, poison_fraction, derive_attack_seed(seed, client_id, "backdoor"))

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image, label = self.base[index]
        if self.active and index in self.poisoned_indices:
            return (self.trigger.apply(image) if self.apply_trigger else image), _relabel(label, self.target_class)
        return image, label


class LabelFlipDataset(_RoundAwareDataset):
    def __init__(self, base: Dataset, *, source_class: int, target_class: int, flip_fraction: float,
                 onset_round: int, seed: int, client_id: str, labels: Sequence[int] | None = None) -> None:
        super().__init__(base, onset_round)
        if source_class == target_class:
            raise ValueError("source_class and target_class must differ")
        labels = list(labels) if labels is not None else labels_from_metadata(base)
        if len(labels) != len(base):
            raise ValueError("labels must align with the dataset")
        self.target_class = target_class
        candidates = [i for i, y in enumerate(labels) if y == source_class]
        self.flipped_indices = _select(candidates, flip_fraction, derive_attack_seed(seed, client_id, "label_flip"))

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image, label = self.base[index]
        if self.active and index in self.flipped_indices:
            return image, _relabel(label, self.target_class)
        return image, label


class TriggeredEvaluationDataset(Dataset):
    """Test-time view for attack success rate: non-target-class samples with the trigger applied.

    Returns ``(triggered_image, original_label)``; success = prediction == target class.
    """

    def __init__(self, base: Dataset, *, trigger: PatchTrigger, target_class: int,
                 labels: Sequence[int] | None = None) -> None:
        labels = list(labels) if labels is not None else labels_from_metadata(base)
        self.base = base
        self.trigger = trigger
        self.indices = [i for i, y in enumerate(labels) if y != target_class]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        image, label = self.base[self.indices[index]]
        return self.trigger.apply(image), label


@dataclass
class ConstraintRecord:
    round_id: int
    status: str
    raw_norm: float
    submitted_norm: float
    scale: float
    z_raw: float | None
    z_submitted: float | None


@dataclass
class NormConstrainedTransform:
    """Detector-aware prototype: keep the attacker's own ``z_m`` <= ``kappa`` (PROVISIONAL)."""

    malicious_client_id: str
    onset_round: int
    t_ref: int
    kappa: float
    trainable_keys: frozenset[str]
    omniscient: bool = False
    margin: float = 1e-6
    own_norms: dict[int, float] = field(default_factory=dict)
    records: list[ConstraintRecord] = field(default_factory=list)

    constrained_signals: tuple[str, ...] = ("m",)
    unconstrained_signals: tuple[str, ...] = ("t_res",)

    def __post_init__(self) -> None:
        if self.omniscient:
            raise NotImplementedError("an omniscient attacker (peer-update access) is not implemented")
        if self.kappa <= 0:
            raise ValueError("kappa must be > 0")
        if self.onset_round < self.t_ref:
            raise ValueError("onset_round < t_ref: the attacker's own calibration window would be poisoned")

    def calibration(self) -> tuple[float, float] | None:
        values = [n for r, n in sorted(self.own_norms.items()) if r < self.t_ref]
        if len(values) < 2:
            return None
        sd = statistics.stdev(values)
        return (statistics.mean(values), sd) if sd > 0 else None

    def __call__(self, client_id: str, round_id: int, state: Mapping[str, torch.Tensor],
                 pre_round_global_state: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if client_id != self.malicious_client_id:
            return state
        norm = compute_update_l2_norm(state, pre_round_global_state, self.trainable_keys)
        if round_id < self.onset_round:
            self.own_norms[round_id] = norm
            return state
        cal = self.calibration()
        if cal is None:
            self.records.append(ConstraintRecord(round_id, "NO_CALIBRATION", norm, norm, 1.0, None, None))
            return state
        mu, sd = cal
        upper, lower = mu + self.kappa * sd, mu - self.kappa * sd
        scale = 1.0
        if norm > upper:
            scale = upper * (1.0 - self.margin) / norm
        elif norm < lower and norm > 0:
            scale = lower * (1.0 + self.margin) / norm
        if scale == 1.0:
            new_state = state
        else:
            new_state = dict(state)
            for key in self.trainable_keys:
                delta = state[key] - pre_round_global_state[key]
                new_state[key] = pre_round_global_state[key] + delta * scale
        submitted = compute_update_l2_norm(new_state, pre_round_global_state, self.trainable_keys)
        z_sub = abs(submitted - mu) / sd
        status = "SATISFIED" if z_sub <= self.kappa else "VIOLATED"
        self.records.append(ConstraintRecord(round_id, status, norm, submitted, scale, abs(norm - mu) / sd, z_sub))
        return new_state


def make_dataset_wrapper(attack: Mapping[str, Any], seed: int, registry: list[_RoundAwareDataset],
                         trigger: PatchTrigger | None = None):
    """``(client_id, dataset) -> dataset`` wrapping only the malicious client's dataset."""
    kind = attack["type"]
    malicious = attack["malicious_client_id"]
    onset = int(attack["onset_round"])

    def wrapper(client_id: str, dataset: Dataset) -> Dataset:
        if client_id != malicious:
            return dataset
        if kind in ("backdoor", "constrained_backdoor", "relabel_only"):
            params = attack["backdoor"]
            wrapped: _RoundAwareDataset = BackdoorPoisonedDataset(
                dataset, target_class=int(params["target_class"]), poison_fraction=float(params["poison_fraction"]),
                onset_round=onset, trigger=trigger or cifar100_white_patch(int(params["trigger_size"])),
                seed=seed, client_id=client_id, apply_trigger=kind != "relabel_only")
        elif kind == "label_flip":
            params = attack["label_flip"]
            wrapped = LabelFlipDataset(
                dataset, source_class=int(params["source_class"]), target_class=int(params["target_class"]),
                flip_fraction=float(params["flip_fraction"]), onset_round=onset, seed=seed, client_id=client_id)
        else:
            raise ValueError(f"no dataset wrapper for attack type {kind!r}")
        registry.append(wrapped)
        return wrapped

    return wrapper


def set_round_on_all(registry: Iterable[_RoundAwareDataset]):
    def hook(round_id: int) -> None:
        for dataset in registry:
            dataset.set_round(round_id)

    return hook
