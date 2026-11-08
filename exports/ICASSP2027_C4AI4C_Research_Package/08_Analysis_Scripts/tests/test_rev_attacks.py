"""Attack interfaces: trigger/label correctness, onset, data immutability, RNG isolation, constraint."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset, Subset

from src.analysis.fedtrace_detector import compute_longitudinal_calibration
from src.attacks.traceability import compute_update_l2_norm
from src.rev.attacks import (
    BackdoorPoisonedDataset,
    LabelFlipDataset,
    NormConstrainedTransform,
    PatchTrigger,
    TriggeredEvaluationDataset,
    cifar100_white_patch,
    labels_from_metadata,
    make_dataset_wrapper,
    set_round_on_all,
)
from src.rev.hashing import states_bit_identical
from src.rev.hooks import HookedRunner, RevHooks
from tests.rev_toy_federation import (
    AugmentedDataset,
    ImageLikeDataset,
    TinyNet,
    build_and_run,
    rng_snapshot,
    same_rng,
    seed_everything,
)

TRIGGER = PatchTrigger(size=2, value=(9.0, 8.0, 7.0))


def _backdoor(base: Dataset, onset: int = 3, fraction: float = 0.5, seed: int = 1, client: str = "c") -> BackdoorPoisonedDataset:
    return BackdoorPoisonedDataset(base, target_class=0, poison_fraction=fraction, onset_round=onset, trigger=TRIGGER,
                                   seed=seed, client_id=client)


def test_trigger_writes_only_the_bottom_right_patch_and_never_mutates_input() -> None:
    image = torch.zeros(3, 4, 4)
    out = TRIGGER.apply(image)
    assert torch.equal(image, torch.zeros(3, 4, 4))
    for c, v in enumerate(TRIGGER.value):
        assert torch.all(out[c, 2:, 2:] == v)
    mask = torch.ones(3, 4, 4, dtype=torch.bool)
    mask[:, 2:, 2:] = False
    assert torch.all(out[mask] == 0)
    with pytest.raises(ValueError):
        TRIGGER.apply(torch.zeros(1, 4, 4))
    white = cifar100_white_patch(3)
    assert white.size == 3 and all(v > 1.5 for v in white.value)  # (1 - mean) / std in normalised space


def test_backdoor_is_inactive_before_onset_and_while_the_round_is_unknown() -> None:
    base = ImageLikeDataset(40, seed=2)
    poisoned = _backdoor(base, onset=3)
    for round_id in (None, 0, 2):
        if round_id is not None:
            poisoned.set_round(round_id)
        for i in range(len(base)):
            seed_everything(5)
            a = base[i]
            seed_everything(5)
            b = poisoned[i]
            assert torch.equal(a[0], b[0]) and a[1] == b[1]


def test_backdoor_after_onset_triggers_and_relabels_exactly_the_selected_non_target_samples() -> None:
    base = ImageLikeDataset(40, seed=2)
    snapshot = base.x.clone(), list(base.targets)
    poisoned = _backdoor(base, onset=3, fraction=0.5)
    non_target = [i for i, y in enumerate(base.targets) if y != 0]
    assert len(poisoned.poisoned_indices) == round(0.5 * len(non_target))
    assert poisoned.poisoned_indices <= set(non_target)
    poisoned.set_round(3)
    for i in range(len(base)):
        seed_everything(5)
        clean_x, clean_y = base[i]
        seed_everything(5)
        x, y = poisoned[i]
        if i in poisoned.poisoned_indices:
            assert y == 0 and torch.equal(x, TRIGGER.apply(clean_x))
        else:
            assert y == clean_y and torch.equal(x, clean_x)
    assert torch.equal(base.x, snapshot[0]) and base.targets == snapshot[1]  # base data never mutated


def test_poisoned_index_selection_is_isolated_and_deterministic() -> None:
    base = ImageLikeDataset(60, seed=2)
    seed_everything(9)
    before = rng_snapshot()
    a = _backdoor(base, seed=1, client="client_002")
    a.set_round(5)
    _ = [a[i] for i in sorted(a.poisoned_indices)[:3]]
    # sample access draws only what the base dataset itself draws (one torch.rand per item here)
    seed_everything(9)
    _ = [base[i] for i in sorted(a.poisoned_indices)[:3]]
    after_base_only = rng_snapshot()
    seed_everything(9)
    b = _backdoor(base, seed=1, client="client_002")
    assert same_rng(before, rng_snapshot())  # construction draws nothing from the global RNGs
    b.set_round(5)
    _ = [b[i] for i in sorted(b.poisoned_indices)[:3]]
    assert same_rng(after_base_only, rng_snapshot())
    assert a.poisoned_indices == b.poisoned_indices
    assert _backdoor(base, seed=1, client="client_003").poisoned_indices != a.poisoned_indices
    assert _backdoor(base, seed=2, client="client_002").poisoned_indices != a.poisoned_indices


def test_wrapper_reads_each_base_sample_exactly_once() -> None:
    base = AugmentedDataset(10, seed=4)
    labels = labels_from_metadata(base)
    wrapped = LabelFlipDataset(base, source_class=labels[0], target_class=(labels[0] + 1) % 4, flip_fraction=1.0,
                               onset_round=0, seed=1, client_id="c")
    wrapped.set_round(0)
    base.access_count = 0
    _ = [wrapped[i] for i in range(len(wrapped))]
    assert base.access_count == len(base) and len(wrapped) == len(base)


def test_label_flip_changes_only_selected_source_class_labels_after_onset() -> None:
    base = ImageLikeDataset(80, seed=3)
    source = base.targets[0]
    flip = LabelFlipDataset(base, source_class=source, target_class=(source + 1) % 4, flip_fraction=0.5,
                            onset_round=2, seed=1, client_id="c")
    sources = [i for i, y in enumerate(base.targets) if y == source]
    assert flip.flipped_indices <= set(sources) and len(flip.flipped_indices) == round(0.5 * len(sources))
    flip.set_round(1)
    assert all(flip[i][1] == base.targets[i] for i in range(len(base)))
    flip.set_round(2)
    for i in range(len(base)):
        expected = (source + 1) % 4 if i in flip.flipped_indices else base.targets[i]
        assert flip[i][1] == expected
    with pytest.raises(ValueError):
        LabelFlipDataset(base, source_class=1, target_class=1, flip_fraction=1.0, onset_round=0, seed=1, client_id="c")


def test_labels_come_from_metadata_never_from_iterating_samples() -> None:
    base = AugmentedDataset(12, seed=4)
    base.access_count = 0
    sub = Subset(base, [3, 1, 7])
    assert labels_from_metadata(sub) == [base.targets[3], base.targets[1], base.targets[7]]
    assert labels_from_metadata(base) == base.targets
    assert base.access_count == 0

    class NoLabels(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, i):
            raise AssertionError("must not be read")

    with pytest.raises(TypeError):
        labels_from_metadata(NoLabels())


def test_triggered_evaluation_view_excludes_the_target_class() -> None:
    base = ImageLikeDataset(30, seed=6)
    view = TriggeredEvaluationDataset(base, trigger=TRIGGER, target_class=0)
    assert len(view) == sum(1 for y in base.targets if y != 0)
    x, y = view[0]
    assert y != 0 and torch.all(x[0, 2:, 2:] == 9.0)


def _toy_state(delta_scale: float, seed: int) -> tuple[dict, dict]:
    g = torch.Generator().manual_seed(seed)
    pre = {"w": torch.randn(20, generator=g), "b": torch.randn(3, generator=g),
           "bn.running_mean": torch.randn(3, generator=g), "bn.num_batches_tracked": torch.tensor(5)}
    post = dict(pre)
    post["w"] = pre["w"] + delta_scale * torch.randn(20, generator=g)
    post["b"] = pre["b"] + delta_scale * torch.randn(3, generator=g)
    post["bn.running_mean"] = pre["bn.running_mean"] + 0.1
    post["bn.num_batches_tracked"] = torch.tensor(9)
    return pre, post


def test_constrained_transform_keeps_own_z_m_within_kappa_as_measured_by_the_frozen_detector() -> None:
    keys = frozenset({"w", "b"})
    t = NormConstrainedTransform(malicious_client_id="client_002", onset_round=10, t_ref=10, kappa=2.5,
                                 trainable_keys=keys)
    rows = []
    for r in range(16):
        pre, post = _toy_state(delta_scale=1.0 + 0.05 * (r % 3) + (6.0 if r >= 10 else 0.0), seed=r)
        honest_pre, honest_post = _toy_state(1.0, seed=100 + r)
        assert t("client_000", r, honest_post, honest_pre) is honest_post  # honest clients untouched
        out = t("client_002", r, post, pre)
        if r < 10:
            assert out is post  # nothing changes before onset
        else:
            assert states_bit_identical({k: out[k] for k in ("bn.running_mean", "bn.num_batches_tracked")},
                                        {k: post[k] for k in ("bn.running_mean", "bn.num_batches_tracked")})
        rows.append({"round_id": r, "client_id": "client_002", "update_l2_norm": compute_update_l2_norm(out, pre, keys)})
    cal = compute_longitudinal_calibration(rows, "client_002", "update_l2_norm", t_ref=10)
    post_onset = [abs(row["update_l2_norm"] - cal.mean) / cal.std for row in rows if row["round_id"] >= 10]
    assert max(post_onset) <= 2.5
    assert all(rec.status == "SATISFIED" for rec in t.records)
    assert all(rec.scale < 1.0 and rec.z_raw > 2.5 for rec in t.records)  # the raw poisoned updates violated it
    assert t.constrained_signals == ("m",) and t.unconstrained_signals == ("t_res",)


def test_constrained_transform_limits_and_refusals() -> None:
    keys = frozenset({"w", "b"})
    with pytest.raises(NotImplementedError):
        NormConstrainedTransform(malicious_client_id="c", onset_round=20, t_ref=20, kappa=2.5, trainable_keys=keys,
                                 omniscient=True)
    with pytest.raises(ValueError):
        NormConstrainedTransform(malicious_client_id="c", onset_round=5, t_ref=20, kappa=2.5, trainable_keys=keys)
    t = NormConstrainedTransform(malicious_client_id="c", onset_round=2, t_ref=2, kappa=2.5, trainable_keys=keys)
    pre, post = _toy_state(1.0, seed=1)
    t("c", 0, post, pre)  # only one calibration observation
    assert t("c", 2, post, pre) is post and t.records[-1].status == "NO_CALIBRATION"


def test_data_poisoning_through_the_runner_keeps_the_pre_onset_prefix_identical(tmp_path: Path) -> None:
    clean = build_and_run(tmp_path / "clean", rounds=5)
    registry: list = []
    attack = {"type": "backdoor", "malicious_client_id": "client_002", "onset_round": 3,
              "backdoor": {"target_class": 0, "poison_fraction": 1.0, "trigger_size": 1}}
    wrapper = make_dataset_wrapper(attack, seed=7, registry=registry, trigger=PatchTrigger(size=1, value=(5.0,)))

    def toy_wrapper(cid, ds):
        return wrapper(cid, ds) if cid != "client_002" else _VectorBackdoor(ds, onset_round=3, registry=registry)

    hooks = RevHooks(dataset_wrapper=toy_wrapper, round_start_hooks=[set_round_on_all(registry)])
    attacked = build_and_run(tmp_path / "att", rounds=5, dataset_wrapper=hooks.dataset_wrapper,
                             runner_factory=functools.partial(HookedRunner, rev_hooks=hooks))
    for r in range(3):
        assert states_bit_identical(clean.globals_after[r], attacked.globals_after[r]), r
    assert not states_bit_identical(clean.globals_after[3], attacked.globals_after[3])
    assert len(registry) == 1 and registry[0].current_round == 4
    assert same_rng(clean.final_rng, attacked.final_rng)  # honest clients' RNG stream is unaffected


class _VectorBackdoor(BackdoorPoisonedDataset):
    """BackdoorPoisonedDataset for the toy 1-D feature vectors (trigger = last feature set to 5)."""

    def __init__(self, base, onset_round, registry):
        super().__init__(base, target_class=0, poison_fraction=1.0, onset_round=onset_round,
                         trigger=PatchTrigger(size=1, value=(5.0,)), seed=7, client_id="client_002")
        registry.append(self)

    def __getitem__(self, index):
        x, y = self.base[index]
        if self.active and index in self.poisoned_indices:
            x = x.clone()
            x[-1] = 5.0
            return x, 0
        return x, y


def test_toy_model_has_buffers_for_dtype_coverage() -> None:
    state = TinyNet().state_dict()
    assert state["bn.num_batches_tracked"].dtype == torch.int64
