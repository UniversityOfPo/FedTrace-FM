"""Benign class-composition drift: correct labels, no trigger, determinism, isolation, invariants, config, evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.rev import benign_drift as bd
from src.rev.attacks import BenignClassDriftDataset, labels_from_metadata, make_dataset_wrapper
from src.rev.config import RevConfigError, load_rev_config
from tests.rev_toy_federation import ImageLikeDataset, rng_snapshot, same_rng, seed_everything

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cifar-100-python"


def _drift(base, seed=1, client="client_002", onset=3, fraction=0.5):
    return BenignClassDriftDataset(base, retained_sample_fraction=fraction, onset_round=onset, seed=seed, client_id=client)


def test_labels_stay_correct_and_images_untouched() -> None:
    base = ImageLikeDataset(60, seed=4)
    d = _drift(base)
    d.set_round(3)
    for i in range(len(base)):
        seed_everything(7)
        x0, y0 = base[d.mapping[i]]
        seed_everything(7)
        x1, y1 = d[i]
        assert y1 == y0 == base.targets[d.mapping[i]]  # true label of the returned sample
        assert torch.equal(x1, x0)                      # no trigger, no image change
        assert y1 in d.retained_classes


def test_inactive_before_onset_and_length_invariant() -> None:
    base = ImageLikeDataset(60, seed=4)
    d = _drift(base, onset=3)
    assert len(d) == len(base)
    for round_id in (None, 0, 2):
        if round_id is not None:
            d.set_round(round_id)
        for i in range(len(base)):
            seed_everything(9)
            a = base[i]
            seed_everything(9)
            b = d[i]
            assert torch.equal(a[0], b[0]) and a[1] == b[1]


def test_retained_classes_cover_the_fraction_and_mapping_is_consistent() -> None:
    base = ImageLikeDataset(80, seed=5)
    labels = labels_from_metadata(base)
    d = _drift(base, fraction=0.5)
    retained = sum(1 for y in labels if y in d.retained_classes)
    assert retained >= 0.5 * len(labels)
    assert all(d.mapping[i] == i for i, y in enumerate(labels) if y in d.retained_classes)
    assert all(labels[d.mapping[i]] in d.retained_classes for i in range(len(labels)))
    assert d.replaced_fraction == pytest.approx(1 - retained / len(labels))


def test_selection_is_deterministic_and_isolated() -> None:
    base = ImageLikeDataset(80, seed=5)
    seed_everything(3)
    before = rng_snapshot()
    a = _drift(base, seed=11)
    assert same_rng(before, rng_snapshot())  # construction draws nothing from global RNGs
    b = _drift(base, seed=11)
    assert a.mapping == b.mapping and a.retained_classes == b.retained_classes
    assert _drift(base, seed=12).mapping != a.mapping
    assert _drift(base, seed=11, client="client_001").mapping != a.mapping
    a.set_round(5)
    seed_everything(3)
    _ = [a[i] for i in range(10)]
    after_wrapped = rng_snapshot()
    seed_everything(3)
    _ = [base[a.mapping[i]] for i in range(10)]
    assert same_rng(after_wrapped, rng_snapshot())  # access draws exactly what the base dataset draws


def test_wrapper_targets_only_the_drifted_client() -> None:
    attack = {"type": "benign_class_drift", "malicious_client_id": None, "intervened_client_id": "client_002",
              "onset_round": 20, "benign_drift": {"retained_sample_fraction": 0.5}}
    registry: list = []
    wrap = make_dataset_wrapper(attack, 7, registry)
    honest = ImageLikeDataset(20, seed=1)
    assert wrap("client_001", honest) is honest
    assert isinstance(wrap("client_002", ImageLikeDataset(20, seed=2)), BenignClassDriftDataset) and len(registry) == 1


@pytest.mark.parametrize("seed", ["123", "2026"])
def test_drift_config_has_no_malicious_client_and_matches_the_clean_base(seed: str) -> None:
    drift = load_rev_config(ROOT / f"configs/rev/X4_drift_d{seed}.rev.yaml", ROOT)
    clean = load_rev_config(ROOT / f"configs/rev/X1_clean_d{seed}.rev.yaml", ROOT)
    assert drift.base_raw == clean.base_raw and drift.base_config_path == clean.base_config_path
    assert drift.attack["malicious_client_id"] is None and drift.attack["intervened_client_id"] == "client_002"
    assert drift.attack["onset_round"] == 20 and drift.attack["benign_drift"] == {"retained_sample_fraction": 0.5}
    assert drift.output_dir.as_posix() == f"results_rev/X4_drift_d{seed}"


def test_drift_config_rejects_a_malicious_client(tmp_path) -> None:
    import yaml

    raw = yaml.safe_load((ROOT / "configs/rev/X4_drift_d123.rev.yaml").read_text(encoding="utf-8"))
    raw["rev_attack"]["malicious_client_id"] = "client_002"
    path = tmp_path / "x.rev.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(RevConfigError, match="no malicious client"):
        load_rev_config(path, ROOT)


def test_evaluation_rules() -> None:
    assert bd.norm_class(0.85).startswith("DECREASE") and bd.norm_class(1.2).startswith("INCREASE")
    assert bd.norm_class(1.0).startswith("NO MATERIAL")
    per = {"123": {"E8_any_honest_accusation": {"v3": False, "zp_norm": True, "cusum_norm": False}},
           "2026": {"E8_any_honest_accusation": {"v3": False, "zp_norm": False, "cusum_norm": False}}}
    v = bd.verdicts(per)
    assert v["zp_norm"].startswith("FALSIFIED") and "123" in v["zp_norm"] and v["v3"].startswith("NOT FALSIFIED")
    assert bd.FIXED_THRESHOLDS == {"Q-ZP-m": 1.75, "Q-CUSUM-m": 11.75}


@pytest.fixture(scope="module")
def real_partitions():
    if not DATA.exists():
        pytest.skip("local CIFAR-100 not available")
    import os

    import src.experiments.run_federated_cifar100 as ndss
    from src.data.cifar100 import split_train_validation

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        out = {}
        for seed in ("123", "2026"):
            cfg = load_rev_config(ROOT / f"configs/rev/X4_drift_d{seed}.rev.yaml", ROOT)
            train, _ = split_train_validation("data", download=False, validation_fraction=cfg.base.validation_fraction,
                                              seed=cfg.base.seed)
            out[seed] = (cfg, ndss.select_partitions(cfg.base, train, labels=ndss._labels_for_subset(train)))
    finally:
        os.chdir(cwd)
    return out


@pytest.mark.parametrize("seed", ["123", "2026"])
def test_real_pipeline_drift_is_correctly_labelled_and_prefix_identical(real_partitions, seed: str) -> None:
    cfg, parts = real_partitions[seed]
    subset = parts["client_002"]
    d = make_dataset_wrapper(cfg.attack, cfg.base.seed, [])("client_002", subset)
    labels = labels_from_metadata(subset)
    assert len(d) == len(subset) and 0.0 < d.replaced_fraction <= 0.5
    assert all(labels[d.mapping[i]] in d.retained_classes for i in range(len(labels)))
    d.set_round(19)

    def first_batch(dataset):
        torch.manual_seed(5)
        return next(iter(DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)))

    (xa, ya), (xb, yb) = first_batch(subset), first_batch(d)  # spawned workers, before onset
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    d.set_round(20)
    probe = next(i for i in range(len(labels)) if d.mapping[i] != i)
    seed_everything(2)
    x0, y0 = subset[d.mapping[probe]]
    seed_everything(2)
    x1, y1 = d[probe]
    assert y1 == y0 and torch.equal(x1, x0) and y1 != labels[probe]  # replaced by a retained-class sample, true label
