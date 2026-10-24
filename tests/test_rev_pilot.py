"""Stage C pilot: BadNets wrapper on the real NDSS data pipeline, matched-prefix check, effect and outcome analysis."""

from __future__ import annotations

import functools
import pickle
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.attacks.traceability import save_client_round_trace_csv
from src.rev.attacks import BackdoorPoisonedDataset, PatchTrigger, cifar100_white_patch, labels_from_metadata, make_dataset_wrapper
from src.rev.hooks import HookedRunner, RevHooks
from src.rev.metrics import attack_success_counts
from src.rev.online import OnlineConfig
from src.rev.pilot import forensic_outcome, matched_prefix, per_round_effect
from src.rev.update_log import UpdateLogConfig, UpdateLogger
from tests.rev_toy_federation import ImageLikeDataset, build_and_run, seed_everything
from tests.test_rev_attacks import _VectorBackdoor
from tests.test_rev_online import make_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cifar-100-python"


@pytest.fixture(scope="module")
def ndss_clients():
    """Client datasets built exactly as the NDSS script builds them for the clean seed-123 config (no download)."""
    if not DATA.exists():
        pytest.skip("local CIFAR-100 not available")
    import os

    import src.experiments.run_federated_cifar100 as ndss
    from src.data.cifar100 import split_train_validation
    from src.rev.config import load_rev_config

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        rev = load_rev_config(ROOT / "configs/rev/X2_badnets_d123.rev.yaml", ROOT)
        config = rev.base
        train_subset, _ = split_train_validation("data", download=False, validation_fraction=config.validation_fraction,
                                                 seed=config.seed)
        labels = ndss._labels_for_subset(train_subset)
        partitions = ndss.select_partitions(config, train_subset, labels=labels)
    finally:
        os.chdir(cwd)
    return rev, train_subset, labels, partitions


def test_badnets_wrapper_on_the_real_ndss_client_datasets(ndss_clients) -> None:
    rev, train_subset, labels, partitions = ndss_clients
    assert {c: len(s) for c, s in partitions.items()}["client_002"] == 7202  # archived partition summary
    registry: list = []
    wrapper = make_dataset_wrapper(rev.attack, rev.base.seed, registry)
    wrapped = {c: wrapper(c, s) for c, s in partitions.items()}
    assert [c for c, d in wrapped.items() if d is not partitions[c]] == ["client_002"] and len(registry) == 1
    poisoned: BackdoorPoisonedDataset = wrapped["client_002"]
    subset = partitions["client_002"]
    client_labels = labels_from_metadata(subset)  # nested Subset(Subset(CIFAR100))
    assert client_labels == [labels[i] for i in subset.indices]
    non_target = [i for i, y in enumerate(client_labels) if y != 0]
    assert len(poisoned.poisoned_indices) == round(0.5 * len(non_target))
    assert all(client_labels[i] != 0 for i in poisoned.poisoned_indices)

    trigger = cifar100_white_patch(3)
    probe = sorted(poisoned.poisoned_indices)[:4]
    clean_probe = next(i for i in range(len(subset)) if i not in poisoned.poisoned_indices)
    for round_id, active in ((19, False), (20, True)):
        poisoned.set_round(round_id)
        for i in probe + [clean_probe]:
            seed_everything(11)
            x0, y0 = subset[i]
            seed_everything(11)
            x1, y1 = poisoned[i]
            assert x0.shape == (3, 32, 32)
            if active and i in poisoned.poisoned_indices:
                assert y1 == 0 and torch.equal(x1, trigger.apply(x0))
                assert torch.allclose(x1[:, -3:, -3:], torch.tensor(trigger.value).view(3, 1, 1).expand(3, 3, 3))
            else:
                assert y1 == y0 and torch.equal(x1, x0)

    clone = pickle.loads(pickle.dumps(poisoned))  # DataLoader workers on Windows receive a pickled copy
    assert clone.current_round == 20 and clone.poisoned_indices == poisoned.poisoned_indices


def test_inactive_wrapper_yields_identical_batches_through_spawned_workers(ndss_clients) -> None:
    rev, _, _, partitions = ndss_clients
    subset = partitions["client_002"]
    poisoned = make_dataset_wrapper(rev.attack, rev.base.seed, [])("client_002", subset)
    poisoned.set_round(19)

    def first_batch(dataset):
        torch.manual_seed(5)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
        return next(iter(loader))

    (xa, ya), (xb, yb) = first_batch(subset), first_batch(poisoned)
    assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_triggered_test_view_excludes_target_class_images() -> None:
    if not DATA.exists():
        pytest.skip("local CIFAR-100 not available")
    from src.data.cifar100 import build_datasets
    from src.rev.attacks import TriggeredEvaluationDataset

    _, test = build_datasets(ROOT / "data", download=False)
    view = TriggeredEvaluationDataset(test, trigger=cifar100_white_patch(3), target_class=0)
    assert len(test) == 10000 and len(view) == 9900
    assert all(test.targets[i] != 0 for i in view.indices)


def _logged_pair(tmp_path: Path, attacked_seed: int = 7):
    def run(name, wrapper, seed):
        registry: list = []
        hooks = RevHooks(update_logger=UpdateLogger.open(UpdateLogConfig(root=tmp_path / name, run_id=name, min_free_bytes=0)))
        if wrapper:
            hooks.dataset_wrapper = lambda c, d: _VectorBackdoor(d, 3, registry) if c == "client_002" else d
            from src.rev.attacks import set_round_on_all

            hooks.round_start_hooks = [set_round_on_all(registry)]
        build_and_run(tmp_path / f"{name}_run", rounds=5, seed=seed, dataset_wrapper=hooks.dataset_wrapper,
                      runner_factory=functools.partial(HookedRunner, rev_hooks=hooks))
        hooks.update_logger.seal()
        return tmp_path / name

    return run("clean", False, 7), run("attacked", True, attacked_seed)


def test_matched_prefix_valid_for_a_true_pair(tmp_path: Path) -> None:
    clean, attacked = _logged_pair(tmp_path)
    report = matched_prefix(clean, attacked, onset=3)
    assert report["valid_matched_pair"], report
    assert report["pre_onset_globals_checked"] == 4 and report["pre_onset_submissions_checked"] == 9
    assert report["first_differing_global_round"] == 3
    assert report["first_differing_submission"] == [3, "client_002"]
    assert report["submissions_identical_at_onset"] == ["client_000", "client_001"]


def test_matched_prefix_invalid_when_anything_else_differs(tmp_path: Path) -> None:
    clean, attacked = _logged_pair(tmp_path, attacked_seed=8)  # a different seed is not a matched control
    report = matched_prefix(clean, attacked, onset=3)
    assert not report["valid_matched_pair"] and report["first_differing_global_round"] == -1


class _TinyImageNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(48, 4)

    def forward(self, x):
        return self.fc(x.flatten(1))


class _LabelledImages(Dataset):
    def __init__(self) -> None:
        base = ImageLikeDataset(40, seed=3)
        self.x, self.targets = base.x, base.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


def test_per_round_effect_matches_direct_evaluation(tmp_path: Path) -> None:
    torch.manual_seed(0)
    logger = UpdateLogger.open(UpdateLogConfig(root=tmp_path / "log", run_id="e", min_free_bytes=0))
    states = [_TinyImageNet().state_dict() for _ in range(3)]
    logger.log_client("c", 0, states[1], states[0], num_samples=1)
    logger.log_round_end(0, states[1], "completed", ("c",))
    logger.log_client("c", 1, states[2], states[1], num_samples=1)
    logger.log_round_end(1, states[2], "completed", ("c",))
    logger.seal()
    data = _LabelledImages()
    trigger = PatchTrigger(size=2, value=(9.0, 9.0, 9.0))
    rows = per_round_effect(tmp_path / "log", data, trigger, 0, _TinyImageNet)
    assert [r["round_id"] for r in rows] == [-1, 0, 1]
    model = _TinyImageNet()
    model.load_state_dict(states[2])
    with torch.no_grad():
        clean = (model(data.x).argmax(1) == torch.tensor(data.targets)).float().mean().item()
        keep = [i for i, y in enumerate(data.targets) if y != 0]
        trig = torch.stack([trigger.apply(data.x[i]) for i in keep])
        asr = attack_success_counts(model(trig).argmax(1).tolist(), 0)
    assert rows[-1]["clean_top1"] == pytest.approx(clean) and rows[-1]["asr_k"] == asr.k
    assert rows[-1]["asr_n"] == len(keep) == sum(1 for y in data.targets if y != 0)


def _trace_csv(path: Path, attacker: str | None) -> Path:
    rows = []
    for r in make_rows(rounds=40, attacker=attacker, onset=25):
        rows.append({**r, "selected": True, "attack_active": False, "attack_type": "none", "attack_strength_rho": 0.0,
                     "attack_configured": False, "malicious_client_id": None, "attack_start_round": None,
                     "peer_mean_cosine_similarity": 0.5, "naive_threshold": None, "naive_flag": False,
                     "first_flagged_round": None})
    save_client_round_trace_csv(rows, path)
    return path


def test_forensic_outcome_for_attacked_and_clean_runs(tmp_path: Path) -> None:
    attacked, trajectories = forensic_outcome(_trace_csv(tmp_path / "a.csv", "client_002"), "a", 1, "client_002", 25)
    out = attacked["outcome"]
    assert out["detected"] and out["attribution_correct"] and out["alarm_round"] == 27 and out["estimated_onset"] == 25
    assert out["delay"] == 2 and out["honest_alarm_clients"] == ()
    kinds = {(t["round_id"] < 20, t["kind"]) for t in trajectories}
    assert kinds == {(True, "retrospective"), (False, "causal")}
    clean, _ = forensic_outcome(_trace_csv(tmp_path / "c.csv", None), "c", 1, None, None, OnlineConfig())
    assert clean["outcome"]["detected"] is None and clean["outcome"]["honest_alarm_clients"] == ()
