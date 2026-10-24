"""Reviewer-response configuration: matched-by-construction configs, confinement of outputs, attack validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.experiments.run_federated_cifar100 import load_federated_experiment_config
from src.rev.config import RevConfigError, existing_outputs, load_rev_config
from src.rev.run_rev_experiment import main as rev_main

ROOT = Path(__file__).resolve().parents[1]
EXP016 = "configs/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123.yaml"
CLEAN123 = "configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml"
PATH_KEYS = ("output_dir", "checkpoint_dir")


def test_r0_config_is_exp016_with_only_output_locations_redirected() -> None:
    original = (ROOT / EXP016).read_text(encoding="utf-8").splitlines()
    r0 = (ROOT / "configs/rev/R0_exp016_repro.yaml").read_text(encoding="utf-8").splitlines()
    body = r0[r0.index(next(line for line in r0 if line.startswith("# ---"))) + 1:]
    assert len(body) == len(original)
    differing = [(a, b) for a, b in zip(original, body) if a != b]
    assert [b.split(":")[0] for _, b in differing] == list(PATH_KEYS)
    a = vars(load_federated_experiment_config(ROOT / EXP016))
    b = vars(load_federated_experiment_config(ROOT / "configs/rev/R0_exp016_repro.yaml"))
    assert {k: v for k, v in a.items() if k not in PATH_KEYS} == {k: v for k, v in b.items() if k not in PATH_KEYS}
    assert str(b["output_dir"]).replace("\\", "/") == "results_rev/R0_exp016_repro"
    assert str(b["checkpoint_dir"]).replace("\\", "/") == "checkpoints_rev/R0_exp016_repro"


def test_r1_rev_config_resolves_to_exp016_with_logging() -> None:
    cfg = load_rev_config(ROOT / "configs/rev/R1_exp016_logged.rev.yaml", ROOT)
    assert cfg.base_config_path.as_posix() == EXP016
    assert {k: v for k, v in cfg.resolved_raw.items() if k not in PATH_KEYS} == {
        k: v for k, v in cfg.base_raw.items() if k not in PATH_KEYS}
    assert cfg.update_log_enabled and cfg.update_log_dir.as_posix().startswith("logs_rev/")
    assert cfg.pre_transform_clients == ("client_002",) and cfg.attack == {"type": "none"}


def test_dry_run_validates_and_creates_nothing(capsys) -> None:
    before = {p for p in ("results_rev", "checkpoints_rev", "logs_rev") if (ROOT / p).exists()}
    assert rev_main(["--config", str(ROOT / "configs/rev/R1_exp016_logged.rev.yaml"), "--dry-run"]) == 0
    assert '"expected_client_states": 300' in capsys.readouterr().out
    assert {p for p in ("results_rev", "checkpoints_rev", "logs_rev") if (ROOT / p).exists()} == before


def _write(tmp_path: Path, **changes) -> Path:
    raw = yaml.safe_load((ROOT / "configs/rev/R1_exp016_logged.rev.yaml").read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            raw.pop(key, None)
        else:
            raw[key] = value
    path = tmp_path / "x.rev.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


GOOD_OUT = {"output_dir": "results_rev/x", "checkpoint_dir": "checkpoints_rev/x"}
BACKDOOR = {"type": "backdoor", "provisional": True, "malicious_client_id": "client_002", "onset_round": 20,
            "backdoor": {"target_class": 0, "poison_fraction": 0.5, "trigger_size": 3}}


@pytest.mark.parametrize("overrides, match", [
    ({"output_dir": "results/federated/x", "checkpoint_dir": "checkpoints_rev/x"}, "inside results_rev"),
    ({"output_dir": "results_rev/../results/x", "checkpoint_dir": "checkpoints_rev/x"}, "'..'"),
    ({"output_dir": "C:/x/results_rev/y", "checkpoint_dir": "checkpoints_rev/x"}, "relative"),
    ({"output_dir": "results_rev", "checkpoint_dir": "checkpoints_rev/x"}, "inside results_rev"),
    ({**GOOD_OUT, "seed": 5}, "exactly output_dir and checkpoint_dir"),
])
def test_outputs_are_confined_and_training_fields_cannot_be_overridden(tmp_path, overrides, match) -> None:
    with pytest.raises(RevConfigError, match=match):
        load_rev_config(_write(tmp_path, output_overrides=overrides), ROOT)


def test_update_log_must_be_inside_logs_rev(tmp_path) -> None:
    with pytest.raises(RevConfigError, match="logs_rev"):
        load_rev_config(_write(tmp_path, update_log={"enabled": True, "directory": "results_rev/log"}), ROOT)


@pytest.mark.parametrize("attack, match", [
    (BACKDOOR, "attack_type: none"),  # EXP_016 already has the NDSS Gaussian attack
])
def test_rev_attack_requires_a_clean_base(tmp_path, attack, match) -> None:
    with pytest.raises(RevConfigError, match=match):
        load_rev_config(_write(tmp_path, rev_attack=attack), ROOT)


@pytest.mark.parametrize("attack, match", [
    ({**BACKDOOR, "provisional": False}, "provisional"),
    ({**BACKDOOR, "onset_round": 10}, "contaminates calibration"),
    ({**BACKDOOR, "backdoor": {"target_class": 100, "poison_fraction": 0.5}}, "out of range"),
    ({**BACKDOOR, "type": "constrained_backdoor", "constraint": {"kappa": 2.5, "omniscient": True}}, "omniscient"),
    ({**BACKDOOR, "type": "constrained_backdoor", "onset_round": 10, "allow_calibration_contamination": True,
      "constraint": {"kappa": 2.5}}, "own calibration"),
    ({**BACKDOOR, "type": "label_flip", "label_flip": {"source_class": 3, "target_class": 3, "flip_fraction": 1}},
     "invalid classes"),
    ({**BACKDOOR, "type": "omniscient_lie"}, "rev_attack.type"),
])
def test_rev_attack_validation(tmp_path, attack, match) -> None:
    path = _write(tmp_path, base_config=CLEAN123, rev_attack=attack)
    with pytest.raises(RevConfigError, match=match):
        load_rev_config(path, ROOT)


def test_valid_provisional_attacks_load(tmp_path) -> None:
    cfg = load_rev_config(_write(tmp_path, base_config=CLEAN123, rev_attack=BACKDOOR), ROOT)
    assert cfg.attack["provisional"] is True and cfg.attack["backdoor"]["target_class"] == 0
    constrained = {**BACKDOOR, "type": "constrained_backdoor", "constraint": {"kappa": 2.5}}
    cfg = load_rev_config(_write(tmp_path, base_config=CLEAN123, rev_attack=constrained), ROOT)
    assert cfg.attack["constraint"] == {"kappa": 2.5, "signals": ["m"], "omniscient": False}
    contamination = {**BACKDOOR, "onset_round": 10, "allow_calibration_contamination": True}
    assert load_rev_config(_write(tmp_path, base_config=CLEAN123, rev_attack=contamination), ROOT).attack[
        "allow_calibration_contamination"] is True


def test_existing_outputs_are_reported_so_a_run_never_overwrites(tmp_path) -> None:
    cfg = load_rev_config(ROOT / "configs/rev/R1_exp016_logged.rev.yaml", ROOT)
    assert existing_outputs(cfg, tmp_path) == []
    (tmp_path / cfg.output_dir).mkdir(parents=True)
    assert existing_outputs(cfg, tmp_path) == []  # empty directories are fine
    (tmp_path / cfg.output_dir / "summary.json").write_text("{}", encoding="utf-8")
    assert existing_outputs(cfg, tmp_path) == [cfg.output_dir.as_posix()]
