"""Reviewer-response run configuration (``configs/rev/*.rev.yaml``).

A rev configuration points at an NDSS-format ``base_config`` and may change only
the output locations (``output_dir`` / ``checkpoint_dir``); every training,
federation and NDSS-attack field is taken from the base config unchanged, so a
rev run and an NDSS run of the same base config are matched by construction.
Outputs are confined to the git-ignored ``results_rev/``, ``checkpoints_rev/``
and ``logs_rev/`` trees, and existing outputs are never overwritten.

Reviewer-response attacks (``rev_attack``) require the base config to have
``attack_type: none`` and must be marked ``provisional: true`` until the PI
pre-registers their parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from src.experiments.run_federated_cifar100 import FederatedExperimentConfig, load_federated_experiment_config

REV_SCHEMA = "fedtrace-rev-run-v1"
REV_ATTACK_TYPES = ("none", "backdoor", "label_flip", "constrained_backdoor", "relabel_only", "benign_class_drift")
ROOTS = {"output_dir": "results_rev", "checkpoint_dir": "checkpoints_rev", "update_log": "logs_rev"}
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_GIB = 1024**3


class RevConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RevConfig:
    path: Path
    run_id: str
    stage: str
    base_config_path: Path
    base_raw: dict[str, Any]
    resolved_raw: dict[str, Any]
    base: FederatedExperimentConfig
    output_dir: Path
    checkpoint_dir: Path
    update_log_enabled: bool
    update_log_dir: Path | None
    pre_transform_clients: tuple[str, ...]
    max_log_bytes: int | None
    reserve_bytes: int
    attack: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _confined(value: Any, root: str, what: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RevConfigError(f"{what} must be a non-empty relative path string")
    posix = PurePosixPath(value.replace("\\", "/"))
    if posix.is_absolute() or ":" in value or ".." in posix.parts:
        raise RevConfigError(f"{what} must be a relative path without '..'")
    if posix.parts[0] != root or len(posix.parts) < 2:
        raise RevConfigError(f"{what} must be inside {root}/")
    return Path(*posix.parts)


def _require(mapping: dict[str, Any], key: str, what: str) -> Any:
    if key not in mapping:
        raise RevConfigError(f"{what}: missing '{key}'")
    return mapping[key]


def _validate_attack(attack: dict[str, Any], base: FederatedExperimentConfig) -> dict[str, Any]:
    kind = attack.get("type", "none")
    if kind not in REV_ATTACK_TYPES:
        raise RevConfigError(f"rev_attack.type must be one of {REV_ATTACK_TYPES}")
    if kind == "none":
        return {"type": "none"}
    if base.attack_type != "none":
        raise RevConfigError("a rev_attack requires the base config to have attack_type: none")
    if attack.get("provisional") is not True:
        raise RevConfigError("rev_attack parameters are provisional until pre-registered: set provisional: true")
    if kind == "benign_class_drift":  # legitimate drift of an honest client: there is no malicious client
        if "malicious_client_id" in attack:
            raise RevConfigError("benign_class_drift has no malicious client; use drifted_client_id")
        malicious = None
        intervened = str(_require(attack, "drifted_client_id", "rev_attack"))
    else:
        malicious = str(_require(attack, "malicious_client_id", "rev_attack"))
        intervened = malicious
    onset = int(_require(attack, "onset_round", "rev_attack"))
    if not 0 <= onset < base.communication_rounds:
        raise RevConfigError("rev_attack.onset_round must be within the run")
    t_ref = int(attack.get("t_ref", 20))
    if onset < t_ref and attack.get("allow_calibration_contamination") is not True:
        raise RevConfigError("onset_round < t_ref contaminates calibration; set allow_calibration_contamination: true "
                             "only for the dedicated calibration-contamination stress test")
    out = {"type": kind, "malicious_client_id": malicious, "intervened_client_id": intervened, "onset_round": onset,
           "t_ref": t_ref, "provisional": True,
           "allow_calibration_contamination": bool(attack.get("allow_calibration_contamination", False))}
    if kind in ("backdoor", "constrained_backdoor", "relabel_only"):
        b = dict(_require(attack, "backdoor", "rev_attack"))
        target = int(_require(b, "target_class", "rev_attack.backdoor"))
        fraction = float(_require(b, "poison_fraction", "rev_attack.backdoor"))
        size = int(b.get("trigger_size", 3))
        if not 0 <= target < base.num_classes or not 0 < fraction <= 1 or not 1 <= size <= 8:
            raise RevConfigError("rev_attack.backdoor: target_class, poison_fraction or trigger_size out of range")
        out["backdoor"] = {"target_class": target, "poison_fraction": fraction, "trigger_size": size}
    if kind == "benign_class_drift":
        d = dict(_require(attack, "benign_drift", "rev_attack"))
        fraction = float(_require(d, "retained_sample_fraction", "rev_attack.benign_drift"))
        if not 0 < fraction < 1:
            raise RevConfigError("rev_attack.benign_drift.retained_sample_fraction must be in (0, 1)")
        out["benign_drift"] = {"retained_sample_fraction": fraction}
    if kind == "label_flip":
        f = dict(_require(attack, "label_flip", "rev_attack"))
        src, dst = int(_require(f, "source_class", "label_flip")), int(_require(f, "target_class", "label_flip"))
        fraction = float(_require(f, "flip_fraction", "label_flip"))
        if src == dst or not (0 <= src < base.num_classes and 0 <= dst < base.num_classes) or not 0 < fraction <= 1:
            raise RevConfigError("rev_attack.label_flip: invalid classes or flip_fraction")
        out["label_flip"] = {"source_class": src, "target_class": dst, "flip_fraction": fraction}
    if kind == "constrained_backdoor":
        c = dict(_require(attack, "constraint", "rev_attack"))
        if c.get("omniscient", False):
            raise RevConfigError("omniscient attacker behaviour is not implemented")
        kappa = float(_require(c, "kappa", "rev_attack.constraint"))
        if not 0 < kappa:
            raise RevConfigError("rev_attack.constraint.kappa must be > 0")
        if onset < t_ref:
            raise RevConfigError("the constrained attack needs clean own calibration rounds (onset_round >= t_ref)")
        out["constraint"] = {"kappa": kappa, "signals": ["m"], "omniscient": False}
    return out


def load_rev_config(path: Path | str, repo_root: Path | str = ".") -> RevConfig:
    path, repo_root = Path(path), Path(repo_root)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("rev_schema") != REV_SCHEMA:
        raise RevConfigError(f"{path.name}: rev_schema must be {REV_SCHEMA}")
    run_id = str(_require(raw, "run_id", "config"))
    if not _RUN_ID.match(run_id):
        raise RevConfigError("run_id may contain only letters, digits, '_', '.', '-'")
    base_path = Path(str(_require(raw, "base_config", "config")))
    if base_path.is_absolute() or ".." in base_path.parts:
        raise RevConfigError("base_config must be a repository-relative path")
    base_raw = yaml.safe_load((repo_root / base_path).read_text(encoding="utf-8"))

    overrides = dict(_require(raw, "output_overrides", "config"))
    if set(overrides) != {"output_dir", "checkpoint_dir"}:
        raise RevConfigError("output_overrides must set exactly output_dir and checkpoint_dir")
    output_dir = _confined(overrides["output_dir"], ROOTS["output_dir"], "output_dir")
    checkpoint_dir = _confined(overrides["checkpoint_dir"], ROOTS["checkpoint_dir"], "checkpoint_dir")
    resolved = dict(base_raw)
    resolved["output_dir"], resolved["checkpoint_dir"] = output_dir.as_posix(), checkpoint_dir.as_posix()

    log = dict(raw.get("update_log") or {"enabled": False})
    enabled = bool(log.get("enabled", False))
    log_dir = _confined(_require(log, "directory", "update_log"), ROOTS["update_log"], "update_log.directory") if enabled else None
    max_gib = log.get("max_gib")

    base = load_federated_experiment_config(repo_root / base_path)
    attack = _validate_attack(dict(raw.get("rev_attack") or {"type": "none"}), base)
    return RevConfig(
        path=path, run_id=run_id, stage=str(raw.get("stage", "")), base_config_path=base_path, base_raw=base_raw,
        resolved_raw=resolved, base=base, output_dir=output_dir, checkpoint_dir=checkpoint_dir,
        update_log_enabled=enabled, update_log_dir=log_dir,
        pre_transform_clients=tuple(str(c) for c in log.get("pre_transform_clients", [])),
        max_log_bytes=int(float(max_gib) * _GIB) if max_gib is not None else None,
        reserve_bytes=int(float(log.get("reserve_gib", 100)) * _GIB), attack=attack, raw=raw,
    )


def existing_outputs(config: RevConfig, repo_root: Path | str = ".") -> list[str]:
    """Output locations that already exist and are non-empty (a run never overwrites them)."""
    repo_root = Path(repo_root)
    paths = [config.output_dir, config.checkpoint_dir] + ([config.update_log_dir] if config.update_log_dir else [])
    return [p.as_posix() for p in paths if (repo_root / p).exists() and any((repo_root / p).iterdir())]
