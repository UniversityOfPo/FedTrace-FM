"""Reviewer-response experiment entry point (Stage B onward; NOT run in Stage A).

    python -m src.rev.run_rev_experiment --config configs/rev/<run>.rev.yaml [--dry-run]

Runs the unmodified NDSS ``main()`` of ``src.experiments.run_federated_cifar100``
on a resolved copy of the base config (only output locations changed), with
reviewer-response hooks installed (update logging, rev attacks; see
:mod:`src.rev.hooks`). Writes, next to the NDSS outputs:

    resolved_ndss_config.yaml  the exact config NDSS main() ran
    rev_provenance.json        pre-run git/environment/config provenance, plus
                               post-run torch/GPU/driver, partition and model
                               initialization fingerprints
    rev_status.json            COMPLETED or FAILED (exception type only)
    rev_attack_records.json    constraint records / attack success (rev attacks)

``--dry-run`` validates the configuration, checks that no output exists, and
prints the plan; it trains nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from src.rev.config import RevConfig, existing_outputs, load_rev_config
from src.rev.hashing import atomic_write_bytes, load_tensor_state, sha256_file, state_content_sha256
from src.rev.provenance import (
    assert_no_private_strings,
    capture_provenance,
    partition_fingerprint,
    torch_environment,
)


def _write_json(path: Path, payload: dict[str, Any], repo_root: Path) -> None:
    assert_no_private_strings(payload, repo_root)
    atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def plan(config: RevConfig) -> dict[str, Any]:
    base = config.base
    states_per_round = base.clients_per_round + len(config.pre_transform_clients)
    return {
        "run_id": config.run_id,
        "stage": config.stage,
        "base_config": config.base_config_path.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "checkpoint_dir": config.checkpoint_dir.as_posix(),
        "update_log_dir": config.update_log_dir.as_posix() if config.update_log_dir else None,
        "expected_client_states": states_per_round * base.communication_rounds if config.update_log_enabled else 0,
        "expected_global_states": base.communication_rounds + 1 if config.update_log_enabled else 0,
        "rev_attack": config.attack,
        "ndss_attack": base.attack_type,
        "seed": base.seed,
    }


def run(config: RevConfig, repo_root: Path) -> int:
    import src.experiments.run_federated_cifar100 as ndss
    from src.rev.attacks import NormConstrainedTransform, make_dataset_wrapper, set_round_on_all
    from src.rev.hooks import RevHooks, ndss_hooks_installed
    from src.rev.update_log import UpdateLogConfig, UpdateLogger

    out = repo_root / config.output_dir
    out.mkdir(parents=True, exist_ok=True)  # main() has already refused non-empty outputs
    resolved_path = out / "resolved_ndss_config.yaml"
    atomic_write_bytes(resolved_path, yaml.safe_dump(config.resolved_raw, sort_keys=False).encode("utf-8"))
    resolved = ndss.load_federated_experiment_config(resolved_path)
    if {k: v for k, v in vars(resolved).items() if k not in ("output_dir", "checkpoint_dir")} != {
        k: v for k, v in vars(config.base).items() if k not in ("output_dir", "checkpoint_dir")
    }:
        raise RuntimeError("resolved config differs from the base config beyond output locations")

    provenance = capture_provenance(
        repo_root, config_files=[config.path, config.base_config_path, Path(config.base_raw["dataset_config"])],
        parsed_config={"rev": config.raw, "resolved_ndss": config.resolved_raw}, seed=config.base.seed,
        query_gpu=False,
    )
    provenance["run_id"] = config.run_id
    _write_json(out / "rev_provenance.json", provenance, repo_root)

    hooks = RevHooks()
    p = plan(config)
    if config.update_log_enabled:
        hooks.update_logger = UpdateLogger.open(UpdateLogConfig(
            root=repo_root / config.update_log_dir, run_id=config.run_id,
            pre_transform_clients=config.pre_transform_clients, max_bytes=config.max_log_bytes,
            min_free_bytes=config.reserve_bytes, expected_client_states=p["expected_client_states"],
            expected_global_states=p["expected_global_states"],
        ))
    registry: list = []
    constraint: NormConstrainedTransform | None = None
    if config.attack["type"] != "none":
        hooks.dataset_wrapper = make_dataset_wrapper(config.attack, config.base.seed, registry)
        hooks.round_start_hooks = [set_round_on_all(registry)]
        if config.attack["type"] == "constrained_backdoor":
            constraint = NormConstrainedTransform(
                malicious_client_id=config.attack["malicious_client_id"], onset_round=config.attack["onset_round"],
                t_ref=config.attack["t_ref"], kappa=config.attack["constraint"]["kappa"],
                trainable_keys=frozenset(), omniscient=False,
            )
            hooks.attack_transform = _bind_trainable_keys(constraint, hooks)

    original_parse_args = ndss.parse_args
    ndss.parse_args = lambda: argparse.Namespace(config=resolved_path)
    status: dict[str, Any] = {"run_id": config.run_id, "status": "RUNNING"}
    try:
        with ndss_hooks_installed(ndss, hooks):
            ndss.main()
    except BaseException as exc:
        status = {"run_id": config.run_id, "status": "FAILED", "error_type": type(exc).__name__,
                  "update_log": "left unsealed (interrupted); rerun under a new run id"}
        _write_json(out / "rev_status.json", status, repo_root)
        raise
    finally:
        ndss.parse_args = original_parse_args

    seal = hooks.update_logger.seal() if hooks.update_logger is not None else None
    post: dict[str, Any] = {"torch_post_run": torch_environment(query_gpu=True),
                            "model_init_content_sha256": hooks.init_global_sha256}
    partition_file = out / "client_partition_indices.json"
    if partition_file.exists():
        post["partition_sha256"] = partition_fingerprint(json.loads(partition_file.read_text(encoding="utf-8")))
    final = repo_root / config.checkpoint_dir / "final_global_model.pt"
    if final.exists():
        import torch

        payload = torch.load(final, map_location="cpu", weights_only=False)
        post["final_global_content_sha256"] = state_content_sha256(payload["model_state_dict"])
    provenance["post_run"] = post
    _write_json(out / "rev_provenance.json", provenance, repo_root)

    if config.attack["type"] != "none":
        _write_json(out / "rev_attack_records.json", _attack_records(config, constraint, final, repo_root), repo_root)
    status = {"run_id": config.run_id, "status": "COMPLETED", "update_log_seal": seal,
              "update_log_manifest_sha256": seal["manifest_sha256"] if seal else None}
    _write_json(out / "rev_status.json", status, repo_root)
    return 0


def _bind_trainable_keys(constraint, hooks):
    """Fill the constraint's trainable keys from the client's live model parameter names (no RNG)."""

    def transform(client_id, round_id, state, pre_round_global_state):
        if not constraint.trainable_keys:
            model = hooks.runners[0].clients[client_id].model
            constraint.trainable_keys = frozenset(name for name, _ in model.named_parameters())
        return constraint(client_id, round_id, state, pre_round_global_state)

    return transform


def _attack_records(config: RevConfig, constraint, final: Path, repo_root: Path) -> dict[str, Any]:
    import torch

    from src.data.cifar100 import build_datasets
    from src.models.resnet18_cifar import resnet18_cifar
    from src.rev.attacks import TriggeredEvaluationDataset, cifar100_white_patch
    from src.rev.metrics import attack_success_counts, predict
    from src.utils.config import load_data_config

    records: dict[str, Any] = {"attack": config.attack, "provisional": True}
    if constraint is not None:
        records["constraint_records"] = [vars(r) for r in constraint.records]
        records["constraint_violations"] = sum(r.status == "VIOLATED" for r in constraint.records)
    if config.attack["type"] in ("backdoor", "constrained_backdoor") and final.exists():
        data_config = load_data_config(Path(config.base_raw["dataset_config"]))
        _, test = build_datasets(data_config.data_root, download=False)
        params = config.attack["backdoor"]
        view = TriggeredEvaluationDataset(test, trigger=cifar100_white_patch(params["trigger_size"]),
                                          target_class=params["target_class"])
        model = resnet18_cifar(num_classes=config.base.num_classes)
        model.load_state_dict(torch.load(final, map_location="cpu", weights_only=False)["model_state_dict"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        success = attack_success_counts(predict(model.to(device), view, device), params["target_class"])
        records["final_attack_success"] = {"k": success.k, "n": success.n, "rate": success.rate}
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path.cwd()
    config = load_rev_config(args.config, repo_root)
    existing = existing_outputs(config, repo_root)
    if existing:
        print(f"refusing to run: outputs already exist: {existing}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps(plan(config), indent=2, sort_keys=True))
        return 0
    return run(config, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
