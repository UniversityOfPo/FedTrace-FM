"""Probe-set feasibility and leakage audit for protocol 16 (read-only, CPU, no download).

    python -m src.rev.probe_audit --seeds 123 2026

CLAUDE-REPORTED, DEVELOPMENT ONLY. For each development seed: reconstructs the 45,000 / 5,000 train / validation
split exactly as the training runner does, proves the reconstruction by re-deriving the Dirichlet client partition
and comparing it with the partition saved by the run, checks that the candidate probe images (the seed's validation
subset) are disjoint from every client's training data and from the official test set (indices and exact pixel
duplicates), writes a deterministic label-free probe manifest with a fixed fit / evaluation split, and checks that
the model states needed by protocol 16 exist in the sealed update logs and load into the architecture.

Labels are read here only to reproduce the runner's stratified split and partition (an exact replay of the training
pipeline); the manifest and the probe tensors contain no labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CONFIGS = {"123": "configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml",
           "2026": "configs/cifar100_fedavg_dirichlet_a01_seed2026_50rounds.yaml"}
RUNS = ("X1_clean_d{s}", "X2_badnets_d{s}", "X3_relabel_d{s}", "X4_drift_d{s}")
ROUNDS = range(16, 30)  # protocol 16: client submissions of rounds 16..29 and their pre-round globals (rounds 15..28)
SPLIT_TAG = "FedTrace-probe-split-v1"
MANIFEST_VERSION = 2  # v2: exact pixel duplicates of client-training or test images are excluded (label-free rule)


def fit_or_eval(seed: str, absolute_index: int) -> str:
    """Label-free, deterministic fit / evaluation assignment of a probe image."""
    digest = hashlib.sha256(f"{SPLIT_TAG}|{seed}|{absolute_index}".encode()).digest()
    return "fit" if digest[0] % 2 == 0 else "eval"


def _raw(root: Path, name: str) -> np.ndarray:
    with (root / "data" / "cifar-100-python" / name).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")["data"]


def _pixel_hashes(images: np.ndarray, indices: Sequence[int] | None = None) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i in (range(len(images)) if indices is None else indices):
        out.setdefault(hashlib.sha256(images[i].tobytes()).hexdigest(), []).append(int(i))
    return out


def reconstruct(root: Path, seed: str) -> dict[str, Any]:
    from src.data.cifar100 import split_train_validation
    from src.experiments.run_federated_cifar100 import _labels_for_subset, load_federated_experiment_config, select_partitions

    config = load_federated_experiment_config(root / CONFIGS[seed])
    train_subset, val_subset = split_train_validation(root / "data", download=False,
                                                      validation_fraction=config.validation_fraction, seed=config.seed)
    partitions = select_partitions(config, train_subset, labels=_labels_for_subset(train_subset))
    derived = {c: [int(i) for i in p.indices] for c, p in partitions.items()}
    saved = json.loads((root / "results_rev" / RUNS[0].format(s=seed) / "client_partition_indices.json").read_text())
    return {"train_abs": [int(i) for i in train_subset.indices], "val_abs": [int(i) for i in val_subset.indices],
            "derived_partition": derived, "saved_partition": saved, "config_seed": config.seed,
            "validation_fraction": config.validation_fraction}


def audit_seed(root: Path, seed: str, train_raw: np.ndarray, test_hashes: dict[str, list[int]]) -> dict[str, Any]:
    rec = reconstruct(root, seed)
    train_abs, val_abs = rec["train_abs"], rec["val_abs"]
    partition_matches = rec["derived_partition"] == rec["saved_partition"]
    for run in RUNS[1:]:  # every arm of the seed uses the same partition
        other = json.loads((root / "results_rev" / run.format(s=seed) / "client_partition_indices.json").read_text())
        partition_matches = partition_matches and other == rec["saved_partition"]
    client_abs = {c: {train_abs[i] for i in idx} for c, idx in rec["saved_partition"].items()}
    all_client = set().union(*client_abs.values())
    val = set(val_abs)
    val_hashes = _pixel_hashes(train_raw, val_abs)
    client_hashes = _pixel_hashes(train_raw, sorted(all_client))
    excluded = {}
    for h, idx in val_hashes.items():
        reasons = (["duplicate_of_client_training_image"] if h in client_hashes else []) +                   (["duplicate_of_test_image"] if h in test_hashes else [])
        for i in idx:
            if reasons:
                excluded[i] = {"reasons": reasons, "client_train_indices": client_hashes.get(h, []),
                               "test_indices": test_hashes.get(h, [])}
    manifest = [{"absolute_train_index": i, "split": fit_or_eval(seed, i)} for i in sorted(val_abs) if i not in excluded]
    kept_hashes = {h for h, idx in val_hashes.items() if not set(idx) & set(excluded)}
    blob = json.dumps({"seed": seed, "tag": SPLIT_TAG, "version": MANIFEST_VERSION, "probes": manifest}, sort_keys=True).encode()
    return {
        "n_train": len(train_abs), "n_validation_probe_pool": len(val_abs),
        "partition_reconstruction_matches_all_arms": partition_matches,
        "clients_cover_train_subset": sorted(all_client) == sorted(train_abs),
        "probe_index_overlap_with_any_client": len(val & all_client),
        "probe_index_overlap_with_train_subset": len(val & set(train_abs)),
        "probe_exact_pixel_duplicates_in_client_data": sum(1 for h in val_hashes if h in client_hashes),
        "probe_exact_pixel_duplicates_in_test_set": sum(1 for h in val_hashes if h in test_hashes),
        "probe_internal_exact_duplicates": sum(len(v) - 1 for v in val_hashes.values() if len(v) > 1),
        "excluded_probes": {str(k): v for k, v in sorted(excluded.items())},
        "manifest_duplicates_in_client_data": sum(1 for h in kept_hashes if h in client_hashes),
        "manifest_duplicates_in_test_set": sum(1 for h in kept_hashes if h in test_hashes),
        "near_duplicates_checked": False,
        "n_fit": sum(m["split"] == "fit" for m in manifest), "n_eval": sum(m["split"] == "eval" for m in manifest),
        "manifest": manifest, "manifest_sha256": hashlib.sha256(blob).hexdigest(),
    }


def audit_states(root: Path, seed: str) -> dict[str, Any]:
    from src.rev.update_log import read_manifest

    out: dict[str, Any] = {}
    for run in RUNS:
        log = root / "logs_rev" / run.format(s=seed) / "update_log"
        records = read_manifest(log)
        glob = {r["round_id"]: r for r in records if r.get("kind") == "global"}
        sub = {(r["round_id"], r["client_id"]): r for r in records if r.get("kind") == "submitted"}
        clients = sorted({c for _, c in sub})
        missing = [f"global r{r - 1}" for r in ROUNDS if (r - 1) not in glob]
        missing += [f"submitted r{r} {c}" for r in ROUNDS for c in clients if (r, c) not in sub]
        files = [glob[r - 1]["file"] for r in ROUNDS if (r - 1) in glob] + \
                [sub[(r, c)]["file"] for r in ROUNDS for c in clients if (r, c) in sub]
        out[run.format(s=seed)] = {"sealed": (log / "SEAL.json").exists(), "clients": clients, "missing": missing,
                                   "files_present": all((log / f).exists() for f in files), "n_states": len(files)}
    return out


def load_check(root: Path, seed: str) -> dict[str, Any]:
    """One submitted state loads strictly into the architecture and predicts without retraining."""
    import torch

    from src.models.resnet18_cifar import resnet18_cifar
    from src.rev.hashing import load_tensor_state, state_content_sha256
    from src.rev.update_log import read_manifest

    log = root / "logs_rev" / RUNS[0].format(s=seed) / "update_log"
    rec = next(r for r in read_manifest(log) if r.get("kind") == "submitted" and r["round_id"] == 16)
    state = load_tensor_state(log / rec["file"])
    model = resnet18_cifar(num_classes=100)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 32, 32))
    return {"content_sha256_matches": state_content_sha256(state) == rec["content_sha256"],
            "strict_load": True, "output_shape": list(out.shape), "is_full_state_not_delta": "fc.weight" in state
            and any(k.endswith("running_mean") for k in state)}


def run(root: Path, seeds: Sequence[str]) -> dict[str, Any]:
    train_raw, test_raw = _raw(root, "train"), _raw(root, "test")
    test_hashes = _pixel_hashes(test_raw)
    report: dict[str, Any] = {"status": "CLAUDE-REPORTED, DEVELOPMENT ONLY (protocol 16 feasibility gate)",
                              "n_test": len(test_raw), "per_seed": {}}
    for s in seeds:
        rec = audit_seed(root, s, train_raw, test_hashes)
        rec["states"] = audit_states(root, s)
        rec["load_check"] = load_check(root, s)
        report["per_seed"][s] = rec
    gates = {}
    for s, rec in report["per_seed"].items():
        gates[s] = {
            "G1_partition_reconstructed": rec["partition_reconstruction_matches_all_arms"] and rec["clients_cover_train_subset"],
            "G2_probes_disjoint_from_client_data": rec["probe_index_overlap_with_any_client"] == 0
            and rec["manifest_duplicates_in_client_data"] == 0,
            "G3_probes_disjoint_from_final_evaluation": rec["manifest_duplicates_in_test_set"] == 0,
            "G4_states_complete": all(v["sealed"] and not v["missing"] and v["files_present"] for v in rec["states"].values()),
            "G5_predictions_without_retraining": rec["load_check"]["content_sha256_matches"] and rec["load_check"]["is_full_state_not_delta"],
        }
    report["gates"] = gates
    report["all_gates_pass"] = all(all(g.values()) for g in gates.values())
    out = root / "results_rev" / "pilot" / "label_shift"
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_audit.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for s, rec in report["per_seed"].items():
        (out / f"probe_manifest_{s}.json").write_text(
            json.dumps({"seed": s, "tag": SPLIT_TAG, "version": MANIFEST_VERSION, "probes": rec["manifest"]}, sort_keys=True),
            encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe-set feasibility and leakage audit (read-only).")
    parser.add_argument("--seeds", nargs="+", default=["123", "2026"])
    args = parser.parse_args(argv)
    rep = run(Path("."), args.seeds)
    print(json.dumps({"gates": rep["gates"], "all_gates_pass": rep["all_gates_pass"]}, indent=1))
    return 0 if rep["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
