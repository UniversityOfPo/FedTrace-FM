"""Deterministic comparison of two federated runs (Stage B gate evidence; CPU only).

    python -m src.rev.compare_runs --preset R0-vs-EXP016
    python -m src.rev.compare_runs --preset R1-vs-R0
    python -m src.rev.compare_runs --preset S0b-vs-S0a
    python -m src.rev.compare_runs --reference-results DIR --reference-checkpoints DIR --reference-config FILE \
        --candidate-results DIR --candidate-checkpoints DIR --candidate-config FILE [--candidate-log DIR] --out FILE

Everything is read-only. Each artifact gets one finding with a status:

    IDENTICAL   bit-for-bit equal (floats compared by their IEEE-754 bytes, tensors by dtype, shape and bytes)
    NUMERICAL   same structure, different numbers; reports count, max |diff|, max relative diff, first differing
                round / path
    VALUE       same structure, different non-numeric values (identifiers, strings, booleans)
    STRUCTURAL  different keys, columns, rows, shapes or dtypes
    MISSING     artifact absent on one side
    EXCLUDED    deliberately not compared (wall-clock timing, absolute output paths, rendered plots, run.log);
                listed explicitly, never silently dropped
    INFO        recorded for provenance, not a pass/fail item (e.g. recorded git commits)

Categories: CONFIG, PARTITION, RNG_PROXY, HISTORY, TRACE, TRACE_ORDER_DEPENDENT (peer_mean_cosine_similarity and
the naive-baseline fields derived from it; see ``src.rev.replay``), DETECTOR, GLOBAL_STATE, CLIENT_STATE, METRICS,
RUNNER, UPDATE_LOG, PROVENANCE. Bitwise and numerical differences are counted separately.

No RNG state is observable from the NDSS outputs; RNG behaviour is assessed through proxies: the client-selection
sequence (server RNG), the per-round local-training metrics and round-0 update norms (initialisation and
DataLoader order), and the attacker's post-onset update geometry (dedicated attack generator).

Exit code: 0 = no NUMERICAL / STRUCTURAL / MISSING finding; 1 = differences; 2 = invalid invocation.
Interpreting a difference (Level A vs Level B) is a human decision and is not automated here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

TIMING_KEYS = frozenset({
    "started_at_utc", "completed_at_utc", "created_at_utc", "registered_at_utc", "round_started_at_utc",
    "elapsed_seconds", "round_seconds", "cumulative_seconds", "total_experiment_seconds",
})
PATH_KEYS = frozenset({"checkpoint_paths", "output_dir", "checkpoint_dir"})
ORDER_DEPENDENT_FIELDS = ("peer_mean_cosine_similarity", "naive_threshold", "naive_flag", "first_flagged_round")
EXCLUDED_FILES = ("run.log", "communication_round_duration.png", "confusion_matrix.png", "local_training_loss.png",
                  "validation_loss.png", "validation_top1.png", "validation_top5.png")
DIFFERENCE_STATUSES = ("NUMERICAL", "VALUE", "STRUCTURAL", "MISSING")

ARCHIVE_NAME = "cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123"
PRESETS: dict[str, dict[str, str | None]] = {
    "R0-vs-EXP016": {
        "reference_results": f"results/federated/{ARCHIVE_NAME}",
        "reference_checkpoints": f"checkpoints/federated/{ARCHIVE_NAME}",
        "reference_config": f"configs/{ARCHIVE_NAME}.yaml",
        "candidate_results": "results_rev/R0_exp016_repro",
        "candidate_checkpoints": "checkpoints_rev/R0_exp016_repro",
        "candidate_config": "configs/rev/R0_exp016_repro.yaml",
        "candidate_log": None,
    },
    "R1-vs-R0": {
        "reference_results": "results_rev/R0_exp016_repro",
        "reference_checkpoints": "checkpoints_rev/R0_exp016_repro",
        "reference_config": "configs/rev/R0_exp016_repro.yaml",
        "candidate_results": "results_rev/R1_exp016_logged",
        "candidate_checkpoints": "checkpoints_rev/R1_exp016_logged",
        "candidate_config": "results_rev/R1_exp016_logged/resolved_ndss_config.yaml",
        "candidate_log": "logs_rev/R1_exp016_logged/update_log",
    },
    "S0b-vs-S0a": {
        "reference_results": "results_rev/S0a_smoke_ndss",
        "reference_checkpoints": "checkpoints_rev/S0a_smoke_ndss",
        "reference_config": "configs/rev/S0a_smoke_ndss.yaml",
        "candidate_results": "results_rev/S0b_smoke_logged",
        "candidate_checkpoints": "checkpoints_rev/S0b_smoke_logged",
        "candidate_config": "results_rev/S0b_smoke_logged/resolved_ndss_config.yaml",
        "candidate_log": "logs_rev/S0b_smoke_logged/update_log",
    },
}


@dataclass
class Finding:
    category: str
    item: str
    status: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    reference: dict[str, Any]
    candidate: dict[str, Any]
    findings: list[Finding] = field(default_factory=list)

    def add(self, category: str, item: str, status: str, **detail: Any) -> None:
        self.findings.append(Finding(category, item, status, detail))

    @property
    def differences(self) -> list[Finding]:
        return [f for f in self.findings if f.status in DIFFERENCE_STATUSES]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.status] = out.get(f.status, 0) + 1
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"reference": self.reference, "candidate": self.candidate, "counts": self.counts(),
                "bitwise_identical": not self.differences, "findings": [asdict(f) for f in self.findings]}


# ----------------------------------------------------------------------
# Value comparison
# ----------------------------------------------------------------------
@dataclass
class Diff:
    leaves: int = 0
    identical: int = 0
    numerical: int = 0
    value: int = 0
    structural: int = 0
    excluded: list[str] = field(default_factory=list)
    max_abs: float = 0.0
    max_rel: float = 0.0
    examples: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.structural:
            return "STRUCTURAL"
        if self.value:
            return "VALUE"
        return "NUMERICAL" if self.numerical else "IDENTICAL"

    def note(self, path: str) -> None:
        if len(self.examples) < 10:
            self.examples.append(path)

    def numeric(self, a: float, b: float) -> None:
        if math.isfinite(a) and math.isfinite(b):
            d = abs(a - b)
            self.max_abs = max(self.max_abs, d)
            scale = max(abs(a), abs(b))
            if scale > 0:
                self.max_rel = max(self.max_rel, d / scale)

    def detail(self) -> dict[str, Any]:
        return {"leaves": self.leaves, "identical": self.identical, "numerical": self.numerical,
                "value": self.value, "structural": self.structural, "max_abs_diff": self.max_abs, "max_rel_diff": self.max_rel,
                "first_differences": self.examples, "excluded_keys": sorted(set(self.excluded))}


def _tensor_bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    flat_a, flat_b = a.detach().cpu().reshape(-1).contiguous(), b.detach().cpu().reshape(-1).contiguous()
    return flat_a.view(torch.uint8).numpy().tobytes() == flat_b.view(torch.uint8).numpy().tobytes()


def _float_bits_equal(a: float, b: float) -> bool:
    return struct.pack("<d", a) == struct.pack("<d", b)


def compare_values(a: Any, b: Any, path: str = "", diff: Diff | None = None,
                   exclude: frozenset[str] = TIMING_KEYS | PATH_KEYS) -> Diff:
    """Recursive comparison of nested dict/list/tensor/scalar structures."""
    diff = diff or Diff()
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        diff.leaves += 1
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.dtype != b.dtype or a.shape != b.shape:
            diff.structural += 1
            diff.note(path)
        elif _tensor_bits_equal(a, b):
            diff.identical += 1
        else:
            diff.numerical += 1
            diff.note(path)
            if a.is_floating_point():
                d = (a.double() - b.double()).abs()
                diff.max_abs = max(diff.max_abs, float(d.max()))
                scale = torch.maximum(a.double().abs(), b.double().abs())
                nz = scale > 0
                if bool(nz.any()):
                    diff.max_rel = max(diff.max_rel, float((d[nz] / scale[nz]).max()))
        return diff
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for key in sorted(set(a) | set(b), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if str(key) in exclude:
                diff.excluded.append(str(key))
                continue
            if key not in a or key not in b:
                diff.leaves += 1
                diff.structural += 1
                diff.note(sub + " (missing on one side)")
                continue
            compare_values(a[key], b[key], sub, diff, exclude)
        return diff
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diff.leaves += 1
            diff.structural += 1
            diff.note(f"{path} (length {len(a)} != {len(b)})")
            return diff
        for i, (x, y) in enumerate(zip(a, b)):
            compare_values(x, y, f"{path}[{i}]", diff, exclude)
        return diff
    diff.leaves += 1
    if isinstance(a, float) and isinstance(b, float):
        if _float_bits_equal(a, b):
            diff.identical += 1
        else:
            diff.numerical += 1
            diff.note(path)
            diff.numeric(a, b)
    elif (a is None) != (b is None):
        diff.value += 1  # a value (e.g. an alarm round) present on one side only
        diff.note(path)
    elif type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                          and not isinstance(a, bool) and not isinstance(b, bool)):
        diff.structural += 1
        diff.note(f"{path} (type {type(a).__name__} != {type(b).__name__})")
    elif a == b:
        diff.identical += 1
    else:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            diff.numerical += 1
            diff.numeric(float(a), float(b))
        else:
            diff.value += 1
        diff.note(path)
    return diff


# ----------------------------------------------------------------------
# CSV comparison
# ----------------------------------------------------------------------
def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def compare_csv(ref: Path, cand: Path, key: Sequence[str], columns: Sequence[str] | None = None,
                exclude: Sequence[str] = ()) -> tuple[str, dict[str, Any]]:
    """Cell-by-cell comparison keyed by ``key``; cell text equality is bitwise for Python-written floats."""
    ref_cols, ref_rows = _read_csv(ref)
    cand_cols, cand_rows = _read_csv(cand)
    detail: dict[str, Any] = {"excluded_columns": list(exclude)}
    if ref_cols != cand_cols:
        return "STRUCTURAL", {**detail, "reference_columns": ref_cols, "candidate_columns": cand_cols}
    index_ref = {tuple(r[k] for k in key): r for r in ref_rows}
    index_cand = {tuple(r[k] for k in key): r for r in cand_rows}
    if set(index_ref) != set(index_cand) or len(index_ref) != len(ref_rows):
        return "STRUCTURAL", {**detail, "reference_rows": len(ref_rows), "candidate_rows": len(cand_rows)}
    selected = [c for c in (columns or ref_cols) if c not in exclude and c not in key]
    per_column: dict[str, dict[str, Any]] = {}
    status = "IDENTICAL"
    for col in selected:
        diff = Diff()
        first_round = None
        for k in sorted(index_ref, key=lambda t: tuple(int(x) if x.isdigit() else x for x in t)):
            a, b = index_ref[k][col], index_cand[k][col]
            diff.leaves += 1
            if a == b:
                diff.identical += 1
                continue
            fa, fb = _as_float(a), _as_float(b)
            if fa is not None and fb is not None:
                diff.numerical += 1
                diff.numeric(fa, fb)
            else:
                diff.value += 1
            diff.note("/".join(k))
            if first_round is None:
                first_round = k[0]
        if diff.status != "IDENTICAL":
            per_column[col] = {**diff.detail(), "first_differing_key": first_round}
            rank = ("IDENTICAL", "NUMERICAL", "VALUE", "STRUCTURAL")
            status = max(status, diff.status, key=rank.index)
    return status, {**detail, "rows": len(ref_rows), "columns_compared": selected, "differing_columns": per_column}


# ----------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------
def _load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)  # local, trusted NDSS checkpoints


def _check_file(report: Report, category: str, item: str, ref: Path, cand: Path) -> bool:
    missing = [side for side, p in (("reference", ref), ("candidate", cand)) if not p.exists()]
    if missing:
        report.add(category, item, "MISSING", missing_on=missing)
        return False
    return True


def check_config(report: Report, ref_cfg: Path, cand_cfg: Path) -> None:
    from src.experiments.run_federated_cifar100 import load_federated_experiment_config

    if not _check_file(report, "CONFIG", "experiment configuration", ref_cfg, cand_cfg):
        return
    plain = lambda cfg: {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in vars(cfg).items()}
    diff = compare_values(plain(load_federated_experiment_config(ref_cfg)),
                          plain(load_federated_experiment_config(cand_cfg)),
                          exclude=frozenset({"output_dir", "checkpoint_dir"}))
    report.add("CONFIG", "experiment configuration (parsed; output locations excluded)", diff.status, **diff.detail())


def check_partition(report: Report, ref: Path, cand: Path) -> None:
    for name in ("client_partition_indices.json", "client_partition_summary.csv"):
        if not _check_file(report, "PARTITION", name, ref / name, cand / name):
            continue
        same_bytes = (ref / name).read_bytes() == (cand / name).read_bytes()
        if name.endswith(".json"):
            diff = compare_values(json.loads((ref / name).read_text(encoding="utf-8")),
                                  json.loads((cand / name).read_text(encoding="utf-8")))
            report.add("PARTITION", name, diff.status, file_bytes_identical=same_bytes, **diff.detail())
        else:
            report.add("PARTITION", name, "IDENTICAL" if same_bytes else "VALUE", file_bytes_identical=same_bytes)


def check_history(report: Report, ref: Path, cand: Path) -> None:
    name = "round_history.csv"
    if not _check_file(report, "HISTORY", name, ref / name, cand / name):
        return
    report.add("HISTORY", "round_history.csv timing columns", "EXCLUDED", columns=["round_seconds", "cumulative_seconds"])
    status, detail = compare_csv(ref / name, cand / name, key=("round_id",),
                                 exclude=("round_seconds", "cumulative_seconds"))
    report.add("HISTORY", name, status, **detail)
    s_status, s_detail = compare_csv(ref / name, cand / name, key=("round_id",),
                                     columns=("selected_client_ids", "successful_client_ids"))
    report.add("RNG_PROXY", "client-selection sequence (server RNG)", s_status, **s_detail)
    t_status, t_detail = compare_csv(ref / name, cand / name, key=("round_id",),
                                     columns=("train_loss", "train_top1", "train_top5"))
    report.add("RNG_PROXY", "local-training trajectory (initialisation + DataLoader order)", t_status, **t_detail)


def check_trace(report: Report, ref: Path, cand: Path) -> None:
    name = "client_round_trace.csv"
    if not _check_file(report, "TRACE", name, ref / name, cand / name):
        return
    cols, _ = _read_csv(ref / name)
    exact = [c for c in cols if c not in ORDER_DEPENDENT_FIELDS]
    status, detail = compare_csv(ref / name, cand / name, key=("round_id", "client_id"), columns=exact)
    report.add("TRACE", "client_round_trace.csv (order-independent fields)", status, **detail)
    dep = [c for c in ORDER_DEPENDENT_FIELDS if c in cols]
    status, detail = compare_csv(ref / name, cand / name, key=("round_id", "client_id"), columns=dep)
    report.add("TRACE_ORDER_DEPENDENT", "peer_mean_cosine_similarity and naive-baseline fields", status,
               note="fp64 peer mean depends on Python set order (PYTHONHASHSEED); not used by FedTrace v3", **detail)
    round0_status, round0_detail = compare_csv(ref / name, cand / name, key=("round_id", "client_id"),
                                               columns=("update_l2_norm",))
    first = round0_detail.get("differing_columns", {}).get("update_l2_norm", {}).get("first_differing_key")
    report.add("RNG_PROXY", "client update norms (first differing round)", round0_status, first_differing_round=first)


def check_detector(report: Report, ref: Path, cand: Path) -> None:
    from src.analysis.fedtrace_detector import compute_all_scores_v3, evaluate_detector_v3, load_trace_rows
    from src.rev.online import online_detect_v3

    name = "client_round_trace.csv"
    if not ((ref / name).exists() and (cand / name).exists()):
        report.add("DETECTOR", "FedTrace v3 outputs", "MISSING")
        return
    rows_a, rows_b = load_trace_rows(ref / name), load_trace_rows(cand / name)

    def events(rows):
        offline = evaluate_detector_v3(rows, "H_fedtrace_v3", ("m", "t_res"))
        online = online_detect_v3(rows)
        return {
            "offline_predicted": offline.predicted_client,
            "offline": {c: [r.first_anomalous_round, r.confirmation_round, r.onset_estimate]
                        for c, r in sorted(offline.per_client.items())},
            "online_attributed": online.attributed_client,
            "online": {c: [e.alarm_round, e.estimated_onset, e.actionable_round] for c, e in online.per_client.items()},
        }

    diff = compare_values(events(rows_a), events(rows_b))
    report.add("DETECTOR", "FedTrace v3 decisions (offline and online)", diff.status, **diff.detail())
    scores = lambda rows: {f"{s.round_id}/{s.client_id}": [s.z_m, s.z_t_res] for s in compute_all_scores_v3(rows)}
    diff = compare_values(scores(rows_a), scores(rows_b))
    report.add("DETECTOR", "FedTrace v3 component scores z_m, z_t_res", diff.status, **diff.detail())


def check_states(report: Report, ref_ck: Path, cand_ck: Path) -> None:
    final = "final_global_model.pt"
    if _check_file(report, "GLOBAL_STATE", final, ref_ck / final, cand_ck / final):
        a, b = _load_pt(ref_ck / final), _load_pt(cand_ck / final)
        diff = compare_values(a["model_state_dict"], b["model_state_dict"])
        report.add("GLOBAL_STATE", "final global model tensors", diff.status, **diff.detail())
        meta = compare_values({k: str(v) for k, v in a.items() if k != "model_state_dict"},
                              {k: str(v) for k, v in b.items() if k != "model_state_dict"})
        report.add("PROVENANCE", "final checkpoint metadata (experiment id, seed, torch/CUDA)", meta.status, **meta.detail())
    server = "runner_latest_server.pt"
    if _check_file(report, "GLOBAL_STATE", server, ref_ck / server, cand_ck / server):
        diff = compare_values(_load_pt(ref_ck / server), _load_pt(cand_ck / server))
        report.add("GLOBAL_STATE", "server checkpoint (global state, registrations, round records; timing excluded)",
                   diff.status, **diff.detail())
    runner = "runner_latest.pt"
    if _check_file(report, "RUNNER", runner, ref_ck / runner, cand_ck / runner):
        diff = compare_values(_load_pt(ref_ck / runner), _load_pt(cand_ck / runner))
        report.add("RUNNER", "runner checkpoint (config, round history; timing excluded)", diff.status, **diff.detail())
    names = sorted(p.name for p in ref_ck.glob("runner_latest_client_*.pt")) if ref_ck.exists() else []
    if not names:
        report.add("CLIENT_STATE", "client checkpoints", "MISSING", missing_on=["reference"])
    for name in names:
        if _check_file(report, "CLIENT_STATE", name, ref_ck / name, cand_ck / name):
            diff = compare_values(_load_pt(ref_ck / name), _load_pt(cand_ck / name))
            report.add("CLIENT_STATE", f"{name} (last local model, optimizer state)", diff.status, **diff.detail())


def check_metrics(report: Report, ref: Path, cand: Path) -> None:
    for name in ("final_test_metrics.json", "per_class_accuracy.csv"):
        if not _check_file(report, "METRICS", name, ref / name, cand / name):
            continue
        if name.endswith(".json"):
            diff = compare_values(json.loads((ref / name).read_text(encoding="utf-8")),
                                  json.loads((cand / name).read_text(encoding="utf-8")))
            report.add("METRICS", name, diff.status, **diff.detail())
        else:
            status, detail = compare_csv(ref / name, cand / name, key=("class_index",))
            report.add("METRICS", name, status, **detail)
    name = "summary.json"
    if _check_file(report, "METRICS", name, ref / name, cand / name):
        a = json.loads((ref / name).read_text(encoding="utf-8"))
        b = json.loads((cand / name).read_text(encoding="utf-8"))
        report.add("PROVENANCE", "recorded git commit", "INFO", reference=a.get("git_commit"), candidate=b.get("git_commit"))
        diff = compare_values(a, b, exclude=TIMING_KEYS | PATH_KEYS | {"git_commit"})
        report.add("METRICS", "summary.json (timing, output paths and git commit excluded)", diff.status, **diff.detail())
    report.add("METRICS", "rendered plots and run.log", "EXCLUDED", files=list(EXCLUDED_FILES),
               reason="contain wall-clock timing or renderer metadata; their source data is compared above")


def replay_inputs(cand_cfg: Path, cand: Path) -> tuple[Any, frozenset[str], list[str]]:
    """Attack config, trainable keys and client ids exactly as the NDSS script derives them."""
    from src.attacks.traceability import AttackConfig
    from src.experiments.run_federated_cifar100 import load_federated_experiment_config
    from src.models.resnet18_cifar import resnet18_cifar

    config = load_federated_experiment_config(cand_cfg)
    attack = AttackConfig(attack_type=config.attack_type, malicious_client_id=config.malicious_client_id,
                          attack_start_round=config.attack_start_round, attack_strength_rho=config.attack_strength_rho,
                          directional_target_cosine=config.directional_target_cosine,
                          attack_active_windows=config.attack_active_windows)
    trainable = frozenset(n for n, _ in resnet18_cifar(num_classes=config.num_classes).named_parameters())
    clients = sorted(json.loads((cand / "client_partition_indices.json").read_text(encoding="utf-8")))
    return attack, trainable, clients


def check_update_log(report: Report, log: Path, ref: Path, ref_ck: Path, cand: Path, cand_ck: Path,
                     attack: Any, trainable: frozenset[str], clients: Sequence[str]) -> None:
    from src.rev.hashing import load_tensor_state
    from src.rev.replay import compare_trace_rows, recompute_trace_rows, replay_aggregation, verify_update_log
    from src.rev.update_log import read_manifest
    from src.analysis.fedtrace_detector import load_trace_rows

    if not log.exists():
        report.add("UPDATE_LOG", "update log", "MISSING", missing_on=["candidate"])
        return
    verification = verify_update_log(log)
    report.add("UPDATE_LOG", "log verification (seal, hashes, chain)", "IDENTICAL" if verification.ok else "STRUCTURAL",
               records=verification.records, problems=verification.problems)
    if not verification.ok:
        report.add("UPDATE_LOG", "replay from log", "STRUCTURAL",
                   reason="not run: the log failed verification, so its contents cannot be trusted")
        return
    replay = replay_aggregation(log)
    bad = [r for r, ok in replay.items() if not ok]
    report.add("UPDATE_LOG", "re-aggregation of logged submissions reproduces logged globals",
               "IDENTICAL" if replay and not bad else "NUMERICAL", rounds=len(replay), failing_rounds=bad)
    records = read_manifest(log)
    globals_ = sorted((r for r in records if r.get("kind") == "global"), key=lambda r: r["round_id"])
    if globals_:
        last = load_tensor_state(log / globals_[-1]["file"])
        for side, ck in (("candidate", cand_ck), ("reference", ref_ck)):
            final = ck / "final_global_model.pt"
            if final.exists():
                diff = compare_values(last, _load_pt(final)["model_state_dict"])
                report.add("UPDATE_LOG", f"last logged global == {side} final model", diff.status, **diff.detail())
        provenance = cand / "rev_provenance.json"
        if provenance.exists():
            init_hash = json.loads(provenance.read_text(encoding="utf-8")).get("post_run", {}).get("model_init_content_sha256")
            report.add("UPDATE_LOG", "logged initial global == runner initialisation fingerprint",
                       "IDENTICAL" if init_hash == globals_[0]["content_sha256"] else "NUMERICAL",
                       provenance=init_hash, logged=globals_[0]["content_sha256"])
    replayed = recompute_trace_rows(log, clients, trainable, attack)
    for side, base in (("candidate", cand), ("reference", ref)):
        trace = base / "client_round_trace.csv"
        if not trace.exists():
            continue
        live = load_trace_rows(trace)
        replayed_typed = _retype(replayed, live)
        exact = compare_trace_rows(live, replayed_typed, ignore_fields=ORDER_DEPENDENT_FIELDS)
        report.add("UPDATE_LOG", f"trace recomputed from log == {side} trace (order-independent fields)",
                   "IDENTICAL" if not exact else "NUMERICAL", mismatches=exact[:10], n_mismatches=len(exact))
        dep = compare_trace_rows(live, replayed_typed,
                                 ignore_fields=[k for k in live[0] if k not in ORDER_DEPENDENT_FIELDS])
        report.add("TRACE_ORDER_DEPENDENT", f"trace recomputed from log (sorted peers) vs {side} trace",
                   "IDENTICAL" if not dep else "NUMERICAL", mismatches=dep[:10], n_mismatches=len(dep))
    report.add("RNG_PROXY", "logged pre-round globals form an unbroken chain from the initialisation",
               "IDENTICAL" if verification.ok else "STRUCTURAL")


def _retype(replayed: Sequence[Mapping[str, Any]], live: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Round-trip recomputed rows through the NDSS CSV writer/reader so they compare like archived rows."""
    import tempfile

    from src.analysis.fedtrace_detector import load_trace_rows
    from src.attacks.traceability import save_client_round_trace_csv

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "replayed.csv"
        save_client_round_trace_csv([dict(r) for r in replayed], path)
        return load_trace_rows(path)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def compare(paths: Mapping[str, str | None], repo_root: Path = Path(".")) -> Report:
    p = {k: (repo_root / v if v else None) for k, v in paths.items()}
    report = Report(reference={k: v for k, v in paths.items() if k.startswith("reference")},
                    candidate={k: v for k, v in paths.items() if k.startswith("candidate")})
    ref, cand = p["reference_results"], p["candidate_results"]
    ref_ck, cand_ck = p["reference_checkpoints"], p["candidate_checkpoints"]
    check_config(report, p["reference_config"], p["candidate_config"])
    check_partition(report, ref, cand)
    check_history(report, ref, cand)
    check_trace(report, ref, cand)
    check_detector(report, ref, cand)
    check_states(report, ref_ck, cand_ck)
    check_metrics(report, ref, cand)
    if p.get("candidate_log") is not None:
        try:
            attack, trainable, clients = replay_inputs(p["candidate_config"], cand)
        except (OSError, ValueError, KeyError) as exc:
            report.add("UPDATE_LOG", "replay inputs (config, partition)", "MISSING", error=type(exc).__name__)
        else:
            check_update_log(report, p["candidate_log"], ref, ref_ck, cand, cand_ck, attack, trainable, clients)
    else:
        report.add("UPDATE_LOG", "update log", "EXCLUDED", reason="no update log for this pair")
    return report


def render(report: Report) -> str:
    lines = [f"{'CATEGORY':22s} {'STATUS':11s} ITEM"]
    for f in report.findings:
        extra = ""
        if f.status == "NUMERICAL":
            extra = f"  (max|d|={f.detail.get('max_abs_diff', '?')}, max rel={f.detail.get('max_rel_diff', '?')})"
        lines.append(f"{f.category:22s} {f.status:11s} {f.item}{extra}")
    lines.append(f"counts: {report.counts()}  bitwise_identical={not report.differences}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic comparison of two federated runs (read-only).")
    parser.add_argument("--preset", choices=sorted(PRESETS))
    for side in ("reference", "candidate"):
        for kind in ("results", "checkpoints", "config"):
            parser.add_argument(f"--{side}-{kind}")
    parser.add_argument("--candidate-log")
    parser.add_argument("--out", help="JSON report path (default: results_rev/comparisons/<preset>.json)")
    args = parser.parse_args(argv)
    if args.preset:
        paths = dict(PRESETS[args.preset])
    else:
        paths = {f"{s}_{k}": getattr(args, f"{s}_{k}") for s in ("reference", "candidate")
                 for k in ("results", "checkpoints", "config")}
        paths["candidate_log"] = args.candidate_log
        if any(v is None for k, v in paths.items() if k != "candidate_log"):
            parser.error("without --preset, all reference/candidate results, checkpoints and config are required")
    report = compare(paths)
    out = Path(args.out) if args.out else Path("results_rev") / "comparisons" / f"{args.preset or 'custom'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(render(report))
    print(f"report: {out.as_posix()}")
    return 1 if report.differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
