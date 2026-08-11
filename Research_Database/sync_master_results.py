"""Synchronize Research_Database/FedTrace_Master_Results.xlsx from existing
experiment artifacts.

Pipeline:
    results/**/summary.json (+ final_test_metrics.json, multikrum_diagnostics.csv,
    client_round_trace.csv where present) + configs/*.yaml (dirichlet_alpha only)
    + analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv (index of which EXP IDs exist,
    their category/purpose, and pointers for qualitative FedTrace fields)
        -> this script
        -> Research_Database/FedTrace_Master_Results.xlsx

Design principles (see analysis/FEDTRACE_MASTER_RESULTS_SYNC_REPORT.md for the
full audit this script's design was based on):

  * Read-only against every experiment artifact. This script never modifies
    anything under results/, checkpoints/, configs/, src/, or tests/, and never
    re-runs any experiment, attack, or FedTrace/Multi-Krum analysis.
  * Numeric ground truth (loss/top1/top5/macro metrics/hyperparameters/attack
    metadata) is read FRESH from each experiment's own summary.json every run
    -- never hard-coded, never copied from analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv
    without a cross-check.
  * FedTrace-attribution and Multi-Krum-proxy fields that require judgment
    already made in a prior, human-reviewed analysis task (e.g. "which signal
    was dominant", "was attribution correct") are NOT re-derived by parsing
    prose at runtime (fragile, unauditable). They are instead a small, explicit,
    clearly-cited constant table (see FEDTRACE_ATTRIBUTION_FACTS /
    ABLATION_FACTS below), each entry traceable to the exact analysis document
    and section it was verified against. Multi-Krum per-run rejection summary
    statistics (rejection rate, switch round, proxy attribution) ARE computed
    fresh from each run's own multikrum_diagnostics.csv, because that artifact
    already exposes the exact raw selected/rejected data needed -- applying the
    already-documented, mechanical P_krum=3 persistence rule to it is a
    deterministic derivation, not a new judgment call.
  * A missing value is written as the literal string "NOT RECORDED" (data that
    should exist somewhere but was not safely recoverable in this pass) or
    "NOT APPLICABLE" (data that does not apply to this experiment's type) --
    never fabricated, never silently left as an empty/blank cell.
  * The workbook's eight pre-existing sheets (Experiment Summary,
    Hyperparameters, Hardware, Federated Results, Attack Results, Attribution
    Results, Paper Tables, Notes) are loaded into memory and never modified in
    any way -- this script only adds NEW sheets alongside them. On every run,
    this script's own managed sheets (see MANAGED_SHEETS) are deleted and
    rebuilt from scratch before being re-added, which is what makes the sync
    idempotent (no duplicate rows can ever accumulate) without touching the
    legacy sheets at all.
  * A timestamped backup of the workbook is written to
    Research_Database/backup/ before every sync, and the new workbook is
    written to a temp file and atomically replaced (os.replace) over the
    target path.

Usage:
    python Research_Database/sync_master_results.py [--dry-run]

--dry-run performs every read/derive/validate step and prints the summary,
but does not back up or write the workbook.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    import openpyxl
except ImportError as exc:  # pragma: no cover - environment guard, not test-covered
    raise SystemExit(
        "openpyxl is required to run this script (pip install openpyxl). "
        "It was not present in this project's requirements.txt as of this "
        "script's authoring; add it there if not already done."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"
REGISTRY_CSV = REPO_ROOT / "analysis" / "FEDTRACE_EXPERIMENT_REGISTRY.csv"
WORKBOOK_PATH = REPO_ROOT / "Research_Database" / "FedTrace_Master_Results.xlsx"
BACKUP_DIR = REPO_ROOT / "Research_Database" / "backup"

NOT_RECORDED = "NOT RECORDED"
NOT_APPLICABLE = "NOT APPLICABLE"

# Sheets this script owns outright: deleted and rebuilt from scratch on every
# run (idempotent by construction), never partially edited in place.
MANAGED_SHEETS = (
    "Experiment Registry",
    "FedTrace Attribution",
    "MultiKrum Diagnostics Summary",
    "Ablation Detector Results",
    "Reproducibility Provenance",
    "Sync Metadata",
)

# Sheets present in the original hand-authored workbook. Verified present by
# direct inspection (openpyxl) before this script was written; never deleted,
# never written to, by any code path in this script.
LEGACY_SHEETS = (
    "Experiment Summary",
    "Hyperparameters",
    "Hardware",
    "Federated Results",
    "Attack Results",
    "Attribution Results",
    "Paper Tables",
    "Notes",
)


class SyncError(RuntimeError):
    """Raised for malformed/missing required data -- this script fails loudly
    rather than silently producing an incomplete or incorrect workbook row."""


# ---------------------------------------------------------------------------
# Manually verified, explicitly cited constant tables for fields that require
# judgment already exercised in a prior, human-reviewed analysis task. Each
# entry is traceable to an exact source document/section re-read (not merely
# recalled) while this script was authored. These are NOT derived from
# artifacts at runtime -- see the module docstring for why.
# ---------------------------------------------------------------------------

FEDTRACE_ATTRIBUTION_FACTS: dict[str, list[dict[str, Any]]] = {
    # EXP_014: development pilot 1 (Gaussian, seed 42). Source:
    # analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md Section 8.
    "EXP_014": [
        {
            "fedtrace_version": "v1 (development; magnitude-only -- t-signal "
            "not evaluable, self_prev_round_cosine absent from this trace's "
            "14-field schema)",
            "predicted_client": "client_002",
            "true_client": "client_002",
            "attribution_correct": True,
            "first_anomaly_round": 20,
            "confirmation_round": 22,
            "onset_estimate": 20,
            "true_onset": 20,
            "onset_error": 0,
            "detection_delay": 2,
            "honest_persistent_fp_count": 0,
            "dominant_signal": "z_m only (t not evaluable this trace)",
            "threshold": 3.0,
            "persistence_p": 3,
            "evaluable": "PARTIAL (magnitude only; temporal signal absent from this trace's schema)",
            "source_analysis_document": "analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md Section 8",
        }
    ],
    # EXP_015: development pilot 2 (directional, seed 42), the central
    # diagnostic run across all three FedTrace design iterations. Source:
    # analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md Sections 8-9;
    # analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md Sections (attribution table);
    # analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md.
    "EXP_015": [
        {
            "fedtrace_version": "v1 (development)",
            "predicted_client": "client_002",
            "true_client": "client_002",
            "attribution_correct": True,
            "first_anomaly_round": NOT_RECORDED,
            "confirmation_round": 22,
            "onset_estimate": 20,
            "true_onset": 20,
            "onset_error": 0,
            "detection_delay": 2,
            "honest_persistent_fp_count": 3,
            "dominant_signal": "z_t (v1's static pre-attack calibration; non-stationarity flaw)",
            "threshold": 3.0,
            "persistence_p": 3,
            "evaluable": True,
            "notes": "3 honest false positives despite correct attribution: "
            "client_004 onset 41/confirm 43; client_003 onset 45/confirm 47 "
            "(isolated break at round 44); client_001 onset 46/confirm 48. "
            "client_000 never anomalous under v1 (max z~2.98, round 49). "
            "Attribution survived only because client_002 confirmed first (round 22).",
            "source_analysis_document": "analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md Section 8-9",
        },
        {
            "fedtrace_version": "v2 (development)",
            "predicted_client": "client_000",
            "true_client": "client_002",
            "attribution_correct": False,
            "first_anomaly_round": 9,
            "confirmation_round": 11,
            "onset_estimate": 9,
            "true_onset": 20,
            "onset_error": NOT_APPLICABLE,
            "detection_delay": NOT_APPLICABLE,
            "honest_persistent_fp_count": 1,
            "dominant_signal": "z_t_rel (v2's same-round cross-sectional MAD scale; small-sample instability flaw)",
            "threshold": 3.0,
            "persistence_p": 3,
            "evaluable": True,
            "notes": "ATTRIBUTION-BREAKING FAILURE: client_000 confirmed "
            "persistent at round 11 (onset estimate 9), entirely before the "
            "true attack begins at round 20. client_002 itself is separately "
            "detected correctly (onset 20, confirm 22) but is not the earliest "
            "confirmed client, so the detector's predicted_client is the wrong "
            "client_000. Onset error/delay are not meaningful against the "
            "wrong predicted client and are marked NOT APPLICABLE, not computed "
            "against client_002.",
            "source_analysis_document": "analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md",
        },
        {
            "fedtrace_version": "v3 (frozen; this is the version used for all "
            "subsequent Stage B / generalization runs)",
            "predicted_client": "client_002",
            "true_client": "client_002",
            "attribution_correct": True,
            "first_anomaly_round": 20,
            "confirmation_round": 22,
            "onset_estimate": 20,
            "true_onset": 20,
            "onset_error": 0,
            "detection_delay": 2,
            "honest_persistent_fp_count": 0,
            "dominant_signal": "z_t_res (30/30 post-attack rounds, exhaustively checked)",
            "threshold": 3.0,
            "persistence_p": 3,
            "evaluable": True,
            "notes": "Zero honest persistent false positives anywhere, at any "
            "round, for any client -- exhaustive 49-round x 5-client check.",
            "source_analysis_document": "analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md",
        },
    ],
    # EXP_016-019: Stage B independent validation, frozen v3, unmodified.
    # Source: analysis/EXP_01{6,7,8,9}_STAGE_B_*.md +
    # analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md Sections 5-11.
    "EXP_016": [{
        "fedtrace_version": "v3 (frozen)", "predicted_client": "client_002",
        "true_client": "client_002", "attribution_correct": True,
        "first_anomaly_round": 20, "confirmation_round": 22, "onset_estimate": 20,
        "true_onset": 20, "onset_error": 0, "detection_delay": 2,
        "honest_persistent_fp_count": 0, "dominant_signal": "z_m (30/30, 100%)",
        "threshold": 3.0, "persistence_p": 3, "evaluable": True,
        "source_analysis_document": "analysis/EXP_016_STAGE_B_GAUSSIAN_SEED123_VALIDATION.md; FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md Section 6",
    }],
    "EXP_017": [{
        "fedtrace_version": "v3 (frozen)", "predicted_client": "client_002",
        "true_client": "client_002", "attribution_correct": True,
        "first_anomaly_round": 20, "confirmation_round": 22, "onset_estimate": 20,
        "true_onset": 20, "onset_error": 0, "detection_delay": 2,
        "honest_persistent_fp_count": 0,
        "dominant_signal": "mixed (z_m 10/30, z_t_res 20/30) -- genuine seed-dependent difference",
        "threshold": 3.0, "persistence_p": 3, "evaluable": True,
        "source_analysis_document": "analysis/EXP_017_STAGE_B_GAUSSIAN_SEED2026_VALIDATION.md; FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md Section 6",
    }],
    "EXP_018": [{
        "fedtrace_version": "v3 (frozen)", "predicted_client": "client_002",
        "true_client": "client_002", "attribution_correct": True,
        "first_anomaly_round": 20, "confirmation_round": 22, "onset_estimate": 20,
        "true_onset": 20, "onset_error": 0, "detection_delay": 2,
        "honest_persistent_fp_count": 0, "dominant_signal": "z_t_res (30/30, 100%)",
        "threshold": 3.0, "persistence_p": 3, "evaluable": True,
        "source_analysis_document": "analysis/EXP_018_STAGE_B_DIRECTIONAL_SEED123_VALIDATION.md Sections 4-15",
    }],
    "EXP_019": [{
        "fedtrace_version": "v3 (frozen)", "predicted_client": "client_002",
        "true_client": "client_002", "attribution_correct": True,
        "first_anomaly_round": 20, "confirmation_round": 22, "onset_estimate": 20,
        "true_onset": 20, "onset_error": 0, "detection_delay": 2,
        "honest_persistent_fp_count": 0, "dominant_signal": "z_t_res (30/30, 100%)",
        "threshold": 3.0, "persistence_p": 3, "evaluable": True,
        "source_analysis_document": "analysis/EXP_019_STAGE_B_DIRECTIONAL_SEED2026_VALIDATION.md; FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md Section 7",
    }],
    # EXP_026/EXP_027: generalization gate. Directly computed by this same
    # author, in this same overall session, via the unmodified
    # src/analysis/fedtrace_detector.py module -- source documents contain the
    # exact figures transcribed here.
    "EXP_026": [{
        "fedtrace_version": "v3 (frozen; unmodified)", "predicted_client": "client_004",
        "true_client": "client_004", "attribution_correct": True,
        "first_anomaly_round": 20, "confirmation_round": 22, "onset_estimate": 20,
        "true_onset": 20, "onset_error": 0, "detection_delay": 2,
        "honest_persistent_fp_count": 0, "dominant_signal": "z_t_res (30/30, 100%); z_m never confirmed at all in this run",
        "threshold": 3.0, "persistence_p": 3, "evaluable": True,
        "source_analysis_document": "analysis/EXP_026_ATTACKER_IDENTITY_GENERALIZATION_ANALYSIS.md Sections 3-4",
    }],
    "EXP_027": [{
        "fedtrace_version": "v3 (frozen; unmodified)", "predicted_client": "client_002",
        "true_client": "client_002", "attribution_correct": True,
        "first_anomaly_round": 30, "confirmation_round": 32, "onset_estimate": 30,
        "true_onset": 30, "onset_error": 0, "detection_delay": 2,
        "honest_persistent_fp_count": 0, "dominant_signal": "z_t_res (20/20, 100%)",
        "threshold": 3.0, "persistence_p": 3, "evaluable": True,
        "source_analysis_document": "analysis/EXP_027_ONSET_GENERALIZATION_ANALYSIS.md Sections 3-4",
    }],
}

# Ablation / detector comparison rows (magnitude-only vs temporal-residual-only
# vs full v3), for the runs where a clean, single-version v3 comparison exists.
# Source: analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md Section 12;
# analysis/EXP_026_ATTACKER_IDENTITY_GENERALIZATION_ANALYSIS.md Section 5;
# analysis/EXP_027_ONSET_GENERALIZATION_ANALYSIS.md Section 5.
ABLATION_FACTS: dict[str, list[dict[str, Any]]] = {
    "EXP_016": [
        {"detector": "A_magnitude_only", "onset_error": 0, "detection_delay": 2, "notes": "matches full v3"},
        {"detector": "E_temporal_residual_only", "onset_error": 0, "detection_delay": 2, "notes": "matches full v3"},
        {"detector": "H_fedtrace_v3_full", "onset_error": 0, "detection_delay": 2, "notes": "identical to magnitude-only here"},
    ],
    "EXP_017": [
        {"detector": "A_magnitude_only", "onset_error": 0, "detection_delay": 2, "notes": "matches full v3"},
        {"detector": "E_temporal_residual_only", "onset_error": 1, "detection_delay": 3, "notes": "lags magnitude-only by 1 round this seed"},
        {"detector": "H_fedtrace_v3_full", "onset_error": 0, "detection_delay": 2, "notes": "matches magnitude-only (the faster of the two here)"},
    ],
    "EXP_018": [
        {"detector": "A_magnitude_only", "onset_error": 27, "detection_delay": 29, "notes": "far outside pre-registered criteria; late honest-counterfactual drift, not attack-driven"},
        {"detector": "E_temporal_residual_only", "onset_error": 0, "detection_delay": 2, "notes": "matches full v3"},
        {"detector": "H_fedtrace_v3_full", "onset_error": 0, "detection_delay": 2, "notes": "identical to temporal-residual-only"},
    ],
    "EXP_019": [
        {"detector": "A_magnitude_only", "onset_error": 24, "detection_delay": 26, "notes": "far outside pre-registered criteria"},
        {"detector": "E_temporal_residual_only", "onset_error": 0, "detection_delay": 2, "notes": "matches full v3"},
        {"detector": "H_fedtrace_v3_full", "onset_error": 0, "detection_delay": 2, "notes": "identical to temporal-residual-only"},
    ],
    "EXP_026": [
        {"detector": "A_magnitude_only", "onset_error": NOT_APPLICABLE, "detection_delay": NOT_APPLICABLE, "notes": "never confirms any client at all in this run (max post-attack z_m=1.91)"},
        {"detector": "E_temporal_residual_only", "onset_error": 0, "detection_delay": 2, "notes": "matches full v3"},
        {"detector": "H_fedtrace_v3_full", "onset_error": 0, "detection_delay": 2, "notes": "identical to temporal-residual-only"},
    ],
    "EXP_027": [
        {"detector": "A_magnitude_only", "onset_error": 17, "detection_delay": 19, "notes": "attribution correct but far outside pre-registered criteria"},
        {"detector": "E_temporal_residual_only", "onset_error": 0, "detection_delay": 2, "notes": "matches full v3"},
        {"detector": "H_fedtrace_v3_full", "onset_error": 0, "detection_delay": 2, "notes": "identical to temporal-residual-only"},
    ],
}

MULTIKRUM_PERSISTENCE_P = 3  # P_krum, as used throughout the Multi-Krum analysis series


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass
class RegistryRow:
    experiment_id: str
    experiment_name: str
    dataset: str
    model: str
    aggregation: str
    seed: str
    num_clients: str
    partition_type: str
    dirichlet_alpha: str
    attack_present: str
    attack_family: str
    malicious_client: str
    attack_onset_round: str
    attack_severity: str
    purpose: str
    paired_control: str
    status: str


def load_registry() -> list[RegistryRow]:
    """Load analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv, reject duplicate IDs,
    return rows sorted numerically by the integer suffix of Experiment_ID."""
    if not REGISTRY_CSV.exists():
        raise SyncError(f"Required index file not found: {REGISTRY_CSV}")

    rows: list[RegistryRow] = []
    seen: set[str] = set()
    with REGISTRY_CSV.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            eid = raw["Experiment_ID"].strip()
            if eid in seen:
                raise SyncError(f"Duplicate Experiment_ID in registry CSV: {eid}")
            seen.add(eid)
            rows.append(
                RegistryRow(
                    experiment_id=eid,
                    experiment_name=raw["Experiment_Name"].strip(),
                    dataset=raw["Dataset"].strip(),
                    model=raw["Model"].strip(),
                    aggregation=raw["Aggregation"].strip(),
                    seed=raw["Seed"].strip(),
                    num_clients=raw["Num_Clients"].strip(),
                    partition_type=raw["Partition_Type"].strip(),
                    dirichlet_alpha=raw["Dirichlet_Alpha"].strip(),
                    attack_present=raw["Attack_Present"].strip(),
                    attack_family=raw["Attack_Family"].strip(),
                    malicious_client=raw["Malicious_Client"].strip(),
                    attack_onset_round=raw["Attack_Onset_Round"].strip(),
                    attack_severity=raw["Attack_Severity_Or_Target_Cosine"].strip(),
                    purpose=raw["Purpose"].strip(),
                    paired_control=raw["Paired_Control"].strip(),
                    status=raw["Status"].strip(),
                )
            )

    def sort_key(row: RegistryRow) -> int:
        try:
            return int(row.experiment_id.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise SyncError(f"Cannot parse numeric ID from {row.experiment_id!r}") from exc

    rows.sort(key=sort_key)
    return rows


def discover_result_dirs() -> dict[str, Path]:
    """Scan every results/**/summary.json exactly once; return {exp_id: result_dir}.

    Keyed by each artifact's own embedded 'experiment_id' field, with exactly
    one documented, auditable exception (EXP_001's centralized baseline, whose
    summary.json predates that field being written). Smoke-test artifacts
    (e.g. EXP_002_SMOKE, or the centralized smoke run with no embedded ID at
    all) are deliberately excluded -- they were never part of the numbered
    EXP_001-EXP_027 sequence.
    """
    mapping: dict[str, Path] = {}
    for summary_path in sorted(RESULTS_DIR.rglob("summary.json")):
        with summary_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        exp_id = data.get("experiment_id")
        if exp_id is None:
            if summary_path.parent.name == "cifar100_resnet18" and "centralized" in str(summary_path.parent):
                exp_id = "EXP_001"
            else:
                continue
        if exp_id in mapping:
            raise SyncError(
                f"Duplicate experiment_id {exp_id!r} found in both "
                f"{mapping[exp_id]} and {summary_path.parent}"
            )
        mapping[exp_id] = summary_path.parent
    return mapping


# ---------------------------------------------------------------------------
# Per-experiment field extraction (fresh from artifacts every run)
# ---------------------------------------------------------------------------


def _get(d: dict, *keys, default=NOT_RECORDED):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def load_dirichlet_alpha(experiment_name: str, partition_type: str) -> Any:
    """Read dirichlet_alpha from the matching configs/<name>.yaml, if it exists.

    summary.json does not carry this field for federated runs; the config is
    the only reliable source. Returns NOT_APPLICABLE for non-dirichlet runs,
    NOT_RECORDED if a dirichlet run's config cannot be located/parsed.
    """
    if partition_type != "dirichlet":
        return NOT_APPLICABLE
    config_path = CONFIGS_DIR / f"{experiment_name}.yaml"
    if not config_path.exists():
        return NOT_RECORDED
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config.get("dirichlet_alpha", NOT_RECORDED)


def resolve_config_path(experiment_name: str) -> str:
    candidate = CONFIGS_DIR / f"{experiment_name}.yaml"
    if candidate.exists():
        return str(candidate.relative_to(REPO_ROOT)).replace("\\", "/")
    return NOT_RECORDED


def build_experiment_registry_row(reg: RegistryRow, result_dir: Path | None) -> dict[str, Any]:
    """The main per-experiment record. All numeric/config fields are read
    fresh from summary.json / the matching YAML config -- never copied
    unchecked from the registry CSV, which is used here only as the index of
    which experiments exist and their qualitative category/purpose."""

    if result_dir is None:
        # EXP_013: synthesis-only, no independent run -- registered honestly,
        # per this task's explicit Step 5 instruction, not invented.
        return {
            "experiment_id": reg.experiment_id,
            "experiment_name": reg.experiment_name,
            "experiment_category": "SYNTHESIS_ONLY_NO_INDEPENDENT_RUN",
            "dataset": reg.dataset,
            "model": reg.model,
            "aggregation": NOT_APPLICABLE,
            "seed": NOT_APPLICABLE,
            "partition": reg.partition_type,
            "dirichlet_alpha": reg.dirichlet_alpha if reg.dirichlet_alpha else NOT_APPLICABLE,
            "num_clients": reg.num_clients,
            "clients_per_round": NOT_APPLICABLE,
            "communication_rounds": NOT_APPLICABLE,
            "local_epochs": NOT_APPLICABLE,
            "optimizer": NOT_APPLICABLE,
            "learning_rate": NOT_APPLICABLE,
            "batch_size": NOT_APPLICABLE,
            "attack_present": "No",
            "attack_type": NOT_APPLICABLE,
            "malicious_client_id": NOT_APPLICABLE,
            "attack_start_round": NOT_APPLICABLE,
            "attack_strength_rho": NOT_APPLICABLE,
            "directional_target_cosine": NOT_APPLICABLE,
            "final_test_loss": NOT_APPLICABLE,
            "final_test_top1": NOT_APPLICABLE,
            "final_test_top5": NOT_APPLICABLE,
            "best_validation_top1": NOT_APPLICABLE,
            "best_validation_round": NOT_APPLICABLE,
            "macro_precision": NOT_APPLICABLE,
            "macro_recall": NOT_APPLICABLE,
            "macro_f1": NOT_APPLICABLE,
            "balanced_accuracy": NOT_APPLICABLE,
            "git_commit": NOT_APPLICABLE,
            "result_dir": NOT_APPLICABLE,
            "config_path": NOT_APPLICABLE,
            "notes_status": (
                "Synthesis-only entry: consolidates EXP_007/008/009/010/011/012 "
                "(FedAvg vs FedProx three-seed comparison). No independent "
                "training run, no summary.json of its own. See "
                "analysis/EXP_013_FedAvg_vs_FedProx_ThreeSeed_Synthesis.md."
            ),
        }

    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        raise SyncError(f"{reg.experiment_id}: expected summary.json not found at {summary_path}")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    is_federated = "aggregation_strategy" in summary
    attack = summary.get("attack")
    best_val = summary.get("best_validation", {})
    final_test = summary.get("final_test", {})

    attack_present = "Yes" if (attack and attack.get("attack_type") not in (None, "none")) else "No"

    return {
        "experiment_id": reg.experiment_id,
        "experiment_name": summary.get("experiment_name", reg.experiment_name),
        "experiment_category": reg.status,
        "dataset": summary.get("dataset", reg.dataset),
        "model": summary.get("model", reg.model),
        "aggregation": summary.get("aggregation_strategy", NOT_APPLICABLE if not is_federated else NOT_RECORDED),
        "seed": summary.get("seed", NOT_RECORDED),
        "partition": summary.get("partition", NOT_APPLICABLE if not is_federated else NOT_RECORDED),
        "dirichlet_alpha": load_dirichlet_alpha(reg.experiment_name, reg.partition_type),
        "num_clients": summary.get("num_clients", NOT_APPLICABLE if not is_federated else NOT_RECORDED),
        "clients_per_round": summary.get("clients_per_round", NOT_APPLICABLE if not is_federated else NOT_RECORDED),
        "communication_rounds": summary.get(
            "communication_rounds_completed",
            summary.get("completed_epochs", NOT_RECORDED),
        ),
        "local_epochs": summary.get("local_epochs", NOT_APPLICABLE if not is_federated else NOT_RECORDED),
        "optimizer": summary.get("optimizer", NOT_RECORDED),
        "learning_rate": summary.get("learning_rate", NOT_RECORDED),
        "batch_size": summary.get("batch_size", NOT_RECORDED),
        "attack_present": attack_present,
        "attack_type": _get(summary, "attack", "attack_type", default=NOT_APPLICABLE) if attack else NOT_APPLICABLE,
        "malicious_client_id": _get(summary, "attack", "malicious_client_id", default=NOT_APPLICABLE) if attack else NOT_APPLICABLE,
        "attack_start_round": _get(summary, "attack", "attack_start_round", default=NOT_APPLICABLE) if attack else NOT_APPLICABLE,
        "attack_strength_rho": _get(summary, "attack", "attack_strength_rho", default=NOT_APPLICABLE) if attack else NOT_APPLICABLE,
        "directional_target_cosine": _get(summary, "attack", "directional_target_cosine", default=NOT_APPLICABLE) if attack else NOT_APPLICABLE,
        "final_test_loss": final_test.get("loss", NOT_RECORDED),
        "final_test_top1": final_test.get("top1", NOT_RECORDED),
        "final_test_top5": final_test.get("top5", NOT_RECORDED),
        "best_validation_top1": best_val.get("top1", NOT_RECORDED),
        "best_validation_round": best_val.get("round_id", best_val.get("epoch", NOT_RECORDED)),
        "macro_precision": final_test.get("macro_precision", NOT_RECORDED),
        "macro_recall": final_test.get("macro_recall", NOT_RECORDED),
        "macro_f1": final_test.get("macro_f1", NOT_RECORDED),
        "balanced_accuracy": final_test.get("balanced_accuracy", NOT_RECORDED),
        "git_commit": summary.get("git_commit", NOT_RECORDED),
        "result_dir": str(result_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "config_path": resolve_config_path(reg.experiment_name),
        "notes_status": f"{reg.purpose} | Paired control: {reg.paired_control}",
    }


def cross_validate_final_test_metrics(exp_id: str, result_dir: Path, summary: dict) -> list[str]:
    """Cross-check summary.json's final_test block against the independently
    written final_test_metrics.json, where present. Returns a list of
    human-readable discrepancy strings (empty if consistent or not
    applicable) -- this script does not silently prefer one source over the
    other if they disagree."""
    metrics_path = result_dir / "final_test_metrics.json"
    if not metrics_path.exists():
        return []
    with metrics_path.open(encoding="utf-8") as handle:
        standalone = json.load(handle)
    final_test = summary.get("final_test", {})
    discrepancies = []
    for key in ("loss", "top1", "top5"):
        a, b = final_test.get(key), standalone.get(key)
        if a is not None and b is not None and a != b:
            discrepancies.append(f"{exp_id}: summary.json final_test.{key}={a} != final_test_metrics.json {key}={b}")
    return discrepancies


# ---------------------------------------------------------------------------
# Multi-Krum diagnostics (computed fresh from each run's own raw CSV)
# ---------------------------------------------------------------------------


def load_multikrum_summary(exp_id: str, result_dir: Path, malicious_client: str, attack_onset: Any) -> dict[str, Any] | None:
    diag_path = result_dir / "multikrum_diagnostics.csv"
    if not diag_path.exists():
        return None

    with diag_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["round_id"] = int(row["round_id"])
        row["multikrum_rejected"] = row["multikrum_rejected"].strip().lower() == "true"

    by_client: dict[str, list[dict]] = {}
    for row in rows:
        by_client.setdefault(row["client_id"], []).append(row)
    for client_rows in by_client.values():
        client_rows.sort(key=lambda r: r["round_id"])

    onset = attack_onset if isinstance(attack_onset, int) else None
    has_attacker = malicious_client not in (NOT_APPLICABLE, "", None)

    # Pre-attack (or full-run, for clean controls) most-rejected honest client.
    pre_window = [
        r for r in rows
        if (onset is None or r["round_id"] < onset) and r["client_id"] != malicious_client
    ]
    pre_rejection_counts: dict[str, int] = {}
    for r in pre_window:
        if r["multikrum_rejected"]:
            pre_rejection_counts[r["client_id"]] = pre_rejection_counts.get(r["client_id"], 0) + 1
    pre_attack_rejected_client = NOT_APPLICABLE
    pre_attack_rejection_count = 0
    if pre_rejection_counts:
        pre_attack_rejected_client = max(pre_rejection_counts, key=lambda c: (pre_rejection_counts[c], c))
        pre_attack_rejection_count = pre_rejection_counts[pre_attack_rejected_client]

    post_attacker_rejection_count = NOT_APPLICABLE
    post_attacker_rejection_rate = NOT_APPLICABLE
    post_attacker_selection_count = NOT_APPLICABLE
    rejection_switch_round = NOT_APPLICABLE
    proxy_predicted_client = NOT_APPLICABLE
    proxy_attribution_correct = NOT_APPLICABLE
    proxy_confirmation_round = NOT_APPLICABLE

    if has_attacker and onset is not None:
        attacker_rows = [r for r in by_client.get(malicious_client, []) if r["round_id"] >= onset]
        rejected = sum(1 for r in attacker_rows if r["multikrum_rejected"])
        post_attacker_rejection_count = rejected
        post_attacker_rejection_rate = round(rejected / len(attacker_rows), 4) if attacker_rows else NOT_APPLICABLE
        post_attacker_selection_count = len(attacker_rows) - rejected

        # Regime switch: first post-attack round where the attacker becomes
        # rejected and, from that round on, is rejected in every remaining
        # round observed (i.e. a durable, not merely momentary, flip).
        for r in attacker_rows:
            remainder = [x for x in attacker_rows if x["round_id"] >= r["round_id"]]
            if r["multikrum_rejected"] and all(x["multikrum_rejected"] for x in remainder):
                rejection_switch_round = r["round_id"]
                break

        # Full-run rejection-persistence proxy (P_krum=3), applied mechanically
        # to every client's own rejected/selected sequence across the WHOLE
        # run (matching the already-documented Multi-Krum analysis series'
        # own protocol) -- earliest-confirmed client wins.
        confirmations: dict[str, int] = {}
        for client_id, client_rows in by_client.items():
            streak = 0
            for r in client_rows:
                if r["multikrum_rejected"]:
                    streak += 1
                    if streak >= MULTIKRUM_PERSISTENCE_P:
                        confirmations[client_id] = r["round_id"]
                        break
                else:
                    streak = 0
        if confirmations:
            proxy_predicted_client = min(confirmations, key=lambda c: (confirmations[c], c))
            proxy_confirmation_round = confirmations[proxy_predicted_client]
            proxy_attribution_correct = proxy_predicted_client == malicious_client

    return {
        "pre_attack_rejected_client": pre_attack_rejected_client,
        "pre_attack_rejection_count": pre_attack_rejection_count,
        "post_attack_attacker_rejection_count": post_attacker_rejection_count,
        "post_attack_attacker_rejection_rate": post_attacker_rejection_rate,
        "post_attack_attacker_selection_count": post_attacker_selection_count,
        "rejection_switch_round": rejection_switch_round,
        "proxy_predicted_client": proxy_predicted_client,
        "proxy_attribution_correct": proxy_attribution_correct,
        "proxy_confirmation_round": proxy_confirmation_round,
    }


# ---------------------------------------------------------------------------
# Workbook assembly
# ---------------------------------------------------------------------------


def backup_workbook() -> Path | None:
    if not WORKBOOK_PATH.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = BACKUP_DIR / f"FedTrace_Master_Results_{stamp}.xlsx"
    shutil.copy2(WORKBOOK_PATH, backup_path)
    return backup_path


def write_sheet(wb, name: str, header: list[str], rows: list[list[Any]]) -> None:
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    ws.append(header)
    for row in rows:
        ws.append(row)


def run_sync(dry_run: bool = False) -> dict[str, Any]:
    registry = load_registry()
    result_dirs = discover_result_dirs()

    registry_rows: list[dict[str, Any]] = []
    fedtrace_rows: list[dict[str, Any]] = []
    multikrum_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    discrepancies: list[str] = []

    for reg in registry:
        result_dir = result_dirs.get(reg.experiment_id)
        if reg.experiment_id != "EXP_013" and result_dir is None:
            raise SyncError(
                f"{reg.experiment_id} is registered in {REGISTRY_CSV.name} but no "
                f"matching results/**/summary.json with that experiment_id was found."
            )

        row = build_experiment_registry_row(reg, result_dir)
        registry_rows.append(row)

        if result_dir is not None:
            with (result_dir / "summary.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)
            discrepancies.extend(cross_validate_final_test_metrics(reg.experiment_id, result_dir, summary))

            provenance_rows.append({
                "experiment_id": reg.experiment_id,
                "git_commit": summary.get("git_commit", NOT_RECORDED),
                "result_dir": str(result_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
                "checkpoint_dir": str(_get(summary, "checkpoint_paths", "final_global_model", default=NOT_RECORDED)),
                "config_path": resolve_config_path(reg.experiment_name),
                "device": summary.get("device", NOT_RECORDED),
                "gpu_name": summary.get("gpu_name", NOT_RECORDED),
                "total_experiment_seconds": summary.get(
                    "total_experiment_seconds", summary.get("total_training_seconds", NOT_RECORDED)
                ),
            })

            attack = summary.get("attack")
            malicious_client = attack.get("malicious_client_id") if attack else NOT_APPLICABLE
            attack_onset = attack.get("attack_start_round") if attack else None
            mk_summary = load_multikrum_summary(
                reg.experiment_id, result_dir, malicious_client or NOT_APPLICABLE, attack_onset
            )
            if mk_summary is not None:
                multikrum_rows.append({
                    "experiment_id": reg.experiment_id,
                    "clean_or_attacked": "Attacked" if attack_onset is not None else "Clean",
                    "seed": summary.get("seed", NOT_RECORDED),
                    "attack_type": _get(summary, "attack", "attack_type", default=NOT_APPLICABLE),
                    "true_attacker": malicious_client or NOT_APPLICABLE,
                    "f": 1,
                    "m": 4,
                    "neighbors_used": 4,
                    "source_analysis_document": f"analysis/EXP_{reg.experiment_id.split('_')[1]}_*.md; "
                    "analysis/MULTIKRUM_*_SYNTHESIS.md (see analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md Section 5)",
                    **mk_summary,
                })

        for fact in FEDTRACE_ATTRIBUTION_FACTS.get(reg.experiment_id, []):
            fedtrace_rows.append({"experiment_id": reg.experiment_id, **fact})

        for fact in ABLATION_FACTS.get(reg.experiment_id, []):
            ablation_rows.append({"experiment_id": reg.experiment_id, **fact})

    summary_info = {
        "registry_count": len(registry_rows),
        "fedtrace_row_count": len(fedtrace_rows),
        "multikrum_row_count": len(multikrum_rows),
        "ablation_row_count": len(ablation_rows),
        "discrepancies": discrepancies,
        "experiment_ids": [r["experiment_id"] for r in registry_rows],
    }

    if dry_run:
        return summary_info

    backup_path = backup_workbook()
    wb = openpyxl.load_workbook(WORKBOOK_PATH) if WORKBOOK_PATH.exists() else openpyxl.Workbook()
    if "Sheet" in wb.sheetnames and wb["Sheet"].max_row == 1 and wb["Sheet"].max_column == 1 and wb["Sheet"]["A1"].value is None:
        del wb["Sheet"]  # default empty sheet from a brand-new Workbook(), if ever created fresh

    registry_header = list(registry_rows[0].keys()) if registry_rows else []
    write_sheet(wb, "Experiment Registry", registry_header, [list(r.values()) for r in registry_rows])

    fedtrace_header = [
        "experiment_id", "fedtrace_version", "predicted_client", "true_client",
        "attribution_correct", "first_anomaly_round", "confirmation_round",
        "onset_estimate", "true_onset", "onset_error", "detection_delay",
        "honest_persistent_fp_count", "dominant_signal", "threshold",
        "persistence_p", "evaluable", "notes", "source_analysis_document",
    ]
    fedtrace_out = []
    for r in fedtrace_rows:
        fedtrace_out.append([r.get(k, NOT_RECORDED if k != "notes" else "") for k in fedtrace_header])
    write_sheet(wb, "FedTrace Attribution", fedtrace_header, fedtrace_out)

    multikrum_header = [
        "experiment_id", "clean_or_attacked", "seed", "attack_type", "true_attacker",
        "pre_attack_rejected_client", "pre_attack_rejection_count",
        "post_attack_attacker_rejection_count", "post_attack_attacker_rejection_rate",
        "post_attack_attacker_selection_count", "rejection_switch_round",
        "proxy_predicted_client", "proxy_attribution_correct", "proxy_confirmation_round",
        "f", "m", "neighbors_used", "source_analysis_document",
    ]
    multikrum_out = [[r.get(k, NOT_RECORDED) for k in multikrum_header] for r in multikrum_rows]
    write_sheet(wb, "MultiKrum Diagnostics Summary", multikrum_header, multikrum_out)

    ablation_header = ["experiment_id", "detector", "onset_error", "detection_delay", "notes"]
    ablation_out = [[r.get(k, NOT_RECORDED) for k in ablation_header] for r in ablation_rows]
    write_sheet(wb, "Ablation Detector Results", ablation_header, ablation_out)

    provenance_header = [
        "experiment_id", "git_commit", "result_dir", "checkpoint_dir", "config_path",
        "device", "gpu_name", "total_experiment_seconds",
    ]
    provenance_out = [[r.get(k, NOT_RECORDED) for k in provenance_header] for r in provenance_rows]
    write_sheet(wb, "Reproducibility Provenance", provenance_header, provenance_out)

    sync_meta_header = ["field", "value"]
    sync_meta_rows = [
        ["sync_timestamp_utc", datetime.now(timezone.utc).isoformat()],
        ["script_path", "Research_Database/sync_master_results.py"],
        ["source_registry", "analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv"],
        ["experiments_registered", len(registry_rows)],
        ["fedtrace_attribution_rows", len(fedtrace_rows)],
        ["multikrum_diagnostic_rows", len(multikrum_rows)],
        ["ablation_rows", len(ablation_rows)],
        ["cross_validation_discrepancies", len(discrepancies)],
        ["missing_value_policy", "NOT RECORDED = recoverable in principle but not safely "
         "extracted this run; NOT APPLICABLE = does not apply to this experiment's type. "
         "Never fabricated."],
        ["managed_sheets", ", ".join(MANAGED_SHEETS)],
        ["legacy_sheets_untouched", ", ".join(LEGACY_SHEETS)],
    ]
    write_sheet(wb, "Sync Metadata", sync_meta_header, sync_meta_rows)

    tmp_path = WORKBOOK_PATH.with_suffix(".xlsx.tmp")
    wb.save(tmp_path)
    tmp_path.replace(WORKBOOK_PATH)

    summary_info["backup_path"] = str(backup_path.relative_to(REPO_ROOT)).replace("\\", "/") if backup_path else None
    return summary_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Compute and print the sync summary without writing the workbook.")
    args = parser.parse_args()

    try:
        result = run_sync(dry_run=args.dry_run)
    except SyncError as exc:
        print(f"SYNC FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Experiments registered: {result['registry_count']}")
    print(f"FedTrace attribution rows: {result['fedtrace_row_count']}")
    print(f"Multi-Krum diagnostic rows: {result['multikrum_row_count']}")
    print(f"Ablation rows: {result['ablation_row_count']}")
    print(f"Experiment IDs: {', '.join(result['experiment_ids'])}")
    if result["discrepancies"]:
        print("DISCREPANCIES FOUND (summary.json vs final_test_metrics.json):")
        for d in result["discrepancies"]:
            print(f"  - {d}")
    else:
        print("No summary.json / final_test_metrics.json discrepancies found.")
    if not args.dry_run:
        print(f"Backup written to: {result.get('backup_path')}")
        print(f"Workbook updated at: {WORKBOOK_PATH.relative_to(REPO_ROOT)}")
    else:
        print("(dry run -- workbook not written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
