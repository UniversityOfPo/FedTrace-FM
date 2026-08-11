"""Focused tests for Research_Database/sync_master_results.py.

CPU-only, no GPU, no training. Tests exercise the sync script's parsing,
validation, and derivation logic -- both against the real repository
artifacts (read-only) and against small synthetic fixtures for the
duplicate/malformed-input error paths that the real repository does not
naturally exhibit.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT_PATH = REPO_ROOT / "Research_Database" / "sync_master_results.py"


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_master_results", SYNC_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sync_master_results"] = module
    spec.loader.exec_module(module)
    return module


sync = _load_sync_module()


# ---------------------------------------------------------------------------
# Experiment ID parsing / sorting
# ---------------------------------------------------------------------------


def test_registry_loads_and_sorts_numerically():
    rows = sync.load_registry()
    ids = [r.experiment_id for r in rows]
    numeric = [int(i.split("_")[1]) for i in ids]
    assert numeric == sorted(numeric), "registry rows must be sorted by numeric EXP ID"
    assert numeric == list(range(1, 28)), "expected exactly EXP_001..EXP_027"


def test_registry_rejects_duplicate_experiment_ids(tmp_path, monkeypatch):
    bad_csv = tmp_path / "bad_registry.csv"
    header = (
        "Experiment_ID,Experiment_Name,Dataset,Model,Aggregation,Seed,Num_Clients,"
        "Partition_Type,Dirichlet_Alpha,Attack_Present,Attack_Family,Malicious_Client,"
        "Attack_Onset_Round,Attack_Severity_Or_Target_Cosine,Purpose,Paired_Control,"
        "Final_Test_Loss,Final_Test_Top1,Final_Test_Top5,FedTrace_Version,"
        "Attribution_Result,Estimated_Onset,Detection_Delay,Honest_Persistent_FP,"
        "MultiKrum_Rejection_Behavior,Scientific_Interpretation,Status\n"
    )
    row = ",".join(["EXP_999"] + ["x"] * 26) + "\n"
    bad_csv.write_text(header + row + row, encoding="utf-8")

    monkeypatch.setattr(sync, "REGISTRY_CSV", bad_csv)
    with pytest.raises(sync.SyncError, match="Duplicate Experiment_ID"):
        sync.load_registry()


# ---------------------------------------------------------------------------
# EXP_013 synthesis-only handling
# ---------------------------------------------------------------------------


def test_exp_013_has_no_result_dir():
    rows = sync.load_registry()
    exp_013 = next(r for r in rows if r.experiment_id == "EXP_013")
    result_dirs = sync.discover_result_dirs()
    assert "EXP_013" not in result_dirs, "EXP_013 is synthesis-only and must not resolve to a training-run directory"
    row = sync.build_experiment_registry_row(exp_013, None)
    assert row["experiment_category"] == "SYNTHESIS_ONLY_NO_INDEPENDENT_RUN"
    assert row["final_test_loss"] == sync.NOT_APPLICABLE
    assert row["final_test_top1"] == sync.NOT_APPLICABLE
    assert row["seed"] == sync.NOT_APPLICABLE
    assert "synthesis" in row["notes_status"].lower() or "Synthesis" in row["notes_status"]


def test_exp_013_metrics_are_not_fabricated():
    rows = sync.load_registry()
    exp_013 = next(r for r in rows if r.experiment_id == "EXP_013")
    row = sync.build_experiment_registry_row(exp_013, None)
    for numeric_field in ("final_test_loss", "final_test_top1", "final_test_top5",
                           "best_validation_top1", "macro_f1"):
        assert row[numeric_field] == sync.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Missing optional artifact handling
# ---------------------------------------------------------------------------


def test_clean_run_without_client_round_trace_returns_no_multikrum_summary():
    # EXP_008 is a clean FedAvg run: no client_round_trace.csv, no
    # multikrum_diagnostics.csv. The sync script must tolerate this, not crash.
    result_dirs = sync.discover_result_dirs()
    result_dir = result_dirs["EXP_008"]
    assert not (result_dir / "multikrum_diagnostics.csv").exists()
    summary = sync.load_multikrum_summary("EXP_008", result_dir, sync.NOT_APPLICABLE, None)
    assert summary is None


def test_fedavg_directional_attack_run_has_no_multikrum_diagnostics():
    # EXP_018 is FedAvg (not Multi-Krum) but IS attack-configured -- confirms
    # multikrum_diagnostics.csv absence is about aggregation strategy, not
    # attack presence.
    result_dirs = sync.discover_result_dirs()
    result_dir = result_dirs["EXP_018"]
    assert not (result_dir / "multikrum_diagnostics.csv").exists()


# ---------------------------------------------------------------------------
# Deterministic output ordering / idempotent synchronization logic
# ---------------------------------------------------------------------------


def test_dry_run_is_deterministic_across_two_invocations():
    result_1 = sync.run_sync(dry_run=True)
    result_2 = sync.run_sync(dry_run=True)
    assert result_1["experiment_ids"] == result_2["experiment_ids"]
    assert result_1["registry_count"] == result_2["registry_count"] == 27
    assert result_1["fedtrace_row_count"] == result_2["fedtrace_row_count"]
    assert result_1["multikrum_row_count"] == result_2["multikrum_row_count"]
    assert result_1["ablation_row_count"] == result_2["ablation_row_count"]
    assert result_1["discrepancies"] == result_2["discrepancies"] == []


def test_managed_sheets_do_not_overlap_legacy_sheets():
    assert set(sync.MANAGED_SHEETS).isdisjoint(set(sync.LEGACY_SHEETS))


# ---------------------------------------------------------------------------
# Representative clean-run extraction
# ---------------------------------------------------------------------------


def test_clean_run_field_extraction_exp_008():
    rows = sync.load_registry()
    reg = next(r for r in rows if r.experiment_id == "EXP_008")
    result_dir = sync.discover_result_dirs()["EXP_008"]
    row = sync.build_experiment_registry_row(reg, result_dir)

    with (result_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    assert row["aggregation"] == "fedavg"
    assert row["seed"] == 123
    assert row["partition"] == "dirichlet"
    assert row["dirichlet_alpha"] == 0.1
    assert row["attack_present"] == "No"
    assert row["attack_type"] == sync.NOT_APPLICABLE
    assert row["final_test_top1"] == summary["final_test"]["top1"]
    assert row["final_test_loss"] == summary["final_test"]["loss"]


# ---------------------------------------------------------------------------
# Attack metadata extraction
# ---------------------------------------------------------------------------


def test_attack_metadata_extraction_exp_018():
    rows = sync.load_registry()
    reg = next(r for r in rows if r.experiment_id == "EXP_018")
    result_dir = sync.discover_result_dirs()["EXP_018"]
    row = sync.build_experiment_registry_row(reg, result_dir)

    assert row["attack_present"] == "Yes"
    assert row["attack_type"] == "norm_preserving_directional"
    assert row["malicious_client_id"] == "client_002"
    assert row["attack_start_round"] == 20
    assert row["directional_target_cosine"] == 0.0


def test_attack_metadata_extraction_exp_027_shifted_onset():
    rows = sync.load_registry()
    reg = next(r for r in rows if r.experiment_id == "EXP_027")
    result_dir = sync.discover_result_dirs()["EXP_027"]
    row = sync.build_experiment_registry_row(reg, result_dir)

    assert row["malicious_client_id"] == "client_002"
    assert row["attack_start_round"] == 30


def test_attack_metadata_extraction_exp_026_shifted_identity():
    rows = sync.load_registry()
    reg = next(r for r in rows if r.experiment_id == "EXP_026")
    result_dir = sync.discover_result_dirs()["EXP_026"]
    row = sync.build_experiment_registry_row(reg, result_dir)

    assert row["malicious_client_id"] == "client_004"
    assert row["attack_start_round"] == 20


# ---------------------------------------------------------------------------
# Multi-Krum run metadata extraction
# ---------------------------------------------------------------------------


def test_multikrum_directional_attack_zero_rejection_exp_024():
    result_dirs = sync.discover_result_dirs()
    result_dir = result_dirs["EXP_024"]
    summary = sync.load_multikrum_summary("EXP_024", result_dir, "client_002", 20)
    assert summary is not None
    assert summary["post_attack_attacker_rejection_count"] == 0
    assert summary["post_attack_attacker_rejection_rate"] == 0.0
    assert summary["rejection_switch_round"] == sync.NOT_APPLICABLE
    # The true attacker is never rejected under the directional attack, so the
    # full-run persistence proxy must not name client_002 correctly.
    assert summary["proxy_attribution_correct"] is False


def test_multikrum_gaussian_attack_full_rejection_exp_022():
    result_dirs = sync.discover_result_dirs()
    result_dir = result_dirs["EXP_022"]
    summary = sync.load_multikrum_summary("EXP_022", result_dir, "client_002", 20)
    assert summary is not None
    assert summary["post_attack_attacker_rejection_count"] == 30
    assert summary["post_attack_attacker_rejection_rate"] == 1.0
    assert summary["rejection_switch_round"] == 20


def test_multikrum_clean_control_has_no_attacker_fields_exp_020():
    result_dirs = sync.discover_result_dirs()
    result_dir = result_dirs["EXP_020"]
    summary = sync.load_multikrum_summary("EXP_020", result_dir, sync.NOT_APPLICABLE, None)
    assert summary is not None
    assert summary["post_attack_attacker_rejection_count"] == sync.NOT_APPLICABLE
    assert summary["proxy_predicted_client"] == sync.NOT_APPLICABLE
    # A clean run still has a most-rejected honest client under the proxy's
    # pre-attack/full-run window.
    assert summary["pre_attack_rejected_client"] != sync.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# No accidental mutation of scientific result values
# ---------------------------------------------------------------------------


def test_registry_extraction_does_not_alter_source_summary_json():
    result_dirs = sync.discover_result_dirs()
    result_dir = result_dirs["EXP_018"]
    summary_path = result_dir / "summary.json"
    before = summary_path.read_bytes()

    rows = sync.load_registry()
    reg = next(r for r in rows if r.experiment_id == "EXP_018")
    sync.build_experiment_registry_row(reg, result_dir)
    sync.load_multikrum_summary("EXP_018", result_dir, "client_002", 20)

    after = summary_path.read_bytes()
    assert before == after, "sync script must never write to an experiment's own summary.json"


def test_final_test_values_are_not_rounded_or_altered():
    result_dirs = sync.discover_result_dirs()
    result_dir = result_dirs["EXP_026"]
    with (result_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    rows = sync.load_registry()
    reg = next(r for r in rows if r.experiment_id == "EXP_026")
    row = sync.build_experiment_registry_row(reg, result_dir)
    assert row["final_test_loss"] == summary["final_test"]["loss"]
    assert row["final_test_top1"] == summary["final_test"]["top1"]
    assert row["final_test_top5"] == summary["final_test"]["top5"]


# ---------------------------------------------------------------------------
# FedTrace attribution / ablation constant-table sanity
# ---------------------------------------------------------------------------


def test_fedtrace_attribution_facts_cover_every_attack_run():
    rows = sync.load_registry()
    attack_run_ids = {
        r.experiment_id for r in rows
        if r.attack_present == "Yes" and r.experiment_id not in ("EXP_020", "EXP_021", "EXP_022", "EXP_023", "EXP_024", "EXP_025")
    }
    # FedTrace attribution facts are curated for the FedAvg attack runs
    # (development + validation + generalization) -- Multi-Krum attack runs
    # are covered by the Multi-Krum sheet instead, not this one.
    assert attack_run_ids == set(sync.FEDTRACE_ATTRIBUTION_FACTS.keys())


def test_exp_015_has_three_fedtrace_versions_recorded():
    facts = sync.FEDTRACE_ATTRIBUTION_FACTS["EXP_015"]
    assert len(facts) == 3
    versions = [f["fedtrace_version"] for f in facts]
    assert any("v1" in v for v in versions)
    assert any("v2" in v for v in versions)
    assert any("v3" in v for v in versions)
    v2_fact = next(f for f in facts if "v2" in f["fedtrace_version"])
    assert v2_fact["attribution_correct"] is False
    assert v2_fact["predicted_client"] == "client_000"


def test_round0_wording_policy_not_violated_in_notes():
    # Guard against reintroducing the overclaim phrase flagged in
    # analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md Section 7.
    forbidden = "no honest client ever exceeds"
    for facts in sync.FEDTRACE_ATTRIBUTION_FACTS.values():
        for fact in facts:
            note_text = str(fact.get("notes", "")).lower()
            assert forbidden not in note_text
