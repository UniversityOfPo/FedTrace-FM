"""Stage B comparison script and preflight (CPU only; synthetic run directories built from archived small files)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import torch

from src.rev import compare_runs as cr
from src.rev import preflight as pf

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/federated" / cr.ARCHIVE_NAME
EXP016_CONFIG = f"configs/{cr.ARCHIVE_NAME}.yaml"
SMALL_FILES = ("client_round_trace.csv", "round_history.csv", "client_partition_indices.json",
               "client_partition_summary.csv", "final_test_metrics.json", "per_class_accuracy.csv", "summary.json")


def _checkpoints(directory: Path, bump: float = 0.0) -> None:
    directory.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    state = {"w": torch.randn(4, 3, generator=g), "bn.num_batches_tracked": torch.tensor(7)}
    state["w"][0, 0] += bump
    torch.save({"model_state_dict": state, "experiment_id": "EXP_016", "seed": 123}, directory / "final_global_model.pt")
    torch.save({"global_model_state_dict": state, "round_history": [{"round_id": 0, "started_at_utc": "x"}]},
               directory / "runner_latest_server.pt")
    torch.save({"history": [{"round_id": 0, "metadata": {"elapsed_seconds": 1.0}}]}, directory / "runner_latest.pt")
    torch.save({"model_state_dict": state, "optimizer_state_dict": {"state": {}}}, directory / "runner_latest_client_client_000.pt")


@pytest.fixture()
def pair(tmp_path: Path) -> dict[str, str]:
    if not ARCHIVE.exists():
        pytest.skip("archived EXP_016 results not available")
    for side in ("ref", "cand"):
        (tmp_path / side / "results").mkdir(parents=True)
        for name in SMALL_FILES:
            shutil.copyfile(ARCHIVE / name, tmp_path / side / "results" / name)
        _checkpoints(tmp_path / side / "ck")
    return {"reference_results": str(tmp_path / "ref/results"), "reference_checkpoints": str(tmp_path / "ref/ck"),
            "reference_config": str(ROOT / EXP016_CONFIG), "candidate_results": str(tmp_path / "cand/results"),
            "candidate_checkpoints": str(tmp_path / "cand/ck"),
            "candidate_config": str(ROOT / "configs/rev/R0_exp016_repro.yaml"), "candidate_log": None}


def _status(report: cr.Report, category: str, contains: str = "") -> str:
    found = [f.status for f in report.findings if f.category == category and contains in f.item]
    assert found, (category, contains)
    return found[0]


def _edit_csv(path: Path, key: dict[str, str], column: str, value: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        if all(row[k] == v for k, v in key.items()):
            row[column] = value
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_identical_runs_report_no_difference(pair) -> None:
    report = cr.compare(pair, Path("."))
    assert not report.differences, [f for f in report.differences]
    assert _status(report, "CONFIG") == "IDENTICAL"
    assert {f.category for f in report.findings} >= {"CONFIG", "PARTITION", "RNG_PROXY", "HISTORY", "TRACE",
                                                    "TRACE_ORDER_DEPENDENT", "DETECTOR", "GLOBAL_STATE",
                                                    "CLIENT_STATE", "METRICS", "RUNNER", "PROVENANCE"}
    assert any(f.status == "EXCLUDED" and "timing" in f.item for f in report.findings)


def test_peer_field_difference_is_reported_separately_not_ignored(pair) -> None:
    trace = Path(pair["candidate_results"]) / "client_round_trace.csv"
    with trace.open(newline="", encoding="utf-8") as handle:
        row = next(r for r in csv.DictReader(handle) if r["round_id"] == "3" and r["client_id"] == "client_001")
    value = float(row["peer_mean_cosine_similarity"])
    _edit_csv(trace, {"round_id": "3", "client_id": "client_001"}, "peer_mean_cosine_similarity",
              repr(value + abs(value) * 2e-16 + 1e-300))
    report = cr.compare(pair, Path("."))
    assert _status(report, "TRACE_ORDER_DEPENDENT") == "NUMERICAL"
    assert _status(report, "TRACE", "order-independent") == "IDENTICAL"
    assert [f.category for f in report.differences] == ["TRACE_ORDER_DEPENDENT"]


def test_update_difference_reports_first_round_and_reaches_the_detector(pair) -> None:
    trace = Path(pair["candidate_results"]) / "client_round_trace.csv"
    for r in range(25, 30):
        _edit_csv(trace, {"round_id": str(r), "client_id": "client_000"}, "update_l2_norm", "1000.0")
    report = cr.compare(pair, Path("."))
    finding = next(f for f in report.findings if f.category == "TRACE" and "order-independent" in f.item)
    assert finding.status == "NUMERICAL"
    assert finding.detail["differing_columns"]["update_l2_norm"]["first_differing_key"] == "25"
    assert _status(report, "RNG_PROXY", "update norms") == "NUMERICAL"
    assert _status(report, "DETECTOR", "decisions") == "VALUE"  # an alarm appears where there was none


def test_timing_columns_are_excluded_but_metric_columns_are_not(pair) -> None:
    history = Path(pair["candidate_results"]) / "round_history.csv"
    _edit_csv(history, {"round_id": "4"}, "round_seconds", "999.0")
    assert not cr.compare(pair, Path(".")).differences
    _edit_csv(history, {"round_id": "4"}, "train_loss", "0.5")
    report = cr.compare(pair, Path("."))
    assert next(f.status for f in report.findings if f.item == "round_history.csv") == "NUMERICAL"
    assert _status(report, "RNG_PROXY", "local-training") == "NUMERICAL"
    assert _status(report, "RNG_PROXY", "client-selection") == "IDENTICAL"


def test_tensor_difference_missing_file_and_git_commit_info(pair) -> None:
    shutil.rmtree(pair["candidate_checkpoints"])
    _checkpoints(Path(pair["candidate_checkpoints"]), bump=1e-7)
    (Path(pair["candidate_results"]) / "final_test_metrics.json").unlink()
    summary = Path(pair["candidate_results"]) / "summary.json"
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["git_commit"] = "0" * 40
    summary.write_text(json.dumps(data), encoding="utf-8")
    report = cr.compare(pair, Path("."))
    final = next(f for f in report.findings if f.item == "final global model tensors")
    assert final.status == "NUMERICAL" and 0 < final.detail["max_abs_diff"] < 1e-6
    assert _status(report, "METRICS", "final_test_metrics.json") == "MISSING"
    assert _status(report, "PROVENANCE", "git commit") == "INFO"
    assert _status(report, "METRICS", "summary.json") == "IDENTICAL"


def test_config_difference_is_detected() -> None:
    report = cr.Report({}, {})
    cr.check_config(report, ROOT / EXP016_CONFIG, ROOT / "configs/rev/S0a_smoke_ndss.yaml")
    finding = report.findings[0]
    assert finding.status == "VALUE"  # experiment_id/name differ (strings) besides the numeric fields
    assert finding.detail["numerical"] == 2 and finding.detail["value"] == 2
    assert any("communication_rounds" in p for p in finding.detail["first_differences"])


def test_value_comparison_is_bitwise() -> None:
    assert cr.compare_values(0.0, -0.0).status == "NUMERICAL"
    assert cr.compare_values(torch.tensor([0.0]), torch.tensor([-0.0])).status == "NUMERICAL"
    assert cr.compare_values(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64)).status == "STRUCTURAL"
    nan = float("nan")
    assert cr.compare_values(torch.tensor([nan]), torch.tensor([nan])).status == "IDENTICAL"
    d = cr.compare_values({"a": 1, "elapsed_seconds": 3}, {"a": 1, "elapsed_seconds": 4})
    assert d.status == "IDENTICAL" and d.excluded == ["elapsed_seconds"]
    assert cr.compare_values({"a": 1}, {"b": 1}).status == "STRUCTURAL"
    assert cr.compare_values({"a": "x"}, {"a": "y"}).status == "VALUE"
    assert cr.compare_values({"a": None}, {"a": None}).status == "IDENTICAL"
    assert cr.compare_values({"a": None}, {"a": 3}).status == "VALUE"


def test_presets_point_into_rev_output_trees() -> None:
    for name, paths in cr.PRESETS.items():
        assert paths["candidate_results"].startswith("results_rev/"), name
        assert paths["candidate_checkpoints"].startswith("checkpoints_rev/"), name


def test_preflight_disk_arithmetic_and_failures(tmp_path: Path, monkeypatch) -> None:
    r1 = pf.PLANNED["R1"]
    assert pf.planned_bytes(r1) == pf.RUN_OUTPUT_BYTES + (6 * 50 + 51) * pf.STATE_BYTES
    assert pf.planned_bytes(pf.PLANNED["R0"]) == pf.RUN_OUTPUT_BYTES

    class Usage:
        free = 50 * 1024**3

    monkeypatch.setattr(pf.shutil, "disk_usage", lambda _p: Usage())
    assert pf.check_disk(tmp_path, [r1], reserve_gib=100).status == "FAIL"
    assert pf.check_archive(tmp_path, tmp_path).status == "FAIL"  # archive files absent


def test_preflight_detects_existing_outputs(tmp_path: Path) -> None:
    (tmp_path / "configs/rev").mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/rev/S0a_smoke_ndss.yaml", tmp_path / "configs/rev/S0a_smoke_ndss.yaml")
    run = pf.PLANNED["S0a"]
    import os

    cwd = os.getcwd()
    os.chdir(ROOT)  # the NDSS loader resolves dataset_config relative to the working directory
    try:
        assert pf.check_outputs(tmp_path, [run]).status == "PASS"
        (tmp_path / "results_rev/S0a_smoke_ndss").mkdir(parents=True)
        (tmp_path / "results_rev/S0a_smoke_ndss/summary.json").write_text("{}", encoding="utf-8")
        assert pf.check_outputs(tmp_path, [run]).status == "FAIL"
    finally:
        os.chdir(cwd)


def _materialise(run, directory: Path, attack) -> tuple[Path, Path]:
    """Write a toy run's trace, partition stub and final model in the NDSS output layout."""
    from src.attacks.traceability import apply_naive_detector, save_client_round_trace_csv
    from tests.rev_toy_federation import CLIENT_IDS

    results, ck = directory / "results", directory / "ck"
    results.mkdir(parents=True)
    ck.mkdir()
    save_client_round_trace_csv(apply_naive_detector(run.trace_rows, attack), results / "client_round_trace.csv")
    (results / "client_partition_indices.json").write_text(json.dumps({c: [] for c in CLIENT_IDS}), encoding="utf-8")
    torch.save({"model_state_dict": run.globals_after[-1]}, ck / "final_global_model.pt")
    return results, ck


def test_update_log_checks_on_a_real_cpu_log(tmp_path: Path) -> None:
    import functools

    from src.rev.hooks import HookedRunner, RevHooks
    from src.rev.update_log import UpdateLogConfig, UpdateLogger
    from tests.rev_toy_federation import CLIENT_IDS, build_and_run
    from tests.test_rev_hooks_replay import GAUSSIAN, TRAINABLE, _gaussian

    plain = build_and_run(tmp_path / "a", rounds=4, transform=_gaussian, attack_config=GAUSSIAN)
    hooks = RevHooks(update_logger=UpdateLogger.open(UpdateLogConfig(
        root=tmp_path / "log", run_id="t", min_free_bytes=0, pre_transform_clients=("client_002",))))
    logged = build_and_run(tmp_path / "b", rounds=4, transform=_gaussian, attack_config=GAUSSIAN,
                           runner_factory=functools.partial(HookedRunner, rev_hooks=hooks))
    hooks.update_logger.seal()
    ref, ref_ck = _materialise(plain, tmp_path / "ref", GAUSSIAN)
    cand, cand_ck = _materialise(logged, tmp_path / "cand", GAUSSIAN)

    report = cr.Report({}, {})
    cr.check_update_log(report, tmp_path / "log", ref, ref_ck, cand, cand_ck, GAUSSIAN, TRAINABLE, CLIENT_IDS)
    statuses = {f.item: f.status for f in report.findings}
    assert set(statuses.values()) == {"IDENTICAL"}, statuses
    assert any("re-aggregation" in item for item in statuses) and any("reference trace" in item for item in statuses)

    # a corrupted log is reported, not ignored
    target = next((tmp_path / "log" / "states").glob("r0002_submitted_client_001.pt"))
    data = bytearray(target.read_bytes())
    data[-30] ^= 0xFF
    target.write_bytes(bytes(data))
    report = cr.Report({}, {})
    cr.check_update_log(report, tmp_path / "log", ref, ref_ck, cand, cand_ck, GAUSSIAN, TRAINABLE, CLIENT_IDS)
    assert next(f for f in report.findings if f.item.startswith("log verification")).status == "STRUCTURAL"
