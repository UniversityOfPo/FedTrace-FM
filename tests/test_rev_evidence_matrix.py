"""Evidence matrix: eight conditions, labels, detection vs attribution, honest accusations, matched-clean diagnostics,
missing / undefined values, read-only behaviour, reconciliation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from src.rev import evidence_matrix as em

CLIENTS = [f"client_00{i}" for i in range(5)]
HEADER = ["round_id", "client_id", "selected", "attack_configured", "attack_active", "attack_type", "attack_strength_rho",
          "malicious_client_id", "attack_start_round", "update_l2_norm", "peer_mean_cosine_similarity", "naive_threshold",
          "naive_flag", "first_flagged_round", "directional_target_cosine", "delta_attack_norm_ratio",
          "realized_cosine_to_honest", "self_prev_round_cosine"]
# seed "123": client_002 above the median, client_003 always the median (zero calibration sd -> undefined);
# seed "2026": client_002 lowest, client_000 always the median; the intervention lifts client_002 above it.
BASE_T = {"123": [0.20, 0.24, 0.36, 0.28, 0.32], "2026": [0.28, 0.32, 0.20, 0.36, 0.24]}
TOP1 = {"clean": 0.58, "badnets": 0.57, "relabel": 0.565, "drift": 0.56}


def _series(seed: str, cond: str, missing_self_cos: int | None = None) -> list[dict]:
    rows = []
    for r in range(50):
        for i, c in enumerate(CLIENTS):
            norm = 10 + i + 0.1 * math.sin(r + i)
            t = None if r == 0 else BASE_T[seed][i] + 0.002 * math.sin(r + 2 * i)
            if c == "client_002" and r >= 20 and cond in ("badnets", "relabel", "drift"):
                norm *= {"badnets": 0.8, "relabel": 0.86, "drift": 0.88}[cond]
                t = 0.40
            if c == "client_004" and r >= 20 and cond == "badnets" and seed == "2026":
                norm *= 0.7  # honest client confirmed before the attacker: detection without attribution
            if c == "client_002" and r == missing_self_cos:
                t = None
            rows.append({"round_id": r, "client_id": c, "selected": True, "attack_configured": False,
                         "attack_active": False, "attack_type": "none", "attack_strength_rho": 0.0,
                         "malicious_client_id": "", "attack_start_round": "", "update_l2_norm": norm,
                         "peer_mean_cosine_similarity": 0.1, "naive_threshold": "", "naive_flag": False,
                         "first_flagged_round": "", "directional_target_cosine": "", "delta_attack_norm_ratio": "",
                         "realized_cosine_to_honest": "", "self_prev_round_cosine": "" if t is None else t})
    return rows


def _write_run(root: Path, run: str, rows: list[dict], top1: float) -> None:
    d = root / "results_rev" / run
    d.mkdir(parents=True)
    with (d / "client_round_trace.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)
    (d / "final_test_metrics.json").write_text(json.dumps({"top1": top1}))


def _write_pair(root: Path, seed: str, cond: str, asr: float) -> None:
    run, clean = f"{em.CONDITIONS[cond][0].format(s=seed)}", em.CONDITIONS["clean"][0].format(s=seed)
    d = root / "results_rev/pilot" / f"{run}__vs__{clean}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text("{}")
    for name, a, top in ((run, asr, TOP1[cond] - 0.0002), (clean, 0.003, TOP1["clean"] - 0.0002)):
        (d / f"effect_{name}.csv").write_text(
            "round_id,clean_top1,untriggered_target_rate,asr\n-1,0.01,0,0\n" f"49,{top},0.001,{a}\n")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    for seed in ("123", "2026"):
        for cond, (pattern, _) in em.CONDITIONS.items():
            _write_run(tmp_path, pattern.format(s=seed), _series(seed, cond, missing_self_cos=25 if (seed, cond) == ("123", "relabel") else None), TOP1[cond])
        _write_pair(tmp_path, seed, "badnets", 0.6)
        _write_pair(tmp_path, seed, "relabel", 0.004)
    _write_pair(tmp_path, "123", "drift", 0.002)  # seed-2026 drift pair deliberately absent
    return tmp_path


def _hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "evidence_matrix" not in p.parts}


def test_eight_conditions_and_labels(root: Path) -> None:
    rep = em.build(root, ["123", "2026"], with_logs=False)
    assert sorted(rep["rows"]) == sorted(f"{s}/{c}" for s in ("123", "2026") for c in em.CONDITIONS)
    for key, e in rep["rows"].items():
        cond = key.split("/")[1]
        assert e["condition"] == cond and e["malicious"] == (cond in ("badnets", "relabel"))
        assert e["malicious_client"] == ("client_002" if cond in ("badnets", "relabel") else None)
        assert e["intervened_client"] == (None if cond == "clean" else "client_002")
        assert ("diagnostic" in e) == (cond != "clean")


def test_benign_drift_alarm_is_an_honest_accusation_not_detection_of_an_attacker(root: Path) -> None:
    e = em.build(root, ["123"], with_logs=False)["rows"]["123/drift"]["decisions"]["zp_norm"]
    assert e["first_attributed"] == "client_002" and e["honest_false_accusations"] == ["client_002"]
    assert e["attacker_detected"] is None and e["attribution_correct"] is None


def test_detection_without_attribution(root: Path) -> None:
    rows = em.build(root, ["2026"], with_logs=False)["rows"]
    z = rows["2026/badnets"]["decisions"]["zp_norm"]
    assert z["attacker_detected"] and z["first_attributed"] == "client_004" and not z["attribution_correct"]
    assert z["honest_false_accusations"] == ["client_004"] and z["n_honest_false_accusations"] == 1
    for e in rows.values():  # v3 scoring order agrees with the online wrapper everywhere
        assert e["decisions"]["v3"]["online_wrapper_attributed"] == e["decisions"]["v3"]["first_attributed"]


def test_clean_runs_have_no_accusations(root: Path) -> None:
    rows = em.build(root, ["123", "2026"], with_logs=False)["rows"]
    for s in ("123", "2026"):
        for d in rows[f"{s}/clean"]["decisions"].values():
            assert not d["detected_any"] and d["honest_false_accusations"] == []


def test_matched_clean_diagnostics_and_reference_movement_sign(root: Path) -> None:
    rows = em.build(root, ["123", "2026"], with_logs=False)["rows"]
    d = rows["123/badnets"]["diagnostic"]
    assert d["norm_ratio_r25_35"] == pytest.approx(0.8)
    assert d["self_cos_change_r21_29"] == pytest.approx(0.40 - 0.36, abs=0.003)
    # client_002 stays above the median: it never moves the inclusive median; client_003 (always median) undefined
    assert d["reference_movement_undefined_clients"] == ["client_003"]
    assert all(v == pytest.approx(0.0) for v in d["reference_movement_on_others"].values())
    up = rows["2026/badnets"]["diagnostic"]
    assert up["reference_movement_undefined_clients"] == ["client_000"]
    assert all(v > 0 for v in up["reference_movement_on_others"].values())  # lifted above the median -> pushed up
    assert up["reference_movement_max_abs"]["value"] == max(up["reference_movement_on_others"].values())


def test_max_abs_and_max_signed_movement_are_distinct(monkeypatch, root: Path) -> None:
    fake = {"client_000": [{"attacker_induced": -0.7}], "client_001": [{"attacker_induced": 0.1}], "client_002": [], "client_003": []}
    monkeypatch.setattr(em, "decompose", lambda *a, **k: fake)
    d = em.build(root, ["123"], with_logs=False)["rows"]["123/drift"]["diagnostic"]
    assert d["reference_movement_max_signed"] == {"client": "client_001", "value": 0.1}
    assert d["reference_movement_max_abs"] == {"client": "client_000", "value": -0.7}
    assert d["reference_movement_undefined_clients"] == ["client_003"]


def test_accuracy_change_sign_and_sources(root: Path) -> None:
    rows = em.build(root, ["2026"], with_logs=False)["rows"]
    assert rows["2026/badnets"]["top1_change_vs_clean_pp"] == pytest.approx(-1.0)
    assert rows["2026/clean"]["top1_change_vs_clean_pp"] == 0.0
    assert rows["2026/badnets"]["pilot_top1_r49"] == pytest.approx(0.5698)
    assert rows["2026/clean"]["final_effect"]["asr"] == pytest.approx(0.003)  # read via the relabel pair


def test_missing_values_are_reported_not_imputed(root: Path) -> None:
    rep = em.build(root, ["123", "2026"], with_logs=False)
    assert rep["rows"]["123/relabel"]["diagnostic"]["self_cos_change_r21_29"] is None  # round 25 missing
    assert rep["rows"]["123/relabel"]["diagnostic"]["signed_residual_change_r21_29"] is None
    assert rep["trajectories"]["123/relabel"][25]["self_cos_change"] is None
    d = rep["rows"]["123/relabel"]["diagnostic"]
    assert d["reference_movement_on_others"] == {} and d["reference_movement_max_abs"] is None
    assert d["reference_movement_missing_inputs"] == [[25, "client_002"]]
    assert rep["rows"]["2026/drift"]["final_effect"] is None and rep["rows"]["2026/drift"]["pilot_top1_r49"] is None


def test_required_artifact_missing_fails_loudly(root: Path) -> None:
    (root / "results_rev/X4_drift_d123/final_test_metrics.json").unlink()
    with pytest.raises(FileNotFoundError):
        em.build(root, ["123"], with_logs=False)


def test_read_only_and_single_output(root: Path) -> None:
    before = _hashes(root)
    em.build(root, ["123", "2026"], with_logs=False)
    assert _hashes(root) == before
    out = root / "results_rev/pilot/evidence_matrix"
    assert [p.name for p in out.iterdir()] == ["evidence_matrix.json"]


def test_reconcile_detects_mismatch(root: Path) -> None:
    rep = em.build(root, ["123"], with_logs=False)
    assert em.reconcile(rep, root) == []
    p = root / "results_rev/pilot/phase_s"
    p.mkdir(parents=True)
    saved = {"attributed": "client_001", "honest_false_alarms": ["client_001"]}
    (p / "phase_s.json").write_text(json.dumps({"methods": {"Q-ZP-m": {"123/badnets": saved}}}))
    assert em.reconcile(rep, root) == ["123/badnets zp_norm: phase_s client_001/['client_001']"]


def test_classifier_concentration(monkeypatch) -> None:
    import torch

    w = torch.zeros(100, 4)
    w[7] = 3.0
    w[1] = 1.0
    monkeypatch.setattr(em, "update_geometry", lambda log, client, rounds: {r: {"fc.weight": w.clone()} for r in rounds})
    out = em.classifier_concentration(Path("unused"), "client_002", [5])
    assert out[5]["max_row_class"] == 7
    assert out[5]["max_row_share"] == pytest.approx(6.0 / math.sqrt(40.0))
    assert out[5]["energy_concentration"] == pytest.approx(0.9 ** 2 + 0.1 ** 2)
