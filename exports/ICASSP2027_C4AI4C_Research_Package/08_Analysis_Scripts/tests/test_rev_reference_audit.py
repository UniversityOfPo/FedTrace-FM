"""Reference-contamination audit: reference definitions, decomposition, causality, read-only behaviour, reproducibility."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from src.analysis.fedtrace_detector import compute_all_scores_v3
from src.rev import reference_audit as ra
from src.rev.hashing import sha256_file
from src.rev.online import online_detect_v3
from src.rev.update_log import UpdateLogConfig, UpdateLogger

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = ["c0", "c1", "c2", "c3", "c4"]


def make_rows(attacker_t=None, seed=1, rounds=40, base=(0.20, 0.23, 0.27, 0.32, 0.38)):
    rng = random.Random(seed)
    rows = []
    for r in range(rounds):
        for c, b in zip(CLIENTS, base):
            t = None if r == 0 else b + rng.gauss(0, 0.01)
            if attacker_t and c == "c2" and r in attacker_t:
                t = attacker_t[r]
            rows.append({"round_id": r, "client_id": c, "self_prev_round_cosine": t,
                         "update_l2_norm": 10 + rng.gauss(0, 0.2)})
    return rows


def test_reference_definitions_hand_computed() -> None:
    t = {(5, c): v for c, v in zip(CLIENTS, (0.1, 0.2, 0.3, 0.4, 0.9))}
    assert ra.inclusive_median(t, 5, "c4", CLIENTS) == pytest.approx(0.3)
    assert ra.leave_one_out_median(t, 5, "c4", CLIENTS) == pytest.approx(0.25)   # median of 0.1, 0.2, 0.3, 0.4
    assert ra.leave_one_out_median(t, 5, "c2", CLIENTS) == pytest.approx(0.3)    # median of 0.1, 0.2, 0.4, 0.9
    assert ra.trimmed_mean(t, 5, "c0", CLIENTS) == pytest.approx(0.3)            # mean of 0.2, 0.3, 0.4
    hist = {(r, "c1"): 0.1 * r for r in range(1, 20)}
    assert ra.historical_own_mean(hist, 25, "c1", CLIENTS) == pytest.approx(1.0)  # mean of 0.1 ... 1.9


def test_inclusive_residual_is_v3s_before_clipping() -> None:
    rows = make_rows(attacker_t={r: 0.45 for r in range(20, 40)})
    res = ra.residuals(rows, ra.inclusive_median)
    for s in compute_all_scores_v3(rows):
        if s.round_id >= ra.T_REF and s.z_t_res is not None:
            assert s.z_t_res == pytest.approx(max(0.0, res[s.client_id][s.round_id]["z"]))


def test_inclusive_detection_reproduces_online_v3() -> None:
    rows = make_rows(attacker_t={r: 0.60 for r in range(20, 40)})
    det = ra.detect(rows, ra.residuals(rows, ra.inclusive_median))
    online = online_detect_v3(rows)
    assert det["alarms"] == {c: (e.alarm_round, e.estimated_onset) for c, e in online.per_client.items() if e.alarm_round is not None}
    assert det["attributed"] == online.attributed_client


@pytest.mark.parametrize("name", list(ra.REFERENCES))
def test_every_reference_is_causal_from_t_ref(name: str) -> None:
    rows = make_rows(attacker_t={r: 0.45 for r in range(20, 40)})
    full = ra.residuals(rows, ra.REFERENCES[name])
    for cut in (20, 26, 33):
        prefix = ra.residuals([r for r in rows if r["round_id"] <= cut], ra.REFERENCES[name])
        for c in CLIENTS:
            if full[c]:  # a client with zero calibration sd has no z at all (as in frozen v3)
                assert prefix[c][cut]["z"] == full[c][cut]["z"]
            else:
                assert prefix[c] == {}


def test_decomposition_identity_and_attacker_attribution() -> None:
    clean = make_rows()
    attacked = make_rows(attacker_t={r: 0.45 for r in range(20, 40)})  # only the attacker's t differs
    dec = ra.decompose(clean, attacked, "c2", range(20, 30))
    assert any(dec.values())
    for k, rows in dec.items():
        for row in rows:
            assert row["delta_z"] == pytest.approx(row["from_reference"] + row["from_own_update"], abs=1e-12)
            if k != "c2":
                assert row["from_own_update"] == 0.0
                assert row["attacker_induced"] == pytest.approx(row["from_reference"], abs=1e-12)
    # the attacker was the median client (base 0.27): honest residuals rise only through the reference
    assert dec["c0"] and all(r["from_reference"] > 0 for r in dec["c0"])


def test_pre_onset_rank() -> None:
    assert ra.pre_onset_rank(make_rows(), "c2", range(15, 20)) == [3] * 5
    assert ra.pre_onset_rank(make_rows(), "c4", range(15, 20)) == [5] * 5


def _toy_log(root: Path) -> Path:
    torch.manual_seed(0)
    logger = UpdateLogger.open(UpdateLogConfig(root=root, run_id="t", min_free_bytes=0))
    lin = torch.nn.Linear(4, 100)
    g = {"fc.weight": lin.weight.detach().clone(), "fc.bias": lin.bias.detach().clone(),
         "layer4.w": torch.randn(6), "bn.running_mean": torch.zeros(3)}
    for r in range(4):
        sub = {k: v + 0.01 * (r + 1) * torch.randn_like(v) for k, v in g.items()}
        logger.log_client("c2", r, sub, g, num_samples=1)
        g = {k: v + 0.001 for k, v in sub.items()}
        logger.log_round_end(r, g, "completed", ("c2",))
    logger.seal()
    return root


def test_update_geometry_is_read_only_and_excludes_running_stats(tmp_path: Path) -> None:
    log = _toy_log(tmp_path / "log")
    before = {p.name: sha256_file(p) for p in log.rglob("*") if p.is_file()}
    d = ra.update_geometry(log, "c2", range(1, 4))
    after = {p.name: sha256_file(p) for p in log.rglob("*") if p.is_file()}
    assert before == after
    assert set(d[1]) == {"fc.weight", "fc.bias", "layer4.w"}
    rows = ra.geometry_summary(d, d)
    assert rows[0]["self_cos_attacked"] == pytest.approx(rows[0]["self_cos_clean"])
    assert 0 < rows[0]["target_row_norm_share_attacked"] < 1


def test_decomposition_is_deterministic() -> None:
    clean, attacked = make_rows(), make_rows(attacker_t={r: 0.45 for r in range(20, 40)})
    assert ra.decompose(clean, attacked, "c2", range(20, 30)) == ra.decompose(clean, attacked, "c2", range(20, 30))


def test_reproduces_recorded_development_audit() -> None:
    path = ROOT / "results_rev/pilot/reference_audit_posthoc/reference_audit_posthoc.json"
    if not path.exists() or not (ROOT / "results_rev/X2_badnets_d2026/client_round_trace.csv").exists():
        pytest.skip("development audit outputs not available on this machine")
    recorded = json.loads(path.read_text(encoding="utf-8"))["task3_decomposition_rounds20_29"]
    for seed in ("123", "2026"):
        clean = ra.load_trace_rows(ROOT / f"results_rev/X1_clean_d{seed}/client_round_trace.csv")
        attacked = ra.load_trace_rows(ROOT / f"results_rev/X2_badnets_d{seed}/client_round_trace.csv")
        for k, rows in ra.decompose(clean, attacked, "client_002", range(20, 30)).items():
            mean = sum(r["delta_z"] for r in rows) / len(rows)
            assert mean == pytest.approx(recorded[seed][k]["mean_delta_z"], abs=1e-12)


def test_zero_calibration_sd_is_reported_undefined_not_crashing() -> None:
    rows = []
    for r in range(30):
        for c, b in zip(CLIENTS, (0.1, 0.2, 0.3, 0.4, 0.5)):
            rows.append({"round_id": r, "client_id": c, "self_prev_round_cosine": None if r == 0 else b,
                         "update_l2_norm": 10.0 + 0.1 * (r % 3)})
    res = ra.residuals(rows, ra.inclusive_median)
    assert res["c2"] == {}  # constant median client: d == 0 in every calibration round
    assert ra.decompose(rows, rows, "c4", range(20, 25))["c2"] == []
