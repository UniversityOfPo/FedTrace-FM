"""Phase S harness: fidelity to frozen v3, calibration discipline, scoring semantics, undefined handling."""

from __future__ import annotations

import random

import pytest

from src.rev import phase_s as ps
from src.rev.online import online_detect_v3

CLIENTS = ["client_000", "client_001", "client_002", "client_003", "client_004"]


def rows(seed=1, attacker_norm=None, attacker_t=None, rounds=40):
    rng = random.Random(seed)
    out = []
    for r in range(rounds):
        for i, c in enumerate(CLIENTS):
            n = 10 + i + rng.gauss(0, 0.2)
            t = None if r == 0 else 0.20 + 0.04 * i + rng.gauss(0, 0.01)
            if c == "client_002" and attacker_norm and r in attacker_norm:
                n = attacker_norm[r]
            if c == "client_002" and attacker_t and r in attacker_t:
                t = attacker_t[r]
            out.append({"round_id": r, "client_id": c, "update_l2_norm": n, "self_prev_round_cosine": t})
    return out


def test_protocol_constants() -> None:
    assert (ps.T_REF, ps.PERSISTENCE, ps.K_CUSUM, ps.DELTA_PH) == (20, 3, 0.5, 0.5)
    assert ps.GRID[0] == 0.25 and ps.GRID[-1] == 60.0 and len(ps.GRID) == 240
    assert ps.FEATURES == ("m", "t", "dA", "dB", "dC", "dD")


def test_group_v_inclusive_is_frozen_v3() -> None:
    data = rows(attacker_norm={r: 14.0 for r in range(25, 40)})
    alarms, _ = ps.v_alarms(data, "inclusive_median")
    online = online_detect_v3(data)
    assert {c: a[:2] for c, a in alarms.items()} == {c: (e.alarm_round, e.estimated_onset)
                                                   for c, e in online.per_client.items() if e.alarm_round is not None}
    assert ps.evaluate_run(alarms, "client_002", 20)["attributed"] == online.attributed_client


def test_calibration_depends_on_clean_runs_only() -> None:
    clean = [rows(seed=1), rows(seed=2)]
    h = ps.calibrate(clean, "CUSUM", "m")
    assert h is not None
    for c in clean:  # zero clean alarms by construction
        assert ps.q_alarms(c, "CUSUM", "m", h)[0] == {}
    # the function signature accepts no attacked run; recomputation is deterministic
    assert ps.calibrate(clean, "CUSUM", "m") == h


def test_detection_is_not_attribution_when_an_honest_client_is_first() -> None:
    alarms = {"client_004": (23, 21, 4.0), "client_001": (23, 21, 3.5), "client_002": (24, 22, 3.1)}
    ev = ps.evaluate_run(alarms, "client_002", 20)
    assert ev["first_alarmed"] == "client_004" and ev["attacker_detected"] and not ev["attribution_correct"]
    assert ev["honest_false_alarms"] == ["client_001", "client_004"] and ev["delay"] == 4 and ev["onset_error"] == 2
    missed = ps.evaluate_run({}, "client_002", 20)
    assert not missed["attacker_detected"] and missed["delay"] is None and missed["delay_censored_at"] == 29
    clean = ps.evaluate_run({"client_003": (40, 38, 1.0)}, None, None)
    assert clean["n_honest_false_alarms"] == 1 and "attacker_detected" not in clean


def test_undefined_calibration_is_reported_not_scored() -> None:
    data = [dict(r, update_l2_norm=5.0) if r["client_id"] == "client_001" else r for r in rows()]
    alarms, undefined = ps.q_alarms(data, "ZP", "m", 3.0)
    assert undefined == ["client_001"] and "client_001" not in alarms


def test_two_sided_detectors_catch_a_drop() -> None:
    data = rows(attacker_norm={r: 8.0 for r in range(25, 40)})  # attacker norm falls below its calibration
    for det in ("ZP", "CUSUM", "PH"):
        alarms, _ = ps.q_alarms(data, det, "m", 3.0)
        assert "client_002" in alarms, det


def test_residual_features_use_the_named_reference() -> None:
    data = rows()
    t = {(r["round_id"], r["client_id"]): r["self_prev_round_cosine"] for r in data}
    dA = ps.feature_series(data, "dA")["client_004"][25]
    import statistics as st

    assert dA == pytest.approx(st.median(t[(25, c)] for c in CLIENTS) - t[(25, "client_004")])
    dB = ps.feature_series(data, "dB")["client_004"][25]
    assert dB == pytest.approx(st.median(t[(25, c)] for c in CLIENTS if c != "client_004") - t[(25, "client_004")])
