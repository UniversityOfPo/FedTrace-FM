"""Read-only pair diagnostics: consistency with frozen v3, causality, and the preregistered verdict rules."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from src.analysis.fedtrace_detector import compute_all_scores_v3
from src.rev import diagnostics as dg

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = ("client_000", "client_001", "client_002", "client_003", "client_004")


def rows(attacker_cos=None, attacker_norm=None, clean_dip=None, seed=3, rounds=40):
    """Synthetic trace rows; attacker_* map round -> value overrides for client_002."""
    rng = random.Random(seed)
    out = []
    for r in range(rounds):
        for c in CLIENTS:
            norm = 10.0 + rng.gauss(0, 0.2) + (3.0 if r < 2 else 0.0)  # burn-in transient in rounds 0-1
            cos = None if r == 0 else 0.30 + rng.gauss(0, 0.02)
            if c == "client_002" and attacker_cos and r in attacker_cos:
                cos = attacker_cos[r]
            if c == "client_002" and attacker_norm and r in attacker_norm:
                norm = attacker_norm[r]
            if clean_dip and c == clean_dip[0] and r in clean_dip[1]:
                cos = 0.60
            out.append({"round_id": r, "client_id": c, "update_l2_norm": norm, "self_prev_round_cosine": cos})
    return out


def test_signed_residual_is_v3s_residual_before_clipping() -> None:
    data = rows(attacker_cos={r: 0.55 for r in range(20, 40)})
    res = dg.signed_residuals(data)
    for s in compute_all_scores_v3(data):
        if s.round_id >= dg.T_REF and s.z_t_res is not None:
            assert s.z_t_res == pytest.approx(max(0.0, res[s.client_id][s.round_id]["signed_z"]))
    assert res["client_002"][25]["signed_z"] < -3 and res["client_002"][5]["kind"] == "retrospective"


def test_signed_residual_is_causal_from_t_ref() -> None:
    data = rows(attacker_cos={r: 0.55 for r in range(20, 40)})
    full = dg.signed_residuals(data)
    for cut in (20, 27, 35):
        prefix = dg.signed_residuals([r for r in data if r["round_id"] <= cut])
        for c in CLIENTS:
            assert prefix[c][cut]["signed_z"] == full[c][cut]["signed_z"]


def test_observation_d_supported_falsified_and_inconclusive() -> None:
    clean = dg.signed_residuals(rows())
    supported = dg.observation_d_verdict(clean, dg.signed_residuals(rows(attacker_cos={r: 0.55 for r in range(20, 40)})),
                                         "client_002", 20, 39)
    assert supported["verdict"] == "SUPPORTED" and supported["S1"] and supported["S2"]
    no_shift = dg.observation_d_verdict(clean, dg.signed_residuals(rows()), "client_002", 20, 39)
    assert no_shift["verdict"] == "FALSIFIED" and no_shift["F1"]
    benign = dg.signed_residuals(rows(clean_dip=("client_004", range(30, 34))))
    attacked = dg.signed_residuals(rows(attacker_cos={r: 0.55 for r in range(20, 40)}))
    assert dg.observation_d_verdict(benign, attacked, "client_002", 20, 39)["verdict"] == "FALSIFIED"  # F2
    weak = dg.signed_residuals(rows(attacker_cos={r: 0.33 + (0.2 if r in (21, 23, 25, 27) else 0.0) for r in range(20, 40)}))
    verdict = dg.observation_d_verdict(clean, weak, "client_002", 20, 39)
    assert verdict["verdict"] == "INCONCLUSIVE" and verdict["S1"] and not verdict["S2"] and not verdict["F1"]


def test_norm_verdict_thresholds() -> None:
    clean = rows()
    for factor, expected in ((0.80, "REPLICATED"), (0.93, "PARTIAL"), (0.99, "NOT_REPLICATED"), (1.20, "NOT_REPLICATED (INCREASE)")):
        attacked = [dict(r) for r in clean]
        for r in attacked:
            if r["client_id"] == "client_002" and r["round_id"] >= 20:
                r["update_l2_norm"] *= factor
        verdict = dg.norm_verdict(dg.norm_ratios(clean, attacked), "client_002", 20)
        assert verdict["verdict"] == expected and verdict["attacker_mean_ratio"] == pytest.approx(factor)


def test_calibration_spread_and_verdict() -> None:
    spread = dg.calibration_spread(rows(), (2, 1, 3))
    assert all(v["sd_ratio_b2"] > 1.5 for v in spread.values())
    assert dg.spread_verdict(spread)["verdict"] == "REPLICATED"
    flat = dg.calibration_spread([dict(r, update_l2_norm=10.0 + 0.1 * (r["round_id"] % 3)) for r in rows()], (2,))
    assert dg.spread_verdict(flat)["verdict"] == "NOT_REPLICATED"


def test_asr_classification_boundaries() -> None:
    assert dg.asr_classification(0.50, 0.01) == "EFFECTIVE"
    assert dg.asr_classification(0.4354, 0.0073) == "PARTIALLY_EFFECTIVE"
    assert dg.asr_classification(0.08, 0.01) == "INEFFECTIVE"


def test_v3_round_table_streak_and_alarm_match_online_v3() -> None:
    from src.rev.online import online_detect_v3

    data = rows(attacker_norm={r: 14.0 for r in range(25, 40)})
    table = dg.v3_round_table(data, 20, 39)
    alarm = online_detect_v3(data).per_client["client_002"].alarm_round
    first_alarm_row = min(t["round"] for t in table if t["client"] == "client_002" and t["alarm_state"])
    assert first_alarm_row == alarm


def test_preregistered_constants_are_fixed() -> None:
    assert {k: getattr(dg, k) for k in dg.PREREGISTERED} == {
        "T_REF": 20, "Z_CUT": 3.0, "RUN_LENGTH": 3, "EARLY_WINDOW": 3, "POST_WINDOW": 10, "FALSIFY_MEDIAN_Z": -1.0,
        "NORM_WINDOW": (5, 15), "NORM_REPLICATED_MAX": 0.90, "NORM_NOT_REPLICATED_MIN": 0.97,
        "HONEST_RATIO_BAND": (0.97, 1.03), "BURN_IN_PRIMARY": 2, "BURN_IN_EXPLORATORY": (1, 3, 5),
        "SPREAD_RATIO_CUT": 1.5, "ASR_EFFECTIVE": 0.50, "ASR_PARTIAL_MIN_GAIN": 0.10}


def test_cli_refuses_missing_runs() -> None:
    with pytest.raises(SystemExit):
        dg.main(["diagnose", "--clean", "no_such_run", "--attacked", "no_such_run_2", "--malicious", "c", "--onset", "20"])


def test_exploratory_reproduces_recorded_pilot_numbers() -> None:
    path = ROOT / "results_rev/pilot/X2_badnets_d123__vs__X1_clean_d123/diagnostics_posthoc/exploratory_posthoc.json"
    if not path.exists():
        pytest.skip("pilot P1 outputs not available on this machine")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["CUSUM_norm"]["threshold_from_clean"] == 8.75 and report["CUSUM_norm"]["alarm"] == 25
    assert report["onset_at_v3_alarm"]["X2_badnets_d123"]["v3_streak_onset"] == 28
