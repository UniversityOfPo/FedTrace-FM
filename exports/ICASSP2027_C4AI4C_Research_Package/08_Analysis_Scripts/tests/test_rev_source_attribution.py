"""D1 source localization (protocol 14): synthetic data only — no development outcome is used here."""

from __future__ import annotations

import hashlib
import inspect
import math
import statistics
from pathlib import Path

import pytest

from src.rev import source_attribution as sa
from tests.test_rev_evidence_matrix import _series, _write_run

CLIENTS = [f"client_00{i}" for i in range(5)]


def trace(base, change=None, rounds=50, noise=0.004, norm_change=None):
    """Self-cosine base_i + deterministic jitter; ``change`` = {client: (from_round, new_level)}."""
    out = []
    for r in range(rounds):
        for i, c in enumerate(CLIENTS):
            t = None if r == 0 else base[i] + noise * math.sin(1.7 * r + 2.3 * i)
            if change and c in change and r >= change[c][0]:
                t = change[c][1] + noise * math.sin(1.7 * r + 2.3 * i)
            n = 10 + i + 0.1 * math.sin(r + i)
            if norm_change and c in norm_change and r >= norm_change[c][0]:
                n *= norm_change[c][1]
            out.append({"round_id": r, "client_id": c, "self_prev_round_cosine": t, "update_l2_norm": n})
    return out


BASE = [0.28, 0.32, 0.20, 0.36, 0.24]  # client_002 lowest, client_000 at the median


def test_ranking_inputs_exclude_identity_onset_and_counterfactual() -> None:
    assert list(inspect.signature(sa.analyse).parameters) == ["rows", "estimators", "eval_round"]
    assert list(inspect.signature(sa.per_round_scores).parameters) == ["rows", "estimator"]


def test_causal_scores_ignore_future_rounds() -> None:
    a = trace(BASE, change={"client_002": (20, 0.40)})
    b = [dict(r, self_prev_round_cosine=0.9 if r["round_id"] > 29 and r["self_prev_round_cosine"] else r["self_prev_round_cosine"],
              update_l2_norm=r["update_l2_norm"] * (3 if r["round_id"] > 29 else 1)) for r in a]
    for e in ("MED", "MEAN"):
        ra, rb = sa.analyse(a, (e,))[e], sa.analyse(b, (e,))[e]
        for m in sa.METHODS:
            assert ra[m]["at_eval_scores"] == rb[m]["at_eval_scores"], (e, m)


def test_permutation_equivariance_identity_not_used() -> None:
    a = trace(BASE, change={"client_002": (20, 0.40)})
    perm = {"client_000": "client_003", "client_001": "client_004", "client_002": "client_000",
            "client_003": "client_001", "client_004": "client_002"}
    b = [dict(r, client_id=perm[r["client_id"]]) for r in a]
    sa_, sb = sa.analyse(a, ("MED",))["MED"], sa.analyse(b, ("MED",))["MED"]
    for m in sa.METHODS:
        for c, v in sa_[m]["at_eval_scores"].items():
            other = sb[m]["at_eval_scores"][perm[c]]
            assert (v is None and other is None) or v == pytest.approx(other, abs=1e-12)


def test_source_moving_the_median_is_localized_by_B() -> None:
    res = sa.analyse(trace(BASE, change={"client_002": (20, 0.40)}), ("MED",))["MED"]
    assert res["B"]["at_eval"]["top"] == "client_002"
    assert res["B"]["at_eval_scores"]["client_002"] > 0


def test_stationary_run_has_small_influence_scores_calibration_absorbs_mechanical_offset() -> None:
    clean = sa.analyse(trace(BASE), ("MED",))["MED"]["B"]["at_eval_scores"]
    moved = sa.analyse(trace(BASE, change={"client_002": (20, 0.40)}), ("MED",))["MED"]["B"]["at_eval_scores"]
    assert max(abs(v) for v in clean.values() if v is not None) < 0.1 * moved["client_002"]


def test_A_and_B_differ_when_an_extreme_client_moves_further_out() -> None:
    # client_003 is already the highest; rising further changes its own residual (A) but not the median (B)
    res = sa.analyse(trace(BASE, change={"client_003": (20, 0.60)}), ("MED",))["MED"]
    assert res["A"]["at_eval"]["top"] == "client_003"
    assert abs(res["B"]["at_eval_scores"]["client_003"]) < 0.5
    assert res["SELFCOS"]["at_eval"]["top"] == "client_003"


def test_median_leave_one_out_reference_takes_three_values_for_k5() -> None:
    rows = trace(BASE)
    t = sa.series(rows, "self_prev_round_cosine")
    for r in range(20, 30):
        pos = sa.positions(t, r)
        refs = {c: statistics.median([t[r][k] for k in t[r] if k != c]) for c in t[r]}
        by_side = {}
        for c, p in pos.items():
            by_side.setdefault("low" if p < 3 else "mid" if p == 3 else "high", set()).add(round(refs[c], 15))
        assert all(len(v) == 1 for v in by_side.values())
        per = sa.per_round_scores(rows, "MED")["ABS"][r]
        low = [per[c] for c, p in pos.items() if p < 3]
        assert low[0] == pytest.approx(low[1], abs=1e-15)  # same-side candidates: identical displacement


def test_undefined_calibration_is_reported_and_ranked_last() -> None:
    rows = [dict(r, self_prev_round_cosine=0.5 if r["client_id"] == "client_001" and r["round_id"] else r["self_prev_round_cosine"])
            for r in trace([0.28, 0.5, 0.20, 0.36, 0.24])]
    res = sa.analyse(rows, ("MED",))["MED"]
    # client_001 is constant: own-change standardisation undefined
    assert res["SELFCOS"]["at_eval"]["undefined"] == ["client_001"]
    assert res["SELFCOS"]["at_eval"]["order"][-1] == "client_001"


def test_rank_ties_and_margins() -> None:
    r = sa.rank({"client_001": 1.0, "client_000": 1.0, "client_002": 0.5, "client_003": None})
    assert r["order"] == ["client_000", "client_001", "client_002", "client_003"]
    assert r["exact_tie_at_top"] and r["tie_groups"] == [["client_000", "client_001"]] and r["undefined"] == ["client_003"]
    r = sa.rank({"client_000": 1.0, "client_001": 0.995, "client_002": 0.2})
    assert not r["exact_tie_at_top"] and r["near_tie_at_top"] and r["margin"] == pytest.approx(0.005)


def test_aggregation_rules_for_displacement() -> None:
    pr = {"ABS": {20: {"a": 1.0}, 21: {"a": -1.0}}, "SIGNED": {20: {"a": 1.0}, 21: {"a": -1.0}}}
    assert sa.cumulative(pr, "ABS", 21) == {"a": 1.0}
    assert sa.cumulative(pr, "SIGNED", 21) == {"a": 0.0}


def test_evaluation_is_separate_and_kendall() -> None:
    res = sa.analyse(trace(BASE, change={"client_002": (20, 0.40)}), ("MED",))
    ev = sa.evaluate(res, "client_002")
    assert ev["MED"]["B"]["rank_of_true_source"] == 1 and ev["MED"]["B"]["top_is_true_source"]
    assert sa.evaluate(res, None)["MED"]["B"]["top_is_true_source"] is None
    assert sa.kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert sa.kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_stability_and_positions_reported() -> None:
    st = sa.analyse(trace(BASE, change={"client_002": (20, 0.40)}), ("MED",))["MED"]["B"]["stability"]
    assert st["window_rounds"] == 10 and sum(st["per_round_top_position_counts"].values()) <= 10
    assert st["cumulative_top_unchanged_from_round"] is not None


def test_run_is_read_only(tmp_path: Path) -> None:
    for s in ("123", "2026"):
        for cond, pattern in sa.CONDITIONS.items():
            _write_run(tmp_path, pattern.format(s=s), _series(s, cond), 0.5)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    out = sa.run(tmp_path, ["123", "2026"])
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file() and "source_attribution" not in p.parts}
    assert after == before
    assert set(out["verdicts"]) == {"F1_motivating_case", "F2_vs_ABS", "F2_vs_SIGNED", "F3_clean_winner",
                                    "F4_unstable_runs", "F5_estimator_dependent"}
