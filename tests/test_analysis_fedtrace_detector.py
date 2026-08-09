"""Focused tests for the offline FedTrace evaluator (src/analysis/fedtrace_detector.py).

Uses small synthetic in-memory row dicts throughout (never a real
``client_round_trace.csv`` file, never a dataset/dataloader/GPU) -- mirrors
the synthetic-data style already used by
``tests/test_attacks_traceability.py``. Per the Stage A / v2- / v3-
implementation tasks' explicit restrictions, these tests are written but
deliberately **not executed** in any of them.

The v1 tests (everything above the "FedTrace v2" section marker) are
unmodified from Stage A. The v2 tests (between the v2 and v3 markers)
exercise only the v2 additions (the robust cross-sectional temporal
calibration, ``z_t_rel``, ``ScoredRowV2``, ``apply_persistence_rule_v2``,
``evaluate_detector_v2``) and are unmodified from the v2 task. The v3 tests
below the "FedTrace v3" marker exercise only the new v3 additions (the
common-round median reference, the historical residual calibration,
``z_t_res``, ``ScoredRowV3``, ``apply_persistence_rule_v3``,
``evaluate_detector_v3``), plus regression checks that v1's and v2's own
evaluators are unaffected by v3's addition.
"""

from __future__ import annotations

from src.analysis.fedtrace_detector import (
    COMPONENT_FIELDNAMES,
    DEFAULT_MAD_SCALE_CONSTANT,
    V2_COMPONENT_FIELDNAMES,
    V3_COMPONENT_FIELDNAMES,
    ClientCalibration,
    apply_persistence_rule_v2,
    apply_persistence_rule_v3,
    compute_all_scores,
    compute_all_scores_v2,
    compute_all_scores_v3,
    compute_common_round_reference,
    compute_cross_sectional_calibration,
    compute_cross_sectional_robust_calibration,
    compute_historical_residual_calibration,
    compute_longitudinal_calibration,
    evaluate_detector,
    evaluate_detector_v2,
    evaluate_detector_v3,
    evaluate_naive_baseline,
    field_available,
)


def _row(
    round_id: int,
    client_id: str,
    *,
    malicious_client_id: str = "client_002",
    attack_start_round: int = 20,
    attack_configured: bool = True,
    m: float | None = None,
    p: float | None = None,
    t: float | None = None,
    naive_flag: bool = False,
) -> dict:
    return {
        "round_id": round_id,
        "client_id": client_id,
        "attack_configured": attack_configured,
        "malicious_client_id": malicious_client_id,
        "attack_start_round": attack_start_round,
        "update_l2_norm": m,
        "peer_mean_cosine_similarity": p,
        "self_prev_round_cosine": t,
        "naive_flag": naive_flag,
    }


# ----------------------------------------------------------------------
# Calibration helpers
# ----------------------------------------------------------------------


def test_longitudinal_calibration_uses_only_pre_t_ref_rows_for_this_client() -> None:
    rows = [
        _row(0, "client_000", m=6.0),
        _row(1, "client_000", m=6.2),
        _row(19, "client_000", m=5.8),
        _row(20, "client_000", m=100.0),  # must be excluded (round_id >= t_ref)
        _row(0, "client_001", m=999.0),  # must be excluded (different client)
    ]
    calibration = compute_longitudinal_calibration(rows, "client_000", "update_l2_norm", t_ref=20)
    assert calibration is not None
    assert calibration.n == 3
    assert 5.9 < calibration.mean < 6.1


def test_longitudinal_calibration_returns_none_below_min_observations() -> None:
    rows = [_row(0, "client_000", m=6.0)]
    assert compute_longitudinal_calibration(rows, "client_000", "update_l2_norm", t_ref=20) is None


def test_longitudinal_calibration_returns_none_for_zero_variance() -> None:
    rows = [_row(r, "client_000", m=6.0) for r in range(5)]
    assert compute_longitudinal_calibration(rows, "client_000", "update_l2_norm", t_ref=20) is None


def test_cross_sectional_calibration_pools_across_clients_at_one_round() -> None:
    rows = [
        _row(20, "client_000", p=0.10),
        _row(20, "client_001", p=0.12),
        _row(20, "client_002", p=-0.50),  # the outlier this round
        _row(21, "client_000", p=999.0),  # must be excluded (different round)
    ]
    calibration = compute_cross_sectional_calibration(rows, 20, "peer_mean_cosine_similarity")
    assert calibration is not None
    assert calibration.n == 3


def test_cross_sectional_calibration_has_no_historical_window() -> None:
    """Unlike the longitudinal calibrators, round 20 stats must not see round 0-19 data at all.

    Round-20 values must be non-identical: compute_cross_sectional_calibration
    intentionally returns None when the same-round sample has zero variance
    (matching compute_longitudinal_calibration's own degenerate-case rule) --
    using two distinct round-20 values keeps that safeguard intact while still
    proving the round-0-19 values (an absurd 1000.0, impossible to average in
    undetected) are never read.
    """
    rows = [_row(r, "client_000", p=1000.0) for r in range(20)] + [
        _row(20, "client_000", p=0.10),
        _row(20, "client_001", p=0.12),
    ]
    calibration = compute_cross_sectional_calibration(rows, 20, "peer_mean_cosine_similarity")
    assert calibration is not None
    assert abs(calibration.mean - 0.11) < 1e-9


# ----------------------------------------------------------------------
# field_available / NOT EVALUABLE
# ----------------------------------------------------------------------


def test_field_available_false_when_field_entirely_missing_from_rows() -> None:
    rows = [_row(0, "client_000", m=6.0)]
    del rows[0]["self_prev_round_cosine"]
    assert field_available(rows, "self_prev_round_cosine") is False


def test_field_available_false_when_field_present_but_always_none() -> None:
    rows = [_row(0, "client_000", m=6.0, t=None)]
    assert field_available(rows, "self_prev_round_cosine") is False


def test_field_available_true_when_at_least_one_value_present() -> None:
    rows = [_row(0, "client_000", m=6.0, t=None), _row(1, "client_000", m=6.1, t=0.3)]
    assert field_available(rows, "self_prev_round_cosine") is True


def test_evaluate_detector_not_evaluable_when_required_signal_entirely_missing() -> None:
    """Mirrors EXP_014's trace file: requesting the 't' component must fail loudly, not silently."""
    rows = [_row(r, "client_000", m=6.0) for r in range(25)]
    for row in rows:
        del row["self_prev_round_cosine"]
    result = evaluate_detector(rows, "C_temporal_only", ("t",))
    assert result.evaluable is False
    assert result.not_evaluable_reason is not None
    assert "self_prev_round_cosine" in result.not_evaluable_reason


# ----------------------------------------------------------------------
# Persistence rule: first-anomalous vs confirmation vs onset-estimate
# ----------------------------------------------------------------------


def _build_two_client_magnitude_run(malicious_m_by_round: dict[int, float]) -> list[dict]:
    """20 honest pre-attack rounds (m~6.0 for both clients) then a caller-specified magnitude
    trajectory for the malicious client from round 20 onward; the honest client stays flat."""
    rows = []
    for r in range(20):
        rows.append(_row(r, "client_000", m=6.0 + 0.01 * (r % 3)))
        rows.append(_row(r, "client_002", m=6.0 + 0.01 * (r % 2)))
    for r in range(20, 30):
        rows.append(_row(r, "client_000", m=6.0))
        rows.append(_row(r, "client_002", m=malicious_m_by_round.get(r, 6.0)))
    return rows


def test_persistence_requires_three_consecutive_anomalous_rounds() -> None:
    """A single-round or two-round spike must NOT confirm; three consecutive rounds must."""
    rows = _build_two_client_magnitude_run(
        {20: 6.0, 21: 50.0, 22: 6.0, 23: 6.0, 24: 6.0}  # isolated single-round spike only
    )
    result = evaluate_detector(rows, "B_magnitude_only", ("m",))
    assert result.evaluable is True
    client_002_result = result.per_client["client_002"]
    assert client_002_result.first_anomalous_round == 21
    assert client_002_result.confirmation_round is None  # never confirmed: not persistent
    assert client_002_result.onset_estimate is None


def test_persistence_confirms_on_third_consecutive_anomalous_round() -> None:
    rows = _build_two_client_magnitude_run({r: 50.0 for r in range(20, 30)})
    result = evaluate_detector(rows, "B_magnitude_only", ("m",))
    client_002_result = result.per_client["client_002"]
    assert client_002_result.first_anomalous_round == 20
    assert client_002_result.onset_estimate == 20  # first round of the persistent run
    assert client_002_result.confirmation_round == 22  # onset + persistence - 1


def test_onset_estimate_uses_the_first_confirmed_run_not_a_later_isolated_flicker() -> None:
    """A pre-run isolated flicker must not become the onset estimate once a later run confirms."""
    rows = _build_two_client_magnitude_run(
        {20: 50.0, 21: 6.0, 22: 50.0, 23: 50.0, 24: 50.0, 25: 50.0}
    )
    result = evaluate_detector(rows, "B_magnitude_only", ("m",))
    client_002_result = result.per_client["client_002"]
    assert client_002_result.first_anomalous_round == 20  # raw first, isolated
    assert client_002_result.onset_estimate == 22  # first round of the confirmed run
    assert client_002_result.confirmation_round == 24


# ----------------------------------------------------------------------
# Attribution / metrics
# ----------------------------------------------------------------------


def test_attribution_picks_earliest_confirmed_client() -> None:
    rows = []
    for r in range(20):
        rows.append(_row(r, "client_000", m=6.0 + 0.01 * (r % 3)))
        rows.append(_row(r, "client_002", m=6.0 + 0.01 * (r % 2)))
    # client_002 (the true malicious client) confirms first, at round 22;
    # client_000 later drifts and also confirms, at round 30 -- must not win attribution.
    for r in range(20, 40):
        m_002 = 50.0 if r < 30 else 6.0
        m_000 = 6.0 if r < 27 else 50.0
        rows.append(_row(r, "client_002", m=m_002))
        rows.append(_row(r, "client_000", m=m_000))
    result = evaluate_detector(rows, "B_magnitude_only", ("m",))
    assert result.predicted_client == "client_002"
    assert result.attribution_correct is True
    assert "client_000" in result.false_positive_clients
    assert result.true_positive_clients == ("client_002",)


def test_perfect_detection_yields_precision_recall_f1_of_one() -> None:
    rows = _build_two_client_magnitude_run({r: 50.0 for r in range(20, 30)})
    result = evaluate_detector(rows, "B_magnitude_only", ("m",))
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.onset_localization_error == 0
    assert result.detection_delay == 2


def test_false_negative_when_malicious_client_never_confirmed() -> None:
    rows = _build_two_client_magnitude_run({})  # malicious client stays at baseline, never flagged
    result = evaluate_detector(rows, "B_magnitude_only", ("m",))
    assert result.predicted_client is None
    assert result.attribution_correct is False
    assert result.false_negative is True
    assert result.recall == 0.0


# ----------------------------------------------------------------------
# Max-combiner: missing component never substitutes as zero
# ----------------------------------------------------------------------


def test_combined_score_ignores_missing_component_never_treats_it_as_zero() -> None:
    """If t is undefined for a row, max(m, p, t) must equal max(m, p), not max(m, p, 0)."""
    rows = []
    for r in range(25):
        # Two clients (so cross-sectional p calibration has n=2) with slightly
        # varying m (so longitudinal m calibration has nonzero std); t always None.
        rows.append(_row(r, "client_000", m=6.0 + 0.01 * (r % 3), p=0.10 + 0.001 * (r % 2), t=None))
        rows.append(_row(r, "client_001", m=5.9 + 0.02 * (r % 2), p=0.09 - 0.001 * (r % 2), t=None))
    scored = compute_all_scores(rows, t_ref=20)
    row20 = next(s for s in scored if s.round_id == 20 and s.client_id == "client_000")
    assert row20.z_m is not None and row20.z_p is not None and row20.z_t is None
    combined_mpt = row20.combined_score(("m", "p", "t"))
    combined_mp = row20.combined_score(("m", "p"))
    assert combined_mpt == combined_mp  # t's absence must not pull the max toward 0


def test_dominant_component_counts_only_count_anomalous_rounds() -> None:
    rows = _build_two_client_magnitude_run({r: 50.0 for r in range(20, 30)})
    result = evaluate_detector(rows, "D_magnitude_peer", ("m", "p"))
    assert result.dominant_component_counts["m"] >= 1
    assert sum(result.dominant_component_counts.values()) == result.dominant_component_counts["m"]


# ----------------------------------------------------------------------
# Detector A wrapper (existing naive baseline)
# ----------------------------------------------------------------------


def test_naive_baseline_wrapper_reflects_existing_naive_flag_column() -> None:
    rows = [_row(r, "client_000", naive_flag=False) for r in range(20)]
    rows += [_row(r, "client_002", naive_flag=False) for r in range(20)]
    rows += [_row(r, "client_000", naive_flag=False) for r in range(20, 30)]
    rows += [
        _row(r, "client_002", naive_flag=(r >= 25)) for r in range(20, 30)
    ]
    result = evaluate_naive_baseline(rows)
    assert result.predicted_client == "client_002"
    assert result.per_client["client_002"].first_anomalous_round == 25


def test_naive_baseline_wrapper_never_detects_when_naive_flag_always_false() -> None:
    rows = [_row(r, "client_000") for r in range(30)] + [_row(r, "client_002") for r in range(30)]
    result = evaluate_naive_baseline(rows)
    assert result.predicted_client is None
    assert result.attribution_correct is False
    assert result.false_negative is True


def test_component_fieldnames_cover_all_three_signal_families() -> None:
    assert set(COMPONENT_FIELDNAMES) == {"m", "p", "t"}


# ============================================================================
# FedTrace v2 (research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md)
# ============================================================================


def _row_v2(round_id: int, client_id: str, *, t: float | None, m: float | None = 6.0) -> dict:
    return _row(round_id, client_id, m=m, t=t)


# ----------------------------------------------------------------------
# compute_cross_sectional_robust_calibration: median / MAD / 1.4826 scaling
# ----------------------------------------------------------------------


def test_robust_calibration_computes_correct_median() -> None:
    """5 clients, odd n, so the median is exactly the literal middle value."""
    rows = [
        _row_v2(10, "client_000", t=0.10),
        _row_v2(10, "client_001", t=0.30),
        _row_v2(10, "client_002", t=0.20),
        _row_v2(10, "client_003", t=0.50),
        _row_v2(10, "client_004", t=0.40),
    ]
    calibration = compute_cross_sectional_robust_calibration(rows, 10, "self_prev_round_cosine")
    assert calibration is not None
    assert calibration.mean == 0.30  # center = median of {0.1,0.2,0.3,0.4,0.5} = 0.3
    assert calibration.n == 5


def test_robust_calibration_mad_and_1_4826_scaling() -> None:
    """Deviations from median 0.30 are {0.20, 0.10, 0, 0.20, 0.10} -> MAD = 0.10."""
    rows = [
        _row_v2(10, "client_000", t=0.10),
        _row_v2(10, "client_001", t=0.30),
        _row_v2(10, "client_002", t=0.20),
        _row_v2(10, "client_003", t=0.50),
        _row_v2(10, "client_004", t=0.40),
    ]
    calibration = compute_cross_sectional_robust_calibration(rows, 10, "self_prev_round_cosine")
    assert calibration is not None
    # deviations sorted: 0, 0.10, 0.10, 0.20, 0.20 -> median (3rd) = 0.10 -> MAD = 0.10
    # calibration.std stores the *scaled* value (1.4826 * MAD), not the raw MAD itself.
    assert DEFAULT_MAD_SCALE_CONSTANT == 1.4826
    implied_mad = calibration.std / DEFAULT_MAD_SCALE_CONSTANT
    assert abs(implied_mad - 0.10) < 1e-9
    assert abs(calibration.std - 1.4826 * 0.10) < 1e-9


def test_robust_calibration_has_no_historical_window() -> None:
    """Unlike z_m, z_t_rel must not see any round other than the one requested."""
    rows = [_row_v2(r, "client_000", t=999.0) for r in range(20)] + [
        _row_v2(20, "client_000", t=0.30),
        _row_v2(20, "client_001", t=0.32),
    ]
    calibration = compute_cross_sectional_robust_calibration(
        rows, 20, "self_prev_round_cosine"
    )
    assert calibration is not None
    assert abs(calibration.mean - 0.31) < 1e-9  # median of {0.30, 0.32}, not contaminated by round<20


def test_robust_calibration_none_when_fewer_than_two_clients() -> None:
    rows = [_row_v2(10, "client_000", t=0.30)]
    assert compute_cross_sectional_robust_calibration(rows, 10, "self_prev_round_cosine") is None


def test_robust_calibration_none_when_mad_is_zero() -> None:
    """4 identical values + 1 outlier -> median absolute deviation is 0 (a tie among the majority)."""
    rows = [
        _row_v2(10, "client_000", t=0.30),
        _row_v2(10, "client_001", t=0.30),
        _row_v2(10, "client_002", t=0.30),
        _row_v2(10, "client_003", t=0.30),
        _row_v2(10, "client_004", t=0.90),
    ]
    assert compute_cross_sectional_robust_calibration(rows, 10, "self_prev_round_cosine") is None


def test_robust_calibration_never_substitutes_epsilon_for_zero_mad() -> None:
    """A degenerate round must propagate as None all the way to z_t_rel, not silently divide by a floor."""
    rows = [_row_v2(10, cid, t=0.30) for cid in ("client_000", "client_001", "client_002")]
    scored = compute_all_scores_v2(rows, t_ref=20)
    row10 = next(s for s in scored if s.round_id == 10 and s.client_id == "client_000")
    assert row10.z_t_rel is None


# ----------------------------------------------------------------------
# One-sided z_t_rel behavior
# ----------------------------------------------------------------------


def test_z_t_rel_is_zero_for_clients_above_the_median_not_negative() -> None:
    rows = [
        _row_v2(10, "client_000", t=0.10),  # below median -> positive z
        _row_v2(10, "client_001", t=0.30),  # median itself -> z = 0
        _row_v2(10, "client_002", t=0.20),
        _row_v2(10, "client_003", t=0.50),  # well above median -> z = 0, not negative
        _row_v2(10, "client_004", t=0.40),
    ]
    scored = compute_all_scores_v2(rows, t_ref=20)
    by_client = {s.client_id: s for s in scored if s.round_id == 10}
    assert by_client["client_000"].z_t_rel > 0  # below median: nonzero
    assert by_client["client_001"].z_t_rel == 0.0  # exactly at median
    assert by_client["client_003"].z_t_rel == 0.0  # above median: clipped to 0, never negative
    assert by_client["client_004"].z_t_rel == 0.0


def test_z_t_rel_never_uses_future_rounds() -> None:
    """A later round's values must not influence an earlier round's center/scale."""
    rows = [
        _row_v2(10, "client_000", t=0.10),
        _row_v2(10, "client_001", t=0.30),
        _row_v2(10, "client_002", t=0.20),
    ]
    calibration_before = compute_cross_sectional_robust_calibration(
        rows, 10, "self_prev_round_cosine"
    )
    rows_with_future = rows + [
        _row_v2(11, "client_000", t=999.0),
        _row_v2(11, "client_001", t=-999.0),
    ]
    calibration_after = compute_cross_sectional_robust_calibration(
        rows_with_future, 10, "self_prev_round_cosine"
    )
    assert calibration_before == calibration_after


def test_z_t_rel_never_reads_attack_start_round_or_malicious_client_id() -> None:
    """z_t_rel must be identical regardless of ground-truth attack metadata on the rows."""
    per_client_offset = {"client_000": 0.00, "client_001": 0.03, "client_002": -0.05}
    rows_a = [
        _row_v2(r, cid, t=0.30 - 0.01 * r + offset)
        for r in range(25)
        for cid, offset in per_client_offset.items()
    ]
    rows_b = [dict(row, attack_start_round=999, malicious_client_id="client_999") for row in rows_a]
    scored_a = compute_all_scores_v2(rows_a, t_ref=20)
    scored_b = compute_all_scores_v2(rows_b, t_ref=20)
    z_a = [(s.round_id, s.client_id, s.z_t_rel) for s in scored_a]
    z_b = [(s.round_id, s.client_id, s.z_t_rel) for s in scored_b]
    assert z_a == z_b


# ----------------------------------------------------------------------
# FedTrace v2 max combination / missing-component handling
# ----------------------------------------------------------------------


def test_v2_combined_score_ignores_missing_t_rel_never_treats_it_as_zero() -> None:
    rows = [
        _row_v2(r, "client_000", m=6.0 + 0.01 * (r % 3), t=None) for r in range(25)
    ] + [_row_v2(r, "client_001", m=5.9 + 0.02 * (r % 2), t=None) for r in range(25)]
    scored = compute_all_scores_v2(rows, t_ref=20)
    row = next(s for s in scored if s.round_id == 20 and s.client_id == "client_000")
    assert row.z_t_rel is None and row.z_m is not None
    combined_both = row.combined_score(("m", "t_rel"))
    combined_m_only = row.combined_score(("m",))
    assert combined_both == combined_m_only


def test_v2_evaluate_detector_not_evaluable_when_t_entirely_missing() -> None:
    """Mirrors EXP_014: requesting the corrected-temporal-only or fedtrace-v2 detector must fail loudly."""
    rows = [_row_v2(r, "client_000", m=6.0, t=None) for r in range(25)]
    for row in rows:
        del row["self_prev_round_cosine"]
    result = evaluate_detector_v2(rows, "D_corrected_temporal_only", ("t_rel",))
    assert result.evaluable is False
    assert "self_prev_round_cosine" in result.not_evaluable_reason


def test_v2_component_fieldnames_cover_magnitude_and_temporal_only() -> None:
    assert set(V2_COMPONENT_FIELDNAMES) == {"m", "t_rel"}
    assert "p" not in V2_COMPONENT_FIELDNAMES  # peer signal must never enter the v2 primary score


# ----------------------------------------------------------------------
# Persistence rule unchanged (same semantics as v1, applied to ScoredRowV2)
# ----------------------------------------------------------------------


def _build_five_client_temporal_run_v2(malicious_t_by_round: dict[int, float]) -> list[dict]:
    """5 clients (mirrors this project's real clients-per-round=5), 4 honest with fixed,
    mutually-distinct temporal values (avoids the degenerate n=2/tied-value MAD cases --
    see test_robust_calibration_none_when_mad_is_zero for why exact ties matter), and
    client_002's caller-specified trajectory (defaults to its own honest-range baseline
    of 0.24 wherever a round is not given explicitly)."""
    honest_fixed = {"client_000": 0.20, "client_001": 0.30, "client_003": 0.35, "client_004": 0.25}
    rows = []
    for r in range(20):
        for cid, value in honest_fixed.items():
            rows.append(_row_v2(r, cid, t=value))
        rows.append(_row_v2(r, "client_002", t=0.24))
    for r in range(20, 30):
        for cid, value in honest_fixed.items():
            rows.append(_row_v2(r, cid, t=value))
        rows.append(_row_v2(r, "client_002", t=malicious_t_by_round.get(r, 0.24)))
    return rows


def test_v2_persistence_requires_three_consecutive_anomalous_rounds() -> None:
    rows = _build_five_client_temporal_run_v2({20: 0.30, 21: -0.50, 22: 0.30, 23: 0.30, 24: 0.30})
    scored = compute_all_scores_v2(rows, t_ref=20)
    per_client = apply_persistence_rule_v2(scored, ("t_rel",))
    client_002_result = per_client["client_002"]
    assert client_002_result.first_anomalous_round == 21
    assert client_002_result.confirmation_round is None  # isolated single-round drop, not persistent


def test_v2_persistence_confirms_on_third_consecutive_anomalous_round() -> None:
    rows = _build_five_client_temporal_run_v2({r: -0.50 for r in range(20, 30)})
    scored = compute_all_scores_v2(rows, t_ref=20)
    per_client = apply_persistence_rule_v2(scored, ("t_rel",))
    client_002_result = per_client["client_002"]
    assert client_002_result.first_anomalous_round == 20
    assert client_002_result.onset_estimate == 20
    assert client_002_result.confirmation_round == 22  # onset + persistence(3) - 1, unchanged from v1


def test_v2_fedtrace_full_detector_attributes_earliest_confirmed_client() -> None:
    rows = _build_five_client_temporal_run_v2({r: -0.50 for r in range(20, 30)})
    result = evaluate_detector_v2(rows, "F_fedtrace_v2", ("m", "t_rel"))
    assert result.predicted_client == "client_002"
    assert result.attribution_correct is True
    assert result.onset_localization_error == 0
    assert result.detection_delay == 2


# ----------------------------------------------------------------------
# v1 behavior unchanged (regression check)
# ----------------------------------------------------------------------


def test_v1_evaluate_detector_output_unaffected_by_v2_additions() -> None:
    """A v1-only detector call must produce results identical to before v2 was added --
    no v2 import or function call should be able to influence v1's own evaluator."""
    rows = _build_two_client_magnitude_run({r: 50.0 for r in range(20, 30)})
    result_v1 = evaluate_detector(rows, "B_magnitude_only", ("m",))
    assert result_v1.predicted_client == "client_002"
    assert result_v1.onset_localization_error == 0
    assert result_v1.detection_delay == 2
    assert result_v1.precision == 1.0 and result_v1.recall == 1.0 and result_v1.f1 == 1.0


# ----------------------------------------------------------------------
# Deterministic evaluation
# ----------------------------------------------------------------------


def test_v2_evaluation_is_deterministic_across_repeated_calls() -> None:
    rows = _build_five_client_temporal_run_v2({r: -0.50 for r in range(20, 30)})
    result_1 = evaluate_detector_v2(rows, "F_fedtrace_v2", ("m", "t_rel"))
    result_2 = evaluate_detector_v2(rows, "F_fedtrace_v2", ("m", "t_rel"))
    assert result_1.predicted_client == result_2.predicted_client
    assert result_1.onset_localization_error == result_2.onset_localization_error
    assert result_1.detection_delay == result_2.detection_delay
    assert result_1.per_client == result_2.per_client


# ============================================================================
# FedTrace v3 (research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md)
# ============================================================================

# Fixture used throughout this section: 4 clients hold fixed t values every
# round (client_001=0.30, client_002=0.24, client_003=0.35, client_004=0.25 --
# client_004 always sits at the exact median given these four, so c(r) stays
# pinned at 0.25 across every round regardless of client_000's own value,
# which is what lets every hand-computed number below stay exact and simple).
# client_000's t value varies by round below, giving a genuinely nonzero
# historical residual variance (unlike v2's fixture, a plain constant value
# per client would make D_k's variance zero and its calibration undefined --
# see test_historical_calibration_zero_variance_returns_none for why that
# matters). t_ref is overridden to 5 (rounds 1-4 as the calibration window)
# purely to keep this hand-verification small -- production code always
# receives the real t_ref=20 default; the parameter is not hard-coded and is
# exercised at both a small and the default value across this section.

_V3_FIXED_HONEST_T = {"client_001": 0.30, "client_003": 0.35, "client_004": 0.25}


def _row_v3(round_id: int, client_id: str, *, t: float | None, m: float | None = 6.0) -> dict:
    return _row(round_id, client_id, m=m, t=t)


def _build_v3_fixture(client_000_t_by_round: dict[int, float], *, t_ref: int = 5) -> list[dict]:
    rows = []
    for r in range(t_ref + 5):
        rows.append(_row_v3(r, "client_000", t=client_000_t_by_round.get(r, 0.20)))
        rows.append(_row_v3(r, "client_002", t=0.24))
        for cid, value in _V3_FIXED_HONEST_T.items():
            rows.append(_row_v3(r, cid, t=value))
    return rows


# Hand-verified calibration for client_000 under this fixture with t_ref=5:
# D_{client_000} = {c(r) - t(client_000,r) : r in [1,4]}
#                = {0.25-0.10, 0.25-0.15, 0.25-0.05, 0.25-0.20} = {0.15, 0.10, 0.20, 0.05}
# mean = 0.125; sample variance = 0.0125/3 = 0.0041667; sample std ~ 0.064550
_V3_HISTORICAL_CENTER_CLIENT_000 = 0.125
_V3_HISTORICAL_SCALE_CLIENT_000 = 0.06455


# ----------------------------------------------------------------------
# compute_common_round_reference: plain median, no scale
# ----------------------------------------------------------------------


def test_common_round_reference_computes_median() -> None:
    rows = [
        _row_v3(5, "client_000", t=0.10),
        _row_v3(5, "client_001", t=0.30),
        _row_v3(5, "client_002", t=0.24),
        _row_v3(5, "client_003", t=0.35),
        _row_v3(5, "client_004", t=0.25),
    ]
    reference = compute_common_round_reference(rows, 5, "self_prev_round_cosine")
    assert reference == 0.25


def test_common_round_reference_none_when_fewer_than_two_clients() -> None:
    rows = [_row_v3(5, "client_000", t=0.20)]
    assert compute_common_round_reference(rows, 5, "self_prev_round_cosine") is None


def test_common_round_reference_no_leave_one_out() -> None:
    """The client whose own row is included must not be excluded from the median computation."""
    rows = [
        _row_v3(5, "client_000", t=0.10),
        _row_v3(5, "client_001", t=0.30),
    ]
    # With only these two rows, the median (mean of the two, n=2) is 0.20 --
    # this would differ if client_000's own row were excluded from its own reference.
    reference = compute_common_round_reference(rows, 5, "self_prev_round_cosine")
    assert reference is not None
    assert abs(reference - 0.20) < 1e-9


def test_common_round_reference_never_uses_future_rounds() -> None:
    rows = [
        _row_v3(5, "client_000", t=0.10),
        _row_v3(5, "client_001", t=0.30),
        _row_v3(6, "client_000", t=999.0),
        _row_v3(6, "client_001", t=-999.0),
    ]
    reference = compute_common_round_reference(rows, 5, "self_prev_round_cosine")
    assert reference is not None
    assert abs(reference - 0.20) < 1e-9  # unaffected by round 6's extreme values


# ----------------------------------------------------------------------
# Residual computation (via compute_all_scores_v3's z_t_res, exercised end-to-end)
# ----------------------------------------------------------------------


def test_residual_and_historical_calibration_hand_verified() -> None:
    """d(client_000, r) = c(r) - t(client_000, r); D_{client_000} matches the module-level comment above."""
    rows = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20})
    scored = compute_all_scores_v3(rows, t_ref=5)
    # round 1: c(1)=0.25 (client_004's fixed value, robust to client_000's varying input),
    # t(client_000,1)=0.10 -> d=0.15; this round is inside the calibration window itself,
    # scored using the same frozen calibration as every other round (matches z_m's own
    # existing convention -- see compute_historical_residual_calibration's docstring).
    row1 = next(s for s in scored if s.round_id == 1 and s.client_id == "client_000")
    expected_z = max(0.0, (0.15 - _V3_HISTORICAL_CENTER_CLIENT_000) / _V3_HISTORICAL_SCALE_CLIENT_000)
    assert row1.z_t_res is not None
    assert abs(row1.z_t_res - expected_z) < 1e-3


# ----------------------------------------------------------------------
# Historical window: excludes future/post-reference rows, sample mean/stdev(n-1)
# ----------------------------------------------------------------------


def test_historical_window_excludes_rounds_at_or_after_t_ref() -> None:
    """Rounds >= t_ref must never enter D_k, however extreme their residual."""
    rows = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20, 5: -100.0, 6: 100.0}, t_ref=5)
    round_references = {
        r: compute_common_round_reference(rows, r, "self_prev_round_cosine")
        for r in sorted({row["round_id"] for row in rows})
    }
    calibration = compute_historical_residual_calibration(
        rows, "client_000", "self_prev_round_cosine", round_references, t_ref=5
    )
    assert calibration is not None
    assert abs(calibration.mean - _V3_HISTORICAL_CENTER_CLIENT_000) < 1e-9
    assert abs(calibration.std - _V3_HISTORICAL_SCALE_CLIENT_000) < 1e-3


def test_historical_window_excludes_round_zero() -> None:
    """t(k,0) is always undefined, so round 0 can never contribute to D_k."""
    rows = _build_v3_fixture({0: -100.0, 1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20}, t_ref=5)
    round_references = {
        r: compute_common_round_reference(rows, r, "self_prev_round_cosine")
        for r in sorted({row["round_id"] for row in rows})
    }
    calibration = compute_historical_residual_calibration(
        rows, "client_000", "self_prev_round_cosine", round_references, t_ref=5
    )
    assert calibration is not None
    assert calibration.n == 4  # rounds 1-4 only, never round 0
    assert abs(calibration.mean - _V3_HISTORICAL_CENTER_CLIENT_000) < 1e-9


def test_historical_calibration_uses_sample_mean() -> None:
    rows = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20}, t_ref=5)
    round_references = {
        r: compute_common_round_reference(rows, r, "self_prev_round_cosine")
        for r in sorted({row["round_id"] for row in rows})
    }
    calibration = compute_historical_residual_calibration(
        rows, "client_000", "self_prev_round_cosine", round_references, t_ref=5
    )
    assert calibration is not None
    assert abs(calibration.mean - 0.125) < 1e-9


def test_historical_calibration_uses_sample_stdev_with_n_minus_1() -> None:
    """Sum of squared deviations 0.0125 / (n-1=3) = 0.0041667 -> std ~ 0.06455, not / n=4 (~0.0559)."""
    rows = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20}, t_ref=5)
    round_references = {
        r: compute_common_round_reference(rows, r, "self_prev_round_cosine")
        for r in sorted({row["round_id"] for row in rows})
    }
    calibration = compute_historical_residual_calibration(
        rows, "client_000", "self_prev_round_cosine", round_references, t_ref=5
    )
    assert calibration is not None
    n_minus_1_std = (0.0125 / 3) ** 0.5
    n_std = (0.0125 / 4) ** 0.5
    assert abs(calibration.std - n_minus_1_std) < 1e-6
    assert abs(calibration.std - n_std) > 1e-3  # decisively not the population (n) variant


def test_historical_calibration_min_observations_rule() -> None:
    """Only 1 pre-t_ref round -> undefined, mirroring MIN_CALIBRATION_OBSERVATIONS=2 everywhere else."""
    rows = _build_v3_fixture({1: 0.10}, t_ref=2)
    round_references = {
        r: compute_common_round_reference(rows, r, "self_prev_round_cosine")
        for r in sorted({row["round_id"] for row in rows})
    }
    calibration = compute_historical_residual_calibration(
        rows, "client_000", "self_prev_round_cosine", round_references, t_ref=2
    )
    assert calibration is None


def test_historical_calibration_zero_variance_returns_none() -> None:
    """If every pre-t_ref residual is identical, the sample standard deviation is exactly 0."""
    rows = _build_v3_fixture({1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20}, t_ref=5)
    round_references = {
        r: compute_common_round_reference(rows, r, "self_prev_round_cosine")
        for r in sorted({row["round_id"] for row in rows})
    }
    calibration = compute_historical_residual_calibration(
        rows, "client_000", "self_prev_round_cosine", round_references, t_ref=5
    )
    assert calibration is None


def test_historical_calibration_never_substitutes_epsilon_or_mad_for_zero_variance() -> None:
    """A degenerate calibration must propagate as None all the way to z_t_res -- no epsilon, no MAD fallback."""
    rows = _build_v3_fixture({1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20, 5: -50.0}, t_ref=5)
    scored = compute_all_scores_v3(rows, t_ref=5)
    row5 = next(s for s in scored if s.round_id == 5 and s.client_id == "client_000")
    assert row5.z_t_res is None


# ----------------------------------------------------------------------
# One-sided z_t_res, undefined propagation, no future leakage
# ----------------------------------------------------------------------


def test_z_t_res_is_one_sided() -> None:
    """A client running consistently *more* self-consistent than its own history scores 0, not negative."""
    rows = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20, 6: 0.24}, t_ref=5)
    # round 6: client_000's t rises to 0.24, close to c(6)=0.25 -> d=0.01, far *below*
    # its own historical center (0.125) -> the "wrong" (safe) direction, must clip to 0.
    scored = compute_all_scores_v3(rows, t_ref=5)
    row6 = next(s for s in scored if s.round_id == 6 and s.client_id == "client_000")
    assert row6.z_t_res == 0.0


def test_z_t_res_undefined_when_t_entirely_missing_for_run() -> None:
    """Mirrors EXP_014: no self_prev_round_cosine column at all -> z_t_res undefined for every row,
    while an independently available, non-degenerate magnitude signal remains usable.

    _build_v3_fixture's rows all share _row_v3's default update_l2_norm=6.0 (constant across
    every round), which is fine for the temporal-only tests elsewhere in this section but gives
    z_m a zero-variance (degenerate, correctly None) calibration by construction -- unrelated to
    self_prev_round_cosine. This test overrides client_000's update_l2_norm with real per-round
    variation so it can actually exercise "z_m stays defined when t is entirely removed".
    """
    rows = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20}, t_ref=5)
    for row in rows:
        if row["client_id"] == "client_000":
            row["update_l2_norm"] = 6.0 + 0.01 * (row["round_id"] % 3)
        del row["self_prev_round_cosine"]
    scored = compute_all_scores_v3(rows, t_ref=5)
    assert all(s.z_t_res is None for s in scored)
    assert any(s.z_m is not None for s in scored)  # z_m is unaffected by t's removal


def test_z_t_res_never_uses_future_rounds() -> None:
    rows_a = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20}, t_ref=5)
    rows_b = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20, 6: -999.0, 7: 999.0}, t_ref=5)
    scored_a = {(s.round_id, s.client_id): s.z_t_res for s in compute_all_scores_v3(rows_a, t_ref=5)}
    scored_b = {(s.round_id, s.client_id): s.z_t_res for s in compute_all_scores_v3(rows_b, t_ref=5)}
    for round_id in range(5):
        assert scored_a[(round_id, "client_000")] == scored_b[(round_id, "client_000")]


def test_z_t_res_never_reads_attack_start_round_or_malicious_client_id() -> None:
    rows_a = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20, 5: -0.15}, t_ref=5)
    rows_b = [dict(row, attack_start_round=999, malicious_client_id="client_999") for row in rows_a]
    z_a = [(s.round_id, s.client_id, s.z_t_res) for s in compute_all_scores_v3(rows_a, t_ref=5)]
    z_b = [(s.round_id, s.client_id, s.z_t_res) for s in compute_all_scores_v3(rows_b, t_ref=5)]
    assert z_a == z_b


# ----------------------------------------------------------------------
# FedTrace v3 max combination / missing-component handling
# ----------------------------------------------------------------------


def test_v3_combined_score_uses_defined_component_when_one_is_missing() -> None:
    rows = [
        _row_v3(r, "client_000", m=6.0 + 0.01 * (r % 3), t=None) for r in range(25)
    ] + [_row_v3(r, "client_001", m=5.9 + 0.02 * (r % 2), t=None) for r in range(25)]
    scored = compute_all_scores_v3(rows, t_ref=20)
    row = next(s for s in scored if s.round_id == 20 and s.client_id == "client_000")
    assert row.z_t_res is None and row.z_m is not None
    combined_both = row.combined_score(("m", "t_res"))
    combined_m_only = row.combined_score(("m",))
    assert combined_both == combined_m_only


def test_v3_combined_score_undefined_when_both_components_missing() -> None:
    rows = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20}, t_ref=5)
    for row in rows:
        row["update_l2_norm"] = None  # z_m undefined (no calibration possible)
        del row["self_prev_round_cosine"]  # z_t_res undefined (field entirely absent)
    scored = compute_all_scores_v3(rows, t_ref=5)
    row5 = next(s for s in scored if s.round_id == 5 and s.client_id == "client_000")
    assert row5.z_m is None and row5.z_t_res is None
    assert row5.combined_score(("m", "t_res")) is None


def test_v3_evaluate_detector_not_evaluable_when_t_entirely_missing() -> None:
    rows = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20}, t_ref=5)
    for row in rows:
        del row["self_prev_round_cosine"]
    result = evaluate_detector_v3(rows, "E_temporal_residual_only", ("t_res",), t_ref=5)
    assert result.evaluable is False
    assert "self_prev_round_cosine" in result.not_evaluable_reason


def test_v3_component_fieldnames_cover_magnitude_and_temporal_residual_only() -> None:
    assert set(V3_COMPONENT_FIELDNAMES) == {"m", "t_res"}
    assert "p" not in V3_COMPONENT_FIELDNAMES  # peer signal must never enter the v3 primary score


# ----------------------------------------------------------------------
# Persistence (P=3 unchanged) and attribution/tie-breaking (unchanged)
# ----------------------------------------------------------------------


def test_v3_persistence_requires_three_consecutive_anomalous_rounds() -> None:
    """A single anomalous round (r5) followed by a non-anomalous one (r6) must not confirm."""
    rows = _build_v3_fixture({1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20, 5: -0.15, 6: 0.12}, t_ref=5)
    scored = compute_all_scores_v3(rows, t_ref=5)
    per_client = apply_persistence_rule_v3(scored, ("t_res",))
    result = per_client["client_000"]
    assert result.first_anomalous_round == 5
    assert result.confirmation_round is None


def test_v3_persistence_confirms_on_third_consecutive_anomalous_round() -> None:
    """Hand-verified: z(r5)=4.26, z(r6)=5.03, z(r7)=3.49 -- all > 3, three consecutive."""
    rows = _build_v3_fixture(
        {1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20, 5: -0.15, 6: -0.20, 7: -0.10}, t_ref=5
    )
    scored = compute_all_scores_v3(rows, t_ref=5)
    per_client = apply_persistence_rule_v3(scored, ("t_res",))
    result = per_client["client_000"]
    assert result.first_anomalous_round == 5
    assert result.onset_estimate == 5
    assert result.confirmation_round == 7  # onset + persistence(3) - 1, unchanged from v1/v2


def test_v3_fedtrace_attributes_earliest_confirmed_client() -> None:
    rows = _build_v3_fixture(
        {1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20, 5: -0.15, 6: -0.20, 7: -0.10}, t_ref=5
    )
    result = evaluate_detector_v3(rows, "H_fedtrace_v3", ("m", "t_res"), t_ref=5)
    assert result.predicted_client == "client_000"
    # _row()'s default ground-truth client is "client_002", but this fixture's anomalous
    # client is "client_000" -- attribution is therefore (correctly) reported as incorrect,
    # and onset_localization_error/detection_delay are left undefined rather than computed
    # against the wrong client (matches evaluate_detector_v3's documented behavior).
    assert result.attribution_correct is False
    assert result.onset_localization_error is None
    per_client_result = result.per_client["client_000"]
    assert per_client_result.onset_estimate == 5
    assert per_client_result.confirmation_round == 7


# ----------------------------------------------------------------------
# v1 and v2 unaffected by v3 (regression checks)
# ----------------------------------------------------------------------


def test_v1_evaluate_detector_unaffected_by_v3_additions() -> None:
    rows = _build_two_client_magnitude_run({r: 50.0 for r in range(20, 30)})
    result_v1 = evaluate_detector(rows, "B_magnitude_only", ("m",))
    assert result_v1.predicted_client == "client_002"
    assert result_v1.onset_localization_error == 0
    assert result_v1.detection_delay == 2
    assert result_v1.precision == 1.0 and result_v1.recall == 1.0 and result_v1.f1 == 1.0


def test_v2_evaluate_detector_unaffected_by_v3_additions() -> None:
    rows = _build_five_client_temporal_run_v2({r: -0.50 for r in range(20, 30)})
    result_v2 = evaluate_detector_v2(rows, "F_fedtrace_v2", ("m", "t_rel"))
    assert result_v2.predicted_client == "client_002"
    assert result_v2.onset_localization_error == 0
    assert result_v2.detection_delay == 2


# ----------------------------------------------------------------------
# Deterministic evaluation
# ----------------------------------------------------------------------


def test_v3_evaluation_is_deterministic_across_repeated_calls() -> None:
    rows = _build_v3_fixture(
        {1: 0.10, 2: 0.15, 3: 0.05, 4: 0.20, 5: -0.15, 6: -0.20, 7: -0.10}, t_ref=5
    )
    result_1 = evaluate_detector_v3(rows, "H_fedtrace_v3", ("m", "t_res"), t_ref=5)
    result_2 = evaluate_detector_v3(rows, "H_fedtrace_v3", ("m", "t_res"), t_ref=5)
    assert result_1.predicted_client == result_2.predicted_client
    assert result_1.per_client == result_2.per_client
