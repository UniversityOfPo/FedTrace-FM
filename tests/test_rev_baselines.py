"""Reference baselines against independently hand-computed toy examples."""

from __future__ import annotations

import math

import pytest

from src.rev.baselines import (
    cusum,
    page_hinkley,
    series_from_trace,
    smallest_threshold_without_client_alarms,
    zscore_persistence,
)


def _series(values, start=0):
    return {start + i: v for i, v in enumerate(values)}


# Calibration rounds 0..3 = [1, 2, 3, 2]: mean 2, sample sd sqrt(2/3) = 0.81650; value 5 -> z = 3/0.8165 = 3.674.
CAL = [1.0, 2.0, 3.0, 2.0]


def test_zscore_persistence_hand_computed() -> None:
    events = zscore_persistence(_series(CAL + [2.0, 5.0, 5.0, 5.0, 2.0]), t_ref=4, threshold=3.0, persistence=3)
    assert events.alarm_round == 7 and events.estimated_onset == 5
    assert events.statistic[4] == 0.0
    assert events.statistic[5] == pytest.approx(3.0 / math.sqrt(2.0 / 3.0))
    assert 0 not in events.statistic  # calibration rounds are never scored


def test_zscore_two_sided_vs_one_sided_and_streak_breaks() -> None:
    low = _series(CAL + [-1.0, -1.0, -1.0])
    assert zscore_persistence(low, t_ref=4, threshold=3.0, persistence=3).alarm_round == 6
    assert zscore_persistence(low, t_ref=4, threshold=3.0, persistence=3, two_sided=False).alarm_round is None
    broken = _series(CAL + [5.0, 5.0, None, 5.0, 5.0])
    assert zscore_persistence(broken, t_ref=4, threshold=3.0, persistence=3).alarm_round is None
    gap = {**_series(CAL), 4: 5.0, 5: 5.0, 7: 5.0}  # round 6 absent: not contiguous
    assert zscore_persistence(gap, t_ref=4, threshold=3.0, persistence=3).alarm_round is None


def test_zscore_without_calibration_never_alarms() -> None:
    assert zscore_persistence(_series([1.0, 9.0, 9.0, 9.0], start=3), t_ref=4, threshold=3.0, persistence=1).alarm_round is None


def test_cusum_hand_computed_raw() -> None:
    # k = 0.5, h = 2: S = 0, 0.5, 1.0, 1.5, 1.0, 2.5 -> alarm at round 5; excursion started at round 1.
    events = cusum(_series([0.0, 1.0, 1.0, 1.0, 0.0, 2.0]), t_ref=0, k=0.5, h=2.0, standardize=False)
    assert [events.statistic[r] for r in range(6)] == [0.0, 0.5, 1.0, 1.5, 1.0, 2.5]
    assert events.alarm_round == 5 and events.estimated_onset == 1


def test_cusum_resets_at_zero_and_restarts_the_excursion() -> None:
    # S = 0.5, 0, 0.5, 2.0, 3.5 -> excursion restarted at round 2; alarm (h=3) at round 4.
    events = cusum(_series([1.0, 0.0, 1.0, 2.0, 2.0]), t_ref=0, k=0.5, h=3.0, standardize=False)
    assert events.statistic[1] == 0.0 and events.alarm_round == 4 and events.estimated_onset == 2


def test_cusum_standardised_uses_calibration_and_starts_at_t_ref() -> None:
    # z at round 4 = (5-2)/0.8165 = 3.674; k = 0.5 -> S4 = 3.174 > h = 3.
    events = cusum(_series(CAL + [5.0]), t_ref=4, k=0.5, h=3.0)
    assert set(events.statistic) == {4}
    assert events.statistic[4] == pytest.approx(3.0 / math.sqrt(2.0 / 3.0) - 0.5)
    assert events.alarm_round == 4 and events.estimated_onset == 4


def test_page_hinkley_hand_computed() -> None:
    # x = 0,0,0,3,3; delta = 0: m = 0,0,0,2.25,...; minimum 0 last attained at round 2 -> PH_3 = 2.25 > 1.5.
    events = page_hinkley(_series([0.0, 0.0, 0.0, 3.0, 3.0]), t_ref=0, delta=0.0, lam=1.5)
    assert events.statistic[3] == pytest.approx(2.25)
    assert events.alarm_round == 3 and events.estimated_onset == 3


def test_page_hinkley_with_drift_allowance_and_no_alarm_before_t_ref() -> None:
    # x = 1,1,4,4,4; delta = 0.5: xbar = 1,1,2,2.5,2.8; increments -0.5,-0.5,1.5,1.0,0.7;
    # m = -0.5,-1.0,0.5,1.5,2.2; min -1.0 at round 1; PH = 0,0,1.5,2.5,3.2.
    series = _series([1.0, 1.0, 4.0, 4.0, 4.0])
    early = page_hinkley(series, t_ref=0, delta=0.5, lam=2.0)
    assert [round(early.statistic[r], 10) for r in range(4)] == [0.0, 0.0, 1.5, 2.5]  # stops at the alarm
    assert early.alarm_round == 3 and early.estimated_onset == 2
    late = page_hinkley(series, t_ref=4, delta=0.5, lam=2.0)
    assert late.alarm_round == 4 and round(late.statistic[4], 10) == 3.2


def test_series_from_trace_and_threshold_selection_on_clean_runs() -> None:
    rows = [{"round_id": r, "client_id": c, "update_l2_norm": v}
            for r, (a, b) in enumerate([(1, 2), (2, 2), (3, 2), (2, 9), (2, 9)]) for c, v in (("a", a), ("b", b))]
    rows.append({"round_id": 5, "client_id": "a", "update_l2_norm": None})
    series = series_from_trace(rows, "update_l2_norm")
    assert series["a"][5] is None and series["b"][3] == 9.0

    clean_runs = [{"a": _series([0.0, 1.0, 2.0])}, {"a": _series([0.0, 3.0])}]

    def detector(s, h):
        return cusum(s, t_ref=0, k=0.0, h=h, standardize=False)

    # cumulative sums peak at 3.0 in both runs -> smallest candidate with no alarm is 3.0
    assert smallest_threshold_without_client_alarms(clean_runs, [1.0, 2.0, 3.0, 4.0], detector) == 3.0
    assert smallest_threshold_without_client_alarms(clean_runs, [1.0, 2.0], detector) is None
