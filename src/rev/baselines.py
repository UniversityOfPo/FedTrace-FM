"""CPU reference implementations of standard sequential change detectors.

All three operate on one client's round-indexed scalar series (``{round: value}``,
``None`` = no observation) and are causal: the decision at round ``r`` uses
observations at rounds ``<= r`` only. Calibration uses rounds ``< t_ref``; no
alarm is raised before ``t_ref``. A round with no observation leaves the state
unchanged and cannot raise an alarm (persistence streaks are broken by it).

    zscore_persistence  z_r = |x_r - mu| / sd (or one-sided), mu/sd = sample mean /
                        sample std (n-1) of calibration rounds; alarm at the
                        ``persistence``-th consecutive round with z_r > threshold;
                        onset estimate = streak start.
    cusum               one-sided upper CUSUM (Page, 1954) on standardised
                        values: S_r = max(0, S_{r-1} + z_r - k), S = 0 before
                        t_ref; alarm at the first r >= t_ref with S_r > h; onset
                        estimate = first round of the current positive excursion.
    page_hinkley        Page-Hinkley test for an increase in the mean (Hinkley,
                        1971): over all observed rounds in order,
                        xbar_T = running mean, m_T = sum_{t<=T} (x_t - xbar_t - delta),
                        M_T = min_{t<=T} m_t; alarm at the first T >= t_ref with
                        m_T - M_T > lam; onset estimate = first observed round
                        after the round attaining M_T (the latest one on ties;
                        the first observed round if m never fell to <= 0).

These are reference implementations from the textbook definitions. They are not
FLDetector, FedREDefense or FLForensics, which are not implemented.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineEvents:
    alarm_round: int | None
    estimated_onset: int | None
    statistic: dict[int, float]


def _calibration(series: Mapping[int, float | None], t_ref: int) -> tuple[float, float] | None:
    values = [float(v) for r, v in series.items() if r < t_ref and v is not None]
    if len(values) < 2:
        return None
    sd = statistics.stdev(values)
    if sd == 0 or not math.isfinite(sd):
        return None
    return statistics.mean(values), sd


def zscore_persistence(
    series: Mapping[int, float | None], *, t_ref: int, threshold: float, persistence: int, two_sided: bool = True
) -> BaselineEvents:
    cal = _calibration(series, t_ref)
    if cal is None:
        return BaselineEvents(None, None, {})
    mu, sd = cal
    stat: dict[int, float] = {}
    streak, start = 0, None
    prev = None
    for r in sorted(series):
        if r < t_ref:
            continue
        x = series[r]
        if x is None:
            streak, start, prev = 0, None, r
            continue
        z = (float(x) - mu) / sd
        z = abs(z) if two_sided else z
        stat[r] = z
        if z > threshold:
            if streak > 0 and prev == r - 1:
                streak += 1
            else:
                streak, start = 1, r
            if streak >= persistence:
                return BaselineEvents(r, start, stat)
        else:
            streak, start = 0, None
        prev = r
    return BaselineEvents(None, None, stat)


def cusum(
    series: Mapping[int, float | None], *, t_ref: int, k: float, h: float, standardize: bool = True
) -> BaselineEvents:
    if standardize:
        cal = _calibration(series, t_ref)
        if cal is None:
            return BaselineEvents(None, None, {})
        mu, sd = cal
    else:
        mu, sd = 0.0, 1.0
    s = 0.0
    excursion_start: int | None = None
    stat: dict[int, float] = {}
    for r in sorted(series):
        if r < t_ref or series[r] is None:
            continue
        z = (float(series[r]) - mu) / sd
        new_s = max(0.0, s + z - k)
        if new_s > 0 and s == 0:
            excursion_start = r
        if new_s == 0:
            excursion_start = None
        s = new_s
        stat[r] = s
        if s > h:
            return BaselineEvents(r, excursion_start, stat)
    return BaselineEvents(None, None, stat)


def page_hinkley(
    series: Mapping[int, float | None], *, t_ref: int, delta: float, lam: float
) -> BaselineEvents:
    total, n = 0.0, 0
    m, m_min, argmin = 0.0, 0.0, None
    stat: dict[int, float] = {}
    for r in sorted(series):
        x = series[r]
        if x is None:
            continue
        n += 1
        total += float(x)
        xbar = total / n
        m += float(x) - xbar - delta
        if m <= m_min:
            m_min, argmin = m, r
        stat[r] = m - m_min
        if r >= t_ref and m - m_min > lam:
            onset = None
            later = [q for q in sorted(series) if (argmin is None or q > argmin) and series[q] is not None]
            if later:
                onset = later[0]
            return BaselineEvents(r, onset, stat)
    return BaselineEvents(None, None, stat)


def series_from_trace(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, dict[int, float | None]]:
    """Client id -> {round: value of ``field``} from trace rows (``None`` kept as ``None``)."""
    out: dict[str, dict[int, float | None]] = {}
    for row in rows:
        value = row.get(field)
        out.setdefault(str(row["client_id"]), {})[int(row["round_id"])] = None if value is None else float(value)
    return out


def smallest_threshold_without_client_alarms(
    clean_runs: Sequence[Mapping[str, Mapping[int, float | None]]],
    candidates: Sequence[float],
    detector,
) -> float | None:
    """Smallest candidate threshold with zero client-level alarms over all clean development runs.

    ``detector(series, threshold) -> BaselineEvents``. Returns ``None`` if no candidate
    achieves zero alarms. Must only be called on clean *development* runs.
    """
    for threshold in sorted(candidates):
        if all(detector(series, threshold).alarm_round is None for run in clean_runs for series in run.values()):
            return threshold
    return None
