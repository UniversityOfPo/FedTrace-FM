"""Causal (online) evaluation around the frozen FedTrace v3 detector.

The v3 functions in ``src.analysis.fedtrace_detector`` are imported and called
unchanged; no score formula is altered. What this module adds is *when* a score
may be used and how the forensic events are defined.

Look-ahead in offline v3 (documented, demonstrated by
:func:`prefix_invariance_violations`):
    * calibration (``z_m`` over rounds ``< t_ref``; the ``z_t_res`` residual scale
      over rounds ``[1, t_ref)``) is computed once from the full trace, and every
      round, *including calibration rounds*, is then scored. A score at a round
      ``r < t_ref`` therefore uses rounds after ``r``;
    * the attribution tie-break (``_select_predicted_client``) uses
      ``mean_score_during_run``, which keeps accumulating scores *after* the
      confirmation round until the streak ends.

Online semantics used here:
    calibration period     rounds ``0 .. t_ref-1`` (no alarm can be raised);
    earliest scoring round ``t_ref`` -- from here a round's score depends only on
                           rounds ``<= r`` (verified by prefix invariance);
    alarm round ``a``      the round of the ``persistence``-th consecutive
                           exceedance among scored rounds ``>= t_ref`` (the v3
                           persistence rule applied to the causal score sequence);
    actionable round       ``a + 1`` for post-aggregation monitoring (the NDSS
                           pipeline computes trace features after aggregating
                           round ``a``); ``a`` if monitoring is pre-aggregation;
    estimated onset        start of the confirming streak, ``a - persistence + 1``
                           (computable at alarm time);
    detection delay        ``a - true_onset`` (computed in :mod:`src.rev.metrics`;
                           a miss is censored, never dropped);
    retrospective onset    v3's offline onset estimate over the full trace,
                           reported separately and labelled retrospective.
    attribution            earliest alarm; ties broken by the mean score of the
                           confirming streak up to the alarm round (causal), then
                           client id.

Limitation: online and offline v3 differ when offline v3 confirms a streak that
starts before ``t_ref`` (the online wrapper cannot raise such an alarm). This is
reported by :func:`online_vs_offline`, never hidden.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.analysis.fedtrace_detector import (
    DEFAULT_PERSISTENCE,
    DEFAULT_T_REF,
    DEFAULT_THRESHOLD,
    ScoredRowV3,
    apply_persistence_rule_v3,
    compute_all_scores_v3,
    evaluate_detector_v3,
)

MONITORING_POINTS = ("post_aggregation", "pre_aggregation")


@dataclass(frozen=True)
class OnlineConfig:
    t_ref: int = DEFAULT_T_REF
    threshold: float = DEFAULT_THRESHOLD
    persistence: int = DEFAULT_PERSISTENCE
    components: tuple[str, ...] = ("m", "t_res")
    monitoring_point: str = "post_aggregation"

    def __post_init__(self) -> None:
        if self.t_ref < 2:
            raise ValueError("t_ref must be >= 2 (v3 needs at least two calibration observations)")
        if self.persistence < 1:
            raise ValueError("persistence must be >= 1")
        if self.monitoring_point not in MONITORING_POINTS:
            raise ValueError(f"monitoring_point must be one of {MONITORING_POINTS}")

    @property
    def earliest_scoring_round(self) -> int:
        return self.t_ref


@dataclass(frozen=True)
class ClientEvents:
    client_id: str
    alarm_round: int | None
    estimated_onset: int | None
    actionable_round: int | None
    streak_mean_score_at_alarm: float | None
    first_exceedance_round: int | None


@dataclass(frozen=True)
class OnlineDetection:
    config: OnlineConfig
    last_round: int
    per_client: dict[str, ClientEvents]
    attributed_client: str | None
    attributed_alarm_round: int | None
    retrospective_onset: dict[str, int | None] = field(default_factory=dict)
    retrospective_attributed_client: str | None = None


def _truncate(rows: Sequence[Mapping[str, Any]], last_round: int) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["round_id"] <= last_round]


def causal_scores_v3(rows: Sequence[Mapping[str, Any]], config: OnlineConfig) -> list[ScoredRowV3]:
    """Scores for rounds ``>= t_ref``, each computed from the trace truncated at that round."""
    rounds = sorted({row["round_id"] for row in rows})
    scored: list[ScoredRowV3] = []
    for r in rounds:
        if r < config.earliest_scoring_round:
            continue
        scored.extend(s for s in compute_all_scores_v3(_truncate(rows, r), t_ref=config.t_ref) if s.round_id == r)
    return scored


def prefix_invariance_violations(
    rows: Sequence[Mapping[str, Any]], config: OnlineConfig, rounds: Sequence[int] | None = None
) -> list[tuple[int, str, float | None, float | None]]:
    """(round, client, offline score, prefix-truncated score) wherever they differ.

    Offline = v3 scored on the full trace; prefix = v3 scored on rows ``<= r``.
    Any entry is look-ahead leakage at that round. ``rounds`` defaults to all rounds.
    """
    offline = {(s.round_id, s.client_id): s for s in compute_all_scores_v3(rows, t_ref=config.t_ref)}
    all_rounds = sorted({row["round_id"] for row in rows}) if rounds is None else list(rounds)
    violations = []
    for r in all_rounds:
        for s in compute_all_scores_v3(_truncate(rows, r), t_ref=config.t_ref):
            if s.round_id != r:
                continue
            full = offline[(r, s.client_id)]
            a = full.combined_score(config.components)
            b = s.combined_score(config.components)
            if a != b:
                violations.append((r, s.client_id, a[0] if a else None, b[0] if b else None))
    return violations


def _streak_mean(scores: Mapping[int, float | None], start: int, end: int) -> float:
    return statistics.mean(float(scores[r]) for r in range(start, end + 1))


def online_detect_v3(rows: Sequence[Mapping[str, Any]], config: OnlineConfig | None = None) -> OnlineDetection:
    """Online FedTrace v3 events for every client (see module docstring)."""
    config = config or OnlineConfig()
    scored = causal_scores_v3(rows, config)
    persistence = apply_persistence_rule_v3(scored, config.components, config.threshold, config.persistence)
    by_client: dict[str, dict[int, float | None]] = {}
    for s in scored:
        combined = s.combined_score(config.components)
        by_client.setdefault(s.client_id, {})[s.round_id] = combined[0] if combined else None

    offset = 1 if config.monitoring_point == "post_aggregation" else 0
    per_client: dict[str, ClientEvents] = {}
    for cid in sorted({row["client_id"] for row in rows}):
        result = persistence.get(cid)
        alarm = result.confirmation_round if result else None
        onset = result.onset_estimate if result else None
        per_client[cid] = ClientEvents(
            client_id=cid,
            alarm_round=alarm,
            estimated_onset=onset,
            actionable_round=alarm + offset if alarm is not None else None,
            streak_mean_score_at_alarm=_streak_mean(by_client[cid], onset, alarm) if alarm is not None else None,
            first_exceedance_round=result.first_anomalous_round if result else None,
        )
    alarmed = [e for e in per_client.values() if e.alarm_round is not None]
    alarmed.sort(key=lambda e: (e.alarm_round, -(e.streak_mean_score_at_alarm or 0.0), e.client_id))

    retrospective = evaluate_detector_v3(rows, "H_fedtrace_v3_retrospective", config.components,
                                         t_ref=config.t_ref, threshold=config.threshold,
                                         persistence=config.persistence)
    return OnlineDetection(
        config=config,
        last_round=max(row["round_id"] for row in rows),
        per_client=per_client,
        attributed_client=alarmed[0].client_id if alarmed else None,
        attributed_alarm_round=alarmed[0].alarm_round if alarmed else None,
        retrospective_onset={cid: r.onset_estimate for cid, r in retrospective.per_client.items()},
        retrospective_attributed_client=retrospective.predicted_client,
    )


def online_vs_offline(rows: Sequence[Mapping[str, Any]], config: OnlineConfig | None = None) -> list[str]:
    """Clients whose online confirmation differs from offline v3 (e.g. a pre-``t_ref`` streak)."""
    config = config or OnlineConfig()
    online = online_detect_v3(rows, config)
    offline = apply_persistence_rule_v3(compute_all_scores_v3(rows, t_ref=config.t_ref), config.components,
                                        config.threshold, config.persistence)
    differences = []
    for cid, events in online.per_client.items():
        off = offline.get(cid)
        off_alarm = off.confirmation_round if off else None
        if off_alarm != events.alarm_round:
            differences.append(f"{cid}: offline confirmation {off_alarm}, online alarm {events.alarm_round}")
    return differences
