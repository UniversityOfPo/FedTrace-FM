"""Offline, retrospective FedTrace multi-signal detector (Stage A / v1, plus v2).

Implements -- exactly, with no redesign -- the frozen v1 method specified
in ``research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md``: three signal
families (magnitude ``m``, cross-sectional peer-cosine ``p``, temporal
self-cosine ``t``), per-signal honest-reference calibration, an
absolute-value max-combiner score, and a fixed-persistence
attribution/onset rule. It operates purely on already-collected
``client_round_trace.csv`` rows (read via :func:`load_trace_rows`); it
trains nothing, runs no federated round, and never reads
``attack_start_round`` for any detection decision -- only for the post-hoc
evaluation metrics in :func:`evaluate_detector`, which is explicitly *not*
part of the detector's own decision process.

The v1 code above the "FedTrace v2" section marker below is **unmodified**
from Stage A -- every v1 function, dataclass, and constant keeps its exact
original behavior, so v1 remains independently reproducible for
before/after comparison. FedTrace v2 (``research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md``)
is implemented entirely as additional code below that marker: a corrected,
robust, per-round cross-sectional temporal calibration
(:func:`compute_cross_sectional_robust_calibration`) replacing v1's ``z_t``
with ``z_t_rel`` for the primary score only, magnitude (``z_m``) carried
over unchanged, and the peer signal (``z_p``) deliberately excluded from
the v2 primary combiner (it remains computable via v1's own functions for
ablation/monitoring purposes, unchanged).

FedTrace v3 (``research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md``)
is likewise implemented entirely as additional code, below its own clearly
marked section, leaving v1 and v2 above it untouched. v3 replaces v2's
same-round cross-sectional *scale* (unstable at ``n=5`` clients, the exact
mechanism found responsible for v2's `client_000` false positive in
``analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md``) with a per-client
*historical* scale estimated from each client's own residuals
(:func:`compute_historical_residual_calibration`), while keeping v2's
same-round median reference (:func:`compute_common_round_reference`) to
remove the shared temporal drift that originally motivated v2. ``z_m`` is
carried over unchanged for a third time; the peer signal remains excluded
from the primary score.

Deliberately placed under ``src/analysis/`` rather than the (currently
empty) ``src/attribution/`` placeholder referenced by
``src/attacks/traceability.py``'s module docstring: this module is a
retrospective, offline *evaluation* tool for the Stage A/A-v2/A-v3 analyses
in ``analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md``,
``analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md``, and
``analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md``, not the eventual
production attribution implementation that placeholder is reserved for
(which, per the method plan's three-stage validation sequence, is not yet
justified before Stage A/B results exist).

This module never modifies ``src/attacks/traceability.py`` or any
experiment result file -- it only reads already-saved
``client_round_trace.csv`` rows.
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_T_REF = 20
DEFAULT_THRESHOLD = 3.0
DEFAULT_PERSISTENCE = 3
MIN_CALIBRATION_OBSERVATIONS = 2

COMPONENT_FIELDNAMES: Mapping[str, str] = {
    "m": "update_l2_norm",
    "p": "peer_mean_cosine_similarity",
    "t": "self_prev_round_cosine",
}

# name -> (components used, human-readable description); mirrors Stage A's
# required detector set A-G (research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md).
# "A" (the existing per-client peer-cosine baseline) is intentionally absent
# here -- it is not a z-score/persistence detector at all, and is evaluated
# separately by evaluate_naive_baseline() directly from the already-computed
# naive_flag/first_flagged_round columns.
DETECTOR_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "B_magnitude_only": ("m",),
    "C_temporal_only": ("t",),
    "D_magnitude_peer": ("m", "p"),
    "E_magnitude_temporal": ("m", "t"),
    "F_peer_temporal": ("p", "t"),
    "G_full_fedtrace": ("m", "p", "t"),
}

_BOOL_TRUE_STRINGS = {"true", "1", "yes"}


def _parse_optional_float(raw: str | None) -> float | None:
    """Parse a CSV cell into a float, or None for empty/missing (never fabricated as 0.0)."""
    if raw is None or raw == "":
        return None
    return float(raw)


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    return int(raw)


def _parse_bool(raw: str | None) -> bool:
    return (raw or "").strip().lower() in _BOOL_TRUE_STRINGS


def load_trace_rows(path: Path) -> list[dict[str, Any]]:
    """Load and type-parse a ``client_round_trace.csv`` file (14- or 18-column).

    Older trace files (e.g. ``EXP_014``'s) lack the four
    ``norm_preserving_directional``-era columns entirely; those keys are
    simply absent from the resulting row dicts rather than being
    back-filled with a fabricated value -- callers must use
    ``row.get(...)``, never ``row[...]``, for those four fields.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for raw_row in reader:
            row: dict[str, Any] = {
                "round_id": int(raw_row["round_id"]),
                "client_id": raw_row["client_id"],
                "selected": _parse_bool(raw_row["selected"]),
                "attack_configured": _parse_bool(raw_row["attack_configured"]),
                "attack_active": _parse_bool(raw_row["attack_active"]),
                "attack_type": raw_row["attack_type"],
                "attack_strength_rho": float(raw_row["attack_strength_rho"]),
                "malicious_client_id": raw_row["malicious_client_id"] or None,
                "attack_start_round": _parse_optional_int(raw_row["attack_start_round"]),
                "update_l2_norm": _parse_optional_float(raw_row.get("update_l2_norm")),
                "peer_mean_cosine_similarity": _parse_optional_float(
                    raw_row.get("peer_mean_cosine_similarity")
                ),
                "naive_threshold": _parse_optional_float(raw_row.get("naive_threshold")),
                "naive_flag": _parse_bool(raw_row.get("naive_flag")),
                "first_flagged_round": _parse_optional_int(raw_row.get("first_flagged_round")),
            }
            if "self_prev_round_cosine" in raw_row:
                row["self_prev_round_cosine"] = _parse_optional_float(
                    raw_row.get("self_prev_round_cosine")
                )
            rows.append(row)
    return rows


def field_available(rows: Sequence[Mapping[str, Any]], field_name: str) -> bool:
    """Whether ``field_name`` is a genuinely populated signal anywhere in ``rows``.

    False both when the column is entirely absent from the row dicts (an
    older-schema trace file) and when the column is present but every
    value is ``None`` -- either way, a detector requiring this signal is
    NOT EVALUABLE on this run, never silently treated as an all-zero
    signal.
    """
    return any(row.get(field_name) is not None for row in rows)


@dataclass(frozen=True)
class ClientCalibration:
    """A single client-or-round honest-reference calibration statistic."""

    mean: float
    std: float
    n: int


def compute_longitudinal_calibration(
    rows: Sequence[Mapping[str, Any]],
    client_id: str,
    field_name: str,
    t_ref: int = DEFAULT_T_REF,
) -> ClientCalibration | None:
    """Per-client longitudinal calibration for signals ``m`` and ``t``.

    Sample mean/standard deviation (``statistics.stdev``, matching
    ``apply_naive_detector``'s existing convention in
    ``src/attacks/traceability.py``) of ``rows[field_name]`` over this
    client's own rows with ``round_id < t_ref`` only. Returns ``None``
    (signal excluded, never treated as zero) if fewer than
    ``MIN_CALIBRATION_OBSERVATIONS`` such values exist, or if the sample
    standard deviation is exactly zero (degenerate; a z-score would be
    undefined).
    """
    values = [
        float(row[field_name])
        for row in rows
        if row["client_id"] == client_id
        and row["round_id"] < t_ref
        and row.get(field_name) is not None
    ]
    if len(values) < MIN_CALIBRATION_OBSERVATIONS:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std == 0:
        return None
    return ClientCalibration(mean=mean, std=std, n=len(values))


def compute_cross_sectional_calibration(
    rows: Sequence[Mapping[str, Any]],
    round_id: int,
    field_name: str,
) -> ClientCalibration | None:
    """Per-round cross-sectional calibration for signal ``p`` (no historical window).

    Sample mean/standard deviation of ``rows[field_name]`` across every
    row with this exact ``round_id`` (i.e. that round's own selected
    clients, including the malicious client itself -- no leave-one-out).
    Returns ``None`` under the same degenerate conditions as
    :func:`compute_longitudinal_calibration`.
    """
    values = [
        float(row[field_name])
        for row in rows
        if row["round_id"] == round_id and row.get(field_name) is not None
    ]
    if len(values) < MIN_CALIBRATION_OBSERVATIONS:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std == 0:
        return None
    return ClientCalibration(mean=mean, std=std, n=len(values))


@dataclass(frozen=True)
class ScoredRow:
    """Per-(client, round) component z-scores; ``None`` means "excluded", never zero."""

    round_id: int
    client_id: str
    z_m: float | None
    z_p: float | None
    z_t: float | None

    def combined_score(self, components: Sequence[str]) -> tuple[float, str] | None:
        """``max`` over whichever requested ``components`` are defined for this row.

        Returns ``(score, dominant_component)``, or ``None`` if none of
        the requested components have a defined value for this row.
        """
        candidates = [
            (z, name)
            for name, z in (("m", self.z_m), ("p", self.z_p), ("t", self.z_t))
            if name in components and z is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pair: pair[0])


def compute_all_scores(
    rows: Sequence[Mapping[str, Any]], t_ref: int = DEFAULT_T_REF
) -> list[ScoredRow]:
    """Compute ``z_m``/``z_p``/``z_t`` for every row, per the frozen calibration rules."""
    client_ids = sorted({row["client_id"] for row in rows})
    round_ids = sorted({row["round_id"] for row in rows})

    m_calibration = {
        client_id: compute_longitudinal_calibration(rows, client_id, "update_l2_norm", t_ref)
        for client_id in client_ids
    }
    t_calibration = {
        client_id: compute_longitudinal_calibration(
            rows, client_id, "self_prev_round_cosine", t_ref
        )
        for client_id in client_ids
    }
    p_calibration = {
        round_id: compute_cross_sectional_calibration(
            rows, round_id, "peer_mean_cosine_similarity"
        )
        for round_id in round_ids
    }

    scored: list[ScoredRow] = []
    for row in rows:
        client_id = row["client_id"]
        round_id = row["round_id"]

        z_m = None
        m_cal = m_calibration.get(client_id)
        if m_cal is not None and row.get("update_l2_norm") is not None:
            z_m = abs(row["update_l2_norm"] - m_cal.mean) / m_cal.std

        z_p = None
        p_cal = p_calibration.get(round_id)
        if p_cal is not None and row.get("peer_mean_cosine_similarity") is not None:
            z_p = abs(row["peer_mean_cosine_similarity"] - p_cal.mean) / p_cal.std

        z_t = None
        t_cal = t_calibration.get(client_id)
        if t_cal is not None and row.get("self_prev_round_cosine") is not None:
            z_t = abs(row["self_prev_round_cosine"] - t_cal.mean) / t_cal.std

        scored.append(ScoredRow(round_id=round_id, client_id=client_id, z_m=z_m, z_p=z_p, z_t=z_t))
    return scored


@dataclass(frozen=True)
class ClientPersistenceResult:
    """The three-way first-anomalous / confirmation / onset-estimate outcome for one client."""

    client_id: str
    first_anomalous_round: int | None
    confirmation_round: int | None
    onset_estimate: int | None
    mean_score_during_run: float | None

    @property
    def detected(self) -> bool:
        return self.confirmation_round is not None


def apply_persistence_rule(
    scored_rows: Sequence[ScoredRow],
    components: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
) -> dict[str, ClientPersistenceResult]:
    """Apply the frozen persistence rule independently for each client.

    Never uses ground-truth ``attack_start_round`` -- purely a function of
    each client's own round-ordered score sequence. ``onset_estimate`` is
    the first round of the confirmed ``persistence``-round consecutive
    run (not the confirmation round itself, which lags by
    ``persistence - 1`` rounds); ``first_anomalous_round`` is the raw
    first single round exceeding ``threshold``, independent of whether it
    was ever confirmed persistent.
    """
    by_client: dict[str, list[ScoredRow]] = {}
    for row in scored_rows:
        by_client.setdefault(row.client_id, []).append(row)

    results: dict[str, ClientPersistenceResult] = {}
    for client_id, client_rows in by_client.items():
        ordered = sorted(client_rows, key=lambda r: r.round_id)

        first_anomalous_round: int | None = None
        confirmation_round: int | None = None
        onset_estimate: int | None = None
        streak_len = 0
        streak_start: int | None = None
        prev_round_id: int | None = None
        prev_anomalous = False
        run_scores: list[float] = []

        for row in ordered:
            combined = row.combined_score(components)
            score = combined[0] if combined is not None else None
            anomalous = score is not None and score > threshold

            if anomalous and first_anomalous_round is None:
                first_anomalous_round = row.round_id

            if anomalous:
                contiguous = prev_anomalous and prev_round_id is not None and row.round_id == prev_round_id + 1
                if contiguous:
                    streak_len += 1
                else:
                    streak_len = 1
                    streak_start = row.round_id
                if score is not None:
                    run_scores.append(score)
                if streak_len >= persistence and confirmation_round is None:
                    confirmation_round = row.round_id
                    onset_estimate = streak_start
            else:
                streak_len = 0
                streak_start = None
                run_scores = []

            prev_anomalous = anomalous
            prev_round_id = row.round_id

        mean_score = statistics.mean(run_scores) if (confirmation_round is not None and run_scores) else None
        results[client_id] = ClientPersistenceResult(
            client_id=client_id,
            first_anomalous_round=first_anomalous_round,
            confirmation_round=confirmation_round,
            onset_estimate=onset_estimate,
            mean_score_during_run=mean_score,
        )
    return results


@dataclass(frozen=True)
class DetectorRunResult:
    """The full evaluation of one detector on one run's trace rows."""

    detector_name: str
    components: tuple[str, ...]
    evaluable: bool
    not_evaluable_reason: str | None
    per_client: dict[str, ClientPersistenceResult] = field(default_factory=dict)
    predicted_client: str | None = None
    true_malicious_client: str | None = None
    attack_start_round: int | None = None
    attribution_correct: bool | None = None
    onset_localization_error: int | None = None
    detection_delay: int | None = None
    true_positive_clients: tuple[str, ...] = ()
    false_positive_clients: tuple[str, ...] = ()
    false_negative: bool = False
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    dominant_component_counts: dict[str, int] = field(default_factory=dict)


def _select_predicted_client(
    per_client: Mapping[str, ClientPersistenceResult],
) -> str | None:
    """Earliest-confirmed client wins; ties broken by higher mean score, then client_id."""
    detected = [result for result in per_client.values() if result.detected]
    if not detected:
        return None
    detected.sort(
        key=lambda r: (
            r.confirmation_round,
            -(r.mean_score_during_run if r.mean_score_during_run is not None else 0.0),
            r.client_id,
        )
    )
    return detected[0].client_id


def evaluate_detector(
    rows: Sequence[Mapping[str, Any]],
    detector_name: str,
    components: Sequence[str],
    t_ref: int = DEFAULT_T_REF,
    threshold: float = DEFAULT_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
) -> DetectorRunResult:
    """Evaluate one detector (a fixed component subset) on one run's already-loaded trace rows.

    Marks the detector NOT EVALUABLE (rather than silently proceeding)
    when none of ``components``' underlying signal fields are populated
    anywhere in ``rows`` -- e.g. requesting ``"t"`` on an ``EXP_014``-era
    trace file that predates ``self_prev_round_cosine`` entirely.
    ``attack_start_round``/``malicious_client_id`` are read here only for
    the post-hoc metrics below -- never inside :func:`apply_persistence_rule`.
    """
    missing = [c for c in components if not field_available(rows, COMPONENT_FIELDNAMES[c])]
    if missing == list(components):
        reason = (
            f"None of the requested component(s) {tuple(components)} have any "
            f"populated values in this run's trace rows (missing field(s): "
            f"{[COMPONENT_FIELDNAMES[c] for c in missing]})."
        )
        return DetectorRunResult(
            detector_name=detector_name,
            components=tuple(components),
            evaluable=False,
            not_evaluable_reason=reason,
        )

    scored_rows = compute_all_scores(rows, t_ref=t_ref)
    per_client = apply_persistence_rule(scored_rows, components, threshold, persistence)
    predicted_client = _select_predicted_client(per_client)

    attack_rows = [row for row in rows if row.get("attack_configured")]
    true_malicious_client = attack_rows[0]["malicious_client_id"] if attack_rows else None
    attack_start_round = attack_rows[0]["attack_start_round"] if attack_rows else None

    attribution_correct: bool | None = None
    onset_error: int | None = None
    delay: int | None = None
    if true_malicious_client is not None:
        attribution_correct = predicted_client == true_malicious_client
        predicted_result = per_client.get(predicted_client) if predicted_client else None
        if predicted_result is not None and attribution_correct:
            if predicted_result.onset_estimate is not None and attack_start_round is not None:
                onset_error = abs(predicted_result.onset_estimate - attack_start_round)
            if predicted_result.confirmation_round is not None and attack_start_round is not None:
                delay = predicted_result.confirmation_round - attack_start_round

    detected_clients = {cid for cid, result in per_client.items() if result.detected}
    true_positive_clients = tuple(sorted(detected_clients & {true_malicious_client})) if true_malicious_client else ()
    false_positive_clients = tuple(sorted(detected_clients - {true_malicious_client}))
    false_negative = bool(true_malicious_client) and true_malicious_client not in detected_clients

    tp = len(true_positive_clients)
    fp = len(false_positive_clients)
    fn = 1 if false_negative else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    dominant_counts: dict[str, int] = {c: 0 for c in components}
    for row in scored_rows:
        combined = row.combined_score(components)
        if combined is not None and combined[0] > threshold:
            dominant_counts[combined[1]] += 1

    return DetectorRunResult(
        detector_name=detector_name,
        components=tuple(components),
        evaluable=True,
        not_evaluable_reason=None,
        per_client=per_client,
        predicted_client=predicted_client,
        true_malicious_client=true_malicious_client,
        attack_start_round=attack_start_round,
        attribution_correct=attribution_correct,
        onset_localization_error=onset_error,
        detection_delay=delay,
        true_positive_clients=true_positive_clients,
        false_positive_clients=false_positive_clients,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        dominant_component_counts=dominant_counts,
    )


def evaluate_naive_baseline(rows: Sequence[Mapping[str, Any]]) -> DetectorRunResult:
    """Wrap the existing ``apply_naive_detector`` output (Detector A) in the same result shape.

    Reads ``naive_flag``/``first_flagged_round`` directly -- this baseline
    is not a z-score/persistence detector and is not recomputed here; the
    values must already have been produced by
    ``src.attacks.traceability.apply_naive_detector`` and saved into
    ``rows``.
    """
    attack_rows = [row for row in rows if row.get("attack_configured")]
    true_malicious_client = attack_rows[0]["malicious_client_id"] if attack_rows else None
    attack_start_round = attack_rows[0]["attack_start_round"] if attack_rows else None

    by_client: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_client.setdefault(row["client_id"], []).append(row)

    per_client: dict[str, ClientPersistenceResult] = {}
    for client_id, client_rows in by_client.items():
        flagged_rows = sorted(
            (r for r in client_rows if r.get("naive_flag")), key=lambda r: r["round_id"]
        )
        first_flagged = flagged_rows[0]["round_id"] if flagged_rows else None
        confirmation_round = first_flagged
        per_client[client_id] = ClientPersistenceResult(
            client_id=client_id,
            first_anomalous_round=first_flagged,
            confirmation_round=confirmation_round,
            onset_estimate=first_flagged,
            mean_score_during_run=None,
        )

    predicted_client = _select_predicted_client(per_client)
    attribution_correct = predicted_client == true_malicious_client if true_malicious_client else None

    onset_error = None
    delay = None
    if attribution_correct and predicted_client is not None and attack_start_round is not None:
        predicted_result = per_client[predicted_client]
        if predicted_result.onset_estimate is not None:
            onset_error = abs(predicted_result.onset_estimate - attack_start_round)
        if predicted_result.confirmation_round is not None:
            delay = predicted_result.confirmation_round - attack_start_round

    detected_clients = {cid for cid, result in per_client.items() if result.detected}
    true_positive_clients = tuple(sorted(detected_clients & {true_malicious_client})) if true_malicious_client else ()
    false_positive_clients = tuple(sorted(detected_clients - {true_malicious_client}))
    false_negative = bool(true_malicious_client) and true_malicious_client not in detected_clients

    tp = len(true_positive_clients)
    fp = len(false_positive_clients)
    fn = 1 if false_negative else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    return DetectorRunResult(
        detector_name="A_existing_peer_cosine_baseline",
        components=(),
        evaluable=True,
        not_evaluable_reason=None,
        per_client=per_client,
        predicted_client=predicted_client,
        true_malicious_client=true_malicious_client,
        attack_start_round=attack_start_round,
        attribution_correct=attribution_correct,
        onset_localization_error=onset_error,
        detection_delay=delay,
        true_positive_clients=true_positive_clients,
        false_positive_clients=false_positive_clients,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        dominant_component_counts={},
    )


def evaluate_all_detectors(
    rows: Sequence[Mapping[str, Any]],
    t_ref: int = DEFAULT_T_REF,
    threshold: float = DEFAULT_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
) -> dict[str, DetectorRunResult]:
    """Evaluate the full Stage A detector set (A-G) on one run's already-loaded trace rows."""
    results: dict[str, DetectorRunResult] = {
        "A_existing_peer_cosine_baseline": evaluate_naive_baseline(rows),
    }
    for name, components in DETECTOR_COMPONENTS.items():
        results[name] = evaluate_detector(
            rows, name, components, t_ref=t_ref, threshold=threshold, persistence=persistence
        )
    return results


# ============================================================================
# FedTrace v2 (research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md)
#
# Everything above this marker is v1, unmodified. Nothing below this marker
# changes, calls into internals of, or otherwise alters v1's behavior --
# v2 reuses v1's public functions (compute_longitudinal_calibration,
# field_available, load_trace_rows, DetectorRunResult, ClientPersistenceResult,
# _select_predicted_client) purely as already-frozen building blocks.
# ============================================================================

DEFAULT_MAD_SCALE_CONSTANT = 1.4826
"""Consistency constant: scales a normal distribution's MAD to its standard deviation."""

V2_COMPONENT_FIELDNAMES: Mapping[str, str] = {
    "m": "update_l2_norm",
    "t_rel": "self_prev_round_cosine",
}

# name -> components used, for the two new-in-v2 detectors this module adds.
# The other four required comparison points (magnitude-only, peer-only,
# old-temporal-only, FedTrace v1) are unchanged v1 detectors, evaluated via
# evaluate_detector()/DETECTOR_COMPONENTS above -- not redefined here.
V2_DETECTOR_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "D_corrected_temporal_only": ("t_rel",),
    "F_fedtrace_v2": ("m", "t_rel"),
}


def compute_cross_sectional_robust_calibration(
    rows: Sequence[Mapping[str, Any]],
    round_id: int,
    field_name: str,
) -> ClientCalibration | None:
    """Per-round, cross-sectional, robust (median/MAD) calibration for FedTrace v2's ``z_t_rel``.

    Per ``research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md`` §3/§14:
    ``center = median`` and ``scale = 1.4826 * MAD`` (``MAD`` = median
    absolute deviation from ``center``) computed across every row with this
    exact ``round_id`` that has a defined ``field_name`` value -- no
    historical window, no leave-one-out (the malicious/outlier client's own
    value is included in computing its own reference, exactly as specified
    and as empirically justified in the v2 plan's §3). Reuses the
    :class:`ClientCalibration` dataclass (``mean``/``std`` fields hold
    ``center``/``scale`` here; the field names are kept for reuse, not
    renamed, to avoid introducing a second near-identical dataclass).

    Returns ``None`` (excluded, never fabricated) if fewer than
    ``MIN_CALIBRATION_OBSERVATIONS`` clients have a defined value this
    round, or if the resulting scale is exactly zero (a degenerate tie
    among all values, or too small an MAD to divide by) -- no epsilon
    floor or substitute constant is used, per the plan's explicit "do not
    substitute zero for missing values" instruction (interpreted here as:
    do not substitute *any* fabricated value at all, whether zero or an
    epsilon).
    """
    values = [
        float(row[field_name])
        for row in rows
        if row["round_id"] == round_id and row.get(field_name) is not None
    ]
    if len(values) < MIN_CALIBRATION_OBSERVATIONS:
        return None
    center = statistics.median(values)
    abs_devs = [abs(v - center) for v in values]
    mad = statistics.median(abs_devs)
    scale = DEFAULT_MAD_SCALE_CONSTANT * mad
    if scale == 0:
        return None
    return ClientCalibration(mean=center, std=scale, n=len(values))


@dataclass(frozen=True)
class ScoredRowV2:
    """Per-(client, round) FedTrace v2 component scores; ``None`` means "excluded", never zero."""

    round_id: int
    client_id: str
    z_m: float | None
    z_t_rel: float | None

    def combined_score(self, components: Sequence[str]) -> tuple[float, str] | None:
        """``max`` over whichever requested ``components`` (subset of ``{"m", "t_rel"}``) are defined."""
        candidates = [
            (z, name)
            for name, z in (("m", self.z_m), ("t_rel", self.z_t_rel))
            if name in components and z is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pair: pair[0])


def compute_all_scores_v2(
    rows: Sequence[Mapping[str, Any]], t_ref: int = DEFAULT_T_REF
) -> list[ScoredRowV2]:
    """Compute ``z_m`` (v1, unchanged) and ``z_t_rel`` (new) for every row.

    ``z_m`` is produced by v1's own :func:`compute_longitudinal_calibration`
    with no changes whatsoever (``research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md``
    §5: "keep the existing z_m definition unchanged. Do not recalibrate or
    retune magnitude behavior."). ``z_t_rel`` uses
    :func:`compute_cross_sectional_robust_calibration` on
    ``self_prev_round_cosine``, one-sided (``max(0, ...)``) per §3/§14 of
    the v2 plan -- an unusually *high* temporal self-consistency this round
    relative to peers is not evidence of attack under this project's threat
    model and scores exactly ``0``, not a large negative number.
    ``attack_start_round`` is never read here.
    """
    client_ids = sorted({row["client_id"] for row in rows})
    round_ids = sorted({row["round_id"] for row in rows})

    m_calibration = {
        client_id: compute_longitudinal_calibration(rows, client_id, "update_l2_norm", t_ref)
        for client_id in client_ids
    }
    t_rel_calibration = {
        round_id: compute_cross_sectional_robust_calibration(
            rows, round_id, "self_prev_round_cosine"
        )
        for round_id in round_ids
    }

    scored: list[ScoredRowV2] = []
    for row in rows:
        client_id = row["client_id"]
        round_id = row["round_id"]

        z_m = None
        m_cal = m_calibration.get(client_id)
        if m_cal is not None and row.get("update_l2_norm") is not None:
            z_m = abs(row["update_l2_norm"] - m_cal.mean) / m_cal.std

        z_t_rel = None
        t_cal = t_rel_calibration.get(round_id)
        if t_cal is not None and row.get("self_prev_round_cosine") is not None:
            # t_cal.mean/t_cal.std hold center/scale here (ClientCalibration reused, see
            # compute_cross_sectional_robust_calibration's docstring).
            z_t_rel = max(0.0, (t_cal.mean - row["self_prev_round_cosine"]) / t_cal.std)

        scored.append(
            ScoredRowV2(round_id=round_id, client_id=client_id, z_m=z_m, z_t_rel=z_t_rel)
        )
    return scored


def apply_persistence_rule_v2(
    scored_rows: Sequence[ScoredRowV2],
    components: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
) -> dict[str, ClientPersistenceResult]:
    """FedTrace v2's persistence rule -- algorithmically identical to v1's :func:`apply_persistence_rule`.

    Duplicated (not shared via a common helper) so that v1's function is
    provably untouched by this addition; the streak/first-anomalous/
    confirmation/onset logic below is byte-for-byte the same algorithm as
    v1's, applied to :class:`ScoredRowV2` instead of :class:`ScoredRow`.
    Never reads ``attack_start_round`` or any other ground-truth field.
    """
    by_client: dict[str, list[ScoredRowV2]] = {}
    for row in scored_rows:
        by_client.setdefault(row.client_id, []).append(row)

    results: dict[str, ClientPersistenceResult] = {}
    for client_id, client_rows in by_client.items():
        ordered = sorted(client_rows, key=lambda r: r.round_id)

        first_anomalous_round: int | None = None
        confirmation_round: int | None = None
        onset_estimate: int | None = None
        streak_len = 0
        streak_start: int | None = None
        prev_round_id: int | None = None
        prev_anomalous = False
        run_scores: list[float] = []

        for row in ordered:
            combined = row.combined_score(components)
            score = combined[0] if combined is not None else None
            anomalous = score is not None and score > threshold

            if anomalous and first_anomalous_round is None:
                first_anomalous_round = row.round_id

            if anomalous:
                contiguous = (
                    prev_anomalous
                    and prev_round_id is not None
                    and row.round_id == prev_round_id + 1
                )
                if contiguous:
                    streak_len += 1
                else:
                    streak_len = 1
                    streak_start = row.round_id
                if score is not None:
                    run_scores.append(score)
                if streak_len >= persistence and confirmation_round is None:
                    confirmation_round = row.round_id
                    onset_estimate = streak_start
            else:
                streak_len = 0
                streak_start = None
                run_scores = []

            prev_anomalous = anomalous
            prev_round_id = row.round_id

        mean_score = (
            statistics.mean(run_scores) if (confirmation_round is not None and run_scores) else None
        )
        results[client_id] = ClientPersistenceResult(
            client_id=client_id,
            first_anomalous_round=first_anomalous_round,
            confirmation_round=confirmation_round,
            onset_estimate=onset_estimate,
            mean_score_during_run=mean_score,
        )
    return results


def evaluate_detector_v2(
    rows: Sequence[Mapping[str, Any]],
    detector_name: str,
    components: Sequence[str],
    t_ref: int = DEFAULT_T_REF,
    threshold: float = DEFAULT_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
) -> DetectorRunResult:
    """Evaluate one FedTrace-v2-family detector (``components`` subset of ``{"m", "t_rel"}``).

    Structurally parallel to v1's :func:`evaluate_detector` (same
    NOT-EVALUABLE rule, same attribution/precision/recall/F1/dominance
    computation), but scores rows via :func:`compute_all_scores_v2` /
    :func:`apply_persistence_rule_v2` instead of v1's scorer. Duplicated
    rather than refactored into a shared helper, for the same
    provably-untouched-v1 reason as :func:`apply_persistence_rule_v2`.
    """
    missing = [c for c in components if not field_available(rows, V2_COMPONENT_FIELDNAMES[c])]
    if missing == list(components):
        reason = (
            f"None of the requested component(s) {tuple(components)} have any "
            f"populated values in this run's trace rows (missing field(s): "
            f"{[V2_COMPONENT_FIELDNAMES[c] for c in missing]})."
        )
        return DetectorRunResult(
            detector_name=detector_name,
            components=tuple(components),
            evaluable=False,
            not_evaluable_reason=reason,
        )

    scored_rows = compute_all_scores_v2(rows, t_ref=t_ref)
    per_client = apply_persistence_rule_v2(scored_rows, components, threshold, persistence)
    predicted_client = _select_predicted_client(per_client)

    attack_rows = [row for row in rows if row.get("attack_configured")]
    true_malicious_client = attack_rows[0]["malicious_client_id"] if attack_rows else None
    attack_start_round = attack_rows[0]["attack_start_round"] if attack_rows else None

    attribution_correct: bool | None = None
    onset_error: int | None = None
    delay: int | None = None
    if true_malicious_client is not None:
        attribution_correct = predicted_client == true_malicious_client
        predicted_result = per_client.get(predicted_client) if predicted_client else None
        if predicted_result is not None and attribution_correct:
            if predicted_result.onset_estimate is not None and attack_start_round is not None:
                onset_error = abs(predicted_result.onset_estimate - attack_start_round)
            if predicted_result.confirmation_round is not None and attack_start_round is not None:
                delay = predicted_result.confirmation_round - attack_start_round

    detected_clients = {cid for cid, result in per_client.items() if result.detected}
    true_positive_clients = (
        tuple(sorted(detected_clients & {true_malicious_client})) if true_malicious_client else ()
    )
    false_positive_clients = tuple(sorted(detected_clients - {true_malicious_client}))
    false_negative = bool(true_malicious_client) and true_malicious_client not in detected_clients

    tp = len(true_positive_clients)
    fp = len(false_positive_clients)
    fn = 1 if false_negative else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    dominant_counts: dict[str, int] = {c: 0 for c in components}
    for row in scored_rows:
        combined = row.combined_score(components)
        if combined is not None and combined[0] > threshold:
            dominant_counts[combined[1]] += 1

    return DetectorRunResult(
        detector_name=detector_name,
        components=tuple(components),
        evaluable=True,
        not_evaluable_reason=None,
        per_client=per_client,
        predicted_client=predicted_client,
        true_malicious_client=true_malicious_client,
        attack_start_round=attack_start_round,
        attribution_correct=attribution_correct,
        onset_localization_error=onset_error,
        detection_delay=delay,
        true_positive_clients=true_positive_clients,
        false_positive_clients=false_positive_clients,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        dominant_component_counts=dominant_counts,
    )


def evaluate_all_detectors_v2(
    rows: Sequence[Mapping[str, Any]],
    t_ref: int = DEFAULT_T_REF,
    threshold: float = DEFAULT_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
) -> dict[str, DetectorRunResult]:
    """Evaluate the full v1-vs-v2 comparison set required by the v2 retrospective task.

    Returns six detectors: the four unchanged v1 ones (magnitude-only,
    peer-only, old-temporal-only, FedTrace v1 -- via v1's own
    :func:`evaluate_detector`, untouched) plus the two new v2 detectors
    (corrected-temporal-only, FedTrace v2 -- via :func:`evaluate_detector_v2`).
    """
    results: dict[str, DetectorRunResult] = {
        "A_magnitude_only": evaluate_detector(
            rows, "A_magnitude_only", ("m",), t_ref=t_ref, threshold=threshold, persistence=persistence
        ),
        "B_peer_only": evaluate_detector(
            rows, "B_peer_only", ("p",), t_ref=t_ref, threshold=threshold, persistence=persistence
        ),
        "C_old_temporal_only": evaluate_detector(
            rows, "C_old_temporal_only", ("t",), t_ref=t_ref, threshold=threshold, persistence=persistence
        ),
        "E_fedtrace_v1": evaluate_detector(
            rows,
            "E_fedtrace_v1",
            ("m", "p", "t"),
            t_ref=t_ref,
            threshold=threshold,
            persistence=persistence,
        ),
    }
    for name, components in V2_DETECTOR_COMPONENTS.items():
        results[name] = evaluate_detector_v2(
            rows, name, components, t_ref=t_ref, threshold=threshold, persistence=persistence
        )
    return results


# ============================================================================
# FedTrace v3 (research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md)
#
# Everything above this marker (v1 and v2) is unmodified by this addition.
# v3 reuses v1's compute_longitudinal_calibration (for z_m, unchanged) and
# v1/v2's DetectorRunResult/ClientPersistenceResult/_select_predicted_client/
# field_available purely as already-frozen building blocks; it does not call
# or alter any v2-specific function (compute_cross_sectional_robust_calibration,
# ScoredRowV2, apply_persistence_rule_v2, evaluate_detector_v2).
# ============================================================================

V3_COMPONENT_FIELDNAMES: Mapping[str, str] = {
    "m": "update_l2_norm",
    "t_res": "self_prev_round_cosine",
}

# name -> components, for the two new-in-v3 detectors this module adds.
# The other comparison points (magnitude-only, peer-only, v1 temporal-only,
# v2 temporal-only, FedTrace v1, FedTrace v2) are unchanged v1/v2 detectors,
# evaluated via evaluate_detector()/evaluate_detector_v2() above.
V3_DETECTOR_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "E_temporal_residual_only": ("t_res",),
    "H_fedtrace_v3": ("m", "t_res"),
}


def compute_common_round_reference(
    rows: Sequence[Mapping[str, Any]],
    round_id: int,
    field_name: str,
) -> float | None:
    """``c(r)``: the plain cross-sectional median of ``field_name`` across every row at ``round_id``.

    Per ``research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md``
    §2/§16 -- median only, no scale/MAD computed here at all (v3's whole
    point is to stop estimating a *scale* statistic from only 5 same-round
    clients; v2's :func:`compute_cross_sectional_robust_calibration` is not
    reused or called by this function). No leave-one-out: every client
    present at ``round_id``, including the one later being scored, is
    included when computing the median (§2 of the v3 plan re-derives and
    reconfirms this choice, rather than merely inheriting it from v2).
    Returns ``None`` (excluded, never fabricated) if fewer than
    ``MIN_CALIBRATION_OBSERVATIONS`` clients have a defined value this
    round. Never reads ``attack_start_round``, ``malicious_client_id``, or
    any future round's rows.
    """
    values = [
        float(row[field_name])
        for row in rows
        if row["round_id"] == round_id and row.get(field_name) is not None
    ]
    if len(values) < MIN_CALIBRATION_OBSERVATIONS:
        return None
    return statistics.median(values)


def compute_historical_residual_calibration(
    rows: Sequence[Mapping[str, Any]],
    client_id: str,
    field_name: str,
    round_references: Mapping[int, float | None],
    t_ref: int = DEFAULT_T_REF,
) -> ClientCalibration | None:
    """Per-client longitudinal calibration of the temporal residual ``d(k,r) = c(r) - t(k,r)``.

    Per ``research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md``
    §3/§4/§16: ``D_k = {d(k,r) : round_id(r) in [1, t_ref)}`` -- round 0 is
    always excluded (``t(k,0)`` is undefined for every client by
    construction, so ``d(k,0)`` is too), and no round ``>= t_ref`` is ever
    included, mirroring :func:`compute_longitudinal_calibration`'s existing
    windowing convention exactly. ``historical_center_k = mean(D_k)``,
    ``historical_scale_k = stdev(D_k)`` -- **sample** standard deviation
    (``statistics.stdev``, ``n-1`` denominator), not a robust (median/MAD)
    alternative -- a deliberate choice explained in the v3 plan's §3 (at
    ``n~19`` observations, the small-sample instability that motivated v2's
    robust cross-sectional scale does not apply, and reusing this project's
    already-established longitudinal-calibration convention, shared with
    :func:`compute_longitudinal_calibration`, avoids introducing a second,
    unevidenced scale-estimation rule).

    ``round_references`` must already hold every round's
    :func:`compute_common_round_reference` result (computed once, by the
    caller, and shared across all 5 clients -- not recomputed per client
    here) -- this function only reads from it, never calls into it, so a
    round's reference is provably computed identically regardless of which
    client's calibration is being built from it.

    Returns ``None`` (excluded, never fabricated, no epsilon/MAD fallback)
    if fewer than ``MIN_CALIBRATION_OBSERVATIONS`` valid residuals exist in
    ``D_k``, or if the resulting sample standard deviation is exactly zero.
    """
    values: list[float] = []
    for row in rows:
        if row["client_id"] != client_id:
            continue
        round_id = row["round_id"]
        if not (1 <= round_id < t_ref):
            continue
        t_value = row.get(field_name)
        if t_value is None:
            continue
        c_r = round_references.get(round_id)
        if c_r is None:
            continue
        values.append(c_r - float(t_value))

    if len(values) < MIN_CALIBRATION_OBSERVATIONS:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    if std == 0:
        return None
    return ClientCalibration(mean=mean, std=std, n=len(values))


@dataclass(frozen=True)
class ScoredRowV3:
    """Per-(client, round) FedTrace v3 component scores; ``None`` means "excluded", never zero."""

    round_id: int
    client_id: str
    z_m: float | None
    z_t_res: float | None

    def combined_score(self, components: Sequence[str]) -> tuple[float, str] | None:
        """``max`` over whichever requested ``components`` (subset of ``{"m", "t_res"}``) are defined."""
        candidates = [
            (z, name)
            for name, z in (("m", self.z_m), ("t_res", self.z_t_res))
            if name in components and z is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pair: pair[0])


def compute_all_scores_v3(
    rows: Sequence[Mapping[str, Any]], t_ref: int = DEFAULT_T_REF
) -> list[ScoredRowV3]:
    """Compute ``z_m`` (v1, unchanged) and ``z_t_res`` (new) for every row.

    ``z_m`` is produced by v1's own :func:`compute_longitudinal_calibration`
    with no changes whatsoever (``research_plans/FEDTRACE_V3_...PLAN.md``
    §5/§6: "reuse existing z_m EXACTLY unchanged"). ``z_t_res`` is computed
    in two passes, exactly per §16 of the v3 plan: first every round's
    common reference ``c(r)`` (:func:`compute_common_round_reference`,
    computed once per round and shared across all clients -- never
    recomputed per client, and never using a future round or any
    ground-truth field), then each client's historical residual calibration
    (:func:`compute_historical_residual_calibration`, using only rounds
    ``[1, t_ref)``), then the one-sided score itself:
    ``z_t_res(k,r) = max(0, [d(k,r) - historical_center_k] / historical_scale_k)``.
    ``attack_start_round``/``malicious_client_id`` are never read anywhere
    in this function.
    """
    client_ids = sorted({row["client_id"] for row in rows})
    round_ids = sorted({row["round_id"] for row in rows})

    m_calibration = {
        client_id: compute_longitudinal_calibration(rows, client_id, "update_l2_norm", t_ref)
        for client_id in client_ids
    }
    round_references = {
        round_id: compute_common_round_reference(rows, round_id, "self_prev_round_cosine")
        for round_id in round_ids
    }
    residual_calibration = {
        client_id: compute_historical_residual_calibration(
            rows, client_id, "self_prev_round_cosine", round_references, t_ref
        )
        for client_id in client_ids
    }

    scored: list[ScoredRowV3] = []
    for row in rows:
        client_id = row["client_id"]
        round_id = row["round_id"]

        z_m = None
        m_cal = m_calibration.get(client_id)
        if m_cal is not None and row.get("update_l2_norm") is not None:
            z_m = abs(row["update_l2_norm"] - m_cal.mean) / m_cal.std

        z_t_res = None
        res_cal = residual_calibration.get(client_id)
        c_r = round_references.get(round_id)
        t_value = row.get("self_prev_round_cosine")
        if res_cal is not None and c_r is not None and t_value is not None:
            d = c_r - float(t_value)
            z_t_res = max(0.0, (d - res_cal.mean) / res_cal.std)

        scored.append(
            ScoredRowV3(round_id=round_id, client_id=client_id, z_m=z_m, z_t_res=z_t_res)
        )
    return scored


def apply_persistence_rule_v3(
    scored_rows: Sequence[ScoredRowV3],
    components: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
) -> dict[str, ClientPersistenceResult]:
    """FedTrace v3's persistence rule -- algorithmically identical to v1's/v2's.

    Duplicated (not shared via a common helper) so that v1's and v2's own
    functions are provably untouched by this addition; the
    streak/first-anomalous/confirmation/onset logic below is byte-for-byte
    the same algorithm as v1's :func:`apply_persistence_rule` and v2's
    :func:`apply_persistence_rule_v2`, applied to :class:`ScoredRowV3`.
    Never reads ``attack_start_round`` or any other ground-truth field. A
    row whose combined score is undefined breaks any in-progress streak
    (treated the same as an explicitly non-anomalous round), per
    ``research_plans/FEDTRACE_V3_...PLAN.md`` §8/§16.
    """
    by_client: dict[str, list[ScoredRowV3]] = {}
    for row in scored_rows:
        by_client.setdefault(row.client_id, []).append(row)

    results: dict[str, ClientPersistenceResult] = {}
    for client_id, client_rows in by_client.items():
        ordered = sorted(client_rows, key=lambda r: r.round_id)

        first_anomalous_round: int | None = None
        confirmation_round: int | None = None
        onset_estimate: int | None = None
        streak_len = 0
        streak_start: int | None = None
        prev_round_id: int | None = None
        prev_anomalous = False
        run_scores: list[float] = []

        for row in ordered:
            combined = row.combined_score(components)
            score = combined[0] if combined is not None else None
            anomalous = score is not None and score > threshold

            if anomalous and first_anomalous_round is None:
                first_anomalous_round = row.round_id

            if anomalous:
                contiguous = (
                    prev_anomalous
                    and prev_round_id is not None
                    and row.round_id == prev_round_id + 1
                )
                if contiguous:
                    streak_len += 1
                else:
                    streak_len = 1
                    streak_start = row.round_id
                if score is not None:
                    run_scores.append(score)
                if streak_len >= persistence and confirmation_round is None:
                    confirmation_round = row.round_id
                    onset_estimate = streak_start
            else:
                streak_len = 0
                streak_start = None
                run_scores = []

            prev_anomalous = anomalous
            prev_round_id = row.round_id

        mean_score = (
            statistics.mean(run_scores) if (confirmation_round is not None and run_scores) else None
        )
        results[client_id] = ClientPersistenceResult(
            client_id=client_id,
            first_anomalous_round=first_anomalous_round,
            confirmation_round=confirmation_round,
            onset_estimate=onset_estimate,
            mean_score_during_run=mean_score,
        )
    return results


def evaluate_detector_v3(
    rows: Sequence[Mapping[str, Any]],
    detector_name: str,
    components: Sequence[str],
    t_ref: int = DEFAULT_T_REF,
    threshold: float = DEFAULT_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
) -> DetectorRunResult:
    """Evaluate one FedTrace-v3-family detector (``components`` subset of ``{"m", "t_res"}``).

    Structurally parallel to v1's :func:`evaluate_detector` and v2's
    :func:`evaluate_detector_v2` (same NOT-EVALUABLE rule, same
    attribution/precision/recall/F1/dominance computation), but scores rows
    via :func:`compute_all_scores_v3` / :func:`apply_persistence_rule_v3`.
    Duplicated rather than refactored into a shared helper, for the same
    provably-untouched-v1/v2 reason as :func:`apply_persistence_rule_v3`.
    """
    missing = [c for c in components if not field_available(rows, V3_COMPONENT_FIELDNAMES[c])]
    if missing == list(components):
        reason = (
            f"None of the requested component(s) {tuple(components)} have any "
            f"populated values in this run's trace rows (missing field(s): "
            f"{[V3_COMPONENT_FIELDNAMES[c] for c in missing]})."
        )
        return DetectorRunResult(
            detector_name=detector_name,
            components=tuple(components),
            evaluable=False,
            not_evaluable_reason=reason,
        )

    scored_rows = compute_all_scores_v3(rows, t_ref=t_ref)
    per_client = apply_persistence_rule_v3(scored_rows, components, threshold, persistence)
    predicted_client = _select_predicted_client(per_client)

    attack_rows = [row for row in rows if row.get("attack_configured")]
    true_malicious_client = attack_rows[0]["malicious_client_id"] if attack_rows else None
    attack_start_round = attack_rows[0]["attack_start_round"] if attack_rows else None

    attribution_correct: bool | None = None
    onset_error: int | None = None
    delay: int | None = None
    if true_malicious_client is not None:
        attribution_correct = predicted_client == true_malicious_client
        predicted_result = per_client.get(predicted_client) if predicted_client else None
        if predicted_result is not None and attribution_correct:
            if predicted_result.onset_estimate is not None and attack_start_round is not None:
                onset_error = abs(predicted_result.onset_estimate - attack_start_round)
            if predicted_result.confirmation_round is not None and attack_start_round is not None:
                delay = predicted_result.confirmation_round - attack_start_round

    detected_clients = {cid for cid, result in per_client.items() if result.detected}
    true_positive_clients = (
        tuple(sorted(detected_clients & {true_malicious_client})) if true_malicious_client else ()
    )
    false_positive_clients = tuple(sorted(detected_clients - {true_malicious_client}))
    false_negative = bool(true_malicious_client) and true_malicious_client not in detected_clients

    tp = len(true_positive_clients)
    fp = len(false_positive_clients)
    fn = 1 if false_negative else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    dominant_counts: dict[str, int] = {c: 0 for c in components}
    for row in scored_rows:
        combined = row.combined_score(components)
        if combined is not None and combined[0] > threshold:
            dominant_counts[combined[1]] += 1

    return DetectorRunResult(
        detector_name=detector_name,
        components=tuple(components),
        evaluable=True,
        not_evaluable_reason=None,
        per_client=per_client,
        predicted_client=predicted_client,
        true_malicious_client=true_malicious_client,
        attack_start_round=attack_start_round,
        attribution_correct=attribution_correct,
        onset_localization_error=onset_error,
        detection_delay=delay,
        true_positive_clients=true_positive_clients,
        false_positive_clients=false_positive_clients,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
        dominant_component_counts=dominant_counts,
    )


def evaluate_all_detectors_v3(
    rows: Sequence[Mapping[str, Any]],
    t_ref: int = DEFAULT_T_REF,
    threshold: float = DEFAULT_THRESHOLD,
    persistence: int = DEFAULT_PERSISTENCE,
) -> dict[str, DetectorRunResult]:
    """Evaluate the full v1-vs-v2-vs-v3 comparison set (detectors A-H) required by the v3 retrospective task.

    Returns eight detectors: the four unchanged v1 ones (magnitude-only,
    peer-only, v1-temporal-only, FedTrace v1), the two v2 ones
    (v2-temporal-only, FedTrace v2), and the two new v3 ones
    (temporal-residual-only, FedTrace v3).
    """
    results: dict[str, DetectorRunResult] = {
        "A_magnitude_only": evaluate_detector(
            rows, "A_magnitude_only", ("m",), t_ref=t_ref, threshold=threshold, persistence=persistence
        ),
        "B_peer_only": evaluate_detector(
            rows, "B_peer_only", ("p",), t_ref=t_ref, threshold=threshold, persistence=persistence
        ),
        "C_v1_temporal_only": evaluate_detector(
            rows, "C_v1_temporal_only", ("t",), t_ref=t_ref, threshold=threshold, persistence=persistence
        ),
        "D_v2_temporal_only": evaluate_detector_v2(
            rows, "D_v2_temporal_only", ("t_rel",), t_ref=t_ref, threshold=threshold, persistence=persistence
        ),
        "F_fedtrace_v1": evaluate_detector(
            rows,
            "F_fedtrace_v1",
            ("m", "p", "t"),
            t_ref=t_ref,
            threshold=threshold,
            persistence=persistence,
        ),
        "G_fedtrace_v2": evaluate_detector_v2(
            rows, "G_fedtrace_v2", ("m", "t_rel"), t_ref=t_ref, threshold=threshold, persistence=persistence
        ),
    }
    for name, components in V3_DETECTOR_COMPONENTS.items():
        results[name] = evaluate_detector_v3(
            rows, name, components, t_ref=t_ref, threshold=threshold, persistence=persistence
        )
    return results
