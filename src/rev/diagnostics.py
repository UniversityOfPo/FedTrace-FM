"""Read-only forensic diagnostics for a matched clean / attacked pair (CPU; sealed run outputs only).

    python -m src.rev.diagnostics diagnose --clean X1_clean_d2026 --attacked X2_badnets_d2026 \
        --malicious client_002 --onset 20
    python -m src.rev.diagnostics exploratory --clean X1_clean_d123 --attacked X2_badnets_d123 \
        --malicious client_002 --onset 20 [--include-archived]

``diagnose`` computes the PREREGISTERED diagnostic endpoints of ``paper_plan/08_G1_preregistration.md`` and applies
its fixed verdict rules (constants below; they must not change after results are seen). It writes
``results_rev/pilot/<attacked>__vs__<clean>/diagnostics/``.

``exploratory`` reproduces the POST-HOC EXPLORATORY comparisons of pilot P1 (established onset estimators and
two-sided change detectors with thresholds from the clean run only). Its outputs go to ``diagnostics_posthoc/`` and
are never evidence of generalisation.

Causality: every detector-side quantity for round r >= T_ref uses rounds <= r only. The matched clean counterfactual
(norm ratios) and the true onset are used for EXPERIMENTAL DIAGNOSIS ONLY and never enter a detector. Values for
rounds < T_ref are calibration-window values and are labelled ``retrospective``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.analysis.fedtrace_detector import (
    compute_common_round_reference,
    compute_historical_residual_calibration,
    compute_longitudinal_calibration,
    load_trace_rows,
)
from src.rev.baselines import cusum, page_hinkley, smallest_threshold_without_client_alarms
from src.rev.online import OnlineConfig, causal_scores_v3, online_detect_v3

# ---- Preregistered constants (paper_plan/08_G1_preregistration.md) -- do not edit after G1 results exist ----
T_REF = 20
Z_CUT = 3.0                 # magnitude of a signed residual counted as extreme
RUN_LENGTH = 3              # consecutive extreme rounds that count as a sustained shift
EARLY_WINDOW = 3            # rounds onset .. onset+2
POST_WINDOW = 10            # rounds onset .. onset+9
FALSIFY_MEDIAN_Z = -1.0     # attacker median signed z above this in the post window falsifies Observation D
NORM_WINDOW = (5, 15)       # rounds onset+5 .. onset+15 for the norm-ratio endpoint
NORM_REPLICATED_MAX = 0.90
NORM_NOT_REPLICATED_MIN = 0.97
HONEST_RATIO_BAND = (0.97, 1.03)
BURN_IN_PRIMARY = 2
BURN_IN_EXPLORATORY = (1, 3, 5)
SPREAD_RATIO_CUT = 1.5
ASR_EFFECTIVE = 0.50
ASR_PARTIAL_MIN_GAIN = 0.10
PREREGISTERED = ("T_REF", "Z_CUT", "RUN_LENGTH", "EARLY_WINDOW", "POST_WINDOW", "FALSIFY_MEDIAN_Z", "NORM_WINDOW",
                 "NORM_REPLICATED_MAX", "NORM_NOT_REPLICATED_MIN", "HONEST_RATIO_BAND", "BURN_IN_PRIMARY",
                 "BURN_IN_EXPLORATORY", "SPREAD_RATIO_CUT", "ASR_EFFECTIVE", "ASR_PARTIAL_MIN_GAIN")


def _by_key(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {(int(r["round_id"]), str(r["client_id"])): r for r in rows}


def clients_of(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(r["client_id"]) for r in rows})


def signed_residuals(rows: Sequence[Mapping[str, Any]], t_ref: int = T_REF) -> dict[str, dict[int, dict[str, Any]]]:
    """Signed temporal residual z = (d - mu_k)/sd_k with v3's own residual calibration (rounds [1, t_ref)).

    d = c(r) - t(k, r) as in v3; v3 scores only max(0, z). Available online from t_ref on: c(r) is the median of
    round-r self-cosines (all submissions of round r), t(k, r) needs only k's previous update.
    """
    rounds = sorted({int(r["round_id"]) for r in rows})
    refs = {r: compute_common_round_reference(rows, r, "self_prev_round_cosine") for r in rounds}
    by = _by_key(rows)
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for c in clients_of(rows):
        cal = compute_historical_residual_calibration(rows, c, "self_prev_round_cosine", refs, t_ref)
        out[c] = {}
        for r in rounds:
            row = by.get((r, c))
            t = None if row is None else row.get("self_prev_round_cosine")
            d = None if (t is None or refs[r] is None) else refs[r] - float(t)
            z = None if (d is None or cal is None) else (d - cal.mean) / cal.std
            out[c][r] = {"residual_d": d, "signed_z": z, "kind": "causal" if r >= t_ref else "retrospective"}
    return out


def norm_ratios(clean: Sequence[Mapping[str, Any]], attacked: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, float]]:
    """Attacked / matched-clean update norm per client and round (DIAGNOSIS ONLY: uses the counterfactual run)."""
    a, b = _by_key(clean), _by_key(attacked)
    out: dict[str, dict[int, float]] = {}
    for (r, c), row in b.items():
        base = a.get((r, c))
        if base and base.get("update_l2_norm") and row.get("update_l2_norm") is not None:
            out.setdefault(c, {})[r] = float(row["update_l2_norm"]) / float(base["update_l2_norm"])
    return out


def calibration_spread(rows: Sequence[Mapping[str, Any]], burn_ins: Iterable[int], t_ref: int = T_REF) -> dict[str, dict[str, Any]]:
    """Sample sd of the update norm over rounds < t_ref, with and without the first b rounds (diagnostic only)."""
    out: dict[str, dict[str, Any]] = {}
    for c in clients_of(rows):
        values = [float(r["update_l2_norm"]) for r in sorted(rows, key=lambda x: int(x["round_id"]))
                  if r["client_id"] == c and int(r["round_id"]) < t_ref and r.get("update_l2_norm") is not None]
        rec: dict[str, Any] = {"sd_all": statistics.stdev(values), "mean_all": statistics.mean(values)}
        for b in burn_ins:
            rest = values[b:]
            sd = statistics.stdev(rest)
            rec[f"sd_without_first_{b}"] = sd
            rec[f"sd_ratio_b{b}"] = rec["sd_all"] / sd if sd > 0 else None
        out[c] = rec
    return out


def _max_run(values: Sequence[float | None], predicate) -> int:
    best = run = 0
    for v in values:
        run = run + 1 if (v is not None and predicate(v)) else 0
        best = max(best, run)
    return best


def observation_d_verdict(clean_res: Mapping[str, Mapping[int, Mapping[str, Any]]],
                          attacked_res: Mapping[str, Mapping[int, Mapping[str, Any]]],
                          attacker: str, onset: int, last_round: int) -> dict[str, Any]:
    """Preregistered support / falsification rule for Observation D (attacker becomes more self-consistent)."""
    z = lambda res, c, rounds: [res[c].get(r, {}).get("signed_z") for r in rounds]
    early = [v for v in z(attacked_res, attacker, range(onset, onset + EARLY_WINDOW)) if v is not None]
    post = [v for v in z(attacked_res, attacker, range(onset, onset + POST_WINDOW)) if v is not None]
    scored = range(T_REF, last_round + 1)
    clean_runs = {c: _max_run(z(clean_res, c, scored), lambda v: v <= -Z_CUT) for c in clean_res}
    honest_runs = {c: _max_run(z(attacked_res, c, scored), lambda v: v <= -Z_CUT) for c in attacked_res if c != attacker}
    s1 = bool(early) and min(early) <= -Z_CUT
    median_post = statistics.median(post) if post else None
    s2 = median_post is not None and median_post <= -Z_CUT
    s3 = all(n < RUN_LENGTH for n in clean_runs.values())
    s4 = all(n < RUN_LENGTH for n in honest_runs.values())
    f1 = median_post is not None and median_post > FALSIFY_MEDIAN_Z
    f2 = not s3
    verdict = "FALSIFIED" if (f1 or f2) else ("SUPPORTED" if (s1 and s2 and s3 and s4) else "INCONCLUSIVE")
    return {"verdict": verdict, "S1_min_early_z": min(early) if early else None, "S1": s1,
            "S2_median_post_z": median_post, "S2": s2, "S3_clean_max_negative_run": clean_runs, "S3": s3,
            "S4_attacked_honest_max_negative_run": honest_runs, "S4": s4, "F1": f1, "F2": f2}


def norm_verdict(ratios: Mapping[str, Mapping[int, float]], attacker: str, onset: int) -> dict[str, Any]:
    window = range(onset + NORM_WINDOW[0], onset + NORM_WINDOW[1] + 1)
    mean = lambda c: statistics.mean(ratios[c][r] for r in window if r in ratios[c])
    att = mean(attacker)
    honest = {c: mean(c) for c in ratios if c != attacker}
    honest_ok = all(HONEST_RATIO_BAND[0] <= v <= HONEST_RATIO_BAND[1] for v in honest.values())
    if att <= NORM_REPLICATED_MAX and honest_ok:
        verdict = "REPLICATED"
    elif att >= NORM_NOT_REPLICATED_MIN:
        verdict = "NOT_REPLICATED" + (" (INCREASE)" if att > HONEST_RATIO_BAND[1] else "")
    else:
        verdict = "PARTIAL"
    return {"verdict": verdict, "attacker_mean_ratio": att, "honest_mean_ratios": honest, "honest_within_band": honest_ok}


def spread_verdict(spread: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    key = f"sd_ratio_b{BURN_IN_PRIMARY}"
    hits = sorted(c for c, rec in spread.items() if rec.get(key) is not None and rec[key] >= SPREAD_RATIO_CUT)
    n = len(spread)
    verdict = "REPLICATED" if len(hits) >= 3 else ("NOT_REPLICATED" if len(hits) <= 1 else "PARTIAL")
    return {"verdict": verdict, "clients_with_ratio_at_least_cut": hits, "n_clients": n, "burn_in": BURN_IN_PRIMARY}


def asr_classification(final_asr_attacked: float, final_asr_clean: float) -> str:
    if final_asr_attacked >= ASR_EFFECTIVE:
        return "EFFECTIVE"
    if final_asr_attacked - final_asr_clean >= ASR_PARTIAL_MIN_GAIN:
        return "PARTIALLY_EFFECTIVE"
    return "INEFFECTIVE"


def v3_round_table(rows: Sequence[Mapping[str, Any]], first: int, last: int, config: OnlineConfig | None = None) -> list[dict[str, Any]]:
    """Per client and round: norm, z_m, z_t_res, combined score, exceedance, streak and online alarm state."""
    config = config or OnlineConfig()
    causal = {(s.round_id, s.client_id): s for s in causal_scores_v3(rows, config)}
    mcal = {c: compute_longitudinal_calibration(rows, c, "update_l2_norm", config.t_ref) for c in clients_of(rows)}
    by = _by_key(rows)
    streak = {c: 0 for c in clients_of(rows)}
    alarm: dict[str, int | None] = {c: None for c in clients_of(rows)}
    out = []
    for r in range(config.t_ref, last + 1):  # streak state must be accumulated from t_ref
        for c in clients_of(rows):
            s = causal.get((r, c))
            comb = s.combined_score(config.components) if s else None
            exceed = bool(comb and comb[0] > config.threshold)
            streak[c] = streak[c] + 1 if exceed else 0
            if streak[c] >= config.persistence and alarm[c] is None:
                alarm[c] = r
            if r >= first:
                row = by.get((r, c), {})
                out.append({"round": r, "client": c, "norm": row.get("update_l2_norm"),
                            "norm_over_cal_mean": (row.get("update_l2_norm") / mcal[c].mean) if mcal[c] and row.get("update_l2_norm") else None,
                            "z_m": s.z_m if s else None, "z_t_res": s.z_t_res if s else None,
                            "score": comb[0] if comb else None, "dominant": comb[1] if comb else None,
                            "exceeds_tau": exceed, "streak": streak[c],
                            "alarm_state": alarm[c] is not None and r >= alarm[c]})
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    q = statistics.quantiles(values, n=20, method="inclusive")
    return {"n": len(values), "min": min(values), "p5": q[0], "median": statistics.median(values), "p95": q[-1],
            "max": max(values), "frac_le_minus3": sum(v <= -Z_CUT for v in values) / len(values),
            "frac_ge_plus3": sum(v >= Z_CUT for v in values) / len(values)}


def diagnose(root: Path, clean: str, attacked: str, attacker: str, onset: int) -> dict[str, Any]:
    pair = root / "results_rev" / "pilot" / f"{attacked}__vs__{clean}"
    out = pair / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    rows_c = load_trace_rows(root / "results_rev" / clean / "client_round_trace.csv")
    rows_a = load_trace_rows(root / "results_rev" / attacked / "client_round_trace.csv")
    last = max(int(r["round_id"]) for r in rows_a)

    res_c, res_a = signed_residuals(rows_c), signed_residuals(rows_a)
    for name, res in ((clean, res_c), (attacked, res_a)):
        _write_csv(out / f"signed_residuals_{name}.csv",
                   [{"round": r, "client": c, **v} for c in sorted(res) for r, v in sorted(res[c].items())])
    ratios = norm_ratios(rows_c, rows_a)
    _write_csv(out / "norm_ratio_attacked_over_clean.csv",
               [{"round": r, **{c: ratios[c].get(r) for c in sorted(ratios)}} for r in sorted(next(iter(ratios.values())))])
    spread = calibration_spread(rows_c, (BURN_IN_PRIMARY, *BURN_IN_EXPLORATORY))
    for name, rows in ((clean, rows_c), (attacked, rows_a)):
        _write_csv(out / f"v3_rounds_{name}.csv", v3_round_table(rows, max(T_REF, onset - 5), min(last, onset + 15)))

    honest_dist = {}
    for name, res, exclude in ((clean, res_c, None), (attacked, res_a, attacker)):
        values = [v["signed_z"] for c, per in res.items() if c != exclude for r, v in per.items()
                  if r >= T_REF and v["signed_z"] is not None]
        honest_dist[name] = _quantiles(values)

    effect = pair / "summary.json"
    summary = json.loads(effect.read_text(encoding="utf-8")) if effect.exists() else {}
    asr_att, asr_cln = summary.get(f"final_asr_{attacked}"), summary.get(f"final_asr_{clean}")
    report = {
        "status": "PREREGISTERED DIAGNOSTIC ENDPOINTS (DEVELOPMENT ONLY)",
        "constants": {k: globals()[k] for k in PREREGISTERED},
        "endpoint_1_norm_ratio": norm_verdict(ratios, attacker, onset),
        "endpoint_2_attacker_signed_residual": {r: res_a[attacker][r]["signed_z"] for r in range(onset - 5, min(last, onset + 15) + 1)},
        "endpoint_3_honest_signed_residual_distribution": honest_dist,
        "endpoint_4_calibration_spread": {"primary_b2_verdict": spread_verdict(spread), "per_client": spread,
                                          "note": f"b={BURN_IN_PRIMARY} preregistered; b in {BURN_IN_EXPLORATORY} EXPLORATORY; the v3 detector is unchanged"},
        "endpoint_5_v3": {name: {c: vars(e) for c, e in online_detect_v3(rows).per_client.items()}
                          for name, rows in ((clean, rows_c), (attacked, rows_a))},
        "endpoint_6_asr_classification": asr_classification(asr_att, asr_cln) if asr_att is not None else "NOT AVAILABLE (run src.rev.pilot first)",
        "observation_D": observation_d_verdict(res_c, res_a, attacker, onset, last),
    }
    (out / "preregistered_endpoints.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------- POST-HOC EXPLORATORY (pilot P1) -------------
K_CUSUM, DELTA_PH = 0.5, 0.5
GRID = [x * 0.25 for x in range(1, 241)]
KAPPAS = (0.5, 1.0, 1.5, 2.0)
ARCHIVED = {
    "EXP_014": ("results/federated/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed42", "client_002", 20),
    "EXP_015": ("results/federated/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed42", "client_002", 20),
    "EXP_016": ("results/federated/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123", "client_002", 20),
    "EXP_017": ("results/federated/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed2026", "client_002", 20),
    "EXP_018": ("results/federated/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123", "client_002", 20),
    "EXP_019": ("results/federated/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed2026", "client_002", 20),
    "EXP_026": ("results/federated/cifar100_fedavg_dirichlet_a01_direction_attack_client004_round20_seed123", "client_004", 20),
    "EXP_027": ("results/federated/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round30_seed123", "client_002", 30),
    "E1_gaussian_K10": ("results_ndss/E1_10clients/gaussian_client002_round20_seed123", "client_002", 20),
    "E1_directional_K10": ("results_ndss/E1_10clients/directional_client002_round20_seed123", "client_002", 20),
    "E2_gaussian_rho050": ("results_ndss/E2_attack_severity/gaussian_rho050_client002_round20_seed123", "client_002", 20),
    "E2_directional_c050": ("results_ndss/E2_attack_severity/directional_c050_client002_round20_seed123", "client_002", 20),
}


def standardized_feature(rows: Sequence[Mapping[str, Any]], kind: str, t_ref: int = T_REF) -> dict[str, dict[int, float]]:
    """Per client: feature standardised with its own rounds < t_ref; only rounds >= t_ref are returned (causal)."""
    res = signed_residuals(rows, t_ref) if kind == "resid" else None
    out = {}
    for c in clients_of(rows):
        if kind == "norm":
            series = {int(r["round_id"]): float(r["update_l2_norm"]) for r in rows
                      if r["client_id"] == c and r.get("update_l2_norm") is not None}
        else:
            series = {r: v["residual_d"] for r, v in res[c].items() if v["residual_d"] is not None}
        cal = [v for r, v in series.items() if r < t_ref]
        mu, sd = statistics.mean(cal), statistics.stdev(cal)
        out[c] = {r: (v - mu) / sd for r, v in series.items() if r >= t_ref}
    return out


def _two_sided(detector, series, threshold):
    up, down = detector(series, threshold), detector({r: -v for r, v in series.items()}, threshold)
    cands = [e for e in (up, down) if e.alarm_round is not None]
    return min(cands, key=lambda e: e.alarm_round) if cands else up


EXPLORATORY_DETECTORS = {
    "CUSUM_norm": ("norm", lambda s, h: cusum(s, t_ref=T_REF, k=K_CUSUM, h=h, standardize=False)),
    "CUSUM_residual": ("resid", lambda s, h: cusum(s, t_ref=T_REF, k=K_CUSUM, h=h, standardize=False)),
    "PH_norm": ("norm", lambda s, h: page_hinkley(s, t_ref=T_REF, delta=DELTA_PH, lam=h)),
    "PH_residual": ("resid", lambda s, h: page_hinkley(s, t_ref=T_REF, delta=DELTA_PH, lam=h)),
}


def change_time(scores: Mapping[int, float | None], alarm: int, kappa: float, t_ref: int = T_REF) -> int:
    """CUSUM change time on the v3 score at the alarm round (rounds t_ref..alarm only)."""
    c, last_zero = 0.0, t_ref - 1
    for r in range(t_ref, alarm + 1):
        c = max(0.0, c + (scores.get(r) or 0.0) - kappa)
        if c == 0.0 and r < alarm:
            last_zero = r
    return last_zero + 1


def exploratory(root: Path, clean: str, attacked: str, attacker: str, onset: int, include_archived: bool) -> dict[str, Any]:
    out = root / "results_rev" / "pilot" / f"{attacked}__vs__{clean}" / "diagnostics_posthoc"
    out.mkdir(parents=True, exist_ok=True)
    rows_c = load_trace_rows(root / "results_rev" / clean / "client_round_trace.csv")
    rows_a = load_trace_rows(root / "results_rev" / attacked / "client_round_trace.csv")
    report: dict[str, Any] = {"status": "POST-HOC EXPLORATORY, DEVELOPMENT ONLY; not method selection; not generalisation"}
    for name, (kind, det) in EXPLORATORY_DETECTORS.items():
        h = smallest_threshold_without_client_alarms([standardized_feature(rows_c, kind)], GRID,
                                                     lambda s, thr: _two_sided(det, s, thr))
        if h is None:
            report[name] = {"threshold_from_clean": None}
            continue
        events = {c: _two_sided(det, s, h) for c, s in standardized_feature(rows_a, kind).items()}
        alarmed = sorted((e.alarm_round, c) for c, e in events.items() if e.alarm_round is not None)
        report[name] = {"threshold_from_clean": h, "attributed": alarmed[0][1] if alarmed else None,
                        "alarm": alarmed[0][0] if alarmed else None,
                        "alarmed_clients": {c: [e.alarm_round, e.estimated_onset] for c, e in events.items() if e.alarm_round is not None}}
    traces = {attacked: (root / "results_rev" / attacked, attacker, onset)}
    if include_archived:
        traces.update({k: (root / p, a, o) for k, (p, a, o) in ARCHIVED.items()})
    onsets = {}
    for name, (path, who, true_onset) in traces.items():
        rows = load_trace_rows(Path(path) / "client_round_trace.csv")
        ev = online_detect_v3(rows).per_client[who]
        scores = {}
        for s in causal_scores_v3(rows, OnlineConfig()):
            if s.client_id == who:
                comb = s.combined_score(("m", "t_res"))
                scores[s.round_id] = comb[0] if comb else None
        rec = {"true_onset": true_onset, "v3_alarm": ev.alarm_round, "v3_streak_onset": ev.estimated_onset}
        if ev.alarm_round is not None:
            rec.update({f"change_time_k{k}": change_time(scores, ev.alarm_round, k) for k in KAPPAS})
        onsets[name] = rec
    report["onset_at_v3_alarm"] = onsets
    (out / "exploratory_posthoc.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only pair diagnostics.")
    parser.add_argument("mode", choices=("diagnose", "exploratory"))
    parser.add_argument("--clean", required=True)
    parser.add_argument("--attacked", required=True)
    parser.add_argument("--malicious", required=True)
    parser.add_argument("--onset", type=int, required=True)
    parser.add_argument("--include-archived", action="store_true")
    args = parser.parse_args(argv)
    root = Path(".")
    for name in (args.clean, args.attacked):
        if not (root / "results_rev" / name / "client_round_trace.csv").is_file():
            parser.error(f"missing trace for run {name}")
    if args.mode == "diagnose":
        report = diagnose(root, args.clean, args.attacked, args.malicious, args.onset)
        print(json.dumps({k: report[k] for k in ("endpoint_1_norm_ratio", "endpoint_6_asr_classification", "observation_D")},
                         indent=1, default=str))
        print(json.dumps(report["endpoint_4_calibration_spread"]["primary_b2_verdict"], default=str))
    else:
        report = exploratory(root, args.clean, args.attacked, args.malicious, args.onset, args.include_archived)
        print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
