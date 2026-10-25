"""Phase S harness: standard baselines on the development runs (protocol ``paper_plan/12_phase_s_protocol.md``).

    python -m src.rev.phase_s --seeds 123 2026

Read-only, CPU, POST-HOC EXPLORATORY, DEVELOPMENT ONLY. Wires existing code together exactly as the frozen protocol
specifies; no new detector logic. Writes ``results_rev/pilot/phase_s/``. Thresholds of group Q are calibrated on
clean development runs only (``calibrate`` never receives attacked runs); attacker identity and onset are used only
by ``evaluate_run`` to score outcomes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.analysis.fedtrace_detector import load_trace_rows
from src.rev.baselines import BaselineEvents, cusum, page_hinkley, zscore_persistence
from src.rev.reference_audit import REFERENCES, clients_of, decompose, detect, residuals, self_cosines

T_REF, PERSISTENCE, K_CUSUM, DELTA_PH = 20, 3, 0.5, 0.5
GRID = tuple(x * 0.25 for x in range(1, 241))
ATTACKER, ONSET = "client_002", 20
REF_IDS = {"dA": "inclusive_median", "dB": "leave_one_out_median", "dC": "trimmed_mean", "dD": "historical_own_mean"}
FEATURES = ("m", "t", "dA", "dB", "dC", "dD")


# ---------------------------------------------------------------- features (online-computable)
def feature_series(rows: Sequence[Mapping[str, Any]], feature: str) -> dict[str, dict[int, float]]:
    clients = clients_of(rows)
    if feature == "m":
        return {c: {int(r["round_id"]): float(r["update_l2_norm"]) for r in rows
                    if r["client_id"] == c and r.get("update_l2_norm") is not None} for c in clients}
    t = self_cosines(rows)
    if feature == "t":
        return {c: {r: v for (r, k), v in t.items() if k == c and v is not None} for c in clients}
    ref = REFERENCES[REF_IDS[feature]]
    out = {}
    for c in clients:
        series = {}
        for (r, k), v in t.items():
            if k != c or v is None:
                continue
            value = ref(t, r, c, clients)
            if value is not None:
                series[r] = value - v
        out[c] = series
    return out


def calibration_defined(series: Mapping[int, float]) -> bool:
    cal = [v for r, v in series.items() if r < T_REF]
    return len(cal) >= 2 and statistics.stdev(cal) > 0


def standardized(series: Mapping[int, float]) -> dict[int, float]:
    cal = [v for r, v in series.items() if r < T_REF]
    mu, sd = statistics.mean(cal), statistics.stdev(cal)
    return {r: (v - mu) / sd for r, v in series.items() if r >= T_REF}


# ---------------------------------------------------------------- group Q detectors (two-sided, conventional)
def _earliest(up: BaselineEvents, down: BaselineEvents) -> BaselineEvents:
    cands = [e for e in (up, down) if e.alarm_round is not None]
    return min(cands, key=lambda e: e.alarm_round) if cands else up


def run_q(detector: str, series: Mapping[int, float], h: float) -> BaselineEvents:
    if detector == "ZP":
        return zscore_persistence(series, t_ref=T_REF, threshold=h, persistence=PERSISTENCE, two_sided=True)
    if detector == "CUSUM":
        return _earliest(cusum(series, t_ref=T_REF, k=K_CUSUM, h=h, standardize=True),
                         cusum({r: -v for r, v in series.items()}, t_ref=T_REF, k=K_CUSUM, h=h, standardize=True))
    if detector == "PH":
        z = standardized(series)
        return _earliest(page_hinkley(z, t_ref=T_REF, delta=DELTA_PH, lam=h),
                         page_hinkley({r: -v for r, v in z.items()}, t_ref=T_REF, delta=DELTA_PH, lam=h))
    raise ValueError(detector)


def calibrate(clean_runs: Sequence[Sequence[Mapping[str, Any]]], detector: str, feature: str) -> float | None:
    """Smallest grid value with zero client-level alarms over all clean development runs (protocol §4)."""
    series = [s for rows in clean_runs for s in feature_series(rows, feature).values() if calibration_defined(s)]
    for h in GRID:
        if all(run_q(detector, s, h).alarm_round is None for s in series):
            return h
    return None


def q_alarms(rows: Sequence[Mapping[str, Any]], detector: str, feature: str, h: float) -> tuple[dict, list[str]]:
    alarms, undefined = {}, []
    for c, s in feature_series(rows, feature).items():
        if not calibration_defined(s):
            undefined.append(c)
            continue
        e = run_q(detector, s, h)
        if e.alarm_round is not None:
            alarms[c] = (e.alarm_round, e.estimated_onset, e.statistic.get(e.alarm_round, 0.0))
    return alarms, undefined


def v_alarms(rows: Sequence[Mapping[str, Any]], reference: str) -> tuple[dict, list[str]]:
    res = residuals(rows, REFERENCES[reference])
    out = detect(rows, res)
    undefined = [c for c, v in res.items() if not v]
    ranking = out["ranking"]
    return {c: (a[0], a[1], -ranking.index(c)) for c, a in out["alarms"].items()}, undefined


# ---------------------------------------------------------------- outcome scoring (evaluation only)
def evaluate_run(alarms: Mapping[str, tuple], attacker: str | None, onset: int | None) -> dict[str, Any]:
    order = sorted(alarms, key=lambda c: (alarms[c][0], -alarms[c][2], c))
    first = order[0] if order else None
    honest = sorted(c for c in alarms if c != attacker)
    res: dict[str, Any] = {"first_alarmed": first, "attributed": first, "honest_false_alarms": honest,
                           "n_honest_false_alarms": len(honest),
                           "first_alarm_round": alarms[first][0] if first else None,
                           "actionable_round": alarms[first][0] + 1 if first else None,
                           "alarm_rounds": {c: alarms[c][0] for c in order}}
    if attacker is not None:
        att = alarms.get(attacker)
        detected = att is not None and att[0] >= onset
        res.update({"attacker_detected": detected, "attribution_correct": first == attacker and detected,
                    "attacker_alarm": att[0] if att else None,
                    "delay": (att[0] - onset) if detected else None, "delay_censored_at": None if detected else 29,
                    "estimated_onset": att[1] if detected else None,
                    "onset_error": abs(att[1] - onset) if detected and att[1] is not None else None})
    return res


def contamination(clean, attacked, reference: str, attacker: str = ATTACKER) -> dict[str, Any]:
    """DIAGNOSIS ONLY (uses the clean counterfactual)."""
    dec = decompose(clean, attacked, attacker, range(ONSET, ONSET + 10), REFERENCES[reference])
    rc, ra = residuals(clean, REFERENCES[reference]), residuals(attacked, REFERENCES[reference])
    att = [ra[attacker][r]["z"] - rc[attacker][r]["z"] for r in range(21, 30)
           if r in ra.get(attacker, {}) and r in rc.get(attacker, {})]
    return {"attacker_induced_movement": {k: statistics.mean(x["attacker_induced"] for x in v) for k, v in dec.items() if k != attacker and v},
            "honest_standardised_change": {k: statistics.mean(x["delta_z"] for x in v) for k, v in dec.items() if k != attacker and v},
            "attacker_standardised_residual_change": statistics.mean(att) if len(att) == 9 else None}


def run_phase_s(root: Path, seeds: Sequence[str]) -> dict[str, Any]:
    load = lambda run: load_trace_rows(root / "results_rev" / run / "client_round_trace.csv")
    runs = {s: {"clean": load(f"X1_clean_d{s}"), "badnets": load(f"X2_badnets_d{s}"), "relabel": load(f"X3_relabel_d{s}")}
            for s in seeds}
    report: dict[str, Any] = {"status": "CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY (protocol 12)",
                              "methods": {}, "contamination": {}, "not_evaluable": ["FLDetector", "FLForensics", "FedREDefense"]}
    clean_runs = [runs[s]["clean"] for s in seeds]
    methods: dict[str, Callable] = {}
    for ref in REF_IDS.values():
        methods[f"V-{ref}"] = lambda rows, ref=ref: v_alarms(rows, ref)
    thresholds = {}
    for det in ("ZP", "CUSUM", "PH"):
        for feat in FEATURES:
            h = calibrate(clean_runs, det, feat)
            thresholds[f"Q-{det}-{feat}"] = h
            if h is not None:
                methods[f"Q-{det}-{feat}"] = lambda rows, det=det, feat=feat, h=h: q_alarms(rows, det, feat, h)
    report["thresholds"] = thresholds
    for name, fn in methods.items():
        per = {}
        for s in seeds:
            for arm, rows in runs[s].items():
                alarms, undefined = fn(rows)
                ev = evaluate_run(alarms, None if arm == "clean" else ATTACKER, None if arm == "clean" else ONSET)
                ev["undefined_clients"] = undefined
                per[f"{s}/{arm}"] = ev
        report["methods"][name] = per
    for s in seeds:
        for arm in ("badnets", "relabel"):
            report["contamination"][f"{s}/{arm}"] = {ref: contamination(runs[s]["clean"], runs[s][arm], ref) for ref in REF_IDS.values()}
    out = root / "results_rev" / "pilot" / "phase_s"
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase_s.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase S standard baselines (read-only, exploratory).")
    parser.add_argument("--seeds", nargs="+", default=["123", "2026"])
    args = parser.parse_args(argv)
    report = run_phase_s(Path("."), args.seeds)
    print(json.dumps(report["thresholds"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------- calibration sensitivity (report only; no selection)
def lag1_autocorrelation(values: Sequence[float]) -> float | None:
    if len(values) < 3:
        return None
    mu = statistics.mean(values)
    den = sum((v - mu) ** 2 for v in values)
    if den == 0:
        return None
    return sum((a - mu) * (b - mu) for a, b in zip(values, values[1:])) / den


def effective_n(n: int, rho: float | None) -> float | None:
    """AR(1) effective sample size n (1 - rho) / (1 + rho), floored at 1 (rho clipped to [-0.99, 0.99])."""
    if rho is None:
        return None
    rho = max(-0.99, min(0.99, rho))
    return max(1.0, n * (1 - rho) / (1 + rho))


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b) or len(a) < 3:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((y - mb) ** 2 for y in b) ** 0.5
    return None if sa == 0 or sb == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def calibration_sensitivity(clean_runs: Mapping[str, Sequence[Mapping[str, Any]]], feature: str = "m") -> dict[str, Any]:
    """Thresholds from each clean run alone and together, the binding client, and dependence diagnostics."""
    out: dict[str, Any] = {"thresholds": {}, "per_client": {}, "dependence": {}}
    names = sorted(clean_runs)
    for det in ("ZP", "CUSUM"):
        out["thresholds"][det] = {name: calibrate([clean_runs[name]], det, feature) for name in names}
        out["thresholds"][det]["all"] = calibrate([clean_runs[n] for n in names], det, feature)
    for name in names:
        rows = clean_runs[name]
        series = {c: s for c, s in feature_series(rows, feature).items() if calibration_defined(s)}
        z = {c: standardized(s) for c, s in series.items()}
        per = {}
        for c, zc in z.items():
            rounds = sorted(zc)
            absz = [abs(zc[r]) for r in rounds]
            zp_stat = max(min(absz[i:i + PERSISTENCE]) for i in range(len(absz) - PERSISTENCE + 1))
            rho = lag1_autocorrelation([zc[r] for r in rounds])
            per[c] = {"zp_binding_statistic": zp_stat, "lag1_autocorrelation": rho,
                      "n_rounds": len(rounds), "effective_n_ar1": effective_n(len(rounds), rho)}
        out["per_client"][name] = per
        clients = sorted(z)
        common = sorted(set.intersection(*(set(z[c]) for c in clients)))
        pairs = [_pearson([z[a][r] for r in common], [z[b][r] for r in common])
                 for i, a in enumerate(clients) for b in clients[i + 1:]]
        pairs = [p for p in pairs if p is not None]
        out["dependence"][name] = {"mean_pairwise_cross_client_correlation": statistics.mean(pairs) if pairs else None,
                                   "nominal_client_rounds": sum(v["n_rounds"] for v in per.values()),
                                   "sum_effective_n_ar1": sum(v["effective_n_ar1"] or 0 for v in per.values())}
    return out
