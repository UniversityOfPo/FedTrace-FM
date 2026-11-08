"""D1 explaining-away source localization: exploratory feasibility audit (protocol 14; CPU, read-only).

    python -m src.rev.source_attribution --seeds 123 2026

CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY. Not a detector and not an attribution of malice.

Ranking functions (``analyse``) take only the server-side trace of one run: per-round self-cosines and update norms of
every client. They receive no client identity, onset, matched clean run or future round; the score at round r uses
rounds <= r only. Ground truth enters only ``evaluate`` (retrospective). Nothing is fitted.

Notation (protocol 14 §3): t_k(r) self-cosine of client k; C_r clients with defined t_k(r); ref_E(S, r) reference
estimator E over set S; d_k(r; S) = ref_E(S, r) - t_k(r); calibration rounds 1..T_REF-1 under the same set rule;
z_k(r; S) = (d_k(r; S) - mu) / sd.
    A   leave-one-out candidate score     A_k(r) = z_k(r; C \\ {k})^2
    B   leave-one-source-out influence    I_j(r) = mean_{k != j} [ z_k(r; C)^2 - z_k(r; C \\ {j})^2 ]
    ABS / SIGNED  median displacement     D_j(r) = ref_E(C, r) - ref_E(C \\ {j}, r); scores mean|D_j|, |mean D_j|
    SELFCOS / NORM  own change            |mean_{T_REF..r} x_j - mu_pre(x_j)| / sd_pre(x_j)
All scores are aggregated as a cumulative mean over rounds T_REF..r and ranked descending.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.analysis.fedtrace_detector import load_trace_rows

T_REF, EVAL_ROUND, TIE_EPS, NEAR_TIE_REL = 20, 29, 1e-12, 0.01
METHODS = ("B", "A", "ABS", "SIGNED", "SELFCOS", "NORM")
CONDITIONS = {"clean": "X1_clean_d{s}", "badnets": "X2_badnets_d{s}", "relabel": "X3_relabel_d{s}", "drift": "X4_drift_d{s}"}


def _median(v: Sequence[float]) -> float:
    return statistics.median(v)


def _trimmed(v: Sequence[float]) -> float:
    s = sorted(v)
    return statistics.mean(s[1:-1]) if len(s) >= 3 else statistics.mean(s)


ESTIMATORS: dict[str, Callable[[Sequence[float]], float]] = {"MED": _median, "TRIM": _trimmed, "MEAN": statistics.mean}


def series(rows: Sequence[Mapping[str, Any]], field: str) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for r in rows:
        if r.get(field) is not None:
            out.setdefault(int(r["round_id"]), {})[r["client_id"]] = float(r[field])
    return out


def _sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    sd = statistics.stdev(values)
    return sd if sd > TIE_EPS else None


class _Residuals:
    """z_k(r; S) under a set rule, calibrated under the same rule over rounds 1..T_REF-1:
    "full" = C_r (inclusive), ("minus", j) = C_r without source j (k != j), "loo" = C_r without k itself."""

    def __init__(self, t: Mapping[int, Mapping[str, float]], estimator: str) -> None:
        self.t, self.est = t, ESTIMATORS[estimator]
        self._cal: dict[tuple, tuple[float, float] | None] = {}

    def _d(self, r: int, k: str, rule) -> float | None:
        vals = self.t.get(r, {})
        if k not in vals:
            return None
        if rule == "loo":
            members = [c for c in vals if c != k]
        else:
            if rule != "full" and rule[1] == k:
                return None
            members = [c for c in vals if rule == "full" or c != rule[1]]
        if len(members) < 2:
            return None
        return self.est([vals[c] for c in members]) - vals[k]

    def z(self, r: int, k: str, rule) -> float | None:
        key = (k, rule)
        if key not in self._cal:
            cal = [d for q in range(1, T_REF) if (d := self._d(q, k, rule)) is not None]
            sd = _sd(cal)
            self._cal[key] = (statistics.mean(cal), sd) if sd is not None else None
        d = self._d(r, k, rule)
        if d is None or self._cal[key] is None:
            return None
        mu, sd = self._cal[key]
        return (d - mu) / sd


def per_round_scores(rows: Sequence[Mapping[str, Any]], estimator: str = "MED") -> dict[str, dict[int, dict[str, float | None]]]:
    """Per method, round (>= T_REF) and candidate: the per-round score (None = undefined)."""
    t, n = series(rows, "self_prev_round_cosine"), series(rows, "update_l2_norm")
    res = _Residuals(t, estimator)
    est = ESTIMATORS[estimator]
    pre: dict[str, dict[str, tuple[float, float] | None]] = {"SELFCOS": {}, "NORM": {}}
    for name, x in (("SELFCOS", t), ("NORM", n)):
        for c in {c for v in x.values() for c in v}:
            cal = [x[q][c] for q in range(1, T_REF) if c in x.get(q, {})]
            sd = _sd(cal)
            pre[name][c] = (statistics.mean(cal), sd) if sd is not None else None
    out: dict[str, dict[int, dict[str, float | None]]] = {m: {} for m in METHODS}
    for r in sorted(q for q in t if q >= T_REF):
        clients = sorted(t[r])
        full = est([t[r][c] for c in clients])
        for j in clients:
            terms = []
            for k in clients:
                if k == j:
                    continue
                a, b = res.z(r, k, "full"), res.z(r, k, ("minus", j))
                if a is not None and b is not None:
                    terms.append(a * a - b * b)
            out["B"].setdefault(r, {})[j] = statistics.mean(terms) if len(terms) >= 2 else None
            za = res.z(r, j, "loo")
            out["A"].setdefault(r, {})[j] = za * za if za is not None else None
            rest = [t[r][c] for c in clients if c != j]
            disp = full - est(rest) if len(rest) >= 2 else None
            out["ABS"].setdefault(r, {})[j] = disp       # aggregated as mean |D| below
            out["SIGNED"].setdefault(r, {})[j] = disp    # aggregated as |mean D| below
            for name, x in (("SELFCOS", t), ("NORM", n)):
                out[name].setdefault(r, {})[j] = x.get(r, {}).get(j)  # raw value; standardised in aggregation
    out["_pre"] = pre  # type: ignore[assignment]
    return out


def cumulative(per_round: Mapping[str, Any], method: str, upto: int) -> dict[str, float | None]:
    """Online aggregate at round ``upto``: cumulative mean over rounds T_REF..upto (undefined rounds skipped;
    None if no defined round)."""
    rounds = [r for r in sorted(per_round[method]) if T_REF <= r <= upto]
    clients = sorted({c for r in rounds for c in per_round[method][r]})
    out: dict[str, float | None] = {}
    for c in clients:
        vals = [per_round[method][r].get(c) for r in rounds]
        vals = [v for v in vals if v is not None]
        if not vals:
            out[c] = None
        elif method == "ABS":
            out[c] = statistics.mean(abs(v) for v in vals)
        elif method == "SIGNED":
            out[c] = abs(statistics.mean(vals))
        elif method in ("SELFCOS", "NORM"):
            p = per_round["_pre"][method].get(c)
            out[c] = abs(statistics.mean(vals) - p[0]) / p[1] if p is not None else None
        else:
            out[c] = statistics.mean(vals)
    return out


def rank(scores: Mapping[str, float | None]) -> dict[str, Any]:
    """Descending score; exact ties (|diff| <= 1e-12) broken by client id and flagged; undefined ranked last."""
    defined = sorted((c for c in scores if scores[c] is not None), key=lambda c: (-scores[c], c))
    order = defined + sorted(c for c in scores if scores[c] is None)
    top, second = (defined + [None, None])[:2]
    margin = scores[top] - scores[second] if top is not None and second is not None else None
    groups: list[list[str]] = []
    for c in defined:  # consecutive candidates whose scores differ by <= TIE_EPS
        if groups and scores[groups[-1][-1]] - scores[c] <= TIE_EPS:
            groups[-1].append(c)
        else:
            groups.append([c])
    exact_tie_top = margin is not None and margin <= TIE_EPS
    near_tie_top = margin is not None and abs(scores[top]) > 0 and margin <= NEAR_TIE_REL * abs(scores[top])
    return {"order": order, "top": top, "margin": margin, "exact_tie_at_top": exact_tie_top,
            "near_tie_at_top": near_tie_top, "undefined": sorted(c for c in scores if scores[c] is None),
            "tie_groups": [g for g in groups if len(g) > 1]}


def positions(t: Mapping[int, Mapping[str, float]], r: int) -> dict[str, int]:
    """Order-statistic position of each client's self-cosine in round r (1 = lowest)."""
    order = sorted(t[r], key=lambda c: (t[r][c], c))
    return {c: i + 1 for i, c in enumerate(order)}


def analyse(rows: Sequence[Mapping[str, Any]], estimators: Sequence[str] = ("MED", "TRIM", "MEAN"),
            eval_round: int = EVAL_ROUND) -> dict[str, Any]:
    """Source rankings from one run's trace only. No identity, onset or counterfactual is an input."""
    t = series(rows, "self_prev_round_cosine")
    out: dict[str, Any] = {}
    last = max(t)
    for e in estimators:
        pr = per_round_scores(rows, e)
        rec: dict[str, Any] = {}
        for m in METHODS:
            rec[m] = {"at_eval": rank(cumulative(pr, m, eval_round)), "at_eval_scores": cumulative(pr, m, eval_round),
                      "cumulative_top_by_round": {r: rank(cumulative(pr, m, r))["top"] for r in range(T_REF, last + 1)}}
        b = pr["B"]
        per_round_top = {r: rank(b[r]) for r in range(T_REF, last + 1) if r in b}
        window = [r for r in range(T_REF, eval_round + 1) if r in per_round_top]
        cum_top = rec["B"]["at_eval"]["top"]
        rec["B"]["per_round"] = {r: {"scores": b[r], "top": per_round_top[r]["top"], "margin": per_round_top[r]["margin"],
                                     "exact_tie": per_round_top[r]["exact_tie_at_top"], "near_tie": per_round_top[r]["near_tie_at_top"],
                                     "top_position": positions(t, r).get(per_round_top[r]["top"]) if per_round_top[r]["top"] else None}
                                 for r in per_round_top}
        stable_from = None
        for r in range(eval_round, T_REF - 1, -1):
            if rec["B"]["cumulative_top_by_round"].get(r) != cum_top:
                break
            stable_from = r
        rec["B"]["stability"] = {
            "rounds_per_round_top_equals_cumulative_top": sum(per_round_top[r]["top"] == cum_top for r in window),
            "window_rounds": len(window),
            "distinct_per_round_tops": sorted({per_round_top[r]["top"] for r in window if per_round_top[r]["top"]}),
            "cumulative_top_unchanged_from_round": stable_from,
            "exact_ties_in_window": sum(per_round_top[r]["exact_tie_at_top"] for r in window),
            "near_ties_in_window": sum(per_round_top[r]["near_tie_at_top"] for r in window),
            "per_round_top_position_counts": {p: sum(rec["B"]["per_round"][r]["top_position"] == p for r in window) for p in range(1, 6)},
        }
        out[e] = rec
    return out


def kendall_tau(a: Sequence[str], b: Sequence[str]) -> float:
    pos = {c: i for i, c in enumerate(b)}
    pairs = list(itertools.combinations(a, 2))
    s = sum(1 if pos[x] < pos[y] else -1 for x, y in pairs)
    return s / len(pairs)


def evaluate(analysis: Mapping[str, Any], true_source: str | None) -> dict[str, Any]:
    """RETROSPECTIVE ONLY: ground truth enters here, never in ``analyse``."""
    out: dict[str, Any] = {}
    for e, rec in analysis.items():
        out[e] = {}
        for m in METHODS:
            order = rec[m]["at_eval"]["order"]
            out[e][m] = {"rank_of_true_source": order.index(true_source) + 1 if true_source in order else None,
                         "top_is_true_source": rec[m]["at_eval"]["top"] == true_source if true_source else None}
        out[e]["kendall_B_vs_ABS"] = kendall_tau(rec["B"]["at_eval"]["order"], rec["ABS"]["at_eval"]["order"])
        out[e]["kendall_B_vs_SIGNED"] = kendall_tau(rec["B"]["at_eval"]["order"], rec["SIGNED"]["at_eval"]["order"])
    return out


def verdicts(results: Mapping[str, Any]) -> dict[str, Any]:
    """Primary falsification criteria F1-F5 (protocol 14 §10), MED unless stated."""
    rows = results["runs"]
    med = lambda k: rows[k]["evaluation"]["MED"]
    out: dict[str, Any] = {}
    motivating = [k for k in ("2026/badnets", "2026/relabel") if k in rows]
    out["F1_motivating_case"] = ("FALSIFIED" if any(not med(k)["B"]["top_is_true_source"] for k in motivating)
                                 else "NOT FALSIFIED (2 development runs)")
    intervened = [k for k in rows if not k.endswith("/clean")]
    for base in ("ABS", "SIGNED"):
        better = sum(med(k)["B"]["rank_of_true_source"] < med(k)[base]["rank_of_true_source"] for k in intervened)
        worse = sum(med(k)["B"]["rank_of_true_source"] > med(k)[base]["rank_of_true_source"] for k in intervened)
        out[f"F2_vs_{base}"] = {"better": better, "worse": worse,
                                "verdict": "ADDS RANK INFORMATION" if better >= 1 and worse == 0 else "NO EVIDENCE OF ADDED INFORMATION"}
    correct_margins = [rows[k]["analysis"]["MED"]["B"]["at_eval"]["margin"] for k in intervened if med(k)["B"]["top_is_true_source"]]
    clean_margins = {k: rows[k]["analysis"]["MED"]["B"]["at_eval"]["margin"] for k in rows if k.endswith("/clean")}
    floor = min(correct_margins) if correct_margins else None
    out["F3_clean_winner"] = {"smallest_correct_margin": floor, "clean_margins": clean_margins,
                              "verdict": ("NOT EVALUABLE (no correct localization)" if floor is None else
                                          "CLEAN WINNER AS DECISIVE AS A CORRECT LOCALIZATION"
                                          if any(v is not None and v >= floor for v in clean_margins.values())
                                          else "CLEAN MARGINS BELOW EVERY CORRECT LOCALIZATION")}
    out["F4_unstable_runs"] = sorted(k for k in rows if rows[k]["analysis"]["MED"]["B"]["stability"]["rounds_per_round_top_equals_cumulative_top"] < 5)
    out["F5_estimator_dependent"] = sorted(k for k in motivating
                                           if len({rows[k]["analysis"][e]["B"]["at_eval"]["top"] for e in rows[k]["analysis"]}) > 1)
    return out


def run(root: Path, seeds: Sequence[str], out_dir: Path | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {"status": "CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY (protocol 14)", "runs": {}}
    for s in seeds:
        for cond, pattern in CONDITIONS.items():
            rows = load_trace_rows(root / "results_rev" / pattern.format(s=s) / "client_round_trace.csv")
            analysis = analyse(rows)
            truth = None if cond == "clean" else "client_002"  # evaluation only
            results["runs"][f"{s}/{cond}"] = {
                "run": pattern.format(s=s), "true_intervention_source_EVALUATION_ONLY": truth,
                "source_is_malicious_EVALUATION_ONLY": cond in ("badnets", "relabel"),
                "analysis": analysis, "evaluation": evaluate(analysis, truth)}
    results["verdicts"] = verdicts(results)
    out = out_dir or root / "results_rev" / "pilot" / "source_attribution"
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_attribution.json").write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D1 source-localization feasibility audit (read-only).")
    parser.add_argument("--seeds", nargs="+", default=["123", "2026"])
    args = parser.parse_args(argv)
    run(Path("."), args.seeds)
    print("written results_rev/pilot/source_attribution/source_attribution.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
