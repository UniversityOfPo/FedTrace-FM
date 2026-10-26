"""Order-statistic reference contamination in FedTrace v3 (protocol 25; CPU, read-only).

    python -m src.rev.contamination

CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY. Not a detector: nothing here is fitted or thresholded for
decisions; the constants below are fixed reporting levels from protocol 25 §5.

For round r, candidate source j and other client k (protocol 25 §2):
    c(r)            inclusive median of all defined self-cosines (exactly v3's round reference)
    band_j(r)       the median as a function of j's value with the others fixed: c(x) = med(others ∪ {x})
    Δc_j^dep(r)     = c(r) − c_{x = x̂_j(r)}(r), with x̂_j(r) = m_{−j}(r) + o_j: j placed at its historical offset
                     o_j = mean over calibration rounds 1..19 of (t_j − m_{−j}) from the median m_{−j} of the others
                     (DEPLOYABLE: own and peers' history and the current round only; a common-mode shift of all
                     clients leaves Δc^dep unchanged, a single client's move does not)
    Δc_j^retro(r)   = c(r) − c_{x = t_j^clean(r)}(r)  (RETROSPECTIVE: uses the matched clean run; diagnosis only)
    Δz_{k←j}        = Δc_j / σ_k, σ_k = k's v3 calibration sd of d_k = c − t_k (rounds 1..19)
    Δc^obs(r)       = c(r) − mean_{r' in 1..19} c(r')   (BASELINE B0, deployable: observed reference movement only)
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis.fedtrace_detector import ScoredRowV3, apply_persistence_rule_v3, load_trace_rows
from src.rev.online import OnlineConfig, causal_scores_v3
from src.rev.reference_audit import clients_of, inclusive_median, residuals, self_cosines

T_REF, TAU, P = 20, 3.0, 3
AFFECT_LEVEL = 1.0      # reporting level for "affected" client-rounds (z units)
SIGN_LEVEL = 0.5        # client-rounds with |retrospective displacement| >= this enter the sign test
SIGN_PASS = 0.9         # protocol 25 H-C2
SIGMA_SCALES = (0.8, 1.25)
CONFIG = OnlineConfig()


# ---------------------------------------------------------------- exact algebra

def median_with(others: Sequence[float], x: float) -> float:
    """Median of the others together with value x (exactly v3's inclusive median when x is j's value)."""
    return statistics.median(list(others) + [x])


def clip_form(others: Sequence[float], x: float) -> float:
    """Closed form of median_with (protocol 25 Lemma A): odd K: clip(x, s_{m-1}, s_m); even K: mean of two clips."""
    s = sorted(others)
    n = len(s)  # = K - 1
    clip = lambda v, lo, hi: min(max(v, lo), hi)
    if n % 2 == 0:  # K odd, K = n + 1 = 2m - 1
        m = (n + 2) // 2
        return clip(x, s[m - 2], s[m - 1])
    q = (n + 1) // 2  # K even, K = 2q
    lo, mid, hi = s[q - 2] if q >= 2 else float("-inf"), s[q - 1], s[q] if q < n else float("inf")
    return 0.5 * (clip(x, lo, mid) + clip(x, mid, hi))


# ---------------------------------------------------------------- per-run quantities

def _values(t, r, clients):
    return {c: t[(r, c)] for c in clients if t.get((r, c)) is not None}


def analyse_run(rows: Sequence[Mapping[str, Any]], clean_rows: Sequence[Mapping[str, Any]] | None = None,
                source: str | None = None) -> dict[str, Any]:
    """All per-round quantities for one run. ``clean_rows`` / ``source`` are retrospective inputs (evaluation only)."""
    t = self_cosines(rows)
    clients = clients_of(rows)
    res = residuals(rows, inclusive_median)
    last = max(r for r, _ in t)
    sigma = {k: next(iter(v.values()))["cal_sd"] for k, v in res.items() if v}
    offset = {}
    for j in clients:
        diffs = [t[(r, j)] - statistics.median(o) for r in range(1, T_REF)
                 if t.get((r, j)) is not None and len(o := [v for (rr, c), v in t.items() if rr == r and c != j and v is not None]) >= 1]
        offset[j] = statistics.mean(diffs)
    cal_c = statistics.mean(m for r in range(1, T_REF) if (m := inclusive_median(t, r, None, clients)) is not None)
    tc = self_cosines(clean_rows) if clean_rows is not None else None
    res_c = residuals(clean_rows, inclusive_median) if clean_rows is not None else None
    per: dict[int, Any] = {}
    identity_err = 0.0
    for r in range(T_REF, last + 1):
        vals = _values(t, r, clients)
        if len(vals) < 2:
            continue
        c = statistics.median(vals.values())
        rec: dict[str, Any] = {"c": c, "dc_obs": c - cal_c, "dep": {}, "retro": {}}
        for j in vals:
            others = [v for i, v in vals.items() if i != j]
            xhat = statistics.median(others) + offset[j]
            identity_err = max(identity_err, abs(clip_form(others, vals[j]) - c), abs(clip_form(others, xhat) - median_with(others, xhat)))
            rec["dep"][j] = c - median_with(others, xhat)
            if tc is not None and tc.get((r, j)) is not None:
                rec["retro"][j] = c - median_with(others, tc[(r, j)])
        if tc is not None:
            cv = _values(tc, r, clients)
            rec["c_clean"] = statistics.median(cv.values()) if len(cv) >= 2 else None
            rec["dz_total"] = {k: (res[k][r]["d"] - res_c[k][r]["d"]) / sigma[k]
                               for k in clients if k in sigma and r in res.get(k, {}) and r in (res_c.get(k) or {})}
        per[r] = rec
    return {"clients": clients, "sigma": sigma, "offset": offset, "cal_c": cal_c, "per": per, "res": res,
            "identity_max_abs_error": identity_err, "source": source, "has_clean": clean_rows is not None}


# ---------------------------------------------------------------- v3 alarms with components (frozen rule)

def v3_alarms(rows, res, components=("m", "t_res")) -> dict[str, tuple[int, int, float]]:
    clients = clients_of(rows)
    last = max(int(r["round_id"]) for r in rows)
    zm = {(s.round_id, s.client_id): s.z_m for s in causal_scores_v3(rows, CONFIG)}
    scored = [ScoredRowV3(round_id=r, client_id=k, z_m=zm.get((r, k)),
                          z_t_res=max(0.0, res[k][r]["z"]) if r in res.get(k, {}) else None)
              for k in clients for r in range(T_REF, last + 1)]
    per = apply_persistence_rule_v3(scored, components, TAU, P)
    score = {(s.round_id, s.client_id): (s.combined_score(components) or (None,))[0] for s in scored}
    out = {}
    for k, p in per.items():
        if p.confirmation_round is not None:
            streak = [score[(r, k)] for r in range(p.onset_estimate, p.confirmation_round + 1)]
            out[k] = (p.confirmation_round, p.onset_estimate, statistics.mean(streak))
    return out


def explain_alarms(rows, run: Mapping[str, Any], honest: Sequence[str], sigma_scale: float = 1.0) -> list[dict[str, Any]]:
    """Necessity test (protocol 25 §4 D) for every confirmed honest client."""
    res, per, sigma = run["res"], run["per"], run["sigma"]
    zm = {(s.round_id, s.client_id): s.z_m for s in causal_scores_v3(rows, CONFIG)}
    alarms = v3_alarms(rows, res)
    out = []
    for k, (a, onset, _) in sorted(alarms.items()):
        if k not in honest:
            continue
        streak = list(range(a - P + 1, a + 1))
        zt = {r: res[k][r]["z"] for r in streak}
        m = {r: zm.get((r, k)) or 0.0 for r in streak}
        norm_driven_all = all(m[r] >= TAU for r in streak)
        t_needed = any(m[r] < TAU for r in streak)

        def breaks(dz: Mapping[int, float]) -> int:
            return sum(max(m[r], max(0.0, zt[r] - dz[r])) < TAU for r in streak)

        sc = sigma[k] * sigma_scale
        cand = {}
        for j in run["clients"]:
            if j == k:
                continue
            dz = {r: per[r]["dep"].get(j, 0.0) / sc for r in streak}
            cand[j] = (breaks(dz), sum(dz.values()))
        jstar = max(cand, key=lambda j: (cand[j][0], cand[j][1], j)) if cand else None
        rec = {"client": k, "alarm": a, "onset_est": onset, "streak": streak,
               "z_t_signed": [round(zt[r], 3) for r in streak], "z_m": [round(m[r], 3) for r in streak],
               "classification": "norm-driven (z_m >= tau throughout; outside the mechanism)" if norm_driven_all else "t_res-necessary",
               "dep_best_source": jstar, "dep_rounds_broken": cand[jstar][0] if jstar else 0,
               "dep_explained": bool(jstar and cand[jstar][0] >= 1 and t_needed),
               "B0_rounds_broken": breaks({r: per[r]["dc_obs"] / sc for r in streak}),
               "sigma_scale": sigma_scale}
        rec["B0_explained"] = rec["B0_rounds_broken"] >= 1 and t_needed
        if run["has_clean"] and run["source"]:
            dz = {r: per[r]["retro"].get(run["source"], 0.0) / sc for r in streak}
            rec["retro_rounds_broken"] = breaks(dz)
            rec["retro_explained"] = rec["retro_rounds_broken"] >= 1 and t_needed
        out.append(rec)
    return out


def displacement_tests(run: Mapping[str, Any], honest: Sequence[str], window: range) -> dict[str, Any]:
    """Sign and magnitude agreement (protocol 25 §4 B, C) on honest client-rounds of matched runs, rounds onset..onset+9."""
    src, per, sigma = run["source"], run["per"], run["sigma"]
    rows = []
    for r in window:
        if r not in per or src not in per[r]["retro"]:
            continue
        for k in honest:
            if k not in sigma or k not in per[r].get("dz_total", {}):
                continue
            rows.append({"r": r, "k": k, "retro": per[r]["retro"][src] / sigma[k], "dep": per[r]["dep"][src] / sigma[k],
                         "obs": per[r]["dc_obs"] / sigma[k], "total": per[r]["dz_total"][k],
                         "total_median": (per[r]["c"] - per[r]["c_clean"]) / sigma[k]})
    big = [x for x in rows if abs(x["retro"]) >= SIGN_LEVEL]
    sgn = lambda v: (v > 0) - (v < 0)
    ef = lambda pred: (1 - sum((x["total"] - x[pred]) ** 2 for x in rows) / sum(x["total"] ** 2 for x in rows)
                       if rows and sum(x["total"] ** 2 for x in rows) > 0 else None)
    mae = lambda pred: statistics.mean(abs(x[pred] - x["retro"]) for x in rows) if rows else None
    return {"n_client_rounds": len(rows), "n_sign_test": len(big),
            "sign_agree_dep": sum(sgn(x["dep"]) == sgn(x["retro"]) for x in big),
            "sign_agree_B0": sum(sgn(x["obs"]) == sgn(x["retro"]) for x in big),
            "mae_dep_vs_retro": mae("dep"), "mae_B0_vs_retro": mae("obs"),
            "explained_fraction_of_total_by_source_term": ef("retro"),
            "explained_fraction_of_total_by_total_median_change": ef("total_median"),
            "rows": rows}


def clean_rates(run: Mapping[str, Any]) -> dict[str, Any]:
    """Protocol 25 §4 F: in runs with no intervention, how often does each description flag an honest client-round?"""
    per, sigma = run["per"], run["sigma"]
    n = mech = b0 = 0
    for r, rec in per.items():
        for k in sigma:
            if k not in rec["dep"]:
                continue
            n += 1
            mech += max((rec["dep"][j] / sigma[k] for j in rec["dep"] if j != k), default=0.0) >= AFFECT_LEVEL
            b0 += abs(rec["dc_obs"] / sigma[k]) >= AFFECT_LEVEL
    return {"client_rounds": n, "mechanism_flag_rate": mech / n if n else None, "B0_flag_rate": b0 / n if n else None}


# ---------------------------------------------------------------- driver

ARCHIVE = sorted(glob.glob("results/federated/*/client_round_trace.csv") + glob.glob("results_ndss/*/*/client_round_trace.csv"))
DEV = {f"X{i}_{n}_d{s}": (n, s) for s in ("123", "2026") for i, n in ((1, "clean"), (2, "badnets"), (3, "relabel"), (4, "drift"))}


def truth_of(rows, name: str) -> tuple[str | None, int | None, bool]:
    """(intervened client, onset, malicious) — evaluation only."""
    if name in DEV:
        cond = DEV[name][0]
        return (None, None, False) if cond == "clean" else ("client_002", 20, cond in ("badnets", "relabel"))
    att = [r for r in rows if r.get("attack_configured")]
    return (att[0]["malicious_client_id"], att[0]["attack_start_round"], True) if att else (None, None, False)


def matched_clean(root: Path, rows, name: str, seed: str | None):
    """X1 clean of the same seed if its trace rows equal this run's rows before the onset (protocol 25 §3)."""
    if seed not in ("123", "2026") or name.startswith("X1_"):
        return None
    clean = load_trace_rows(root / f"results_rev/X1_clean_d{seed}/client_round_trace.csv")
    ck = {(r["round_id"], r["client_id"]): r for r in clean}
    _, onset, _ = truth_of(rows, name)
    if onset is None or len({r["client_id"] for r in rows}) != len({r["client_id"] for r in clean}):
        return None
    fields = ("update_l2_norm", "self_prev_round_cosine", "peer_mean_cosine_similarity")
    same = all(all(ck.get((r["round_id"], r["client_id"]), {}).get(f) == r.get(f) for f in fields)
               for r in rows if r["round_id"] < onset)
    return clean if same else None


def run_all(root: Path) -> dict[str, Any]:
    traces = {Path(p).parent.parent.name[:12] + "/" + Path(p).parent.name: Path(p) for p in ARCHIVE}
    traces.update({n: root / f"results_rev/{n}/client_round_trace.csv" for n in DEV})
    report: dict[str, Any] = {"status": "CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY (protocol 25)", "runs": {}}
    for name, path in traces.items():
        rows = load_trace_rows(path)
        seed = DEV[name][1] if name in DEV else next((s for s in ("123", "2026", "42") if name.endswith(f"seed{s}") or f"seed{s}_" in name), None)
        src, onset, malicious = truth_of(rows, name)
        clean = matched_clean(root, rows, name, seed)
        run = analyse_run(rows, clean, src)
        honest = [c for c in run["clients"] if c != src or not malicious]
        honest_others = [c for c in run["clients"] if c != src]
        entry = {"seed": seed, "intervened": src, "onset": onset, "malicious": malicious, "matched_clean": clean is not None,
                 "K": len(run["clients"]), "identity_max_abs_error": run["identity_max_abs_error"],
                 "sigma": run["sigma"], "alarms_v3": {k: list(v) for k, v in v3_alarms(rows, run["res"]).items()},
                 "honest_alarm_explanations": explain_alarms(rows, run, honest),
                 "sensitivity": {str(s): explain_alarms(rows, run, honest, s) for s in SIGMA_SCALES},
                 "source_dep_dc": {r: rec["dep"].get(src) for r, rec in run["per"].items()} if src else None,
                 "source_retro_dc": {r: rec["retro"].get(src) for r, rec in run["per"].items()} if src and clean is not None else None,
                 "dc_obs": {r: rec["dc_obs"] for r, rec in run["per"].items()}}
        if src and clean is not None:
            entry["displacement_tests"] = displacement_tests(run, honest_others, range(onset, onset + 10))
        affected = {}
        for r, rec in run["per"].items():
            for k in run["sigma"]:
                best = max(((rec["dep"][j] / run["sigma"][k], j) for j in rec["dep"] if j != k), default=(0.0, None))
                if best[0] >= AFFECT_LEVEL:
                    affected.setdefault(k, []).append((r, best[1], round(best[0], 3)))
        entry["affected_client_rounds_dep"] = affected
        if src is None:
            entry["clean_rates"] = clean_rates(run)
        report["runs"][name] = entry
    report["verdicts"] = verdicts(report)
    out = root / "results_rev/pilot/contamination"
    out.mkdir(parents=True, exist_ok=True)
    (out / "contamination.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report


def verdicts(report: Mapping[str, Any]) -> dict[str, Any]:
    runs = report["runs"]
    v: dict[str, Any] = {}
    v["A_identity_max_abs_error"] = max(e["identity_max_abs_error"] for e in runs.values())
    tests = [e["displacement_tests"] for e in runs.values() if "displacement_tests" in e]
    n = sum(t["n_sign_test"] for t in tests)
    agree = sum(t["sign_agree_dep"] for t in tests)
    agree_b0 = sum(t["sign_agree_B0"] for t in tests)
    rows = [x for t in tests for x in t["rows"]]
    mae_dep = statistics.mean(abs(x["dep"] - x["retro"]) for x in rows) if rows else None
    mae_b0 = statistics.mean(abs(x["obs"] - x["retro"]) for x in rows) if rows else None
    v["H_C2"] = {"n_sign_test": n, "sign_agreement_dep": agree / n if n else None, "sign_agreement_B0": agree_b0 / n if n else None,
                 "mae_dep": mae_dep, "mae_B0": mae_b0,
                 "verdict": "NOT EVALUABLE" if not n else ("PASS" if agree / n >= SIGN_PASS and mae_dep < mae_b0 else "FAIL")}
    t_alarms = [(name, a) for name, e in runs.items() for a in e["honest_alarm_explanations"] if a["classification"] == "t_res-necessary"]
    unexplained = [(name, a["client"]) for name, a in t_alarms if not a["dep_explained"]]
    wrong_source = [(name, a["client"], a["dep_best_source"]) for name, a in t_alarms
                    if runs[name]["intervened"] and a["dep_explained"] and a["dep_best_source"] != runs[name]["intervened"]]
    v["H_C3"] = {"t_res_necessary_honest_alarms": [(n_, a["client"], a["alarm"]) for n_, a in t_alarms],
                 "unexplained": unexplained, "explained_with_wrong_source": wrong_source,
                 "norm_driven_honest_alarms": [(n_, a["client"], a["alarm"]) for n_, e in runs.items()
                                               for a in e["honest_alarm_explanations"] if a["classification"] != "t_res-necessary"],
                 "B0_explains": sum(a["B0_explained"] for _, a in t_alarms),
                 "verdict": "NOT EVALUABLE (no t_res-necessary honest alarm)" if not t_alarms else
                            ("PASS" if not unexplained and not wrong_source else "FAIL")}
    clean = {name: e["clean_rates"] for name, e in runs.items() if "clean_rates" in e}
    lower = sum(c["mechanism_flag_rate"] < c["B0_flag_rate"] for c in clean.values() if c["client_rounds"])
    v["H_C4"] = {"clean_runs": clean, "runs_mechanism_lower": lower, "n_clean_runs": len(clean),
                 "verdict": "PASS" if clean and lower >= len(clean) - 1 else "FAIL"}
    v["overall"] = ("MECHANISM SUPPORTED (retrospective + deployable, development / archived scope)"
                    if v["H_C2"]["verdict"] == "PASS" and v["H_C3"]["verdict"] == "PASS" else "MECHANISM CLAIM NOT SUPPORTED")
    v["non_trivial_vs_B0"] = v["H_C4"]["verdict"] == "PASS" and v["H_C2"]["verdict"] == "PASS"
    return v


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Protocol 25 contamination validation (read-only).").parse_args(argv)
    rep = run_all(Path("."))
    print(json.dumps(rep["verdicts"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
