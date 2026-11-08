"""P1 causal failure analysis (read-only, CPU). Rounds 15-35 of X2 (BadNets) with X1 (clean) as counterfactual."""

import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.analysis.fedtrace_detector import (  # noqa: E402
    compute_common_round_reference,
    compute_historical_residual_calibration,
    compute_longitudinal_calibration,
    load_trace_rows,
)
from src.rev.online import OnlineConfig, causal_scores_v3, online_detect_v3  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
cfg = OnlineConfig()
X = {name: load_trace_rows(Path(f"results_rev/{name}/client_round_trace.csv")) for name in ("X1_clean_d123", "X2_badnets_d123")}
CLIENTS = sorted({r["client_id"] for r in X["X2_badnets_d123"]})


def table(rows):
    refs = {r: compute_common_round_reference(rows, r, "self_prev_round_cosine") for r in sorted({x["round_id"] for x in rows})}
    mcal = {c: compute_longitudinal_calibration(rows, c, "update_l2_norm", cfg.t_ref) for c in CLIENTS}
    rcal = {c: compute_historical_residual_calibration(rows, c, "self_prev_round_cosine", refs, cfg.t_ref) for c in CLIENTS}
    causal = {(s.round_id, s.client_id): s for s in causal_scores_v3(rows, cfg)}
    by = {(r["round_id"], r["client_id"]): r for r in rows}
    out, streak = [], {c: 0 for c in CLIENTS}
    alarm = {c: None for c in CLIENTS}
    for rnd in range(15, 36):
        for c in CLIENTS:
            row = by[(rnd, c)]
            s = causal.get((rnd, c))
            score = s.combined_score(cfg.components) if s else None
            exceed = bool(score and score[0] > cfg.threshold)
            if rnd >= cfg.t_ref:
                streak[c] = streak[c] + 1 if exceed else 0
                if streak[c] >= cfg.persistence and alarm[c] is None:
                    alarm[c] = rnd
            t = row.get("self_prev_round_cosine")
            d = refs[rnd] - t if (t is not None and refs[rnd] is not None) else None
            out.append({
                "round": rnd, "client": c, "norm": row["update_l2_norm"],
                "norm_over_cal_mean": row["update_l2_norm"] / mcal[c].mean,
                "cal_mean": mcal[c].mean, "cal_sd": mcal[c].std,
                "self_cos": t, "round_median_self_cos": refs[rnd], "residual_d": d,
                "res_cal_mean": rcal[c].mean if rcal[c] else None, "res_cal_sd": rcal[c].std if rcal[c] else None,
                "peer_cos": row.get("peer_mean_cosine_similarity"),
                "z_m": s.z_m if s else None, "z_t_res": s.z_t_res if s else None,
                "score": score[0] if score else None, "dominant": score[1] if score else None,
                "exceeds_tau": exceed if rnd >= cfg.t_ref else None,
                "streak": streak[c] if rnd >= cfg.t_ref else None,
                "alarm_state": (alarm[c] is not None and rnd >= alarm[c]) if rnd >= cfg.t_ref else None,
                "kind": "causal" if rnd >= cfg.t_ref else "calibration (not scored online)",
            })
    return out, mcal


summary = {}
for name, rows in X.items():
    tab, mcal = table(rows)
    with (OUT / f"rounds15_35_{name}.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(tab[0]))
        w.writeheader()
        w.writerows(tab)
    det = online_detect_v3(rows, cfg)
    summary[name] = {"events": {c: vars(e) for c, e in det.per_client.items()},
                     "calibration_norm": {c: [mcal[c].mean, mcal[c].std, mcal[c].std / mcal[c].mean] for c in CLIENTS}}

# counterfactual: client_002 norm in X2 vs X1, and honest clients
by1 = {(r["round_id"], r["client_id"]): r for r in X["X1_clean_d123"]}
by2 = {(r["round_id"], r["client_id"]): r for r in X["X2_badnets_d123"]}
cf = []
for rnd in range(15, 36):
    rec = {"round": rnd}
    for c in CLIENTS:
        rec[c] = by2[(rnd, c)]["update_l2_norm"] / by1[(rnd, c)]["update_l2_norm"]
    cf.append(rec)
summary["norm_ratio_attacked_over_clean"] = cf
cal_rows = [r["update_l2_norm"] for r in X["X2_badnets_d123"] if r["client_id"] == "client_002" and r["round_id"] < 20]
summary["client_002_calibration_norms_rounds_0_19"] = cal_rows
(OUT / "diagnosis_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")

a = [r for r in table(X["X2_badnets_d123"])[0] if r["client"] == "client_002"]
print("client_002 (X2) round | norm  norm/calmean  z_m   self_cos  med_cos  d      z_t_res score exceed streak alarm")
for r in a:
    f = lambda v, n=4: "None" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))
    print(r["round"], f(r["norm"], 3), f(r["norm_over_cal_mean"], 3), f(r["z_m"], 2), f(r["self_cos"], 3), f(r["round_median_self_cos"], 3),
          f(r["residual_d"], 3), f(r["z_t_res"], 2), f(r["score"], 2), r["exceeds_tau"], r["streak"], r["alarm_state"])
print("calibration (mean, sd, cv) per client X2:", {c: [round(v, 3) for v in summary["X2_badnets_d123"]["calibration_norm"][c]] for c in CLIENTS})
print("client_002 calibration norms r0-19:", [round(v, 2) for v in cal_rows])
print("norm ratio attacked/clean (round: c000 c001 c002 c003 c004):")
for rec in cf:
    print(rec["round"], *[f"{rec[c]:.3f}" for c in CLIENTS])
