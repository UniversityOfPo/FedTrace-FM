"""Reference-contamination and misattribution audit (read-only, CPU). POST-HOC EXPLORATORY, DEVELOPMENT ONLY.

Usage: python reference_audit.py <out_dir>
Seeds 123 and 2026 (development). Clean counterfactuals are used for DIAGNOSIS ONLY.
"""

import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, ".")
import torch  # noqa: E402

from src.analysis.fedtrace_detector import (  # noqa: E402
    ScoredRowV3,
    apply_persistence_rule_v3,
    compute_common_round_reference,
    compute_historical_residual_calibration,
    load_trace_rows,
)
from src.rev.hashing import load_tensor_state  # noqa: E402
from src.rev.online import OnlineConfig, causal_scores_v3, online_detect_v3  # noqa: E402
from src.rev.update_log import read_manifest  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
CFG = OnlineConfig()
T_REF, ATT, ONSET = 20, "client_002", 20
SEEDS = {"123": ("X1_clean_d123", "X2_badnets_d123"), "2026": ("X1_clean_d2026", "X2_badnets_d2026")}
CLIENTS = [f"client_00{i}" for i in range(5)]
report: dict = {"status": "POST-HOC EXPLORATORY, DEVELOPMENT ONLY; clean counterfactual used for diagnosis only"}


def trace(run):
    return load_trace_rows(Path("results_rev") / run / "client_round_trace.csv")


def tvals(rows):
    return {(r["round_id"], r["client_id"]): r.get("self_prev_round_cosine") for r in rows}


# ---------------------------------------------------------------- references (all use round-r submissions only)
def ref_inclusive(t, rnd, k):
    vals = [t[(rnd, c)] for c in CLIENTS if t.get((rnd, c)) is not None]
    return st.median(vals) if len(vals) >= 2 else None


def ref_loo(t, rnd, k):
    vals = [t[(rnd, c)] for c in CLIENTS if c != k and t.get((rnd, c)) is not None]
    return st.median(vals) if len(vals) >= 2 else None


def ref_trimmed(t, rnd, k):
    vals = sorted(t[(rnd, c)] for c in CLIENTS if t.get((rnd, c)) is not None)
    return st.mean(vals[1:-1]) if len(vals) >= 3 else None


def ref_historical(t, rnd, k):  # client's own calibration-window mean (rounds 1..T_REF-1); no cross-client input
    vals = [t[(r, k)] for r in range(1, T_REF) if t.get((r, k)) is not None]
    return st.mean(vals) if len(vals) >= 2 else None


REFS = {"A_inclusive_median": ref_inclusive, "B_leave_one_out_median": ref_loo,
        "C_trimmed_mean_inclusive": ref_trimmed, "D_historical_own_mean": ref_historical}


def residual_z(rows, ref):
    """Signed z of d = ref - t with per-client calibration over rounds [1, T_REF) (v3's residual convention)."""
    t = tvals(rows)
    out = {}
    for k in CLIENTS:
        d = {r: ref(t, r, k) - t[(r, k)] for r in range(1, 50) if t.get((r, k)) is not None and ref(t, r, k) is not None}
        cal = [v for r, v in d.items() if r < T_REF]
        mu, sd = st.mean(cal), st.stdev(cal)
        out[k] = {r: {"d": v, "z": (v - mu) / sd, "mu": mu, "sd": sd} for r, v in d.items()}
    return out


def detect(rows, zres):
    """Frozen v3 persistence (tau 3, P 3) on max(z_m, max(0, z_ref)) from T_REF; z_m unchanged; online tie-break."""
    zm = {(s.round_id, s.client_id): s.z_m for s in causal_scores_v3(rows, CFG)}
    scored = [ScoredRowV3(round_id=r, client_id=k, z_m=zm.get((r, k)),
                          z_t_res=max(0.0, zres[k][r]["z"]) if r in zres[k] else None)
              for k in CLIENTS for r in range(T_REF, 50)]
    per = apply_persistence_rule_v3(scored, ("m", "t_res"), CFG.threshold, CFG.persistence)
    score = {(s.round_id, s.client_id): (s.combined_score(("m", "t_res")) or (None,))[0] for s in scored}
    alarms = {k: (p.confirmation_round, p.onset_estimate) for k, p in per.items() if p.confirmation_round is not None}
    ranked = sorted(alarms, key=lambda k: (alarms[k][0], -st.mean(score[(r, k)] for r in range(alarms[k][1], alarms[k][0] + 1)), k))
    maxima = {k: max(v for (r, c), v in score.items() if c == k and v is not None) for k in CLIENTS}
    return {"alarms": alarms, "attributed": ranked[0] if ranked else None, "ranking": ranked, "max_score": maxima}


# ---------------------------------------------------------------- Task 2: seed-2026 causal chain, rounds 19-25
rows_a = trace("X2_badnets_d2026")
t = tvals(rows_a)
zA = residual_z(rows_a, ref_inclusive)
causal = {(s.round_id, s.client_id): s for s in causal_scores_v3(rows_a, CFG)}
online = online_detect_v3(rows_a, CFG)
chain, streak, alarm = [], {k: 0 for k in CLIENTS}, {k: None for k in CLIENTS}
for r in range(19, 26):
    inc = compute_common_round_reference(rows_a, r, "self_prev_round_cosine")
    assert abs(inc - st.median([t[(r, c)] for c in CLIENTS])) < 1e-15  # reference includes the candidate itself
    for k in CLIENTS:
        s = causal.get((r, k))
        comb = s.combined_score(CFG.components) if s else None
        if r >= T_REF:
            streak[k] = streak[k] + 1 if (comb and comb[0] > CFG.threshold) else 0
            if streak[k] >= CFG.persistence and alarm[k] is None:
                alarm[k] = r
        chain.append({"round": r, "client": k, "self_cos": t[(r, k)], "ref_inclusive_median": inc,
                      "ref_without_k": st.median([t[(r, c)] for c in CLIENTS if c != k]),
                      "residual_d": zA[k][r]["d"], "signed_z": zA[k][r]["z"], "res_cal_mu": zA[k][r]["mu"],
                      "res_cal_sd": zA[k][r]["sd"], "z_m": s.z_m if s else None, "z_t_res": s.z_t_res if s else None,
                      "score": comb[0] if comb else None, "dominant": comb[1] if comb else None,
                      "streak": streak[k] if r >= T_REF else None, "alarm": alarm[k]})
with (OUT / "seed2026_chain_r19_25.csv").open("w", newline="", encoding="utf-8") as h:
    w = csv.DictWriter(h, fieldnames=list(chain[0]))
    w.writeheader()
    w.writerows(chain)
report["task2_online_events"] = {k: vars(e) for k, e in online.per_client.items()}
report["task2_reference_includes_candidate"] = True

# ---------------------------------------------------------------- Task 3: decomposition (diagnosis only)
decomp = {}
for seed, (clean, att) in SEEDS.items():
    rc, ra = trace(clean), trace(att)
    tc, ta = tvals(rc), tvals(ra)
    zcal = residual_z(rc, ref_inclusive)  # same calibration in both arms (rounds < 20 identical)
    rec = {}
    for k in CLIENTS:
        sd = zcal[k][1]["sd"]
        rows = []
        for r in range(20, 30):
            dc = ref_inclusive(tc, r, k) - tc[(r, k)]
            da = ref_inclusive(ta, r, k) - ta[(r, k)]
            ref_shift = ref_inclusive(ta, r, k) - ref_inclusive(tc, r, k)
            own_shift = -(ta[(r, k)] - tc[(r, k)])
            rows.append({"round": r, "delta_d_z": (da - dc) / sd, "from_reference_z": ref_shift / sd,
                         "from_own_update_z": own_shift / sd})
        tot = sum(x["delta_d_z"] for x in rows)
        rec[k] = {"rounds": rows, "mean_delta_z": tot / len(rows),
                  "mean_from_reference_z": st.mean(x["from_reference_z"] for x in rows),
                  "mean_from_own_update_z": st.mean(x["from_own_update_z"] for x in rows)}
    decomp[seed] = rec
report["task3_decomposition_rounds20_29"] = {s: {k: {x: v[x] for x in ("mean_delta_z", "mean_from_reference_z", "mean_from_own_update_z")}
                                                 for k, v in rec.items()} for s, rec in decomp.items()}

# ---------------------------------------------------------------- Task 4: reference alternatives
alts = {}
for seed, (clean, att) in SEEDS.items():
    for arm, run in (("clean", clean), ("attacked", att)):
        rows = trace(run)
        for name, ref in REFS.items():
            z = residual_z(rows, ref)
            res = detect(rows, z)
            res["attacker_signed_z_r20_29"] = [round(z[ATT][r]["z"], 2) for r in range(20, 30)]
            res["honest_max_positive_z"] = max(z[k][r]["z"] for k in CLIENTS if k != ATT for r in range(20, 50) if r in z[k])
            alts[f"{seed}/{arm}/{name}"] = res
report["task4_alternatives"] = alts

# ---------------------------------------------------------------- Task 5: mechanism (update logs)
names = None


def deltas(run, client, rounds):
    log = Path("logs_rev") / run / "update_log"
    recs = read_manifest(log)
    glob = {r["round_id"]: r for r in recs if r.get("kind") == "global"}
    sub = {r["round_id"]: r for r in recs if r.get("kind") == "submitted" and r["client_id"] == client}
    out = {}
    for r in rounds:
        g = load_tensor_state(log / glob[r - 1]["file"])
        s = load_tensor_state(log / sub[r]["file"])
        out[r] = {k: (s[k] - g[k]).double().reshape(-1) for k in s if s[k].is_floating_point()
                  and not k.endswith(("running_mean", "running_var"))}
    return out


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def group(key):
    return "classifier" if key.startswith(("fc", "linear", "classifier", "head")) else key.split(".")[0]


mech = {}
for seed, (clean, att) in SEEDS.items():
    rounds = range(17, 31)
    da, dc = deltas(att, ATT, rounds), deltas(clean, ATT, rounds)
    keys = sorted(da[17])
    head = [k for k in keys if group(k) == "classifier"]
    groups = sorted({group(k) for k in keys})
    flat = lambda d, ks: torch.cat([d[k] for k in ks])
    per_round = []
    for r in range(18, 31):
        row = {"round": r,
               "self_cos_attacked": cos(flat(da[r], keys), flat(da[r - 1], keys)),
               "self_cos_clean": cos(flat(dc[r], keys), flat(dc[r - 1], keys))}
        for gname in groups:
            ks = [k for k in keys if group(k) == gname]
            row[f"{gname}_attacked"] = cos(flat(da[r], ks), flat(da[r - 1], ks))
            row[f"{gname}_clean"] = cos(flat(dc[r], ks), flat(dc[r - 1], ks))
        if r >= 21:
            u, up = flat(da[r], keys) - flat(dc[r], keys), flat(da[r - 1], keys) - flat(dc[r - 1], keys)
            row["attack_component_self_cos"] = cos(u, up)
            row["attack_component_share_of_norm"] = float(u.norm() / flat(da[r], keys).norm())
        # class-0 row of the classifier weight vs other rows
        w = [k for k in head if k.endswith("weight")]
        if w:
            Wa, Wp = da[r][w[0]].reshape(100, -1), da[r - 1][w[0]].reshape(100, -1)
            Wc, Wcp = dc[r][w[0]].reshape(100, -1), dc[r - 1][w[0]].reshape(100, -1)
            row["class0_row_norm_share_attacked"] = float(Wa[0].norm() / Wa.norm())
            row["class0_row_norm_share_clean"] = float(Wc[0].norm() / Wc.norm())
            row["classifier_without_class0_attacked"] = cos(Wa[1:].reshape(-1), Wp[1:].reshape(-1))
            row["classifier_without_class0_clean"] = cos(Wc[1:].reshape(-1), Wcp[1:].reshape(-1))
        per_round.append(row)
    honest_dt = {k: st.mean(tvals(trace(att))[(r, k)] - tvals(trace(clean))[(r, k)] for r in range(21, 30))
                 for k in CLIENTS}
    mech[seed] = {"per_round": per_round, "groups": groups, "classifier_keys": head,
                  "mean_self_cos_change_r21_29_attacked_minus_clean": honest_dt}
report["task5_mechanism"] = mech

# class composition of the attacker (labels from metadata, no download)
from torchvision.datasets import CIFAR100  # noqa: E402

train = CIFAR100("data", train=True, download=False)
import src.data.cifar100 as dc  # noqa: E402,F401

comp = {}
for seed, (clean, _) in SEEDS.items():
    parts = json.loads((Path("results_rev") / clean / "client_partition_indices.json").read_text())
    # indices are relative to the 45,000-image training subset; map through the NDSS split
    from src.data.cifar100 import split_train_validation  # noqa: E402

    sub, _ = split_train_validation("data", download=False, validation_fraction=0.1, seed=int(seed))
    labels = [int(sub.dataset.targets[i]) for i in sub.indices]
    lab = [labels[i] for i in parts[ATT]]
    counts = {c: lab.count(c) for c in set(lab)}
    comp[seed] = {"n": len(lab), "n_classes": len(counts), "class0_samples": counts.get(0, 0),
                  "top5": sorted(counts.items(), key=lambda x: -x[1])[:5]}
report["task5_attacker_class_composition"] = comp
(OUT / "reference_audit_posthoc.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
print("written", OUT)
