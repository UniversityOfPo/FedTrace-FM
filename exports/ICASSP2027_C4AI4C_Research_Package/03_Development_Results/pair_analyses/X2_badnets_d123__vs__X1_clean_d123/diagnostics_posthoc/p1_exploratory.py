"""POST-HOC EXPLORATORY (CPU, read-only): established onset estimators and change detectors vs frozen v3.

Pre-specified choices (fixed before looking at results of this script, from textbook defaults):
  * CUSUM reference k = 0.5 (detects a 1-sd mean shift), Page-Hinkley drift delta = 0.5 (same scale);
  * alarm thresholds h / lambda: smallest grid value with ZERO client alarms on the clean development run
    X1_clean_d123 (all 5 clients, rounds >= T_ref); grid 0.25 ... 60 step 0.25;
  * onset-at-alarm estimator (CUSUM change time on the v3 score S): kappa0 in {0.5, 1, 1.5, 2} -- ALL reported;
  * features: standardized update norm (two-sided) and the temporal residual d = c(r) - t (two-sided), each
    standardized with the client's own calibration rounds < T_ref (causal from T_ref on).
Frozen v3 outputs are recomputed unchanged and reported first.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.analysis.fedtrace_detector import compute_common_round_reference, load_trace_rows  # noqa: E402
from src.rev.baselines import cusum, page_hinkley, smallest_threshold_without_client_alarms  # noqa: E402
from src.rev.online import OnlineConfig, causal_scores_v3, online_detect_v3  # noqa: E402

T_REF, K, DELTA = 20, 0.5, 0.5
GRID = [x * 0.25 for x in range(1, 241)]
KAPPAS = (0.5, 1.0, 1.5, 2.0)
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
cfg = OnlineConfig()


def features(rows):
    clients = sorted({r["client_id"] for r in rows})
    refs = {r: compute_common_round_reference(rows, r, "self_prev_round_cosine") for r in sorted({x["round_id"] for x in rows})}
    norm = {c: {} for c in clients}
    resid = {c: {} for c in clients}
    for r in rows:
        c, rnd = r["client_id"], r["round_id"]
        norm[c][rnd] = r["update_l2_norm"]
        t = r.get("self_prev_round_cosine")
        resid[c][rnd] = None if (t is None or refs[rnd] is None) else refs[rnd] - t
    return clients, norm, resid


def standardized(series):
    import statistics as st
    cal = [v for r, v in series.items() if r < T_REF and v is not None]
    mu, sd = st.mean(cal), st.stdev(cal)
    return {r: (v - mu) / sd for r, v in series.items() if r >= T_REF and v is not None}


def two_sided(detector, series, threshold):
    """Run a one-sided detector on +series and -series; earliest alarm wins (ties: upper side)."""
    up = detector(series, threshold)
    down = detector({r: -v for r, v in series.items()}, threshold)
    cands = [e for e in (up, down) if e.alarm_round is not None]
    return min(cands, key=lambda e: e.alarm_round) if cands else up


DETECTORS = {
    "CUSUM_norm": ("norm", lambda s, h: cusum(s, t_ref=T_REF, k=K, h=h, standardize=False)),
    "CUSUM_residual": ("resid", lambda s, h: cusum(s, t_ref=T_REF, k=K, h=h, standardize=False)),
    "PH_norm": ("norm", lambda s, h: page_hinkley(s, t_ref=T_REF, delta=DELTA, lam=h)),
    "PH_residual": ("resid", lambda s, h: page_hinkley(s, t_ref=T_REF, delta=DELTA, lam=h)),
}


def per_client_series(rows, kind):
    clients, norm, resid = features(rows)
    src = norm if kind == "norm" else resid
    return {c: standardized(src[c]) for c in clients}


def run_detector(rows, name, h):
    kind, det = DETECTORS[name]
    events = {c: two_sided(det, s, h) for c, s in per_client_series(rows, kind).items()}
    alarmed = sorted((e.alarm_round, c) for c, e in events.items() if e.alarm_round is not None)
    return {"attributed": alarmed[0][1] if alarmed else None, "alarm": alarmed[0][0] if alarmed else None,
            "onset": events[alarmed[0][1]].estimated_onset if alarmed else None,
            "alarmed_clients": {c: (e.alarm_round, e.estimated_onset) for c, e in events.items() if e.alarm_round is not None}}


def change_time(scores, alarm, kappa):
    """CUSUM change time on the v3 score S at the alarm round (causal: rounds T_ref..alarm only)."""
    c, last_zero = 0.0, T_REF - 1
    for r in range(T_REF, alarm + 1):
        c = max(0.0, c + (scores.get(r) or 0.0) - kappa)
        if c == 0.0 and r < alarm:
            last_zero = r
    return last_zero + 1


report = {"note": "POST-HOC EXPLORATORY; development data only; not evidence of generalisation"}
clean = load_trace_rows(Path("results_rev/X1_clean_d123/client_round_trace.csv"))
attacked = load_trace_rows(Path("results_rev/X2_badnets_d123/client_round_trace.csv"))

# 1. frozen v3, unchanged
for name, rows in (("X1_clean_d123", clean), ("X2_badnets_d123", attacked)):
    det = online_detect_v3(rows, cfg)
    report[f"v3_{name}"] = {"attributed": det.attributed_client, "alarm": det.attributed_alarm_round,
                            "onset": det.per_client[det.attributed_client].estimated_onset if det.attributed_client else None}

# 2. established detectors with thresholds from the clean development run only
for name in DETECTORS:
    kind, det = DETECTORS[name]
    clean_series = per_client_series(clean, kind)
    h = smallest_threshold_without_client_alarms([clean_series], GRID, lambda s, thr: two_sided(det, s, thr))
    report[name] = {"threshold_from_X1_clean": h,
                    "X2": run_detector(attacked, name, h) if h is not None else "no zero-alarm threshold in grid"}

# 3. onset-at-alarm estimators on the v3 alarm (X2 + archived development traces with a single known onset)
TRACES = {
    "X2_badnets_d123": ("results_rev/X2_badnets_d123", "client_002", 20),
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
onsets = {}
for name, (path, attacker, true_onset) in TRACES.items():
    rows = load_trace_rows(Path(path) / "client_round_trace.csv")
    det = online_detect_v3(rows, cfg)
    ev = det.per_client[attacker]
    scores = {}
    for s in causal_scores_v3(rows, cfg):
        if s.client_id == attacker:
            comb = s.combined_score(cfg.components)
            scores[s.round_id] = comb[0] if comb else None
    rec = {"true_onset": true_onset, "v3_alarm": ev.alarm_round, "v3_streak_onset": ev.estimated_onset}
    if ev.alarm_round is not None:
        for kappa in KAPPAS:
            rec[f"cusum_change_time_k{kappa}"] = change_time(scores, ev.alarm_round, kappa)
    onsets[name] = rec
report["onset_at_v3_alarm"] = onsets
(OUT / "exploratory_posthoc.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")

print(json.dumps({k: v for k, v in report.items() if k != "onset_at_v3_alarm"}, indent=1, default=str))
print("onset at v3 alarm: name | true | v3 alarm | v3 streak onset | change time kappa0=0.5,1,1.5,2")
for name, r in onsets.items():
    print(name, r["true_onset"], r["v3_alarm"], r["v3_streak_onset"], [r.get(f"cusum_change_time_k{k}") for k in KAPPAS])
