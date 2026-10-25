"""Preregistered evaluation of the benign-drift control (``paper_plan/13_benign_drift_preregistration.md``).

    python -m src.rev.benign_drift --seeds 123 2026

Read-only, CPU, DEVELOPMENT ONLY. Compares X4_drift_d<seed> with its matched clean run X1_clean_d<seed>. There is
NO malicious client: every confirmed client is an honest false accusation. Detector definitions and thresholds are
those of Phase S (protocol 12); the two group-Q thresholds below are the values recorded by Phase S and are fixed
here before any drift outcome exists -- they are never recalibrated on drift runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.analysis.fedtrace_detector import load_trace_rows
from src.rev import phase_s as ps
from src.rev.diagnostics import norm_ratios
from src.rev.online import online_detect_v3
from src.rev.reference_audit import decompose, inclusive_median
from src.rev.relabel_compare import self_cos_change

DRIFTED, ONSET = "client_002", 20
FIXED_THRESHOLDS = {"Q-ZP-m": 1.75, "Q-CUSUM-m": 11.75}   # Phase S (L-56), retrospective clean-development calibration
POISONING_NORM_RATIO_RANGE = (0.79, 0.87)                 # attacker norm ratios observed under BadNets / relabel (L-50)
POISONING_SELF_COS_MIN = 0.075                            # smallest attacker self-cosine rise observed (relabel, seed 123)


def norm_class(ratio: float) -> str:
    if ratio <= 0.90:
        return "DECREASE (<= 0.90; comparable to poisoning 0.79-0.87)"
    if ratio >= 1.10:
        return "INCREASE (>= 1.10)"
    return "NO MATERIAL CHANGE (0.90-1.10)"


def evaluate_seed(root: Path, seed: str) -> dict[str, Any]:
    clean = load_trace_rows(root / "results_rev" / f"X1_clean_d{seed}" / "client_round_trace.csv")
    drift = load_trace_rows(root / "results_rev" / f"X4_drift_d{seed}" / "client_round_trace.csv")
    out: dict[str, Any] = {}
    det = online_detect_v3(drift)
    out["E1_v3_confirmed_clients"] = sorted(c for c, e in det.per_client.items() if e.alarm_round is not None)
    out["E1_v3_alarms"] = {c: e.alarm_round for c, e in det.per_client.items() if e.alarm_round is not None}
    for name, key in (("E2_zp_norm", "Q-ZP-m"), ("E3_cusum_norm", "Q-CUSUM-m")):
        detector = "ZP" if key == "Q-ZP-m" else "CUSUM"
        alarms, undefined = ps.q_alarms(drift, detector, "m", FIXED_THRESHOLDS[key])
        clean_alarms, _ = ps.q_alarms(clean, detector, "m", FIXED_THRESHOLDS[key])
        out[name] = {"threshold": FIXED_THRESHOLDS[key], "confirmed_clients": sorted(alarms),
                     "alarm_rounds": {c: a[0] for c, a in alarms.items()}, "undefined": undefined,
                     "clean_control_confirmed": sorted(clean_alarms)}
    ratios = norm_ratios(clean, drift)
    window = range(25, 36)
    mean_ratio = {c: statistics.mean(ratios[c][r] for r in window) for c in ratios}
    out["E4_norm_ratio_r25_35"] = {"per_client": mean_ratio, "drifted_client_class": norm_class(mean_ratio[DRIFTED])}
    dt = self_cos_change(clean, drift, DRIFTED)
    out["E5_self_cos_change_r21_29"] = {"drifted_client": dt,
                                        "comparable_to_poisoning": dt is not None and dt >= POISONING_SELF_COS_MIN}
    dec = decompose(clean, drift, DRIFTED, range(ONSET, ONSET + 10), inclusive_median)
    out["E6_reference_movement_from_drifted_client"] = {k: statistics.mean(r["attacker_induced"] for r in v)
                                                         for k, v in dec.items() if k != DRIFTED and v}
    pair = root / "results_rev" / "pilot" / f"X4_drift_d{seed}__vs__X1_clean_d{seed}"
    summary = json.loads((pair / "summary.json").read_text(encoding="utf-8")) if (pair / "summary.json").exists() else {}
    out["E7_final_top1"] = {"clean": summary.get(f"final_test_top1_X1_clean_d{seed}"),
                            "drift": summary.get(f"final_test_top1_X4_drift_d{seed}")}
    out["E8_any_honest_accusation"] = {"v3": bool(out["E1_v3_confirmed_clients"]),
                                       "zp_norm": bool(out["E2_zp_norm"]["confirmed_clients"]),
                                       "cusum_norm": bool(out["E3_cusum_norm"]["confirmed_clients"])}
    out["prefix_valid"] = json.loads((pair / "prefix.json").read_text(encoding="utf-8"))["valid_matched_pair"] \
        if (pair / "prefix.json").exists() else None
    return out


def verdicts(per_seed: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Preregistered H-BEN verdict per detector: FALSIFIED if any client is confirmed in any drift run."""
    res = {}
    for name in ("v3", "zp_norm", "cusum_norm"):
        accused = [s for s, v in per_seed.items() if v["E8_any_honest_accusation"][name]]
        res[name] = ("FALSIFIED (honest accusation in seed(s) " + ", ".join(accused) + ")") if accused else \
            "NOT FALSIFIED on these development runs (no confirmed client; not a false-positive guarantee)"
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benign-drift control evaluation (read-only).")
    parser.add_argument("--seeds", nargs="+", default=["123", "2026"])
    args = parser.parse_args(argv)
    root = Path(".")
    per_seed = {s: evaluate_seed(root, s) for s in args.seeds}
    report = {"status": "CLAUDE-REPORTED, DEVELOPMENT ONLY (protocol 13)", "fixed_thresholds": FIXED_THRESHOLDS,
              "per_seed": per_seed, "H-BEN": verdicts(per_seed)}
    out = root / "results_rev" / "pilot" / "benign_drift"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benign_drift.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["H-BEN"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
