"""Unified, evidence-linked outcome table over all compatible traces (protocol 25 §6; CPU, read-only).

    python -m src.rev.unified_outcomes

CLAUDE-REPORTED. Frozen detectors with frozen thresholds only (no tuning): FedTrace v3 (inclusive median; magnitude +
temporal residual), its magnitude-only and temporal-residual-only ablations, v3 with leave-one-out / trimmed-mean /
client-historical references, and the Phase S norm detectors (two-sided z-score + persistence h = 1.75, CUSUM
h = 11.75, Page-Hinkley h = 6.5; thresholds from the two clean development runs, protocol 12). One row per trace and
detector; nothing is pooled across attack families, aggregation rules or client counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.analysis.fedtrace_detector import load_trace_rows
from src.rev import phase_s as ps
from src.rev.contamination import ARCHIVE, CONF, DEV, RUNS, truth_of, v3_alarms
from src.rev.reference_audit import REFERENCES, residuals

Q = {"ZP-norm": ("ZP", 1.75), "CUSUM-norm": ("CUSUM", 11.75), "PH-norm": ("PH", 6.5)}


def family(name: str, rows) -> str:
    if name in RUNS:
        tag = " (confirmation)" if name in CONF else ""
        return {"clean": "clean (dev)" if name in DEV else "clean (confirmation)", "badnets": "BadNets data poisoning", "relabel": "relabel-only data poisoning",
                "drift": "benign class-composition drift"}[RUNS[name][0]] + (tag if RUNS[name][0] != "clean" else "")
    att = [r for r in rows if r.get("attack_configured")]
    agg = "Multi-Krum" if "multikrum" in name else "FedAvg"
    k = len({r["client_id"] for r in rows})
    kind = "clean" if not att else {"gaussian_update_noise": "Gaussian update perturbation",
                                    "norm_preserving_directional": "directional update perturbation"}.get(att[0]["attack_type"], att[0]["attack_type"])
    if "E3" in name:
        kind += " (intermittent)"
    if "E2" in name:
        kind += " (reduced severity)"
    return f"{kind} | {agg} | K={k}"


def detectors(rows) -> dict[str, dict]:
    res_inc = residuals(rows, REFERENCES["inclusive_median"])
    out = {"v3": v3_alarms(rows, res_inc), "v3 magnitude-only": v3_alarms(rows, res_inc, ("m",)),
           "v3 t_res-only": v3_alarms(rows, res_inc, ("t_res",))}
    for ref, label in (("leave_one_out_median", "v3 LOO-ref"), ("trimmed_mean", "v3 trimmed-ref"), ("historical_own_mean", "v3 historical-ref")):
        out[label] = ps.v_alarms(rows, ref)[0]
    for label, (det, h) in Q.items():
        out[label] = ps.q_alarms(rows, det, "m", h)[0]
    return out


def effect(root: Path, name: str, path: Path) -> dict[str, Any]:
    fm = path.parent / "final_test_metrics.json"
    top1 = json.loads(fm.read_text())["top1"] if fm.exists() else None
    asr = None
    if name in RUNS:
        m = root / ("results_rev/pilot/evidence_matrix_confirmation/evidence_matrix.json" if name in CONF
                    else "results_rev/pilot/evidence_matrix/evidence_matrix.json")
        if m.exists():
            cond, s = RUNS[name]
            row = json.loads(m.read_text())["rows"].get(f"{s}/{cond}")
            asr = (row.get("final_effect") or {}).get("asr") if row else None
    return {"final_top1": top1, "asr": asr}


def run_all(root: Path, confirmation: bool = False) -> dict[str, Any]:
    if confirmation:
        traces = {n: root / f"results_rev/{n}/client_round_trace.csv" for n in CONF}
    else:
        traces = {Path(p).parent.parent.name[:12] + "/" + Path(p).parent.name: Path(p) for p in ARCHIVE}
        traces.update({n: root / f"results_rev/{n}/client_round_trace.csv" for n in DEV})
    table = []
    for name, path in traces.items():
        rows = load_trace_rows(path)
        src, onset, malicious = truth_of(rows, name)
        eff = effect(root, name, path)
        for det, alarms in detectors(rows).items():
            ev = ps.evaluate_run(alarms, src if malicious else None, onset if malicious else None)
            row = {"trace": name, "family": family(name, rows), "intervened": src, "malicious": malicious, "onset": onset,
                   "detector": det, "any_confirmation": bool(alarms), "first_attributed": ev["attributed"],
                   "honest_false_accusations": ev["honest_false_alarms"], "attack_effect": eff}
            if malicious:
                row.update({"attacker_detected": ev["attacker_detected"], "attribution_correct": ev["attribution_correct"],
                            "onset_error": ev["onset_error"], "delay": ev["delay"]})
            table.append(row)
    out = root / "results_rev/pilot/unified_outcomes"
    out.mkdir(parents=True, exist_ok=True)
    rep = {"status": "CLAUDE-REPORTED (archived traces) / DEVELOPMENT ONLY (rev runs); frozen detectors and thresholds",
           "rows": table}
    (out / ("unified_outcomes_confirmation.json" if confirmation else "unified_outcomes.json")).write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    return rep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified outcome table (read-only).")
    parser.add_argument("--confirmation", action="store_true", help="protocol 26 runs W1 / W2 only (separate output)")
    args = parser.parse_args(argv)
    rep = run_all(Path("."), args.confirmation)
    print("rows", len(rep["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
