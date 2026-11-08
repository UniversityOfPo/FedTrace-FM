"""Protocol 37 outcome table: seeds W3-W5 (K = 5) and K = 10 at seeds 123 / 2026 (CPU, read-only).

    python -m src.rev.extension_outcomes                 # every protocol-37 run whose trace exists
    python -m src.rev.extension_outcomes --check-w1w2    # regression: reproduce unified_outcomes_confirmation.json

Frozen detectors and thresholds only (the same `unified_outcomes.detectors`); nothing is tuned or pooled. Per seed
block it reports the frozen facts F1-F5 and the "trade-off pattern" verdict of paper_plan/37 section 5. Output:
results_rev/pilot/extension/extension_outcomes.json (never overwrites development or confirmation outputs).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.analysis.fedtrace_detector import load_trace_rows
from src.rev import phase_s as ps
from src.rev.unified_outcomes import detectors

ARMS = ("clean", "badnets", "drift")
INTERVENED, ONSET = "client_002", 20
BLOCKS = {**{w: {a: f"{w}_{a}" for a in ARMS} for w in ("W3", "W4", "W5")},
          **{f"K10_d{s}": {a: f"K10_{a}_d{s}" for a in ARMS} for s in (123, 2026)}}
REFERENCE_BLOCKS = {w: {a: f"{w}_{a}" for a in ARMS} for w in ("W1", "W2")}  # protocol 26, for the regression
NORM = ("ZP-norm", "CUSUM-norm")


def _asr(root: Path, run: str, clean: str) -> float | None:
    """Final-round ASR from the pilot pair analysis (src.rev.pilot), if it has been run."""
    for d in (root / "results_rev/pilot" / f"{run}__vs__{clean}", root / "results_rev/pilot/extension_pairs" / f"{run}__vs__{clean}"):
        f = d / f"effect_{run}.csv"
        if f.exists():
            last = list(csv.DictReader(f.open(encoding="utf-8")))[-1]
            return float(last["asr"]) if last.get("asr") not in (None, "") else None
    return None


def _top1(root: Path, run: str) -> float | None:
    f = root / "results_rev" / run / "final_test_metrics.json"
    return json.loads(f.read_text())["top1"] if f.exists() else None


def evaluate_block(root: Path, block: dict[str, str]) -> dict[str, Any] | None:
    traces = {a: root / "results_rev" / r / "client_round_trace.csv" for a, r in block.items()}
    if not all(p.exists() for p in traces.values()):
        return None
    out: dict[str, Any] = {"runs": block, "arms": {}}
    clean_top1 = _top1(root, block["clean"])
    for arm, run in block.items():
        malicious = arm == "badnets"
        rows = load_trace_rows(traces[arm])
        dets = {}
        for det, alarms in detectors(rows).items():
            ev = ps.evaluate_run(alarms, INTERVENED if malicious else None, ONSET if malicious else None)
            d = {"confirmed": sorted(alarms), "first_attributed": ev["attributed"],
                 "honest_false_accusations": ev["honest_false_alarms"]}
            if malicious:
                d.update({"attacker_detected": ev["attacker_detected"], "attribution_correct": ev["attribution_correct"],
                          "onset_error": ev["onset_error"], "delay": ev["delay"]})
            dets[det] = d
        top1 = _top1(root, run)
        out["arms"][arm] = {"run": run, "clients": len({r["client_id"] for r in rows}), "final_top1": top1,
                            "top1_change_vs_clean_pp": None if arm == "clean" or top1 is None or clean_top1 is None
                            else 100 * (top1 - clean_top1),
                            "asr": _asr(root, run, block["clean"]) if arm == "badnets" else None, "detectors": dets}
    a = out["arms"]
    facts = {
        "F1_v3_attacker": {k: a["badnets"]["detectors"]["v3"][k] for k in ("attacker_detected", "attribution_correct")},
        "F2_baselines_attacker": {det: {k: a["badnets"]["detectors"][det][k] for k in ("attacker_detected", "attribution_correct")}
                                  for det in ("ZP-norm", "CUSUM-norm", "PH-norm")},
        "F3_drift_honest_confirmations": {det: a["drift"]["detectors"][det]["honest_false_accusations"] for det in a["drift"]["detectors"]},
        "F4_clean_honest_confirmations": {det: a["clean"]["detectors"][det]["honest_false_accusations"] for det in a["clean"]["detectors"]},
        "F5_effect": {"asr": a["badnets"]["asr"], "clean_top1": a["clean"]["final_top1"],
                      "badnets_top1_change_pp": a["badnets"]["top1_change_vs_clean_pp"],
                      "drift_top1_change_pp": a["drift"]["top1_change_vs_clean_pp"]},
    }
    # protocol 37 section 5 (frozen): all three conditions must hold
    c1 = any(a["badnets"]["detectors"][d]["attribution_correct"] for d in NORM)
    c2 = any(a[arm]["detectors"][d]["honest_false_accusations"] for d in NORM for arm in ("drift", "clean"))
    c3 = not a["clean"]["detectors"]["v3"]["honest_false_accusations"]
    out["facts"] = facts
    out["tradeoff_pattern"] = {"norm_baseline_attributes_attacker_first": c1,
                               "norm_baseline_accuses_honest_in_drift_or_clean": c2,
                               "v3_no_clean_alarm": c3, "present": c1 and c2 and c3}
    return out


def run_all(root: Path, blocks: dict[str, dict[str, str]]) -> dict[str, Any]:
    rep = {"status": "CLAUDE-REPORTED; protocol 37; frozen detectors and thresholds; seed-level, not pooled",
           "blocks": {}}
    for name, block in blocks.items():
        res = evaluate_block(root, block)
        rep["blocks"][name] = res if res is not None else "NOT AVAILABLE (missing trace)"
    return rep


def check_w1w2(root: Path) -> list[str]:
    """Regression: the per-run detector outcomes for W1 / W2 must equal unified_outcomes_confirmation.json."""
    saved = json.loads((root / "results_rev/pilot/unified_outcomes/unified_outcomes_confirmation.json").read_text())["rows"]
    rep = run_all(root, REFERENCE_BLOCKS)
    issues = []
    for s in saved:
        w, arm = s["trace"].split("_")
        mine = rep["blocks"][w]["arms"][arm]["detectors"][s["detector"]]
        for k in ("first_attributed", "honest_false_accusations", "onset_error", "delay"):
            if k in s and s[k] != mine.get(k):
                issues.append(f"{s['trace']} {s['detector']} {k}: saved {s[k]} vs {mine.get(k)}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Protocol 37 outcome table (read-only).")
    parser.add_argument("--check-w1w2", action="store_true")
    args = parser.parse_args(argv)
    root = Path(".")
    if args.check_w1w2:
        issues = check_w1w2(root)
        print("W1/W2 regression:", "OK (identical to unified_outcomes_confirmation.json)" if not issues else issues)
        return 0 if not issues else 1
    rep = run_all(root, BLOCKS)
    out = root / "results_rev/pilot/extension"
    out.mkdir(parents=True, exist_ok=True)
    (out / "extension_outcomes.json").write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    for name, b in rep["blocks"].items():
        print(name, b if isinstance(b, str) else f"trade-off pattern present: {b['tradeoff_pattern']['present']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
