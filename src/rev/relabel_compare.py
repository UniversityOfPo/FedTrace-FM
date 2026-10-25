"""Clean vs BadNets vs relabel-only comparison (protocol 09 §3 and its 2026-10-25 amendment). Read-only, CPU.

    python -m src.rev.relabel_compare --seeds 123 2026

DEVELOPMENT ONLY. Reads the sealed traces of X1_clean_d<seed>, X2_badnets_d<seed>, X3_relabel_d<seed> and the pair
analyses; writes ``results_rev/pilot/relabel_comparison/``. The clean run is the matched counterfactual and is used
for diagnosis only. The ratio rho is descriptive (amendment B), never causal proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.analysis.fedtrace_detector import load_trace_rows
from src.rev.diagnostics import norm_ratios, signed_residuals
from src.rev.reference_audit import decompose

ATTACKER, ONSET = "client_002", 20
SELF_COS_ROUNDS = range(21, 30)        # protocol 09 §3 endpoint 1
NORM_ROUNDS = range(25, 36)            # endpoint 3
REFERENCE_ROUNDS = range(20, 30)       # endpoint 4
EPSILON = 0.02                         # amendment: near-zero threshold for effects (5x honest self-cosine noise)
LOW, HIGH = 0.25, 0.75                 # original preregistered thresholds (unchanged)


def self_cos_change(clean: Sequence[Mapping[str, Any]], arm: Sequence[Mapping[str, Any]], client: str = ATTACKER,
                    rounds=SELF_COS_ROUNDS) -> float | None:
    c = {r["round_id"]: r.get("self_prev_round_cosine") for r in clean if r["client_id"] == client}
    a = {r["round_id"]: r.get("self_prev_round_cosine") for r in arm if r["client_id"] == client}
    diffs = [a[r] - c[r] for r in rounds if a.get(r) is not None and c.get(r) is not None]
    return statistics.mean(diffs) if len(diffs) == len(rounds) else None


def norm_effect(clean, arm, client: str = ATTACKER, rounds=NORM_ROUNDS) -> float | None:
    ratios = norm_ratios(clean, arm).get(client, {})
    vals = [ratios[r] for r in rounds if r in ratios]
    return 1.0 - statistics.mean(vals) if len(vals) == len(rounds) else None


def signed_residual_change(clean, arm, client: str = ATTACKER, rounds=SELF_COS_ROUNDS) -> float | None:
    zc, za = signed_residuals(clean).get(client, {}), signed_residuals(arm).get(client, {})
    diffs = [za[r]["signed_z"] - zc[r]["signed_z"] for r in rounds
             if r in za and r in zc and za[r]["signed_z"] is not None and zc[r]["signed_z"] is not None]
    return statistics.mean(diffs) if len(diffs) == len(rounds) else None


def classify_rho(relabel: float | None, badnets: float | None, eps: float = EPSILON) -> dict[str, Any]:
    """Per-seed rho with the amendment's handling of undefined, near-zero and opposite-signed effects."""
    if relabel is None or badnets is None:
        return {"rho": None, "class": "UNDEFINED", "reason": "missing measurement"}
    if abs(badnets) < eps:
        return {"rho": None, "class": "UNDEFINED", "reason": f"|BadNets effect| < {eps} (near-zero denominator)"}
    rho = relabel / badnets
    if abs(relabel) >= eps and (relabel > 0) != (badnets > 0):
        return {"rho": rho, "class": "LOW", "flag": "OPPOSITE_SIGN",
                "reason": "relabel-only effect opposite in direction to BadNets (counted as <= 0.25 by the original rule)"}
    if abs(relabel) < eps:
        return {"rho": rho, "class": "LOW", "flag": "NO_DETECTABLE_RELABEL_EFFECT", "reason": f"|relabel effect| < {eps}"}
    cls = "HIGH" if rho >= HIGH else ("LOW" if rho <= LOW else "MID")
    return {"rho": rho, "class": cls}


def overall(classes: Sequence[str]) -> str:
    """Original preregistered cross-seed classification (unchanged)."""
    if any(c == "UNDEFINED" for c in classes):
        return "UNDEFINED"
    if all(c == "HIGH" for c in classes):
        return "relabelling sufficient"
    if all(c == "LOW" for c in classes):
        return "trigger-specific"
    if all(c == "MID" for c in classes):
        return "mixed"
    return "inconsistent across seeds"


CAUTIOUS = {
    "relabelling sufficient": "relabel-only reproduces most of the BadNets change; consistent with relabelling being sufficient (descriptive, not causal proof)",
    "trigger-specific": "relabel-only does not reproduce the BadNets change; consistent with a trigger-specific component (descriptive, not causal proof)",
    "mixed": "relabel-only reproduces part of the change in both seeds (descriptive)",
    "inconsistent across seeds": "the two development seeds disagree; no conclusion",
    "UNDEFINED": "ratio undefined for at least one seed; no classification",
}


def _final_effect(pair_dir: Path, run: str) -> dict[str, float] | None:
    path = pair_dir / f"effect_{run}.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    last = rows[-1]
    return {k: float(last[k]) for k in ("clean_top1", "asr", "untriggered_target_rate") if last.get(k) not in (None, "")}


def compare(root: Path, seeds: Sequence[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "CLAUDE-REPORTED, DEVELOPMENT ONLY; rho descriptive (protocol 09 amendment)",
                              "epsilon": EPSILON, "thresholds": [LOW, HIGH], "seeds": {}}
    classes: dict[str, list[str]] = {"self_cos": [], "norm": []}
    for seed in seeds:
        runs = {"clean": f"X1_clean_d{seed}", "badnets": f"X2_badnets_d{seed}", "relabel": f"X3_relabel_d{seed}"}
        rows = {k: load_trace_rows(root / "results_rev" / v / "client_round_trace.csv") for k, v in runs.items()}
        pairs = {"badnets": root / "results_rev" / "pilot" / "reanalysis_target_rate" / f"{runs['badnets']}__vs__{runs['clean']}",
                 "relabel": root / "results_rev" / "pilot" / f"{runs['relabel']}__vs__{runs['clean']}"}
        arms: dict[str, Any] = {}
        for arm in ("badnets", "relabel"):
            dec = decompose(rows["clean"], rows[arm], ATTACKER, REFERENCE_ROUNDS)
            outcome = json.loads((pairs[arm] / f"outcome_{runs[arm]}.json").read_text(encoding="utf-8"))["outcome"]
            arms[arm] = {
                "self_cos_change": self_cos_change(rows["clean"], rows[arm]),
                "signed_residual_change": signed_residual_change(rows["clean"], rows[arm]),
                "norm_effect": norm_effect(rows["clean"], rows[arm]),
                "attacker_induced_reference_movement": {k: statistics.mean(r["attacker_induced"] for r in v)
                                                        for k, v in dec.items() if k != ATTACKER and v},
                "v3": {k: outcome[k] for k in ("detected", "attribution_correct", "attributed_client", "alarm_round",
                                               "estimated_onset", "onset_abs_error", "delay", "honest_alarm_clients")},
                "final": _final_effect(pairs[arm], runs[arm]),
                "prefix_valid": json.loads((pairs[arm] / "prefix.json").read_text(encoding="utf-8"))["valid_matched_pair"],
            }
        arms["clean_final"] = _final_effect(pairs["relabel"], runs["clean"])
        rho_t = classify_rho(arms["relabel"]["self_cos_change"], arms["badnets"]["self_cos_change"])
        rho_n = classify_rho(arms["relabel"]["norm_effect"], arms["badnets"]["norm_effect"])
        classes["self_cos"].append(rho_t["class"])
        classes["norm"].append(rho_n["class"])
        report["seeds"][seed] = {"arms": arms, "rho_self_cos": rho_t, "rho_norm": rho_n}
    for key in ("self_cos", "norm"):
        verdict = overall(classes[key])
        report[f"overall_{key}"] = {"preregistered_class": verdict, "cautious_interpretation": CAUTIOUS[verdict]}
    out = root / "results_rev" / "pilot" / "relabel_comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "relabel_comparison.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean vs BadNets vs relabel-only comparison (read-only).")
    parser.add_argument("--seeds", nargs="+", default=["123", "2026"])
    args = parser.parse_args(argv)
    report = compare(Path("."), args.seeds)
    print(json.dumps({k: v for k, v in report.items() if k.startswith("overall")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
