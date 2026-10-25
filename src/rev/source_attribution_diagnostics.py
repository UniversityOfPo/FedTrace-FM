"""Post-hoc mechanism checks for protocol 14 (D1), computed after outcomes (L-74). Read-only, CPU.

    python -m src.rev.source_attribution_diagnostics --seeds 123 2026

CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY. Separate from the frozen implementation
(`source_attribution.py`, hash recorded in protocol 14), which it only imports.

``leakage`` splits each candidate's influence I_j (rounds 20..29, MED) into the part contributed by one named
client's own term z_k(C)^2 - z_k(C \\ {j})^2 and the part from the remaining clients. The named client is supplied
by the analyst (evaluation only); it is not an input of the ranking.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.analysis.fedtrace_detector import load_trace_rows
from src.rev import source_attribution as sa


def leakage(rows: Sequence[Mapping[str, Any]], named: str, rounds=range(20, 30), estimator: str = "MED") -> dict[str, dict[str, float]]:
    t = sa.series(rows, "self_prev_round_cosine")
    res = sa._Residuals(t, estimator)
    out = {}
    for j in sorted(t[rounds[0]]):
        from_named, rest = [], []
        for r in rounds:
            terms = {}
            for k in t[r]:
                if k == j:
                    continue
                a, b = res.z(r, k, "full"), res.z(r, k, ("minus", j))
                if a is not None and b is not None:
                    terms[k] = a * a - b * b
            if len(terms) < 2:
                continue
            from_named.append(terms.get(named, 0.0) / len(terms))
            rest.append(sum(v for k, v in terms.items() if k != named) / len(terms))
        out[j] = {"I": statistics.mean(a + b for a, b in zip(from_named, rest)),
                  "from_named_client_term": statistics.mean(from_named), "from_other_terms": statistics.mean(rest)}
    return out


def run(root: Path, seeds: Sequence[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY (protocol 14, L-74)", "runs": {}}
    for s in seeds:
        for cond, pattern in sa.CONDITIONS.items():
            rows = load_trace_rows(root / "results_rev" / pattern.format(s=s) / "client_round_trace.csv")
            t = sa.series(rows, "self_prev_round_cosine")
            res = sa._Residuals(t, "MED")
            report["runs"][f"{s}/{cond}"] = {
                "leakage_of_client_002_term": leakage(rows, "client_002"),
                "z_client_002_inclusive_r20_29": [res.z(r, "client_002", "full") for r in range(20, 30)],
                "positions_r20_29": {r: sa.positions(t, r) for r in range(20, 30)},
            }
    out = root / "results_rev" / "pilot" / "source_attribution"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mechanism_checks.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D1 post-hoc mechanism checks (read-only).")
    parser.add_argument("--seeds", nargs="+", default=["123", "2026"])
    args = parser.parse_args(argv)
    run(Path("."), args.seeds)
    print("written results_rev/pilot/source_attribution/mechanism_checks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
