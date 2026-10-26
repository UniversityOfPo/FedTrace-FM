import json, hashlib, sys
from pathlib import Path
src = Path("results_rev/pilot/unified_outcomes/unified_outcomes.json")
R = json.loads(src.read_text())
sha = hashlib.sha256(src.read_bytes()).hexdigest()
DET = ["v3", "v3 magnitude-only", "v3 t_res-only", "v3 LOO-ref", "v3 trimmed-ref", "v3 historical-ref", "ZP-norm", "CUSUM-norm", "PH-norm"]
by = {}
for r in R["rows"]:
    by.setdefault(r["trace"], {})[r["detector"]] = r
def cell(r):
    fa = r["honest_false_accusations"]
    fa_s = ("; FA " + ",".join(c[-1] for c in fa)) if fa else ""
    if not r["malicious"]:
        return ("none" if not r["any_confirmation"] else "**accuses " + ",".join(c[-1] for c in fa) + "**")
    if r["attribution_correct"]:
        return f"✓ e{r['onset_error']} d{r['delay']}{fa_s}"
    if r["attacker_detected"]:
        return f"✗ first {r['first_attributed'][-1]} (attacker d{r['delay']}){fa_s}"
    return ("missed" + (f"; **FA {','.join(c[-1] for c in fa)}**" if fa else ""))
lines = ["# 28 — Unified Outcome Table (generated; evidence-linked)", "",
         f"Generated 2026-10-26 by `src/rev/unified_outcomes.py` (protocol 25 §6, frozen in `164029e`); source "
         f"`results_rev/pilot/unified_outcomes/unified_outcomes.json` (SHA-256 `{sha}`). Frozen detectors and thresholds; "
         "one row per trace; **no pooling**. Archived rows: CLAUDE-REPORTED from archived traces (perturbation attacks — "
         "not data poisoning). Development rows (X*): CLAUDE-REPORTED, DEVELOPMENT ONLY. Norm thresholds (ZP 1.75, "
         "CUSUM 11.75, PH 6.5) were set on the two clean K = 5 FedAvg development runs: Multi-Krum and K = 10 rows are "
         "outside that calibration regime.", "",
         "Cell legend — malicious runs: ✓ = attacker confirmed first (e = onset error, d = delay in rounds); ✗ = attacker "
         "confirmed but an honest client first; missed = attacker never confirmed; FA = honest clients confirmed (client "
         "index). Clean / benign-drift runs: none = no confirmation; **accuses** = honest client(s) confirmed (in the drift "
         "runs the intervened client_002 is honest).", "",
         "| Trace | Family | Seed / attacker / onset | Final top-1 | ASR | " + " | ".join(DET) + " |",
         "|" + "---|" * (5 + len(DET))]
order = sorted(by, key=lambda t: (t.startswith("X"), t))
for t in order:
    d = by[t]; r0 = d["v3"]
    seed = next((s for s in ("123", "2026", "42") if s in t), "?")
    top1 = r0["attack_effect"]["final_top1"]; asr = r0["attack_effect"]["asr"]
    lines.append(f"| `{t}` | {r0['family'].replace(' | ', ', ')} | {seed} / {r0['intervened'] or '–'} / {r0['onset'] or '–'} | "
                 f"{'' if top1 is None else f'{100*top1:.2f}'} | {'' if asr is None else f'{100*asr:.1f} %'} | "
                 + " | ".join(cell(d[x]) for x in DET) + " |")
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("rows", len(order))
