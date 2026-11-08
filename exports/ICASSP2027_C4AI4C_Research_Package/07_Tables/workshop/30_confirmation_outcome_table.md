# 30 — Confirmation Outcome Table, seeds W1 / W2 (generated; protocol 26)

Generated 2026-10-26 by `src/rev/unified_outcomes.py` (protocol 25 §6, frozen in `164029e`); source `results_rev/pilot/unified_outcomes/unified_outcomes.json` (SHA-256 `bf3bd3a12813cb31a1ff5a723d82015bd46148c8b211444eaa125dd9cc163ba2`). Frozen detectors and thresholds; one row per trace; **no pooling**. Archived rows: CLAUDE-REPORTED from archived traces (perturbation attacks — not data poisoning). Development rows (X*): CLAUDE-REPORTED, DEVELOPMENT ONLY. Norm thresholds (ZP 1.75, CUSUM 11.75, PH 6.5) were set on the two clean K = 5 FedAvg development runs: Multi-Krum and K = 10 rows are outside that calibration regime.

Cell legend — malicious runs: ✓ = attacker confirmed first (e = onset error, d = delay in rounds); ✗ = attacker confirmed but an honest client first; missed = attacker never confirmed; FA = honest clients confirmed (client index). Clean / benign-drift runs: none = no confirmation; **accuses** = honest client(s) confirmed (in the drift runs the intervened client_002 is honest).

| Trace | Family | Seed / attacker / onset | Final top-1 | ASR | v3 | v3 magnitude-only | v3 t_res-only | v3 LOO-ref | v3 trimmed-ref | v3 historical-ref | ZP-norm | CUSUM-norm | PH-norm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `W1_badnets` | BadNets data poisoning (confirmation) | W1 / client_002 / 20 | 57.08 | 63.3 % | missed | missed | missed | missed; **FA 3** | missed; **FA 3** | missed; **FA 0,1,3,4** | ✓ e1 d3; FA 4 | ✓ e0 d6; FA 4 | missed; **FA 4** |
| `W1_clean` | clean (confirmation) | W1 / – / – | 58.04 | 0.8 % | none | none | none | none | none | **accuses 0,2,3,4** | **accuses 4** | **accuses 4** | **accuses 4** |
| `W1_drift` | benign class-composition drift (confirmation) | W1 / client_002 / 20 | 55.75 | 0.7 % | **accuses 2** | **accuses 2** | none | **accuses 2,3,4** | **accuses 2,4** | **accuses 0,1,2,3,4** | **accuses 2,4** | **accuses 2,4** | **accuses 4** |
| `W2_badnets` | BadNets data poisoning (confirmation) | W2 / client_002 / 20 | 59.36 | 44.2 % | missed | missed | missed | missed | missed | missed; **FA 1,3** | ✓ e1 d3 | ✓ e0 d6 | missed |
| `W2_clean` | clean (confirmation) | W2 / – / – | 60.22 | 0.0 % | none | none | none | none | none | **accuses 1,3** | none | none | none |
| `W2_drift` | benign class-composition drift (confirmation) | W2 / client_002 / 20 | 58.75 | 0.1 % | none | none | none | none | none | **accuses 1,3** | **accuses 2** | **accuses 2** | none |
