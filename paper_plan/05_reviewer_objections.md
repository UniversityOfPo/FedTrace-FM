# 05 — Reviewer Objection Map (NDSS 2027, submission #368)

Status: living document, created 2026-10-24. Both reviews recommended Reject. Source: the review texts as pasted
by the PI (#368A, #368B); paraphrased here — keep the original texts with the PI's records and quote them exactly
in any response letter.

Columns: objection → what we will do → evidence / experiment → current status → residual risk.

## Review #368A

| # | Objection (paraphrase) | Response plan | Evidence / experiment | Status | Residual risk |
|---|---|---|---|---|---|
| A1 | Evaluation is small (few seeds, one setting) | Runs as statistical unit; 5 fresh pre-registered validation seeds per main condition, CP intervals, per-seed appendix; dev/validation separation | X1–X3 (V), T3, A4 | PLANNED | 5 seeds still small (5/5 → CI lower 0.48); state honestly; X8 if GPU allows |
| A2 | Attacks are unrealistic (noise / geometry) | Reclassify Gaussian/directional as Class A stress tests; add Class B poisoning with measured harmful objective (backdoor success, paired clean accuracy) | X2 (+X7), T1, T3 | PLANNED | Backdoor effectiveness at K=5, α=0.1 unknown until dev runs |
| A3 | Attacks are not adaptive | Class C detector-aware norm-constrained backdoor (white-box detector knowledge, own history only), measured success vs stealth trade-off | X3, T5, F3 | PLANNED (prototype implemented) | Only z_m constrained; t_res not (needs peer information) — state the limitation; an omniscient attacker is future work |
| A4 | Strong assumption: clean calibration window | Calibration-contamination experiment + analytic masking bound (what calibration cannot see) | X4, A8, L-14 | PLANNED (SHOULD) | If detection collapses, report as limitation; do not tune |
| A5 | Assumes full participation | Partial participation K=10, 5 per round with frozen v3 | X5, T7 | PLANNED (SHOULD) | Frozen persistence breaks on unselected rounds; expected degradation is a finding, not a bug |
| A6 | Single attacker only | Two colluding backdoor clients if engineering ≤ 0.5 day; otherwise explicit limitation | X6 | CONDITIONAL | May not fit in 7 days |
| A7 | Limited technical novelty | Reposition contributions: forensic formulation + causal event semantics, calibration-masking analysis, evaluation methodology, robust-aggregation ≠ attribution; include textbook change detectors as baselines to show what is and is not new | blueprint §4–5, C1, C2 | PLANNED | High: core detector is a calibrated z-score with persistence |
| A8 | Weak baseline (only Multi-Krum); compare with FedREDefense and FLForensics | Sequential detectors at matched FPR (CPU); FedREDefense/FLForensics only after reading the papers/code; otherwise qualitative discussion and explicit "not compared" | C1, C4, T4 | PARTIAL (CUSUM/PH/z-score implemented; published methods UNVERIFIED) | High if published baselines cannot be run |

## Review #368B

| # | Objection (paraphrase) | Response plan | Evidence | Status | Residual risk |
|---|---|---|---|---|---|
| B1 | Threat model unclear | Dedicated Threat Model section and table: defender view, attacker capability/knowledge/objective, timing, out-of-scope | blueprint §2, T1 | READY TO WRITE | — |
| B2 | No comparison with prior work [10] | Identify [10] from the submitted PDF's bibliography (draft has 8 references; submitted version may differ), then compare or discuss | — | **BLOCKED: [10] unidentified** | Do not guess |
| B3 | Narrow, non-security setup | Security objective (backdoor attribution) as the main condition; state single-dataset limitation; second dataset only with download approval | X2, X9 | PARTIAL | Single dataset remains |
| B4 | Typo p.2 "Post-hoc forensic attribution.:" | fix | manuscript | TODO | — |
| B5 | RQ labels inconsistent | fixed RQ1–RQ6 list used everywhere | blueprint §3 | TODO (manuscript) | — |
| B6 | Move §III.H (Experimental Protocol and Implementation Details) to §IV.A | new section order (blueprint §8) | manuscript | TODO | — |

## Anticipated CVPR 2027 reviewer expectations (venue changed, D-018)

The NDSS objections above remain valid; CVPR reviewers will additionally probe:

| # | Expected objection | Response plan | Status |
|---|---|---|---|
| V1 | One dataset / one backbone is not enough for CVPR | add at least one more vision dataset (and ideally a second backbone) under the same protocol; needs download + licence approval (P-16) and GPU budget | BLOCKED on P-16 |
| V2 | Contribution is statistical monitoring, not vision methodology | frame as heterogeneity-aware evidence for federated *vision* training; candidates A/B/C each evaluated by single-swap ablation; claim only adopted candidates | PLANNED (blueprint §0) |
| V3 | Backdoor attacks must be state-of-the-art for vision | BadNets first (pilot); stronger vision backdoors only after their papers/code are reviewed | PLANNED |
| V4 | Comparison with published FL backdoor defences/forensics | C4 (sources needed, P-07); otherwise explicit "not compared" | BLOCKED on sources |
| V5 | Benign drift vs attack is not separated convincingly | primary endpoints: clean non-IID specificity + BadNets sensitivity, matched pairs, runs as units | PLANNED |

## Cross-cutting topics requested by the PI

| Topic | Where addressed | Status |
|---|---|---|
| Realistic poisoning | A2 → X2, X7 | PLANNED |
| Adaptive attackers | A3 → X3 | PLANNED |
| Technical novelty | A7 | PLANNED (risk high) |
| FLDetector and comparable forensic baselines | A8, C4 — FLDetector has no bibliography entry; mechanics UNVERIFIED | BLOCKED on sources |
| Calibration contamination | A4 → X4, A8 bound | PLANNED |
| Partial participation | A5 → X5 | PLANNED |
| Multiple attackers | A6 → X6 | CONDITIONAL |
| Security-relevant benchmark | B3 | PARTIAL |
| Statistical reproducibility | A1; Stage B (B-R0/R1), matched controls, provenance (commit, dirty flag, environment), sealed update logs, per-seed appendix; disclosure of archived dirty-tree runs | PLANNED |
| Threat-model clarity | B1 | READY |
| Writing quality | B4–B6; full proofread; consistent notation with equation library | TODO |
| Specificity (not raised, but self-identified) | L-11 clean Multi-Krum honest alarm; X1 clean FedAvg | MUST address before submission |
