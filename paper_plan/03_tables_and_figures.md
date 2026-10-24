# 03 — Planned Tables and Figures

Status: living document, created 2026-10-24. **No cell is filled with an estimate.** Evidence status per item:
ARCHIVED (exists in hash-sealed NDSS artifacts), PENDING (needs runs in `02_experiment_matrix.md`), PARTIAL.
Proposed output location for new assets: `paper_assets_rev/{tables,figures}/` with a provenance CSV per item
(the NDSS `paper_assets/` tree stays frozen). Source runs: `results_rev/<run_id>/` (trace
`client_round_trace.csv`, `summary.json`, `rev_attack_records.json`, `rev_provenance.json`), update logs
`logs_rev/<run_id>/update_log/`, CPU analysis outputs `analysis_rev/`.

Uncertainty conventions: binary outcomes k/n with 95 % Clopper–Pearson; continuous per-seed values listed with
mean; delays with censoring marked "≥"; no p-values from fewer than 5 seeds; client-round rates secondary.

## Main paper

| ID | Exact title | Scientific question | Layout | Metrics / units | Data required | Uncertainty | Status |
|---|---|---|---|---|---|---|---|
| T1 | Threat model and attack classes | What can each attacker do and what does it want? | rows: Class A (Gaussian, directional), B (backdoor, label flip), C (norm-constrained backdoor); cols: capability, detector knowledge, objective, effect metric, stealth constraint, unconstrained comparator | — | none (design) | — | READY TO WRITE |
| T2 | Experimental setup | — | setting / value (update archived `table01`) | — | configs | — | PARTIAL (archived; add attack params, seeds after sealing) |
| T3 | Online attribution, onset localisation and false alarms across conditions | RQ1, RQ2 | rows: clean (X1), backdoor (X2), constrained backdoor (X3), Gaussian (archived), directional (archived); cols: runs n, detection k/n [CI], attribution k/n [CI], onset abs. error (n detected), alarm delay (rounds; censored ≥), runs with honest alarm k/n, honest alarms /1,000 client-rounds, attack success % (mean; per seed in App.), Δ clean top-1 pp (paired) | rounds, %, pp | X1–X3 V seeds; archived EXP_016–019 for Class A | CP; per-seed list | PENDING (Class A rows ARCHIVED but computed offline; recompute online) |
| T4 | Comparison with change-detection and forensic baselines at matched false-alarm level | RQ5 | rows: FedTrace (online), z-score+P on m, CUSUM, Page–Hinkley, naive peer-cosine, Multi-Krum rejection proxy, [published methods only if verified]; cols per attack (backdoor, constrained, Gaussian, directional): attribution k/n, median delay, honest-alarm runs k/n | rounds | X1–X3 + archived traces; thresholds from D clean runs | CP | PENDING |
| T5 | What a detector-aware attacker gives up | RQ3 | rows: seeds (V); cols: unconstrained success, constrained success, constraint satisfied rounds/total, constrained detected?, alarm round, signal that fired | %, rounds | X2, X3 | per-seed (no pooling claim beyond k/n) | PENDING |
| T6 | Evidence components and sensitivity | RQ6 | rows: m only, t_res only, v3 (m+t_res), τ ∈ {2.5, 3, 3.5}, P ∈ {2, 3, 4}; cols: attribution k/n and median delay per attack, clean honest-alarm runs k/n | rounds | all traces | CP | PARTIAL (archived ablation `table03` for Class A, 2 seeds) |
| T7 | Robustness to deployment assumptions | RQ4 | rows: calibration contamination (X4), partial participation (X5), two attackers (X6, if run); cols as T3 | as T3 | X4–X6 | CP | PENDING (SHOULD) |
| F1 | FedTrace overview with online event timeline | method | diagram: signals → calibration (rounds < T_ref) → scores → persistence; timeline marking t₀, alarm a, actionable a+1, onset â | — | none | — | PARTIAL (archived `fig01`; add timeline) |
| F2 | Per-client online scores under clean, backdoor and constrained backdoor (one validation seed) | RQ1/RQ3 illustration | 3 panels; x: round (0–49); y: S_k^r (max(z_m, z_t_res)); τ line; shaded calibration window; markers t₀, alarm | z-units | X1–X3 one pre-chosen V seed (choose seed *before* seeing results) | illustrative only — caption states seed | PENDING |
| F3 | Attack success versus detection delay | RQ3 trade-off | scatter; x: attack success (%) ; y: alarm delay (rounds, censored plotted at top with marker); colour: condition; one point per run | %, rounds | X2, X3 (+X4) | every run shown | PENDING |
| F4 | Which signal fires for which attack | RQ6 | stacked bars: share of above-τ rounds by dominant component per attack class | % of rounds | all attacked traces | per-run dots | PARTIAL (archived `fig03` for Class A) |
| F5 | Honest-client score drift: clean FedAvg vs clean Multi-Krum | RQ2 / specificity | x: round; y: z_m of each honest client; panels: FedAvg clean (X1), Multi-Krum clean (EXP_020/021) | z-units | X1 + archived | all clients shown | PARTIAL (MK ARCHIVED; FedAvg PENDING) |

## Appendix

| ID | Exact title | Question | Layout | Data | Status |
|---|---|---|---|---|---|
| A1 | Reproduction of EXP_016 and logging non-interference | Is the code revision equivalent to the archive? Does logging change training? (Engineering evidence only — not an additional seed; blueprint §10a) | rows = comparison categories of `src/rev/compare_runs.py`: CONFIG, PARTITION, RNG_PROXY (client selection; local-training trajectory; update norms), HISTORY (timing excluded), TRACE (order-independent fields), TRACE_ORDER_DEPENDENT (peer cosine + naive-baseline fields, reported separately), DETECTOR (v3 offline/online decisions and component scores), GLOBAL_STATE (final model, server checkpoint), CLIENT_STATE (last local models + optimizer), METRICS, RUNNER, UPDATE_LOG (R1 only: verification, re-aggregation, logged final = final model, replayed trace = live trace); cols: A = R0 vs archived EXP_016, B = R1 vs R0; cell = IDENTICAL / NUMERICAL (max abs, max rel, first round) / VALUE / STRUCTURAL / MISSING, plus EXCLUDED items listed (timing columns, absolute output paths, plots, run.log) | B-R0, B-R1; reports `results_rev/comparisons/{R0-vs-EXP016,R1-vs-R0}.json` | DATA AVAILABLE (CLAUDE-REPORTED, L-20/L-21): A = 23 IDENTICAL / 0 differences; B = 33 IDENTICAL / 0 differences; table not yet rendered |
| A2 | Sensitivity heatmaps (τ × P, T_ref) | robustness of conclusions | heatmap per attack; cell = attribution k/n | all traces | PENDING |
| A3 | Run registry | provenance | run id, condition, seed, commit, execution_clean, config hash, partition hash, init fingerprint, log manifest hash | `rev_provenance.json`, `rev_status.json` | PENDING |
| A4 | Per-seed results | transparency | every run, every T3 metric | all | PENDING |
| A5 | Online versus offline FedTrace v3 on archived traces | Does causality change any archived outcome? | rows: 21 archived traces; cols: score differences at r ≥ T_ref, online alarm/attribution, offline attribution | archived (CPU) | CLAUDE-REPORTED (0 differences; needs independent re-run) |
| A6 | Multi-Krum: rejection versus attribution | robust aggregation ≠ attribution | archived `table04` + FedTrace on clean MK | archived | ARCHIVED + CLAUDE-REPORTED clean-MK alarm |
| A7 | Geometry stress tests (Class A) | which signal sees which geometry | archived EXP_014–019, 026, 027, E1–E3 | archived | ARCHIVED (relabel as stress tests) |
| A8 | Calibration-masking bound | what calibration cannot see | z_max(n, k) table for n = 20, k = 1..5 (analytic) | none | READY (derivation; general case HYPOTHESIS) |
| A9 | Utility | cost of attack / defence | final top-1 per condition, paired | X1–X3 | PENDING |
