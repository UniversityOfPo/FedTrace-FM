# 18 — Original FedTrace (NDSS) Asset Inventory

> **Review note — 2026-10-26 (Prompt 33, P-49).** Status label of EXP_016/017 corrected (no row is PI-VERIFIED). New §5: matched-counterfactual availability for archived traces (checked on CPU, read-only).

Prepared 2026-10-26 (Prompt 32). Read-only inventory of the preserved NDSS archive; nothing in it was modified.
Labels: **VERIFIED** (PI sign-off — none yet, per ledger convention), **CLAUDE-REPORTED** (checked by Claude from
artifacts with a command), **UNVERIFIED**, **PROPOSED**. HEAD at preparation: `5f79305`.

Recomputation used here: frozen `src/analysis/fedtrace_detector.py::evaluate_detector_v3` (offline) and
`src/rev/online.py::online_detect_v3` (causal) on every archived `client_round_trace.csv` (21 traces), CPU, read-only;
output in the scratchpad (`recompute_archive.json`), not committed. All 21 recomputed attributions match the archive.

## 1. Manuscript and paper assets

| Asset | Path | Status |
|---|---|---|
| NDSS manuscript source (`.tex` / `.pdf` / `.docx`) | **not found in this repository** (search over `*.tex`, `*.pdf`, `*.docx`, excluding data / logs / checkpoints) | UNVERIFIED — the PI must supply the manuscript or its location |
| Paper blueprint (rev-era) | `paper_plan/01_paper_blueprint.md` | CLAUDE-REPORTED (planning, not the manuscript) |
| Main tables (CSV + LaTeX) | `paper_assets/tables/main/table0{1..5}_*.csv`, `paper_assets/tables/latex/*.tex` | present; provenance `paper_assets/tables/TABLE_DATA_PROVENANCE.csv` |
| Appendix tables | `paper_assets/tables/appendix/tableA{1,2,3}_*.csv` | present |
| Main figures | `paper_assets/figures/main/fig01_fedtrace_overview.pdf`, `fig02_onset_trajectory(_horizontal).pdf`, `fig03_signal_complementarity.pdf`, `fig04_multikrum_geometry.pdf`, `fig05_aggregation_vs_attribution.pdf` | present; provenance `paper_assets/figures/FIGURE_DATA_PROVENANCE.csv` |
| Appendix figures | `paper_assets/figures/appendix/fig06_utility_2x2.pdf`, `fig07_partition_fairness.pdf`, `fig08_generalization_check.pdf` | present |
| Experiment registry | `Research_Database/Experiment_Index.csv`, `analysis/FEDTRACE_EXPERIMENT_REGISTRY.csv`, `Research_Database/FedTrace_Master_Results.{md,xlsx}` | present |
| Method / analysis documents | `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md`, `FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md`, `FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md`, `FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md`, `analysis/FEDTRACE_*_RETROSPECTIVE_ANALYSIS.md` | present |
| NDSS E1–E4 extensions | `results_ndss/E{1..4}_*/` (summaries, CSV, PNG, logs), configs `configs/ndss_e*.yaml`, scripts `scripts/ndss_e*_postprocess.py`, `scripts/ndss_e4_sensitivity.py` | present |

## 2. Original method (from code and archived documents)

| Element | Original FedTrace v3 (frozen at `7f6ca8b`; code `src/analysis/fedtrace_detector.py`) | Status |
|---|---|---|
| Problem formulation | Server-side, monitor-only forensic attribution of a single malicious client and its attack onset in cross-silo FedAvg (K = 5, full participation), from per-round per-client update statistics | CLAUDE-REPORTED |
| Threat model (as evaluated) | One attacker from a fixed onset; **model-update perturbations**: Gaussian update noise (ρ = 1, 0.5) and a norm-preserving directional perturbation (target cosine c = 0, 0.5); intermittent variants (E3); no data poisoning in the archive | CLAUDE-REPORTED |
| Evidence | (i) **magnitude**: z_m = two-sided z of the update L2 norm against the client's own calibration (rounds < T_ref = 20); (ii) **temporal residual**: d = c(r) − t_k(r), with t_k the client's self-cosine (update vs its previous update) and c(r) the **inclusive same-round median** of self-cosines; z_t_res = one-sided max(0, ·) z of d against the client's own calibration; score S = max(z_m, z_t_res) | CLAUDE-REPORTED |
| Peer-relative evidence | v1's peer signal z_p (peer-mean cosine) — dropped in v3 (Table 3: never contributes a detection); v3's only peer-relative element is the median reference c(r) | CLAUDE-REPORTED |
| Calibration | Per-client mean / sd over rounds < 20 (offline: self-inclusive; online wrapper `src/rev/online.py` causal) | CLAUDE-REPORTED (L-12, L-13) |
| Decision rule | Threshold τ = 3, persistence P = 3 consecutive rounds → confirmation; attribution = earliest confirmation (tie-break by streak mean) | CLAUDE-REPORTED |
| Onset localisation | Start of the confirming streak (alarm round − P + 1) | CLAUDE-REPORTED |
| Method evolution | v1 (non-stationarity flaw), v2 (small-sample MAD instability), v3 (temporal residual calibration) — `tableA2_method_evolution.csv` | CLAUDE-REPORTED |
| Computational overhead | Per round O(K) scalar statistics from the trace; no timing reported in the archive | CLAUDE-REPORTED (no measurement) |
| Setting | CIFAR-100, ResNet-18 (CIFAR variant), K = 5, full participation, 50 rounds, 1 local epoch, SGD lr 0.01, batch 32, Dirichlet α = 0.1; E1 uses K = 10 | CLAUDE-REPORTED (`table01_experimental_setup.csv`) |
| Seeds | 42 (development: EXP_014/015), 123 and 2026 ("validation" EXP_016–019; the same seeds were later re-used as development seeds in the rev phase — no independent seed exists outside the sealed confirmation seeds) | CLAUDE-REPORTED |
| Metrics | Attribution correct, first anomaly, confirmation round, onset error, detection delay, honest persistent false positives, dominant signal, final top-1 | CLAUDE-REPORTED |
| Baselines in the archive | Peer-only / magnitude-only / temporal-residual-only ablations; Multi-Krum (f = 1) as an aggregation baseline with a P = 3 rejection-persistence proxy; naive peer-agreement detector (`apply_naive_detector`); E4 τ × P sensitivity grid | CLAUDE-REPORTED |
| Reproducibility | Stage B: EXP_016 reproduced bit-for-bit (L-20); logging non-interference (L-21); dirty-tree provenance caveat for EXP_020–025 and E1–E3 (L-17, ledger header) | CLAUDE-REPORTED |

## 3. Result inventory

Columns: (1) ID, (2) source, (3) dataset, (4) model, (5) attack, (6) K, (7) seeds, (8) metric, (9) reported result,
(10) raw artifact, (11) recomputable, (12) workshop suitability. All CIFAR-100 / ResNet-18-CIFAR / Dirichlet 0.1 /
50 rounds unless stated.

| ID | Source (trace / table) | Attack | K | Seeds | Reported result | Raw | Recomputed 2026-10-26 | Class | Notes |
|---|---|---|---|---|---|---|---|---|---|
| EXP_001 | `results/centralized/cifar100_resnet18` | none (centralised) | – | 42 | top-1 62.71 % | summary | not recomputed | REUSABLE WITH QUALIFICATION | utility reference only |
| EXP_002–007, 010–012 | `results/federated/*` | none | 5 | 42 / 123 / 2026 | utility (FedAvg IID, α 0.5 / 0.1, FedProx) | summaries | not recomputed | RELATED ONLY | development utility studies; not forensic |
| EXP_008 / 009 | `results/federated/*seed{123,2026}_50rounds` | none (clean FedAvg) | 5 | 123, 2026 | top-1 56.83 / 58.26 % | summary; **no trace** (L-18) | — | INCOMPATIBLE for specificity (no trace) | clean specificity measured later by X1 (rev) |
| EXP_014 / 015 | `results/federated/*seed42` | Gaussian / directional | 5 | 42 | attribution correct (development) | trace | yes: client_002, alarm 22, onset 20 | REUSABLE WITH QUALIFICATION | development runs (used to design v3) |
| EXP_016 / 017 | `table02`; `results/federated/*gaussian*seed{123,2026}` | Gaussian ρ = 1 | 5 | 123, 2026 | correct, onset 20, confirm 22, delay 2, 0 honest FP | trace; EXP_016 reproduced bit-exactly (L-20) | yes: identical | **REUSABLE WITH QUALIFICATION** (CLAUDE-REPORTED: recomputed 2026-10-26; no PI-VERIFIED row exists) | Gaussian noise is a geometry stress test, not poisoning (L-01) |
| EXP_018 / 019 | `table02`; `*direction_attack_client002_round20_seed{123,2026}` | directional c = 0 | 5 | 123, 2026 | correct, onset 20, delay 2, 0 honest FP; signal z_t_res | trace | yes: identical | same as above | stress test, not poisoning |
| Table 3 ablation | `table03_signal_ablation.csv` | Gaussian / directional | 5 | 123, 2026 | magnitude-only late on directional (err 24–27); t_res-only fast on directional; full = coverage | traces | not re-run here (recomputable from traces) | REUSABLE WITH QUALIFICATION | "coverage, not fusion" framing (L-06) |
| EXP_020 / 021 | `*multikrum*seed{123,2026}_50rounds` | none (clean, Multi-Krum) | 5 | 123, 2026 | MK proxy implicates largest-partition honest client 49/50 rounds; **v3 confirms an honest client at round 46 (onset 44)** | trace | yes: client_004 (46, 44) / client_001 (46, 44) | REUSABLE WITH QUALIFICATION (negative result, L-11) | dirty-tree caveat (L-17) |
| EXP_022 / 023 | `table04`; `*multikrum_gaussian*` | Gaussian under MK | 5 | 123, 2026 | MK rejects attacker every round; v3 correct | trace | yes: client_002 (22, 20), no honest alarm | REUSABLE WITH QUALIFICATION | dirty-tree caveat |
| EXP_024 / 025 | `table04`; `*multikrum_direction*` | directional under MK | 5 | 123, 2026 | MK never rejects attacker; v3 correct | trace | yes: client_002 (22, 20) **plus a late honest alarm** (seed 123: client_004 at 47; seed 2026: client_001 at 46) | REUSABLE WITH QUALIFICATION — the late honest alarm is **not shown in Table 4** and must be reported | dirty-tree caveat |
| EXP_026 | `table05`; `*client004_round20_seed123` | directional, attacker client_004 | 5 | 123 | correct, onset 20, delay 2 | trace | yes: identical | REUSABLE WITH QUALIFICATION | n = 1 (L-05) |
| EXP_027 | `table05`; `*round30_seed123` | directional, onset 30 | 5 | 123 | correct, onset 30, delay 2 | trace | yes: client_002 (32, 30) | REUSABLE WITH QUALIFICATION | the only onset ≠ T_ref run |
| E1 | `results_ndss/E1_10clients/` | Gaussian / directional / clean | **10** | 123 | correct (onset 20, delay 2); clean K = 10: no alarm | traces | yes: identical; clean: no alarm | REUSABLE WITH QUALIFICATION | n = 1 per condition; dirty-tree caveat |
| E2 | `results_ndss/E2_attack_severity/` | Gaussian ρ = 0.5; directional c = 0.5 | 5 | 123 | ρ 0.5: **late** (onset est. 41, confirm 43, error 21); c 0.5: onset 20, delay 2 | traces | yes: identical | REUSABLE WITH QUALIFICATION (late detection is a limitation, L-10) | n = 1 |
| E3 | `results_ndss/E3_intermittent/` | intermittent Gaussian / directional | 5 | 123 | correct, onset 20, delay 2 | traces | yes: identical | REUSABLE WITH QUALIFICATION | n = 1 |
| E4 | `results_ndss/E4_detector_sensitivity/` | (re-analysis of EXP_016–019) | 5 | 123, 2026 | τ ∈ {2.5, 3, 3.5}, P ∈ {2, 3, 4}: P = 2 breaks attribution; P ≥ 3 preserves conclusions | summary CSV | recomputable from traces (not re-run here) | REUSABLE WITH QUALIFICATION | post-hoc sensitivity on the same four runs |
| Utility (Table A3, Fig 6) | `tableA3_utility_metrics.csv` | various | 5 | 123, 2026 | top-1 per run | summaries | not recomputed | REUSABLE WITH QUALIFICATION | n = 2 per cell; no significance |
| Claim "no honest persistent FP" | Table 2 column | attacked FedAvg runs only | 5 | 123, 2026 | 0 / 16 honest client-runs | traces | consistent | **REUSABLE ONLY AS SCOPED**: attacked FedAvg runs; contradicted for MK clean / MK directional runs, for BadNets / relabel / drift at seed 2026 (doc 19) | must never be stated unscoped |
| Claim "FedTrace attributes poisoning" | (not in the archive) | — | — | — | — | — | — | UNSUPPORTED from the archive (archive has no data poisoning); see doc 19 | |

**Incompatibility rules for the workshop paper.** Archived runs (perturbation attacks, NDSS runner, no update logs)
and rev runs (data poisoning / drift, rev runner with logs) share dataset, model, partition, seeds and FedAvg, so
their **trace-level detector outcomes are comparable**; their attack families are different and must be reported in
separate rows, never pooled into one rate. ASR exists only for rev runs. Multi-Krum runs use a different aggregation
rule and must not be pooled with FedAvg runs.

## 4. Negative and qualifying results that must be kept

L-11 (v3 honest confirmation in clean Multi-Krum runs), E2 ρ = 0.5 late detection (L-10), the newly recorded late
honest alarms in the two Multi-Krum directional runs, L-13/L-14 (offline calibration is self-inclusive; masking bound),
L-17 (dirty-tree provenance), L-18 (no clean FedAvg trace in the archive).

## 5. Matched clean counterfactual for archived traces (CLAUDE-REPORTED, 2026-10-26, Prompt 33)

Trace rows (update norm, self-cosine, peer field) of every archived FedAvg K = 5 run at seeds 123 / 2026 are
identical to the development clean run of the same seed (`results_rev/X1_clean_d{123,2026}`) in every round before
the attack onset, and first differ exactly at the onset round: EXP_016, 017, 018, 019, 026, 027 (onset 30), E2 (two
runs), E3 (two runs). Hence X1 serves as a **trace-level** matched counterfactual for these ten runs (retrospective
diagnosis only). No counterfactual exists for the Multi-Krum runs (different aggregation; differ from round 1), the
K = 10 runs (E1) or the seed-42 development runs (EXP_014/015). This is trace-level equality, not state-level.
