# 22 — Experiment Priority Plan (nothing executed; PI approval required for every item)

> **Review note — 2026-10-26 (Prompt 33, P-49).** P-50 S1 selected (new seeds; sealed seeds untouched); GPU-1 deferred until the frozen protocol (doc 26) is reviewed (P-53); GPU-2…6 deferred (P-55); CPU-4 (FLDetector) reduced to a feasibility audit (P-54, doc 27); CPU-1/2/3 executed in Prompt 33 under doc 25.

Prepared 2026-10-26 (Prompt 32). Labels: **PROPOSED** unless stated. Cost basis (CLAUDE-REPORTED from run logs):
one 50-round K = 5 run on the RTX 5060 laptop GPU takes 69–84 min of training loop (X runs 4,132–5,027 s) plus
evaluation ≈ **1.2–1.4 GPU-h**; K = 10 runs took 1.9–2.9 h (E1). Storage: 1.1 MB results + 0.5 GB checkpoints per
run; **+13 GiB** if update logging is enabled; free disk 505 GB. A matched-prefix pair costs two runs unless an
existing clean run of the same seed is reused.

## 1. Seed policy for any new run (decision required, P-50)

- Seeds 123 / 2026: development only (registry). Seed 42: used in NDSS development (EXP_014/015).
- **Option S1 — SELECTED by the PI (P-50):** derive two **new development-confirmation seeds** by a published rule (e.g.
  SHA-256("FedTrace-workshop-confirmation|2026-10-27")), register them, and freeze a confirmatory protocol for
  **already-frozen** methods (v3, ZP-norm h 1.75, CUSUM-norm h 11.75, protocol 13 drift) before any run. Independent
  evidence without consuming the sealed seeds reserved for CVPR.
- **Option S2 — REJECTED (P-50); C1 / C2 / spare stay sealed:** unseal C1 / C2 (978310317, 478485063) now, since every method to be evaluated is frozen. Consumes
  the CVPR confirmation seeds (spare 1082418221 remains).
- Either way: protocol frozen and committed before the first run; no threshold re-calibrated on new outcomes.

## 2. Experiments ranked by information gain per GPU-hour

| Rank | ID | Scientific question | Expected information gain | Reusable code | Required changes | GPU-h | Storage | Baselines evaluated | Failure / falsification criteria (pre-stated) | Manuscript support |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **CPU-1** Contamination characterisation test (Option A, doc 21) | Do upward median crossings predict exactly which honest clients v3 accuses? | High (turns a one-seed mechanism into a tested proposition) | `src/rev/reference_audit.py`, `phase_s.py`, archived + dev traces | new read-only script + tests; preregistration of the prediction before computing on archived traces | **0** | < 10 MB | v3 (inclusive), LOO, trimmed | falsified if any honest t_res confirmation lacks a preceding upward crossing, or a predicted ≥ τ shift for P rounds yields no confirmation | Fig. 1, Prop. 1 |
| 2 | **CPU-2** Unified outcome table over all existing traces (archive perturbations + dev poisoning + drift + clean) incl. v3 ablations and reference ablations | Which evidence attributes which attack family, and at what false-accusation cost? | High (consolidation; no new data) | `evidence_matrix.py`, `phase_s.py`, `fedtrace_detector.py` | extend the matrix to archived traces (no ASR there) | **0** | < 10 MB | v3, magnitude-only, t_res-only, ZP-norm, CUSUM-norm, PH, LOO / trimmed / historical | — (descriptive); report every cell incl. failures | Table 1, Fig. 2 |
| 3 | **CPU-3** Uncertainty and sensitivity | How much do conclusions depend on τ, P, h and on two seeds? | Medium | `scripts/ndss_e4_sensitivity.py`, `phase_s.calibration_sensitivity` | extend E4 grid to dev runs; exact (Clopper–Pearson) intervals on counts; effective-n statement | **0** | negligible | v3, ZP, CUSUM | report; conclusions that flip within the grid are labelled fragile | text, Table 1 footnote |
| 4 | **GPU-1** Confirmatory replication on two new seeds: clean, BadNets, benign drift (frozen methods; traces only, logging off) | Do the seed-2026 contamination / false-accusation pattern and the drift falsification replicate? Out-of-sample test of Option A | **Highest GPU value** | `run_rev_experiment`, configs X1/X2/X4 (seed fields only), pilot, benign_drift, evidence_matrix | new configs (seed only), preregistration (doc), seed registration | **6 runs ≈ 7.5–8.5** | ≈ 3 GB (no logs) | all direct baselines | pre-stated: report attribution / FA / onset per detector; Option A predictions computed before looking at detector outcomes | Table 1 (confirmatory rows), Fig. 2 |
| 5 | **GPU-2** Relabel-only on the same two new seeds | Does the "attacker missed / honest accused" pattern of relabel-only replicate? | Medium | X3 configs | seed only | 2 runs ≈ 2.5–2.8 | ≈ 1 GB | all direct | as GPU-1 | Table 1 |
| 6 | **GPU-3** BadNets with onset 30 (new seeds, reuse their clean runs) | Is onset localisation genuine when onset ≠ T_ref? | Medium-high (removes the onset = T_ref confound, L-34) | X2 configs | onset field | 2 runs ≈ 2.5–2.8 | ≈ 1 GB | all direct (onset errors) | onset error reported per detector; no re-tuning | Table 1 onset column |
| 7 | **CPU-4** FLDetector per-client consistency score (re-implemented, monitor-only) on logged runs | Does the closest published temporal evidence behave differently under drift / contamination? | Medium (partial baseline) | `update_log` readers | re-implementation (~1 day + tests), L-BFGS Hessian approximation; labelled "our adaptation" | 0 (CPU hours on 8 logged runs; new runs would need logging +13 GiB each) | negligible | FLDetector component | its published clustering is degenerate at K = 5 — report the score, not a re-tuned decision | Table 1 extra column or text |
| 8 | **GPU-4** Second benign-drift type (e.g. symmetric label noise or new-class acquisition), new seeds | Are drift false accusations specific to class-composition drift? | Medium | `BenignClassDriftDataset` pattern | new benign wrapper + tests + preregistration | 2 runs ≈ 2.5–2.8 | ≈ 1 GB | all direct | H-BEN per detector as protocol 13 | text / Table 1 |
| 9 | GPU-5 Stronger BadNets at seed 123 (higher poison rate) | Is the 43.5 % ASR result due to attack strength? | Low (changes the attack after seeing outcomes) | X2 | poison rate | 2 runs | ≈ 1 GB | – | – | not recommended before submission; report as is |
| 10 | GPU-6 K = 10 poisoning / drift | Does contamination weaken with more clients? | Medium, expensive | NDSS E1 configs + rev attacks | K = 10 configs | 3 runs ≈ 6–9 | ≈ 1.5 GB | all direct | – | not before submission |
| 11 | FLTracer adaptation | Fair comparison to the closest prior | Low per hour (degenerate cross-client channels at K = 5; constants out of regime) | – | log export to npz, reference policy, running third-party code (PI approval) | 0–2 | – | FLTracer | – | qualitative positioning only (Table 2) |

## 3. Minimum credible set (PROPOSED)

CPU-1, CPU-2, CPU-3 (0 GPU) + GPU-1 (≈ 8 GPU-h) — **≈ 8 GPU-h total**. Recommended if the GPU is free for one more
evening: + GPU-3 (≈ 2.7 GPU-h) and GPU-2 (≈ 2.7 GPU-h) — **≈ 13–14 GPU-h total**. CPU-4 only if Day 2 shows a clean
re-implementation path. Items 9–11 are deferred.

## 4. Rules

No GPU run before (i) PI approval of the item, (ii) a committed preregistration, (iii) a seed decision (§1). Every
new run uses frozen methods and frozen thresholds; no re-calibration on new outcomes; sealed seeds untouched unless
S2 is chosen explicitly; no tracked-file edits while a run executes (D-039).
