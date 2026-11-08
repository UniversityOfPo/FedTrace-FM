# 33 — Workshop Results Selection (4 pages + references)

Prepared 2026-10-27 (Prompt 35). PROPOSED for PI / author approval; claim IDs refer to doc 32.

## 1. Contribution recommendation (Task 3)

| Proposed | Supported? | Recommended wording |
|---|---|---|
| C1 Forensic reliability decomposition (detection / attribution / onset / honest accusation / effectiveness) | **Yes as an evaluation methodology, not as a new concept** — FLDetector reports per-client FPR / FNR, FLTracer reports attack time and type (doc 36 §6). What is supported: applying the five outcomes *separately and at seed level* to matched interventions, including benign drift | "We evaluate client-level forensics with five separately reported outcomes on matched clean, poisoning and benign-drift interventions" |
| C2 Matched-intervention evaluation (poisoning, benign drift, clean controls) | **Yes** — bit-identical pre-intervention histories (all pairs valid), preregistered confirmation seeds | merge with C1 into one contribution |
| C3 Cross-seed analysis of calibration failure and conditional reference contamination | **Yes, with qualification** — calibration failure (v3 misses, threshold transfer) confirmed on new seeds; contamination is an exact elementary identity plus a conditional explanation observed in one seed | "We show … and characterise a rank-conditioned reference-contamination route to false attribution, observed in one of four seeds" |

**Final recommendation:** two contributions — (1) a matched, five-outcome forensic evaluation of an existing
framework (FedTrace v3) and standard change detectors on four seeds, two preregistered; (2) the cross-seed findings
(C-01 – C-04, C-07) with the conditional contamination analysis (C-06) — plus an honest limitations statement. No
novelty claim for median algebra, z-scores, CUSUM or the outcome decomposition.

## 2. Minimum evidence for the paper (Task 4)

| Item | Content | Source | Claims |
|---|---|---|---|
| **Table 1** (`tables/table1_seed_level.tex`) | 12 rows (BadNets / drift / clean × 123, 2026, W1, W2) × {ASR, Δtop-1, v3, v3 magnitude-only, v3 t_res-only, ZP-norm, CUSUM-norm}; cell = outcome, onset error / delay, honest accusations | unified tables (docs 28, 30) | C-01–C-04, C-07, C-09, C-10 |
| **Fig. 1** (`figures/fig1_reference_contamination.pdf`) | (a) seed 2026: client_002 crosses the others' band upward, median moves; (b) honest residual scores pass τ and are confirmed first; (c) W1: client_002 starts above the band, median unchanged | X2_badnets_d2026, W1_badnets traces | C-05, C-06a, C-06b |
| **Fig. 2** (`figures/fig2_reliability_grid.pdf`) | outcome grid (detectors × seeds) for BadNets, drift, clean — the reliability trade-off: no detector is both attributing and drift-silent | same as Table 1 | C-01–C-04 |
| Text only | archived perturbation context (one sentence, C-11); relabel-only (one sentence, development); identifiability statement (C-13, one sentence); falsified exploratory analyses (Limitations, one sentence) | docs 18, 15, 19 | C-11, C-13 |
| Optional, only if space | positioning table (methods × online / attribution / onset / data poisoning / benign drift / clients evaluated) from doc 20 | doc 20, doc 36 §6 | U-04, U-05 guard |

Excluded (space; not central): Multi-Krum runs, K = 10, severity / intermittent detail, PH-norm and reference
variants (one sentence: historical reference accuses in clean runs, C-12), D1, label-shift, FLDetector audit.

Note on Table 1 vs Fig. 2: they encode the same outcomes; if space is short keep Table 1 (exact numbers) and Fig. 1,
and drop Fig. 2.
