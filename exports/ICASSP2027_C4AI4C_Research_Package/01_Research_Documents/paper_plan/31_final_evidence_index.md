# 31 — Final Evidence Index (frozen for the C4AI4C'27 manuscript)

Frozen 2026-10-27 (Prompt 35) at HEAD `4f7b871`. Labels: **CONFIRMATION** (preregistered seeds W1, W2),
**DEVELOPMENT** (123, 2026), **ARCHIVED** (NDSS traces; model-update perturbations), **ANALYTIC**. All entries
CLAUDE-REPORTED (no PI-VERIFIED row). Nothing is pooled across rows.

## 1. Immutability check (performed 2026-10-27)

| Evidence group | Check | Result |
|---|---|---|
| 21 archived NDSS traces | git-tracked; `git status` clean | unchanged |
| NDSS freeze manifest (EXP_016 artifacts) | preflight `archive` check | PASS (8 files, 0 changed) |
| 8 development runs (142 files) | SHA-256 vs the list recorded 2026-10-25 | 0 mismatches |
| Development analyses (`contamination.json`, `unified_outcomes.json`, `evidence_matrix.json`) | SHA-256 vs pre-Prompt-34 values | unchanged |
| Everything cited by the paper | **`paper_plan/31_evidence_manifest.sha256`** — 370 files (traces, summaries, run directories, update-log manifests and seals, pair analyses, analysis JSONs); verify with `sha256sum -c paper_plan/31_evidence_manifest.sha256` | self-verifies |

## 2. Index

| Experiment | Seed(s) | Configuration | Artifact path(s) | Commit (code / run) | Outcome (seed-level) | Status | Manuscript claim (doc 32) |
|---|---|---|---|---|---|---|---|
| Archived Gaussian ρ = 1 (EXP_016/017) | 123, 2026 | `configs/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed{123,2026}.yaml` | `results/federated/*gaussian*seed{123,2026}/` | `7f6ca8b`, `bfe9171` | v3 correct, onset error 0, delay 2 | ARCHIVED | C-11 |
| Archived directional c = 0 (EXP_018/019/026/027) | 123, 2026 | `…direction_attack…` configs | `results/federated/*direction*` | `12d030a` | v3 correct, onset error 0, delay 2 (incl. onset 30, attacker client_004) | ARCHIVED | C-11 |
| Archived E1 (K = 10), E2 (severity), E3 (intermittent) | 123 | `configs/ndss_e*.yaml` | `results_ndss/E{1,2,3}_*/` | `df4d814` (dirty-tree caveat, L-17) | v3 correct; E2 ρ 0.5 late (onset error 21) | ARCHIVED | C-11 (context) |
| Archived Multi-Krum (EXP_020–025) | 123, 2026 | `configs/cifar100_multikrum_*` | `results/federated/*multikrum*` | `12d030a` (dirty-tree caveat) | honest confirmations in clean and directional runs (norm-driven) | ARCHIVED | not used in the paper (different aggregation) |
| Development clean X1 | 123, 2026 | `configs/rev/X1_clean_d{123,2026}.rev.yaml` | `results_rev/X1_clean_d*/`, `logs_rev/X1_*` | `bb995c1`, `a7290fc` | no confirmation (v3, ZP, CUSUM) | DEVELOPMENT | C-02, C-04 |
| Development BadNets X2 | 123, 2026 | `X2_badnets_d*.rev.yaml` | `results_rev/X2_badnets_d*/`, pair dirs | `bb995c1`, `a7290fc` | ASR 43.5 % / 64.7 %; v3 correct-late / **wrong client**; ZP, CUSUM correct | DEVELOPMENT | C-02, C-05, C-07, C-09 |
| Development relabel-only X3 | 123, 2026 | `X3_relabel_d*.rev.yaml` | `results_rev/X3_relabel_d*/` | `3f2f3b4` | v3 missed both; honest accusations at 2026 | DEVELOPMENT | C-05 (supporting) |
| Development drift X4 | 123, 2026 | `X4_drift_d*.rev.yaml` | `results_rev/X4_drift_d*/` | `934c9c0` (2026: execution_clean False, accepted, L-68) | honest accusations by ZP, CUSUM (both), v3 (2026, reference-driven) | DEVELOPMENT | C-03, C-05 |
| Confirmation W1 / W2 (clean, BadNets, drift) | 1178015705, 1463065807 | `configs/rev/W{1,2}_{clean,badnets,drift}.rev.yaml` | `results_rev/W*/`, `logs_rev/W*/`, `results_rev/pilot/W*__vs__*` | `4720670` (all execution_clean) | ASR 63.3 % / 44.2 %; v3 **missed** both; ZP, CUSUM correct (+ honest alarm W1); drift accusations both; W1 clean honest alarm by ZP / CUSUM | CONFIRMATION | C-01, C-02, C-03, C-04, C-07 |
| Phase S baselines | 123, 2026 | protocol 12 (`f706150`) | `results_rev/pilot/phase_s/` | `f706150` | thresholds ZP 1.75, CUSUM 11.75, PH 6.5 | DEVELOPMENT | baseline definition |
| Unified outcome tables | all | `src/rev/unified_outcomes.py` | `results_rev/pilot/unified_outcomes/*.json`; docs 28, 30 | `164029e` / `4720670` | 9 frozen detectors × 35 traces | as rows | Table 1, Fig. 2 |
| Contamination analysis | all | protocol 25 (`164029e`, A1 `0f6a313`, A2 `4720670`) | `results_rev/pilot/contamination/*.json` | as stated | identity exact; 6 reference-driven accusations explained (2026 only); not exercised on W1 / W2; proxy unreliable on W | ANALYTIC + DEVELOPMENT + CONFIRMATION | C-05, C-06, C-08 |
| Identifiability analysis | — | doc 15 (+ A1) | `paper_plan/15_*` | `e0531ce` | two-point bound scope; precedence lemma | ANALYTIC | C-13 |
| Falsified exploratory analyses | 123, 2026 | protocols 14, 16 | `results_rev/pilot/{source_attribution,label_shift}/` | `f66cc62`, `a7b6aa2` | D1 and label-shift correction falsified | DEVELOPMENT | Limitations only |
| Manuscript assets | — | `manuscript_c4ai4c27/scripts/make_assets.py` | `manuscript_c4ai4c27/{figures,tables}/`, `ASSET_PROVENANCE.json` | uncommitted (this prompt) | Fig. 1, Fig. 2, Table 1 | generated | — |
