# EXP_012 — FedProx vs. FedAvg, Paired Seed 2026 Analysis (Dirichlet α=0.1, μ=0.01)

**Analysis date:** 2026-08-07
**Subject experiments (paired, seed 2026 only):**
- `EXP_009` — FedAvg, Dirichlet α=0.1, 50 rounds, seed 2026
- `EXP_012` — FedProx (μ=0.01), Dirichlet α=0.1, 50 rounds, seed 2026

**Data sources (verified, no invented or estimated values):**
- `results/federated/cifar100_fedavg_dirichlet_a01_seed2026_50rounds/summary.json`,
  `final_test_metrics.json`, `client_partition_summary.csv`
- `results/federated/cifar100_fedprox_dirichlet_a01_mu001_seed2026_50rounds/summary.json`,
  `final_test_metrics.json`, `client_partition_summary.csv`

This document performs analysis only. No code, config, test, README, or
Research Database file was created or modified in the process of producing
it.

---

## 1. Pairing Check (from result files)

Both runs' `client_partition_summary.csv` report identical per-client
sample counts: `client_000`: 5,233; `client_001`: 12,318; `client_002`:
9,171; `client_003`: 6,556; `client_004`: 11,722. Both `summary.json`
files report `seed=2026`, `dataset="CIFAR-100"`, `partition="dirichlet"`,
`num_clients=5`, `clients_per_round=5`, `communication_rounds_completed=50`,
`local_epochs=1`, `aggregation_strategy="fedavg"`, `learning_rate=0.01`,
`total_parameters=11,220,132`.

---

## 2. Reported Metrics

| Metric | EXP_009 (FedAvg) | EXP_012 (FedProx, μ=0.01) |
|---|---:|---:|
| Final test loss | 1.446613697052002 | 1.473337201499939 |
| Test top-1 | 58.26% | 57.63% |
| Test top-5 | 88.56% | 87.74% |
| Macro precision | 0.6237342308141861 | 0.6216831240921957 |
| Macro recall | 0.5827 | 0.5763 |
| Macro F1 | 0.5641049541589388 | 0.554339915901785 |
| Balanced accuracy | 0.5827 | 0.5763 |
| Best validation top-1 | 58.08% | 57.22% |
| Best validation round | 49 | 49 |

---

## 3. Paired Differences (FedProx − FedAvg)

| Metric | Δ (EXP_012 − EXP_009) |
|---|---:|
| Final test loss | +0.026723504447937 |
| Test top-1 | −0.63pp |
| Test top-5 | −0.82pp |
| Macro precision | −0.0020511067219904 |
| Macro recall | −0.64pp |
| Macro F1 | −0.0097650382571538 |
| Balanced accuracy | −0.64pp |
| Best validation top-1 | −0.86pp |
| Best validation round | 0 (both round 49) |

---

## 4. Direction Comparison with Seed 42 and Seed 123

Directions below use the same sign convention as this document's §3
(FedProx − FedAvg), and reuse the paired differences already reported in
`analysis/EXP_010_FedProx_Seed42_Analysis.md` (§§2–3) and
`analysis/EXP_011_FedProx_Seed123_Analysis.md` (§§2–3).

| Metric | Seed 42 direction | Seed 123 direction | Seed 2026 direction | Agrees with 42? | Agrees with 123? |
|---|---|---|---|---|---|
| Final test loss | lower (better) | higher (worse) | higher (worse) | Disagrees | Agrees |
| Test top-1 | lower (worse) | higher (better) | lower (worse) | Agrees | Disagrees |
| Test top-5 | lower (worse) | lower (worse) | lower (worse) | Agrees | Agrees |
| Macro precision | lower (worse) | lower (worse) | lower (worse) | Agrees | Agrees |
| Macro recall | lower (worse) | higher (better) | lower (worse) | Agrees | Disagrees |
| Macro F1 | lower (worse) | higher (better) | lower (worse) | Agrees | Disagrees |
| Balanced accuracy | lower (worse) | higher (better) | lower (worse) | Agrees | Disagrees |
| Best validation top-1 | lower (worse) | lower (worse) | lower (worse) | Agrees | Agrees |
| Best validation round | same (49 vs. 49) | different (47 vs. 49) | same (49 vs. 49) | Agrees (same) | Disagrees (differs from 123's "different") |

This document intentionally does not draw any overall conclusion across the three seeds. That synthesis is reserved for the next analysis.
