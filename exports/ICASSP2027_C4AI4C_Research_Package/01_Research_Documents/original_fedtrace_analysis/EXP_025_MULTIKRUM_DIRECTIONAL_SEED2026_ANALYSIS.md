# EXP_025 — Second Directional Attacked Multi-Krum Run: Seed 2026

**Status:** Analysis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run. Multi-Krum's `f=1`/`m=4`, tie-breaking, weighting, buffer handling, and the pre-registered `P_krum=3` rejection proxy were not modified or tuned. FedTrace v3 remains frozen. Multi-Krum's exclusion pattern is referred to throughout as the **"Multi-Krum rejection proxy"**, never as an attribution detector.

Paired experiments used: `EXP_021` (clean Multi-Krum, seed 2026), `EXP_019` (directional-attacked FedAvg, seed 2026, including its already-frozen FedTrace v3 result), `EXP_009` (clean FedAvg, seed 2026). Cross-seed comparison uses `EXP_024` (directional Multi-Krum, seed 123).

---

## 1. Pairing Verification

| Field | EXP_019 (attacked FedAvg) | EXP_025 (attacked Multi-Krum) | EXP_021 (clean Multi-Krum) | Match? |
|---|---|---|---|---|
| `dataset`/`model` | CIFAR-100 / resnet18_cifar | same | same | ✅ |
| `seed` | 2026 | 2026 | 2026 | ✅ |
| `dirichlet_alpha` | 0.1 | 0.1 | 0.1 | ✅ |
| `num_clients`/`clients_per_round` | 5/5 | 5/5 | 5/5 | ✅ |
| `communication_rounds_completed` | 50/50 | 50/50 | 50/50 | ✅ |
| `optimizer`/`learning_rate`/`batch_size` | sgd/0.01/32 | sgd/0.01/32 | sgd/0.01/32 | ✅ |
| `client_sample_counts` | `{000:5233,001:12318,002:9171,003:6556,004:11722}` | **identical, integer-for-integer** | **identical** | ✅ same partition realization throughout |
| `attack_type`/`malicious_client_id`/`attack_start_round`/`directional_target_cosine` | norm_preserving_directional / client_002 / 20 / 0.0 | identical | — | ✅ |
| `aggregation_strategy` | fedavg | **multikrum** | multikrum | **The sole scientific difference between EXP_025 and EXP_019** |

**No discrepancy found.** EXP_025's rounds 0–19 are byte-identical to EXP_021's (round 19 validation Top-1: 38.10% in both), directly confirming the shared partition realization.

## 2. Directional Attack Correctness

`client_002`'s `attack_active` is `False` for `round_id<20` and `True` for `round_id≥20`; no other client's `attack_active` is ever `True` (all 250 rows inspected).

| Field | Min | Max | Max abs. deviation from target |
|---|---|---|---|
| `delta_attack_norm_ratio` (target 1.0) | `0.9999999997870611` (round 40) | `1.0000000001384177` (round 49) | `≈2.1×10⁻¹⁰` |
| `realized_cosine_to_honest` (target 0.0) | `-6.86×10⁻¹⁰` (round 44) | `7.52×10⁻¹⁰` (round 48) | `≈7.5×10⁻¹⁰` |

**All 30 active rounds satisfy both frozen attack invariants to floating-point precision.** The attack is genuinely norm-preserving and correctly targeted throughout — consistent with `EXP_019`'s own already-frozen verification (`analysis/EXP_019_STAGE_B_DIRECTIONAL_SEED2026_VALIDATION.md` §3) and with `EXP_024`'s independent seed-123 confirmation.

## 3. Multi-Krum Diagnostics Audit

`multikrum_diagnostics.csv`: 251 lines (250 data rows). **Integrity — all checks pass:** 50×5=250 rows; ranks 1–5 exactly once per round every round; exactly 4 selected/1 rejected every round; no missing or malformed rows. Rounds 0–19 identical, to the full floating-point digit, to `EXP_021`'s corresponding rounds.

| Client | Total rejections (of 50) | R0–R19 | R20–R49 | Pre-attack rate | Post-attack rate | Longest streak | First rejection | Mean rank pre | Mean rank post |
|---|---|---|---|---|---|---|---|---|---|
| `client_000` | 0 | 0 | 0 | 0% | 0% | 0 | never | 1.00 | 1.00 |
| `client_003` | 0 | 0 | 0 | 0% | 0% | 0 | never | 2.00 | 2.00 |
| `client_002` (malicious) | **0** | 0 | **0** | 0% | **0%** | 0 | never | 3.00 | **3.00** |
| `client_004` | 1 | 1 | 0 | 5% | 0% | 1 (R1) | 1 | 4.02 | 4.00 |
| `client_001` | **49** | 19 | **30** | 95% | **100%** | **48 (R2–R49)** | 0 | 4.98 | 5.00 |

**`client_002`'s rank is exactly 3 in every one of the 50 rounds, with zero exceptions anywhere in the entire run — pre-attack and post-attack alike.** This is even more extreme than `EXP_024`'s seed-123 result, where `client_002` at least moved from rank 2 to rank 3 during rounds 20–22; here there is no rank movement at all, at any point.

**Mean Krum score (`client_002`):** pre-attack ≈103–115 (steady, matching `EXP_021`'s clean trend exactly); post-attack rises from 113.18 (round 20) to 147.56 (round 49) — a real, measurable elevation relative to its own pre-attack trend (and relative to `EXP_021`'s clean-control counterfactual, e.g. round 49: 147.56 here vs. 131.16 in `EXP_021`, a +16.4 difference) — but never remotely close to `client_001`'s post-attack range (≈150–180).

## 4. Critical Replication Question

**Independently confirmed, not assumed:** `EXP_025` reproduces the `EXP_024` phenomenon, and in one respect even more starkly.

- `client_002` rejected post-attack: **0 of 30 (0%)**.
- `client_002` selected post-attack: **30 of 30 (100%)**.
- Rank at round 19: **3**.
- Rank at round 20: **3** (unchanged — unlike `EXP_024`, where the attacker's rank was 2 at round 19/20 before shifting to 3 at round 22, here there is no baseline shift to observe because `client_002` was already rank 3 pre-attack, per the seed-2026 clean pattern established in `EXP_021`).
- Rank trajectory, rounds 20–49: **constant at 3**, no variation.
- Longest rejection streak: **0**.
- Longest selection streak: **50 rounds** (the entire run).

**Classification: A — complete evasion**, an even more absolute case than `EXP_024`'s (zero rank movement at all, versus a small, transient shift at seed 123). This replication is not forced — it is the direct, independent outcome of this run's own diagnostics.

**Honest client `client_001` (this seed's largest partition, 12,318/45,000 = 27.4%) is rejected in 30/30 post-attack rounds** — the same role `client_004` played at seed 123, occupied here by a different client identity but the same structural position (largest-partition client, per Part 10 of `analysis/MULTIKRUM_GAUSSIAN_CROSS_SEED_SYNTHESIS.md`'s already-established fairness pattern).

## 5. Rejection-Target Behavior, Rounds 15–25

| Round | Rejected client | `client_002` rank | `client_002` score | `client_001` (largest-partition) rank | `attack_active` |
|---|---|---|---|---|---|
| 15 | `client_001` | 3 | 111.49 | **5 (rejected)** | False |
| 16 | `client_001` | 3 | 111.59 | **5 (rejected)** | False |
| 17 | `client_001` | 3 | 112.62 | **5 (rejected)** | False |
| 18 | `client_001` | 3 | 111.50 | **5 (rejected)** | False |
| 19 | `client_001` | 3 | 112.26 | **5 (rejected)** | False |
| **20 (attack starts)** | **`client_001`** | **3** | **113.18** | **5 (rejected)** | **True** |
| 21 | `client_001` | 3 | 115.78 | 5 (rejected) | True |
| 22 | `client_001` | 3 | 117.41 | 5 (rejected) | True |
| 23 | `client_001` | 3 | 118.67 | 5 (rejected) | True |
| 24 | `client_001` | 3 | 122.12 | 5 (rejected) | True |
| 25 | `client_001` | 3 | 122.76 | 5 (rejected) | True |

**No regime switch occurs, at any round. `client_002` never becomes rank 5.** This result closely resembles `EXP_024`, structurally identical in every respect that matters (no switch, attacker never rejected, honest largest-partition client permanently rejected).

## 6. Full-Run Rejection Proxy (`P_krum=3`, Not Reset at Round 20)

- **`client_001`:** rejected round 0 (streak 1), selected round 1 (reset), rejected rounds 2–49 (48 consecutive, unbroken). First reaches 3 consecutive at round 4 — **persistently confirmed at round 4**. Onset estimate = round 2.
- **`client_004`:** rejected only once (round 1) — never confirmed.
- **`client_002`:** never rejected at any round — **never confirmed, at any point in the run.**
- **`client_000`, `client_003`:** never rejected — never confirmed.

**Predicted suspicious client: `client_001`** — confirmed round 4. **Incorrect.** Malicious `client_002` is never confirmed. **This exactly replicates `EXP_024`'s outcome, including the identical confirmation round (4) for the honest false positive** — the same structural mechanism (a round-1 crossover artifact plus a rejection pattern stable from round 2 onward) produces the same timing at this second seed too.

## 7. EXP_025 vs. EXP_021 (Attack Effect Under Multi-Krum)

| Metric | EXP_021 (clean MK) | EXP_025 (attacked MK) | Difference (025 − 021) |
|---|---|---|---|
| Final test loss | 2.0016 | 2.5107 | **+0.5091** |
| Test Top-1 | 51.29% | 44.50% | **−6.79 pp** |
| Test Top-5 | 79.62% | 72.26% | **−7.36 pp** |
| Macro precision | 53.86% | 46.78% | **−7.08 pp** |
| Macro recall | 51.30% | 44.50% | **−6.80 pp** |
| Macro F1 | 47.52% | 39.53% | **−7.99 pp** |
| Balanced accuracy | 51.30% | 44.50% | **−6.80 pp** |
| Best validation Top-1 | 51.26% (round 47) | 44.54% (round 48) | **−6.72 pp** |

**Trajectory:** rounds 0–19 byte-identical (round 19 validation Top-1: 38.10% both). At round 20, `EXP_025` immediately drops **4.22 pp** below `EXP_021` (36.24% vs. 40.46%) — an even sharper immediate hit than `EXP_024` showed at seed 123 (−1.34 pp at round 20). The gap persists and settles around **−6 to −7 pp** for the remainder of the run (round 49: 43.98% vs. 50.98%, −7.00 pp). **Classification: immediate degradation, persistent plateau — matching `EXP_024` qualitatively, with an even sharper initial onset-round drop.**

## 8. EXP_025 vs. EXP_019 (Multi-Krum vs. FedAvg Under the Same Attack)

| Metric | EXP_019 (attacked FedAvg) | EXP_025 (attacked Multi-Krum) | Difference (025 − 019) |
|---|---|---|---|
| Final test loss | 1.7139 | 2.5107 | **+0.7969** |
| Test Top-1 | 54.56% | 44.50% | **−10.06 pp** |
| Test Top-5 | 83.87% | 72.26% | **−11.61 pp** |
| Macro precision | 57.76% | 46.78% | **−10.98 pp** |
| Macro recall | 54.55% | 44.50% | **−10.05 pp** |
| Macro F1 | 50.99% | 39.53% | **−11.46 pp** |
| Balanced accuracy | 54.55% | 44.50% | **−10.05 pp** |
| Best validation Top-1 | 54.44% (round 49) | 44.54% (round 48) | **−9.90 pp** |

**Multi-Krum worsens utility relative to FedAvg under this attack — decisively, and by an even larger margin than at seed 123** (−10.06 pp vs. −8.74 pp Top-1). Robust-exclusion behavior (Part 4: complete evasion, 0/30 rejections) is separated here from utility: the exclusion failure alone does not fully explain the gap's size, which — as established mechanistically in `analysis/GAUSSIAN_VS_DIRECTIONAL_MULTIKRUM_SEED123_SYNTHESIS.md` §4 — is compounded by Multi-Krum's unweighted 25% (1/4) averaging of `client_002`'s poisoned update whenever selected (every round here), versus FedAvg's sample-weighted 20.4% (9,171/45,000) — again more relative influence for the attacker under Multi-Krum than under FedAvg, the same mechanism identified at seed 123 (there, 25% vs. 16.0%).

## 9. Seed-2026 Directional 2×2

|  | CLEAN | DIRECTIONAL ATTACK |
|---|---|---|
| **FedAvg** | EXP_009 | EXP_019 |
| **Multi-Krum** | EXP_021 | EXP_025 |

**Top-1:** EXP_009=58.26%, EXP_019=54.56%, EXP_021=51.29%, EXP_025=44.50%.
1. Clean aggregation effect (EXP_021−EXP_009): **−6.97 pp**
2. Attack effect under FedAvg (EXP_019−EXP_009): **−3.70 pp**
3. Attack effect under Multi-Krum (EXP_025−EXP_021): **−6.79 pp**
4. Aggregation effect under attack (EXP_025−EXP_019): **−10.06 pp**

**Top-5:** EXP_009=88.56%, EXP_019=83.87%, EXP_021=79.62%, EXP_025=72.26%.
1. Clean aggregation effect: **−8.94 pp**
2. Attack effect under FedAvg: **−4.69 pp**
3. Attack effect under Multi-Krum: **−7.36 pp**
4. Aggregation effect under attack: **−11.61 pp**

**Comparison with seed 123 (Top-1):** clean-agg effect −6.92 pp (vs. −6.97 pp — nearly identical); attack-under-FedAvg −3.73 pp (vs. −3.70 pp — nearly identical); attack-under-Multi-Krum −5.55 pp (vs. −6.79 pp — same direction, larger at seed 2026); aggregation-under-attack −8.74 pp (vs. −10.06 pp — same direction, larger at seed 2026). **All four effects replicate in sign at both seeds; the two attack-dependent effects (3, 4) are consistently somewhat larger in magnitude at seed 2026.**

---

## Final Report

1. **Files created:** `analysis/EXP_025_MULTIKRUM_DIRECTIONAL_SEED2026_ANALYSIS.md`, `analysis/MULTIKRUM_DIRECTIONAL_CROSS_SEED_SYNTHESIS.md`, `analysis/MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md`.
2. **EXP_025 pairing verdict:** valid — differs from EXP_019 only by `aggregation: fedavg→multikrum`; identical partition to EXP_021.
3. **Attack correctness:** confirmed — `delta_attack_norm_ratio≈1.0` (max deviation ≈2.1×10⁻¹⁰), `realized_cosine_to_honest≈0.0` (max deviation ≈7.5×10⁻¹⁰), all 30 active rounds.
4. **Diagnostics integrity:** pass — 250/250 rows valid.
5. **Pre/post rejection counts:** `client_001`=19 pre/**30 post**; `client_004`=1 pre/0 post; **`client_002`=0 pre/0 post**; `client_000`/`client_003`=0/0.
6. **`client_002` post-attack rejection/selection:** **0/30 rejected (0%), 30/30 selected (100%)** — complete evasion, replicating `EXP_024`.
7. **Attacker rank behavior:** constant rank 3, zero variation, pre- and post-attack alike.
8. **Rejection-switch result:** none — `client_001` remains rejected every round, uninterrupted.
9. **Full-run `P_krum` attribution:** misattributes `client_001` (confirmed round 4, identical round to `EXP_024`'s honest FP); `client_002` never confirmed.
10. **EXP_025 vs. EXP_021:** −6.79 pp Top-1, −7.36 pp Top-5 (attacked worse than clean) — replicates `EXP_024`'s direction, with a sharper immediate onset-round drop.
11. **EXP_025 vs. EXP_019:** −10.06 pp Top-1, −11.61 pp Top-5 (Multi-Krum much worse than FedAvg) — replicates and slightly exceeds `EXP_024`'s gap.
12. **Seed-2026 2×2:** clean-agg −6.97 pp; attack-under-FedAvg −3.70 pp; attack-under-MK −6.79 pp; aggregation-under-attack −10.06 pp — all four effects match seed 123 in sign, attack-dependent effects somewhat larger.
13–24. **See `analysis/MULTIKRUM_DIRECTIONAL_CROSS_SEED_SYNTHESIS.md` and `analysis/MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md` for cross-seed replication classification, Gaussian-vs-directional synthesis across both seeds, FedTrace comparison, fairness synthesis, paper-level claim matrix, external-baseline branch verdict, and the final Multi-Krum external-baseline program synthesis.**
