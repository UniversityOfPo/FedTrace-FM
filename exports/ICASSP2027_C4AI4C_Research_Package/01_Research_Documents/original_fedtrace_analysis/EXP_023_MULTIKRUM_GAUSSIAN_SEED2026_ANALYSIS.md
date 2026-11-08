# EXP_023 — Second Attacked Multi-Krum Run: Gaussian Attack, Seed 2026

**Status:** Analysis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run. Multi-Krum's `f=1`/`m=4`, tie-breaking, weighting, buffer handling, and the pre-registered `P_krum=3` rejection proxy were not modified or tuned. FedTrace v3 remains frozen. Multi-Krum's exclusion pattern is referred to throughout as the **"Multi-Krum rejection proxy"**, never as an attribution detector.

Paired experiments used: `EXP_021` (clean Multi-Krum, seed 2026), `EXP_017` (Gaussian-attacked FedAvg, seed 2026, including its already-frozen FedTrace v3 result from `analysis/EXP_017_STAGE_B_GAUSSIAN_SEED2026_VALIDATION.md`), `EXP_009` (clean FedAvg, seed 2026, for the 2×2 in Part 7). Cross-seed comparison uses `EXP_022`/`EXP_020` (seed 123).

---

## Part 1 — Pairing Verification

| Field | EXP_017 (attacked FedAvg) | EXP_023 (attacked Multi-Krum) | EXP_021 (clean Multi-Krum) | Match? |
|---|---|---|---|---|
| `dataset` | CIFAR-100 | CIFAR-100 | CIFAR-100 | ✅ |
| `model` | resnet18_cifar | resnet18_cifar | resnet18_cifar | ✅ |
| `seed` | 2026 | 2026 | 2026 | ✅ |
| `dirichlet_alpha` | 0.1 | 0.1 | 0.1 | ✅ |
| `num_clients`/`clients_per_round` | 5/5 | 5/5 | 5/5 | ✅ |
| `communication_rounds_completed` | 50/50 | 50/50 | 50/50 | ✅ |
| `local_epochs` | 1 | 1 | — | ✅ |
| `optimizer`/`learning_rate`/`batch_size` | sgd/0.01/32 | sgd/0.01/32 | sgd/0.01/32 | ✅ |
| `client_sample_counts` | `{000:5233,001:12318,002:9171,003:6556,004:11722}` | **identical, integer-for-integer** | **identical, integer-for-integer** | ✅ same partition realization across all three runs |
| `malicious_client_id` | client_002 | client_002 | — | ✅ |
| `attack_start_round` | 20 | 20 | — | ✅ |
| `attack_strength_rho` | 1.0 | 1.0 | — | ✅ |
| `aggregation_strategy` | fedavg | **multikrum** | multikrum | **The sole scientific difference between EXP_023 and EXP_017** |

**No discrepancy found.** EXP_023 differs from EXP_017 scientifically only by `aggregation: fedavg → multikrum`. EXP_023 shares an identical partition realization with EXP_021 — confirmed both by sample counts and, more strongly, by EXP_023's own rounds 0–19 being byte-identical to EXP_021's (verified in Part 2).

## Part 2 — Multi-Krum Diagnostics Audit

`multikrum_diagnostics.csv` read in full: 251 lines (1 header + 250 data rows). **Integrity — all checks pass:** 50 rounds × 5 clients = 250 rows; ranks 1–5 assigned exactly once per round in every round; exactly `m=4` selected/1 rejected every round with no exception; no missing or malformed rows.

**Determinism cross-check:** rounds 0–19 are identical, to the full floating-point digit, to `EXP_021`'s corresponding rounds (e.g. round 0: `client_001` score `202.77658040864628`, rank 5, in both; round 19: `client_001` score `150.08138462965343`, rank 5, in both) — as expected, since the attack is inactive before round 20.

**Preliminary observations, verified exactly, not assumed:**
- **"Pre-attack: `client_001` appears rejected 19/20 rounds."** **Confirmed exactly** — `client_001` is rejected at round 0 and rounds 2–19 (19 rounds); `client_004` is rejected only at round 1; `client_000`, `client_002`, `client_003` are never rejected pre-attack. 19 + 1 = 20.
- **"Post-attack: `client_002` appears rejected 30/30 rounds."** **Confirmed exactly** — every one of the 30 rows for `client_002` from round 20 through round 49 shows `krum_rank=5`, `multikrum_selected=False`, no exception.

**Full rejection/rank statistics (exact, computed from the CSV):**

| Client | Total rejections (of 50) | Rejections R0–R19 | Rejections R20–R49 | Pre-attack rate | Post-attack rate | Longest streak | First rejection round | Mean rank pre | Mean rank post | Median rank pre | Median rank post |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `client_000` | 0 | 0 | 0 | 0% | 0% | 0 | never | 1.00 | 1.00 | 1 | 1 |
| `client_003` | 0 | 0 | 0 | 0% | 0% | 0 | never | 2.00 | 2.00 | 2 | 2 |
| `client_002` (malicious) | **30** | 0 | **30** | 0% | **100%** | **30 (R20–R49, unbroken)** | **20** | 3.00 | **5.00** | 3 | 5 |
| `client_004` | 1 | 1 | 0 | 5% | 0% | 1 (R1, isolated) | 1 | 4.05 | 3.00 | 4 | 3 |
| `client_001` | 19 | 19 | 0 | 95% | **0%** | 18 (R2–R19) | 0 | 4.95 | 4.00 | 5 | 4 |

## Part 3 — Regime Switch (Rounds 15–25)

| Round | Rejected client | `client_002` rank | `client_002` score | `client_001` rank | `client_001` score |
|---|---|---|---|---|---|
| 15 | `client_001` | 3 | 111.49 | **5 (rejected)** | 146.60 |
| 16 | `client_001` | 3 | 111.59 | **5 (rejected)** | 147.62 |
| 17 | `client_001` | 3 | 112.62 | **5 (rejected)** | 149.70 |
| 18 | `client_001` | 3 | 111.50 | **5 (rejected)** | 148.14 |
| 19 | `client_001` | 3 | 112.26 | **5 (rejected)** | 150.08 |
| **20 (attack starts)** | **`client_002`** | **5 (rejected)** | **183.52** | **4** | **150.10** |
| 21 | `client_002` | 5 (rejected) | 187.93 | 4 | 144.21 |
| 22 | `client_002` | 5 (rejected) | 188.16 | 4 | 143.33 |
| 23 | `client_002` | 5 (rejected) | 189.50 | 4 | 141.58 |
| 24 | `client_002` | 5 (rejected) | 194.91 | 4 | 143.64 |
| 25 | `client_002` | 5 (rejected) | 195.25 | 4 | 141.69 |

**The switch is again instantaneous — one round, zero lag.** At round 19, `client_002` sits at rank 3 (score 112.26), comfortably selected; at round 20, its score jumps to 183.52 (a **+71.26, +63.5% single-round increase** — a larger relative jump than seed 123's +48.7%) and it becomes the rejected client, while `client_001`'s own score barely moves (150.08→150.10) but is no longer the worst because `client_002` now scores far higher.

**Attack onset round = 20. Rejection-switch round = 20. Lag = 0.** This exactly replicates `EXP_022`'s seed-123 finding: "heterogeneity-dominated clean rejection → attack-dominated rejection regime switch," with zero transition delay, **at this second seed too**. This is reported as a two-seed observation, not claimed to be universal.

## Part 4 — Full-Run Rejection Proxy (`P_krum=3`, Not Reset at Round 20)

Applying the frozen rule to the complete 50-round run, with no knowledge of the true attack onset used in scoring:

- **`client_001`:** rejected round 0 (streak 1), selected round 1 (reset), rejected rounds 2–19 (18 consecutive). First reaches 3 consecutive at round 4 (rounds 2/3/4) — **persistently confirmed at round 4**. Onset estimate = round 2.
- **`client_004`:** rejected only once (round 1) — never persistently confirmed.
- **`client_002`:** rejected rounds 20–49. First reaches 3 consecutive at round 22 (rounds 20/21/22) — **persistently confirmed at round 22**. Onset estimate = round 20 (exact).
- **`client_000`, `client_003`:** never rejected — never confirmed.

**Predicted suspicious client over the full run (earliest confirmed, per the proxy's own rule): `client_001`** — confirmed at round 4, **18 rounds before** the true attacker `client_002`'s own confirmation (round 22). **This is incorrect** — `client_001` is honest. `client_002` itself is only confirmed at round 22.

**This exactly replicates `EXP_022`'s seed-123 misattribution**, including the identical confirmation rounds (round 4 for the honest false positive, round 22 for the true attacker, in both seeds) and the identical 18-round margin. This numerical exactness is not a coincidence specific to either seed's raw scores — it follows structurally from two facts that held at both seeds: the pre-attack rejection pattern stabilizes by round 2 (after an identical round-1 crossover artifact in both runs), so `P_krum=3` confirms at round 4 in both; and the attack causes a zero-lag rejection switch at round 20 in both, so `P_krum=3` confirms the true attacker at round 22 in both. **Given those two structural facts (both independently confirmed by direct measurement, not assumed), round 4 and round 22 are the mechanically determined confirmation rounds regardless of the specific score values — this is a genuine two-seed replication of the *mechanism*, not merely a coincidental match of two numbers.**

## Part 5 — EXP_023 vs. EXP_021 (Attacked vs. Clean Multi-Krum)

| Metric | EXP_021 (clean MK) | EXP_023 (attacked MK) | Difference (023 − 021) |
|---|---|---|---|
| Final test loss | 2.0016 | 1.6749 | **−0.3267** (lower/better) |
| Test Top-1 | 51.29% | 54.78% | **+3.49 pp** |
| Test Top-5 | 79.62% | 84.27% | **+4.65 pp** |
| Macro precision | 53.86% | 58.45% | **+4.59 pp** |
| Macro recall | 51.30% | 54.77% | **+3.47 pp** |
| Macro F1 | 47.52% | 51.78% | **+4.26 pp** |
| Balanced accuracy | 51.30% | 54.77% | **+3.47 pp** |
| Best validation Top-1 | 51.26% (round 47) | 54.72% (round 49) | **+3.46 pp** |

**EXP_023 again outperforms EXP_021 on every metric — replicating `EXP_022`'s counter-intuitive seed-123 result.** Reported exactly as found; no causal "attack improves learning" claim is made.

**Trajectory shape differs from seed 123 in an instructive way.** Round 0–19 validation Top-1 is byte-identical between `EXP_023` and `EXP_021` (e.g. round 19: 38.10% in both). But unlike seed 123 (where `EXP_022` was already ahead of `EXP_020` at round 20 itself), at seed 2026 **`EXP_023` briefly *trails* `EXP_021` right around attack onset**: round 20, 38.44% vs. 40.46% (**−2.02 pp**); round 25, 43.68% vs. 44.02% (−0.34 pp) — before crossing over and pulling ahead from round 30 onward (46.10% vs. 44.70%, +1.40 pp) and widening to +3.7–4.7 pp by the end of the run. **The final-metric advantage replicates; the exact shape of how it emerges does not** — at seed 123 the gain is immediate and monotonic from round 20; at seed 2026 there is a brief 5–6 round dip before the gain appears. This is reported as a genuine difference between the two runs, not smoothed over.

**Hypothesis (unchanged from `EXP_022`'s analysis, not re-derived, offered again as the most parsimonious explanation):** the rejection target switches from `client_001` (this seed's largest partition, 12,318/45,000 = 27.4% of all data) to `client_002` (a smaller, 9,171-sample partition) at round 20 — restoring `client_001`'s previously-excluded, larger local signal to the aggregate for the run's second half, plausibly explaining the net improvement independent of the attack's content being informative in any way. **Not directly demonstrated in this task.**

## Part 6 — EXP_023 vs. EXP_017 (Attacked Multi-Krum vs. Attacked FedAvg)

| Metric | EXP_017 (attacked FedAvg) | EXP_023 (attacked Multi-Krum) | Difference (023 − 017) |
|---|---|---|---|
| Final test loss | 1.4764 | 1.6749 | **+0.1985** (worse) |
| Test Top-1 | 58.08% | 54.78% | **−3.30 pp** |
| Test Top-5 | 88.07% | 84.27% | **−3.80 pp** |
| Macro precision | 63.81% | 58.45% | **−5.36 pp** |
| Macro recall | 58.08% | 54.77% | **−3.31 pp** |
| Macro F1 | 56.24% | 51.78% | **−4.46 pp** |
| Balanced accuracy | 58.08% | 54.77% | **−3.31 pp** |
| Best validation Top-1 | 58.18% (round 48) | 54.72% (round 49) | **−3.46 pp** |

**Multi-Krum's perfect malicious-client rejection again translates into *worse*, not better, net utility than simply using FedAvg — replicating `EXP_022`'s seed-123 finding.** The magnitude is somewhat larger here on Top-1 (−3.30 pp vs. −2.07 pp at seed 123) and comparable on macro precision (−5.36 pp vs. −5.45 pp). Per `EXP_017`'s own already-frozen result (`analysis/EXP_017_STAGE_B_GAUSSIAN_SEED2026_VALIDATION.md` §9), the Gaussian attack is again "essentially neutral" against FedAvg at this seed too (−0.18 pp Top-1, −0.49 pp Top-5 relative to `EXP_009`) — so, exactly as at seed 123, there is very little attack damage for Multi-Krum's exclusion to have prevented, and Multi-Krum's own structural clean-performance cost dominates the comparison.

## Part 7 — Complete Seed-2026 2×2

|  | CLEAN | GAUSSIAN ATTACK |
|---|---|---|
| **FedAvg** | EXP_009 | EXP_017 |
| **Multi-Krum** | EXP_021 | EXP_023 |

**Top-1:** EXP_009=58.26%, EXP_017=58.08%, EXP_021=51.29%, EXP_023=54.78%.
1. Clean aggregation effect (EXP_021−EXP_009): **−6.97 pp**
2. Attack effect under FedAvg (EXP_017−EXP_009): **−0.18 pp**
3. Attack effect under Multi-Krum (EXP_023−EXP_021): **+3.49 pp**
4. Aggregation effect under attack (EXP_023−EXP_017): **−3.30 pp**

**Top-5:** EXP_009=88.56%, EXP_017=88.07%, EXP_021=79.62%, EXP_023=84.27%.
1. Clean aggregation effect: **−8.94 pp**
2. Attack effect under FedAvg: **−0.49 pp**
3. Attack effect under Multi-Krum: **+4.65 pp**
4. Aggregation effect under attack: **−3.80 pp**

**Comparison to the seed-123 2×2 (`analysis/EXP_022_MULTIKRUM_GAUSSIAN_SEED123_ANALYSIS.md` §F), Top-1:** clean aggregation effect −6.92 pp (vs. −6.97 pp here — nearly identical); attack-under-FedAvg −0.33 pp (vs. −0.18 pp — both small/near-neutral); attack-under-Multi-Krum +4.52 pp (vs. +3.49 pp — same direction, broadly similar magnitude); aggregation-under-attack −2.07 pp (vs. −3.30 pp — same direction, somewhat larger at seed 2026). **All four effects replicate in direction at both seeds; magnitudes are broadly, not exactly, similar. No claim of statistical significance is made from two seeds.**

---

## Final Report

1. **Files created:** `analysis/EXP_023_MULTIKRUM_GAUSSIAN_SEED2026_ANALYSIS.md`, `analysis/MULTIKRUM_GAUSSIAN_CROSS_SEED_SYNTHESIS.md`.
2. **EXP_023 pairing verdict:** valid — differs from EXP_017 only by `aggregation: fedavg→multikrum`; identical partition to EXP_021 (sample counts and byte-identical pre-attack diagnostics/trajectories through round 19).
3. **Diagnostics integrity:** pass — 250/250 rows, fully structurally valid, no anomalies.
4. **Pre/post rejection counts:** `client_001`=19 pre/0 post; `client_004`=1 pre/0 post; **`client_002`=0 pre/30 post (100%)**; `client_000`/`client_003`=0/0.
5. **Regime-switch result:** instantaneous, zero-lag, at round 20 exactly — replicates EXP_022.
6. **Full-run rejection-proxy result:** misattributes `client_001` (confirmed round 4), 18 rounds before `client_002`'s own confirmation (round 22) — replicates EXP_022's misattribution exactly, including the confirmation rounds.
7. **EXP_023 vs. EXP_021:** EXP_023 outperforms EXP_021 on every metric (+3.49 pp Top-1, +4.65 pp Top-5) — replicates EXP_022's direction; trajectory shape differs (brief dip near onset before pulling ahead, vs. immediate gain at seed 123).
8. **EXP_023 vs. EXP_017:** EXP_023 underperforms EXP_017 on every metric (−3.30 pp Top-1, −3.80 pp Top-5) — replicates EXP_022's finding.
9. **Seed-2026 2×2:** clean aggregation cost −6.97 pp Top-1; attack-under-FedAvg −0.18 pp; attack-under-Multi-Krum +3.49 pp; aggregation-under-attack −3.30 pp — same directions as seed 123, broadly similar magnitudes.
10–19. **See `analysis/MULTIKRUM_GAUSSIAN_CROSS_SEED_SYNTHESIS.md` for the full cross-seed replication classification, FedTrace comparison, fairness synthesis, paper-level verdicts, limitations, and next-gate decision.**
