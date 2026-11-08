# EXP_021 — Clean Multi-Krum Control, Seed 2026: Analysis

**Status:** Analysis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it (this document is the only file written in this half of the task). No experiment was run. Multi-Krum's `f=1`/`m=4`, tie-breaking, weighting, buffer handling, and the pre-registered `P_krum=3` rejection proxy were not modified or tuned. FedTrace v3 remains frozen. Multi-Krum's per-round exclusion pattern is referred to throughout as the **"Multi-Krum rejection proxy"**, never as an attribution detector.

---

## Part 1 — EXP_021 Pairing Verification (vs. EXP_009)

Confirmed directly from each run's own `summary.json`:

| Field | EXP_009 (FedAvg) | EXP_021 (Multi-Krum) | Match? |
|---|---|---|---|
| `seed` | 2026 | 2026 | Yes |
| `dataset` | CIFAR-100 | CIFAR-100 | Yes |
| `partition` | dirichlet | dirichlet | Yes |
| `dirichlet_alpha` (config) | 0.1 | 0.1 | Yes (same config, only `aggregation` changed) |
| `model` | resnet18_cifar | resnet18_cifar | Yes |
| `total_parameters`/`trainable_parameters` | 11,220,132 / 11,220,132 | 11,220,132 / 11,220,132 | Yes |
| `optimizer` | sgd | sgd | Yes |
| `learning_rate` | 0.01 | 0.01 | Yes |
| `local_epochs` | 1 | 1 | Yes |
| `batch_size` | 32 | 32 | Yes |
| `communication_rounds_completed` | 50 | 50 | Yes |
| `client_sample_counts` | `{000:5233, 001:12318, 002:9171, 003:6556, 004:11722}` | `{000:5233, 001:12318, 002:9171, 003:6556, 004:11722}` | **Identical, integer-for-integer** — confirms the same Dirichlet(α=0.1, seed=2026) partition realization (partitioning depends only on `config.seed`/`config.dirichlet_alpha`, never `config.aggregation`) |
| Client participation | all 5 selected & successful, every round (`round_history.csv`) | all 5 selected & successful, every round (`round_history.csv`) | Yes — no failed clients in either run |
| Evaluation settings (`evaluate_every`, validation set) | identical config | identical config | Yes |
| Attack present? | No (config predates Phase 5; no `attack_type` field at all) | No (`"attack": {"attack_type": "none", ...}` explicit) | Both clean |
| `aggregation_strategy` | `fedavg` | `multikrum` | **The sole scientific difference — confirmed** |

**No discrepancy found.** The pairing is scientifically valid; the only scientific difference between EXP_021 and EXP_009 is `aggregation: fedavg → multikrum`.

## Part 2 — EXP_021 Multi-Krum Diagnostics

`multikrum_diagnostics.csv` (`results/federated/cifar100_multikrum_dirichlet_a01_seed2026_50rounds/`) was read in full: 251 lines (1 header + 250 data rows).

**Integrity checks — all pass:**
- 50 rounds present (R0–R49), 5 client rows each = 250 total rows, exactly as expected.
- Every round assigns ranks 1–5 exactly once each, no duplicates or gaps.
- Every round has exactly `m=4` `multikrum_selected=True` rows and exactly 1 `multikrum_selected=False` row (`n-m=f=1`), with no exception across all 50 rounds.
- The rejected client is always the rank-5 client in every round (structurally guaranteed by `m=4` selecting ranks 1–4).
- No missing or malformed rows found in any round.

**Rejection counts and streaks, computed exactly from the CSV:**

| Client | Sample count | Rounds rejected (of 50) | Rejection % | Longest consecutive streak | First rejection round | Mean krum_rank | Median krum_rank |
|---|---|---|---|---|---|---|---|
| `client_000` | 5,233 | 0 | 0% | 0 | never | 1.00 | 1 |
| `client_003` | 6,556 | 0 | 0% | 0 | never | 2.00 | 2 |
| `client_002` | 9,171 | 0 | 0% | 0 | never | 3.00 | 3 |
| `client_004` | 11,722 | 1 | 2% | 1 (round 1, isolated) | round 1 | 4.02 | 4 |
| `client_001` | 12,318 | **49** | **98%** | **48 (rounds 2–49, unbroken)** | **round 0** | 4.98 | 5 |

**Rank stability is total in this run**: `client_000` is rank 1, `client_003` rank 2, `client_002` rank 3 in **every one of the 50 rounds**, with no exception. Only the `client_001`/`client_004` pair swaps — `client_004` briefly occupies rank 5 (rejected) at round 1 only, otherwise `client_001` occupies rank 5 for all 49 remaining rounds. This is structurally the same pattern shape observed in EXP_020 (a stable 3-client core plus one round-1 crossover between the two most-distant clients) — see Part 5 for the cross-seed comparison.

**Persistent-rejection proxy (`P_krum=3`, pre-registered, applied unmodified):**
- `client_001`: rejected round 0 (streak=1, below `P_krum`), selected round 1 (streak resets), then rejected rounds 2, 3, 4 — three consecutive rejected rounds — **persistently confirmed at round 4**. Onset estimate (first round of that confirmed streak) = **round 2**. The streak then continues unbroken through round 49.
- `client_004`: rejected only once (round 1, isolated) — never reaches 3 consecutive rounds. **Not** persistently confirmed.
- `client_000`, `client_002`, `client_003`: never rejected. Not persistently confirmed.

**Because EXP_021 has zero attacker, this persistent confirmation is a clean false positive**, exactly analogous to EXP_020's finding:
- Clean persistent false positive: **Yes.**
- Predicted suspicious client (Multi-Krum rejection proxy): **`client_001`.**
- First anomalous/rejected round: **round 0.**
- Onset estimate: **round 2.**
- Confirmation round: **round 4.**
- Total honest clients persistently flagged: **1** (`client_001` only).

## Part 3 — EXP_021 Clean Performance Cost (vs. EXP_009)

All differences computed directly from `final_test_metrics.json`/`summary.json`, EXP_021 − EXP_009:

| Metric | EXP_009 (FedAvg) | EXP_021 (Multi-Krum) | Difference |
|---|---|---|---|
| Final test loss | 1.4466 | 2.0016 | **+0.5550** (higher/worse) |
| Test Top-1 | 58.26% | 51.29% | **−6.97 pp** |
| Test Top-5 | 88.56% | 79.62% | **−8.94 pp** |
| Macro precision | 62.37% | 53.86% | **−8.51 pp** |
| Macro recall | 58.27% | 51.30% | **−6.97 pp** |
| Macro F1 | 56.41% | 47.52% | **−8.89 pp** |
| Balanced accuracy | 58.27% | 51.30% | **−6.97 pp** |
| Best validation Top-1 | 58.08% (round 49) | 51.26% (round 47) | **−6.82 pp** |

**Round-history trajectory (validation Top-1), not just final metrics:**

| Round | FedAvg | Multi-Krum | Gap |
|---|---|---|---|
| 0 | 1.0% | 1.4% | +0.4 pp |
| 10 | 33.44% | 28.64% | −4.80 pp |
| 20 | 45.36% | 40.46% | −4.90 pp |
| 30 | 52.20% | 45.64% | −6.56 pp |
| 40 | 56.04% | 48.78% | −7.26 pp |
| 49 (final) | 58.08% | 50.98% | −7.10 pp |
| Best | 58.08% (r49) | 51.26% (r47) | −6.82 pp |

The gap **appears early** (already ~4.8–4.9 pp by round 10–20), **grows gradually** through roughly round 30–40, and then **remains stable** at ~7 pp through the end of training — it does not close. (FedAvg's own best validation happens to land on the very last round here, i.e. FedAvg was still slowly improving at round 49; Multi-Krum's best was two rounds earlier and essentially flat/slightly noisy at the end — a minor, non-overinterpreted difference.)

**Clean-performance-cost verdict: SUBSTANTIAL.** A consistent ~6.8–8.9 percentage-point degradation across every accuracy-family metric, plus a ~38% relative increase in test loss, present from early training (round 10) and never closing by round 49. This is quantitatively comparable to (if anything, slightly larger on Top-5/F1 than) EXP_020's seed-123 result.

## Part 4 — Client/Partition Interpretation (Seed 2026)

`client_partition_summary.csv` (identical between EXP_021 and EXP_009, confirming the same partition realization):

| Client | Sample count |
|---|---|
| `client_000` | 5,233 |
| `client_003` | 6,556 |
| `client_002` | 9,171 |
| `client_004` | 11,722 |
| `client_001` | 12,318 |

The persistently rejected client, `client_001`, has the **largest** sample count of all five (12,318). The most consistently trusted client (rank 1 in every round), `client_000`, has the **smallest** sample count (5,233). This is reported strictly as an **association**: the Krum-rank ordering observed in Part 2 (`client_000` < `client_003` < `client_002` < `client_004` < `client_001`, ranks 1→5) matches the sample-count ordering **exactly**, ascending. A plausible (not confirmed) mechanism for this association: every client here trains for exactly `local_epochs=1`, but a larger local partition means more mini-batches — and therefore more local SGD steps — within that one epoch than a smaller partition gets, which would mechanically produce a larger-magnitude parameter update in Euclidean space, independent of any actual anomalous or malicious behavior. This mechanism was not directly measured in this task (e.g. no separate per-client raw update-norm log was inspected here) and is offered as a plausible explanation for the association, not a proven causal claim. See Part 5 for whether this same association reproduces at the other tested seed.

## Part 5 — See `analysis/MULTIKRUM_CLEAN_CONTROL_CROSS_SEED_SYNTHESIS.md`

The cross-seed synthesis comparing EXP_020+EXP_021 is reported in the separate synthesis document, per this task's file-creation instructions.

---

## Summary Table (This Experiment Only)

| Item | Value |
|---|---|
| Pairing valid | Yes |
| Diagnostics integrity | Pass (250/250 rows, no anomalies) |
| Persistently rejected client | `client_001` |
| Rejection count | 49/50 (98%) |
| Longest streak | 48 rounds (R2–R49) |
| Clean persistent false positive | Yes (onset R2, confirmed R4) |
| Test Top-1 clean cost | −6.97 pp |
| Test Top-5 clean cost | −8.94 pp |
| Best-validation clean cost | −6.82 pp |
| Clean-performance-cost verdict | Substantial |
