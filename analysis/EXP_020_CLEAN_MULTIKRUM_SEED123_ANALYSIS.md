# EXP_020 — Clean Multi-Krum Control, Seed 123: Analysis

**Status:** Analysis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it (this document is the only file written). No experiment was run. Multi-Krum's `f=1`/`m=4` were not tuned. FedTrace v3 remains frozen and untouched.

---

## 1. Verify Pairing (EXP_020 vs. EXP_008)

Confirmed identical on every field that must match, drawn directly from each run's own `summary.json` (not re-derived or assumed):

| Field | EXP_008 (FedAvg) | EXP_020 (Multi-Krum) | Match? |
|---|---|---|---|
| `seed` | 123 | 123 | Yes |
| `partition` | dirichlet | dirichlet | Yes |
| `dirichlet_alpha` (config) | 0.1 | 0.1 | Yes (same config file, only `aggregation` changed — verified in the prior task's config-creation report) |
| `model` | resnet18_cifar | resnet18_cifar | Yes |
| `total_parameters` / `trainable_parameters` | 11,220,132 / 11,220,132 | 11,220,132 / 11,220,132 | Yes |
| `optimizer` | sgd | sgd | Yes |
| `learning_rate` | 0.01 | 0.01 | Yes |
| `batch_size` | 32 | 32 | Yes |
| `communication_rounds_completed` | 50 | 50 | Yes |
| `local_epochs` | 1 | 1 | Yes |
| `client_sample_counts` | `{000:6024, 001:9628, 002:7202, 003:10361, 004:11785}` | `{000:6024, 001:9628, 002:7202, 003:10361, 004:11785}` | **Identical, integer-for-integer** — confirms the same Dirichlet(α=0.1, seed=123) partition realization, since `select_partitions`/`partition_dataset_dirichlet` depends only on `config.seed`/`config.dirichlet_alpha`/the dataset, never on `config.aggregation` |
| `client_partition_summary.csv` | same 5 rows | same 5 rows | Identical |
| Client participation per round | all 5 clients selected & successful, every round (`round_history.csv`) | all 5 clients selected & successful, every round (`round_history.csv`) | Yes — `clients_per_round=5=num_clients` in both, no failed clients in either |
| Attack present? | No (`attack_type` field absent from EXP_008's config entirely — it predates Phase 5) | No (`"attack": {"attack_type": "none", ...}` explicit in `summary.json`) | Both clean; equivalent, just expressed by two different code eras (see note below) |
| `aggregation_strategy` | `fedavg` | `multikrum` | **The sole scientific difference — confirmed** |

**Note on provenance:** EXP_008's `git_commit` (`1768aaf5c...`) differs from EXP_020's (`12d030a7a...`) — EXP_008 was run before the Phase-5 attack infrastructure and Multi-Krum code existed, which is why its `summary.json` has no `attack` block at all (the field didn't exist yet) rather than an explicit `"none"`. This is a benign provenance artifact, not a scientific confound: the attack and Multi-Krum additions are both purely additive (confirmed in `analysis/MULTIKRUM_IMPLEMENTATION_VERIFICATION.md` Section F), so nothing about data loading, partitioning, model architecture, or FedAvg's own behavior changed between the two commits in a way that would affect this specific FedAvg-vs-Multi-Krum comparison.

**Conclusion: pairing is valid.** The sole scientific difference between EXP_020 and EXP_008 is `aggregation: fedavg → multikrum`, exactly as required.

## 2. Verify Multi-Krum Integration

`multikrum_diagnostics.csv` exists at `results/federated/cifar100_multikrum_dirichlet_a01_seed123_50rounds/multikrum_diagnostics.csv` and was read in full (251 lines: 1 header + 250 data rows).

- **250 data rows = 50 rounds × 5 clients, exactly** — no missing or extra rows for any round.
- **Every round has exactly 5 client rows** (`client_000`–`client_004`), confirmed for all 50 rounds (R0–R49).
- **Every round assigns ranks 1–5 exactly once each** (no ties, no duplicates, no gaps) — consistent with the algorithm's deterministic `(score, client_id)` total ordering.
- **Every round has exactly 4 `multikrum_selected=True` rows and 1 `multikrum_selected=False` row** — consistent with `m=4`, `f=1`(`=n-m`), `n=5` for all 50 rounds, with no exception.
- **The rejected client is always the rank-5 client** in every round (structurally guaranteed: `m=4` selects ranks 1–4) — confirmed with no counterexample.
- `f`, `m`, `neighbors_used` are not persisted as explicit per-row columns in `multikrum_diagnostics.csv` (only `round_id, client_id, krum_score, krum_rank, multikrum_selected, multikrum_rejected` are, matching the schema `save_multikrum_diagnostics_csv` was designed to write). Their values are indirectly but conclusively confirmed by the structural counts above (`m=4` from the 4-True-per-round count, `f=1` from `n-m=1` rejected per round, `n=5` from the row count) and directly by the already-audited source code (`_MULTIKRUM_F=1` in `run_federated_cifar100.py`; `neighbors_used=n-f-2=2` computed inside `multikrum.py`, unit-tested in `tests/test_multikrum_aggregation.py`). `neighbors_used` is not independently re-derivable from this CSV alone (raw per-client update deltas are not logged, by design — the same trace-only limitation already documented in `research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` §7), so this analysis relies on the already-verified implementation rather than re-deriving the score formula from scratch.

**No malformed or missing diagnostic rows were found across all 50 rounds.**

## 3. Clean Rejection Behavior

Because EXP_020 has no attacker, every client is honest by construction. Exact counts from `multikrum_diagnostics.csv`:

| Client | Sample count | Rounds rejected (of 50) | Rejection % | Longest consecutive rejection streak | First rejection round | Mean krum_rank | Median krum_rank |
|---|---|---|---|---|---|---|---|
| `client_000` | 6,024 | 0 | 0% | 0 | never | 1.00 | 1 |
| `client_001` | 9,628 | 0 | 0% | 0 | never | 2.98 | 3 |
| `client_002` | 7,202 | 0 | 0% | 0 | never | 2.02 | 2 |
| `client_003` | 10,361 | 1 | 2% | 1 (round 1, isolated) | round 1 | 4.02 | 4 |
| `client_004` | 11,785 | **49** | **98%** | **48 (rounds 2–49, unbroken)** | **round 0** | 4.98 | 5 |

(`client_001`'s and `client_002`'s ranks are 3 and 2 respectively in 49/50 rounds, swapping to 2/3 once, at round 47 — a single, isolated one-round crossover between the two closest-to-median clients that never changes who is selected; not scientifically significant.)

Mean Krum scores were not hand-summed to full numerical precision across all 250 values (a manual arithmetic error risk not worth taking for a number that isn't decision-relevant); instead, the steady-state per-round score **range** (excluding round 0, an initial-round outlier common to all five clients before local training has meaningfully diverged) is reported, which is what actually establishes the separation:

- `client_000`: ≈97–124 (rank 1 throughout)
- `client_001`: ≈111–135
- `client_002`: ≈103–135
- `client_003`: ≈125–159
- `client_004`: ≈132–193 — **consistently the highest of all five every single round**, and the gap to the next-highest client (`client_003`) widens over the run (from roughly +7 at round 4 to roughly +33 by round 49)

**Determination: rejection is systematically concentrated on a single honest client (`client_004`), not roughly distributed across clients.** Four of five clients (`client_000`–`client_002`, and `client_003` for all but one round) are essentially never rejected; one client absorbs 49 of the 50 round-level rejections, including one unbroken 48-round streak spanning nearly the entire second half of training.

## 4. Persistent-Rejection Proxy (Pre-Registered, Applied Exactly, `P_krum=3`)

Applying the plan's Section 11 definition without modification:

- **`client_004`**: rejected round 0 (streak = 1, below `P_krum`), selected round 1 (streak resets to 0), then rejected rounds 2, 3, 4 — three consecutive rejected rounds — **persistently confirmed at round 4**. Onset estimate (first round of that confirmed streak) = **round 2**. The rejection streak then continues, unbroken, through round 49 (46 further consecutive confirmed rounds) — `client_004` remains persistently confirmed for the rest of the run.
- **`client_003`**: rejected only once (round 1, isolated) — never reaches 3 consecutive rejected rounds. **Not** persistently confirmed.
- **`client_000`, `client_001`, `client_002`**: never rejected. Not persistently confirmed.

**Because EXP_020 has zero attacker, this persistent confirmation is unambiguously a false positive.**

- Whether any honest client is persistently confirmed: **Yes.**
- Which client: **`client_004`.**
- Onset round: **2.**
- Confirmation round: **4.**
- Number of persistent false-positive clients: **1** (`client_004` only — `client_003`'s single-round rejection correctly does not meet the `P_krum=3` bar).

This is a direct, empirical confirmation of the fairness risk pre-registered in `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` Section 6: under this project's strong Non-IID condition (α=0.1), Multi-Krum's distance-based selection produces a sustained false attribution against a genuinely honest client, with no attacker present at all. Per that plan's Section 15 ("scientific failure ≠ implementation failure") and this task's explicit instruction, this result is retained exactly as observed, not adjusted, and `P_krum` was not tuned.

## 5. Client Heterogeneity Relation (Association Only, No Causal Claim)

Comparing rejection behavior against `client_partition_summary.csv`'s sample counts (the only per-client partition statistic available in the artifacts read for this task — a class-distribution breakdown would require additional computation against `client_partition_indices.json` and the CIFAR-100 label array that was not performed here):

| Client | Sample count | Rank by sample count (1=fewest) | Rejected? |
|---|---|---|---|
| `client_000` | 6,024 | 1 (fewest) | Never |
| `client_002` | 7,202 | 2 | Never |
| `client_001` | 9,628 | 3 | Never |
| `client_003` | 10,361 | 4 | Once |
| `client_004` | 11,785 | 5 (most) | 49/50 rounds |

The persistently-rejected client (`client_004`) has the **largest** sample count of all five, and the client with the **smallest** sample count (`client_000`) is the one Multi-Krum trusts most consistently (rank 1 in every round). This is the opposite of what a "less data → noisier, more divergent update → more likely rejected" hypothesis would predict, and is reported here strictly as an **observed association, not a causal claim** — sample count alone does not explain the ranking (`client_001` and `client_003` do not follow a monotonic sample-count-to-rank relationship: `client_001` has fewer samples than `client_003` but a *better* rank). The more directly supported explanation, visible from the score data itself (Section 3), is simply that `client_004`'s locally-trained update is, under this specific Dirichlet(α=0.1, seed=123) partition, consistently the most geometrically distant from the other four in Euclidean update-space — a property of *this client's particular partition realization* (likely its local class-distribution skew, not measured directly here) interacting with Krum's distance metric, not a property of dataset size.

## 6. Clean Performance Cost (EXP_020 − EXP_008)

All differences computed directly from each run's `final_test_metrics.json`/`summary.json`, in percentage points (pp) for accuracy-type metrics:

| Metric | EXP_008 (FedAvg) | EXP_020 (Multi-Krum) | Difference |
|---|---|---|---|
| Final test loss | 1.4654 | 2.1901 | **+0.7247** (higher/worse) |
| Test Top-1 | 56.83% | 49.91% | **−6.92 pp** |
| Test Top-5 | 87.79% | 80.06% | **−7.73 pp** |
| Macro precision | 63.38% | 54.31% | **−9.07 pp** |
| Macro recall | 56.83% | 49.91% | **−6.92 pp** |
| Macro F1 | 54.86% | 46.16% | **−8.70 pp** |
| Balanced accuracy | 56.83% | 49.91% | **−6.92 pp** |
| Best validation Top-1 | 58.26% (round 47) | 51.44% (round 48) | **−6.82 pp** |
| Best validation round | 47 | 48 | +1 round (negligible) |

**This is the clean cost of switching FedAvg → Multi-Krum under the identical seed-123, α=0.1, 50-round setting: a consistent ~6.8–9.1 percentage-point degradation across every accuracy-family metric, plus a ~49% relative increase in final test loss.** This is not a marginal or noise-level effect — every metric moves in the same direction by a comparable, substantial magnitude.

## 7. Trajectory Comparison (Validation Top-1, `round_history.csv`)

| Round | FedAvg val_top1 | Multi-Krum val_top1 | Gap |
|---|---|---|---|
| 0 | 1.32% | 1.12% | −0.20 pp |
| 10 | 32.40% | 28.40% | −4.00 pp |
| 20 | 46.08% | 39.86% | −6.22 pp |
| 30 | 51.92% | 44.70% | −7.22 pp |
| 40 | 55.90% | 48.22% | −7.68 pp |
| 49 (final) | 57.78% | 50.44% | −7.34 pp |
| Best | 58.26% (r47) | 51.44% (r48) | −6.82 pp |

- **Converges more slowly:** yes — a clear, non-trivial gap (~4 pp) is already visible by round 10, well before either run approaches its ceiling.
- **Reaches a lower ceiling:** yes — the gap widens through roughly round 30 and then stabilizes in the ~6.8–7.7 pp range for the remainder of the run; it does not close as training continues, consistent with a persistent per-round information loss (one client's update permanently discarded every round) rather than a merely-slower optimization trajectory that would eventually converge to the same point.
- **Noisier/smoother:** no clear difference — both runs show comparable round-to-round validation fluctuation (roughly 1–2 pp swings, e.g. both dip and recover around rounds 27–29); this analysis does not overinterpret those small fluctuations.
- **Diverges early or late:** early — the gap is already substantial by round 10–20 and plateaus rather than continuing to grow indefinitely or closing late.

## 8. Scientific Interpretation

**A. Does Multi-Krum impose a meaningful clean-performance penalty under this strong Non-IID realization?**
**Yes.** A consistent ~6.8–9.1 pp degradation across every accuracy metric and a ~49% relative increase in test loss, present from early training and stable through the end of the run — not a marginal effect.

**B. Does Multi-Krum systematically reject honest clients?**
**Yes, decisively.** One honest client (`client_004`, the one with the *largest* sample count) is rejected in 49 of 50 rounds, including one unbroken 48-round streak; the other four clients are rejected 0 or 1 times each. This is a strongly concentrated, non-uniform pattern, not roughly-even rotation across the honest client pool.

**C. Does the pre-registered rejection proxy already produce a clean-run false attribution?**
**Yes.** Applying `P_krum=3` exactly as pre-registered, `client_004` is persistently confirmed at round 4 (onset round 2) and remains confirmed continuously through round 49 — a sustained false positive with zero attacker present. This is kept strictly separate from (A)/(B) above: it is an attribution-proxy finding, not a robustness/aggregation finding, per the plan's Section 12 Table A/Table B separation discipline.

**D. Is the integration/logging path technically valid?**
**Yes.** `multikrum_diagnostics.csv` is structurally complete and internally consistent across all 50 rounds (Section 2); `summary.json`, `final_test_metrics.json`, and `round_history.csv` are all well-formed and mutually consistent with each other and with the config. This closes the one disclosed, non-blocking limitation from `analysis/MULTIKRUM_IMPLEMENTATION_VERIFICATION.md` Section G (the integration wiring had no prior automated test coverage; it has now been exercised end-to-end on a real run and produced valid output).

**Robustness (A/B) and attribution-proxy (C) interpretations are reported separately above and must not be merged into a single verdict**, consistent with the plan's Table A/Table B discipline.

## 9. EXP_021 Gate

**Decision: A — EXP_020 is technically valid and scientifically interpretable; proceed to the seed-2026 clean Multi-Krum control, EXP_021.**

No integration/logging problem was found (Section 2/8-D: structurally complete, internally consistent). No algorithmic correctness issue was found (the observed behavior — persistent rejection of the largest-sample client, a substantial clean-performance cost, and a clean-run false positive under the pre-registered proxy — is exactly the kind of outcome the frozen plan's Section 6 pre-registered as a real possibility, not a symptom of a bug). Per this task's explicit instruction, poor clean performance and honest-client rejection are scientific findings, not implementation failures, and do not block EXP_021 on their own. `EXP_021` (the seed-2026 clean Multi-Krum control, paired with `EXP_009`) was **not created in this task**.

---

## Final Report

1. **File created:** `analysis/EXP_020_CLEAN_MULTIKRUM_SEED123_ANALYSIS.md`.
2. **Files modified:** none.
3. **Pairing verification:** valid — same seed (123), same Dirichlet(α=0.1) partition realization (identical client sample counts, integer-for-integer), same model/optimizer/LR/50-rounds/5-of-5 client participation, no attack in either run; the sole scientific difference is `aggregation: fedavg → multikrum` (Section 1).
4. **Diagnostics integrity:** `multikrum_diagnostics.csv` — 250/250 expected rows present across 50 rounds, 5 clients/round, ranks 1–5 assigned exactly once per round, exactly 4 selected/1 rejected per round every round, no missing or malformed rows (Section 2).
5. **Rejection count per client (of 50 rounds):** `client_000`=0, `client_001`=0, `client_002`=0, `client_003`=1, `client_004`=49 (Section 3).
6. **Longest rejection streak per client:** `client_000`=0, `client_001`=0, `client_002`=0, `client_003`=1, `client_004`=48 consecutive rounds (rounds 2–49) (Section 3).
7. **Whether the persistent-rejection proxy creates a clean false positive:** yes — `client_004` is persistently confirmed (Section 4).
8. **Predicted suspicious client:** `client_004`; onset round 2, confirmation round 4, persisting continuously through round 49; 1 persistent false-positive client total (Section 4).
9. **EXP_020 vs. EXP_008 Top-1 difference:** −6.92 pp (test), −6.82 pp (best validation) (Section 6).
10. **EXP_020 vs. EXP_008 Top-5 difference:** −7.73 pp (test) (Section 6).
11. **Best-validation difference:** −6.82 pp Top-1 (58.26%→51.44%), best round 47→48 (Section 6).
12. **Other metric differences:** test loss +0.7247, macro precision −9.07 pp, macro recall −6.92 pp, macro F1 −8.70 pp, balanced accuracy −6.92 pp (Section 6).
13. **Clean-performance-cost verdict:** substantial and consistent — ~6.8–9.1 pp degradation across every accuracy metric, present from early training and not closing by round 49 (Sections 6–8A).
14. **Honest-rejection-behavior verdict:** systematically concentrated on one honest client (`client_004`, the largest-sample client), not distributed roughly evenly across the honest pool (Sections 3, 5, 8B).
15. **Integration verdict:** technically valid — diagnostics artifact is structurally complete and internally consistent across all 50 rounds; the previously-disclosed test-coverage gap for this wiring is now closed by this real run (Sections 2, 8D).
16. **EXP_021 gate decision:** **A** — proceed to the seed-2026 clean Multi-Krum control (Section 9).
17. **Confirmation Multi-Krum was not modified:** confirmed — `src/federated/aggregation/multikrum.py` was not opened or touched in this task; `f=1`/`m=4` were not tuned.
18. **Confirmation FedTrace v3 unchanged:** confirmed — `src/analysis/fedtrace_detector.py` was not opened or touched.
19. **Confirmation no experiment/config/code/test/database changed:** confirmed — only pre-existing result artifacts (`summary.json`, `final_test_metrics.json`, `round_history.csv`, `multikrum_diagnostics.csv`, `client_partition_summary.csv` for both EXP_020 and EXP_008) and prior analysis/plan documents were read; no config, code, test, README, or Research Database file was created, edited, or run.
20. **Confirmation whether any commands were executed:** none — this task used only read-only file inspection (`Read`) and file-pattern search (`Glob`) tool calls; no shell, PowerShell, Bash, or Python command was executed.
