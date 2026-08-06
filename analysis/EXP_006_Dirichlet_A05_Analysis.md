# EXP_006 — Non-IID Dirichlet(α=0.5) Analysis (EXP_004 vs. EXP_006)

**Analysis date:** 2026-08-05
**Subject experiments:** `EXP_004` (FedAvg IID, 50 rounds — frozen reference
protocol) vs. `EXP_006` (FedAvg Non-IID, Dirichlet α=0.5, 50 rounds)

**Data sources (verified, no invented values):**
- `results/federated/cifar100_fedavg_iid_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`
- `results/federated/cifar100_fedavg_dirichlet_a05_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`

Round-block marginal-gain figures are simple arithmetic (subtraction and
division) applied directly to the verified `validation_top1`/
`validation_loss` columns of each run's `round_history.csv`, using the
same previous-block-endpoint-baseline convention established in
`analysis/EXP_004_Convergence_Analysis.md` and
`analysis/EXP_005_Cosine_Scheduler_Analysis.md`.

This document performs analysis only. No code, config, test, or Research
Database file was created or modified in the process of producing it.

---

## 1. Controlled-Variable Confirmation

Per `research_plans/EXP_006_NonIID_Dirichlet_Study_Plan.md` and
`configs/cifar100_fedavg_dirichlet_a05_50rounds.yaml`, the declared single
scientific variable is `partition` (`iid → dirichlet`), with
`dirichlet_alpha: 0.5` as its required companion (inert unless
`partition="dirichlet"`). Both runs' own `summary.json` confirm every
other tracked setting is identical:

| Field | EXP_004 | EXP_006 |
|---|---|---|
| `num_clients` / `clients_per_round` / `minimum_updates` | 5 / 5 / 5 | 5 / 5 / 5 |
| `communication_rounds_configured` / `_completed` | 50 / 50 | 50 / 50 |
| `local_epochs` | 1 | 1 |
| `aggregation_strategy` | fedavg | fedavg |
| `optimizer` / `learning_rate` | sgd / 0.01 | sgd / 0.01 |
| `batch_size` | 32 | 32 |
| `seed` | 42 | 42 |
| `total_parameters` | 11,220,132 | 11,220,132 |
| `gpu_name` | NVIDIA GeForce RTX 5060 Laptop GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| `partition` | `iid` | `dirichlet` |
| `dirichlet_alpha` | *(not applicable)* | `0.5` |

**Round 0 is *not* numerically identical between these two runs** — unlike
the `EXP_004`→`EXP_005` comparison (same IID partition, only the
round-level LR schedule changed), `EXP_004` and `EXP_006` use genuinely
different partitioning *strategies*. Even at a matched seed, changing the
partitioning function necessarily changes which samples land on which
client (`EXP_006`'s round 0: validation loss 4.79901640625, top-1 1.36%,
top-5 6.98%; `EXP_004`'s round 0: validation loss 4.7639240234375, top-1
1.86%, top-5 7.30%). This is expected and was flagged in advance in the
study plan (Section 4/9's "Imperfect comparability" risk) — it is a
property of comparing two partitioning strategies, not a data error.

---

## 2. IID vs. Dirichlet(α=0.5) Comparison — Summary

| | EXP_004 (IID) | EXP_006 (Dirichlet α=0.5) |
|---|---:|---:|
| Final test top-1 | 68.52% | 66.22% |
| Final test top-5 | 91.27% | 90.63% |
| Best validation round | 49 (final round) | 46 (not final) |
| Client sample counts | 9,000 / 9,000 / 9,000 / 9,000 / 9,000 | 9,696 / 9,079 / 8,487 / 9,255 / 8,483 |
| Total experiment time | 4,055.95 s | 4,241.69 s |

`EXP_006` trails `EXP_004` on every accuracy metric, but the gap is
modest — see Sections 3–5 for exact figures and Section 8 for a severity
assessment. Both runs still exceed the centralized `EXP_001` baseline
(62.71% test top-1): `EXP_004` by +5.81pp, `EXP_006` by +3.51pp.

---

## 3. Final Validation and Test Metrics

| Metric | EXP_004 (IID) | EXP_006 (Dirichlet α=0.5) | Δ (EXP_006 − EXP_004) |
|---|---:|---:|---:|
| Final test loss | 1.1049091817855834 | 1.1717977018356323 | +0.0669 (worse) |
| Final test top-1 | 0.6852 | 0.6622 | **−0.0230** |
| Final test top-5 | 0.9127 | 0.9063 | **−0.0064** |
| Final validation loss | 1.115539186668396 | 1.200572752380371 | +0.0850 (worse) |
| Final validation top-1 | 0.6868 | 0.6536 | **−0.0332** |
| Final validation top-5 | 0.9140 | 0.9084 | **−0.0056** |

---

## 4. Best Validation Round

| | EXP_004 | EXP_006 |
|---|---|---|
| Best validation round | **49** (the final round) | **46** (not the final round) |
| Best validation top-1 | 68.68% | 65.70% |

`EXP_004`'s best round is its last round — validation top-1 was still
climbing at round 49. `EXP_006`'s validation top-1 peaked at round 46
(65.70%) and then wobbled over the final three rounds:
round 46→47: 65.70%→65.06% (−0.64pp), round 47→48: 65.06%→65.48%
(+0.42pp), round 48→49: 65.48%→65.36% (−0.12pp) — a net −0.34pp from peak
to final round (`round_history.csv`, `validation_top1` column). Validation
*loss* over the same window is noisy but trends down overall (round 46:
1.2107 → round 49: 1.2006), the same "noisy accuracy plateau despite
continued marginal loss improvement" pattern observed in `EXP_005`
(`analysis/EXP_005_Cosine_Scheduler_Analysis.md` → Section 3), though
milder here.

---

## 5. Test Loss, Top-1, Top-5, Macro Precision, Macro Recall, Macro F1, Balanced Accuracy

| Metric | EXP_004 | EXP_006 | Δ |
|---|---:|---:|---:|
| Loss | 1.1049091817855834 | 1.1717977018356323 | +0.0669 |
| Top-1 | 0.6852 | 0.6622 | −0.0230 |
| Top-5 | 0.9127 | 0.9063 | −0.0064 |
| Macro precision | 0.6901 | 0.6785 | −0.0116 |
| Macro recall | 0.6850 | 0.6625 | −0.0225 |
| Macro F1 | 0.6814 | 0.6589 | −0.0225 |
| Balanced accuracy | 0.6850 | 0.6625 | −0.0225 |

Every macro metric moves in the same direction and by a broadly comparable
magnitude (−0.012 to −0.023) — consistent with a general, broad-based
accuracy reduction rather than a specific failure concentrated in a few
classes.

---

## 6. Client Sample-Count Imbalance

From `client_partition_summary.csv`:

| Client | EXP_004 (IID) | EXP_006 (Dirichlet α=0.5) |
|---|---:|---:|
| client_000 | 9,000 | 9,696 |
| client_001 | 9,000 | 9,079 |
| client_002 | 9,000 | 8,487 |
| client_003 | 9,000 | 9,255 |
| client_004 | 9,000 | 8,483 |
| **Total** | **45,000** | **45,000** |

`EXP_006`'s per-client totals range from 8,483 to 9,696 (spread of 1,213
samples), a mean of exactly 9,000 (matching `EXP_004`'s uniform
allocation) with a standard deviation of ≈466 samples — a coefficient of
variation of **≈5.2%**. This is a real but *mild* total-sample-count
imbalance.

**This total-count imbalance is not, by itself, a direct measurement of
the intended Non-IID variable.** `dirichlet_alpha` primarily controls each
client's *per-class* label distribution — how much of each of the 100
classes a client receives — not its total sample count. A client's total
count is the sum of its (possibly very skewed) per-class allocations
across 100 classes, and by the law of large numbers that sum tends to
concentrate closer to the dataset-wide average than any single class's
allocation does, even under substantial per-class skew. `client_partition_summary.csv`
records only per-client **totals**, not a per-class breakdown (this gap
was flagged in advance in `research_plans/EXP_006_NonIID_Dirichlet_Study_Plan.md`
→ Sections 7 and 10 as a needed-but-not-yet-implemented deliverable).
**Consequently, this analysis cannot directly verify from available files
how skewed `EXP_006`'s per-client *class* distributions actually were** —
only that per-client total sample counts were mildly, not severely,
imbalanced. The accuracy/convergence differences in Sections 3–5 and 7 are
the best available indirect evidence that real heterogeneity was present.

---

## 7. Round-by-Round Convergence Comparison

**Validation top-1 at decade-block boundaries** (`round_history.csv`):

| Round | 0 | 9 | 19 | 29 | 39 | 49 |
|---|---:|---:|---:|---:|---:|---:|
| EXP_004 | 1.86% | 37.54% | 54.22% | 61.10% | 65.50% | 68.68% |
| EXP_006 | 1.36% | 36.00% | 51.60% | 58.86% | 63.44% | 65.36% |

**Average validation-top-1 gain per round, by decade block**
(previous-block endpoint as baseline):

| Block | EXP_004 (pp/round) | EXP_006 (pp/round) |
|---|---:|---:|
| Rounds 0–9 | ≈3.964 | ≈3.849 |
| Rounds 10–19 | ≈1.668 | ≈1.560 |
| Rounds 20–29 | ≈0.688 | ≈0.726 |
| Rounds 30–39 | ≈0.440 | ≈0.458 |
| Rounds 40–49 | ≈0.318 | ≈0.192 |

**Average validation-loss reduction per round, by decade block:**

| Block | EXP_004 (loss/round) | EXP_006 (loss/round) |
|---|---:|---:|
| Rounds 0–9 | ≈0.2649 | ≈0.2680 |
| Rounds 10–19 | ≈0.0707 | ≈0.0669 |
| Rounds 20–29 | ≈0.0288 | ≈0.0257 |
| Rounds 30–39 | ≈0.0167 | ≈0.0182 |
| Rounds 40–49 | ≈0.0102 | ≈0.0079 |

Through round 39, `EXP_006`'s per-block marginal gains track `EXP_004`'s
closely — in two blocks (20–29, 30–39) `EXP_006` is marginally *ahead*.
The clearest, most consistent divergence is the final block (rounds
40–49): `EXP_006`'s marginal top-1 gain (≈0.192pp/round) is about 60% of
`EXP_004`'s (≈0.318pp/round), and its loss reduction (≈0.0079/round) is
about 78% of `EXP_004`'s (≈0.0102/round). This is a real, measurable
late-training slowdown relative to IID — but it is far milder than
`EXP_005`'s late-training near-stall (rounds 40–49: ≈0.026pp/round top-1,
≈0.0022/round loss — roughly 8–12% of `EXP_004`'s rate;
`analysis/EXP_005_Cosine_Scheduler_Analysis.md` → Section 5).

---

## 8. Is the Degradation Mild, Moderate, or Severe?

**Mild.** Supporting evidence:

- Final test top-1 drops by 2.30 percentage points (68.52% → 66.22%), a
  relative reduction of about 3.4% — `EXP_006` still exceeds the
  centralized `EXP_001` baseline (62.71%) by +3.51pp.
- Macro metrics move together by 0.012–0.023, indicating broad, even
  degradation rather than a severe failure concentrated in specific
  classes or a collapse in any one metric.
- Round-block marginal gains track `EXP_004` closely through round 39;
  the only clear slowdown is confined to the final decade block, and even
  there `EXP_006` retains roughly 60–78% of `EXP_004`'s marginal
  efficiency (Section 7) — not the near-total stall seen in `EXP_005`
  under a different (scheduler-related) failure mode.
- The measured total-sample-count imbalance across clients is itself mild
  (≈5.2% coefficient of variation, Section 6), consistent with
  `alpha=0.5` being a moderate, not extreme, skew setting in the
  Dirichlet-partitioning literature.

This pattern — a real but contained accuracy and convergence-rate
reduction, with no collapse in any tracked metric — is consistent with
`alpha=0.5` producing genuine but moderate client heterogeneity, not a
severe one. Whether a more extreme heterogeneity setting produces a
correspondingly larger gap is exactly what Section 10/11 proposes testing
next.

---

## 9. One Seed Is Not Enough for a Universal Claim

This is a single-seed (42), single-`alpha` (0.5) comparison of one
Dirichlet partition against `EXP_004`'s frozen IID reference, on one
dataset/model/aggregation combination (CIFAR-100, ResNet-18, FedAvg, 5
clients, 50 rounds). It does **not** establish that:

- A 2.30pp accuracy gap is the "true" cost of Dirichlet(0.5) heterogeneity
  for this pipeline — no repeated-seed trials exist for either `EXP_004`
  or `EXP_006` in this project, so the exact size of this gap has not been
  checked against ordinary run-to-run noise (CUDA nondeterminism,
  floating-point accumulation order, or the specific random partition
  drawn at seed 42 versus another seed at the same `alpha`).
- This mild-degradation finding generalizes to other `alpha` values, other
  client counts, other round budgets, other local-epoch counts, or a
  persistent (non-reset) client optimizer-state policy.
- The gap would hold under a different model architecture or dataset —
  CIFAR-100 with ResNet-18 is this project's only tested combination so
  far.

The verified, narrow conclusion is: **at seed 42, this specific Dirichlet(0.5)
partition produced a mild accuracy and convergence-rate degradation
relative to `EXP_004`'s IID reference** — not a general characterization of
FedAvg's heterogeneity sensitivity.

---

## 10. Should Stronger Heterogeneity Be Tested?

**Yes.** Two independent reasons:

1. Section 8 classifies `EXP_006`'s degradation as mild — this is
   consistent with `alpha=0.5` being a moderate skew setting, and does not
   yet stress-test how this pipeline behaves under the more severe
   heterogeneity that realistic federated deployments (and, later,
   adversarial conditions in Phase 4) can present. A single data point at
   one moderate `alpha` cannot characterize the shape of the
   accuracy-vs-heterogeneity relationship — whether it degrades gradually,
   or has a threshold beyond which it degrades sharply.
2. Section 6 notes this analysis cannot directly verify per-client
   *class*-distribution skew from currently available files, only the
   (mild) total-sample-count imbalance. A lower `alpha` produces a more
   extreme, more visibly distinguishable-from-IID split, making the
   intended heterogeneity effect easier to detect and interpret even
   without a per-class breakdown, and more clearly worth the (still
   pending) per-class-breakdown instrumentation effort if pursued later.

---

## 11. Defining `EXP_007`

Per the same controlled-variable discipline used for every prior
experiment in this project: exactly one scientific variable changes
relative to `EXP_006`.

| Field | EXP_006 (frozen) | EXP_007 (planned) |
|---|---|---|
| `dirichlet_alpha` | `0.5` | **`0.1`** |
| `partition` | `dirichlet` | `dirichlet` (unchanged) |
| `experiment_id` / `experiment_name` / `output_dir` / `checkpoint_dir` | `EXP_006` / ... | new, non-colliding identity only (bookkeeping) |
| Every other field (model, optimizer, learning rate, momentum, weight decay, batch size, local epochs, aggregation, client count, communication rounds, seed, evaluation/checkpoint cadence) | — | **unchanged from `EXP_006`/`EXP_004`** |

A lower `alpha` (0.1 vs. 0.5) produces more strongly skewed per-client
class distributions — the standard, literature-consistent direction for
increasing Dirichlet-partition heterogeneity. This document does not
create `EXP_007`'s config or run it; per this project's standing
workflow, that remains a separate, explicit step.

---

## 12. Falsifiable Hypothesis for `EXP_007`

**Hypothesis:** Reducing `dirichlet_alpha` from 0.5 to 0.1, with every
other setting identical to `EXP_006`, will produce a global model with
final test top-1 accuracy **lower than `EXP_006`'s verified 66.22%**, and
a larger gap relative to `EXP_004`'s IID reference (68.52%) than
`EXP_006`'s current 2.30-percentage-point gap. This follows directly from
Section 10's premise: a smaller `alpha` produces more severe per-client
class-distribution skew, which the FedAvg literature (and this project's
own Section 6–8 reasoning) predicts degrades convergence further via
increased client drift.

**Falsification condition:** If `EXP_007`'s final test top-1 is not lower
than `EXP_006`'s 66.22% by a clearly non-trivial margin, **or** if the gap
to `EXP_004`'s 68.52% does not widen beyond `EXP_006`'s 2.30 percentage
points, the hypothesis is falsified. That outcome would indicate the
relationship between `alpha` and FedAvg accuracy is not simply monotonic
in this configuration (round budget = 50, `local_epochs=1`, 5 clients) —
itself a notable finding requiring its own explanation, not a result to
assume away or rerun until a "confirming" outcome appears.
