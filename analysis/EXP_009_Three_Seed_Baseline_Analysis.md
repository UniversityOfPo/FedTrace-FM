# EXP_009 — Three-Seed Dirichlet(α=0.1) Baseline Analysis (Seeds 42, 123, 2026)

**Analysis date:** 2026-08-05
**Subject experiments:**
- `EXP_007` — FedAvg Non-IID, Dirichlet α=0.1, 50 rounds, seed **42**
- `EXP_008` — FedAvg Non-IID, Dirichlet α=0.1, 50 rounds, seed **123**
- `EXP_009` — FedAvg Non-IID, Dirichlet α=0.1, 50 rounds, seed **2026**
- Reference: `EXP_004` — FedAvg IID, 50 rounds (frozen reference protocol)

**Data sources (verified, no invented values):**
- `results/federated/cifar100_fedavg_dirichlet_a01_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`
- `results/federated/cifar100_fedavg_dirichlet_a01_seed123_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`
- `results/federated/cifar100_fedavg_dirichlet_a01_seed2026_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`
- `results/federated/cifar100_fedavg_iid_50rounds/summary.json`,
  `final_test_metrics.json` (`EXP_004` reference)

Mean, sample standard deviation (n−1 denominator), minimum, maximum, and
range are simple arithmetic computed directly from the three seeds'
verified `final_test_metrics.json` values. Round-block figures use the
same previous-block-endpoint-baseline convention established in
`analysis/EXP_004_Convergence_Analysis.md` and reused throughout this
series.

This document performs analysis only. No code, config, test, README, or
Research Database file was created or modified in the process of
producing it. **No claim of statistical significance is made anywhere in
this document — three seeds support descriptive summary statistics only
(see § 9).**

---

## 1. Strict Replication Confirmation

Per `configs/cifar100_fedavg_dirichlet_a01_50rounds.yaml`,
`configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml`, and
`configs/cifar100_fedavg_dirichlet_a01_seed2026_50rounds.yaml`, the
**only** field that differs across `EXP_007`/`EXP_008`/`EXP_009` is
`seed` (`42` / `123` / `2026`). All three runs' own `summary.json`
confirm every other tracked setting is identical: `partition=dirichlet`,
`dirichlet_alpha=0.1`, `num_clients=5`, `clients_per_round=5`,
`minimum_updates=5`, `communication_rounds=50` (all three completed
50/50), `local_epochs=1`, `aggregation_strategy=fedavg`, `optimizer=sgd`,
`learning_rate=0.01`, `batch_size=32`, `total_parameters=11,220,132`,
same GPU (NVIDIA GeForce RTX 5060 Laptop GPU). All three runs completed
every round with 5/5 successful clients and 0 rejected updates
(`round_history.csv`, `round_status=completed` on all 50 rows in all
three files). This is a genuine three-way strict replication.

---

## 2. Per-Seed Validation and Test Metrics

| Metric | EXP_007 (seed 42) | EXP_008 (seed 123) | EXP_009 (seed 2026) |
|---|---:|---:|---:|
| Final validation loss | 1.4239414876937866 | 1.4697468055725098 | 1.4886625778198241 |
| Final validation top-1 | 59.22% | 57.78% | 58.08% |
| Final validation top-5 | 88.26% | 87.98% | 87.94% |
| Best validation round | 49 (final) | 47 (not final) | 49 (final) |
| Best validation top-1 | 59.22% | 58.26% | 58.08% |
| Final test loss | 1.3758617393493653 | 1.4654396366119384 | 1.446613697052002 |
| Final test top-1 | 59.68% | 56.83% | 58.26% |
| Final test top-5 | 88.65% | 87.79% | 88.56% |
| Macro precision | 0.6509177318450269 | 0.6337611188249366 | 0.6237342308141861 |
| Macro recall | 0.5965 | 0.5683 | 0.5827 |
| Macro F1 | 0.5875828060131298 | 0.5485726596416799 | 0.5641049541589388 |
| Balanced accuracy | 0.5965 | 0.5683 | 0.5827 |
| Total experiment time | 4,136.48 s | 4,008.92 s | 5,094.58 s |
| Client sample-count CV | ≈10.9% | ≈23.4% | ≈30.9% |

---

## 3. Three-Seed Summary Statistics

Computed from the three `final_test_metrics.json` values above (mean and
sample standard deviation use n=3, n−1=2):

| Metric | Mean | Sample Std. Dev. | Min | Max | Range |
|---|---:|---:|---:|---:|---:|
| Test loss | 1.4293 | 0.0472 | 1.3759 (seed 42) | 1.4654 (seed 123) | 0.0896 |
| Test top-1 | 58.26% | 1.42pp | 56.83% (seed 123) | 59.68% (seed 42) | 2.85pp |
| Test top-5 | 88.33% | 0.47pp | 87.79% (seed 123) | 88.65% (seed 42) | 0.86pp |
| Macro precision | 0.6361 | 0.0137 | 0.6237 (seed 2026) | 0.6509 (seed 42) | 0.0272 |
| Macro recall | 0.5825 | 0.0141 | 0.5683 (seed 123) | 0.5965 (seed 42) | 0.0282 |
| Macro F1 | 0.5668 | 0.0196 | 0.5486 (seed 123) | 0.5876 (seed 42) | 0.0390 |
| Balanced accuracy | 0.5825 | 0.0141 | 0.5683 (seed 123) | 0.5965 (seed 42) | 0.0282 |

Seed 42 (`EXP_007`) is the best-performing seed on every single metric;
seed 123 (`EXP_008`) is the worst-performing on every metric except
macro precision, where seed 2026 (`EXP_009`) is lowest. The
coefficient of variation on test top-1 (std/mean ≈ 1.42/58.26 ≈ 2.4%) is
modest relative to the mean, but the absolute spread (2.85pp) is not
small in the context of effect sizes this project has otherwise reported
(§ 9 below).

---

## 4. Three-Seed α=0.1 Mean vs. `EXP_004` (IID)

| Metric | EXP_004 (IID) | α=0.1 Mean (n=3) | α=0.1 Range |
|---|---:|---:|---:|
| Test loss | 1.1049091817855834 | 1.4293 | 1.3759–1.4654 |
| Test top-1 | 68.52% | 58.26% | 56.83%–59.68% |
| Test top-5 | 91.27% | 88.33% | 87.79%–88.65% |
| Macro precision | 0.6901 | 0.6361 | 0.6237–0.6509 |
| Macro recall | 0.6850 | 0.5825 | 0.5683–0.5965 |
| Macro F1 | 0.6814 | 0.5668 | 0.5486–0.5876 |
| Balanced accuracy | 0.6850 | 0.5825 | 0.5683–0.5965 |

**All three individual seeds, not just the mean, fall below both
`EXP_004` (68.52%) and the centralized `EXP_001` reference (62.71%) on
test top-1** — seed 42: 59.68% (−3.03pp vs. `EXP_001`); seed 123: 56.83%
(−5.88pp); seed 2026: 58.26% (−4.45pp). This is a unanimous 3-of-3
result, not a single-seed artifact (see § 10).

---

## 5. Quantifying the Mean IID-to-α=0.1 Degradation

Using the three-seed mean as the best available point estimate:

| Metric | Mean Degradation (EXP_004 − α=0.1 mean) |
|---|---:|
| Test loss | +0.3244 (worse) |
| Test top-1 | **−10.26pp** |
| Test top-5 | −2.94pp |
| Macro precision | −0.0540 |
| Macro recall | −0.1025 |
| Macro F1 | −0.1146 |
| Balanced accuracy | −0.1025 |

The per-seed top-1 degradation estimates individually were −8.84pp
(seed 42), −11.69pp (seed 123), and −10.26pp (seed 2026) — the mean
(−10.26pp) sits within that 2.85pp-wide range, roughly in the middle.
This mean is a more defensible estimate of the IID-to-α=0.1 accuracy cost
than any single prior single-seed report, but § 9 explains why it is
still not a precise or statistically validated number.

---

## 6. Best-Validation Rounds and Convergence Variability

| | EXP_007 (seed 42) | EXP_008 (seed 123) | EXP_009 (seed 2026) |
|---|---|---|---|
| Best validation round | 49 (final) | 47 (not final) | 49 (final) |

**2 of 3 seeds have their final round as their best round; 1 of 3 peaks
early.** `EXP_008` peaked at round 47 (58.26%) and declined to round 49
(57.78%, net −0.48pp from peak); `EXP_007` and `EXP_009` were both still
improving (or exactly at their best) at round 49.

**Validation loss converges smoothly and monotonically, block over
block, in all three seeds**, with no block-level reversal in any of
them — this is a genuinely 3-of-3 reproducible property.

**Validation top-1's late-training behavior does *not* reproduce 3-of-3.**
`analysis/EXP_008_Replication_Analysis.md` § 6 observed that both
`EXP_007` and `EXP_008` showed a higher marginal top-1 gain in the final
decade block (rounds 40–49) than in the second-to-last block (30–39) —
and, based on only those two seeds, described this as a pattern that
"replicates in both seeds." **`EXP_009` does not show this pattern**:

| Block | EXP_007 (pp/round) | EXP_008 (pp/round) | EXP_009 (pp/round) |
|---|---:|---:|---:|
| Rounds 30–39 | ≈0.310 | ≈0.286 | ≈0.646 |
| Rounds 40–49 | ≈0.368 | ≈0.300 | **≈0.152** |

`EXP_009`'s block 40–49 marginal gain (≈0.152pp/round) is *lower* than
its block 30–39 rate (≈0.646pp/round) — the ordinary decelerating
pattern `EXP_004`/`EXP_006` also showed, not the uptick `EXP_007`/`EXP_008`
showed. **With the third seed, this feature is 2-of-3, not 3-of-3** — it
is not a reliable property of this configuration, and the prior
two-seed document's tentative conclusion is explicitly superseded here.
This is precisely the kind of correction
`analysis/EXP_008_Replication_Analysis.md` § 10 warned could happen with
a third observation.

---

## 7. Client Sample-Count Imbalance Across Seeds

| Client | EXP_007 (seed 42) | EXP_008 (seed 123) | EXP_009 (seed 2026) |
|---|---:|---:|---:|
| client_000 | 10,883 | 6,024 | 5,233 |
| client_001 | 8,003 | 9,628 | 12,318 |
| client_002 | 8,868 | 7,202 | 9,171 |
| client_003 | 8,574 | 10,361 | 6,556 |
| client_004 | 8,672 | 11,785 | 11,722 |
| **Total** | **45,000** | **45,000** | **45,000** |
| Range (max − min) | 2,880 | 5,761 | **7,085** |
| Coefficient of variation | ≈10.9% | ≈23.4% | **≈30.9%** |

At the **identical** `dirichlet_alpha=0.1`, the realized total-sample-count
imbalance varies enormously across seeds: coefficient of variation ranges
from ≈10.9% to ≈30.9%, essentially tripling. This confirms and
strengthens the finding in `analysis/EXP_008_Replication_Analysis.md`
§ 5: at this `alpha`, the Dirichlet partition realization itself is
highly seed-sensitive, quite apart from any subsequent training noise.
Client identities also shift substantially — `client_000` is the
largest client in `EXP_007` (10,883) but the smallest in both `EXP_008`
(6,024) and `EXP_009` (5,233); `client_004` is mid-sized in `EXP_007`
(8,672) but the largest or near-largest in `EXP_008`/`EXP_009`
(11,785 / 11,722).

---

## 8. What Replicated Qualitatively (3-of-3)

- **Substantial, consistent degradation vs. the IID reference.** All
  three seeds show a large gap to `EXP_004` (8.84–11.69pp test top-1,
  mean 10.26pp) — never small, never absent.
- **All three seeds fall below the centralized `EXP_001` baseline**
  (62.71% test top-1) — a unanimous reversal of the pattern `EXP_004`,
  `EXP_005`, and `EXP_006` established.
- **Full pipeline stability.** 50/50 rounds completed, 5/5 successful
  clients, 0 rejected updates, in all three seeds — no collapse or
  failure in any run.
- **Smooth, monotonic validation-loss convergence** in every block, for
  all three seeds, with no reversal.
- **Broad, consistent macro-metric movement.** All macro metrics (not
  just overall accuracy) move together and substantially below `EXP_004`
  in every seed (§ 4) — general degradation, not a class-specific
  failure, in all three runs.

**What did *not* replicate 3-of-3** (revising the two-seed document's
tentative claims):
- The "final round is best round" feature — 2 of 3, not universal (§ 6).
- The "final-block top-1 marginal-gain uptick" pattern — 2 of 3, not
  universal, and now understood to be seed-specific rather than a
  property of `alpha=0.1` itself (§ 6).
- The exact magnitude of degradation — genuinely different per seed
  (8.84pp to 11.69pp), not a fixed number.

---

## 9. What Three Seeds Support — and Still Cannot Prove

**Supported by this three-seed evidence:**
- The qualitative direction of the effect (Dirichlet α=0.1 substantially
  degrades FedAvg accuracy relative to both the IID reference and the
  centralized baseline, at this round budget and client count) is now
  observed consistently in three independent draws, not one.
- A descriptive point estimate and spread for the degradation magnitude
  (mean −10.26pp test top-1, observed range 8.84–11.69pp) — useful as a
  working estimate, not a precise or final number.
- Direct evidence that both training-dynamics noise *and*
  partition-realization variance (§ 7) contribute to seed-to-seed
  variability at this `alpha`.

**Cannot be proven by three seeds, and no claim to the contrary is made
in this document:**
- **No statistical significance test is valid with n=3.** A sample
  standard deviation from three points is itself a very high-variance
  estimator; confidence intervals or p-values computed from it would be
  unreliable and are not reported here.
- Whether the true population mean degradation is closer to 8.84pp,
  11.69pp, 10.26pp, or a value outside this observed range entirely.
- Whether any of `EXP_007`/`EXP_008`/`EXP_009`'s remaining
  *qualitative* agreements (§ 8's "3-of-3" list) would still hold at a
  fourth or fifth seed — § 6 already shows one previously-assumed 2-of-2
  pattern failed to survive a third seed.
- Generalization beyond this exact configuration (CIFAR-100, ResNet-18,
  FedAvg, 5 clients, 50 rounds, `local_epochs=1`, `alpha=0.1`
  specifically) to any other heterogeneity level, dataset, architecture,
  or aggregation strategy.

---

## 10. Is the Strong-Heterogeneity FedAvg Baseline Sufficiently Established?

**Yes, for the qualitative/directional finding — sufficient to proceed to
a controlled comparison method; no, for a precise magnitude.** Three
independent seeds unanimously show plain FedAvg degrading substantially
under `alpha=0.1` heterogeneity, unanimously falling below both the IID
and centralized references, with no collapse or instability in any run.
That directional consistency (3-of-3, § 8) is a reasonable basis for
treating "FedAvg has a real, non-trivial heterogeneity problem at this
`alpha`" as established enough to justify testing a heterogeneity-aware
aggregation strategy against it. The exact size of the problem (§ 9) is
not established beyond an approximate range, and this document does not
claim otherwise.

---

## 11. Recommend FedProx as the Next Controlled Method

Per `research_plans/PHASE4_RESEARCH_STRATEGY.md` § 7, FedProx's evidence
bar was "a substantially larger degradation than `EXP_006`'s mild
2.30pp gap" — the three-seed mean degradation (10.26pp, range
8.84–11.69pp) clears that bar in every individual seed, not only in a
single (possibly noisy) observation. Per § 10, this is now a sufficiently
established qualitative finding to act on. **FedProx (a proximal term
added to each client's local objective, specifically targeting the
client-drift mechanism this heterogeneity level plausibly induces) is
the recommended next controlled method** — the next single-variable
change should be `aggregation: fedavg → fedprox` (plus its required
proximal-term coefficient), with every other setting identical to
`EXP_007`/`EXP_008`/`EXP_009`.

---

## 12. FedProx Must Use the Same Three Seeds: 42, 123, 2026

To isolate the aggregation-strategy effect from the now-confirmed,
substantial partition-realization variance (§ 7: coefficient of variation
ranging ≈10.9%–30.9% across these three seeds alone), the FedProx
follow-up experiments **must reuse seeds 42, 123, and 2026** — not a
fresh set of seeds. This gives a **paired comparison**: at each seed, the
FedAvg and FedProx runs share the identical Dirichlet partition
realization (same client sample counts, same per-client composition),
so any difference between FedAvg and FedProx at a given seed is
attributable to the aggregation strategy itself, not to a different,
uncontrolled partition draw. This directly controls for the largest
confound this analysis identified (§ 7) and yields three paired
differences (FedProx − FedAvg, at seeds 42/123/2026 respectively) that
are far more informative than three independent, unpaired samples would
be — without requiring any additional experiments beyond the three
FedProx runs themselves.
