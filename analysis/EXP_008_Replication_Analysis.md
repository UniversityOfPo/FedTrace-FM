# EXP_008 — Seed-Replication Analysis (EXP_007 seed 42 vs. EXP_008 seed 123)

**Analysis date:** 2026-08-05
**Subject experiments:**
- `EXP_007` — FedAvg Non-IID, Dirichlet α=0.1, 50 rounds, seed 42
- `EXP_008` — FedAvg Non-IID, Dirichlet α=0.1, 50 rounds, **seed 123**
  (strict single-variable replication of `EXP_007`, per
  `configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml`)

**Data sources (verified, no invented values):**
- `results/federated/cifar100_fedavg_dirichlet_a01_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`
- `results/federated/cifar100_fedavg_dirichlet_a01_seed123_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`

Round-block marginal-gain figures use the same previous-block-endpoint-baseline
arithmetic convention established in `analysis/EXP_004_Convergence_Analysis.md`
and reused in every subsequent analysis document in this series.

This document performs analysis only. No code, config, test, README, or
Research Database file was created or modified in the process of producing it.

---

## 1. Strict Replication Confirmation

Per `configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml`, the
**only** field changed relative to `EXP_007`
(`configs/cifar100_fedavg_dirichlet_a01_50rounds.yaml`) is `seed`
(`42 → 123`); `experiment_id`/`experiment_name`/`output_dir`/
`checkpoint_dir` differ only as non-scientific bookkeeping. Both runs'
own `summary.json` confirm every other tracked setting is identical:
`partition=dirichlet`, `dirichlet_alpha` (both configs specify `0.1`),
`num_clients=5`, `clients_per_round=5`, `minimum_updates=5`,
`communication_rounds=50` (both completed 50/50), `local_epochs=1`,
`aggregation_strategy=fedavg`, `optimizer=sgd`, `learning_rate=0.01`,
`batch_size=32`, `total_parameters=11,220,132`, same GPU
(NVIDIA GeForce RTX 5060 Laptop GPU). Both runs completed all 50 rounds
with 5/5 successful clients and 0 rejected updates
(`round_history.csv`, `round_status=completed` on every row for both).
This is a genuine strict replication: `seed` is the only variable that
differs.

---

## 2. Final Validation and Test Comparison

| Metric | EXP_007 (seed 42) | EXP_008 (seed 123) | Δ (008 − 007) |
|---|---:|---:|---:|
| Final test loss | 1.3758617393493653 | 1.4654396366119384 | +0.0896 (worse) |
| Final test top-1 | 59.68% | 56.83% | **−2.85pp** |
| Final validation loss | 1.4239414876937866 | 1.4697468055725098 | +0.0458 (worse) |
| Final validation top-1 | 59.22% | 57.78% | **−1.44pp** |
| Final validation top-5 | 88.26% | 87.98% | −0.28pp |
| Total experiment time | 4,136.48 s | 4,008.92 s | −127.56 s |

`EXP_008` underperforms `EXP_007` on every reported metric, though the
gap is smaller in absolute terms than either run's gap to the IID
reference (`EXP_004`, 68.52% test top-1).

---

## 3. Test Loss, Top-1, Top-5, Macro Precision, Macro Recall, Macro F1, Balanced Accuracy

| Metric | EXP_007 (seed 42) | EXP_008 (seed 123) | Δ |
|---|---:|---:|---:|
| Loss | 1.3758617393493653 | 1.4654396366119384 | +0.0896 |
| Top-1 | 0.5968 | 0.5683 | −0.0285 |
| Top-5 | 0.8865 | 0.8779 | −0.0086 |
| Macro precision | 0.6509177318450269 | 0.6337611188249366 | −0.0172 |
| Macro recall | 0.5965 | 0.5683 | −0.0282 |
| Macro F1 | 0.5875828060131298 | 0.5485726596416799 | −0.0390 |
| Balanced accuracy | 0.5965 | 0.5683 | −0.0282 |

Every metric moves in the same direction (`EXP_008` worse), by a broadly
consistent magnitude (roughly 0.017–0.039 for the fractional metrics) —
a general shift, not a collapse concentrated in one metric.

---

## 4. Best Validation Round Comparison

| | EXP_007 (seed 42) | EXP_008 (seed 123) |
|---|---|---|
| Best validation round | **49** (the final round) | **47** (not the final round) |
| Best validation top-1 | 59.22% | 58.26% |

`EXP_007`'s final round is also its best round (§ 5 of
`analysis/EXP_007_Dirichlet_A01_Analysis.md`). `EXP_008` peaked two
rounds earlier: round 47 (58.26%) → round 48 (56.40%, **−1.86pp**) →
round 49 (57.78%, +1.38pp) — a net −0.48pp from peak to final round.
This "final round is not the best round" pattern matches what `EXP_005`
and `EXP_006` showed (and `EXP_007` did not) — i.e. **this specific
qualitative feature of `EXP_007` did not replicate** under a different
seed.

---

## 5. Client Sample-Count Comparison

| Client | EXP_007 (seed 42) | EXP_008 (seed 123) |
|---|---:|---:|
| client_000 | 10,883 | 6,024 |
| client_001 | 8,003 | 9,628 |
| client_002 | 8,868 | 7,202 |
| client_003 | 8,574 | 10,361 |
| client_004 | 8,672 | 11,785 |
| **Total** | **45,000** | **45,000** |
| Range (max − min) | 2,880 | **5,761** |
| Coefficient of variation | ≈10.9% | **≈23.4%** |

At the identical `dirichlet_alpha=0.1`, the two seeds produced
substantially different total-sample-count imbalance: `EXP_008`'s
coefficient of variation (≈23.4%) is roughly **double** `EXP_007`'s
(≈10.9%), and its range (5,761 samples) is almost exactly double as well.
**The Dirichlet partition realization itself — not only the subsequent
training dynamics — varies substantially between seeds at this `alpha`.**
This is direct, quantitative evidence that part of any accuracy
difference between `EXP_007` and `EXP_008` could originate from `EXP_008`
having drawn a harder (more imbalanced) partition by chance, independent
of any noise introduced later in SGD/aggregation.

---

## 6. Round-by-Round Convergence Comparison

**Validation top-1 at decade-block boundaries** (`round_history.csv`):

| Round | 0 | 9 | 19 | 29 | 39 | 49 |
|---|---:|---:|---:|---:|---:|---:|
| EXP_007 (seed 42) | 1.06% | 30.24% | 46.00% | 52.44% | 55.54% | 59.22% |
| EXP_008 (seed 123) | 1.32% | 30.54% | 44.82% | 51.92% | 54.78% | 57.78% |

**Average validation-top-1 gain per round, by decade block:**

| Block | EXP_007 (pp/round) | EXP_008 (pp/round) |
|---|---:|---:|
| Rounds 0–9 | ≈3.242 | ≈3.247 |
| Rounds 10–19 | ≈1.576 | ≈1.428 |
| Rounds 20–29 | ≈0.644 | ≈0.710 |
| Rounds 30–39 | ≈0.310 | ≈0.286 |
| Rounds 40–49 | ≈0.368 | ≈0.300 |

**Average validation-loss reduction per round, by decade block:**

| Block | EXP_007 (loss/round) | EXP_008 (loss/round) |
|---|---:|---:|
| Rounds 0–9 | ≈0.2359 | ≈0.2511 |
| Rounds 10–19 | ≈0.0746 | ≈0.0658 |
| Rounds 20–29 | ≈0.0242 | ≈0.0251 |
| Rounds 30–39 | ≈0.0149 | ≈0.0145 |
| Rounds 40–49 | ≈0.0092 | ≈0.0101 |

The two seeds' block-level shapes are broadly similar — both show the
same overall decade-by-decade deceleration pattern, and neither is
uniformly ahead of the other block by block (`EXP_008` is marginally
ahead in blocks 0–9 and 20–29; `EXP_007` is ahead in 10–19, 30–39, and
40–49). Loss decreases smoothly and monotonically block-over-block for
**both** seeds, with no reversal in either. Top-1, however, shows the
same qualitative feature in both runs that
`analysis/EXP_007_Dirichlet_A01_Analysis.md` § 4 flagged as unusual:
marginal top-1 gain in the final block (40–49) is *higher* than in the
second-to-last block (30–39) — `EXP_007`: 0.310 → 0.368; `EXP_008`:
0.286 → 0.300. The magnitude of this late-block uptick differs (larger
in `EXP_007`), but **its direction replicates in both seeds**, suggesting
it is a real, if modest, property of this configuration rather than a
seed-42-specific artifact.

---

## 7. Did the Qualitative Degradation Under α=0.1 Replicate?

**Yes, in overall magnitude and direction; no, in every specific detail.**

Replicated:
- Non-IID(α=0.1) substantially underperforms the IID reference (`EXP_004`,
  68.52% test top-1) in both seeds: `EXP_007` by 8.84pp, `EXP_008` by
  11.69pp (68.52% − 56.83%).
- Both seeds fall **below** the centralized `EXP_001` baseline (62.71%):
  `EXP_007` by −3.03pp, `EXP_008` by −5.88pp — the centralized-baseline
  reversal documented in `analysis/EXP_007_Dirichlet_A01_Analysis.md` § 2
  replicates and is, if anything, larger under seed 123.
- The final-block top-1 marginal-gain uptick (§ 6) replicates in
  direction.
- Loss converges smoothly in both seeds with no block-level reversal.
- Both runs remain fully stable (50/50 rounds completed, 0 rejected
  updates) — no collapse or divergence in either seed.

Did not replicate exactly:
- The specific magnitude (8.84pp vs. 11.69pp gap to `EXP_004` — a
  meaningfully different number, not just noise-level variation).
- `EXP_007`'s "final round is best round" feature (§ 4) — `EXP_008` peaks
  two rounds early instead, matching `EXP_005`/`EXP_006`'s pattern rather
  than `EXP_007`'s.

---

## 8. Was the Pre-Registered Numerical-Closeness Threshold Met?

No document in this project pre-registers an exact **1–1.5 percentage-point**
closeness band. The only pre-registered closeness criterion on record is
the qualitative framing in `analysis/EXP_007_Dirichlet_A01_Analysis.md`
§ 9: replication would be considered supportive if `EXP_008` "reproduces
a test top-1 accuracy **within a few points** of `EXP_007`'s 59.68%."
This document evaluates the observed result against both that qualitative
criterion and the specific 1–1.5pp band as posed:

- **The observed gap is 2.85 percentage points** (test top-1: 59.68% →
  56.83%).
- This **exceeds a strict 1–1.5pp band** by roughly 2x.
- Whether it satisfies the looser "within a few points" framing is
  borderline: 2.85pp is larger than the 2.30pp gap `EXP_006` (α=0.5)
  showed relative to `EXP_004` (IID) — i.e., the seed-to-seed gap here is
  *larger* than a gap this project's own prior analysis classified as a
  real, reportable heterogeneity effect (§ 9 below expands on why this
  matters).

**Conclusion: the 1–1.5pp threshold, as posed, was not met.**

---

## 9. Why a 2.85pp Top-1 Difference Means Seed Variance Is Non-Negligible

Two runs that differ **only** in `seed` — same `alpha`, same model, same
optimizer, same round budget, same everything else — produced test top-1
accuracies 2.85 percentage points apart. This number needs to be judged
against the effect sizes this project has otherwise treated as
scientifically meaningful:

- `analysis/EXP_006_Dirichlet_A05_Analysis.md` classified `EXP_004`
  (IID) vs. `EXP_006` (α=0.5) — a **change in a real scientific
  variable** (`partition`/`dirichlet_alpha`) — as producing a "mild"
  effect of **2.30 percentage points**.
- The pure seed-to-seed noise observed here (2.85pp) is **larger** than
  that 2.30pp "mild heterogeneity" effect.

This means that, without a variance estimate, `EXP_006`'s 2.30pp gap
could not have been confidently distinguished from ordinary run-to-run
noise at this project's current single-seed-per-condition practice — a
possibility flagged as an open limitation in every analysis document in
this series but not, until now, quantified. § 5 above additionally shows
this noise is not purely an optimization-trajectory effect: the Dirichlet
partition realization itself varies substantially between seeds at
α=0.1 (coefficient of variation 10.9% vs. 23.4%), so at least part of the
2.85pp gap plausibly reflects `EXP_008` having drawn a harder partition,
not only different SGD/aggregation noise. Both sources of variance are
real properties of this experimental setup, not artifacts to be
explained away.

---

## 10. Two Seeds Are Insufficient for a Reliable Variance Estimate

`EXP_007` and `EXP_008` establish that seed-to-seed variance at α=0.1 is
non-negligible (§ 9) — they do **not** establish its magnitude reliably.
With only two data points:

- A range or single pairwise difference can be computed (2.85pp, § 2),
  but no standard deviation, standard error, or confidence interval can
  be meaningfully estimated — variance estimates computed from n=2 are
  themselves extremely unstable and can swing enormously with a third
  observation.
- There is no way to tell, from two points alone, whether `EXP_007` or
  `EXP_008` is closer to a "typical" outcome for this configuration, or
  whether both are draws from the tails of a wider distribution.
- Every qualitative feature identified as "replicated" or "not
  replicated" in § 7 is itself based on only two observations and could
  change with a third.

No claim in this document about the *magnitude* of seed variance at
α=0.1 should be treated as more than a lower bound (variance is at least
this large) until a third independent seed is available.

---

## 11. Recommend EXP_009: Third Strict Replication at Seed 2026

Per § 10, the next scientifically justified experiment is a **second**
replication, giving three independent seeds at the same configuration.
**`EXP_009`** should change only `seed` (`42`/`123` → `2026`) relative to
`EXP_007`/`EXP_008`, with `dirichlet_alpha=0.1` and every other field
identical — the same single-variable-change discipline used for
`EXP_007 → EXP_008`. This requires no new code, only a new config file
(not created by this document). With three points, a rough central
tendency (e.g. median or mean test top-1) and spread (e.g. range across
the three seeds) become meaningful for the first time in this project's
federated experiments, and the § 7 "did the late-block uptick replicate"
and "did the final-round-is-best-round feature replicate" questions can
be evaluated on 2-of-3 or 3-of-3 evidence rather than a single pairwise
comparison.

---

## 12. FedProx Implementation Should Wait for the Third-Seed Baseline

`research_plans/PHASE4_RESEARCH_STRATEGY.md` § 7 set an evidence bar for
implementing FedProx — a "substantially larger degradation than
`EXP_006`'s mild 2.30pp gap" — and `EXP_007` alone appeared to cross it
(8.84pp). **This document's finding changes that assessment:** § 9 shows
that a 2.85pp swing can arise from seed variance *alone*, at the same
`alpha`, with no change to the heterogeneity setting at all. A single
additional seed (`EXP_008`) already materially changed the picture (the
gap to `EXP_004` widened from 8.84pp to 11.69pp, and one qualitative
feature failed to replicate) — two points are not enough to know whether
the true FedAvg-under-α=0.1 effect is closer to `EXP_007`'s number,
`EXP_008`'s number, or something else entirely. **Implementing FedProx
now, on the strength of `EXP_007` alone, would risk building and
evaluating a mitigation against a problem whose size has not been
reliably measured.** Per § 11, `EXP_009` (a third independent seed)
should be completed first; only once three seeds give a stable-enough
picture of the α=0.1 effect size should the `research_plans/PHASE4_RESEARCH_STRATEGY.md`
§ 7 evidence bar be considered formally re-evaluated and, if still met,
acted on.
