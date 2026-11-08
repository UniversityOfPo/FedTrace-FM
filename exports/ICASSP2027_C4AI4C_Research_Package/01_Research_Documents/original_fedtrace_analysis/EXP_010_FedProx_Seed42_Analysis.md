# EXP_010 — FedProx vs. FedAvg, Paired Seed 42 Analysis (Dirichlet α=0.1, μ=0.01)

**Analysis date:** 2026-08-05
**Subject experiments (paired, seed 42 only):**
- `EXP_007` — FedAvg, Dirichlet α=0.1, 50 rounds, seed 42
- `EXP_010` — FedProx (μ=0.01), Dirichlet α=0.1, 50 rounds, seed 42

**Data sources (verified, no invented values):**
- `results/federated/cifar100_fedavg_dirichlet_a01_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`
- `results/federated/cifar100_fedprox_dirichlet_a01_mu001_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`

Round-block marginal-gain figures use the same previous-block-endpoint-baseline
convention established in `analysis/EXP_004_Convergence_Analysis.md` and
reused throughout this series. This document performs analysis only. No
code, config, test, README, or Research Database file was created or
modified in the process of producing it.

**This is one paired seed only (seed 42 of the pre-registered 42/123/2026
triad in `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md`).** Per
that plan's § 22, no claim of statistical significance is made anywhere
in this document.

---

## 1. Pairing Verification

Per `configs/cifar100_fedprox_dirichlet_a01_mu001_50rounds.yaml`, the
**only** declared scientific change relative to `EXP_007` is
`local_training_method` (`fedavg → fedprox`), with `fedprox_mu: 0.01` as
its required companion. Both runs' own `summary.json` confirm every other
tracked setting matches exactly: `dataset="CIFAR-100"`, `seed=42`,
`partition="dirichlet"`, `num_clients=5`, `clients_per_round=5`,
`minimum_updates=5`, `communication_rounds_configured/_completed=50/50`,
`local_epochs=1`, `optimizer="sgd"`, `learning_rate=0.01`, `batch_size=32`,
`total_parameters=11,220,132`, `aggregation_strategy="fedavg"` (unchanged
— § 8 below), same GPU (NVIDIA GeForce RTX 5060 Laptop GPU).

**Partition realization is confirmed identical**, not merely configured
identically: `client_partition_summary.csv` reports the exact same
per-client sample counts in both runs — `client_000`: 10,883; `client_001`:
8,003; `client_002`: 8,868; `client_003`: 8,574; `client_004`: 8,672. This
is a genuine paired comparison at the sample-composition level, not just
at the config level.

**Round 0 is *not* numerically identical between the two runs**
(`EXP_007`: validation loss 4.77608671875, top-1 1.06%; `EXP_010`:
validation loss 4.6875078125, top-1 1.28%) — and this is expected,
**not** a pairing failure. Unlike the `EXP_004`→`EXP_005` round-level-LR
comparison (where round 0's cosine learning rate coincided exactly with
the constant rate by construction), FedProx's proximal term is active
from round 0 onward: even though both clients start round 0 from the
same random initial global model, the proximal penalty begins pulling
each client's local update back toward that shared starting point
immediately, so the two runs' local training already diverges within
round 0 itself. All 50 rounds completed in both runs with 5/5 successful
clients and 0 rejected updates.

---

## 2. Final Validation and Test Comparison

| Metric | EXP_007 (FedAvg) | EXP_010 (FedProx, μ=0.01) | Δ (010 − 007) |
|---|---:|---:|---:|
| Final validation loss | 1.4239414876937866 | 1.4156399635314942 | −0.0083 (better) |
| Final validation top-1 | 59.22% | 59.02% | **−0.20pp** |
| Final validation top-5 | 88.26% | 88.58% | **+0.32pp** |
| Best validation round | 49 (final) | 49 (final) | — |
| Best validation top-1 | 59.22% | 59.02% | −0.20pp |
| Final test loss | 1.3758617393493653 | 1.3706392387390136 | −0.0052 (better) |
| Final test top-1 | 59.68% | 59.63% | **−0.05pp** |
| Final test top-5 | 88.65% | 88.59% | **−0.06pp** |
| Macro precision | 0.6509177318450269 | 0.6460172564159314 | −0.0049 |
| Macro recall | 0.5965 | 0.5963 | −0.0002 |
| Macro F1 | 0.5875828060131298 | 0.5858452197428362 | −0.0017 |
| Balanced accuracy | 0.5965 | 0.5963 | −0.0002 |

Both runs' best validation round is the final round (49) — the pattern
`EXP_007` already showed on its own is unchanged under FedProx.

---

## 3. Quantified FedProx-Minus-FedAvg Differences

| Metric | Δ (FedProx − FedAvg) | Direction |
|---|---:|---|
| Test loss | −0.0052225 | FedProx slightly lower (better) |
| Test top-1 | −0.05pp | FedProx slightly lower (worse) |
| Test top-5 | −0.06pp | FedProx slightly lower (worse) |
| Macro precision | −0.0049 | FedProx slightly lower (worse) |
| Macro recall | −0.02pp | FedProx slightly lower (worse) |
| Macro F1 | −0.0017 | FedProx slightly lower (worse) |
| Balanced accuracy | −0.02pp | FedProx slightly lower (worse) |

Every difference is small and mixed in sign relative to loss vs.
accuracy-style metrics (test loss marginally improves; every accuracy/F1
metric marginally declines by a fraction of a percentage point). For
scale: **the largest accuracy-metric difference here (test top-5, 0.06pp)
is roughly 1/50th the size of the 2.85-percentage-point seed-to-seed
noise floor already measured for plain FedAvg alone** at this exact
`alpha` (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` § 3, seeds
42/123/2026). None of these differences approach a magnitude that would
be distinguishable from ordinary run-to-run variation.

---

## 4. Round-by-Round Convergence and Final-Decade Behavior

**Validation top-1 at decade-block boundaries:**

| Round | 0 | 9 | 19 | 29 | 39 | 49 |
|---|---:|---:|---:|---:|---:|---:|
| EXP_007 (FedAvg) | 1.06% | 30.24% | 46.00% | 52.44% | 55.54% | 59.22% |
| EXP_010 (FedProx) | 1.28% | 28.66% | 44.62% | 52.46% | 55.92% | 59.02% |

**Average validation-top-1 gain per round, by decade block:**

| Block | EXP_007 (pp/round) | EXP_010 (pp/round) |
|---|---:|---:|
| Rounds 0–9 | ≈3.242 | ≈3.042 |
| Rounds 10–19 | ≈1.576 | ≈1.596 |
| Rounds 20–29 | ≈0.644 | ≈0.784 |
| Rounds 30–39 | ≈0.310 | ≈0.346 |
| Rounds 40–49 | ≈0.368 | ≈0.310 |

**Average validation-loss reduction per round, by decade block:**

| Block | EXP_007 (loss/round) | EXP_010 (loss/round) |
|---|---:|---:|
| Rounds 0–9 | ≈0.2359 | ≈0.2190 |
| Rounds 10–19 | ≈0.0746 | ≈0.0735 |
| Rounds 20–29 | ≈0.0242 | ≈0.0313 |
| Rounds 30–39 | ≈0.0149 | ≈0.0135 |
| Rounds 40–49 | ≈0.0092 | ≈0.0119 |

Neither run consistently leads the other, block by block, on either
metric — each is ahead in some blocks and behind in others, with no
sustained pattern. Notably, `EXP_007`'s final-block marginal-gain uptick
(rounds 30–39 → 40–49: 0.310 → 0.368, flagged as an unusual,
seed-specific feature in `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`
§ 6) does **not** appear in `EXP_010` under the same seed and partition
(0.346 → 0.310, an ordinary decrease) — consistent with that feature
already having been shown to be seed-dependent rather than a property of
`alpha=0.1` itself.

**Final-decade (rounds 40–49) round-to-round detail (`EXP_010`):**
validation top-1 moves 55.92% → 55.68% → 56.20% → 56.40% → 58.50% →
58.04% → 58.10% → 57.88% → 57.68% → 58.54% → 59.02%, with **4 negative
single-round transitions** out of 10 (rounds 40, 44, 46, 47) — the same
count of negative transitions `EXP_007`'s equivalent window showed (4 of
10). FedProx at this `mu` shows no visible reduction in late-round
round-to-round volatility relative to plain FedAvg in this single paired
seed.

**Note on training loss (per this task's explicit instruction):**
`round_history.csv`'s `train_loss` column is **not** compared between
these two runs anywhere in this document. `EXP_010`'s logged `train_loss`
includes the `(mu/2)*||w-w_global||^2` proximal penalty term (documented
in `src/training/engine.py::train_one_epoch`'s docstring), while
`EXP_007`'s does not — the two columns measure different quantities and
are not directly comparable as equivalent training objectives. Only
validation and test metrics (unaffected by this difference) are compared
above.

---

## 5. Does Seed 42 Show Improvement, Degradation, or Neutrality?

**Effectively neutral.** Every validation/test metric difference in §§ 2–3
is small, several are in the "improved" direction (test loss, test
loss-derived metrics) and several are in the "declined" direction (test
top-1/top-5, macro metrics), and none approaches a magnitude
distinguishable from the already-measured 2.85-percentage-point
FedAvg-only seed-to-seed noise floor. This is neither a meaningful
improvement nor a meaningful degradation — at seed 42, FedProx (μ=0.01)
produced a result statistically indistinguishable, by inspection, from
plain FedAvg on every reported metric.

---

## 6. A Plausible (Not Demonstrated) Reason: `local_epochs=1`

**This section states a hypothesis, not a demonstrated causal
explanation** — no controlled experiment isolating this variable exists
in this project yet.

FedProx's proximal term exists to counteract *client drift*: the
divergence a client's local model accumulates from the global model over
many local optimization steps before the next synchronization. This
project's federated configuration has used `local_epochs=1` in every
experiment to date (`EXP_002`–`EXP_010`) — each client takes only one
local epoch's worth of SGD steps per round before re-synchronizing, and
(per the unchanged Sprint 4.1 client-state policy) local optimizer
momentum is discarded every round regardless of training method. Under
these conditions, the opportunity for a client's local model to drift far
from `w_global` within a single round is inherently limited *before*
FedProx is even applied — `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md`
§ 10 raised exactly this consideration when justifying `mu=0.01` as a
moderate (not aggressive) starting value. If drift is already small under
`local_epochs=1`, the proximal term has comparatively little excess drift
left to correct, which is *consistent with* (but does not by itself
prove) the near-neutral result observed here. Other explanations — e.g.
`mu=0.01` simply being too small at this `alpha`, or this specific seed's
partition not being a case where drift is the dominant limiting factor —
remain equally plausible and are not ruled out by this single data point.

---

## 7. One Paired Seed Cannot Determine FedProx Efficacy

This document compares exactly one paired seed (42) out of the three
pre-registered in `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md`
§ 3 (42, 123, 2026). `analysis/EXP_008_Replication_Analysis.md` and
`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` already demonstrated,
for plain FedAvg at this same `alpha=0.1`, that a single seed can show a
qualitatively different picture from what a second or third seed reveals
(e.g. a pattern present in 2 of 3 seeds, absent in the third). There is no
reason to expect FedProx-vs-FedAvg comparisons to be less
seed-sensitive. **This single paired result cannot establish whether
FedProx is neutral, mildly helpful, or mildly harmful in general at this
configuration** — it establishes only that seed 42 shows a
near-neutral paired result.

---

## 8. Recommendation: Complete Seeds 123 and 2026 Before Tuning

Per `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §§ 3–4 and 19,
the next step is to **complete the two remaining pre-registered paired
FedProx runs at seeds 123 and 2026** (matching `EXP_008` and `EXP_009`
respectively), using the same `mu=0.01` and every other setting unchanged.
Only once all three paired differences are available should any
conclusion be drawn about FedProx's efficacy at this configuration, per
the plan's per-seed and aggregate comparison rules (§§ 14–15) and its
explicit prohibition on claiming significance from three seeds (§ 22).

**No `mu` sweep is proposed here.** Per the plan § 11, one `mu` is
sufficient for this first controlled study, and per this task's explicit
instruction, this document does not propose one. Changing `mu` or
`local_epochs` before the three-seed paired comparison is complete would
introduce a new variable before the current one-variable comparison has
even been fully run, and would not resolve the open question in § 7 —
completing the pre-registered seeds is the correct next step, not
adjusting the method.

---

## Note on Runtime

`EXP_010` took 5,896.86 seconds (≈98.3 minutes) versus `EXP_007`'s
4,136.48 seconds (≈68.9 minutes) — about 43% longer. This is larger than
the "a few percent" overhead `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md`
§ 16 anticipated from the proximal-term computation alone, and plausibly
reflects some amount of incidental machine-level variance (this project's
prior analyses have repeatedly noted similar run-to-run duration
differences on this single local workstation not attributable to the
scientific variable under test) in addition to the proximal term's actual
compute cost. This document does not attempt to separate those two
contributions from a single paired observation.
