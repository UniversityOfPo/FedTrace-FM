# EXP_011 — FedProx vs. FedAvg, Paired Seed 123 Analysis (Dirichlet α=0.1, μ=0.01)

**Analysis date:** 2026-08-07
**Subject experiments (paired, seed 123 only):**
- `EXP_008` — FedAvg, Dirichlet α=0.1, 50 rounds, seed 123
- `EXP_011` — FedProx (μ=0.01), Dirichlet α=0.1, 50 rounds, seed 123

**Data sources (verified, no invented values):**
- `results/federated/cifar100_fedavg_dirichlet_a01_seed123_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`
- `results/federated/cifar100_fedprox_dirichlet_a01_mu001_seed123_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`
- `configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml`,
  `configs/cifar100_fedprox_dirichlet_a01_mu001_seed123_50rounds.yaml`

Round-block marginal-gain figures use the same previous-block-endpoint-baseline
convention established in `analysis/EXP_004_Convergence_Analysis.md` and reused
throughout this series (`analysis/EXP_010_FedProx_Seed42_Analysis.md` §4 in
particular). This document performs analysis only. No code, config, test,
README, or Research Database file was created or modified in the process of
producing it.

**This is one paired seed only (seed 123 of the pre-registered 42/123/2026
triad in `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md`).** Per that
plan's §22, no claim of statistical significance is made anywhere in this
document. This document also compares its seed-123 finding against the
already-published seed-42 finding (`analysis/EXP_010_FedProx_Seed42_Analysis.md`),
but explicitly does **not** combine the two into a single final conclusion —
that step is reserved for after seed 2026 is also available.

---

## 1. Pairing Verification

Comparing `configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml` and
`configs/cifar100_fedprox_dirichlet_a01_mu001_seed123_50rounds.yaml` directly,
the **only** declared differences are `local_training_method` (`fedavg →
fedprox`) and its required companion `fedprox_mu` (absent/inert → `0.01`).
Every other field is byte-identical: `num_clients=5`, `clients_per_round=5`,
`minimum_updates=5`, `communication_rounds=50`, `local_epochs=1`,
`batch_size=32`, `num_workers=2`, `model=resnet18_cifar`, `num_classes=100`,
`optimizer=sgd`, `learning_rate=0.01`, `momentum=0.9`, `weight_decay=0.0005`,
`scheduler=none`, `mixed_precision=true`, `aggregation=fedavg`,
`partition=dirichlet`, `dirichlet_alpha=0.1`, `validation_fraction=0.1`,
`evaluate_every=1`, `checkpoint_every=1`, `seed=123`,
`fail_round_on_client_error=true`, `continue_after_failed_round=false`.

Both runs' own `summary.json` confirm this at execution time: `dataset="CIFAR-100"`,
`seed=123`, `partition="dirichlet"`, `num_clients=5`, `clients_per_round=5`,
`minimum_updates=5`, `communication_rounds_configured/_completed=50/50`,
`local_epochs=1`, `optimizer="sgd"`, `learning_rate=0.01`, `batch_size=32`,
`total_parameters=11,220,132`, `aggregation_strategy="fedavg"` (unchanged —
FedProx modifies only the client-side local objective, never the
server-side `AggregationStrategy`, per plan §8), same GPU (NVIDIA GeForce
RTX 5060 Laptop GPU).

**Partition realization is confirmed identical**, not merely configured
identically: `client_partition_summary.csv` reports the exact same
per-client sample counts in both runs — `client_000`: 6,024; `client_001`:
9,628; `client_002`: 7,202; `client_003`: 10,361; `client_004`: 11,785
(sum 45,000). This is the same partition realization used by `EXP_008`,
confirming a genuine paired comparison at the sample-composition level, not
just at the config level — and it matches the pre-registered 42/123/2026
paired design in `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §3.

**Round 0 is *not* numerically identical between the two runs**
(`EXP_008`: validation loss 4.88488203125, top-1 1.32%; `EXP_011`: validation
loss 4.81584765625, top-1 1.36%) — as with the seed-42 pair, this is
expected, not a pairing failure: FedProx's proximal term is active from
round 0 onward, so even starting from the same random global initialization,
the two runs' local training diverges within round 0 itself. All 50 rounds
completed in both runs with 5/5 successful clients and 0 rejected updates.

**Conclusion: the only scientific variable that differs between `EXP_008`
and `EXP_011` is the local training objective (FedAvg vs. FedProx μ=0.01).**
All other factors (model, data split, partition realization, optimizer
hyperparameters, round/epoch counts, aggregation rule, seed) are held fixed.

---

## 2. Final Validation and Test Comparison

| Metric | EXP_008 (FedAvg) | EXP_011 (FedProx, μ=0.01) | Δ (011 − 008) |
|---|---:|---:|---:|
| Final validation loss | 1.4697468055725098 | 1.4710012462615967 | +0.0013 (worse) |
| Final validation top-1 | 57.78% | 57.62% | **−0.16pp** |
| Final validation top-5 | 87.98% | 87.42% | **−0.56pp** |
| Best validation round | 47 | 49 (final) | — |
| Best validation top-1 | 58.26% | 57.62% | **−0.64pp** |
| Final test loss | 1.4654396366119384 | 1.4718453573226928 | +0.0064 (worse) |
| Final test top-1 | 56.83% | 57.19% | **+0.36pp** |
| Final test top-5 | 87.79% | 87.36% | **−0.43pp** |
| Macro precision | 0.6337611188249366 | 0.6274460728730231 | −0.0063 |
| Macro recall | 0.5683 | 0.5719 | +0.0036 |
| Macro F1 | 0.5485726596416799 | 0.5515867193058298 | +0.0030 |
| Balanced accuracy | 0.5683 | 0.5719 | +0.0036 |

Note that `EXP_008`'s best validation round (47) is *not* its final round —
round 48 dips to 56.40% before round 49 recovers to 57.78%, and round 47's
58.26% is never re-reached. `EXP_011`'s best validation round is its final
round (49, 57.62%), matching the pattern both seed-42 runs showed
(`analysis/EXP_010_FedProx_Seed42_Analysis.md` §2). This best-round
difference is a property of `EXP_008`'s own late-round fluctuation, not
evidence of a FedProx effect by itself.

---

## 3. Quantified FedProx-Minus-FedAvg Differences

| Metric | Δ (FedProx − FedAvg) | Direction |
|---|---:|---|
| Test loss | +0.0064057 | FedProx higher (worse) |
| Test top-1 | +0.36pp | FedProx higher (better) |
| Test top-5 | −0.43pp | FedProx lower (worse) |
| Macro precision | −0.0063150 | FedProx lower (worse) |
| Macro recall | +0.36pp | FedProx higher (better) |
| Macro F1 | +0.0030141 | FedProx higher (better) |
| Balanced accuracy | +0.36pp | FedProx higher (better) |

As at seed 42, the differences are small and mixed in sign — but unlike
seed 42, here it is test **loss** (not accuracy) that moves in the
"worse" direction while top-1, recall, F1, and balanced accuracy move
"better." The largest single difference (test top-5, 0.43pp; best-validation
top-1, 0.64pp) is still well below the 2.85-percentage-point seed-to-seed
noise floor already measured for plain FedAvg alone at this exact `alpha`
(`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` §3, seeds 42/123/2026) —
roughly 1/4 to 1/6 of that floor, versus roughly 1/50th at seed 42. **The
seed-123 differences are noticeably larger in absolute magnitude than the
seed-42 differences (§8 below), while still remaining inside the established
noise floor.**

---

## 4. Round-by-Round Convergence and Final-Decade Behavior

**Validation top-1 at decade-block boundaries:**

| Round | 0 | 9 | 19 | 29 | 39 | 49 |
|---|---:|---:|---:|---:|---:|---:|
| EXP_008 (FedAvg) | 1.32% | 30.54% | 44.82% | 51.92% | 54.78% | 57.78% |
| EXP_011 (FedProx) | 1.36% | 29.64% | 43.88% | 51.22% | 54.00% | 57.62% |

**Average validation-top-1 gain per round, by decade block:**

| Block | EXP_008 (pp/round) | EXP_011 (pp/round) |
|---|---:|---:|
| Rounds 0–9 | ≈3.247 | ≈3.142 |
| Rounds 10–19 | ≈1.428 | ≈1.424 |
| Rounds 20–29 | ≈0.710 | ≈0.734 |
| Rounds 30–39 | ≈0.286 | ≈0.278 |
| Rounds 40–49 | ≈0.300 | ≈0.362 |

**Average validation-loss reduction per round, by decade block:**

| Block | EXP_008 (loss/round) | EXP_011 (loss/round) |
|---|---:|---:|
| Rounds 0–9 | ≈0.25113 | ≈0.23578 |
| Rounds 10–19 | ≈0.06584 | ≈0.06688 |
| Rounds 20–29 | ≈0.02506 | ≈0.02781 |
| Rounds 30–39 | ≈0.01450 | ≈0.01545 |
| Rounds 40–49 | ≈0.01011 | ≈0.01214 |

Neither run consistently leads the other block by block. Notably, **both**
`EXP_008` and `EXP_011` show a final-block marginal-gain uptick relative to
the rounds-30–39 block (top-1: 0.286→0.300 and 0.278→0.362 respectively),
with `EXP_011`'s uptick larger than `EXP_008`'s. This is a different pattern
from the seed-42 pair, where `EXP_007` showed an uptick (0.310→0.368) but
`EXP_010` did not (0.346→0.310, an ordinary decrease). This project's prior
analyses (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` §6) already
established that this late-round uptick is seed-dependent even for plain
FedAvg alone (present in 2 of 3 FedAvg-only seeds, absent in the third); its
presence or absence in these paired FedProx comparisons is consistent with
that same seed-dependence rather than a FedProx-specific effect, and no
causal claim is made here.

**Final-decade (rounds 40–49) round-to-round detail:**

`EXP_008` validation top-1: 54.78% → 55.90% → 56.24% → 55.58% → 55.76% →
57.58% → 57.20% → 57.38% → 58.26% → 56.40% → 57.78%, with **3 negative
single-round transitions** out of 10 (rounds 41→42, 44→45, 47→48).

`EXP_011` validation top-1: 54.00% → 55.82% → 55.18% → 55.34% → 56.20% →
56.90% → 56.42% → 56.66% → 57.22% → 57.20% → 57.62%, with **3 negative
single-round transitions** out of 10 (rounds 40→41, 44→45, 47→48).

Both runs show the same count (3 of 10) of negative single-round transitions
in the final decade — a smaller count than either seed-42 run showed (4 of
10 each, per `analysis/EXP_010_FedProx_Seed42_Analysis.md` §4). FedProx at
this `mu` shows no visible reduction (nor increase) in late-round
round-to-round volatility relative to plain FedAvg within this single
paired seed.

**Note on training loss (per this task's explicit instruction):**
`round_history.csv`'s `train_loss` column is **not** compared between these
two runs anywhere in this document. `EXP_011`'s logged `train_loss` includes
the `(mu/2)*||w-w_global||^2` proximal penalty term (documented in
`src/training/engine.py::train_one_epoch`'s docstring), while `EXP_008`'s
does not — the two columns measure different quantities and are not
directly comparable as equivalent training objectives. Only validation and
test metrics (unaffected by this difference) are compared above.

---

## 5. Does Seed 123 Show Improvement, Degradation, or Neutral/Mixed Effect?

**Effectively neutral/mixed.** Test top-1, macro recall, macro F1, and
balanced accuracy all move in the "improved" direction (+0.30–0.36pp/+0.003–0.004
absolute); test loss, test top-5, and macro precision all move in the
"declined" direction (test top-5 by −0.43pp being the largest single
accuracy-metric move). No individual metric's difference approaches the
2.85-percentage-point FedAvg-only seed-to-seed noise floor (the largest,
best-validation top-1 at −0.64pp, is roughly 22% of that floor — larger than
seed 42's largest relative share but still well inside it). This is neither
a meaningful improvement nor a meaningful degradation at seed 123: it is a
small, mixed-sign result, indistinguishable by inspection from ordinary
run-to-run variation at this configuration.

---

## 6. Comparison with the Seed-42 Result (Direction Only — Not Combined)

| Metric | Seed 42 direction (Δ) | Seed 123 direction (Δ) | Same direction? |
|---|---|---|---|
| Test loss | better (−0.0052) | worse (+0.0064) | **No** |
| Test top-1 | worse (−0.05pp) | better (+0.36pp) | **No** |
| Test top-5 | worse (−0.06pp) | worse (−0.43pp) | Yes |
| Macro precision | worse (−0.0049) | worse (−0.0063) | Yes |
| Macro recall | worse (−0.02pp) | better (+0.36pp) | **No** |
| Macro F1 | worse (−0.0017) | better (+0.0030) | **No** |
| Balanced accuracy | worse (−0.02pp) | better (+0.36pp) | **No** |

Only 2 of 7 compared metrics (test top-5, macro precision) agree in sign
between the two seeds; the remaining 5 flip sign. Where the sign does agree
(test top-5, macro precision), the seed-123 magnitude is 5–7× larger than
the seed-42 magnitude for the same metric. **This comparison is reported for
transparency only and is explicitly not combined into a single
FedAvg-vs-FedProx conclusion** — per the pre-registered plan, that step
requires all three paired seeds.

---

## 7. Consistency of Direction vs. Magnitude vs. Ordinary Seed Variability

These are three distinct questions and must not be conflated:

- **Direction consistency**: Low. As shown in §6, a majority of the compared
  metrics (5 of 7) do not even agree in sign between seed 42 and seed 123.
  This is a stronger signal of instability than magnitude alone would
  suggest.
- **Magnitude**: Small in both seeds, but larger at seed 123 than seed 42 for
  every metric compared (e.g. test top-1 |Δ| 0.05pp → 0.36pp; test top-5
  |Δ| 0.06pp → 0.43pp). Still, at both seeds every magnitude sits well below
  the 2.85pp FedAvg-only noise floor.
- **Ordinary seed variability**: Given (a) the sign flips in §6 and (b) that
  every observed FedProx-vs-FedAvg difference at both seeds is smaller than
  the noise floor already measured for FedAvg *alone* across seeds
  (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md`), the pattern so far is
  most consistent with these differences being dominated by ordinary
  partition-realization/training-trajectory seed variability rather than by
  a real, direction-stable FedProx effect. This is a description of the
  current two-seed evidence, not a conclusion — see §8.

---

## 8. Two Paired Seeds Are Insufficient for a Final Decision

This document compares two paired seeds (42, 123) out of the three
pre-registered in `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §3
(42, 123, 2026). The direction instability documented in §§6–7 makes this
point sharper, not weaker, than it was after seed 42 alone: with only one
seed, a mixed-but-small result was already inconclusive; with two seeds
disagreeing in sign on most metrics, it is not yet possible to say whether
seed 2026 will echo seed 42, echo seed 123, or show a third pattern. **Two
paired seeds cannot establish whether FedProx (μ=0.01) is neutral, mildly
helpful, or mildly harmful in general at this configuration** — they
establish only that neither of the two available seeds shows a meaningful
effect individually, and that the two do not agree with each other on
direction for most metrics.

---

## 9. Recommendation: Complete Seed 2026 Before Any Tuning

Per `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §§3–4 and 19, the
pre-registered next step is unchanged by this analysis: **complete the third
and final pre-registered paired FedProx run at seed 2026** (matching
`EXP_009`), using the same `mu=0.01` and every other setting unchanged. Only
once all three paired differences are available should any conclusion be
drawn about FedProx's efficacy at this configuration, per the plan's
per-seed and aggregate comparison rules (§§14–15) and its explicit
prohibition on claiming significance from three seeds (§22).

**No `mu` sweep is proposed here. `local_epochs` is not changed. No new
algorithm is proposed.** The direction instability observed between seeds 42
and 123 (§§6–7) is a reason to obtain the third pre-registered seed before
drawing any conclusion, not a reason to deviate from the pre-registered
single-variable design by introducing a second variable (a different `mu`,
a different `local_epochs`) before that design has even been completed.

---

## Note on Runtime

`EXP_011` took 5,654.98 seconds (≈94.2 minutes) versus `EXP_008`'s 4,008.92
seconds (≈66.8 minutes) — about 41% longer, consistent with the ≈43%
overhead already observed between `EXP_010` and `EXP_007` at seed 42
(`analysis/EXP_010_FedProx_Seed42_Analysis.md`, "Note on Runtime"). This
document does not attempt to separate proximal-term compute cost from
incidental machine-level variance from a single paired observation, as
before.
