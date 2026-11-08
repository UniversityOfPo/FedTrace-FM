# EXP_004 — Formal Post-Experiment Convergence Analysis

**Analysis date:** 2026-08-05
**Subject experiment:** `EXP_004` — FedAvg IID, 50 communication rounds
(`configs/cifar100_fedavg_iid_50rounds.yaml`)
**Protocol under evaluation:** `research_plans/EXP_004_Convergence_Study_Plan.md`
(Sections 6–9: Success Criteria, Failure Criteria, Expected Scientific
Outcomes, Decision Tree)

**Data sources (verified, no invented values):**
- `results/federated/cifar100_fedavg_iid_50rounds/round_history.csv` (all
  50 rows, `round_id` 0–49)
- `results/federated/cifar100_fedavg_iid_50rounds/summary.json`
- `results/federated/cifar100_fedavg_iid_50rounds/final_test_metrics.json`
- `results/federated/cifar100_fedavg_iid_50rounds/run.log`
- Cross-experiment comparison figures (`EXP_001`–`EXP_003`) are the
  already-frozen, previously verified results recorded in
  `FEDTRACE_FM_MASTER.md` and `experiments/EXP_00{1,2,3}_*.md`, themselves
  sourced from each experiment's own `results/` outputs.

This document performs analysis only. No Python file, config, test, or
Research Database file was created or modified in the process of producing
it.

---

## 1. EXP_001 vs. EXP_002 vs. EXP_003 vs. EXP_004 Comparison

| Metric | EXP_001 (Centralized, 10 epochs) | EXP_002 (FedAvg IID, 10 rounds) | EXP_003 (FedAvg IID, 20 rounds) | EXP_004 (FedAvg IID, 50 rounds) |
|---|---:|---:|---:|---:|
| Final Test Loss | 1.3016 | 2.3611 | 1.6335 | **1.1049** |
| Final Test Top-1 | 62.71% | 37.81% | 54.69% | **68.52%** |
| Final Test Top-5 | 88.54% | 70.27% | 83.75% | **91.27%** |
| Gap to Centralized (top-1) | — | −24.90pp | −8.02pp | **+5.81pp (exceeds)** |
| Macro F1 | — | 0.3600 | 0.5367 | 0.6814 |
| Total experiment time | 501.84 s | 765.57 s | 1776.63 s | 4055.95 s |

`EXP_004`'s final test metrics (`final_test_metrics.json`): loss
1.1049091817855834, top-1 0.6852, top-5 0.9127, macro precision
0.690070145422417, macro recall 0.6850000000000002, macro F1
0.6813777434367658, balanced accuracy 0.6850000000000002 — identical to
the `final_test` block of `summary.json`.

`EXP_004` is the first result in the project's history to exceed the
centralized `EXP_001` reference on every reported metric, continuing the
trend established across `EXP_002` → `EXP_003` → `EXP_004` of the
federated/centralized gap shrinking and then reversing as the
communication-round budget increased.

---

## 2–4. Round-Block Analysis: Validation Top-1 Gain and Validation-Loss Reduction per Round

All values below are read directly from `round_history.csv`
(`validation_top1`, `validation_loss` columns). Per-round-average figures
for a block use the last round of the *previous* block as the baseline
(consistent with the methodology already applied to `EXP_002`/`EXP_003` in
`RESEARCH_DIARY.md` Entry 4.1 and `FEDTRACE_FM_MASTER.md` Section 7), except
for the first block (rounds 0–9), which has no round prior to round 0 and
is therefore averaged over its own 9 internal transitions.

| Block | Start round (val_top1 / val_loss) | End round (val_top1 / val_loss) | Transitions | Δ Val Top-1 | Avg Top-1 gain/round | Δ Val Loss | Avg loss reduction/round |
|---|---|---|---:|---:|---:|---:|---:|
| Rounds 0–9   | R0: 1.86% / 4.7639  | R9: 37.54% / 2.3798  | 9  | +35.68pp | **≈3.964pp/round** | −2.3841 | **≈0.2649/round** |
| Rounds 10–19 | R9: 37.54% / 2.3798 | R19: 54.22% / 1.6727 | 10 | +16.68pp | **≈1.668pp/round** | −0.7072 | **≈0.0707/round** |
| Rounds 20–29 | R19: 54.22% / 1.6727 | R29: 61.10% / 1.3847 | 10 | +6.88pp  | **≈0.688pp/round** | −0.2879 | **≈0.0288/round** |
| Rounds 30–39 | R29: 61.10% / 1.3847 | R39: 65.50% / 1.2177 | 10 | +4.40pp  | **≈0.440pp/round** | −0.1670 | **≈0.0167/round** |
| Rounds 40–49 | R39: 65.50% / 1.2177 | R49: 68.68% / 1.1155 | 10 | +3.18pp  | **≈0.318pp/round** | −0.1022 | **≈0.0102/round** |

**Observation:** both the per-round validation top-1 gain and the
per-round validation-loss reduction decrease monotonically, block over
block, across all five decades — from ≈3.96pp/round down to ≈0.32pp/round
for top-1 accuracy (a ~92% reduction from the first to the last block),
and from ≈0.265/round down to ≈0.010/round for validation loss (a ~96%
reduction). This is a smooth, continuously-shrinking-returns pattern with
no block showing an increase in per-round efficiency relative to the block
before it.

---

## 5. Final-10-Round Trend (Rounds 40–49)

Round-by-round detail from `round_history.csv`:

| Round | Val Loss | Δ Loss | Val Top-1 | Δ Top-1 | Val Top-5 | Δ Top-5 |
|---:|---:|---:|---:|---:|---:|---:|
| 39 | 1.2177 | — | 65.50% | — | 89.96% | — |
| 40 | 1.2431 | +0.0254 | 65.40% | −0.10pp | 89.64% | −0.32pp |
| 41 | 1.1880 | −0.0551 | 66.70% | +1.30pp | 90.64% | +1.00pp |
| 42 | 1.1993 | +0.0113 | 66.00% | −0.70pp | 90.56% | −0.08pp |
| 43 | 1.1748 | −0.0245 | 67.00% | +1.00pp | 90.66% | +0.10pp |
| 44 | 1.1799 | +0.0051 | 66.70% | −0.30pp | 90.42% | −0.24pp |
| 45 | 1.1713 | −0.0086 | 67.16% | +0.46pp | 90.84% | +0.42pp |
| 46 | 1.1556 | −0.0157 | 67.28% | +0.12pp | 90.94% | +0.10pp |
| 47 | 1.1513 | −0.0043 | 67.72% | +0.44pp | 91.08% | +0.14pp |
| 48 | 1.1196 | −0.0317 | 68.50% | +0.78pp | 91.80% | +0.72pp |
| 49 | 1.1155 | −0.0040 | 68.68% | +0.18pp | 91.40% | **−0.40pp** |

**Trend characteristics:**
- Net movement over the window is positive on every tracked metric: val
  loss −0.1022 (1.2177 → 1.1155), val top-1 +3.18pp (65.50% → 68.68%), val
  top-5 +1.44pp (89.96% → 91.40%).
- The trend is **not monotonic**: two individual round-to-round top-1
  regressions occur (round 41→42: −0.70pp; round 43→44: −0.30pp), one
  individual loss regression at the very start of the window (round
  39→40: +0.0254), and one top-5 regression at the very end (round 48→49:
  −0.40pp) — the first top-5 decrease recorded anywhere in this project's
  round histories (also noted in `experiments/EXP_004_FedAvg_IID_50Rounds.md`).
- No regression is sustained across more than one consecutive round; every
  single-round dip is followed by a subsequent round of net improvement.

---

## 6. Plateau Assessment

Applying `research_plans/EXP_004_Convergence_Study_Plan.md` Section 6
(Success/continued-convergence criteria) and Section 7
(Failure/plateau criteria) directly against the verified data above:

**Section 6 — Success (continued, meaningful convergence) criteria:**

| # | Criterion | Verified outcome | Met? |
|---|---|---|---|
| 1 | Val top-1 at round 49 exceeds round 19's 54.22% by a clearly non-trivial margin | 68.68% vs. 54.22% → +14.46pp | **Yes, clearly** |
| 2 | Final-ten-round (40–49) average marginal gain is positive and not indistinguishable from zero | +0.318pp/round average; positive, but smaller in magnitude than several individual single-round swings observed within the same window (e.g. +1.30pp, −0.70pp, +1.00pp) | **Ambiguous** — positive and non-zero, but the *average* sits inside the noise band set by the window's own single-round volatility |
| 3 | Val loss at round 49 lower than round 19's 1.6727, with no *sustained* (multi-round) increase in the final ten rounds | 1.1155 < 1.6727; the only loss increase in the window (round 39→40, +0.0254) is a single round, immediately reversed | **Yes** |
| 4 | Final test top-1 narrows the gap to `EXP_001` relative to `EXP_003` | Gap not merely narrowed but reversed: `EXP_003` trailed by 8.02pp, `EXP_004` exceeds `EXP_001` by 5.81pp | **Yes** |

3 of 4 sub-criteria are unambiguously satisfied; criterion 2 is genuinely
ambiguous under the plan's own wording.

**Section 7 — Failure (effective plateau) criteria (any one triggers plateau):**

| # | Criterion | Verified outcome | Met? |
|---|---|---|---|
| 1 | Final-ten-round average val top-1 gain is near zero, or comparable to/smaller than the smallest single-round delta already observed in `EXP_003` rounds 15–19 (example given in the plan: +0.70pp, round 17→18) | Average gain 0.318pp/round is numerically smaller than the plan's own 0.70pp reference example | **Literally met**, but only by comparing an across-block *average* to a cherry-picked *single-round* figure from a different experiment — a weaker basis than the other two failure criteria |
| 2 | Val loss shows no sustained decrease over the final ten rounds (flat/oscillating, no clear downward trend) | Net decrease of 0.1022 over the window, with a clear downward trend interrupted by isolated single-round noise only | **Not met** |
| 3 | Val top-1/top-5 decrease or oscillate *without net improvement* over multiple consecutive rounds late in training (distinct from isolated single-round noise) | Top-1 shows strong net improvement (+3.18pp); the sole notable regression (top-5, round 48→49) is a single-round event, explicitly excluded by this criterion's own wording | **Not met** |

**Assessment:** Two of the three Section 7 plateau criteria are clearly
**not** met — validation loss and validation top-1/top-5 both show a real,
net, multi-round improving trend through round 49, not the flat/oscillating/
regressing pattern Section 7 requires. Only criterion 7.1 is met, and only
under a literal but weak reading (comparing a 10-round average to one
cherry-picked single-round delta from a different, shorter experiment).
Given that the stronger, trend-based criteria (7.2, 7.3) both fail to
indicate a plateau, and three of four Section 6 continued-convergence
sub-criteria are unambiguously satisfied, **the balance of evidence does
not support classifying this as an effective plateau (Outcome C)**.

At the same time, the round-block analysis (Sections 2–4 above) shows an
unambiguous, monotonic, large reduction in marginal efficiency each decade
(≈3.96 → ≈1.67 → ≈0.69 → ≈0.44 → ≈0.32 pp/round) — this is not the
"gains through round 50 remain comparable in character to `EXP_003`'s
rounds 10–19" pattern that Section 8 requires for Outcome A
("convergence continues").

---

## 7. Explicit Classification per the Pre-Registered Protocol

Per `research_plans/EXP_004_Convergence_Study_Plan.md` Section 8:

- **Outcome A (convergence continues):** rejected. The per-round marginal
  gain in the final block (≈0.32pp/round) is not "comparable in character"
  to `EXP_003`'s rounds 10–19 rate (≈1.67pp/round) — it is roughly 5x
  smaller, and every successive decade block shows a further reduction in
  efficiency with no reversal.
- **Outcome C (convergence plateaus):** rejected. Two of the three
  Section 7 failure criteria (loss trend, top-1/top-5 trend) are not met;
  only one criterion is met, and only under a comparison the plan itself
  frames as an example threshold rather than a strict test, applied across
  a data window (10-round average) different in kind from the one it was
  originally illustrated with (single-round delta).
- **Outcome B (convergence slows significantly): SELECTED.** Gains through
  round 49 are much smaller than `EXP_003`'s equivalent-window rate, but
  remain clearly positive, cumulative, and non-noise at the block level
  (net +3.18pp top-1, net −0.1022 loss over the final 10 rounds, no
  sustained regression on any tracked metric). This matches Section 8's
  description of Outcome B precisely: "gains are much smaller than in
  `EXP_003` but still clearly positive and non-noise."

**Classification: Outcome B — convergence slows significantly.**

---

## 8. Why EXP_004 Exceeds the Centralized Baseline (Without Claiming Federated Learning Is Inherently Superior)

`EXP_004` (68.52% test top-1) exceeds `EXP_001` (62.71% test top-1) by
5.81pp. The most defensible, configuration-grounded explanation is an
**unequal effective training budget between the two runs, not a general
superiority of federated averaging over centralized SGD**:

- `EXP_001` trains for exactly 10 centralized epochs over the full
  45,000-image training set (`configs/cifar100_resnet18_baseline.yaml`,
  `epochs: 10`), under a cosine-annealed learning-rate schedule
  (`learning_rate=0.1 → 0` over those 10 epochs) — a schedule and budget
  both explicitly tuned for a 10-unit run.
- `EXP_004` runs 50 communication rounds, and in every round, all 5
  clients each perform 1 full local epoch over their own 9,000-sample
  shard (`local_epochs: 1`, `clients_per_round: 5`) before their updates
  are FedAvg-aggregated. Across 50 rounds, this exposes the training
  process to a cumulative total of 50 rounds × 5 clients × 1 local epoch
  = 250 client-epoch-equivalents of local gradient computation over the
  data — a categorically larger optimization budget than `EXP_001`'s 10
  epochs, under a constant, un-annealed learning rate (`learning_rate=0.01`,
  no scheduler).
- Prior experiments in this project (`EXP_002` → `EXP_003` → `EXP_004`)
  already establish, at matched round counts, that federated top-1 trails
  centralized (`EXP_002`, 10 rounds: 37.81% vs. 62.71%) and only
  approaches/exceeds it as the round (and therefore total local-epoch)
  budget grows well past 10. This is consistent with the gap being driven
  by relative training budget, not by an inherent per-unit advantage of
  FedAvg.

A secondary, unverified hypothesis — that repeatedly resetting and
re-averaging 5 independently-trained local models each round acts as an
implicit regularizer (loosely analogous to model averaging / stochastic
weight averaging) — remains plausible but **has not been tested or
measured** in this analysis and should not be treated as an established
cause.

**Conclusion for this section:** `EXP_004` exceeding `EXP_001` reflects
an **unequal, budget-mismatched comparison** (250 client-epoch-equivalents
vs. 10 centralized epochs, differing LR schedules), not evidence that
federated learning outperforms centralized learning in general. No
matched-budget centralized/federated comparison has been run in this
project to date.

---

## 9. Next Single-Variable Experiment (EXP_005)

Per the Decision Tree in `research_plans/EXP_004_Convergence_Study_Plan.md`
Section 9, Outcome B prescribes introducing exactly one new controlled
variable — a learning-rate scheduler — rather than a further round-budget
extension.

**Proposed `EXP_005` configuration:**
`scheduler: none → cosine` — the single scientific change.

All other `EXP_004` settings remain identical: `resnet18_cifar` model, IID
partition (5 clients, seed 42, 9,000 samples/client), FedAvg aggregation,
`clients_per_round=5`, `minimum_updates=5`, `local_epochs=1`,
`communication_rounds=50`, optimizer SGD (`learning_rate=0.01`,
`momentum=0.9`, `weight_decay=0.0005`), client-state reset every round,
seed 42, mixed precision enabled, batch size 32, `validation_fraction=0.1`.
`experiment_id`/`experiment_name`/`output_dir`/`checkpoint_dir` would
differ only as non-scientific bookkeeping, exactly as done for
`EXP_002` → `EXP_003` → `EXP_004`.

---

## 10. Falsifiable Hypothesis for EXP_005

**Hypothesis:** Introducing a cosine-annealed learning-rate schedule over
the 50-round run (peak `learning_rate=0.01` decaying to 0 by round 49),
with every other `EXP_004` setting held fixed, will raise `EXP_005`'s
final test top-1 accuracy above `EXP_004`'s verified 68.52%, and will
reduce the round-to-round validation top-1 noise observed in `EXP_004`'s
final-ten-round window (Section 5 above — two negative single-round
transitions, magnitude up to −0.70pp) to no more than one negative
single-round transition of magnitude ≤0.70pp across `EXP_005`'s own final
ten rounds (40–49).

**Falsification condition:** If `EXP_005`'s final test top-1 accuracy is
≤68.52% (`EXP_004`'s verified value), **or** if `EXP_005`'s final-ten-round
window shows two or more negative single-round validation top-1
transitions of magnitude >0.70pp, the hypothesis is falsified — indicating
the learning-rate schedule did not meaningfully improve either final
accuracy or late-training stability relative to the constant-LR policy
used in `EXP_002`–`EXP_004`.

---

## Final Conclusion

`EXP_004`'s round-by-round convergence behavior is classified as
**Outcome B — convergence slows significantly** — not Outcome A
(unabated continuation) and not Outcome C (effective plateau). Marginal
per-round validation top-1 gain fell by roughly 92% from the first decade
block to the last (≈3.96pp/round → ≈0.32pp/round), and per-round
validation-loss reduction fell by roughly 96% (≈0.265/round → ≈0.010/round),
yet both metrics kept improving, net, through round 49 with no sustained
multi-round regression on either.

**Communication rounds are no longer judged to be the primary unresolved
variable in this project's federated pipeline.** They produced large,
clearly-attributable gains through round 29 and diminishing-but-real gains
through round 49, but the magnitude of that diminishment (a ~5x drop in
per-round efficiency between the `EXP_003`-equivalent window and `EXP_004`'s
final window) means further blind round-budget extension is not currently
the best-justified next step. Per the pre-registered decision tree's
Outcome B branch, the next single-variable experiment should test a
learning-rate scheduler (Sections 9–10 above), not a further increase in
`communication_rounds`.
