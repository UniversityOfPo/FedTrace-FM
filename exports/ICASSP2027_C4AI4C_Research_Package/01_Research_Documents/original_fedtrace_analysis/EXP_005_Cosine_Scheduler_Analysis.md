# EXP_005 — Cosine Round-Level Scheduler Analysis (EXP_004 vs. EXP_005)

**Analysis date:** 2026-08-05
**Subject experiments:** `EXP_004` (FedAvg IID, 50 rounds, constant learning
rate) vs. `EXP_005` (FedAvg IID, 50 rounds, round-level cosine learning-rate
schedule, Sprint 5.1/5.2)

**Data sources (verified, no invented values):**
- `results/federated/cifar100_fedavg_iid_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`
- `results/federated/cifar100_fedavg_iid_50rounds_cosine/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `run.log`

Per-round applied learning-rate values quoted below are read directly from
`run.log`'s `"round N learning rate set to ..."` log lines (written by
`FederatedClient._apply_round_learning_rate`), not recomputed by hand.
Round-block marginal-gain figures are simple arithmetic (a single
subtraction and division) applied directly to the verified
`validation_top1`/`validation_loss` columns of each run's
`round_history.csv`, using the same previous-block-endpoint-baseline
convention established in `analysis/EXP_004_Convergence_Analysis.md`.

This document performs analysis only. No code, config, test, or Research
Database file was created or modified in the process of producing it.

---

## 1. Controlled-Variable Confirmation

Per `configs/cifar100_fedavg_iid_50rounds_cosine.yaml`, the declared single
scientific variable is `round_lr_schedule` (`constant → cosine`), with
`min_learning_rate: 0.0001` as its required companion value (inert unless
`round_lr_schedule="cosine"`). Both runs' own `summary.json` confirm every
other tracked setting is identical:

| Field | EXP_004 | EXP_005 |
|---|---|---|
| `num_clients` / `clients_per_round` / `minimum_updates` | 5 / 5 / 5 | 5 / 5 / 5 |
| `communication_rounds_configured` / `_completed` | 50 / 50 | 50 / 50 |
| `local_epochs` | 1 | 1 |
| `aggregation_strategy` | fedavg | fedavg |
| `optimizer` / `learning_rate` | sgd / 0.01 | sgd / 0.01 |
| `batch_size` | 32 | 32 |
| `seed` | 42 | 42 |
| `total_parameters` | 11,220,132 | 11,220,132 |
| `client_sample_counts` | 9,000 × 5 | 9,000 × 5 |
| `gpu_name` | NVIDIA GeForce RTX 5060 Laptop GPU | NVIDIA GeForce RTX 5060 Laptop GPU |

**Empirical confirmation, not just config-level:** round 0 of both runs is
**numerically identical** in `round_history.csv` — validation loss
4.7639240234375, validation top-1 0.0186, validation top-5 0.073 in both
files. This is expected and correct: round 0's round-level learning rate is
`base_lr` under both `"constant"` and `"cosine"` (`compute_round_learning_rate`
returns exactly `base_lr` at `round_id=0` regardless of schedule), so round 0
is a genuine same-seed, same-partition, same-initial-weights reproduction —
the same internal consistency check already used for
`EXP_002 → EXP_003 → EXP_004`. From round 1 onward the two runs diverge
(EXP_005's round-1 applied learning rate was `0.00998983` per `run.log`,
already marginally below `0.01`), confirming the round-level schedule was
genuinely active and is the sole source of divergence.

---

## 2. Final Validation and Test Comparison

| Metric | EXP_004 (constant) | EXP_005 (cosine) | Δ (EXP_005 − EXP_004) |
|---|---:|---:|---:|
| Final test loss | 1.1049 | 1.2190 | +0.1141 (worse) |
| Final test top-1 | 68.52% | 65.12% | **−3.40pp** |
| Final test top-5 | 91.27% | 89.63% | **−1.64pp** |
| Final validation loss | 1.1155 | 1.2439 | +0.1283 (worse) |
| Final validation top-1 | 68.68% | 64.66% | **−4.02pp** |
| Final validation top-5 | 91.40% | 89.64% | **−1.76pp** |

`EXP_005` trails `EXP_004` on every final metric. `EXP_004` remains the
project's best federated result and the only one to exceed the centralized
`EXP_001` baseline (62.71% test top-1); `EXP_005`'s 65.12% still exceeds
`EXP_001`, but by a smaller margin (+2.41pp vs. `EXP_004`'s +5.81pp).

---

## 3. Best Validation Round Comparison

| | EXP_004 | EXP_005 |
|---|---|---|
| Best validation round | **49** (the final round) | **45** (not the final round) |
| Best validation top-1 | 68.68% | 64.86% |

This is a qualitative, not just quantitative, difference. `EXP_004`'s best
round is its last round — validation top-1 was still climbing (round 48→49:
+0.18pp) when training stopped. `EXP_005`'s validation top-1 peaked at
round 45 (64.86%) and then **declined slightly** over the remaining four
rounds — round 45→49: 64.86% → 64.84% → 64.76% → 64.78% → 64.66%, a net
**−0.20pp** regression from peak to final round (`round_history.csv`,
`validation_top1` column). Validation *loss* over that same window kept
decreasing marginally (round 45: 1.2472 → round 49: 1.2439), so this is a
noisy, near-flat plateau in accuracy rather than outright divergence — but
it is the first run in this project where the final round is not also the
best round.

---

## 4. Loss, Top-1, Top-5, Macro-Metric Comparison (Final Test)

| Metric | EXP_004 | EXP_005 | Δ |
|---|---:|---:|---:|
| Loss | 1.1049091817855834 | 1.2189941571235656 | +0.1141 |
| Top-1 | 0.6852 | 0.6512 | −0.0340 |
| Top-5 | 0.9127 | 0.8963 | −0.0164 |
| Macro precision | 0.6901 | 0.6499 | −0.0402 |
| Macro recall | 0.6850 | 0.6514 | −0.0336 |
| Macro F1 | 0.6814 | 0.6491 | −0.0323 |
| Balanced accuracy | 0.6850 | 0.6514 | −0.0336 |

Every macro metric moves in the same direction and by a comparable
magnitude (−0.032 to −0.040) — the cosine run's degradation is broad
across classes, not concentrated in a few, and is consistent with
under-training relative to `EXP_004` rather than a specific failure mode
(e.g. class collapse) that would show up as a macro-vs-overall divergence.

---

## 5. Round-by-Round Learning-Curve Comparison

**Applied round-level learning rate** (`run.log`, `EXP_005` only —
`EXP_004` has none, by design of `round_lr_schedule="constant"`):

| Round | 0 | 9 | 19 | 24 | 25 | 29 | 39 | 49 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP_005 LR | 0.01 | 0.00919854 | 0.00675956 | 0.00520866 | 0.00489134 | 0.00364159 | 0.001083 | 0.0001 |
| EXP_004 LR | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |

**Validation top-1 at the same round-block boundaries** (`round_history.csv`):

| Round | 0 | 9 | 19 | 29 | 39 | 49 |
|---|---:|---:|---:|---:|---:|---:|
| EXP_004 | 1.86% | 37.54% | 54.22% | 61.10% | 65.50% | 68.68% |
| EXP_005 | 1.86% | 38.00% | 53.80% | 61.32% | 64.40% | 64.66% |

**Average validation-top-1 gain per round, by decade block** (previous-block
endpoint as baseline, matching `analysis/EXP_004_Convergence_Analysis.md`'s
methodology):

| Block | EXP_004 (pp/round) | EXP_005 (pp/round) |
|---|---:|---:|
| Rounds 0–9 | ≈3.964 | ≈4.016 |
| Rounds 10–19 | ≈1.668 | ≈1.580 |
| Rounds 20–29 | ≈0.688 | ≈0.752 |
| Rounds 30–39 | ≈0.440 | ≈0.308 |
| Rounds 40–49 | ≈0.318 | **≈0.026** |

**Average validation-loss reduction per round, by decade block:**

| Block | EXP_004 (loss/round) | EXP_005 (loss/round) |
|---|---:|---:|
| Rounds 0–9 | ≈0.2649 | ≈0.2672 |
| Rounds 10–19 | ≈0.0707 | ≈0.0677 |
| Rounds 20–29 | ≈0.0288 | ≈0.0295 |
| Rounds 30–39 | ≈0.0167 | ≈0.0121 |
| Rounds 40–49 | ≈0.0102 | **≈0.0022** |

Through round 29, the two runs' block-level marginal gains are close and
even trade off which run is ahead (`EXP_005` slightly ahead in blocks 0–9
and 20–29; `EXP_004` slightly ahead in 10–19). The decisive divergence is
**rounds 30–49**, where `EXP_005`'s applied learning rate falls from
`0.00364` (round 29) to `0.0001` (round 49) — an over-90% reduction — while
`EXP_004`'s stays fixed at `0.01`. `EXP_005`'s rounds 40–49 marginal gain
(≈0.026pp/round) is roughly **12x smaller** than `EXP_004`'s equivalent
window (≈0.318pp/round), and its loss reduction in that window is roughly
**5x smaller** (0.0022 vs. 0.0102 per round).

---

## 6. Why Cosine Likely Underperformed in This Setting

The round-block data points to a specific, mechanistic explanation rather
than a general failure of cosine annealing:

- **The schedule reaches its minimum learning rate at exactly round 49**
  (by construction — `compute_round_learning_rate` anchors `round_id=0` to
  `base_lr` and `round_id=total_rounds-1` to `min_lr`), but it spends a
  large share of the round budget already close to that minimum: by round
  39 (10 rounds before the end), the applied learning rate is `0.001083` —
  **89% below** the constant rate `EXP_004` used for the same round. By
  round 29, it is already at `0.00364` (64% below).
- With `local_epochs=1` and every-round optimizer/momentum reset (the
  Sprint 4.1 client-state policy, unchanged and still active for both
  runs), each round's local training is a small, fixed number of SGD steps
  starting from zero momentum. At the very small learning rates the
  schedule reaches in its final third, those steps move the model's
  parameters by very little before the round's weights are averaged away
  again — consistent with the observed near-stall (rounds 40–49: +0.26pp
  total validation-top-1 gain, vs. `EXP_004`'s +3.18pp in the same window).
- `EXP_004`'s own convergence analysis
  (`analysis/EXP_004_Convergence_Analysis.md`) already showed marginal
  gains shrinking substantially by rounds 40–49 even at the *full* constant
  learning rate (Outcome B, "convergence slows significantly"). Compounding
  that already-diminishing-returns regime with a simultaneously
  near-zero learning rate leaves very little room for further improvement
  in `EXP_005`'s final third — which is exactly what the data shows.
- This also explains the Section 3 finding (best round at 45, not 49):
  once the learning rate is low enough that a round's update is
  comparable in scale to ordinary round-to-round noise, whether validation
  top-1 ticks up or down on any given round stops being a reliable signal
  of continued learning.

---

## 7. This Result Does Not Prove Cosine Scheduling Is Universally Inferior

This is a single-seed (42), single-configuration comparison of one specific
cosine parameterization (`min_learning_rate=0.0001`, no warmup, anchored to
exactly 50 rounds, applied at round granularity under `local_epochs=1` and
an every-round momentum reset) against one constant-learning-rate baseline,
on one dataset/partition/model/aggregation combination (CIFAR-100, IID,
ResNet-18, FedAvg). It does **not** establish that:

- A different `min_learning_rate` (e.g. closer to the base rate, so the
  schedule never falls as low as `0.0001`), a different decay shape, or an
  added warmup phase would perform the same way.
- Cosine scheduling is ineffective in federated learning generally —
  `EXP_001` (this project's own centralized baseline) used a cosine
  schedule successfully, and centralized deep learning broadly relies on
  learning-rate decay as standard practice.
- The result would hold under Non-IID partitioning, more local epochs per
  round, a persistent (non-reset) client optimizer-state policy, a
  different round budget, or a different model/aggregation strategy.
- No repeated-seed trials exist for either `EXP_004` or `EXP_005` in this
  project, so the exact size of the −3.40pp gap has not been checked
  against ordinary run-to-run noise — the gap is large enough, and the
  mechanistic explanation (Section 6) specific enough, to treat it as a
  real effect rather than noise, but this has not been formally quantified.

The verified, narrow conclusion is: **this specific cosine configuration,
under this specific federated training policy, underperformed the constant
baseline** — not a general claim about cosine annealing.

---

## 8. Which IID Protocol Is Frozen

**`EXP_004`'s configuration (constant `learning_rate=0.01`, no round-level
schedule, 50 communication rounds) remains the frozen IID FedAvg reference
protocol** for this project. It outperforms `EXP_005` on every reported
metric (Sections 2 and 4) and is the only run whose best validation round
is also its final round (Section 3), with no evidence of a late-training
stall. `EXP_005` is retained as a documented, informative negative result
(Section 6) — not adopted as a replacement default. Any future federated
CIFAR-100 IID experiment intended for comparison against the project's
best result should continue to cite `EXP_004`'s 68.52% test top-1 / 91.27%
test top-5, not `EXP_005`'s.

---

## 9. Is Further Scheduler Tuning Scientifically Justified Right Now?

Section 6 identifies a specific, plausible fix (e.g., a higher
`min_learning_rate`, so the schedule never decays far enough to stall
training, or a schedule with a warmup/later decay onset) — so further
scheduler tuning is not scientifically *unjustified* in principle; a
real, explained failure mode exists and a follow-up ablation could test it
directly. However, it is **not the recommended next priority**:

- `EXP_004` already meets and exceeds Phase 3's "federated ≈ centralized
  under IID" completion criterion (`PROJECT_ROADMAP.md`), so further
  optimizing the IID protocol's peak accuracy is refinement of an
  already-successful result, not resolution of an open completion
  criterion.
- Phase 3's one remaining, structurally required, not-yet-started item is
  Non-IID (Dirichlet/shard-based) client partitioning — this blocks
  progress toward the project's central research question (attribution
  and repair under realistic, non-IID, potentially adversarial client
  data) more directly than a better IID learning-rate schedule does.
- Per the same opportunity-cost reasoning already applied in
  `research_plans/EXP_004_Convergence_Study_Plan.md` and
  `research_plans/EXP_005_Cosine_Scheduler_Study_Plan.md`, each additional
  round-budget/scheduler experiment further delays Non-IID partitioning
  and the attack/attribution phases queued behind it.

**Recommendation:** defer further scheduler tuning to a later, explicitly
scoped ablation (if revisited at all before Phase 6+ foundation-model
work); prioritize Non-IID partitioning next.

---

## 10. Next Research Phase: Non-IID Partitioning

Per `PROJECT_ROADMAP.md` Phase 3's sole remaining unchecked completion
criterion and `FEDTRACE_FM_MASTER.md`'s "Pending Modules", the next phase
of work is:

1. **Implement Non-IID client data partitioning** — a Dirichlet- and/or
   shard-based partitioner added as a sibling function to the existing
   `partition_dataset_iid` in `src/data/federated_partition.py`, without
   modifying `partition_dataset_iid` itself (the same "extend, don't
   modify" discipline already applied throughout this project).
2. **Run a Non-IID federated CIFAR-100 experiment** using `EXP_004`'s
   frozen protocol (Section 8 above) as the controlled baseline — constant
   `learning_rate=0.01`, 50 rounds, FedAvg, 5 clients, `local_epochs=1` —
   changing only the partition strategy (IID → Non-IID), following this
   project's established single-variable-change discipline. Per this
   project's sequential experiment numbering (`EXP_001`–`EXP_005` already
   assigned), this would logically be **`EXP_006`**, though no plan or
   config for it exists yet.
3. **Compare against `EXP_004`'s IID reference** (68.52% test top-1, 91.27%
   test top-5) to characterize how much of a gap Non-IID partitioning
   introduces under otherwise-identical conditions — the direct federated
   analogue of how `EXP_002`/`EXP_003`/`EXP_004` characterized the
   round-budget gap against the centralized `EXP_001` baseline.
4. Only once a Non-IID baseline exists does Phase 4 (attack simulation)
   become well-motivated to start, since realistic label-flipping/backdoor/
   poisoning attacks are typically studied against non-IID client data,
   not the idealized IID case this project has used through `EXP_005`.

This is a pre-registration-style outline, not a run plan — per this
project's standing workflow, a dedicated `research_plans/EXP_006_*.md`
protocol (objective, hypothesis, exact frozen-baseline configuration,
success/failure criteria) would be written and reviewed before any Non-IID
partitioning code or experiment config is created.
