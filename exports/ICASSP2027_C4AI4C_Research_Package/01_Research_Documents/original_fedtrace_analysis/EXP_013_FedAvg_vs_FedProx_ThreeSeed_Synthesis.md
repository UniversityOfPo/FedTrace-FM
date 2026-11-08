# EXP_013 — FedAvg vs. FedProx (μ=0.01), Three-Seed Paired Synthesis (Dirichlet α=0.1)

**Analysis date:** 2026-08-07
**This is a scientific synthesis document, not a new experiment.** It
combines the three already-completed, already-analyzed paired FedProx
studies into one three-seed picture, per the pre-registered
42/123/2026 design in `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md`.

**Subject experiments (three paired seeds):**

| Seed | FedAvg | FedProx (μ=0.01) |
|---|---|---|
| 42 | `EXP_007` | `EXP_010` |
| 123 | `EXP_008` | `EXP_011` |
| 2026 | `EXP_009` | `EXP_012` |

**Sources consulted:**
- `analysis/EXP_010_FedProx_Seed42_Analysis.md`
- `analysis/EXP_011_FedProx_Seed123_Analysis.md`
- `analysis/EXP_012_FedProx_Seed2026_Analysis.md`
- `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`
- `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md`
- Raw result files (re-verified directly, not taken on faith from the
  above documents): `results/federated/cifar100_fedavg_dirichlet_a01_50rounds/`,
  `.../cifar100_fedavg_dirichlet_a01_seed123_50rounds/`,
  `.../cifar100_fedavg_dirichlet_a01_seed2026_50rounds/`,
  `.../cifar100_fedprox_dirichlet_a01_mu001_50rounds/`,
  `.../cifar100_fedprox_dirichlet_a01_mu001_seed123_50rounds/`,
  `.../cifar100_fedprox_dirichlet_a01_mu001_seed2026_50rounds/`
  (`summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`
  in each).

No code, config, test, README, or Research Database file was created or
modified in producing this document. No command was executed. All numbers
below are taken directly from the files listed above; none are estimated
or invented. **No claim of statistical significance is made anywhere in
this document — three paired seeds support descriptive summary statistics
only**, per `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §22.

---

## 1. Paired Design Confirmation

Re-verified directly from the six runs' own `summary.json` and
`client_partition_summary.csv` files:

- **Partition realization is identical within each seed pair.**
  `client_partition_summary.csv` client sample counts match exactly
  between `EXP_007`/`EXP_010` (10,883 / 8,003 / 8,868 / 8,574 / 8,672),
  between `EXP_008`/`EXP_011` (6,024 / 9,628 / 7,202 / 10,361 / 11,785),
  and between `EXP_009`/`EXP_012` (5,233 / 12,318 / 9,171 / 6,556 / 11,722).
- **All six runs share**: `dataset="CIFAR-100"`, `partition="dirichlet"`,
  `dirichlet_alpha=0.1` (via config), `model="resnet18_cifar"`,
  `total_parameters=11,220,132`, `num_clients=5`, `clients_per_round=5`,
  `minimum_updates=5`, `communication_rounds_configured=communication_rounds_completed=50`,
  `local_epochs=1`, `optimizer="sgd"`, `learning_rate=0.01`,
  `batch_size=32`, `aggregation_strategy="fedavg"` (identical for both
  methods — FedProx modifies only the client-side local objective, never
  the server-side aggregation rule), same GPU (NVIDIA GeForce RTX 5060
  Laptop GPU). All six runs completed 50/50 rounds with 5/5 successful
  clients and 0 rejected updates.
- **The only declared scientific difference within each pair** is
  `local_training_method` (`fedavg` → `fedprox`) and its required
  companion `fedprox_mu` (`0` / inert → `0.01`).
- **Seeds differ only across pairs, as intended**: 42, 123, 2026 — each
  pair uses a distinct Dirichlet partition realization, but that
  realization is shared identically by its own FedAvg/FedProx pair.

This is a genuine three-pair, single-variable (local objective) paired
design, matching the pre-registration in
`research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §§3–4 exactly.

---

## 2. Three-Seed Paired Metric Table

| Metric | Seed 42 FedAvg | Seed 42 FedProx | Seed 123 FedAvg | Seed 123 FedProx | Seed 2026 FedAvg | Seed 2026 FedProx |
|---|---:|---:|---:|---:|---:|---:|
| Test loss | 1.3758617393493653 | 1.3706392387390136 | 1.4654396366119384 | 1.4718453573226928 | 1.446613697052002 | 1.473337201499939 |
| Test top-1 | 59.68% | 59.63% | 56.83% | 57.19% | 58.26% | 57.63% |
| Test top-5 | 88.65% | 88.59% | 87.79% | 87.36% | 88.56% | 87.74% |
| Macro precision | 0.6509177318450269 | 0.6460172564159314 | 0.6337611188249366 | 0.6274460728730231 | 0.6237342308141861 | 0.6216831240921957 |
| Macro recall | 0.5965 | 0.5963 | 0.5683 | 0.5719 | 0.5827 | 0.5763 |
| Macro F1 | 0.5875828060131298 | 0.5858452197428362 | 0.5485726596416799 | 0.5515867193058298 | 0.5641049541589388 | 0.554339915901785 |
| Balanced accuracy | 0.5965 | 0.5963 | 0.5683 | 0.5719 | 0.5827 | 0.5763 |
| Best validation top-1 | 59.22% | 59.02% | 58.26% | 57.62% | 58.08% | 57.22% |

---

## 3. Paired Differences (FedProx − FedAvg) by Seed

| Metric | Seed 42 Δ | Seed 123 Δ | Seed 2026 Δ |
|---|---:|---:|---:|
| Test loss | −0.0052225 | +0.0064057 | +0.0267235 |
| Test top-1 | −0.05pp | +0.36pp | −0.63pp |
| Test top-5 | −0.06pp | −0.43pp | −0.82pp |
| Macro precision | −0.0049005 | −0.0063150 | −0.0020511 |
| Macro recall | −0.02pp | +0.36pp | −0.64pp |
| Macro F1 | −0.0017376 | +0.0030141 | −0.0097650 |
| Balanced accuracy | −0.02pp | +0.36pp | −0.64pp |
| Best validation top-1 | −0.20pp | −0.64pp | −0.86pp |

---

## 4. Three-Seed Aggregate Statistics

Mean and sample standard deviation use n=3, n−1=2 denominator, computed
directly from the values in §§2–3.

| Metric | Mean FedAvg | Mean FedProx | Mean Δ | Sample Std. Dev. of Δ | Min Δ | Max Δ |
|---|---:|---:|---:|---:|---:|---:|
| Test loss | 1.4293 | 1.4386 | +0.0093 | 0.0162 | −0.0052 | +0.0267 |
| Test top-1 | 58.26% | 58.15% | −0.11pp | 0.50pp | −0.63pp | +0.36pp |
| Test top-5 | 88.33% | 87.90% | −0.44pp | 0.38pp | −0.82pp | −0.06pp |
| Macro precision | 0.6361 | 0.6317 | −0.0044 | 0.0022 | −0.0063 | −0.0021 |
| Macro recall | 58.25% | 58.15% | −0.10pp | 0.50pp | −0.64pp | +0.36pp |
| Macro F1 | 0.5668 | 0.5639 | −0.0028 | 0.0065 | −0.0098 | +0.0030 |
| Balanced accuracy | 58.25% | 58.15% | −0.10pp | 0.50pp | −0.64pp | +0.36pp |
| Best validation top-1 | 58.52% | 57.95% | −0.57pp | 0.34pp | −0.86pp | −0.20pp |

Per `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §15, no standard
error, confidence interval, or p-value is computed or reported for any of
these paired differences — n=3 pairs does not support such an estimate.

---

## 5. Test Top-1 Paired Differences (Explicit)

The three paired test top-1 differences (FedProx − FedAvg), individually,
before averaging:

- Seed 42: **−0.05pp**
- Seed 123: **+0.36pp**
- Seed 2026: **−0.63pp**
- **Mean: −0.11pp**

Positive in 1 of 3 seeds (123); negative in 2 of 3 seeds (42, 2026).

---

## 6. Consistent Improvement, Consistent Degradation, or Mixed/Near-Neutral?

**Mixed/near-neutral, with a majority-negative sign lean that does not
reach practical significance.** No metric shows FedProx ahead of FedAvg in
2-of-3 or 3-of-3 seeds. The sign pattern by metric:

- **3-of-3 seeds negative (FedProx behind)**: test top-5, macro precision,
  best validation top-1.
- **2-of-3 seeds negative**: test loss (worse at 123, 2026; better at 42),
  test top-1, macro recall, macro F1, balanced accuracy (all worse at 42
  and 2026, better at 123).
- **0-of-3 seeds negative (FedProx ahead in all three)**: none.

No metric is "consistently improved" by FedProx under this protocol. The
pattern is also not a symmetric, evenly-split neutral result — every
metric's majority sign (2-of-3 or 3-of-3) points toward FedProx being
marginally behind FedAvg, not ahead. However, per §7 below, the *size* of
every one of these differences is small enough that "consistently
degrades" would overstate what three seeds actually show. The most
accurate description is: **FedProx (μ=0.01) shows no measurable benefit
over FedAvg at any seed, and a small, direction-inconsistent-in-magnitude-but-mostly-negative-in-sign
effect that does not clear the bar for practical significance (§7).**

---

## 7. Practical Significance

Practical significance requires not just a sign but a magnitude large
enough to matter for the project's purposes. None of the mean paired
differences approach that bar:

- Mean test top-1 difference: **−0.11pp** — roughly a quarter of one
  percentage point.
- Largest mean difference of any accuracy-style metric: best validation
  top-1 at **−0.57pp**.
- Largest single-seed difference of any metric: test loss at seed 2026
  (+0.0267) and best validation top-1 at seed 2026 (−0.86pp) — still a
  fraction of a single percentage point in accuracy terms.

None of these differences would be distinguishable from ordinary run
variation by inspection at this configuration (§8). By the standard set in
`research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §12 (mean paired
test top-1 difference must be positive and clearly larger than the
2.85pp noise floor to count as a measurable improvement) and §13
(negative mean and positive in only 0–1 of 3 seeds counts as a failure),
**this three-seed result formally satisfies the plan's pre-registered
Failure Criteria for test top-1**: the mean difference (−0.11pp) is
negative, and the per-seed difference is positive in only 1 of 3 seeds.

---

## 8. Magnitude Relative to Already-Characterized Seed-to-Seed Variation

`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` §3 established a
2.85-percentage-point test top-1 range (56.83%–59.68%) from FedAvg alone,
across the same three seeds, with no method change at all. Every mean
paired FedProx-minus-FedAvg difference reported in §4 above is small
relative to that figure:

| Metric | Mean |Δ| | As a fraction of the 2.85pp FedAvg-only test-top-1 noise floor |
|---|---:|---:|
| Test top-1 | 0.11pp | ≈4% |
| Test top-5 | 0.44pp | ≈15% |
| Macro recall / balanced accuracy | 0.10pp | ≈4% |
| Best validation top-1 | 0.57pp | ≈20% |

Even the single largest per-seed difference observed anywhere in §3
(best validation top-1, seed 2026, −0.86pp) is still smaller than the
2.85pp noise floor. **The FedProx effect, whatever its true sign, is small
relative to the seed-to-seed variation this project has already
characterized for FedAvg alone at this exact configuration.**

---

## 9. Training Loss Is Not Directly Comparable Between Methods

As already established in `analysis/EXP_010_FedProx_Seed42_Analysis.md`
§4, `analysis/EXP_011_FedProx_Seed123_Analysis.md` §4, and
`src/training/engine.py::train_one_epoch`'s docstring: FedProx's logged
`train_loss` (in `round_history.csv`) includes the
`(mu/2)*||w-w^t||^2` proximal penalty term, while FedAvg's does not. The
two `train_loss` columns measure different quantities and are **not**
compared anywhere in this synthesis, or in any of the three per-seed
analyses it draws on. Only validation and test metrics (loss, top-1,
top-5, macro precision/recall/F1, balanced accuracy — all computed by the
unmodified `evaluate()` path, unaffected by the proximal term) are
compared in §§2–8.

---

## 10. `local_epochs=1` Remains an Unproven Hypothesis, Not a Causal Explanation

`analysis/EXP_010_FedProx_Seed42_Analysis.md` §6 raised
`local_epochs=1` (used unchanged in every experiment `EXP_002`–`EXP_012`)
as a **plausible, not demonstrated** explanation for FedProx's muted
effect: with only one local epoch per round before re-synchronization and
optimizer state reset every round (`FederatedClient.reset_training_state()`,
unconditional), a client's opportunity to drift far from the global model
within a single round is inherently bounded before the proximal term is
even applied, leaving comparatively little drift for FedProx to correct.
**This synthesis does not treat this as demonstrated by three seeds
either.** No experiment in this project has isolated `local_epochs` as a
controlled variable against FedProx's effect — the three seeds analyzed
here all share `local_epochs=1`, so they can neither confirm nor rule out
this hypothesis; they can only show that the effect is muted under the
one `local_epochs` value tested so far.

---

## 11. Should FedProx (μ=0.01) Replace, Remain a Comparison Baseline, or Be Dropped?

**FedAvg remains the primary strong-Non-IID baseline. FedProx (μ=0.01)
should remain documented as a completed, tried comparison — not adopted
as a replacement baseline, and not carried forward as the default method
for future phases.** This follows directly from §7's finding that the
three-seed result formally meets the plan's pre-registered Failure
Criteria (§13) for test top-1: a negative mean paired difference with a
positive result in only 1 of 3 seeds. Per
`research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §19's Failure
branch, this project should "not adopt FedProx for this configuration."
This is not equivalent to concluding FedProx is harmful, or worth erasing
from the record — the effect (§§7–8) is small and does not demonstrate
degradation either — but it does not clear the bar the plan pre-registered
for replacing FedAvg. FedAvg's three-seed characterization
(`analysis/EXP_009_Three_Seed_Baseline_Analysis.md`) remains this
project's reference for the α=0.1 condition.

---

## 12. Is a μ Sweep Justified Now?

**No.** Per `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §11, a
broad `mu` sweep is not the default response to a neutral or negative
result — it is a new experimental variable, not a continuation of the
already-registered single-variable design. §19 of that plan reserves a
"small, targeted follow-up" only for a genuinely **ambiguous** result
(mixed-sign differences with no clear majority, or an improvement close to
but not clearly above the noise floor). §7 above shows this result is not
ambiguous in that sense: it cleanly satisfies the pre-registered Failure
Criteria (§13) rather than landing in an unclear middle ground. A μ sweep
is therefore not scientifically justified by this evidence at this time.

---

## 13. Is Changing `local_epochs` Justified Now?

**Not as a modification to the completed 42/123/2026 paired protocol.**
That protocol is now complete and should remain frozen for historical
comparability with `EXP_007`–`EXP_012`, per this project's standing
single-variable-change discipline. Separately,
`research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §20 already
identifies a `local_epochs > 1` Non-IID stress test as one of **two**
required conditions (alongside FedProx being found insufficient, now
satisfied per §11) before SCAFFOLD design work would begin — but that is
a distinct, not-yet-run, not-yet-planned experiment in its own right, not
a change to the experiments this synthesis analyzes. This document does
not propose or schedule that stress test; it only notes that the
first of the two SCAFFOLD-gate conditions (§20) is now met by this
synthesis's own finding (§11), and that the second remains untested.

---

## 14–15. Recommended Next Research Phase

**Recommendation: shift primary experimental effort toward Phase 4 attack
simulation, using FedAvg (Dirichlet α=0.1, the already three-seed-characterized
configuration) as the baseline — not toward further aggregation-method
tuning (μ sweep, SCAFFOLD, or a fresh `local_epochs` stress test).**

The reasoning:

- The heterogeneity/optimization-characterization track set out in
  `research_plans/PHASE4_RESEARCH_STRATEGY.md` §7/11 has now run its
  pre-registered course for this configuration: three-seed FedAvg
  baseline established (`EXP_009` analysis), FedProx tried as the
  next controlled method across the same three paired seeds (this
  document), and found not to clear its own pre-registered bar (§§7, 11).
  Per §12, a μ sweep is not justified by this result. Per §13, a
  `local_epochs` stress test is a separate, not-yet-scoped experiment,
  not a continuation of the current protocol.
- Per `research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §21, this
  project's flagship goal — a federated learning system that can
  distinguish malicious client behavior from honest Non-IID
  heterogeneity, and identify both the malicious client and the attack
  round — requires, as its own separate gate, **(1) a working Phase 4
  attack simulation (label flipping, backdoor, or gradient/model
  poisoning) that measurably degrades the global model**, evaluated
  against the current heterogeneity baseline, and **(2) a naive
  baseline attribution/anomaly-detection method evaluated against that
  attack and shown to have a specific, documented weakness** — in
  particular, an inability to reliably distinguish malicious clients
  from the honestly-heterogeneous ones this project's own α=0.1 results
  make concrete. **Completing the FedProx comparison (this document)
  does not, by itself, satisfy that gate — the gate has never depended
  on FedProx's outcome.**
- **Readiness assessment**: the project is ready, on the
  heterogeneity/optimization side, to stop investing further effort
  there for now — the three-seed FedAvg-vs-FedProx comparison is
  complete and its own decision criteria have been applied (§§11–13).
  It is **not yet** ready to begin FedTrace-FM traceability/attribution
  design itself, because neither of §21's two required conditions
  (a working attack simulation; a naive baseline shown to have a
  documented weakness) exists yet. **The next research phase should
  therefore be building the Phase 4 attack simulation** — the
  necessary first building block that must exist before any
  malicious-client injection or traceability experiment, or any
  attribution-method comparison, can be run at all — using the
  established FedAvg, Dirichlet α=0.1, three-seed (42/123/2026)
  protocol as the honest-heterogeneity baseline against which attacked
  runs will be compared.

---

## Summary Table

| Question | Answer |
|---|---|
| Does FedProx (μ=0.01) consistently improve, degrade, or show a mixed/near-neutral effect? | Mixed/near-neutral; majority sign leans negative on 6 of 8 metrics, but no metric reaches practical significance (§6) |
| Is the effect practically significant? | No (§7) |
| Is the effect large relative to established seed-to-seed noise? | No — every mean |Δ| is ≤20% of the 2.85pp FedAvg-only noise floor (§8) |
| Does FedProx replace FedAvg as the primary baseline? | No — FedAvg remains primary; FedProx retained only as a completed comparison (§11) |
| Is a μ sweep justified now? | No (§12) |
| Is a `local_epochs` change justified now? | Not to the completed protocol; a separate stress test remains unscoped (§13) |
| Is the project ready to begin the malicious-client/traceability phase? | Not literally yet — its own prerequisite gate (a working attack simulation + a naive baseline shown to have a documented weakness) does not exist yet. The recommended next phase is to build that attack simulation now, since the heterogeneity/optimization track has run its course (§§14–15) |
