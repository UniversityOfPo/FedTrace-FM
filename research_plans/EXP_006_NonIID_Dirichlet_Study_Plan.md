# EXP_006 — Non-IID Dirichlet Partitioning Study Plan

**Status:** Planning document only. `EXP_006` has **NOT** been executed. No
config file for this experiment has been created yet. The Dirichlet
partitioning function itself does not exist yet either — unlike `EXP_004`
(whose plan predated only its config) and `EXP_005` (whose plan predated
only its config and round-level LR plumbing), this plan predates the
underlying partitioning **code** as well. Nothing beyond this document is
implied to exist by it.

---

## 1. Scientific Motivation

This project's central research hypothesis
(`FEDTRACE_FM_MASTER.md` → *Research Hypothesis*) is that malicious or
corrupted client contributions leave detectable signatures usable for
attribution and targeted repair, and that this capability degrades
gracefully — not catastrophically — as scale and domain change. Every
federated result produced so far (`EXP_002`–`EXP_005`) tests this only
under **IID** client data: every client's local class distribution is a
near-identical sample of the same global distribution. IID partitioning is
the easiest case for federated averaging and the least representative of
real deployments — including this project's own eventual medical-imaging
case study (Phase 7), where different sites/institutions almost always
have different patient populations and different disease prevalence.

An attack-detection or attribution method that only works when every
honest client already looks statistically identical to every other honest
client would not be evidence for the project's central hypothesis in any
setting that matters. Establishing a Non-IID baseline is therefore a
scientific prerequisite for Phase 4 (attack simulation) and Phase 5
(FedTrace attribution) to produce results that generalize beyond the
IID special case — not merely a generalization check to run once
convenient.

## 2. Why Non-IID Is the Next Phase

- `PROJECT_ROADMAP.md` Phase 3's completion criteria are fully checked
  except one: *"Non-IID (Dirichlet and/or shard-based) client partitioning
  implemented and tested."* Phase 3 cannot be marked complete without it.
- `EXP_004` already met — and exceeded — the "federated ≈ centralized
  under IID" criterion (68.52% vs. `EXP_001`'s 62.71% test top-1). Further
  optimizing IID accuracy (e.g. the scheduler tuning considered and
  explicitly deprioritized in
  `analysis/EXP_005_Cosine_Scheduler_Analysis.md` → Section 9) is lower
  priority than closing the one remaining, structurally required Phase 3
  gap.
- `FEDTRACE_FM_MASTER.md`'s "Pending Modules" and every session's "Next
  Action" since `EXP_002` (`RESEARCH_DIARY.md`, Entries 3.1–5.1) have
  repeatedly named Non-IID partitioning as the next open module, deferred
  each time in favor of the round-budget/scheduler question that has now
  been settled (`EXP_004` frozen as the IID reference; see
  `analysis/EXP_005_Cosine_Scheduler_Analysis.md` → Section 8).
- Realistic attacks (label flipping, backdoor triggers, gradient
  poisoning — Phase 4) are typically studied, and are more scientifically
  meaningful, against non-IID client data: detecting an anomalous client
  is a fundamentally different (and harder) problem when honest clients
  already differ from each other than when they are all statistically
  interchangeable, as under IID.

## 3. Controlled-Variable Protocol

Following the same single-variable-change discipline used for
`EXP_002 → EXP_003 → EXP_004 → EXP_005`: exactly one scientific variable
changes relative to a frozen reference configuration, and every other
value is held identical.

**Reference/baseline configuration:** `EXP_004`
(`configs/cifar100_fedavg_iid_50rounds.yaml`) — not `EXP_005` — per
`analysis/EXP_005_Cosine_Scheduler_Analysis.md` → Section 8, which
concluded `EXP_004`'s constant-learning-rate configuration remains the
project's frozen IID FedAvg reference protocol.

**Prerequisite implementation (not yet done, out of scope for this plan):**
a Dirichlet-based partitioner, added as a new function (e.g.
`partition_dataset_dirichlet`) in `src/data/federated_partition.py`, as a
**sibling** to the existing `partition_dataset_iid` — without modifying
`partition_dataset_iid` itself, matching the "extend, don't modify"
discipline already applied to every prior addition in this project (e.g.
`round_lr_schedule` added alongside, not in place of, `scheduler`). The
standard construction: for each class, draw a per-client proportion vector
from `Dirichlet(alpha, ..., alpha)` (one concentration parameter per
client) and assign that class's samples to clients according to those
proportions. Smaller `alpha` produces more skewed (more non-IID) client
class distributions; `alpha → ∞` approaches the IID case. This plan is
written in advance of that implementation, per this project's standing
"protocol before implementation" workflow (`PROJECT_CONTEXT.md` → Section
9).

**New config fields required (not yet added to the experiment script):**
- `partition: iid → dirichlet` — the one declared scientific variable.
- `dirichlet_alpha` — a required companion value with no meaning under
  `partition: iid` (mirrors `min_learning_rate`'s relationship to
  `round_lr_schedule` in `EXP_005`). This plan pre-registers a starting
  value of **`alpha = 0.5`**, a moderate-skew value commonly used as a
  default in the federated-learning literature — neither near-IID
  (`alpha` large) nor pathologically extreme (`alpha` near 0, where some
  clients could receive zero samples of most classes).

## 4. Exactly One Scientific Variable Changes: `partition: iid → dirichlet`

| Field | `EXP_004` (frozen reference) | `EXP_006` (planned) |
|---|---|---|
| `partition` | `iid` | **`dirichlet`** |
| `dirichlet_alpha` | *(not applicable)* | **`0.5`** (new, required companion field) |
| `experiment_id` / `experiment_name` / `output_dir` / `checkpoint_dir` | `EXP_004` / `cifar100_fedavg_iid_50rounds` / ... | new, non-colliding identity only (bookkeeping) |
| Every other field | — | **unchanged** (Section 5) |

**On `seed`:** `seed: 42` is held identical to every prior experiment, but
this does **not** mean `EXP_006`'s clients receive the same samples as
`EXP_004`'s. Changing the partitioning *strategy* necessarily changes which
samples land on which client, even at a matched seed — the seed controls
reproducibility of the Dirichlet draw itself (so `EXP_006` can be re-run
and produce the identical partition), not sample-for-sample agreement with
`EXP_004`'s IID partition. This is an inherent property of comparing two
different partitioning strategies, not a design flaw (see Section 9).

## 5. Variables Frozen from `EXP_004`

Copied unchanged from `configs/cifar100_fedavg_iid_50rounds.yaml`:

| Field | Value |
|---|---|
| `model` | `resnet18_cifar` |
| `num_classes` | 100 |
| `optimizer` | `sgd` |
| `learning_rate` | 0.01 |
| `momentum` | 0.9 |
| `weight_decay` | 0.0005 |
| `scheduler` | `none` |
| `round_lr_schedule` (Sprint 5.1/5.2; absent from `EXP_004`'s config, defaults to) | `constant` |
| `mixed_precision` | `true` |
| `aggregation` | `fedavg` |
| `num_clients` / `clients_per_round` / `minimum_updates` | 5 / 5 / 5 |
| `communication_rounds` | 50 |
| `local_epochs` | 1 |
| `batch_size` | 32 |
| `num_workers` | 2 |
| `validation_fraction` | 0.1 |
| `evaluate_every` / `checkpoint_every` | 1 / 1 |
| `seed` | 42 |
| `device` | `auto` |
| `fail_round_on_client_error` | `true` |
| `continue_after_failed_round` | `false` |

## 6. Hypothesis

**Primary hypothesis:** Under Dirichlet(`alpha=0.5`) Non-IID partitioning,
with every other setting identical to `EXP_004`, the global model will
reach a **lower** final test top-1 accuracy than `EXP_004`'s 68.52%. This
follows from FedAvg's well-documented sensitivity to client heterogeneity
("client drift"): when each client's local class distribution diverges
sharply from the global distribution, locally-trained models diverge more
from each other before each round's averaging step, degrading the
aggregated result relative to the IID case at a matched round budget.

**Falsification condition:** If `EXP_006`'s final test top-1 is not lower
than `EXP_004`'s by a clearly non-trivial margin — or is statistically
indistinguishable from it — the hypothesis is falsified. This would itself
be a notable, currently-unexplained finding (not evidence to discard or
re-run until a "better" result appears): it would suggest FedAvg's known
heterogeneity sensitivity does not manifest strongly at this `alpha`,
round budget, and client count, and would require its own follow-up
explanation before being treated as a settled result.

**Secondary hypothesis:** `EXP_006`'s round-by-round validation curve will
show a **slower** rate of convergence (smaller average per-round gain in
the early/middle decade blocks, using the same block methodology as
`analysis/EXP_004_Convergence_Analysis.md` and
`analysis/EXP_005_Cosine_Scheduler_Analysis.md`) than `EXP_004`'s, since
`clients_per_round = num_clients = 5` means every client participates
every round (no client-selection-driven variance) — any additional
round-to-round volatility observed would be attributable to client
heterogeneity in local updates, not sampling noise.

## 7. Success Criteria

`EXP_006` will be considered a **valid, informative experiment** (a
separate question from whether the primary hypothesis is confirmed) if:

- All 50 configured rounds complete (`communication_rounds_completed ==
  50`), matching every prior experiment's reliability record (`EXP_002`–
  `EXP_005` all completed 5/5 successful clients on every round, 0
  failures).
- The realized Dirichlet partition is **verifiably non-IID** — per-client
  per-class sample counts (Section 11) show meaningfully uneven class
  coverage across clients, not an accidentally near-uniform draw. Without
  this check, a completed run would not actually test the intended
  scientific variable, regardless of its final accuracy.
- No client receives zero total training samples (a degenerate partition
  that `partition_dataset_dirichlet` must guard against, analogous to
  `partition_dataset_iid`'s existing `num_clients > len(dataset)` check).
- The comparison against `EXP_004` remains fair per Section 4/9 — same
  seed, same every other config value, only the partition strategy (and
  its required `dirichlet_alpha` companion) differs.

## 8. Failure Criteria

`EXP_006` will be considered **not scientifically usable as run** if any
of the following occur:

- A round fails to complete, or the run aborts before round 50 — likely
  indicating an implementation bug in the new partitioner (e.g. a client
  ending up with too few samples of some classes to form even one batch)
  rather than a genuine federated-learning finding.
- The realized partition turns out statistically close to IID (e.g. all
  clients' per-class proportions cluster near the global proportions) —
  meaning `alpha=0.5` (or a partitioning bug) failed to produce real
  heterogeneity, invalidating any accuracy comparison drawn from the run.
- Any client's local training produces non-finite (NaN/Inf) loss at any
  point, which would need root-causing (most likely a class entirely
  absent from a client's local batches) before any metric from that run
  is treated as valid.

## 9. Risks

- **Implementation risk:** the Dirichlet partitioner does not exist yet;
  this plan cannot be executed until it is implemented and tested
  (analogous to `EXP_005`'s plan pre-registering round-level LR scheduling
  before Sprint 5.1 built it). Writing that code is explicitly out of
  scope for this document.
- **Degenerate-partition risk:** an unlucky combination of `alpha` and
  `seed` could assign a client zero (or near-zero) samples of one or more
  classes. This is a known property of Dirichlet partitioning at low
  `alpha` and must be checked (Section 7/8), not assumed away.
- **Single-`alpha` scope:** one run at `alpha=0.5` characterizes one point
  on the IID-to-extremely-non-IID spectrum, not the full spectrum. A
  finding at this `alpha` should not be generalized to all degrees of
  heterogeneity without a follow-up sweep (out of scope here).
- **Imperfect comparability:** as noted in Section 4, matching `seed`
  across `EXP_004` and `EXP_006` does not yield sample-identical
  partitions the way it did across `EXP_002 → EXP_003 → EXP_004`'s shared
  IID partition — this is an unavoidable consequence of comparing two
  different partitioning *strategies*, and must be stated plainly in any
  resulting analysis rather than implied to be a like-for-like split.
- **Opportunity cost:** as with every prior plan in this project, time
  spent here (a ~68-minute run, comparable to `EXP_004`/`EXP_005`, plus
  the unscoped partitioner implementation time) delays Phase 4 (attack
  simulation), which is itself gated on this experiment producing a valid
  Non-IID baseline.

## 10. Deliverables

- `src/data/federated_partition.py`: new `partition_dataset_dirichlet`
  function (sibling to `partition_dataset_iid`, which remains unmodified)
  plus corresponding tests in `tests/test_federated_partition.py`.
- A small, explicitly-scoped extension so per-client **per-class** sample
  counts are recorded (the existing
  `save_client_partition_outputs`/`client_partition_summary.csv` in
  `src/experiments/run_federated_cifar100.py` records only per-client
  **total** sample counts today — insufficient to verify Section 7's
  non-IID check). This is a deliverable of the implementation step, not
  of this plan.
- `configs/cifar100_fedavg_iid_dirichlet_50rounds.yaml` (or equivalent
  name) — not yet created.
- Full `results/federated/cifar100_fedavg_iid_dirichlet_50rounds/` output
  set (`round_history.csv`, `summary.json`, `final_test_metrics.json`,
  per-client per-class partition breakdown, `run.log`, plots), produced by
  the (otherwise unmodified) experiment script.
- `experiments/EXP_006_FedAvg_NonIID_Dirichlet.md` — narrative writeup,
  written after the run, following this project's established experiment
  documentation format.
- A comparison analysis document (`analysis/EXP_006_NonIID_vs_IID_Analysis.md`
  or equivalent), following the same verified-data-only methodology as
  `analysis/EXP_004_Convergence_Analysis.md` and
  `analysis/EXP_005_Cosine_Scheduler_Analysis.md`.
- `Research_Database/` updates (new `Experiment_Index.csv` row, etc.) and
  a `PROJECT_ROADMAP.md` Phase 3 checkbox update — both performed after
  the run, per this project's standing workflow, not as part of this plan.

## 11. Expected Figures

Produced automatically by the (unmodified) experiment script, matching the
existing output set for every prior federated experiment:
- `validation_loss.png`, `validation_top1.png`, `validation_top5.png` —
  per-round validation curves.
- `local_training_loss.png` — sample-weighted local training loss by round.
- `communication_round_duration.png` — per-round wall-clock duration.
- `confusion_matrix.png`, `per_class_accuracy.csv` — final test-set
  breakdown.

New, specific to this experiment (require the Section 10 deliverable
extension):
- A **per-client × per-class sample-count table** (or heatmap) — the
  direct evidence needed to confirm/quantify the realized partition's
  heterogeneity (Section 7's non-IID success check). Without this, the
  claim "this partition is Non-IID" would rest on the `alpha` value alone
  rather than on the partition actually produced.

Produced at the analysis stage (not by the experiment script):
- A dual-line `EXP_004` vs. `EXP_006` validation-top-1-by-round comparison
  plot/table, mirroring the round-by-round comparison already done for
  `EXP_004` vs. `EXP_005` in
  `analysis/EXP_005_Cosine_Scheduler_Analysis.md` → Section 5.

## 12. Decision Tree for `EXP_006`

- **If the primary hypothesis is confirmed** (Non-IID trails IID by a
  clear, non-trivial margin, Section 6/7 criteria met): this validates
  FedAvg's expected heterogeneity sensitivity in this codebase. Phase 3's
  Non-IID completion criterion is satisfied. Proceed to **Phase 4 (attack
  simulation)**, using `EXP_006` as the clean Non-IID baseline that
  attack experiments are measured against.
- **If the primary hypothesis is falsified** (little or no gap vs.
  `EXP_004`): treat this as an open, unexplained finding — investigate
  whether `alpha=0.5` produced insufficient skew (re-check Section 7's
  per-client per-class evidence before anything else), whether the round
  budget/`local_epochs` masks the expected heterogeneity effect, or
  whether FedAvg under this specific architecture/dataset is simply more
  heterogeneity-robust than the literature's general expectation. Do not
  proceed to Phase 4 claiming a validated Non-IID baseline until this is
  resolved.
- **If Failure Criteria (Section 8) are triggered:** fix the underlying
  partitioner implementation or partition-quality issue first. No metric
  from a failed or degenerate-partition run may be cited as `EXP_006`'s
  result, and Phase 4 does not proceed until a valid Non-IID baseline
  exists.

Exactly one of these three branches applies to any observed `EXP_006`
outcome.

## 13. PI Approval Checklist

Before any part of `EXP_006` is implemented or executed, confirm:

- [ ] This plan has been reviewed and Non-IID (Dirichlet) partitioning is
      agreed to be the correct next phase, ahead of further IID scheduler
      tuning (per `analysis/EXP_005_Cosine_Scheduler_Analysis.md` →
      Section 9).
- [ ] `alpha = 0.5` is agreed as the starting concentration parameter (or
      an alternative value is chosen and this plan is updated before
      implementation begins).
- [ ] `partition_dataset_dirichlet` is implemented and tested as a sibling
      to `partition_dataset_iid`, without modifying the latter.
- [ ] The per-client per-class partition-recording extension (Section 10)
      is implemented, so Section 7's non-IID verification check is
      possible before any accuracy comparison is drawn.
- [ ] Section 4/5's configuration table has been used to create
      `configs/cifar100_fedavg_iid_dirichlet_50rounds.yaml` (or
      equivalent), with every non-`partition`/`dirichlet_alpha` field
      verified identical to `EXP_004`'s config.
- [ ] The bookkeeping identity fields (`experiment_id`, `experiment_name`,
      `output_dir`, `checkpoint_dir`) are set to new, non-colliding values
      so `EXP_004`'s and `EXP_005`'s frozen results are not overwritten.
- [ ] Success/failure criteria (Sections 7–8) are agreed upon in advance,
      before results are seen, to avoid post-hoc reinterpretation.
- [ ] The Decision Tree (Section 12) is agreed upon in advance for all
      three branches.
- [ ] No other file (code, config, test, prior experiment result, or
      Research Database entry) will be modified as part of running
      `EXP_006` until its own results are ready for documentation.
