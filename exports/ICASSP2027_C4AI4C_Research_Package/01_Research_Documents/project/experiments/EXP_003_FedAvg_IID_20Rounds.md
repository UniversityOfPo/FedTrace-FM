# EXP_003 — IID Federated CIFAR-100 ResNet-18 (FedAvg, 20 Rounds)

## Experiment ID
`EXP_003`

## Date
2026-08-04

## Objective
Test, via a single controlled variable change, whether `EXP_002`'s
10-round IID FedAvg result was undertrained rather than converged. The
`EXP_002` round-history analysis found no plateau in validation loss,
validation top-1, or validation top-5 through the final configured round,
and recommended extending the communication-round budget as the single
highest-value next experiment. `EXP_003` is exactly that experiment:
identical to `EXP_002` in every respect except `communication_rounds:
10 → 20`.

## Dataset
CIFAR-100. Identical to `EXP_002`: the same reproducible 45,000/5,000
stratified train/validation split (`validation_fraction=0.1`, seed 42),
the same IID partition of the 45,000-image training subset across 5
clients (`partition_dataset_iid`, seed 42 — 9,000 images per client), the
same centralized 5,000-image validation subset, and the same untouched
10,000-image test set, evaluated exactly once at the end.

## Model
CIFAR-compatible ResNet-18 (`resnet18_cifar`), identical architecture to
`EXP_001` and `EXP_002`. **Total parameters:** 11,220,132 (all trainable).

## Partition
IID (`partition_dataset_iid`, seed 42) — identical partition to `EXP_002`
(same seed, same dataset, same client count): 5 clients, 9,000 training
samples each (`client_000`–`client_004`).

## Aggregation Strategy
FedAvg (`src/federated/aggregation/fedavg.py`) — identical to `EXP_002`.
All 5 clients contributed equally (9,000 samples each) every round.

## Client Configuration
Identical to `EXP_002`:
- `clients_per_round`: 5, `minimum_updates`: 5
- `local_epochs`: 1
- Optimizer: SGD, `learning_rate=0.01`, `momentum=0.9`,
  `weight_decay=0.0005`, no scheduler, mixed precision enabled
- **Client-state policy:** optimizer/scheduler/AMP-scaler reset at the
  start of every round (`FederatedClient.reset_training_state()`, via
  `load_global_weights()`) — unchanged from `EXP_002`.

## The One Changed Variable
`communication_rounds: 10 → 20`. No other configuration value differs
from `configs/cifar100_fedavg_iid.yaml` (`EXP_002`'s config).
`experiment_id`, `experiment_name`, `output_dir`, and `checkpoint_dir`
were also changed (to give this run its own identity and output location
instead of overwriting `EXP_002`'s frozen results) — these are bookkeeping
fields, not scientific variables, and have no effect on training. No
Python file required any modification: the experiment script
(`src/experiments/run_federated_cifar100.py`) already reads
`communication_rounds` from the YAML with no hardcoded round count
anywhere in the code.

## Communication Rounds
20 configured, 20 completed. All 20 rounds finalized with `round_status =
completed` and 5/5 successful client updates (0 failed, 0 rejected) —
`round_history.csv`. Rounds 0–9 exactly reproduce `EXP_002`'s per-round
values (same seed, same partition, same everything up to that point),
confirming this is a genuine continuation rather than an independent run.

## Seed
42 — identical to `EXP_001` and `EXP_002`.

## Hardware
NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM), local single-GPU
execution. Total experiment time: 1776.63 seconds (≈29 minutes 37
seconds) for all 20 rounds.

## Final Validation Metrics
Collected on the shared, centralized 5,000-image validation set after
every round. Best and final validation both occurred at **round 19** (the
last round):
- **Validation Loss:** 1.6727
- **Validation Top-1:** 54.22%
- **Validation Top-5:** 82.72%

## Final Test Metrics
The final global model (server weights after round 19's aggregation),
evaluated exactly once on the untouched 10,000-image test set:

| Metric | Value |
|---|---|
| Final Test Loss | 1.6335 |
| Final Test Top-1 Accuracy | 54.69% |
| Final Test Top-5 Accuracy | 83.75% |
| Macro Precision | 0.5565 |
| Macro Recall | 0.5468 |
| Macro F1 | 0.5367 |
| Balanced Accuracy | 0.5468 |

Full per-round history in `results/federated/
cifar100_fedavg_iid_20rounds/round_history.csv`; full summary in
`results/federated/cifar100_fedavg_iid_20rounds/summary.json`.

## Comparison to EXP_002 and EXP_001
- **Improvement over `EXP_002` (10 rounds):** +16.88 percentage points
  test top-1 (37.81% → 54.69%); +13.48 percentage points test top-5
  (70.27% → 83.75%).
- **Remaining gap to centralized Baseline v1.0 (`EXP_001`, 62.71% test
  top-1):** 8.02 percentage points — down from 24.90 points at `EXP_002`'s
  10-round budget.

## Observations
- Doubling the communication-round budget, with zero other change,
  produced a very large accuracy gain: +16.88 points test top-1. This is
  a substantially bigger effect than any hyperparameter change would
  plausibly have produced at the 10-round budget, and it directly
  confirms the `EXP_002` analysis's central claim: the model was
  undertrained, not converged.
- The gap to the centralized baseline shrank by roughly two-thirds (24.90
  → 8.02 points) from this single change, strongly suggesting most (but,
  at 8.02 points remaining, not necessarily all) of the original gap was
  attributable to insufficient training rounds rather than an inherent
  limitation of FedAvg under IID partitioning.
- Validation accuracy was still improving at round 19 (e.g., round 18→19:
  53.14% → 54.22%, +1.08 points), with no clear flattening visible in the
  last few rounds — the same undertrained signature observed at the end
  of `EXP_002`, now recurring at a higher accuracy level. Whether this
  round-19 rate of improvement is meaningfully smaller than earlier
  rounds' rates (i.e., whether the curve is beginning to bend toward a
  plateau) has not yet been formally analyzed — that analysis is the
  immediate next task (see **Next Experiment**).
- Final test metrics (top-1 54.69%, top-5 83.75%) again closely track
  final validation metrics (top-1 54.22%, top-5 82.72%), consistent with
  `EXP_002`'s finding that the validation set is a reliable, low-variance
  proxy for test performance in this federated setup.
- All 20 rounds completed with 5/5 clients successful and 0 rejected
  updates — the pipeline's reliability observed in `EXP_002` held at
  double the round count with no new failure modes.

## Lessons Learned
- The recommendation produced by a careful, data-only analysis of
  `EXP_002` (extend the round budget; do not change partition, model, or
  hyperparameters yet) turned out to be directly correct and high-value:
  a single-variable, code-free change (just a new YAML config) produced
  the largest accuracy improvement of any experiment in this project to
  date.
- Because the experiment script was already fully config-driven, testing
  this hypothesis required creating one new YAML file and zero lines of
  code changed — a direct payoff of the Sprint 3.4/4.1 design decision to
  keep `communication_rounds` (and everything else) as ordinary
  configuration rather than a hardcoded value.
- Reusing the identical seed and partition let rounds 0–9 serve as an
  exact reproducibility check against `EXP_002` (their values match
  row-for-row), which is a useful, low-cost way to gain confidence that a
  "controlled, single-variable" experiment claim is actually true, not
  just asserted.

## Limitations
- Single seed (42); no repeated runs to characterize variance.
- Still only IID partitioning; Non-IID (Dirichlet/shard) partitioning does
  not exist yet.
- The remaining 8.02-point gap to the centralized baseline is not yet
  explained: it is unknown how much of it would close with a further
  increase in rounds (e.g., 50), versus how much reflects a genuine,
  persistent cost of federated averaging under this configuration.
- No formal plateau/convergence analysis has been performed on this
  round history yet — the "still improving at round 19" observation above
  is qualitative (from the raw numbers), not a quantified trend analysis.
- Client processing remains strictly sequential; no attack, attribution,
  or repair logic is present.

## Future Comparison Targets
- **`EXP_004` (pending decision):** a further round-budget extension (the
  originally suggested figure was 50 rounds), to determine whether the
  remaining 8.02-point gap to `EXP_001` continues to close or begins to
  plateau above it. This decision is the immediate next task (see below).
- **Non-IID federated experiments** — once round-budget questions are
  settled, compared against whichever IID result (`EXP_002`, `EXP_003`, or
  a future longer run) is the appropriate reference point.
- **Attack/attribution/repair experiments** (Phases 4–5) — will need a
  converged (not undertrained) federated baseline as their reference
  point, making the round-budget question directly relevant to when those
  phases can begin meaningfully.

## Git Commit History
As of this experiment, the repository has twelve commits, the most recent
being:

12. `26a9bb2` — **Add 20-round IID FedAvg experiment configuration**
    Added `configs/cifar100_fedavg_iid_20rounds.yaml` only (Sprint 4.2).
    No Python file, existing config, or test was modified.

This commit's config, combined with the already-committed experiment
script (commit 10, `f82ac23`), produced the exact result documented above
(seed 42). See `FEDTRACE_FM_MASTER.md` → *Current Git History* for the
full twelve-commit list.

## Next Experiment
Analysis, not a new experiment, is the immediate next step: analyze
`EXP_003`'s full round-by-round history for a plateau signal, and use
that analysis to decide whether `EXP_004` (planned) should extend to 50
communication rounds before Non-IID partitioning work begins.
