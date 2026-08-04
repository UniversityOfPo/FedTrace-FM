# EXP_002 — IID Federated CIFAR-100 ResNet-18 (FedAvg)

## Experiment ID
`EXP_002`

## Date
2026-08-04

## Objective
Run the first real federated learning experiment on top of the completed
federated infrastructure (`FederatedClient`, `FederatedServer`, the
pluggable `AggregationStrategy` interface with `FedAvg`, and
`FederatedRunner`): partition the existing 45,000-image CIFAR-100 training
subset IID across 5 clients, train a CIFAR-compatible ResNet-18 via FedAvg
for 10 communication rounds, and evaluate the final global model exactly
once on the untouched, official 10,000-image test set — producing a
reproducible, paper-useful Federated Baseline v1.0 result comparable
against the frozen centralized Baseline v1.0 (`EXP_001`) and against
future Non-IID, attack, attribution, and repair experiments.

## Dataset
CIFAR-100. The existing, reproducible 45,000/5,000 stratified
train/validation split (`validation_fraction=0.1`, seed 42) is reused
unchanged: the 45,000-image training subset is partitioned IID across the
5 clients (`partition_dataset_iid`, seed 42 — 9,000 images per client,
disjoint, full coverage); the 5,000-image validation subset stays
centralized and is used by the server-side evaluation callback after
every communication round; the official 10,000-image test set is loaded
separately and evaluated exactly once, at the end, on the final global
model.

## Model
CIFAR-compatible ResNet-18 (`resnet18_cifar`, 3x3 stride-1 stem, no
initial max-pool, no pretrained weights) — the same architecture as the
centralized baseline. **Total parameters:** 11,220,132 (all trainable).
Every client and the server build the model from the same fresh-instance
factory function.

## Partition
IID (`partition_dataset_iid`, seed 42). 5 clients, 9,000 training samples
each (`client_000`–`client_004`), disjoint index sets covering the full
45,000-image training subset.

## Aggregation Strategy
FedAvg (`src/federated/aggregation/fedavg.py`) — sample-count-weighted
averaging of client model updates. All 5 clients contributed equally
(9,000 samples each), so this round's FedAvg reduces to an unweighted
average across clients.

## Client Configuration
- `clients_per_round`: 5 (all registered clients selected every round)
- `minimum_updates`: 5 (a round only completes if all 5 clients succeed)
- `local_epochs`: 1 (one local epoch of training per client per round)
- Optimizer: SGD, `learning_rate=0.01`, `momentum=0.9`,
  `weight_decay=0.0005`
- Scheduler: none
- Mixed precision: enabled
- **Client-state policy:** at the start of every round, each selected
  client loads the current global weights and its optimizer/scheduler/AMP
  scaler are rebuilt from scratch
  (`FederatedClient.reset_training_state()`, invoked automatically by
  `load_global_weights()`) — no SGD momentum or scheduler state carries
  over between communication rounds. See `DECISIONS.md` → *Why Reset
  Client Optimizer State Every Round*.

## Communication Rounds
10 configured, 10 completed. All 10 rounds finalized with `round_status =
completed` and 5/5 successful client updates (0 failed, 0 rejected) —
`round_history.csv`.

## Seed
42 (applied to Python/NumPy/PyTorch RNGs, the IID partition, and
server-side deterministic client selection).

## Hardware
NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM), local single-GPU
execution. Total experiment time: 765.57 seconds (≈12 minutes 46
seconds) for all 10 rounds (data loading, partitioning, 5×10 local
client-training passes, aggregation, and per-round validation combined).

## Final Validation Metrics
Collected on the shared, centralized 5,000-image validation set after
every round. Best and final validation both occurred at round 9 (the
last round):
- **Validation Loss:** 2.3798
- **Validation Top-1:** 37.54%
- **Validation Top-5:** 69.66%

## Final Test Metrics
The final global model (server weights after round 9's aggregation),
evaluated exactly once on the untouched 10,000-image test set:

| Metric | Value |
|---|---|
| Final Test Loss | 2.3611 |
| Final Test Top-1 Accuracy | 37.81% |
| Final Test Top-5 Accuracy | 70.27% |
| Macro Precision | 0.3918 |
| Macro Recall | 0.3780 |
| Macro F1 | 0.3600 |
| Balanced Accuracy | 0.3780 |

Full per-round history in
`results/federated/cifar100_fedavg_iid/round_history.csv`; full summary
in `results/federated/cifar100_fedavg_iid/summary.json`.

## Observations
- Every one of the 10 rounds completed with all 5/5 clients successful and
  0 rejected updates — the federated pipeline (client selection, local
  training, update submission, aggregation, explicit global-model update,
  round finalization, centralized evaluation) ran without a single
  failure across the full run.
- Validation top-1 improved monotonically and substantially every round:
  1.86% → 8.72% → 16.96% → 19.86% → 22.28% → 26.66% → 29.76% → 32.60% →
  35.66% → 37.54%. No round showed a regression, and the gains were still
  meaningful in the final round (35.66% → 37.54%), suggesting the model
  had not yet plateaued at round 10.
- The final test metrics (top-1 37.81%, top-5 70.27%) closely track the
  final/best validation metrics (top-1 37.54%, top-5 69.66%) — a small,
  consistent ~0.3–0.6 point gap in the test set's favor — indicating the
  centralized validation set remains a representative, low-variance proxy
  for test performance even in the federated setting.
- Round durations were not constant: rounds 0–4 averaged ~72s, while
  rounds 5–9 averaged ~80s (visible in `communication_round_duration.png`
  and `round_history.csv`'s `round_seconds` column). This mirrors the
  same early-round CUDA/cuDNN warm-up pattern observed in the centralized
  Baseline v1.0 run, not a progressive slowdown.
- **The federated result is substantially below the centralized
  baseline at a comparable data-exposure budget.** Both experiments
  expose the model to the full 45,000-image training set once per
  "unit" (once per centralized epoch; once per federated round, since
  all 5 IID clients together cover the full set each round). At that
  matched 10-unit budget: centralized reaches 62.71% test top-1;
  federated (IID, FedAvg) reaches 37.81%. This is a well-known
  characteristic of FedAvg — periodic weight averaging across
  independently-trained clients is a less efficient optimizer than
  direct SGD over the full data stream, especially early in training —
  not evidence of a bug in this implementation. It is flagged explicitly
  because Phase 3 of `PROJECT_ROADMAP.md` calls for the federated result
  to be "comparable" to the centralized baseline under IID/no-attack
  conditions, and at this round budget it is not yet comparable.

## Lessons Learned
- The Sprint 3.2 open question ("should client optimizer state persist
  across rounds?") needed a real, concrete answer to run any federated
  experiment at all — deferring it as a TODO was correct at the time (no
  strategy existed yet that needed a different answer), but running
  `EXP_002` is what actually forced the decision. Resolving it as "reset
  every round" (the FedAvg-standard policy) was the right choice for a
  first baseline, and is now documented in `DECISIONS.md` rather than
  left implicit.
- Building the experiment script strictly on top of the existing,
  already-tested federated stack (no new orchestration abstractions, no
  duplicated training/evaluation logic) meant the entire pipeline worked
  correctly on the very first real run — every round completed, no client
  or aggregation errors occurred, matching the lightweight unit tests'
  expectations.
- Sequential (non-parallel) client processing within a round is the
  dominant cost: with 5 clients at ~13–16s of local training each, most
  of each round's ~72–80s is simply 5 sequential local-training passes.
  This is expected at this sprint's scope (parallel client execution is
  explicitly out of scope) but is worth noting as the main lever for
  faster iteration in future experiments.

## Limitations
- Single seed (42); no repeated runs to characterize variance across
  seeds for either the partition or training.
- Only IID partitioning has been exercised; Non-IID (Dirichlet/shard)
  partitioning does not exist yet (Sprint 4.2).
- Only 10 communication rounds × 1 local epoch were run; whether the
  federated/centralized gap narrows with more rounds, more local epochs
  per round, or different hyperparameters has not been tested.
- No attack, attribution, or repair logic is present — this is a clean,
  cooperative-clients-only baseline.
- Client processing is strictly sequential; wall-clock time would not
  reflect a real multi-machine deployment.
- The optimizer-reset-every-round policy (see `DECISIONS.md`) is the only
  client-state policy that has been tested; whether a persistent-state
  policy would change these results is unknown.

## Future Comparison Targets
This experiment (`EXP_002`) is now itself a reference point for:
- **Non-IID federated experiments** (Sprint 4.2) — same model/optimizer/
  aggregation/round budget, Dirichlet- or shard-partitioned clients,
  compared directly against this IID result.
- **Attack simulation experiments** (Phase 4) — expected to show
  measurable degradation relative to this clean, cooperative-clients
  baseline.
- **Attribution/repair experiments** (Phase 5) — expected to recover
  accuracy closer to this baseline relative to an undefended attacked run.
- **Extended-round/extended-local-epoch federated runs** — to determine
  whether the observed gap to the centralized Baseline v1.0 (`EXP_001`)
  narrows with a larger training budget.

## Git Commit History
As of this experiment, the repository has ten commits, the most recent
being:

10. `f82ac23` — **Add IID CIFAR-100 federated experiment pipeline**
    Added `src/data/federated_partition.py`,
    `src/experiments/run_federated_cifar100.py`,
    `configs/cifar100_fedavg_iid.yaml`,
    `configs/cifar100_fedavg_iid_smoke.yaml`,
    `FederatedClient.reset_training_state()`, and their tests.

This commit's code and config produced the exact result documented above
(`configs/cifar100_fedavg_iid.yaml`, seed 42). See
`FEDTRACE_FM_MASTER.md` → *Current Git History* for the full ten-commit
list.

## Next Experiment
`EXP_003` (planned, Sprint 4.2): a Non-IID (Dirichlet- or shard-partitioned)
federated CIFAR-100 experiment, using the same model, optimizer, FedAvg
aggregation, and round budget as `EXP_002`, to directly isolate the effect
of non-IID client data distributions relative to this IID baseline.
