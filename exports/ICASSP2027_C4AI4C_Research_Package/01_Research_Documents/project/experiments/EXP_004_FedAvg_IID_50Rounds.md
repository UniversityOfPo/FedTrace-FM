# EXP_004 — IID Federated CIFAR-100 ResNet-18 (FedAvg, 50 Rounds)

## Experiment ID
`EXP_004`

## Date
2026-08-04

## Objective
Per `research_plans/EXP_004_Convergence_Study_Plan.md`: determine whether
extending the communication-round budget for IID FedAvg on CIFAR-100
continues to close the gap to the centralized baseline (`EXP_001`), or
whether training has begun to plateau by round 20 (`EXP_003`).
`communication_rounds: 20 → 50` is the only variable changed relative to
`EXP_003`; every other configuration value is identical.

## Dataset
CIFAR-100. Identical to `EXP_002`/`EXP_003`: the same reproducible
45,000/5,000 stratified train/validation split (`validation_fraction=0.1`,
seed 42), the same IID partition of the 45,000-image training subset
across 5 clients (`partition_dataset_iid`, seed 42 — 9,000 images per
client), the same centralized 5,000-image validation subset, and the same
untouched 10,000-image test set, evaluated exactly once at the end.

## Model
CIFAR-compatible ResNet-18 (`resnet18_cifar`), identical architecture to
`EXP_001`/`EXP_002`/`EXP_003`. **Total parameters:** 11,220,132 (all
trainable).

## Partition
IID (`partition_dataset_iid`, seed 42) — identical partition to
`EXP_002`/`EXP_003` (same seed, same dataset, same client count): 5
clients, 9,000 training samples each (`client_000`–`client_004`).

## Aggregation Strategy
FedAvg (`src/federated/aggregation/fedavg.py`) — identical to
`EXP_002`/`EXP_003`. All 5 clients contributed equally (9,000 samples
each) every round.

## Client Configuration
Identical to `EXP_002`/`EXP_003`:
- `clients_per_round`: 5, `minimum_updates`: 5
- `local_epochs`: 1
- Optimizer: SGD, `learning_rate=0.01`, `momentum=0.9`,
  `weight_decay=0.0005`, no scheduler, mixed precision enabled
- **Client-state policy:** optimizer/scheduler/AMP-scaler reset at the
  start of every round (`FederatedClient.reset_training_state()`, via
  `load_global_weights()`) — unchanged from `EXP_002`/`EXP_003`.

## The One Changed Variable
`communication_rounds: 20 → 50`. No other configuration value differs
from `configs/cifar100_fedavg_iid_20rounds.yaml` (`EXP_003`'s config).
`experiment_id`, `experiment_name`, `output_dir`, and `checkpoint_dir`
were also changed (bookkeeping only, to avoid overwriting `EXP_003`'s
frozen results), exactly as documented for `EXP_002` → `EXP_003`.

## Communication Rounds
50 configured, 50 completed. All 50 rounds finalized with `round_status =
completed` and 5/5 successful client updates (0 failed, 0 rejected) —
`round_history.csv`. Rounds 0–19 exactly reproduce `EXP_003`'s per-round
values (same seed, same partition, same everything up to that point) —
e.g. round 19: validation loss 1.6726748435974121, top-1 54.22%, top-5
82.72%, matching `EXP_003`'s round 19 exactly — confirming this is a
genuine controlled continuation.

## Seed
42 — identical to `EXP_001`/`EXP_002`/`EXP_003`.

## Hardware
NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM), local single-GPU
execution. Total experiment time: 4055.95 seconds (≈67.6 minutes) for all
50 rounds.

## Final Validation Metrics
Collected on the shared, centralized 5,000-image validation set after
every round. Best and final validation both occurred at **round 49** (the
last round):
- **Validation Loss:** 1.1155
- **Validation Top-1:** 68.68%
- **Validation Top-5:** 91.40%

## Final Test Metrics
The final global model (server weights after round 49's aggregation),
evaluated exactly once on the untouched 10,000-image test set:

| Metric | Value |
|---|---|
| Final Test Loss | 1.1049 |
| Final Test Top-1 Accuracy | 68.52% |
| Final Test Top-5 Accuracy | 91.27% |
| Macro Precision | 0.6901 |
| Macro Recall | 0.6850 |
| Macro F1 | 0.6814 |
| Balanced Accuracy | 0.6850 |

Full per-round history in `results/federated/
cifar100_fedavg_iid_50rounds/round_history.csv`; full summary in
`results/federated/cifar100_fedavg_iid_50rounds/summary.json`.

## Comparison to EXP_003 and EXP_001
- **Improvement over `EXP_003` (20 rounds):** +13.83 percentage points
  test top-1 (54.69% → 68.52%); +7.52 percentage points test top-5
  (83.75% → 91.27%).
- **Comparison to centralized Baseline v1.0 (`EXP_001`, 62.71% test
  top-1):** `EXP_004`'s federated test top-1 (68.52%) now **exceeds** the
  centralized baseline by 5.81 percentage points, and test top-5 (91.27%
  vs. 88.54%) exceeds it by 2.73 percentage points. This reverses the
  gap direction observed in both `EXP_002` (24.90 points below) and
  `EXP_003` (8.02 points below).

## Final-Five-Round Trend (rounds 45–49)

| Round | Val Loss | Δ Loss | Val Top-1 | Δ Top-1 | Val Top-5 | Δ Top-5 |
|---|---|---|---|---|---|---|
| 45 | 1.1713 | — | 67.16% | — | 90.84% | — |
| 46 | 1.1556 | −0.0158 | 67.28% | +0.12pp | 90.94% | +0.10pp |
| 47 | 1.1513 | −0.0042 | 67.72% | +0.44pp | 91.08% | +0.14pp |
| 48 | 1.1196 | −0.0318 | 68.50% | +0.78pp | 91.80% | +0.72pp |
| 49 | 1.1155 | −0.0040 | 68.68% | +0.18pp | 91.40% | −0.40pp |

## Observations
- Extending the round budget from 20 to 50, with zero other change,
  produced another very large accuracy gain: +13.83 points test top-1
  over `EXP_003`. Combined with `EXP_002`→`EXP_003`'s +16.88-point gain,
  the round-budget hypothesis first raised by the `EXP_002` analysis has
  now been confirmed twice in a row by independent, single-variable
  extensions.
- **`EXP_004` is the first federated result in this project to exceed the
  centralized Baseline v1.0** (68.52% vs. 62.71% test top-1, +5.81
  points). This is a data observation, not yet an explained one — no
  root-cause analysis (e.g., effective additional data exposure across 50
  rounds × 1 local epoch vs. 10 centralized epochs, or a beneficial
  averaging/regularization effect of FedAvg itself) has been performed as
  part of this documentation step.
- The final-five-round marginal gains (Section above) are visibly smaller
  than `EXP_003`'s final-five-round gains (`EXP_003` rounds 15–19
  averaged ≈1.17pp/round top-1; `EXP_004` rounds 45–49 average ≈0.38pp/
  round top-1), and round 48→49 shows validation top-5 *decreasing*
  slightly (91.80% → 91.40%, −0.40pp) for the first time in this
  project's round histories. This is consistent with, but does not by
  itself prove, an approaching plateau — a formal convergence
  classification (continues / slows significantly / plateaus, per
  `research_plans/EXP_004_Convergence_Study_Plan.md` Section 8) has not
  been performed as part of this documentation step and remains a
  separate analysis task.
- Final test metrics (top-1 68.52%, top-5 91.27%) again closely track
  final validation metrics (top-1 68.68%, top-5 91.40%), consistent with
  the pattern observed in `EXP_002`/`EXP_003`.
- All 50 rounds completed with 5/5 clients successful and 0 rejected
  updates — the pipeline's reliability held at more than double
  `EXP_003`'s round count with no new failure modes.

## Lessons Learned
- A second consecutive controlled round-budget extension again required
  zero lines of code changed — only a new YAML config — reconfirming the
  value of keeping `communication_rounds` fully config-driven.
- The exact reproduction of `EXP_003`'s round 0–19 values inside
  `EXP_004`'s round history continues to serve as a reliable, low-cost
  reproducibility check for "controlled, single-variable" experiment
  claims.
- A federated result surpassing its centralized reference is a notable
  enough finding that it should prompt a dedicated follow-up analysis
  (why, and is it robust) rather than being treated as simply "a bigger
  number" — flagged here for that follow-up, not resolved by it.

## Limitations
- Single seed (42); no repeated runs to characterize variance — the
  centralized-exceeding result has not been checked for seed sensitivity.
- Still only IID partitioning; Non-IID (Dirichlet/shard) partitioning does
  not exist yet.
- No formal plateau/convergence classification has been performed on this
  round history yet (see Observations above) — whether round 49 represents
  a genuine near-plateau or continued noisy improvement is not yet
  determined.
- Client processing remains strictly sequential; no attack, attribution,
  or repair logic is present.
- The reason `EXP_004` exceeds the centralized `EXP_001` baseline has not
  been investigated (e.g., differing effective data exposure, regularizing
  effect of federated averaging, or another cause) — this is an open
  question, not an invented explanation.

## Future Comparison Targets
- **Convergence analysis of `EXP_004`'s round history** — the immediate
  next task, to classify the observed slowdown per
  `research_plans/EXP_004_Convergence_Study_Plan.md` Section 8/9 and
  determine the single next action (further round extension, or
  introducing a learning-rate scheduler).
- **Non-IID federated experiments** — compared against whichever IID
  result (`EXP_002`, `EXP_003`, or `EXP_004`) is the appropriate reference
  point once the round-budget question is settled.
- **Attack/attribution/repair experiments** (Phases 4–5) — will need a
  settled, well-understood federated baseline as their reference point.

## Git Commit History
As of this experiment, the repository has fourteen commits, the most
recent being:

14. `283902c` — **Prepare EXP_004 convergence study and 50-round
    configuration**
    Added `research_plans/EXP_004_Convergence_Study_Plan.md` and
    `configs/cifar100_fedavg_iid_50rounds.yaml`. No existing Python file,
    config, or test was modified.

This commit's config, combined with the already-committed experiment
script (commit 10, `f82ac23`), produced the exact result documented above
(seed 42). See `FEDTRACE_FM_MASTER.md` → *Current Git History* for the
full commit list.

## Next Experiment
Analysis, not a new experiment, is the immediate next step: formally
classify `EXP_004`'s round-by-round convergence behavior (continues /
slows significantly / plateaus) per
`research_plans/EXP_004_Convergence_Study_Plan.md`'s decision tree, and
use that analysis — together with the unexplained centralized-exceeding
result — to decide the next single controlled experiment.