# EXP_006 — Non-IID Federated CIFAR-100 ResNet-18 (FedAvg, Dirichlet α=0.5, 50 Rounds)

## Experiment ID
`EXP_006`

## Date
2026-08-05

## Objective
Per `research_plans/EXP_006_NonIID_Dirichlet_Study_Plan.md`: determine
whether Non-IID (Dirichlet, α=0.5) client data partitioning degrades FedAvg
CIFAR-100 accuracy relative to the frozen IID reference (`EXP_004`), as a
single controlled-variable follow-up (`partition: iid → dirichlet`, with
`dirichlet_alpha=0.5` as that variable's required companion value). This
is the project's first federated experiment using anything other than IID
client data.

## Configuration
(`configs/cifar100_fedavg_dirichlet_a05_50rounds.yaml`)
- Experiment: `cifar100_fedavg_dirichlet_a05_50rounds` (`EXP_006`); model:
  `resnet18_cifar` (11,220,132 total parameters, all trainable).
- Dataset: CIFAR-100, the existing 45,000/5,000 stratified train/
  validation split (`validation_fraction=0.1`, seed 42); the 45,000-image
  training subset partitioned across 5 clients via
  `partition_dataset_dirichlet` (`alpha=0.5`, seed 42,
  `min_samples_per_client=320`).
- 5 clients, `clients_per_round=5`, `minimum_updates=5`, `local_epochs=1`,
  batch size 32, `num_workers=2`.
- Optimizer: SGD, `learning_rate=0.01`, `momentum=0.9`,
  `weight_decay=0.0005`, no scheduler (`round_lr_schedule` defaults to
  `constant`), mixed precision enabled.
- Aggregation: FedAvg. `communication_rounds=50`, `evaluate_every=1`,
  `checkpoint_every=1`. `fail_round_on_client_error=true`,
  `continue_after_failed_round=false`.
- Seed: 42. GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM).
- Per `run.log`: the Dirichlet partitioner succeeded on its **first**
  attempt (`attempt=1/100`), producing client sizes in the range
  8,483–9,696 — the `min_samples_per_client=320` safety floor was never
  actually invoked as a constraint.

## Controlled Variables
The **only** scientific variable changed relative to `EXP_004`
(`configs/cifar100_fedavg_iid_50rounds.yaml`) is `partition`
(`iid → dirichlet`), together with `dirichlet_alpha=0.5` — a required
companion value with no effect under `partition: iid`, not a second
independent variable. Every other configuration value is identical to
`EXP_004`: model, optimizer, learning rate, momentum, weight decay, batch
size, local epochs, aggregation strategy, client count, communication
rounds, seed, and evaluation/checkpoint cadence.
`experiment_id`/`experiment_name`/`output_dir`/`checkpoint_dir` were also
changed, but only as non-scientific bookkeeping so this run does not
overwrite `EXP_004`'s frozen results.

## Final Validation Metrics
Collected on the shared, centralized 5,000-image validation set after
every round (`round_history.csv`, `summary.json`). Best validation
occurred at **round 46**, not the final round:
- **Best Validation Top-1 (round 46):** 65.70%

Final round (49) validation:
- **Final Validation Loss:** 1.200572752380371
- **Final Validation Top-1:** 65.36%
- **Final Validation Top-5:** 90.84%

## Final Test Metrics
The final global model (server weights after round 49's aggregation),
evaluated exactly once on the untouched 10,000-image test set
(`final_test_metrics.json`):

| Metric | Value |
|---|---|
| Final Test Loss | 1.1717977018356323 |
| Final Test Top-1 Accuracy | 66.22% |
| Final Test Top-5 Accuracy | 90.63% |
| Macro Precision | 0.6784646393057799 |
| Macro Recall | 0.6625 |
| Macro F1 | 0.6589107694908175 |
| Balanced Accuracy | 0.6625 |

Total experiment time: 4,241.69 seconds (≈70.7 minutes) for all 50 rounds
(`summary.json`, `total_experiment_seconds`).

## Convergence Summary
Validation top-1 at decade-block boundaries (`round_history.csv`):

| Round | 0 | 9 | 19 | 29 | 39 | 49 |
|---|---:|---:|---:|---:|---:|---:|
| Validation Top-1 | 1.36% | 36.00% | 51.60% | 58.86% | 63.44% | 65.36% |

Marginal validation-top-1 gain shrank block over block — approximately
3.85pp/round (rounds 0–9), 1.56pp/round (10–19), 0.73pp/round (20–29),
0.46pp/round (30–39), and 0.19pp/round (40–49) — a smooth,
continuously-diminishing pattern with no plateau or collapse. The final
three rounds (47–49) show validation top-1 oscillating narrowly
(65.06% → 65.48% → 65.36%) around its round-46 peak (65.70%) while
validation loss continues a slight net decrease over the same window
(1.2107 → 1.2006), the same "noisy accuracy plateau despite continued
marginal loss improvement" pattern seen in `EXP_005`.

## Comparison Against EXP_004
| Metric | EXP_004 (IID) | EXP_006 (Dirichlet α=0.5) | Δ |
|---|---:|---:|---:|
| Final test loss | 1.1049091817855834 | 1.1717977018356323 | +0.0669 |
| Final test top-1 | 68.52% | 66.22% | **−2.30pp** |
| Final test top-5 | 91.27% | 90.63% | **−0.64pp** |
| Macro F1 | 0.6814 | 0.6589 | −0.0225 |
| Balanced accuracy | 0.6850 | 0.6625 | −0.0225 |
| Best validation round | 49 (final) | 46 (not final) | — |
| Client sample counts | 9,000 × 5 (uniform) | 9,696 / 9,079 / 8,487 / 9,255 / 8,483 | — |

`EXP_006` trails `EXP_004` on every reported metric, but both runs still
exceed the centralized `EXP_001` baseline (62.71% test top-1): `EXP_004`
by +5.81pp, `EXP_006` by +3.51pp. Full analysis, including per-block
convergence-rate comparison and a mild/moderate/severe classification of
the degradation, is in `analysis/EXP_006_Dirichlet_A05_Analysis.md`.

## Key Observations
- This is the first federated CIFAR-100 experiment in this project to use
  anything other than IID partitioning, and the first to exercise
  `partition_dataset_dirichlet` (Sprint 6.1) and the FedAvg
  `num_batches_tracked` max-aggregation fix (also Sprint 6.1) in a real,
  non-synthetic run — all 50 rounds completed with 5/5 successful clients
  and 0 rejected updates, matching the reliability record of every prior
  federated experiment in this project.
- Client sample-count imbalance is present but mild: totals range from
  8,483 to 9,255–9,696 samples (mean exactly 9,000, matching `EXP_004`'s
  uniform allocation), a coefficient of variation of ≈5.2%.
  `client_partition_summary.csv` records only per-client totals, not a
  per-class breakdown, so the actual per-client *class*-distribution skew
  (`dirichlet_alpha`'s primary target) cannot be directly verified from
  currently available output files — see Limitations.
- The accuracy and convergence-rate gap to `EXP_004` is concentrated in
  the final third of training (rounds 30–49); through round 29 the two
  runs' per-round marginal gains are close, and in two blocks `EXP_006`
  is briefly ahead.
- Round 0 is not numerically identical to `EXP_004`'s round 0 (unlike the
  `EXP_004`→`EXP_005` comparison) — expected, since changing the
  partitioning *strategy* changes which samples land on which client even
  at a matched seed.

## Scientific Conclusion
Non-IID (Dirichlet, α=0.5) partitioning produces a real but **mild**
degradation relative to `EXP_004`'s IID reference: −2.30 percentage
points test top-1, −0.64 points test top-5, with all macro metrics moving
together by a comparable, modest margin, and no collapse in any tracked
metric. This is consistent with `alpha=0.5` being a moderate, not extreme,
heterogeneity setting. `EXP_004` remains the project's frozen IID
reference protocol; `EXP_006` establishes the first Non-IID data point for
comparison, satisfying `PROJECT_ROADMAP.md` Phase 3's previously-open
"Non-IID client partitioning implemented and tested" criterion.

## Limitations
- Single seed (42); no repeated runs to characterize variance — the
  2.30-point gap to `EXP_004` has not been checked against ordinary
  run-to-run noise.
- Single `alpha` value (0.5); this result does not characterize how the
  accuracy gap changes under stronger (lower-`alpha`) heterogeneity.
- Per-client per-class label distribution was not recorded by this run's
  outputs — only per-client sample totals — so the realized partition's
  class-level skew cannot be directly verified from available files (see
  Key Observations).
- Client processing remains strictly sequential; no attack, attribution,
  or repair logic is present.
- This is a single dataset/model/aggregation combination (CIFAR-100,
  ResNet-18, FedAvg); the finding does not generalize beyond it without
  further testing.

## Recommendation for EXP_007
Per `analysis/EXP_006_Dirichlet_A05_Analysis.md` → Sections 10–12: run a
controlled follow-up changing only `dirichlet_alpha` (`0.5 → 0.1`), all
other settings identical to `EXP_006`, to test whether stronger
heterogeneity produces a correspondingly larger accuracy gap relative to
`EXP_004`. Falsifiable hypothesis: `EXP_007`'s final test top-1 will be
lower than `EXP_006`'s 66.22%, and its gap to `EXP_004`'s 68.52% will
exceed `EXP_006`'s current 2.30 percentage points; if not, the
relationship between `alpha` and FedAvg accuracy is not simply monotonic
in this configuration, which would itself require explanation rather than
being assumed away. `EXP_007`'s config has not been created yet.
