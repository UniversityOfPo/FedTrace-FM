# EXP_001 — Centralized CIFAR-100 ResNet-18 Baseline

## Experiment ID
`EXP_001`

## Date
2026-08-03

## Objective
Establish a reproducible, leakage-free centralized CIFAR-100 ResNet-18
baseline and freeze its result as **Baseline v1.0** — the fixed reference
point against which every future federated learning, attack simulation,
FedTrace attribution, and repair experiment in this project will be
compared.

## Dataset
CIFAR-100 (50,000 training images, 10,000 test images, 100 classes),
loaded via `src/data/cifar100.py`. The 50,000 training images were split
into a 45,000-image training subset and a 5,000-image validation subset
using a stratified, seeded split (`split_train_validation`), preserving
proportional class representation in both subsets. The 10,000-image test
set was left untouched throughout training and evaluated exactly once,
after training, using the best checkpoint.

## Model
CIFAR-compatible ResNet-18 (`src/models/resnet18_cifar.py`): 3x3 stride-1
stem (no ImageNet-style 7x7/stride-2 stem), no initial max-pooling,
standard `[2, 2, 2, 2]` BasicBlock stages, 100-class output head, no
pretrained weights.
**Total parameters:** 11,220,132 (all trainable).

## Seed
`42` (applied to Python `random`, NumPy, and PyTorch CPU/CUDA RNGs via
`src/utils/seed.py::set_seed`, with deterministic-mode settings enabled).

## Epochs
10 (configured and completed in full — `configured_epochs=10`,
`completed_epochs=10`).

## Optimizer
SGD with Nesterov momentum (`momentum=0.9`, enabled automatically because
`momentum > 0`), `weight_decay=0.0005`.

## Learning Rate
Base learning rate `0.1`, governed by the scheduler below.

## Scheduler
Cosine annealing (`torch.optim.lr_scheduler.CosineAnnealingLR`,
`T_max=10`, no warmup — `warmup_epochs=0`). The learning rate decayed
smoothly from an effective post-epoch-1 value of `0.09755` down to `0.0`
at epoch 10.

## Batch Size
32 (training config's `batch_size`, which takes precedence over the
dataset config's default).

## Validation Split
10% (`validation_fraction=0.1`) of the 50,000-image training set, carved
out via a stratified, seeded split — 45,000 training images / 5,000
validation images. Used for per-epoch model selection; the official test
set was not touched until the single final evaluation.

## Hardware
NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM), local single-GPU
execution, mixed precision enabled (`mixed_precision=true`, via
`torch.amp`). Total training time: 501.84 seconds (≈8 minutes 22 seconds)
for all 10 epochs.

## Final Metrics
Best checkpoint selected at **epoch 10** by validation top-1 accuracy
(`best_validation.top1 = 0.6234`), reloaded, and evaluated exactly once on
the official 10,000-image test set:

| Metric | Value |
|---|---|
| Final Test Loss | 1.3016 |
| Final Test Top-1 Accuracy | 62.71% |
| Final Test Top-5 Accuracy | 88.54% |
| Macro Precision | 0.6287 |
| Macro Recall | 0.6270 |
| Macro F1 | 0.6244 |
| Balanced Accuracy | 0.6270 |

Full per-epoch history is recorded in
`results/centralized/cifar100_resnet18/history.csv`; full summary in
`results/centralized/cifar100_resnet18/summary.json`.

## Key Observations
- Training and validation metrics improved monotonically across all 10
  epochs, with no divergence between them — the ~3.8-point gap between
  final train top-1 (66.18%) and validation top-1 (62.34%) is a modest,
  expected generalization gap, not overfitting.
- The final one-time test evaluation (top-1 62.71%, top-5 88.54%) closely
  matched the best validation metrics recorded during training (top-1
  62.34%, top-5 88.54% — an exact match on top-5), confirming the 10%
  stratified validation split is a representative, low-variance proxy for
  test performance.
- Macro-averaged precision/recall/F1 and balanced accuracy all sit within
  about 1.5 points of overall top-1 accuracy, indicating no severe
  per-class performance imbalance across the 100 classes.
- Validation accuracy was still improving at epoch 10 (the best epoch was
  the last one), suggesting further epochs would likely improve accuracy
  further — intentionally not explored here, since Baseline v1.0 is fixed
  at the originally configured 10 epochs to serve as a stable reference
  point.

## Lessons Learned
- The training pipeline had an initial test-set-leakage bug: per-epoch
  model selection was evaluating directly against the official test set.
  This was caught and fixed (train/validation/test split, per-epoch
  selection on validation only, test set evaluated exactly once) before
  this experiment was run. This experiment's close agreement between
  validation and test metrics is itself evidence the fix was correct.
- Running a cheap 1-epoch smoke test before committing to the full
  10-epoch run caught pipeline-integration issues early (GPU detection,
  checkpoint saving, artifact writing) without spending the full training
  time budget.
- Cosine annealing without warmup produced smooth, stable convergence for
  this model/dataset/batch-size combination; no instability or loss spikes
  were observed at any point in the 10 epochs.

## Limitations
- Only 10 epochs were run, with no hyperparameter search (learning rate,
  weight decay, and batch size were all set to single, untuned values).
  62.71% test top-1 should be read as a first reference point, not an
  upper bound on this architecture's achievable CIFAR-100 accuracy.
- Single-seed result (`seed=42`); run-to-run variance across seeds has not
  been measured.
- Single-GPU, single-machine training only; no distributed or federated
  setting has been exercised yet.
- No hyperparameter or architecture ablations were performed as part of
  this experiment.

## Future Comparison Targets
Baseline v1.0 (this experiment) is the fixed reference point for:
- **Federated learning (IID, no attack)** — expected to approximately
  match this centralized result, as a sanity check that the federated
  simulation itself is implemented correctly (Phase 3).
- **Federated learning under attack, without attribution/repair** —
  expected to show measurable degradation relative to this baseline
  (Phase 4).
- **Federated learning under attack, with FedTrace attribution and
  repair** — expected to recover accuracy closer to this baseline relative
  to the undefended attacked case (Phase 5).
- **Foundation-model backbones (LoRA-adapted ViT/DINOv2/CLIP)** — expected
  to be compared against this CNN-scale result to assess whether
  foundation-model backbones offer an accuracy advantage on CIFAR-100
  under the same federated/attack/attribution conditions (Phase 6).

## Git Commit History
As of this experiment, the repository has four commits:
1. `37f46ff` — Initial research project setup.
2. `0543ba6` — Complete CIFAR-100 data pipeline.
3. `a9f5992` — Add centralized CIFAR-100 ResNet-18 baseline.
4. `7937bb0` — Add project documentation and research roadmap.

The validation-split leakage fix that produced this experiment's result
is **not yet committed** — it exists in the working tree on top of commit
`a9f5992`. This experiment's exact code state therefore corresponds to
commit `a9f5992` plus that uncommitted fix, not to any single commit hash
alone. Committing that fix is the immediate next git action for this
project (see `FEDTRACE_FM_MASTER.md` → *Current Next Task*).

## Next Experiment
`EXP_002` (planned): Federated Learning Infrastructure — an IID,
no-attack federated simulation of this same dataset/model/hyperparameter
configuration, validated against this experiment's frozen Baseline v1.0
result as a correctness sanity check before attack simulation (Phase 4) is
introduced.
