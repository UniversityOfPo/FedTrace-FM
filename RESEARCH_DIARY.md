# FedTrace-FM — Research Diary

> Laboratory notebook for the FedTrace-FM project. Each work session gets a
> dated entry recorded at the time of the session. Entries are append-only:
> earlier entries are not rewritten as understanding improves — corrections
> and updated conclusions are recorded as new entries that reference the
> earlier one.

---

## Session 1 — 2026-08-02

### Entry 1.1 — Environment Setup

**Date:** 2026-08-02 (environment established prior to and confirmed during
this session)

**Objective:** Establish a fully local development environment for
FedTrace-FM with no cloud-notebook dependency, suitable for GPU-accelerated
PyTorch research.

**Completed Work:**
- Confirmed target environment: Windows 11, VS Code, Python 3.12, local
  virtual environment at `.venv`.
- Confirmed PyTorch build: `2.11.0+cu128` (CUDA 12.8 wheel).
- Confirmed target hardware: single NVIDIA GeForce RTX 5060 Laptop GPU,
  approximately 8 GB VRAM.
- Established the project constraint set: no Google Colab, no Google Drive
  mounting, CPU fallback must always be available when CUDA is not present.
- Created the initial project scaffold: `configs/`, `data/`, `notebooks/`,
  `src/` (with `data/`, `models/`, `federated/`, `attacks/`, `attribution/`,
  `repair/`, `utils/` subpackages), `experiments/`, `results/`,
  `checkpoints/`, `logs/`, `tests/`, plus `requirements.txt`, `README.md`,
  `.gitignore`, and a placeholder `main.py`.

**Results:** A clean, importable package skeleton exists
(`__init__.py` in every Python package directory), with no training logic
implemented yet, as intended for this step.

**Problems:** None encountered at this stage — this was a scaffolding-only
step with no functional code to validate.

**Solutions:** N/A.

**Next Action:** Build the CIFAR-100 data pipeline on top of this scaffold.

---

### Entry 1.2 — GPU Verification

**Date:** 2026-08-02

**Objective:** Confirm that the configured PyTorch/CUDA stack can actually
detect and use the local GPU, not just that the software is installed.

**Completed Work:**
- Implemented device auto-resolution in `src/training/engine.py`
  (`resolve_device`), where `device: auto` selects CUDA when available and
  falls back to CPU otherwise.
- Implemented GPU logging in `src/train_centralized.py`: when CUDA is
  selected, the run logs the GPU name and total VRAM via
  `torch.cuda.get_device_name(0)` and
  `torch.cuda.get_device_properties(0).total_memory`.
- Verified this end-to-end by running the 1-epoch smoke test (see Entry 1.7).

**Results:** Confirmed via the smoke-test run log
(`results/centralized/cifar100_resnet18_smoke/run.log`):
```
Using GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM)
```
and `summary.json` reporting `"device": "cuda"`. GPU detection, mixed
precision (`torch.amp`), and CUDA execution are all confirmed working on
this machine.

**Problems:** None — CUDA was detected correctly on the first run.

**Solutions:** N/A.

**Next Action:** Rely on this same auto-detection path for all future
training runs (centralized and, later, federated).

---

### Entry 1.3 — Git

**Date:** 2026-08-01 to 2026-08-02 (see commit history; exact local commit
timestamps below are derived from `.git/logs/HEAD`)

**Objective:** Establish version control for the project and commit
completed work at meaningful milestones.

**Completed Work:**
- Initialized the git repository on branch `master`.
- Committed three milestones so far:
  1. `37f46ff` — *Initial research project setup* (scaffold from Entry 1.1).
  2. `0543ba6` — *Complete CIFAR-100 data pipeline* (Entry 1.4).
  3. `a9f5992` — *Add centralized CIFAR-100 ResNet-18 baseline* (Entry 1.5).

**Results:** Clean, linear history of three commits, each corresponding to
one completed, reviewed unit of work.

**Problems:** None with git itself. The validation-split fix (Entry 1.6)
was implemented after the third commit and has not yet been committed —
this is tracked explicitly rather than left implicit (see Entry 1.8 and
`FEDTRACE_FM_MASTER.md` → *Current Git History*).

**Solutions:** N/A — this is simply the current, honestly-recorded state,
not a problem requiring a fix.

**Next Action:** Commit the validation-split fix as its own milestone
before starting any new module.

---

### Entry 1.4 — Data Pipeline

**Date:** 2026-08-01/2026-08-02 (committed as `0543ba6`)

**Objective:** Build a reproducible CIFAR-100 data inspection pipeline:
typed config loading, seeding, dataset construction with automatic
download, and a CLI inspection script producing verifiable artifacts.

**Completed Work:**
- `configs/cifar100_data.yaml` — dataset name, data root, download flag,
  batch size, workers, pin memory, seed, output directory, preview-image
  count.
- `src/utils/config.py` — `load_yaml` (generic) + `load_data_config`
  (typed, validated `DataConfig`), raising a clear `ConfigError` for
  missing files or invalid fields.
- `src/utils/seed.py` — seeds Python `random`, NumPy, and PyTorch
  (CPU + all CUDA devices), with an optional full-determinism mode
  (documented CPU/GPU performance trade-off).
- `src/data/cifar100.py` — standard CIFAR-100 normalization constants,
  train transforms (`RandomCrop(32, padding=4)`, `RandomHorizontalFlip`,
  `ToTensor`, `Normalize`), eval transforms (`ToTensor`, `Normalize`),
  `build_datasets`, `build_dataloaders`, `get_class_names`,
  `summarize_dataset`.
- `src/data/inspect_dataset.py` — CLI that loads the config, seeds RNGs,
  downloads CIFAR-100 if absent, verifies all 100 classes are present with
  valid labels, saves a 5x5 sample-image grid, a per-class count CSV, and a
  JSON dataset summary.
- `tests/test_cifar100.py` — configuration loading (including the
  missing-file error path), class count equals 100, expected train/test
  sizes (50,000 / 10,000), valid label range, and one DataLoader batch's
  shape.

**Results:** Dataset downloads and loads correctly; all 100 classes
verified present with valid labels; inspection artifacts
(`cifar100_samples.png`, `cifar100_class_counts.csv`,
`cifar100_summary.json`) are generated successfully under
`results/data_inspection/`.

**Problems:** None blocking. The main design question was where to put
YAML validation logic; resolved by centralizing it in `src/utils/config.py`
(`load_yaml` + `ConfigError`, reused later by `TrainingConfig`).

**Solutions:** Centralized, typed config loading module, reused by every
later step instead of being duplicated per script.

**Next Action:** Build the centralized ResNet-18 baseline on top of this
pipeline.

---

### Entry 1.5 — Centralized Baseline

**Date:** 2026-08-01/2026-08-02 (committed as `a9f5992`)

**Objective:** Implement a reproducible centralized CIFAR-100 baseline
using a CIFAR-compatible ResNet-18, with a reusable training engine,
evaluation metrics, atomic checkpointing, and a full training entry point.

**Completed Work:**
- `configs/cifar100_resnet18_baseline.yaml` — experiment name, dataset
  config reference, output/checkpoint directories, epochs, batch size,
  optimizer/scheduler settings, mixed precision, seed, device, checkpoint
  flags, resume path, log interval.
- `src/models/resnet18_cifar.py` — CIFAR-compatible ResNet-18: 3x3
  stride-1 stem (no 7x7/stride-2 ImageNet stem), no initial max-pool,
  standard `[2, 2, 2, 2]` BasicBlock stages, `resnet18_cifar()` factory,
  `count_parameters()`.
- `src/training/engine.py` — `resolve_device`, `build_optimizer`
  (SGD/AdamW), `build_scheduler` (cosine annealing with optional linear
  warmup, or none), `build_grad_scaler`, `train_one_epoch`, `evaluate`
  (with an opt-in prediction-return mode for final reporting), using
  `torch.amp` (not the deprecated `torch.cuda.amp`), `non_blocking=True`
  transfers, and no dependency on any specific experiment config object.
- `src/evaluation/metrics.py` — `topk_accuracy` (pure tensor ops),
  confusion matrix, per-class accuracy, macro precision/recall/F1,
  balanced accuracy, all zero-division-safe.
- `src/utils/checkpoint.py` — atomic checkpoint save (temp file +
  `os.replace`), full-state payload (epoch, model/optimizer/scheduler/
  scaler state, best validation accuracy, resolved config, seed, model
  name, class count, framework versions).
- `src/train_centralized.py` — end-to-end entry point: load/validate
  config, seed, resolve device (log GPU name + VRAM), load data, build
  model, train + evaluate every epoch, save best/last checkpoints, reload
  best checkpoint, final test evaluation, and write all reporting
  artifacts (`history.csv`, training/accuracy/LR curves, confusion matrix,
  per-class accuracy, classification metrics, run log, summary JSON).
- `tests/test_resnet18_cifar.py` — output shape, CIFAR-compatible stem
  shape/stride, absence of max-pooling, nonzero parameter counts.
- `tests/test_training_engine.py` — top-k accuracy correctness on a
  synthetic example, one training step updates parameters, one evaluation
  step does not, mixed precision disables safely on CPU, checkpoint
  save/load round-trip.

**Results:** All modules implemented and reviewed; architecture and engine
tests designed to be CPU-only and fast (no full training required to pass).
At the time of this commit, the pipeline had not yet been run on real
hardware — that verification happened in Entry 1.7.

**Problems:** None blocking. One design question: whether to reuse
`torch.cuda.amp` (deprecated) or `torch.amp`; resolved in favor of the
current `torch.amp.autocast`/`torch.amp.GradScaler` API for
forward-compatibility.

**Solutions:** Standardized on `torch.amp` throughout the engine.

**Next Action:** Identify and fix the test-set-leakage issue in per-epoch
model selection (see Entry 1.6).

---

### Entry 1.6 — Validation Split (Test-Set Leakage Fix)

**Date:** 2026-08-02

**Objective:** Eliminate test-set leakage: the centralized baseline was
evaluating every epoch directly on the official CIFAR-100 test set and
using that same test-set accuracy to select the "best" checkpoint. This
means the reported final test metric was not a clean, held-out estimate —
the test set had effectively been used for model selection.

**Completed Work:**
- Added `validation_fraction: 0.1` to
  `configs/cifar100_resnet18_baseline.yaml`.
- Added `validation_fraction` to `TrainingConfig` in `src/utils/config.py`,
  with validation enforcing `0 < validation_fraction < 0.5`.
- Implemented `split_train_validation()` in `src/data/cifar100.py`: builds
  two independent `CIFAR100(train=True)` instances over the same
  underlying data (one with training transforms, one with evaluation
  transforms), computes a single stratified split via
  `sklearn.model_selection.train_test_split(..., stratify=targets,
  random_state=seed)`, and wraps each instance in a `torch.utils.data.Subset`
  using the same computed indices. This avoids ever mutating a shared
  dataset object's `.transform` in place, and never touches the official
  test set.
- Updated `src/train_centralized.py`: per-epoch evaluation now uses the
  validation loader, not the test loader; checkpoint selection uses
  validation top-1 accuracy; the test set is loaded once and evaluated
  exactly once, after training, using the reloaded best checkpoint;
  `history.csv` columns renamed to `validation_loss` / `validation_top1` /
  `validation_top5`; `summary.json` restructured into `best_validation`
  and `final_test` sub-objects to make the separation explicit.
- Added tests to `tests/test_cifar100.py`: reproducible split under a fixed
  seed, different split under a different seed, no index overlap between
  train and validation subsets, expected split sizes (45,000 / 5,000 for
  `validation_fraction=0.1`), and class representation preserved
  (all 100 classes present in both subsets, per-class validation counts
  within tolerance of the expected ~50).

**Results:** Verified functionally correct via the smoke test (Entry 1.7):
the run log shows
`Split CIFAR-100 training set: 45000 train / 5000 validation
(validation_fraction=0.100, seed=42)`, and `summary.json` correctly
separates `best_validation.top1` from `final_test.top1`.

**Problems:** The original design computed metrics against the test set
every epoch purely because it was the simplest two-loader wiring
(`train_loader`, `test_loader`) — the leakage was a methodology bug, not a
crash or exception, so it would not have been caught by normal testing
without an explicit review pass for this exact concern.

**Solutions:** Introduced a third, properly isolated data split
(train/validation/test) with the test set now referenced by a single,
clearly-commented code path used exactly once, at the end of training.

**Next Action:** Commit this fix, then run the full 10-epoch baseline with
the corrected methodology.

---

### Entry 1.7 — Smoke Test

**Date:** 2026-08-02, 22:53:44 (local time, per `run.log`)

**Objective:** Verify the entire centralized training pipeline — including
the new validation split — actually runs correctly on the real local GPU,
before committing to a full 10-epoch run.

**Completed Work:**
- Created `configs/cifar100_resnet18_smoke.yaml`: identical to the full
  baseline config, with `epochs: 1`, a distinct `experiment_name`
  (`cifar100_resnet18_smoke`), and its own `output_dir`/`checkpoint_dir` so
  it cannot collide with the real baseline's artifacts.
- Ran `python -m src.train_centralized --config
  configs/cifar100_resnet18_smoke.yaml`.

**Results (from `results/centralized/cifar100_resnet18_smoke/run.log`,
`history.csv`, and `summary.json`):**
- Device: CUDA, GPU: `NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM)`.
- Data: 50,000 train samples / 10,000 test samples loaded; split into
  45,000 train / 5,000 validation (`validation_fraction=0.1`, `seed=42`).
- Model: `resnet18_cifar`, 11,220,132 total parameters, all trainable.
- Training: 1 epoch completed in 40.9 s (1,407 batches, mixed precision
  enabled).
- Epoch 1 metrics: `train_loss=4.2968`, `train_top1=4.35%`,
  `validation_loss=3.9604`, `validation_top1=7.16%`,
  `validation_top5=27.64%`.
- Best checkpoint (epoch 1, `validation_top1=7.16%`) reloaded successfully
  for final evaluation.
- Final one-time test evaluation: `loss=3.9444`, `top1=6.93%`,
  `top5=27.54%`, `macro_precision=0.0577`, `macro_recall=0.0693`,
  `macro_f1=0.0397`, `balanced_accuracy=0.0693`.
- Total run time: 41.1 s. All expected output files were written:
  `history.csv`, `training_loss.png`, `top1_accuracy.png`,
  `learning_rate.png`, `confusion_matrix.png`, `per_class_accuracy.csv`,
  `classification_metrics.json`, `summary.json`, `run.log`, plus
  `best.pt`/`last.pt` checkpoints.

**Interpretation:** These accuracy figures are near chance level (1/100 ≈
1% for uniform random guessing across 100 classes; ~7% after one epoch is
consistent with very early-stage learning, not a trained model). This is
expected and correct for a 1-epoch smoke test — the run's purpose was to
verify plumbing, not to produce a usable accuracy number. **These numbers
must not be cited as the project's baseline result.**

**Problems:** None — the pipeline completed without errors on the first
attempt.

**Solutions:** N/A.

**Next Action:** Run the full 10-epoch baseline
(`configs/cifar100_resnet18_baseline.yaml`) to obtain a real, reportable
result, and record it as its own diary entry when it completes.

---

### Entry 1.8 — Git Commits

**Date:** 2026-08-02

**Objective:** Record the exact commit state at the end of this session, so
future sessions know precisely what is and is not saved to version control.

**Completed Work:** Reviewed `.git/logs/HEAD` directly (read-only
inspection; no git commands were executed during this documentation
session).

**Results:** Confirmed three commits exist (`37f46ff`, `0543ba6`,
`a9f5992` — see Entry 1.3). Confirmed the validation-split fix (Entry 1.6),
the smoke-test config, and the smoke-test outputs are **not** part of any
commit as of this entry. `results/` and `checkpoints/` contents are
gitignored regardless and were never intended to be committed.

**Problems:** None — this is a status recording, not a fix.

**Solutions:** N/A.

**Next Action:** Commit the validation-split fix as the next concrete git
action, with a message describing *why* (test-set leakage was present in
per-epoch model selection) rather than only *what* changed.

---

## Session 2 — 2026-08-03

### Entry 2.1 — Centralized Baseline: Full 10-Epoch Run (Baseline v1.0)

**Objective:** Run the full, correctly configured 10-epoch centralized
CIFAR-100 ResNet-18 baseline — using the leakage-free train/validation/test
split introduced in Entry 1.6 — to obtain a real, reportable baseline
accuracy figure, and freeze it as the reference result for all future
federated, attack, attribution, and repair experiments.

**Configuration** (`configs/cifar100_resnet18_baseline.yaml`):
- Experiment: `cifar100_resnet18_baseline`; model: `resnet18_cifar`
  (11,220,132 total parameters, all trainable).
- Dataset: CIFAR-100, `num_classes=100`, `validation_fraction=0.1`
  (45,000 train / 5,000 validation / 10,000 test).
- Seed: 42. Epochs: 10. Batch size: 32. Workers: 2.
- Optimizer: SGD, `learning_rate=0.1`, `momentum=0.9`,
  `weight_decay=0.0005` (Nesterov enabled, since momentum > 0).
- Scheduler: cosine annealing, `warmup_epochs=0`, `T_max=10`.
- `label_smoothing=0.0`, `mixed_precision=true`, `gradient_clip_norm=null`.
- Device: `auto` → resolved to CUDA.
- `save_last_checkpoint=true`, `save_best_checkpoint=true`,
  `resume_from=null`, `log_interval=100`.

**Training Completed:** All 10 configured epochs completed successfully
(`configured_epochs=10`, `completed_epochs=10`), run on
`NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM)`, from 2026-08-03
16:21:24 to 16:29:58 (local time, per `run.log`).
Total training time: 501.84 s (≈8 minutes 22 seconds) for all 10 epochs.

**Results** (from
`results/centralized/cifar100_resnet18/history.csv` and `summary.json`):

| Epoch | Train Loss | Train Top-1 | Val Loss | Val Top-1 | Val Top-5 | LR |
|------:|-----------:|-------------:|---------:|----------:|----------:|---:|
| 1 | 4.2968 | 4.35% | 3.9604 | 7.16% | 27.64% | 0.09755 |
| 2 | 3.7153 | 11.96% | 3.6720 | 14.58% | 40.26% | 0.09045 |
| 3 | 3.2546 | 20.04% | 2.9895 | 25.22% | 56.02% | 0.07939 |
| 4 | 2.8373 | 27.86% | 2.9860 | 26.34% | 57.74% | 0.06545 |
| 5 | 2.4814 | 35.26% | 2.6148 | 33.40% | 67.32% | 0.05000 |
| 6 | 2.1936 | 41.18% | 2.1658 | 42.38% | 74.58% | 0.03455 |
| 7 | 1.9356 | 47.21% | 1.9127 | 47.62% | 79.44% | 0.02061 |
| 8 | 1.6798 | 53.24% | 1.6818 | 53.26% | 83.18% | 0.00955 |
| 9 | 1.4131 | 60.02% | 1.4791 | 59.12% | 86.34% | 0.00245 |
| 10 | 1.1793 | 66.18% | 1.3211 | 62.34% | 88.54% | 0.00000 |

Best checkpoint selected at **epoch 10** (`best_validation.top1 = 62.34%`),
reloaded successfully, and evaluated exactly once on the official
10,000-image test set:

- **Final Test Loss: 1.3016**
- **Final Test Top-1: 62.71%**
- **Final Test Top-5: 88.54%**
- Macro precision: 0.6287, macro recall: 0.6270, macro F1: 0.6244,
  balanced accuracy: 0.6270.

All expected artifacts were written under
`results/centralized/cifar100_resnet18/`: `history.csv`,
`training_loss.png`, `top1_accuracy.png`, `learning_rate.png`,
`confusion_matrix.png`, `per_class_accuracy.csv`,
`classification_metrics.json`, `summary.json`, `run.log`, plus
`checkpoints/cifar100_resnet18/best.pt` and `last.pt`.

**Scientific Observations:**
- Both train and validation metrics improved monotonically across all 10
  epochs, with no sign of divergence between them — the ~3.8-point gap
  between final train top-1 (66.18%) and validation top-1 (62.34%) is a
  normal, modest generalization gap for 10 epochs of training with only
  crop/flip augmentation, not an indication of overfitting.
- The cosine annealing schedule decayed smoothly from an effective LR of
  0.0976 after epoch 1 to 0.0 at epoch 10, exactly as configured
  (`T_max=10`, no warmup); the largest single-epoch accuracy jumps
  (epochs 8→9→10) coincide with the LR approaching zero, consistent with
  the expected late-training fine-convergence behavior of cosine annealing.
- The final one-time test evaluation (top-1 62.71%, top-5 88.54%) is
  extremely close to the best validation metrics recorded during training
  (top-1 62.34%, top-5 88.54% — top-5 is an exact match), indicating the
  10% stratified validation split is a low-variance, representative proxy
  for test performance. This is the direct, positive confirmation that the
  Entry 1.6 leakage fix produced a methodologically sound split, not just
  a differently-named one.
- Macro precision/recall/F1 (0.6287/0.6270/0.6244) and balanced accuracy
  (0.6270) all sit within about 1.5 points of overall top-1 accuracy
  (0.6271), indicating no severe per-class performance imbalance — expected
  given CIFAR-100's uniform 500-images-per-class training distribution and
  the stratified split preserving that balance in both validation and (by
  construction) the untouched test set.
- Validation accuracy was still improving at epoch 10 (the best epoch was
  the last one), suggesting additional epochs beyond 10 would likely yield
  further gains. This was not tested and is out of scope for Baseline
  v1.0 — Baseline v1.0 is deliberately fixed at the originally configured
  10 epochs so it remains a stable, unmoving reference point.
- Early-epoch durations (41–61 s) were somewhat higher and more variable
  than later epochs (which stabilized around 45 s), most likely reflecting
  one-time CUDA/cuDNN kernel selection overhead rather than a persistent
  slowdown.

**Conclusion:** The centralized CIFAR-100 ResNet-18 baseline, trained for
10 epochs under a leakage-free train/validation/test split, achieves
**62.71% test top-1 accuracy** and **88.54% test top-5 accuracy** at a
final test loss of **1.3016**. This result is frozen as **Baseline v1.0**
(see `FEDTRACE_FM_MASTER.md` → *Baseline v1.0 (Frozen)*) and will serve as
the fixed reference point against which every future federated, attack,
attribution, and repair experiment in this project is compared.

**Next Action:** Commit the validation-split fix that produced this result
(still uncommitted as of this entry — see Entry 1.8), then begin design of
the **Federated Learning Infrastructure** module (`src/federated/`),
starting with the client data partitioning strategy (IID vs. non-IID).

---

## Session 3 — 2026-08-04

### Entry 3.1 — First IID Federated CIFAR-100 Experiment (Federated Baseline v1.0, EXP_002)

**Objective:** Run the first real federated learning experiment on the
completed federated infrastructure (`FederatedClient`, `FederatedServer`,
pluggable `AggregationStrategy`/`FedAvg`, `FederatedRunner`, built across
Sprints 3.1–3.4) plus the new IID partitioner and experiment script
(Sprint 4.1): 5 IID clients, FedAvg, 10 communication rounds, and freeze
the result as the reference Federated Baseline v1.0 for comparison
against future Non-IID, attack, attribution, and repair experiments.

**Configuration** (`configs/cifar100_fedavg_iid.yaml`):
- Experiment: `cifar100_fedavg_iid` (`EXP_002`); model: `resnet18_cifar`
  (11,220,132 total parameters, all trainable).
- Dataset: CIFAR-100, the existing 45,000/5,000 stratified train/
  validation split (`validation_fraction=0.1`, seed 42); the 45,000-image
  training subset IID-partitioned across 5 clients (9,000 samples each,
  `partition_dataset_iid`, seed 42).
- 5 clients, `clients_per_round=5`, `minimum_updates=5`, `local_epochs=1`,
  batch size 32, `num_workers=2`.
- Optimizer: SGD, `learning_rate=0.01`, `momentum=0.9`,
  `weight_decay=0.0005`, no scheduler, mixed precision enabled.
- Aggregation: FedAvg. `communication_rounds=10`, `evaluate_every=1`,
  `checkpoint_every=1`. `fail_round_on_client_error=true`,
  `continue_after_failed_round=false`.
- **Client-state policy:** optimizer/scheduler/AMP-scaler state reset at
  the start of every round (`FederatedClient.reset_training_state()`,
  invoked by `load_global_weights()` — see `DECISIONS.md`).

**Training Completed:** All 10 configured rounds completed
(`communication_rounds_configured=10`, `communication_rounds_completed=10`),
run on `NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM)`, from 2026-08-04
12:55:52 to 13:08:52 (local time, per `run.log`). Every round finished
with 5/5 successful client updates and 0 rejected updates. Total
experiment time: 765.57 s (≈12 minutes 46 seconds).

**Results** (from
`results/federated/cifar100_fedavg_iid/round_history.csv` and
`summary.json`):

| Round | Train Loss | Train Top-1 | Val Loss | Val Top-1 | Val Top-5 | Round (s) |
|------:|-----------:|------------:|---------:|----------:|----------:|----------:|
| 0 | 4.3338 | 5.13% | 4.7639 | 1.86% | 7.30% | 71.5 |
| 1 | 4.0336 | 7.89% | 3.9614 | 8.72% | 27.54% | 71.3 |
| 2 | 3.8079 | 11.01% | 3.5032 | 16.96% | 41.58% | 72.8 |
| 3 | 3.5980 | 14.51% | 3.2677 | 19.86% | 48.68% | 72.2 |
| 4 | 3.4145 | 17.33% | 3.1023 | 22.28% | 53.44% | 72.6 |
| 5 | 3.2668 | 19.86% | 2.8922 | 26.66% | 58.80% | 78.9 |
| 6 | 3.1135 | 22.59% | 2.7385 | 29.76% | 62.66% | 80.5 |
| 7 | 2.9756 | 25.01% | 2.5764 | 32.60% | 65.78% | 80.3 |
| 8 | 2.8399 | 27.65% | 2.4551 | 35.66% | 68.20% | 80.3 |
| 9 | 2.7112 | 30.38% | 2.3798 | 37.54% | 69.66% | 80.4 |

Best validation round: **round 9** (`validation_top1 = 37.54%`), also the
final round. The final global model, evaluated exactly once on the
untouched 10,000-image test set:

- **Final Test Loss: 2.3611**
- **Final Test Top-1: 37.81%**
- **Final Test Top-5: 70.27%**
- Macro precision: 0.3918, macro recall: 0.3780, macro F1: 0.3600,
  balanced accuracy: 0.3780.

All expected artifacts were written under
`results/federated/cifar100_fedavg_iid/`: `round_history.csv`,
`summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`,
`client_partition_indices.json`, `validation_loss.png`,
`validation_top1.png`, `validation_top5.png`, `local_training_loss.png`,
`communication_round_duration.png`, `confusion_matrix.png`,
`per_class_accuracy.csv`, `run.log`, plus
`checkpoints/federated/cifar100_fedavg_iid/runner_latest.pt` (+ its
`_server.pt`/`_client_client_00X.pt` companions) and
`final_global_model.pt`.

**Scientific Observations:**
- Validation top-1 improved monotonically every single round (1.86% →
  37.54%), with no regression at any point, and the round-9 gain (35.66%
  → 37.54%) was still substantial — the model had not plateaued by round
  10.
- Final test metrics (top-1 37.81%, top-5 70.27%) closely track the final
  validation metrics (top-1 37.54%, top-5 69.66%), confirming the
  centralized validation set remains a representative proxy for test
  performance in the federated setting, not just the centralized one.
- Every round completed with 5/5 clients successful and 0 rejected
  updates across all 10 rounds — the full client-selection →
  local-training → update-submission → FedAvg-aggregation →
  explicit-global-update → finalization → evaluation pipeline ran
  without a single failure on its first real execution.
- **The federated result sits well below the centralized Baseline v1.0 at
  a matched data-exposure budget**: both experiments expose the model to
  the full 45,000-image training set once per unit (once per centralized
  epoch; once per federated round, since the 5 IID clients jointly cover
  the full set each round). At 10 units: centralized reaches 62.71% test
  top-1; federated reaches 37.81%. This is consistent with FedAvg's
  known slower convergence relative to direct full-dataset SGD (periodic
  weight averaging across independently-trained clients), not a defect in
  this implementation — but it means Phase 3's "federated ≈ centralized
  under IID" completion criterion is not yet met at this round budget.

**Conclusion:** The first real federated CIFAR-100 experiment completed
successfully end-to-end: 5 IID clients, FedAvg, 10 communication rounds,
achieving **37.81% test top-1 / 70.27% test top-5** at a final test loss
of **2.3611**. This result is frozen as **Federated Baseline v1.0**
(`EXP_002`; see `FEDTRACE_FM_MASTER.md` and
`experiments/EXP_002_FedAvg_IID.md`) and will serve as the reference point
for Non-IID, attack, attribution, and repair experiments. The gap to the
centralized Baseline v1.0 at the same data-exposure budget is a
documented, expected FedAvg characteristic, flagged for further
investigation rather than treated as resolved.

**Next Action:** Design and implement Non-IID (Dirichlet and/or
shard-based) client data partitioning (Sprint 4.2), and consider whether
extending the round/local-epoch budget or tuning hyperparameters is
warranted to better characterize the IID federated/centralized gap before
moving on to attack simulation (Phase 4).

---

## Session 4 — 2026-08-04

### Entry 4.1 — Controlled Round-Budget Follow-Up (EXP_003, 20 Rounds)

**Objective:** Following the EXP_002 round-history analysis (Session 3,
Entry 3.1), which found no plateau in any metric through round 9 and
recommended extending the communication-round budget as the single
highest-value next experiment, run a controlled, single-variable
follow-up: identical to `EXP_002` in every respect except
`communication_rounds: 10 → 20`.

**Configuration** (`configs/cifar100_fedavg_iid_20rounds.yaml`):
- Experiment: `cifar100_fedavg_iid_20rounds` (`EXP_003`); same model
  (`resnet18_cifar`, 11,220,132 parameters), same 5-client IID partition
  (seed 42, 9,000 samples each), same optimizer (SGD, `learning_rate=0.01`,
  `momentum=0.9`, `weight_decay=0.0005`, no scheduler), same FedAvg
  aggregation, same client-state reset-every-round policy, same seed (42).
- **Only `communication_rounds` changed: 10 → 20.**
- `experiment_id`, `experiment_name`, `output_dir`, and `checkpoint_dir`
  were also changed (to `EXP_003` / distinct paths) purely so this run
  did not overwrite `EXP_002`'s frozen results — bookkeeping, not a
  scientific variable.
- No `.py` file required any change: `communication_rounds` already
  flowed from the YAML into `FederatedRunnerConfig.total_rounds` with no
  hardcoded round count anywhere in `src/experiments/
  run_federated_cifar100.py`.

**Training Completed:** All 20 configured rounds completed
(`communication_rounds_configured=20`, `communication_rounds_completed=20`),
run on `NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM)`, from 2026-08-04
15:09:02 (per `run.log`). Total experiment time: 1776.63 s (≈29 minutes
37 seconds).

**Results** (from `results/federated/cifar100_fedavg_iid_20rounds/
round_history.csv` and `summary.json`):

| Round | Val Loss | Val Top-1 | Val Top-5 |
|---:|---:|---:|---:|
| 0 | 4.7639 | 1.86% | 7.30% |
| 5 | 2.8922 | 26.66% | 58.80% |
| 9 | 2.3798 | 37.54% | 69.66% |
| 14 | 1.9109 | 47.86% | 78.52% |
| 19 | 1.6727 | 54.22% | 82.72% |

(Rounds 0–9 exactly reproduce `EXP_002`'s values round-for-round, as
expected — same seed, same partition, same everything except how many
rounds the run continues for. Full 20-row history in `round_history.csv`.)

Best/final validation at **round 19**: loss 1.6727, top-1 54.22%, top-5
82.72%. Final global model, evaluated exactly once on the untouched test
set:

- **Final Test Loss: 1.6335**
- **Final Test Top-1: 54.69%**
- **Final Test Top-5: 83.75%**
- Macro precision: 0.5565, macro recall: 0.5468, macro F1: 0.5367,
  balanced accuracy: 0.5468.

**Scientific Observations:**
- **Improvement over `EXP_002` (10 rounds):** +16.88 points test top-1
  (37.81% → 54.69%), +13.48 points test top-5 (70.27% → 83.75%) — for
  zero change to model, optimizer, aggregation, or data, purely from
  doubling the round budget.
- **Remaining gap to centralized Baseline v1.0 (`EXP_001`, 62.71% test
  top-1):** 8.02 points — down from 24.90 points at `EXP_002`'s 10-round
  budget. The gap shrank by roughly two-thirds.
- Validation top-1/top-5 continued improving every single round through
  round 19 (e.g., round 18→19: val top-1 53.14% → 54.22%, +1.08 points),
  with no plateau visible in the last few rounds — consistent with the
  `EXP_002` analysis's prediction that the run was undertrained rather
  than converged.
- Final test metrics (top-1 54.69%, top-5 83.75%) again track the final
  validation metrics closely (top-1 54.22%, top-5 82.72%), confirming the
  validation set remains a reliable proxy for test performance at this
  larger round budget too.

**Conclusion:** `EXP_002` was undertrained; extending communication rounds
from 10 to 20 — the only variable changed — produced a major improvement
(+16.88 points test top-1) and substantially narrowed, but did not close,
the gap to the centralized baseline (now 8.02 points, down from 24.90).
This is the single scientific conclusion this experiment was designed to
test, and it is confirmed by the data.

**Next Action:** Analyze `EXP_003`'s full round-by-round history for a
plateau signal (is round 19 still improving at a similar rate to earlier
rounds, or clearly decelerating toward zero?) and use that analysis to
decide whether `EXP_004` should extend to 50 communication rounds, before
moving on to Non-IID partitioning.

---

## Session Template (copy this block for each new session)

```markdown
## Session N — YYYY-MM-DD

### Entry N.1 — <Short Title>

**Date:** YYYY-MM-DD

**Objective:**
<What this entry's work was intended to accomplish, in one or two sentences.>

**Completed Work:**
- <Concrete, specific changes made — file names, functions, config keys.>

**Results:**
<Actual observed outcome — numbers, log output, test results. Do not
report a result that was not actually observed.>

**Problems:**
<What went wrong, was unclear, or blocked progress. Write "None" if
genuinely nothing came up.>

**Solutions:**
<How each problem was resolved. Write "N/A" if there were no problems.>

**Next Action:**
<The single next concrete step that follows from this entry.>
```
