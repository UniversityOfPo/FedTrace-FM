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
