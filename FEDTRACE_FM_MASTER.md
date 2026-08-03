# FedTrace-FM — Master AI Memory

> **Purpose of this file:** This document is the single authoritative memory
> record for the FedTrace-FM research codebase. It is written so that any AI
> model or human collaborator — with **zero prior conversation history** —
> can read this file and understand what the project is, what exists today,
> what rules govern how work is done here, and what to do next. If this file
> and the actual code ever disagree, the code is the source of truth for
> implementation details, but this file is the source of truth for intent,
> history, and process.

---

## Project Overview

**FedTrace-FM** is a local, GPU-accelerated research codebase investigating
traceability and attribution in federated learning (FL) systems, with an
eventual extension toward foundation-model (FM) scale architectures. The
name combines **Fed**erated learning + **Trace**ability + **F**oundation
**M**odel.

The project currently exists as a single-machine research platform: all
code runs locally on one workstation, with no cloud notebook dependency
(no Google Colab, no Google Drive mounting) and no distributed training
infrastructure yet. It is designed to grow incrementally, phase by phase,
from a centralized image-classification baseline toward a full federated
learning + attack + attribution + repair pipeline, and eventually toward
foundation-model backbones and a medical-imaging case study.

## Research Goal

Investigate how malicious, corrupted, or otherwise anomalous client
contributions in a federated learning system can be:

1. **Detected** — identified as suspicious relative to honest client updates.
2. **Attributed** — traced back to the specific client(s) responsible.
3. **Repaired** — corrected or neutralized in the global model without
   requiring a full retraining cycle from scratch.

The project studies this problem first in a conventional convolutional
setting (CIFAR-100 + ResNet-18) to establish a correct, leakage-free,
reproducible experimental methodology, and later extends the same
methodology to federated settings, adversarial client behavior, and
foundation-model backbones (e.g., LoRA-adapted ViT/DINOv2/CLIP models).

## Long-Term Vision

FedTrace-FM is intended to support **multiple future research papers**,
not a single one-off experiment. The long-term trajectory is:

1. Establish a correct, reproducible **centralized baseline** (current stage).
2. Build a working **federated learning** simulation (multi-client, FedAvg-style).
3. Introduce **attack simulation** (e.g., label flipping, backdoor triggers,
   model poisoning) against that federated simulation.
4. Design and implement **FedTrace**, an attribution mechanism that traces
   anomalous global-model behavior back to the client(s) responsible.
5. Design a **repair** mechanism that corrects the global model using the
   attribution signal, without full retraining.
6. Extend the entire pipeline to **foundation-model backbones** (LoRA
   fine-tuning of ViT / DINOv2 / CLIP) to test whether attribution and
   repair remain effective at foundation-model scale.
7. Apply the validated pipeline to a **medical imaging dataset**, where
   traceability and robustness to corrupted contributions carry direct
   real-world stakes (data quality, patient safety, regulatory trust).
8. Write up the CIFAR-100/ResNet-18-scale results as a first paper.
9. Extend the validated findings into a longer **journal-length** paper
   that includes the foundation-model and medical-imaging results.

Every module built along the way is designed to be reusable across this
entire arc — for example, the training engine (`src/training/engine.py`)
is intentionally decoupled from any specific experiment configuration so
it can be reused unchanged by future federated clients.

## Current Status

As of the most recent work session:

- The **CIFAR-100 data pipeline** is complete and committed.
- The **centralized CIFAR-100 ResNet-18 baseline** (model, training engine,
  evaluation metrics, checkpointing, entry-point script) is complete and
  committed.
- A **train/validation split fix** has been implemented to eliminate
  test-set leakage from per-epoch model selection. This change exists in
  the working tree but **has not yet been committed to git**.
- A **1-epoch GPU smoke test** (`configs/cifar100_resnet18_smoke.yaml`) has
  been run successfully end-to-end on the local RTX 5060 Laptop GPU,
  confirming the full pipeline (data → stratified split → model → AMP
  training → validation evaluation → checkpointing → final test evaluation
  → reporting artifacts) executes correctly on real hardware. This smoke
  run used only 1 epoch, so its accuracy numbers (see the Research Diary)
  are near-random and are **not** representative of the model's eventual
  trained performance — they only confirm the pipeline is functionally
  correct.
- The full 10-epoch baseline configured in
  `configs/cifar100_resnet18_baseline.yaml` has **not** yet been run to
  completion.
- No federated learning, attack simulation, attribution, repair, or
  foundation-model code exists yet. Their package directories
  (`src/federated/`, `src/attacks/`, `src/attribution/`, `src/repair/`)
  exist as empty placeholders (containing only `__init__.py`).

## Current Git History

The repository has three commits so far:

1. `37f46ff` — **Initial research project setup**
   Created the base folder structure, `requirements.txt`, `README.md`,
   `.gitignore`, and `main.py`.
2. `0543ba6` — **Complete CIFAR-100 data pipeline**
   Added `configs/cifar100_data.yaml`, `src/data/cifar100.py`,
   `src/data/inspect_dataset.py`, `src/utils/config.py`, `src/utils/seed.py`,
   and `tests/test_cifar100.py`.
3. `a9f5992` — **Add centralized CIFAR-100 ResNet-18 baseline**
   Added `configs/cifar100_resnet18_baseline.yaml`,
   `src/models/resnet18_cifar.py`, `src/training/engine.py`,
   `src/evaluation/metrics.py`, `src/utils/checkpoint.py`,
   `src/train_centralized.py`, `tests/test_resnet18_cifar.py`, and
   `tests/test_training_engine.py`.

**Uncommitted working-tree changes at the time of writing:**

- The validation-split leakage fix (`validation_fraction` config field,
  `src/data/cifar100.py::split_train_validation`, the corresponding updates
  to `src/train_centralized.py` and `src/utils/config.py`, and new tests in
  `tests/test_cifar100.py`) is present locally but not yet committed.
- `configs/cifar100_resnet18_smoke.yaml` (an ad hoc 1-epoch smoke-test
  config) is untracked.
- Smoke-test outputs under `results/centralized/cifar100_resnet18_smoke/`
  and `checkpoints/cifar100_resnet18_smoke/` exist locally but are excluded
  from version control by `.gitignore` (results and checkpoints are never
  committed — see **Git Rules** below).

Anyone resuming this project should assume the validation-split fix needs
a commit before further work builds on top of it.

## Repository Structure

```
FedTrace-FM/
├── .venv/                              # Local virtual environment (never committed, never modified by tooling)
├── configs/                            # YAML experiment configurations
│   ├── cifar100_data.yaml              # Data pipeline config (download, batch size, seed, ...)
│   ├── cifar100_resnet18_baseline.yaml # Full 10-epoch centralized baseline config
│   └── cifar100_resnet18_smoke.yaml    # Ad hoc 1-epoch smoke-test config (untracked)
├── data/                               # CIFAR-100 downloads land here (gitignored)
├── notebooks/                          # Reserved for future exploratory notebooks (empty)
├── src/
│   ├── data/
│   │   ├── cifar100.py                 # Transforms, dataset construction, stratified train/val split, DataLoaders
│   │   └── inspect_dataset.py          # CLI: dataset inspection, sample grid, class-count CSV, summary JSON
│   ├── models/
│   │   └── resnet18_cifar.py           # CIFAR-compatible ResNet-18 (3x3 stem, no max-pool), parameter counting
│   ├── training/
│   │   └── engine.py                   # Reusable train/eval epoch functions, optimizer/scheduler/AMP builders
│   ├── evaluation/
│   │   └── metrics.py                  # Top-k accuracy, confusion matrix, macro P/R/F1, balanced accuracy
│   ├── utils/
│   │   ├── config.py                   # Typed YAML config loaders (DataConfig, TrainingConfig) + validation
│   │   ├── seed.py                     # Reproducibility: seeds Python/NumPy/PyTorch CPU+CUDA
│   │   └── checkpoint.py               # Atomic checkpoint save/load/restore
│   ├── federated/                      # Reserved for federated learning (empty — __init__.py only)
│   ├── attacks/                        # Reserved for attack simulation (empty — __init__.py only)
│   ├── attribution/                    # Reserved for FedTrace attribution logic (empty — __init__.py only)
│   ├── repair/                         # Reserved for model repair logic (empty — __init__.py only)
│   └── train_centralized.py            # Centralized training entry point (python -m src.train_centralized)
├── experiments/                        # Reserved for future experiment orchestration scripts (empty)
├── results/                            # All generated outputs (gitignored except .gitkeep)
│   ├── data_inspection/                # Outputs of src/data/inspect_dataset.py
│   └── centralized/<experiment_name>/  # Outputs of src/train_centralized.py, one folder per experiment_name
├── checkpoints/                        # All saved model checkpoints (gitignored except .gitkeep)
│   └── <experiment_name>/best.pt, last.pt
├── logs/                               # Reserved (empty; per-run logs currently live under results/.../run.log instead)
├── tests/                              # pytest test suite
│   ├── test_cifar100.py                # Data pipeline + train/val split tests
│   ├── test_resnet18_cifar.py          # Model architecture tests
│   └── test_training_engine.py         # Training engine, metrics, checkpoint round-trip tests
├── requirements.txt                    # Pinned-by-name (not pinned-by-version) Python dependencies
├── README.md                           # User-facing setup and usage instructions
├── .gitignore
└── main.py                             # Placeholder entry point (prints an initialization message)
```

## Completed Modules

1. **Project scaffolding** — full folder structure, `.gitignore`,
   `requirements.txt`, placeholder `main.py`.
2. **CIFAR-100 data pipeline** (`src/data/cifar100.py`,
   `src/data/inspect_dataset.py`) — automatic download, standard CIFAR-100
   normalization, train/eval transform pipelines, dataset inspection CLI
   (sample grid, per-class counts, summary JSON).
3. **Typed configuration system** (`src/utils/config.py`) — `DataConfig` and
   `TrainingConfig` dataclasses, both loaded from YAML via a shared
   `load_yaml`/`ConfigError` foundation, with explicit range/choice
   validation and clear error messages on missing or invalid fields.
4. **Reproducibility utilities** (`src/utils/seed.py`) — seeds Python
   `random`, NumPy, and PyTorch (CPU + all CUDA devices), with an optional
   full-determinism mode (documented performance trade-off).
5. **CIFAR-compatible ResNet-18** (`src/models/resnet18_cifar.py`) — 3x3
   stride-1 stem (no ImageNet-style 7x7/stride-2 stem, no initial
   max-pool), standard `[2, 2, 2, 2]` BasicBlock stages, no pretrained
   weights, parameter-counting utility.
6. **Reusable training engine** (`src/training/engine.py`) — device
   resolution, optimizer/scheduler/AMP-scaler builders, `train_one_epoch`,
   `evaluate` (with an opt-in prediction-return mode for final reporting).
   Deliberately holds no reference to any experiment-specific config or
   output path, so it can be reused unchanged by future federated clients.
7. **Evaluation metrics** (`src/evaluation/metrics.py`) — top-k accuracy,
   confusion matrix, per-class accuracy, macro precision/recall/F1,
   balanced accuracy (all zero-division-safe).
8. **Checkpointing** (`src/utils/checkpoint.py`) — atomic save (via a
   temp-file-then-`os.replace` pattern), full-state checkpoints (model,
   optimizer, scheduler, AMP scaler, best validation accuracy, full
   resolved config, seed, framework versions).
9. **Centralized training entry point** (`src/train_centralized.py`) —
   wires the above modules together: trains for the configured number of
   epochs, evaluates every epoch, saves best/last checkpoints, reloads the
   best checkpoint, performs one final test-set evaluation, and writes a
   full set of paper-ready output artifacts (history CSV, training curves,
   confusion matrix, per-class accuracy, classification metrics, run log,
   summary JSON).
10. **Train/validation split (test-set leakage fix)** —
    `split_train_validation()` in `src/data/cifar100.py` carves a
    stratified, reproducible validation subset out of the 50,000-image
    CIFAR-100 training set, so per-epoch model selection no longer touches
    the official 10,000-image test set. The test set is now evaluated
    exactly once, after training, using the best checkpoint. **This change
    is implemented but not yet committed to git.**

## Current Module

**Centralized CIFAR-100 ResNet-18 baseline, post-leakage-fix.** The
pipeline has been functionally verified end-to-end via a 1-epoch smoke
test on the local GPU (see the Research Diary for the exact run). The
immediate remaining work in this module is:

- Commit the validation-split fix.
- Run the full 10-epoch baseline (`configs/cifar100_resnet18_baseline.yaml`)
  to obtain a real, reportable centralized baseline accuracy figure.
- Treat that run's `summary.json` as the reference baseline number against
  which later federated/attack/attribution/repair experiments are compared.

## Pending Modules

Listed in intended implementation order (see `PROJECT_ROADMAP.md` for full
phase-by-phase detail):

- **Federated learning simulation** (`src/federated/`) — client
  partitioning (IID and non-IID), local client training loops (reusing
  `src/training/engine.py`), a FedAvg-style (or similar) aggregation
  server, multi-round orchestration.
- **Attack simulation** (`src/attacks/`) — malicious client behaviors such
  as label flipping, backdoor triggers, and model/gradient poisoning,
  injected at configurable rates among the simulated clients.
- **FedTrace attribution** (`src/attribution/`) — the project's central
  novel contribution: a mechanism to trace anomalous global-model behavior
  back to the specific client(s) responsible for it.
- **Repair** (`src/repair/`) — a mechanism to correct or neutralize the
  effect of attributed malicious contributions on the global model without
  a full retraining cycle.
- **Foundation-model integration** — extending the client/model
  abstractions to support LoRA-adapted ViT / DINOv2 / CLIP backbones under
  the same federated + attack + attribution + repair pipeline.
- **Medical dataset extension** — applying the validated pipeline to a
  medical imaging dataset, where data corruption and traceability carry
  direct real-world consequences.
- **Paper writing** and **journal extension** — write-up and submission of
  results, followed by an extended journal version incorporating the
  foundation-model and medical results.

## Research Hypothesis

Malicious or corrupted client contributions in a federated learning system
leave detectable statistical or behavioral signatures in their local model
updates (relative to honest clients' updates), even when those
contributions are individually small enough to evade naive outlier
detection. These signatures can be used to (a) attribute anomalous global
model behavior to the responsible client(s), and (b) drive a targeted
repair of the global model that is cheaper than full retraining. The
project further hypothesizes that this attribution/repair capability
degrades gracefully — rather than catastrophically — as the underlying
model architecture scales up to foundation-model-sized backbones (e.g.,
LoRA-adapted ViT/DINOv2/CLIP), and remains useful in higher-stakes domains
such as medical imaging.

## Novel Contributions (Planned)

1. **FedTrace**: a client-level attribution mechanism for federated
   learning that traces anomalous or degraded global-model behavior back
   to its originating client(s), rather than only detecting that an attack
   occurred.
2. A **repair mechanism** that uses the attribution signal to correct the
   global model without a full retraining cycle, quantified against the
   cost/accuracy trade-off of naive alternatives (e.g., discard-and-retrain,
   client exclusion).
3. Extension of both attribution and repair to **foundation-model-scale**
   federated fine-tuning (LoRA-adapted ViT/DINOv2/CLIP), evaluating whether
   the mechanisms generalize beyond small CNNs.
4. Application of the validated pipeline to a **medical imaging** federated
   setting, where traceability of corrupted contributions has direct
   real-world safety and trust implications.
5. A carefully documented, leakage-free, reproducible experimental
   methodology (typed configs, seeded splits, atomic checkpoints, and a
   held-out test set touched exactly once) intended to make all of the
   above results independently verifiable and reproducible.

## Major Design Decisions

(Full rationale, alternatives considered, and trade-offs are recorded in
`DECISIONS.md`. Summary here for quick reference.)

- **Modular `src/` layout** separating `data/`, `models/`, `training/`,
  `evaluation/`, `utils/`, and reserved future packages
  (`federated/`, `attacks/`, `attribution/`, `repair/`), so each concern
  can be developed, tested, and reused independently.
- **YAML + typed dataclass configs** (`DataConfig`, `TrainingConfig`) with
  explicit validation, rather than passing raw dictionaries or CLI-only
  arguments, so every experiment is fully specified, reproducible, and
  fails loudly on misconfiguration.
- **Reproducibility-first**: every entry point seeds Python/NumPy/PyTorch
  from a single configured seed before any data loading or model
  construction happens.
- **CIFAR-compatible ResNet-18** (custom 3x3 stem, no max-pool) chosen over
  the stock torchvision ImageNet ResNet-18, because the ImageNet stem
  would over-downsample 32x32 CIFAR images before the residual stages ever
  run.
- **Atomic checkpointing** (write-to-temp-then-`os.replace`) so a crash
  mid-save can never leave a corrupted checkpoint file.
- **Held-out validation split, carved from the training set, stratified
  and seeded** — added specifically to eliminate test-set leakage from
  per-epoch model selection. The official test set is now touched exactly
  once, after training.
- **Local-only execution** (no Colab, no Drive mounting) — this project
  will eventually touch federated and medical data scenarios where full
  control over data locality and reproducibility matters; building that
  discipline in from the start avoids having to retrofit it later.

## Development Workflow

Work on this codebase proceeds in small, explicitly scoped steps, each
requested and reviewed independently:

1. Each step's requirements are stated up front (what files may be
   created/modified, what must **not** be done).
2. Before editing, the existing repository state is inspected so new code
   integrates with what already exists rather than duplicating it.
3. Proposed changes are shown/explained before or alongside editing.
4. No command (training run, test run, package install, git operation) is
   executed without the user's explicit request or confirmation.
5. After editing, a summary is provided: every file created or modified,
   what each file does, and the exact commands the user would run to
   verify or use the change themselves.
6. Documentation-only sessions (like the one that produced this file) are
   treated as first-class work, not an afterthought — see
   **Documentation Rules** below.

## Coding Rules

- Use type hints and docstrings throughout; avoid ambiguous or untyped
  public function signatures.
- Use `pathlib.Path` for all filesystem paths; avoid hard-coded path
  strings or manual string concatenation for paths.
- Use the `logging` module for structured runtime output; reserve bare
  `print()` for concise, user-facing CLI summaries (e.g., final metrics).
- Raise informative, specific exceptions (e.g., `ConfigError` with the
  exact missing/invalid field named) rather than letting failures surface
  as opaque stack traces.
- Avoid global mutable state; configuration and state are passed
  explicitly through function arguments and typed dataclasses.
- Avoid duplicate logic — shared behavior (YAML loading, top-k accuracy,
  checkpoint state assembly) lives in one place and is imported, not
  copy-pasted, across modules.
- Keep the training engine (`src/training/engine.py`) free of any
  reference to a specific experiment's configuration object, output
  directory, or logging setup, so it remains reusable by future federated
  clients without modification.
- Keep evaluation logic (`src/evaluation/metrics.py`) independent of the
  training entry point, so it can be reused for federated, attack, and
  attribution evaluation later without depending on
  `src/train_centralized.py`.
- Do not add speculative abstractions, feature flags, or
  backwards-compatibility shims for functionality that does not exist yet;
  build the minimum needed for the current step and let future steps
  extend it.
- Do not add dependencies (TensorBoard integration, Weights & Biases,
  distributed training, etc.) until a step explicitly calls for them.

## Documentation Rules

- `FEDTRACE_FM_MASTER.md` (this file) is the single source of project-level
  truth for intent, history, and status. It must always be understandable
  by a reader (human or AI) with no prior conversation context, and should
  be updated whenever project status, structure, or rules materially change.
- `RESEARCH_DIARY.md` is a chronological lab notebook. Every work session
  gets a dated entry recording objective, completed work, results,
  problems encountered, solutions applied, and the next action — written
  at the time of the session, not reconstructed long after the fact.
- `PROJECT_ROADMAP.md` tracks phase-level progress with checkboxes; it is
  updated whenever a phase's objectives are completed or its scope changes.
- `ARCHITECTURE.md` describes the software architecture (folder structure,
  module responsibilities, and data/training/evaluation/checkpoint flows)
  as the code actually is today, plus the intended shape of not-yet-built
  flows (federated, attribution, foundation-model integration). It is
  updated whenever a new module changes the real architecture.
- `DECISIONS.md` records every significant engineering decision with its
  reasoning, the alternatives considered, why they were rejected, and the
  expected long-term benefit — written at decision time, not
  retroactively rationalized.
- No documentation file uses placeholder or filler text (e.g.,
  "Lorem Ipsum"); every section reflects the actual, current state of the
  project.
- Documentation changes are made in dedicated, code-free sessions when
  explicitly requested as such (no source files, configs, or tests are
  touched during a documentation-only session).

## Testing Rules

- Tests are lightweight and CPU-friendly; no test trains a model to
  completion or requires a GPU to pass.
- Tests that need real CIFAR-100 data reuse the already-downloaded dataset
  (via module-scoped fixtures) rather than re-downloading it per test.
- Model/engine tests use tiny synthetic tensors or tiny synthetic datasets
  (a handful of samples) rather than the full 50,000-image training set,
  so the suite runs in seconds.
- Tests verify real properties, not just "it didn't crash": exact output
  shapes, exact architectural properties (e.g., stem kernel size, absence
  of max-pooling), exact split sizes, reproducibility under a fixed seed,
  divergence under a changed seed, no train/validation index overlap, and
  class-balance preservation after a stratified split.
- `DataLoader` instances created inside tests use `num_workers=0` to avoid
  Windows multiprocessing/spawn complications during test collection.
- New functionality is expected to ship with corresponding tests in the
  same step that introduces it, not deferred to a later step.

## Git Rules

- Commits are created only when explicitly requested by the user; work is
  never auto-committed as a side effect of writing code.
- New commits are preferred over amending existing ones; history is not
  rewritten without an explicit request.
- Destructive operations (`reset --hard`, `push --force`, branch deletion,
  `clean -f`) are never run without explicit confirmation for that specific
  action.
- Commit messages describe *why* a change was made, not just *what* changed.
- `results/`, `checkpoints/`, `data/`, and `logs/` contents are excluded
  from version control (see **Why Results are ignored in Git** in
  `DECISIONS.md`); only `.gitkeep` placeholders are tracked for those
  directories. Generated artifacts are reproducible from configs + code
  and are not treated as source.
- `.venv/` is never committed and never modified by any tooling in this
  project.

## Known Limitations

- No federated learning, attack simulation, attribution, or repair logic
  exists yet — the reserved packages (`src/federated/`, `src/attacks/`,
  `src/attribution/`, `src/repair/`) contain only `__init__.py` files.
- Training is currently single-GPU, single-machine only (one RTX 5060
  Laptop GPU, ~8 GB VRAM). This constrains batch size and model scale for
  future foundation-model work and will need revisiting once LoRA-adapted
  ViT/DINOv2/CLIP backbones are introduced.
- Only CIFAR-100 has been used so far; no other benchmark or the intended
  medical imaging dataset has been integrated yet.
- The only training run executed to date used a single epoch as a
  pipeline smoke test; its accuracy figures are near-random by
  construction and must not be cited as baseline performance. The full
  10-epoch baseline has not yet been run.
- Full determinism (`torch.use_deterministic_algorithms(True)`, enabled by
  default in `src/utils/seed.py`) is not guaranteed for every operation;
  operations lacking a deterministic kernel are logged as a warning rather
  than causing a hard failure, and full determinism can reduce GPU
  throughput.
- The validation-split leakage fix, though implemented and functionally
  verified, is not yet committed to git as of this writing.
- `warmup_epochs` in the training config is validated but only exercised
  by the scheduler when greater than 0; it has not yet been tested with a
  nonzero value in an actual run.

## Future Work

Near-term (extends the current centralized-baseline module):
- Commit the validation-split fix.
- Run and record the full 10-epoch centralized baseline.

Mid-term (opens the federated module):
- Design and implement client data partitioning (IID and non-IID).
- Implement a FedAvg-style (or comparable) aggregation server reusing
  `src/training/engine.py` for local client training.
- Validate the federated simulation against the centralized baseline on
  IID partitions (sanity check: federated ≈ centralized under IID, no
  attacks).

Longer-term:
- Implement attack simulation (label flipping, backdoor, poisoning).
- Design and implement the FedTrace attribution mechanism.
- Design and implement the repair mechanism.
- Extend the pipeline to foundation-model backbones (LoRA-adapted
  ViT/DINOv2/CLIP).
- Extend the pipeline to a medical imaging dataset.
- Write the first paper (CIFAR-100/ResNet-18-scale results).
- Write the extended journal version (foundation-model + medical results).

## Current Next Task

1. Commit the validation-split leakage fix (currently uncommitted in the
   working tree).
2. Run the full 10-epoch centralized baseline using
   `configs/cifar100_resnet18_baseline.yaml`, and record its `summary.json`
   as the reference centralized-baseline result.
3. Begin design of the federated learning module (`src/federated/`):
   client data partitioning strategy (IID vs. non-IID) is the first
   concrete design decision to make.
