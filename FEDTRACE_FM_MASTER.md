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

As of commit `f82ac23` (the latest commit; see **Current Git History**):

- The **CIFAR-100 data pipeline** is complete and committed.
- The **centralized CIFAR-100 ResNet-18 baseline** (model, training engine,
  evaluation metrics, checkpointing, entry-point script), including the
  train/validation split (test-set leakage) fix, is complete and
  committed, and its result is **frozen as Baseline v1.0** (see below).
- The **federated learning infrastructure** is complete and committed:
  `FederatedClient` (Sprint 3.1), `FederatedServer` + the `ClientUpdate`/
  `RoundRecord` contracts (Sprint 3.2), the pluggable `AggregationStrategy`
  interface with `FedAvg` (Sprint 3.3), and the strategy-agnostic
  `FederatedRunner` (Sprint 3.4).
- The **first real federated CIFAR-100 experiment** — IID data partitioning
  across 5 clients with FedAvg, 10 communication rounds (Sprint 4.1,
  `EXP_002`) — has been **run to completion** and its result is now
  **frozen as Federated Baseline v1.0** (see below).
- No Non-IID/Dirichlet/shard partitioning, attack simulation, attribution,
  repair, or foundation-model code exists yet. Their package directories
  (`src/attacks/`, `src/attribution/`, `src/repair/`) exist as empty
  placeholders (containing only `__init__.py`).

## Baseline v1.0 (Frozen) — Centralized

- Dataset: CIFAR-100
- Model: ResNet-18
- Epochs: 10
- Validation Split: 10%
- GPU: NVIDIA RTX 5060 Laptop (8 GB)
- Final Test Loss: 1.3016
- Final Test Top-1: 62.71%
- Final Test Top-5: 88.54%

**Status:**
Frozen centralized baseline for all future comparisons.

## Federated Baseline v1.0 (Frozen) — EXP_002, IID FedAvg

- Dataset: CIFAR-100 (45,000-image training subset, IID-partitioned)
- Model: ResNet-18 (`resnet18_cifar`, 11,220,132 parameters, all trainable)
- Aggregation: FedAvg (sample-count-weighted)
- Clients: 5 (`client_000`–`client_004`, 9,000 samples each), all 5
  selected every round (`clients_per_round=5`, `minimum_updates=5`)
- Communication Rounds: 10 (all completed)
- Local Epochs per Round: 1
- Optimizer: SGD, `learning_rate=0.01`, `momentum=0.9`,
  `weight_decay=0.0005`, no scheduler; optimizer/scheduler state reset at
  the start of every round (see `DECISIONS.md` → *Why Reset Client
  Optimizer State Every Round*)
- Seed: 42
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM)
- Total Experiment Time: 765.6 s (≈12.8 minutes)
- Final Validation Loss: 2.3798, Top-1: 37.54%, Top-5: 69.66% (round 9,
  also the best validation round)
- Final Test Loss: 2.3611
- Final Test Top-1: 37.81%
- Final Test Top-5: 70.27%
- Macro precision: 0.3918, macro recall: 0.3780, macro F1: 0.3600,
  balanced accuracy: 0.3780

**Status:**
Frozen IID FedAvg federated baseline for comparison against future
Non-IID, attack, attribution, and repair experiments.

**Next Module:**
Non-IID (Dirichlet/shard) client data partitioning — Sprint 4.2.

## Current Git History

The repository has ten commits so far:

1. `37f46ff` — **Initial research project setup**
   Base folder structure, `requirements.txt`, `README.md`, `.gitignore`,
   `main.py`.
2. `0543ba6` — **Complete CIFAR-100 data pipeline**
   `configs/cifar100_data.yaml`, `src/data/cifar100.py`,
   `src/data/inspect_dataset.py`, `src/utils/config.py`, `src/utils/seed.py`,
   `tests/test_cifar100.py`.
3. `a9f5992` — **Add centralized CIFAR-100 ResNet-18 baseline**
   `configs/cifar100_resnet18_baseline.yaml`, `src/models/resnet18_cifar.py`,
   `src/training/engine.py`, `src/evaluation/metrics.py`,
   `src/utils/checkpoint.py`, `src/train_centralized.py`, and their tests.
4. `7937bb0` — **Add project documentation and research roadmap**
   `FEDTRACE_FM_MASTER.md`, `RESEARCH_DIARY.md`, `PROJECT_ROADMAP.md`,
   `ARCHITECTURE.md`, `DECISIONS.md`.
5. `0d97501` — **Record frozen centralized baseline experiment**
   The train/validation split (test-set leakage) fix, the full 10-epoch
   baseline run, and its documentation as Baseline v1.0.
6. `0d2b17d` — **Add reusable federated client abstraction**
   `src/federated/client.py`, `src/federated/__init__.py`,
   `tests/test_client.py` (Sprint 3.1).
7. `1b929b6` — **Add federated server infrastructure and update contracts**
   `src/federated/server.py`, `src/federated/contracts.py`
   (`ClientUpdate`, `RoundRecord`), `tests/test_server.py` (Sprint 3.2).
8. `10ed49b` — **Add plugin-based aggregation interface and FedAvg strategy**
   `src/federated/aggregation/` (`base.py`, `fedavg.py`),
   `tests/test_aggregation.py` (Sprint 3.3).
9. `692e8a7` — **Add federated orchestration runner**
   `src/federated/runner.py`, `tests/test_runner.py` (Sprint 3.4).
10. `f82ac23` — **Add IID CIFAR-100 federated experiment pipeline**
    `src/data/federated_partition.py`, `src/experiments/`
    (`run_federated_cifar100.py`), `configs/cifar100_fedavg_iid*.yaml`,
    `FederatedClient.reset_training_state()`, and their tests (Sprint 4.1).
    This commit's config/code produced the Federated Baseline v1.0
    (`EXP_002`) result documented above.

As of this commit, the centralized Baseline v1.0 and the Federated
Baseline v1.0 (`EXP_002`) are both fully reproducible from committed code
and configs alone.

## Repository Structure

```
FedTrace-FM/
├── .venv/                              # Local virtual environment (never committed, never modified by tooling)
├── configs/                            # YAML experiment configurations
│   ├── cifar100_data.yaml              # Data pipeline config (download, batch size, seed, ...)
│   ├── cifar100_resnet18_baseline.yaml # Full 10-epoch centralized baseline config
│   ├── cifar100_resnet18_smoke.yaml    # Ad hoc 1-epoch centralized smoke-test config
│   ├── cifar100_fedavg_iid.yaml        # Full 10-round IID FedAvg federated config (EXP_002)
│   └── cifar100_fedavg_iid_smoke.yaml  # 2-round IID FedAvg smoke-test config (EXP_002_SMOKE)
├── data/                               # CIFAR-100 downloads land here (gitignored)
├── notebooks/                          # Reserved for future exploratory notebooks (empty)
├── src/
│   ├── data/
│   │   ├── cifar100.py                 # Transforms, dataset construction, stratified train/val split, DataLoaders
│   │   ├── federated_partition.py      # Generic IID dataset partitioning (partition_dataset_iid)
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
│   ├── federated/
│   │   ├── client.py                   # FederatedClient: local data/model/optimizer, reset_training_state()
│   │   ├── contracts.py                # ClientUpdate, RoundRecord (immutable, validated)
│   │   ├── server.py                   # FederatedServer: global model, registration, round bookkeeping
│   │   ├── aggregation/                # base.py (AggregationStrategy, AggregationResult), fedavg.py (FedAvg)
│   │   └── runner.py                   # FederatedRunner: orchestrates communication rounds
│   ├── experiments/
│   │   └── run_federated_cifar100.py   # IID FedAvg CIFAR-100 experiment entry point (EXP_002)
│   ├── attacks/                        # Reserved for attack simulation (empty — __init__.py only)
│   ├── attribution/                    # Reserved for FedTrace attribution logic (empty — __init__.py only)
│   ├── repair/                         # Reserved for model repair logic (empty — __init__.py only)
│   └── train_centralized.py            # Centralized training entry point (python -m src.train_centralized)
├── results/                            # All generated outputs (gitignored except .gitkeep)
│   ├── data_inspection/                # Outputs of src/data/inspect_dataset.py
│   ├── centralized/<experiment_name>/  # Outputs of src/train_centralized.py
│   └── federated/<experiment_name>/    # Outputs of src/experiments/run_federated_cifar100.py
├── checkpoints/                        # All saved model checkpoints (gitignored except .gitkeep)
│   ├── <experiment_name>/best.pt, last.pt              # Centralized runs
│   └── federated/<experiment_name>/runner_latest*.pt, final_global_model.pt  # Federated runs
├── logs/                               # Reserved (empty; per-run logs currently live under results/.../run.log instead)
├── tests/                              # pytest test suite
│   ├── test_cifar100.py                # Data pipeline + train/val split tests
│   ├── test_resnet18_cifar.py          # Model architecture tests
│   ├── test_training_engine.py         # Training engine, metrics, checkpoint round-trip tests
│   ├── test_client.py                  # FederatedClient tests (incl. optimizer-reset policy)
│   ├── test_server.py                  # FederatedServer + contracts tests
│   ├── test_aggregation.py             # AggregationStrategy/AggregationResult/FedAvg tests
│   ├── test_runner.py                  # FederatedRunner tests
│   ├── test_federated_partition.py     # partition_dataset_iid tests
│   └── test_federated_experiment.py    # IID federated experiment script tests
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
    exactly once, after training, using the best checkpoint.
11. **Federated client abstraction** (`src/federated/client.py`,
    Sprint 3.1) — `FederatedClient` owns a client's local dataset,
    dataloaders, model, optimizer, scheduler; reuses
    `src/training/engine.py` for local training/evaluation; provides
    `initialize_model()`, `load_global_weights()`, `get_model_weights()`,
    `train_one_round()`, `evaluate()`, local checkpointing, and (Sprint 4.1)
    `reset_training_state()`.
12. **Federated server + contracts** (`src/federated/server.py`,
    `src/federated/contracts.py`, Sprint 3.2) — `FederatedServer` owns the
    global model, registered-client metadata, and round bookkeeping;
    `ClientUpdate`/`RoundRecord` are immutable, validated data contracts.
    The server never aggregates on its own.
13. **Pluggable aggregation + FedAvg** (`src/federated/aggregation/`,
    Sprint 3.3) — `AggregationStrategy` (interface) and `AggregationResult`
    (immutable, validated result); `FedAvg` is the first concrete
    strategy (sample-count-weighted averaging, non-floating tensors
    required identical across clients).
14. **Federated round orchestrator** (`src/federated/runner.py`,
    Sprint 3.4) — `FederatedRunner` connects an existing server, clients,
    and aggregation strategy into communication rounds
    (`FederatedRoundResult` per round), with explicit failure/rollback
    semantics and checkpoint save/resume.
15. **IID partitioning + first federated CIFAR-100 experiment**
    (`src/data/federated_partition.py`, `src/experiments/
    run_federated_cifar100.py`, Sprint 4.1) — `partition_dataset_iid()`
    (generic, seeded, near-equal IID split); the experiment script wires
    the data pipeline, IID partitions, and the full federated stack into
    a runnable CIFAR-100 experiment. Produced the frozen Federated
    Baseline v1.0 (`EXP_002`, see above).

## Current Module

**First IID federated CIFAR-100 experiment (`EXP_002`) — complete and
frozen as Federated Baseline v1.0** (see above). The full federated
learning infrastructure (client, server, contracts, pluggable
aggregation/FedAvg, runner) and the IID partitioner are implemented,
tested, and committed; the 10-round IID FedAvg experiment ran to
completion on the local GPU.

The next module to open is **Non-IID (Dirichlet/shard) client data
partitioning** — see `PROJECT_ROADMAP.md` Phase 3, whose completion
criteria still require non-IID partitioning support (Phase 3 is not yet
fully complete: IID partitioning and a working federated run exist, but
non-IID partitioning does not yet, and the IID federated result is
notably below the centralized baseline at this round budget — see
**Known Limitations**).

## Pending Modules

Listed in intended implementation order (see `PROJECT_ROADMAP.md` for full
phase-by-phase detail):

- **Non-IID client data partitioning** (`src/data/`) — Dirichlet- and/or
  shard-based partitioning, extending `partition_dataset_iid`'s sibling
  functions without modifying it.
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
- **Strategy-agnostic federated stack**: `FederatedServer` and
  `FederatedRunner` have no special-cased knowledge of `FedAvg` or any
  other `AggregationStrategy` — the runner only calls `.name`/`.aggregate()`
  through the abstract interface, so future strategies (FedProx, SCAFFOLD,
  etc.) plug in without modifying the server or runner.
- **Reset client optimizer/scheduler state every communication round**
  (`FederatedClient.reset_training_state()`, called from
  `load_global_weights()`) — the baseline FedAvg client-state policy,
  chosen to avoid stale momentum computed against a superseded global
  model. See `DECISIONS.md` for the full rationale.

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

- No Non-IID/Dirichlet/shard partitioning, attack simulation, attribution,
  or repair logic exists yet — `src/attacks/`, `src/attribution/`, and
  `src/repair/` contain only `__init__.py` files.
- **The IID federated result (`EXP_002`, 37.81% test top-1) is well below
  the centralized Baseline v1.0 (62.71% test top-1) at a comparable
  round/epoch budget** (10 communication rounds × 1 local epoch, vs. 10
  centralized epochs — both see the full 45,000-image training set once
  per round/epoch). This is a known, expected characteristic of FedAvg
  (weight-averaging across independently-trained clients converges more
  slowly than direct SGD over the full data stream), not a bug — but the
  gap has not yet been closed or further investigated (e.g., via more
  rounds, more local epochs, or tuned hyperparameters), and Phase 3's
  "federated ≈ centralized under IID" completion criterion is not yet
  satisfied at this round budget. See `experiments/EXP_002_FedAvg_IID.md`.
- Training is currently single-GPU, single-machine only (one RTX 5060
  Laptop GPU, ~8 GB VRAM), and federated clients are processed strictly
  sequentially within a round (no parallel client execution). This
  constrains experiment turnaround time and will need revisiting for
  larger client counts or foundation-model work.
- Only CIFAR-100 has been used so far; no other benchmark or the intended
  medical imaging dataset has been integrated yet.
- Both frozen baseline results (62.71% centralized / 37.81% federated test
  top-1) reflect a single seed, no hyperparameter tuning, and short
  training budgets; they are first reference points, not upper bounds.
- Full determinism (`torch.use_deterministic_algorithms(True)`, enabled by
  default in `src/utils/seed.py`) is not guaranteed for every operation;
  operations lacking a deterministic kernel are logged as a warning rather
  than causing a hard failure, and full determinism can reduce GPU
  throughput.
- `warmup_epochs` in the centralized training config is validated but only
  exercised by the scheduler when greater than 0; it has not yet been
  tested with a nonzero value in an actual run.
- `FederatedClient.load_global_weights()` unconditionally resets local
  optimizer/scheduler/scaler state every round (see `DECISIONS.md` → *Why
  Reset Client Optimizer State Every Round*); a future strategy wanting
  persistent client-side state (e.g. FedProx/SCAFFOLD-style) would need
  this made configurable.
- `FederatedRunner` has no automatic per-round checkpointing of its own
  (`config.checkpoint_every` is validated but not acted on internally);
  `src/experiments/run_federated_cifar100.py` implements this itself via
  a `round_callback`, which is the documented workaround.

## Future Work

Near-term (Sprint 4.2 — opens Non-IID partitioning):
- Implement Non-IID (Dirichlet and/or shard-based) client data
  partitioning, alongside the existing `partition_dataset_iid`.
- Investigate the IID federated/centralized accuracy gap (more rounds,
  more local epochs per round, and/or hyperparameter tuning) so Phase 3's
  "federated ≈ centralized under IID" criterion can be properly assessed.

Mid-term:
- Run and document a Non-IID federated CIFAR-100 experiment, comparing
  against the IID Federated Baseline v1.0 (`EXP_002`).

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

1. Design and implement Non-IID (Dirichlet and/or shard-based) client data
   partitioning (Sprint 4.2), as a sibling to `partition_dataset_iid` in
   `src/data/federated_partition.py`, without modifying the existing IID
   function's behavior.
2. Decide how to investigate/close the IID federated-vs-centralized
   accuracy gap observed in `EXP_002` (more communication rounds, more
   local epochs per round, or hyperparameter tuning) before treating
   Phase 3's "federated ≈ centralized under IID" criterion as satisfied.
3. Run and document a Non-IID federated CIFAR-100 experiment for
   comparison against the frozen Federated Baseline v1.0 (`EXP_002`).
