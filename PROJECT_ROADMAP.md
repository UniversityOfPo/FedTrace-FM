# FedTrace-FM — Project Roadmap

> Phase-by-phase roadmap for the FedTrace-FM research platform. Checkboxes
> are updated as work completes; a phase is only checked off once its
> completion criteria are fully met, not merely started. This roadmap is
> reviewed and updated whenever a phase's scope changes.

---

## Phase 0 — Environment

**Status:** ✅ Complete

**Objectives:**
- Establish a fully local Windows development environment (VS Code,
  Python 3.12, `.venv`) with no cloud-notebook dependency.
- Confirm PyTorch + CUDA are correctly installed and can use the local GPU.
- Create the initial project scaffold with a modular `src/` package layout.

**Expected Outputs:**
- Working `.venv` with `requirements.txt` dependencies installable.
- Folder structure: `configs/`, `data/`, `notebooks/`, `src/` (with
  `data/`, `models/`, `federated/`, `attacks/`, `attribution/`, `repair/`,
  `utils/` subpackages), `experiments/`, `results/`, `checkpoints/`,
  `logs/`, `tests/`.
- `.gitignore`, `README.md`, `main.py`, `requirements.txt` in place.

**Completion Criteria:**
- [x] Project scaffold created with `__init__.py` in every Python package.
- [x] `.venv` present and not modified/deleted by tooling.
- [x] `requirements.txt` lists all dependencies needed so far.
- [x] Git repository initialized with an initial commit.
- [x] CUDA availability confirmed on real hardware (RTX 5060 Laptop GPU,
      ~8 GB VRAM) — verified via the Phase 2 smoke test's GPU log line.

---

## Phase 1 — Dataset

**Status:** ✅ Complete

**Objectives:**
- Build a reproducible CIFAR-100 data pipeline: typed configuration,
  seeding, automatic download, dataset construction, and a CLI inspection
  script that verifies dataset integrity and produces reviewable artifacts.

**Expected Outputs:**
- `configs/cifar100_data.yaml`.
- `src/utils/config.py` (`DataConfig` + validation), `src/utils/seed.py`.
- `src/data/cifar100.py` (transforms, dataset/dataloader construction).
- `src/data/inspect_dataset.py` (CLI inspection script).
- `results/data_inspection/` artifacts: sample image grid, per-class count
  CSV, summary JSON.
- `tests/test_cifar100.py`.

**Completion Criteria:**
- [x] CIFAR-100 downloads automatically into `./data` on first run.
- [x] All 100 classes verified present with valid labels.
- [x] Train/test sizes verified as 50,000 / 10,000.
- [x] Inspection artifacts generated and reviewed.
- [x] Tests passing for config loading, class count, dataset sizes, label
      range, and DataLoader batch shape.
- [x] Committed to git (`0543ba6` — *Complete CIFAR-100 data pipeline*).

---

## Phase 2 — Centralized Learning

**Status:** 🔶 In Progress

**Objectives:**
- Implement a reproducible centralized CIFAR-100 baseline using a
  CIFAR-compatible ResNet-18.
- Provide a reusable training engine, evaluation metrics, and atomic
  checkpointing that later phases (federated learning in particular) can
  reuse without modification.
- Eliminate test-set leakage from model selection by introducing a proper
  train/validation/test split.
- Produce a real, reportable baseline accuracy figure.

**Expected Outputs:**
- `configs/cifar100_resnet18_baseline.yaml`.
- `src/models/resnet18_cifar.py`, `src/training/engine.py`,
  `src/evaluation/metrics.py`, `src/utils/checkpoint.py`,
  `src/train_centralized.py`.
- `tests/test_resnet18_cifar.py`, `tests/test_training_engine.py`.
- `results/centralized/cifar100_resnet18/` — full 10-epoch run artifacts
  (history, curves, confusion matrix, per-class accuracy, classification
  metrics, summary JSON, run log).
- `checkpoints/cifar100_resnet18/best.pt`, `last.pt`.

**Completion Criteria:**
- [x] CIFAR-compatible ResNet-18 implemented (3x3 stride-1 stem, no
      max-pool, no pretrained weights) with parameter counting.
- [x] Reusable training engine implemented (device resolution, optimizer/
      scheduler/AMP builders, train/eval epoch functions), independent of
      any specific experiment configuration.
- [x] Evaluation metrics implemented (top-k accuracy, confusion matrix,
      per-class accuracy, macro precision/recall/F1, balanced accuracy).
- [x] Atomic checkpointing implemented with full training-state payload.
- [x] Centralized training entry point implemented, producing all
      specified output artifacts.
- [x] Train/validation/test split implemented: stratified, reproducible,
      seeded validation split carved from the training set; per-epoch
      model selection uses validation accuracy; the official test set is
      evaluated exactly once, after training.
- [x] Pipeline functionally verified end-to-end via a 1-epoch GPU smoke
      test.
- [ ] Validation-split fix committed to git.
- [ ] Full 10-epoch baseline run completed and its `summary.json` recorded
      as the reference centralized-baseline result.

---

## Phase 3 — Federated Learning

**Status:** 🔶 In Progress

**Objectives:**
- Simulate a federated learning setting on top of the existing centralized
  training engine: partition CIFAR-100 across multiple simulated clients,
  run local training rounds, and aggregate client updates into a global
  model.
- Support both IID and non-IID client data partitioning.
- Validate that the federated simulation reproduces centralized-baseline
  performance under IID partitioning with no attacks present (a sanity
  check that the simulation itself is correct before introducing attacks).

**Expected Outputs:**
- `src/federated/` populated with: client data partitioning utilities, a
  local client training loop (reusing `src/training/engine.py`), an
  aggregation server (e.g., FedAvg-style), and multi-round orchestration.
- A federated training entry point (e.g., `src/train_federated.py`) and
  corresponding YAML configuration(s).
- Federated-simulation results comparable in format to the centralized
  baseline's output artifacts.
- Tests covering partitioning correctness (e.g., class coverage per
  client, partition size, reproducibility under a seed) and aggregation
  correctness (e.g., a known-simple aggregation case produces the expected
  result).

**Completion Criteria:**
- [x] `FederatedClient`/`FederatedServer`/`ClientUpdate`/`RoundRecord`
      implemented and tested (Sprints 3.1–3.2).
- [x] Pluggable `AggregationStrategy` interface implemented and tested,
      with `FedAvg` as the first concrete strategy (Sprint 3.3).
- [x] Strategy-agnostic `FederatedRunner` (multi-round orchestration)
      implemented and tested (Sprint 3.4).
- [x] Local client training loop implemented, reusing the existing
      training engine without modification.
- [x] IID client partitioning implemented and tested
      (`partition_dataset_iid`, Sprint 4.1).
- [ ] Non-IID (Dirichlet and/or shard-based) client partitioning
      implemented and tested (not yet started; Sprints 4.2/4.3 instead
      investigated the round-budget question raised by `EXP_002` — see
      below).
- [x] Multi-round federated training runs end-to-end on local hardware —
      `EXP_002`: 5 IID clients, FedAvg, 10/10 rounds completed, GPU
      (RTX 5060 Laptop). Frozen as Federated Baseline v1.0. `EXP_003`
      (Sprint 4.2): identical configuration, 20/20 rounds completed,
      confirming `EXP_002` was undertrained. `EXP_004` (Sprint 4.3):
      identical configuration, 50/50 rounds completed.
- [x] Federated (IID, no-attack) accuracy validated as comparable to the
      Phase 2 centralized baseline — **met, and exceeded, at 50 rounds**:
      `EXP_002` (10 rounds) trailed the centralized Baseline v1.0 (62.71%
      test top-1) by 24.90 points (37.81%); `EXP_003` (20 rounds,
      otherwise identical) narrowed this to 8.02 points (54.69%); `EXP_004`
      (50 rounds, otherwise identical) **exceeds** it by 5.81 points
      (68.52%) — the first federated result in this project to surpass
      its centralized reference. This confirms the original gap was
      largely an undertraining artifact, though why the federated result
      now exceeds the centralized one is not yet understood — a formal
      convergence classification and root-cause investigation are the
      immediate next task. See `experiments/EXP_004_FedAvg_IID_50Rounds.md`
      → *Observations*.
- [x] Results committed to git (`283902c`); federated module documented in
      `FEDTRACE_FM_MASTER.md`, `RESEARCH_DIARY.md`,
      `experiments/EXP_002_FedAvg_IID.md`,
      `experiments/EXP_003_FedAvg_IID_20Rounds.md`, and
      `experiments/EXP_004_FedAvg_IID_50Rounds.md`.

---

## Phase 4 — Attack Simulation

**Status:** ⬜ Not Started

**Objectives:**
- Introduce configurable malicious client behaviors into the federated
  simulation, so later phases have adversarial conditions to attribute and
  repair against.

**Expected Outputs:**
- `src/attacks/` populated with attack implementations: label flipping,
  backdoor trigger injection, and model/gradient poisoning, each
  configurable (attack type, fraction of malicious clients, attack
  intensity/trigger parameters).
- Attack-aware federated training configuration(s).
- Measured impact of each attack type on global-model accuracy relative to
  the clean federated baseline (Phase 3).
- Tests verifying each attack implementation actually alters client
  updates/labels as intended, in isolation from the full training loop.

**Completion Criteria:**
- [ ] At least label flipping, backdoor, and model-poisoning attacks
      implemented.
- [ ] Attack intensity and malicious-client fraction configurable via YAML.
- [ ] Attack impact quantified against the Phase 3 clean federated
      baseline.
- [ ] Tests covering each attack's isolated behavior.
- [ ] Results and attack configurations committed to git.

---

## Phase 5 — FedTrace Attribution

**Status:** ⬜ Not Started

**Objectives:**
- Design and implement the project's central novel contribution: a
  mechanism that attributes anomalous global-model behavior back to the
  specific client(s) responsible, under the attack conditions established
  in Phase 4.

**Expected Outputs:**
- `src/attribution/` populated with the FedTrace attribution algorithm(s).
- An evaluation protocol quantifying attribution quality (e.g., precision/
  recall of identifying truly malicious clients, across attack types and
  malicious-client fractions).
- A comparison against at least one naive baseline attribution approach
  (e.g., simple update-norm outlier detection), to demonstrate FedTrace's
  added value.
- Tests covering the attribution algorithm's behavior on small, controlled
  synthetic scenarios with known ground-truth malicious clients.

**Completion Criteria:**
- [ ] FedTrace attribution algorithm implemented.
- [ ] Attribution quality quantified across the attack types from Phase 4.
- [ ] Comparison against a naive baseline attribution method completed.
- [ ] Tests covering controlled synthetic attribution scenarios.
- [ ] Method and results documented in preparation for Phase 8 (paper
      writing).

---

## Phase 6 — Foundation Models

**Status:** ⬜ Not Started

**Objectives:**
- Extend the client model, training engine, and attribution/repair
  mechanisms to support LoRA-adapted foundation-model backbones (ViT,
  DINOv2, CLIP), and evaluate whether the findings from Phases 3–5
  generalize beyond a small CNN.

**Expected Outputs:**
- Model factory support for at least one LoRA-adapted ViT/DINOv2/CLIP
  backbone, respecting the local ~8 GB VRAM constraint (e.g., via LoRA
  fine-tuning rather than full fine-tuning).
- Federated + attack + attribution + repair results reproduced on the
  foundation-model backbone, comparable to the CNN-scale results.
- Documentation of any methodology changes required for the
  foundation-model setting (e.g., different attribution signal
  characteristics under LoRA fine-tuning).

**Completion Criteria:**
- [ ] At least one LoRA-adapted foundation-model backbone integrated and
      trainable locally within the available VRAM budget.
- [ ] Federated simulation validated on the new backbone (IID, no attack).
- [ ] Attack simulation reproduced on the new backbone.
- [ ] FedTrace attribution and repair evaluated on the new backbone.
- [ ] Comparative analysis (CNN vs. foundation model) documented.

---

## Phase 7 — Medical Dataset

**Status:** ⬜ Not Started

**Objectives:**
- Apply the validated federated + attack + attribution + repair pipeline
  to a medical imaging dataset, where corrupted or malicious client
  contributions carry direct real-world safety and trust implications.

**Expected Outputs:**
- A selected, appropriately licensed medical imaging dataset integrated
  via a new data module (parallel in structure to `src/data/cifar100.py`).
- Federated + attack + attribution + repair results on this dataset.
- Explicit discussion of domain-specific considerations (e.g., class
  imbalance typical of medical data, higher cost of false negatives,
  any relevant data governance/licensing constraints).

**Completion Criteria:**
- [ ] Medical dataset selected, licensed appropriately, and integrated.
- [ ] Federated baseline validated on the medical dataset (IID, no attack).
- [ ] Attack simulation and FedTrace attribution/repair evaluated on the
      medical dataset.
- [ ] Domain-specific findings and limitations documented.

---

## Phase 8 — Paper Writing

**Status:** ⬜ Not Started

**Objectives:**
- Write up the CIFAR-100/ResNet-18-scale results (Phases 2–5) as a
  submittable research paper.

**Expected Outputs:**
- A complete manuscript covering: motivation, related work, methodology
  (federated simulation, attack simulation, FedTrace attribution, repair),
  experimental results, and limitations.
- Reproducibility artifacts referenced in the paper (configs, seeds,
  scripts) matching what is actually committed in this repository.

**Completion Criteria:**
- [ ] Manuscript draft complete.
- [ ] All reported results reproducible from committed configs and code.
- [ ] Manuscript reviewed and submitted to a target venue.

---

## Phase 9 — Journal Extension

**Status:** ⬜ Not Started

**Objectives:**
- Extend the Phase 8 paper into a longer journal-length manuscript
  incorporating the foundation-model (Phase 6) and medical-imaging
  (Phase 7) results.

**Expected Outputs:**
- An extended manuscript incorporating additional experiments, deeper
  analysis, and the foundation-model/medical results.
- Any additional experiments needed specifically to satisfy journal-level
  reviewer expectations (e.g., broader ablations, additional baselines).

**Completion Criteria:**
- [ ] Extended manuscript draft complete, incorporating Phases 6 and 7.
- [ ] Additional experiments/ablations requested during review completed.
- [ ] Manuscript submitted to a target journal.
