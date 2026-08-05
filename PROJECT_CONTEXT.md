# FedTrace-FM — Project Context

> **Purpose of this file:** This is the single entry point for understanding
> the entire FedTrace-FM project. It is written so that any new session —
> including a new ChatGPT or Claude conversation with zero prior history —
> can read this one file and immediately understand what the project is,
> what has been done, what is verified, what is planned, and what to do
> next. It summarizes and cross-references the project's other documentation
> (`FEDTRACE_FM_MASTER.md`, `RESEARCH_DIARY.md`, `PROJECT_ROADMAP.md`,
> `ARCHITECTURE.md`, `DECISIONS.md`, `experiments/`, `Research_Database/`,
> `research_plans/`) rather than replacing them — for full detail on any
> topic below, follow the referenced file.

---

## 1. Project Overview

**FedTrace-FM** is a local, GPU-accelerated research codebase investigating
traceability and attribution in federated learning (FL) systems, with an
eventual extension toward foundation-model (FM) scale architectures. The
name combines **Fed**erated learning + **Trace**ability + **F**oundation
**M**odel.

The overall research objective is to study how malicious, corrupted, or
otherwise anomalous client contributions in a federated learning system can
be **detected**, **attributed** back to the responsible client(s), and
**repaired** in the global model without a full retraining cycle. The
project runs entirely on a single local workstation (one NVIDIA RTX 5060
Laptop GPU, ~8 GB VRAM) with no cloud-notebook dependency, and is built
incrementally, phase by phase, from a centralized image-classification
baseline toward a full federated + attack + attribution + repair pipeline,
and eventually toward foundation-model backbones and a medical-imaging case
study.

## 2. Research Vision

**Long-term scientific goal:** establish, first in a conventional
convolutional setting (CIFAR-100 + ResNet-18), a correct, leakage-free,
reproducible methodology for federated learning experimentation, then use
that methodology to answer the project's central research question —
whether malicious client contributions leave detectable signatures that
can drive attribution and targeted repair — and finally test whether the
answer generalizes to foundation-model backbones and to a higher-stakes,
real-world domain (medical imaging).

**Why this project exists:** federated learning systems are vulnerable to
malicious or corrupted client contributions, but most existing defenses
focus on *detecting* that an attack occurred, not on *tracing it back* to
the responsible client(s) or *repairing* the global model cheaply. This
project's central hypothesis (see `FEDTRACE_FM_MASTER.md` → *Research
Hypothesis*) is that such contributions leave statistical/behavioral
signatures in client updates that can be exploited for both attribution and
targeted repair, and that this capability degrades gracefully rather than
catastrophically as models scale toward foundation-model size and move
into higher-stakes domains.

**Expected publication direction:** a first paper covering the
CIFAR-100/ResNet-18-scale federated + attack + attribution + repair results
(Phases 2–5), followed by an extended journal-length paper incorporating
foundation-model (Phase 6) and medical-imaging (Phase 7) results (Phases
8–9). See Section 13 below and `PROJECT_ROADMAP.md` Phases 8–9.

## 3. Repository Architecture

(Full detail in `ARCHITECTURE.md`.)

- **`configs/`** — YAML experiment configurations (typed, validated via
  `src/utils/config.py`). Every experiment is fully specified by one
  reviewable file here.
- **`data/`** — CIFAR-100 downloads land here automatically; gitignored.
- **`notebooks/`** — reserved for future exploratory notebooks; currently
  empty.
- **`src/data/`** — dataset construction, transforms, the leakage-free
  stratified train/validation split (`split_train_validation`), generic IID
  client partitioning (`federated_partition.py`), and the dataset
  inspection CLI.
- **`src/models/`** — model architectures; currently the CIFAR-compatible
  ResNet-18 (`resnet18_cifar.py`).
- **`src/training/`** — `engine.py`, a reusable, experiment-agnostic
  train/eval engine (device resolution, optimizer/scheduler/AMP builders,
  `train_one_epoch`, `evaluate`) with no dependency on any specific
  experiment's configuration — deliberately reusable by federated clients.
- **`src/evaluation/`** — `metrics.py`, reusable classification metrics
  (top-k accuracy, confusion matrix, macro precision/recall/F1, balanced
  accuracy), independent of the training engine or any entry point.
- **`src/utils/`** — typed config loading (`config.py`), reproducibility
  seeding (`seed.py`), atomic checkpointing (`checkpoint.py`).
- **`src/federated/`** — the federated learning stack: `client.py`
  (`FederatedClient`), `server.py` (`FederatedServer`), `contracts.py`
  (`ClientUpdate`/`RoundRecord`), `aggregation/` (pluggable
  `AggregationStrategy` + `FedAvg`), `runner.py` (`FederatedRunner`,
  multi-round orchestration).
- **`src/experiments/`** — experiment entry-point scripts;
  `run_federated_cifar100.py` is the IID FedAvg CIFAR-100 experiment used
  for `EXP_002`/`EXP_003`.
- **`src/attacks/`, `src/attribution/`, `src/repair/`** — reserved
  packages for future phases (Phase 4/5); currently contain only
  `__init__.py`.
- **`src/train_centralized.py`** — the centralized training entry point
  (`EXP_001`).
- **`results/`** — all generated run outputs (history, curves, confusion
  matrices, summary JSON, run logs); gitignored except `.gitkeep`, since
  every result is reproducible from its config + code + seed.
- **`checkpoints/`** — saved model checkpoints; gitignored except
  `.gitkeep`, for the same reproducibility reason.
- **`logs/`** — reserved; per-run logs currently live under
  `results/.../run.log` instead.
- **`tests/`** — the pytest suite; CPU-friendly, no test requires a GPU or
  a full training run.
- **`experiments/`** — one narrative Markdown document per completed
  experiment (`EXP_001_...md`, `EXP_002_...md`, `EXP_003_...md`).
- **`research_plans/`** — pre-registered scientific protocols written
  *before* an experiment is run (e.g., `EXP_004_Convergence_Study_Plan.md`).
- **`Research_Database/`** — the permanent, cross-experiment tabular
  record: `FedTrace_Master_Results.xlsx` (main human-readable workbook,
  manually maintained, never programmatically overwritten),
  `FedTrace_Master_Results/*.csv` (8 structured source-table CSVs),
  `Experiment_Index.csv`, `Figure_Index.csv`, `Table_Index.csv`.
- **Root-level documentation** — `FEDTRACE_FM_MASTER.md` (master memory),
  `RESEARCH_DIARY.md` (chronological lab notebook), `PROJECT_ROADMAP.md`
  (phase-by-phase checklist), `ARCHITECTURE.md` (software architecture),
  `DECISIONS.md` (engineering decision log), this file
  (`PROJECT_CONTEXT.md`).

## 4. Current Development Status

**Completed modules** (full list and detail in `FEDTRACE_FM_MASTER.md` →
*Completed Modules*, items 1–16):
- Project scaffolding, CIFAR-100 data pipeline, typed configuration system,
  reproducibility (seeding) utilities.
- CIFAR-compatible ResNet-18, reusable training engine, evaluation metrics,
  atomic checkpointing.
- Centralized training entry point, with the train/validation split
  (test-set-leakage) fix.
- Full federated learning infrastructure: `FederatedClient` (Sprint 3.1),
  `FederatedServer` + `ClientUpdate`/`RoundRecord` contracts (Sprint 3.2),
  pluggable `AggregationStrategy` + `FedAvg` (Sprint 3.3), the
  strategy-agnostic `FederatedRunner` (Sprint 3.4).
- IID client partitioning + the first real federated CIFAR-100 experiment
  (Sprint 4.1, `EXP_002`).
- A controlled round-budget follow-up (Sprint 4.2, `EXP_003`,
  `communication_rounds: 10 → 20`).

**Under development / immediate next task:** analysis of `EXP_003`'s
round-by-round history to decide whether `EXP_004` should extend to a
larger round budget — a planning document for this
(`research_plans/EXP_004_Convergence_Study_Plan.md`, currently specifying
50 rounds) exists, but `EXP_004` itself has **not** been executed and no
config for it has been created yet.

**Not started:**
- Non-IID (Dirichlet/shard-based) client data partitioning (Phase 3,
  remaining criterion).
- Attack simulation (`src/attacks/`, Phase 4).
- FedTrace attribution (`src/attribution/`, Phase 5).
- Repair mechanism (`src/repair/`, Phase 5).
- Foundation-model integration (Phase 6).
- Medical-dataset extension (Phase 7).
- Paper writing and journal extension (Phases 8–9).

## 5. Completed Experiments

### EXP_001 — Centralized ResNet-18 Baseline
- **Objective:** establish a correct, leakage-free centralized CIFAR-100
  baseline as the reference point for all later federated/attack/
  attribution/repair comparisons.
- **Configuration:** CIFAR-compatible ResNet-18, 10 epochs, SGD, cosine
  scheduler, 10% validation split (stratified, seed 42), GPU (RTX 5060
  Laptop, 8 GB VRAM).
- **Major findings:** frozen as **Baseline v1.0** — final test loss 1.3016,
  test top-1 62.71%, test top-5 88.54%. Serves as the reference figure that
  all federated experiments are compared against.
- **Git commit:** `0d97501` — *Record frozen centralized baseline
  experiment*.
- Full detail: `experiments/EXP_001_Centralized_Baseline.md`.

### EXP_002 — FedAvg IID, 10 Communication Rounds
- **Objective:** run the first real federated CIFAR-100 experiment —
  IID data partitioning across 5 simulated clients with FedAvg — and
  establish a federated baseline for comparison against the centralized
  result.
- **Configuration:** same CIFAR-compatible ResNet-18; 5 clients (IID
  partition, seed 42, 9,000 samples each), all 5 selected every round;
  10 communication rounds, 1 local epoch per round; SGD (lr 0.01, momentum
  0.9, weight decay 0.0005, no scheduler); optimizer/scheduler state reset
  every round; seed 42; GPU.
- **Major findings:** frozen as **Federated Baseline v1.0** — final test
  loss 2.3611, test top-1 37.81%, test top-5 70.27% — trailing the
  centralized Baseline v1.0 by 24.90 percentage points top-1 at a matched
  10-unit data-exposure budget. Round-history analysis found no plateau
  through the final round, indicating the model was undertrained rather
  than converged, and recommended extending the communication-round budget
  as the single highest-value next experiment.
- **Git commit:** `f82ac23` — *Add IID CIFAR-100 federated experiment
  pipeline*.
- Full detail: `experiments/EXP_002_FedAvg_IID.md`.

### EXP_003 — FedAvg IID, 20 Communication Rounds
- **Objective:** test, via a single controlled variable change
  (`communication_rounds: 10 → 20`, everything else identical to
  `EXP_002`), whether `EXP_002`'s result was undertrained rather than
  converged.
- **Configuration:** identical to `EXP_002` in every respect (model,
  optimizer, learning rate, scheduler, batch size, local epochs,
  aggregation, client count, partition, seed) except the round count.
- **Major findings:** final test loss 1.6335, test top-1 54.69%, test
  top-5 83.75% — a +16.88 percentage-point top-1 improvement over
  `EXP_002`, narrowing the gap to the centralized Baseline v1.0 from 24.90
  to **8.02 percentage points**. Round-history analysis of the final five
  rounds (15–19) showed continued, non-negligible, non-monotonically-
  shrinking gains with no plateau signature, confirming the round-budget
  hypothesis raised by the `EXP_002` analysis.
- **Git commit:** `26a9bb2` — *Add 20-round IID FedAvg experiment
  configuration*.
- Full detail: `experiments/EXP_003_FedAvg_IID_20Rounds.md`.

## 6. Current Best Results

| Metric | EXP_001 (Centralized, 10 epochs) | EXP_002 (FedAvg IID, 10 rounds) | EXP_003 (FedAvg IID, 20 rounds) |
|---|---|---|---|
| Test Loss | 1.3016 | 2.3611 | 1.6335 |
| Test Top-1 | 62.71% | 37.81% | 54.69% |
| Test Top-5 | 88.54% | 70.27% | 83.75% |
| Gap to Centralized (top-1) | — | 24.90 pp | 8.02 pp |
| Improvement vs. prior FedAvg run (top-1) | — | — | +16.88 pp |

`EXP_003` is the current best federated result; `EXP_001` remains the
current best (and only) centralized result and the project's overall
reference figure.

## 7. Scientific Findings So Far

Only verified conclusions, drawn from the completed analyses of `EXP_002`
and `EXP_003` (see `experiments/EXP_002_FedAvg_IID.md` and
`experiments/EXP_003_FedAvg_IID_20Rounds.md`):

- IID FedAvg federated training substantially trails a centralized
  baseline trained for a matched data-exposure budget when the
  communication-round budget is small (`EXP_002`: 24.90pp gap at 10
  rounds).
- This gap is, to a large extent, an artifact of an insufficient
  communication-round budget rather than an inherent limitation of FedAvg
  under IID partitioning: doubling the round budget alone (`EXP_002` →
  `EXP_003`, no other variable changed) closed roughly two-thirds of the
  gap (24.90pp → 8.02pp).
- `EXP_003`'s final five rounds (15–19) still show consistent, non-trivial
  gains in validation loss, top-1, and top-5 accuracy, with no clear
  plateau — average marginal gain per round shrank from ≈3.96pp/round
  (rounds 0–9) to ≈1.67pp/round (rounds 10–19) to ≈1.17pp/round (rounds
  15–19), which is consistent with **emerging but not yet exhausted**
  diminishing returns.
- Whether the remaining 8.02pp gap fully closes with a larger round
  budget, or plateaus above the centralized result, is not yet determined
  — this is the open question `EXP_004` is designed to answer (see
  `research_plans/EXP_004_Convergence_Study_Plan.md`).
- The federated validation set (shared, centralized) has been observed to
  closely track federated test-set performance in both `EXP_002` and
  `EXP_003`, supporting its use as a reliable proxy during round-by-round
  monitoring.

## 8. Major Design Decisions

(Full rationale for each in `DECISIONS.md`.)

- **Python 3.12 + PyTorch** (CUDA 12.8 build) as the sole language/
  framework, for ecosystem compatibility across the whole planned arc
  (federated, attack, attribution, foundation-model work).
- **YAML + typed dataclass configs** (`DataConfig`, `TrainingConfig`),
  validated via `ConfigError`, rather than raw dicts or Python config
  modules — every experiment is fully specified and reproducible.
- **CIFAR-100** as the first benchmark (fast local iteration on ~8 GB
  VRAM, harder than CIFAR-10, standard in the literature) rather than
  ImageNet.
- **CIFAR-compatible ResNet-18** (custom 3x3 stride-1 stem, no max-pool)
  rather than a deeper CNN or a from-scratch ViT — appropriately sized for
  rapid methodology iteration before the more expensive foundation-model
  phase.
- **Stratified, seeded validation split** carved from the training set,
  restricting the official test set to a single end-of-training evaluation
  — introduced specifically to eliminate test-set leakage from per-epoch
  (or per-round) model selection.
- **Modular `src/` layout**, with the training engine and evaluation
  metrics deliberately kept free of any experiment-specific dependency, so
  the same `train_one_epoch`/`evaluate` functions are reused unchanged by
  federated clients.
- **Strategy-agnostic federated stack**: `FederatedServer` and
  `FederatedRunner` have no special-cased knowledge of `FedAvg` — future
  aggregation strategies (FedProx, SCAFFOLD, etc.) plug in via the
  `AggregationStrategy` interface without modifying the server or runner.
- **Reset client optimizer/scheduler state every communication round**
  (`FederatedClient.reset_training_state()`) — the baseline FedAvg
  client-state policy, chosen because no stateful strategy (e.g.
  FedProx/SCAFFOLD) exists yet to give a persistent-state policy a
  well-defined meaning.
- **Documentation-first**: `FEDTRACE_FM_MASTER.md`, `RESEARCH_DIARY.md`,
  `PROJECT_ROADMAP.md`, `ARCHITECTURE.md`, `DECISIONS.md` are maintained
  continuously, not written retroactively, so any future session (human or
  AI) can resume work without reverse-engineering intent from code alone.
- **Results/checkpoints excluded from git** (`.gitignore`, `.gitkeep`
  placeholders only) — regenerable from versioned configs + code + seed,
  so the repository stays small and fast across every future phase.

## 9. Current Research Workflow

```
Research Question
      ↓
Protocol            (research_plans/EXP_XXX_*.md — objective, hypothesis,
      ↓              controlled variable, success/failure criteria, decision
      ↓              tree, agreed BEFORE the experiment is run)
Implementation      (configs/, and only the minimal src/ code needed —
      ↓              existing reusable modules are extended, not duplicated)
Experiment          (the experiment entry-point script is run; results land
      ↓              under results/ and checkpoints/, gitignored)
Analysis            (data-only review of round_history.csv / summary.json /
      ↓              final_test_metrics.json / run.log; no invented numbers,
      ↓              exactly one recommended next experiment)
Documentation       (FEDTRACE_FM_MASTER.md, RESEARCH_DIARY.md,
      ↓              PROJECT_ROADMAP.md, experiments/EXP_XXX_*.md updated
      ↓              using only verified outputs)
Database            (Research_Database/ CSVs — one row per experiment/
      ↓              figure/table — appended, never overwritten; the
      ↓              XLSX workbook is manually maintained separately)
Git Commit          (created only when explicitly requested; new commits
                     preferred over amending history)
```

This is the exact workflow `EXP_001` → `EXP_002` → `EXP_003` followed, and
the one `EXP_004` is currently mid-way through (protocol written; not yet
implemented or run).

## 10. Current Project Roadmap

(Full phase detail, objectives, and completion criteria in
`PROJECT_ROADMAP.md`.)

- **Phase 0 — Environment:** ✅ Complete.
- **Phase 1 — Dataset:** ✅ Complete.
- **Phase 2 — Centralized Learning:** 🔶 In Progress (baseline run and
  documented; two checklist items remain administrative — validation-split
  fix commit and baseline-run recording — largely reflected already in
  `EXP_001`'s frozen status).
- **Phase 3 — Federated Learning:** 🔶 In Progress. IID partitioning and
  two full IID federated runs (`EXP_002`, `EXP_003`) complete; Non-IID
  partitioning not yet started; the "federated ≈ centralized" completion
  criterion is closer but not yet met (8.02pp gap remaining).
- **Phase 4 — Attack Simulation:** ⬜ Not Started.
- **Phase 5 — FedTrace Attribution:** ⬜ Not Started.
- **Phase 6 — Foundation Models:** ⬜ Not Started.
- **Phase 7 — Medical Dataset:** ⬜ Not Started.
- **Phase 8 — Paper Writing:** ⬜ Not Started.
- **Phase 9 — Journal Extension:** ⬜ Not Started.

## 11. Immediate Next Task

**EXP_004 — FedAvg IID, 50 communication rounds.**

A pre-registered protocol for this experiment already exists at
`research_plans/EXP_004_Convergence_Study_Plan.md`. Per that plan, **only
`communication_rounds` changes, from 20 (`EXP_003`) to 50** — every other
configuration value (model, optimizer, learning rate, scheduler, batch
size, local epochs, aggregation strategy, client count, partition, seed,
evaluation protocol, checkpoint/result format) remains identical to
`EXP_003`. This experiment has **not** been executed yet and its config
file has **not** been created yet — both remain to be done as explicitly
separate steps, following the plan's PI Approval Checklist.

## 12. Future Planned Studies

- **EXP_004 and beyond (round-budget convergence):** determine whether
  extending the communication-round budget continues to close the gap to
  the centralized baseline, slows significantly, or plateaus (see
  `research_plans/EXP_004_Convergence_Study_Plan.md` → Decision Tree for
  the exact single next action under each outcome).
- **Non-IID (Dirichlet/shard-based) client partitioning** and a Non-IID
  federated CIFAR-100 experiment, compared against the IID results.
- **Attack simulation** — label flipping, backdoor triggers, model/
  gradient poisoning, injected at configurable rates.
- **FedTrace attribution** — the project's central novel contribution:
  tracing anomalous global-model behavior back to specific client(s).
- **Repair mechanism** — correcting the global model using the
  attribution signal, without a full retraining cycle.
- **Foundation-model integration** — LoRA-adapted ViT/DINOv2/CLIP
  backbones under the same federated + attack + attribution + repair
  pipeline.
- **Medical-imaging extension** — applying the validated pipeline to a
  medical dataset, where traceability carries direct real-world stakes.

## 13. Paper Vision

Per `FEDTRACE_FM_MASTER.md` → *Long-Term Vision* and `PROJECT_ROADMAP.md`
Phases 8–9, the intended paper structure is:

1. **First paper** (Phase 8): covers the CIFAR-100/ResNet-18-scale results
   from Phases 2–5 — the centralized baseline, the federated (IID and
   Non-IID) simulation, attack simulation, the FedTrace attribution
   mechanism, and the repair mechanism. Structure anticipated: motivation,
   related work, methodology (federated simulation, attack simulation,
   FedTrace attribution, repair), experimental results, and limitations —
   with all reported results reproducible from committed configs and code.
2. **Extended journal version** (Phase 9): incorporates the
   foundation-model (Phase 6) and medical-imaging (Phase 7) results into a
   longer manuscript, adding any additional experiments/ablations needed
   to satisfy journal-level review (e.g., broader baselines).

Both papers are intended to rest on the same leakage-free, reproducible
experimental methodology (typed configs, seeded splits, atomic
checkpoints, held-out test set touched exactly once) established since
Phase 2, so results reported in either paper are independently
verifiable and reproducible from this repository.

## 14. Quick Start for Future Sessions

A new conversation (ChatGPT, Claude, or otherwise) with **zero prior
context** should:

1. Read this file (`PROJECT_CONTEXT.md`) first for the full picture:
   what the project is, what is done, what is verified, and what is next.
2. For deeper detail on any section above, follow its cross-reference:
   - Full project memory / current status → `FEDTRACE_FM_MASTER.md`
   - Chronological session-by-session history → `RESEARCH_DIARY.md`
   - Phase-by-phase completion tracking → `PROJECT_ROADMAP.md`
   - Software architecture and data/training/checkpoint flows →
     `ARCHITECTURE.md`
   - Why each engineering choice was made → `DECISIONS.md`
   - Narrative writeup of any single experiment → `experiments/EXP_XXX_*.md`
   - Pre-registered protocol for a not-yet-run experiment →
     `research_plans/EXP_XXX_*.md`
   - Cross-experiment tabular results → `Research_Database/` (CSVs are the
     structured source of truth; `FedTrace_Master_Results.xlsx` is the
     human-readable workbook, manually maintained, never programmatically
     overwritten)
3. Treat **Section 11 (Immediate Next Task)** above as the current
   priority unless the user states otherwise: `EXP_004` (FedAvg IID, 50
   communication rounds, single-variable change from `EXP_003`) is
   pre-registered but not yet implemented or run.
4. Follow the project's standing rules (see `FEDTRACE_FM_MASTER.md` →
   *Coding Rules*, *Documentation Rules*, *Testing Rules*, *Git Rules*):
   commits only on explicit request; no invented metrics; documentation-
   only sessions touch no code/config/test; new modules reuse existing
   ones rather than duplicating logic; every experiment is fully specified
   by its config and reproducible from committed code.
5. Never modify `Research_Database/FedTrace_Master_Results.xlsx`
   programmatically — it is a manually maintained workbook; the CSVs
   under `Research_Database/FedTrace_Master_Results/` are its structured
   source tables.
