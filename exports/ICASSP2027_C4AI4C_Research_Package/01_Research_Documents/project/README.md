# FedTrace

**Forensic Client Attribution and Attack-Onset Localization in Heterogeneous Federated Learning**

FedTrace analyzes longitudinal client-update traces in federated learning to identify a client
exhibiting persistent malicious behavior and estimate when that behavior began. It targets a
question distinct from Byzantine-robust aggregation: not "which updates should be excluded this
round," but "which client was responsible, and when did it start."

> **Paper status:** This repository is the research artifact for a manuscript currently in
> preparation / under peer review. It is not an accepted or published paper. Please do not cite
> it as an accepted NDSS paper.

---

## 1. Overview

Federated learning lets clients train collaboratively without sharing raw data, but a malicious
client can manipulate its local update while remaining embedded in an otherwise legitimate
population. Robust aggregation (e.g. Multi-Krum) decides which updates influence the model *this
round*; it does not tell you *which client* misbehaved or *when* the behavior started. FedTrace
targets that complementary forensic question directly from the server's own observed update
history, without using ground-truth attacker identity or onset as an input.

## 2. Key Contributions

- A **forensic traceability formulation**: joint client attribution and attack-onset localization
  from longitudinal update histories, explicit separated from per-round robust aggregation.
- **FedTrace**, a frozen detector combining two client-calibrated evidence branches — magnitude
  abnormality and temporal-residual abnormality — with a persistence rule that converts sustained
  evidence into an attribution decision.
- **Primary validation** across two attack families and two held-out seeds, with **zero detector
  retuning**, plus targeted robustness experiments (E1–E4) covering identity/onset changes,
  federation scale, attack severity, intermittent behavior, and detector-parameter sensitivity.
- A matched **Multi-Krum comparison** that empirically distinguishes native robust-aggregation
  rejection from forensic attribution, including an explicit author-constructed rejection-based
  attribution proxy used only as a diagnostic.

## 3. FedTrace Pipeline

1. **Magnitude evidence** `z^m_{k,r}` — two-sided standardized deviation of a client's
   submitted-update L2 norm from its own calibrated reference history.
2. **Temporal-residual evidence** `z^t_res_{k,r}` — one-sided standardized deviation of a
   client's consecutive-round update-cosine residual (after removing a same-round cross-client
   reference) from its own calibrated reference history.
3. **Combined score** `S_{k,r} = max(z^m_{k,r}, z^t_res_{k,r})` — an evidence-coverage rule, not a
   fusion-improves-detection claim; either branch can carry the decision.
4. **Persistence rule** — a client is confirmed only after `P` consecutive rounds with
   `S_{k,r} > τ`; an undefined score breaks an in-progress streak.
5. **Attribution and onset** — the earliest-confirmed client is attributed; the onset estimate is
   the first round of its qualifying persistent run (`r̂_onset = r_conf − P + 1`).

Implementation: `src/analysis/fedtrace_detector.py` (frozen; `DEFAULT_T_REF=20`,
`DEFAULT_THRESHOLD=3.0`, `DEFAULT_PERSISTENCE=3`). Attack simulation:
`src/attacks/traceability.py`.

## 4. Repository Structure

```
src/
  analysis/fedtrace_detector.py   # frozen FedTrace detector (v1/v2/v3; v3 is the reported one)
  attacks/traceability.py         # Gaussian + norm-preserving directional attacks, trace schema
  federated/                      # client/server/runner/aggregation (FedAvg, Multi-Krum)
  experiments/run_federated_cifar100.py   # federated training entry point
  models/resnet18_cifar.py        # CIFAR-adapted ResNet-18 (no pretrained weights)
configs/                          # one YAML per experiment (primary, E1-E4, Multi-Krum, ...)
scripts/                          # paper-asset generation + NDSS E1-E4 post-processing
results/, results_ndss/           # per-experiment raw traces + compact analysis outputs
checkpoints/, checkpoints_ndss/   # model checkpoints (NOT committed; see .gitignore)
paper_assets/                     # generated figures/tables used in the manuscript
tests/                            # unit tests (no GPU/training required)
docs/REPRODUCIBILITY.md           # environment, commands, expected outputs
docs/EXPERIMENT_MAP.md            # repository experiment -> paper table/figure mapping
analysis/                         # internal evidence-audit and synthesis documents
```

## 5. Experimental Setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-100 |
| Model | ResNet-18 (CIFAR variant, no pretrained weights) |
| Primary clients | 5 (full participation) |
| Rounds | 50 |
| Local epochs | 1 |
| Optimizer | SGD, lr=0.01, batch size 32 |
| Partition | Dirichlet, α=0.1 |
| Primary validation seeds | 123, 2026 |
| Primary attack onset | round 20 |
| Primary attacker | `client_002` |
| `T_ref` / `τ` / `P` | 20 / 3.0 / 3 (frozen) |
| Multi-Krum `f` / `m=n−f` | 1 / 4 |

Full field-by-field source-provenance is in `paper_assets/tables/main/table01_experimental_setup.csv`.

## 6. Attacks

- **Gaussian update-noise** (`gaussian_update_noise`): adds isotropic Gaussian noise to the
  honest update, deterministically rescaled so `‖noise‖₂ = ρ · ‖honest update‖₂` exactly.
- **Norm-preserving directional** (`norm_preserving_directional`): replaces the honest update
  with a direction rotated to a target cosine `c` against the honest direction, exactly preserving
  the honest update's norm (verified in every active-attack trace row via
  `delta_attack_norm_ratio`/`realized_cosine_to_honest`).

Both support an optional `attack_active_windows` field (added for E3) for intermittent, rather
than continuous-from-onset, malicious behavior — a strict no-op when omitted (see
`src/attacks/traceability.py::AttackConfig.is_active`).

## 7. Experiments

| Experiment | Question | Configs |
|---|---|---|
| Primary validation | Attribution/onset under Gaussian & directional attacks, 2 seeds | `configs/cifar100_fedavg_dirichlet_a01_*_attack_client002_round20_seed{123,2026}.yaml` |
| Component ablation | Magnitude-only / temporal-residual-only / peer-only vs. full FedTrace | (offline, computed from primary traces) |
| Identity/onset generalization | Attacker → `client_004`; onset → round 30 | `configs/*_client004_round20_seed123.yaml`, `configs/*_client002_round30_seed123.yaml` |
| **E1** — federation scaling | 10 clients (clean + both attacks) | `configs/ndss_e1_10clients_*.yaml` |
| **E2** — attack severity | Gaussian ρ=0.5, directional c=0.5 | `configs/ndss_e2_*.yaml` |
| **E3** — intermittent attacks | 3 five-round windows (20–24, 30–34, 40–44) | `configs/ndss_e3_intermittent_*.yaml` |
| **E4** — detector sensitivity | Offline τ×P sweep on primary traces | `scripts/ndss_e4_sensitivity.py` (no new training) |
| Multi-Krum comparison | Native rejection vs. FedTrace attribution | `configs/cifar100_multikrum_*.yaml` |

## 8. Main Results

All numbers below are read directly from the committed result/summary files, not retyped from
memory — see `docs/EXPERIMENT_MAP.md` for the exact source path of each.

- **Primary validation** (`paper_assets/tables/main/table02_fedtrace_validation.csv`): FedTrace
  correctly attributes `client_002` in all 4 runs (2 attack families × 2 seeds), onset error = 0,
  confirmation delay = 2 (the minimum possible under `P=3`), zero honest persistent false
  positives.
- **Signal complementarity**: magnitude evidence dominates the Gaussian attack (30/30 post-onset
  rounds, one seed; mixed 10/30 vs 20/30 the other); temporal-residual evidence dominates the
  directional attack (30/30 both seeds) — the two branches respond to different attack geometries,
  and the combined score never beats the better individual component's timing in any of the four
  runs (`paper_assets/tables/main/table03_signal_ablation.csv`).
- **Generalization** (`paper_assets/tables/main/table05_generalization.csv`): identity shift to
  `client_004` and onset shift to round 30 both preserve exact attribution/onset with zero
  retuning.
- **Multi-Krum comparison** (`paper_assets/tables/main/table04_multikrum_baseline.csv`):
  Multi-Krum rejects the Gaussian attacker in 30/30 post-onset rounds (both seeds) but the
  norm-preserving directional attacker in 0/30. A retrospective, author-constructed
  rejection-based attribution proxy misattributes a persistently-rejected honest client in all
  four attacked runs. **Multi-Krum's native output is a per-round selection decision, not
  attribution — this proxy is not a Multi-Krum capability.**

## 9. E1–E4 Robustness / Sensitivity Evaluation

Verified from `results_ndss/{E1_10clients,E2_attack_severity,E3_intermittent,E4_detector_sensitivity}/`:

- **E1 (10 clients):** clean control confirms no client; Gaussian and directional attacks both
  reproduce the primary outcome exactly (onset error 0, delay 2, zero honest FP) at double the
  federation size.
- **E2 (attack severity):** directional c=0.5 preserves exact onset/delay; Gaussian ρ=0.5
  preserves correct **attribution** but onset is localized far later (estimated onset round 41,
  confirmation round 43, vs. true onset round 20) — attribution is more robust to reduced severity
  than timely onset localization, at this tested severity value.
- **E3 (intermittent attacks, windows 20–24/30–34/40–44):** both attack families are correctly
  attributed with onset localized at round 20 and confirmation at round 22 — the same as the
  continuous-attack case — with zero honest persistent false positives.
- **E4 (offline τ×P sensitivity):** at the frozen default (τ=3.0, P=3), all 4 primary runs give
  correct attribution, exact onset, and zero honest FP. Across the swept grid (τ∈{2.5,3.0,3.5},
  P∈{2,3,4}), **P=2 is a real failure mode at every tested τ**: mean onset error is large and,
  for 2 of 4 runs, an honest client is confirmed *before* the true attacker, flipping attribution
  itself (not merely a delay). P≥3 gives correct attribution and exact onset at every tested τ.
  This is descriptive, offline evidence — it is not used to retune the reported default.

E1–E4 are each targeted, single-axis stress tests, not exhaustive robustness sweeps; see
`docs/REPRODUCIBILITY.md` and the paper's Limitations section for exact scope.

## 10. Multi-Krum Comparison

Multi-Krum (`f=1`, `m=n−f=4`) is evaluated as an **external Byzantine-robust aggregation
baseline**, not a competing attribution method. `f=1` is set to match the known single-attacker
ground truth in every test — an assumption FedTrace itself never requires. Its native output
(`multikrum_diagnostics.csv`: per-round Krum score, rank, selected/rejected) is reported directly;
any client-level attribution derived from repeated rejection is explicitly an author-constructed
diagnostic proxy, never presented as a Multi-Krum capability. See `src/federated/aggregation/multikrum.py`.

## 11. Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12; PyTorch with CUDA support recommended (all reported experiments used a single local
GPU) but not required — CPU fallback exists, just much slower for the training-based experiments.

## 12. Dataset Preparation

CIFAR-100 downloads automatically into `./data` on first run via `src/data/cifar100.py`
(configured by `configs/cifar100_data.yaml`); no manual download step is required.

## 13. Quick Test

Run the fast, GPU-free unit tests (no training, no dataset download required beyond what's already
cached, typically well under a minute):

```bash
python -m pytest tests/test_attacks_traceability.py tests/test_attack_active_windows.py tests/test_analysis_fedtrace_detector.py -v
```

## 14. Reproducing Experiments

Every experiment is a single YAML config passed to the same entry point:

```bash
python -m src.experiments.run_federated_cifar100 --config configs/<name>.yaml
```

See `docs/REPRODUCIBILITY.md` for the exact command for every primary/E1–E4/Multi-Krum config,
which steps require GPU training vs. are offline post-processing, and expected output locations.

## 15. Expected Outputs

Each training run writes to its config's `output_dir`/`checkpoint_dir` (per-run `summary.json`,
`round_history.csv`, `client_round_trace.csv`, plots, and model checkpoints). Offline
post-processing scripts (`scripts/ndss_e{1,2,3,4}_*.py`) read already-completed traces and write
compact CSV/JSON/PNG summaries alongside them — they never retrain a model.

## 16. Reproducibility Notes

- The frozen detector (`T_ref=20`, `τ=3.0`, `P=3`) is never retuned after seeing any result in any
  experiment in this repository, including E4's own sensitivity sweep.
- Ground-truth attacker identity and onset are used only for post-hoc evaluation columns, never as
  inputs to score or persistence computation (`src/analysis/fedtrace_detector.py`).
- All attack RNG draws are deterministically seeded from `(experiment seed, round_id, client_id)`
  (and attack type, for the directional attack), never the global RNG state.
- Model checkpoints are intentionally excluded from version control (see §10 of `.gitignore`) —
  every reported number is reconstructible from the committed `client_round_trace.csv`/
  `summary.json` files via the frozen detector, without needing the checkpoint weights themselves.

## 17. Current Scope and Limitations

Evaluated conditions: CIFAR-100/ResNet-18, Dirichlet α=0.1, up to 10 clients, full participation,
two attack families, one attacker identity beyond the primary (`client_004`), one onset shift
(round 30), one reduced severity per family, one intermittent schedule, and a local τ/P sensitivity
grid. **Not evaluated:** partial participation, larger federations beyond 10 clients, additional
datasets/architectures, adaptive or colluding adversaries, contaminated reference windows, or other
non-IID regimes. Results are controlled evidence under these conditions, not a universal robustness
guarantee.

## 18. Future Extension: Federated Foundation Models

Earlier planning documents in this repository (`FEDTRACE_FM_MASTER.md`, `PROJECT_ROADMAP.md`)
describe a longer-term, **not-yet-implemented** research direction extending this pipeline to
LoRA-adapted foundation-model backbones (ViT/DINOv2/CLIP) and a medical-imaging case study. **No
foundation-model or pretrained-model code exists in this repository**, and the current FedTrace
paper makes no foundation-model claim. `src/attribution/` and `src/repair/` remain empty
placeholders reserved for that future work. Treat any "FM" references in older planning documents
as future scope, not completed contributions.

## 19. Citation / Paper Status

This repository accompanies a manuscript currently in preparation / under peer review:

> *FedTrace: Forensic Client Attribution and Attack-Onset Localization in Heterogeneous Federated
> Learning.* Anonymous Authors. Manuscript under review.

A formal citation entry will be added once the paper's venue and status are finalized. Please do
not describe this work as an accepted or published paper.
