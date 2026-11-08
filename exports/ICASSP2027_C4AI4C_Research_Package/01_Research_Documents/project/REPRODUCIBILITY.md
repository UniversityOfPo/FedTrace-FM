# Reproducibility Guide

This document lists exact commands, verified against files actually present in this repository.
It does not describe anything that was not run, and every config path named here exists in
`configs/` at the time of writing.

## 1. Environment

- Python 3.12
- PyTorch (CUDA build recommended; CPU fallback works but is much slower for training runs)
- Dependencies: `pip install -r requirements.txt`
- All reported experiments were run locally on a single NVIDIA GPU (no distributed/multi-GPU
  training, no cloud/Colab dependency).

## 2. Dataset Preparation

CIFAR-100 is downloaded automatically into `./data` on first use (`src/data/cifar100.py`,
configured by `configs/cifar100_data.yaml`). No manual step is required.

## 3. Frozen Detector Parameters

All FedTrace results in this repository use, without exception:

```
T_ref = 20   (src/analysis/fedtrace_detector.py: DEFAULT_T_REF)
tau   = 3.0  (DEFAULT_THRESHOLD)
P     = 3    (DEFAULT_PERSISTENCE)
components = ("m", "t_res")   (S = max(z^m, z^t_res))
```

These are never changed to produce a favorable result in any experiment, including E4's own
sensitivity sweep (E4 evaluates alternative τ/P values *offline*, against already-existing traces,
without altering the reported default).

## 4. Lightweight Tests (no GPU, no training)

```bash
python -m pytest tests/test_attacks_traceability.py tests/test_attack_active_windows.py tests/test_analysis_fedtrace_detector.py tests/test_aggregation.py tests/test_multikrum_aggregation.py -v
```

These exercise the attack simulation, the frozen detector's calibration/persistence/attribution
logic, and both aggregation strategies (FedAvg, Multi-Krum) against small synthetic tensors — no
dataset, GPU, or federated training run is required.

## 5. Primary Validation (GPU training required)

```bash
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_fedavg_dirichlet_a01_seed2026_50rounds.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed2026.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed2026.yaml
```

Outputs: `results/federated/<experiment_name>/` (`summary.json`, `round_history.csv`,
`client_round_trace.csv`, plots, `run.log`); checkpoints under
`checkpoints/federated/<experiment_name>/`.

## 6. Generalization (identity/onset changes; GPU training required)

```bash
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_fedavg_dirichlet_a01_direction_attack_client004_round20_seed123.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round30_seed123.yaml
```

## 7. Multi-Krum Comparison (GPU training required)

```bash
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_multikrum_dirichlet_a01_seed123_50rounds.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_multikrum_dirichlet_a01_seed2026_50rounds.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_multikrum_dirichlet_a01_gaussian_attack_client002_round20_seed123.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_multikrum_dirichlet_a01_gaussian_attack_client002_round20_seed2026.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_multikrum_dirichlet_a01_direction_attack_client002_round20_seed123.yaml
python -m src.experiments.run_federated_cifar100 --config configs/cifar100_multikrum_dirichlet_a01_direction_attack_client002_round20_seed2026.yaml
```

Per-round Krum scores/selection are written to `multikrum_diagnostics.csv` in each run's output
directory. `f=1` is fixed internally (`src/experiments/run_federated_cifar100.py::_MULTIKRUM_F`);
it is deliberately not a YAML field, so it cannot be swept per run.

## 8. E1 — Federation Scaling (10 clients; GPU training required)

```bash
python -m src.experiments.run_federated_cifar100 --config configs/ndss_e1_10clients_clean_seed123.yaml
python -m src.experiments.run_federated_cifar100 --config configs/ndss_e1_10clients_gaussian_client002_round20_seed123.yaml
python -m src.experiments.run_federated_cifar100 --config configs/ndss_e1_10clients_directional_client002_round20_seed123.yaml
```

Then, offline (no GPU required):

```bash
python scripts/ndss_e1_postprocess.py
```

Reads the three runs' `client_round_trace.csv`/`summary.json` under
`results_ndss/E1_10clients/` and writes `e1_summary.csv`, `e1_summary.json`,
`e1_roundwise_scores.csv`, `e1_client_partitions.csv`, `e1_run_log.txt`,
`e1_10client_scaling.png` alongside them.

## 9. E2 — Attack Severity (GPU training required)

```bash
python -m src.experiments.run_federated_cifar100 --config configs/ndss_e2_gaussian_rho050_client002_round20_seed123.yaml
python -m src.experiments.run_federated_cifar100 --config configs/ndss_e2_directional_c050_client002_round20_seed123.yaml
```

Then, offline:

```bash
python scripts/ndss_e2_postprocess.py
```

Reads the two new E2 runs **plus** the already-existing primary-validation traces (ρ=1.0/c=0.0,
under `results/federated/`, read-only, never rerun) to build a severity comparison. Writes
`e2_summary.json`, `e2_run_log.txt`, `e2_severity_comparison.csv`, `e2_severity_comparison.png`
under `results_ndss/E2_attack_severity/`.

## 10. E3 — Intermittent Attacks (GPU training required)

```bash
python -m src.experiments.run_federated_cifar100 --config configs/ndss_e3_intermittent_gaussian_client002_seed123.yaml
python -m src.experiments.run_federated_cifar100 --config configs/ndss_e3_intermittent_directional_client002_seed123.yaml
```

Both configs set `attack_active_windows: [[20,24],[30,34],[40,44]]` — the malicious client is
active only in those three windows and behaves as an honest client everywhere else (implemented in
`src/attacks/traceability.py::AttackConfig.is_active`; a strict no-op when this field is absent).

Then, offline:

```bash
python scripts/ndss_e3_postprocess.py
```

Writes `e3_summary.csv`, `e3_summary.json`, `e3_burst_diagnostics.csv`, `e3_gap_diagnostics.csv`,
`e3_run_log.txt`, `e3_intermittent_scores.png` under `results_ndss/E3_intermittent/`.

## 11. E4 — Detector-Parameter Sensitivity (offline only — no GPU, no new training)

```bash
python scripts/ndss_e4_sensitivity.py
```

Reads the four already-existing primary-validation traces (`results/federated/...`, read-only)
and re-evaluates the frozen `compute_all_scores_v3`/`evaluate_detector_v3` across
τ∈{2.5,3.0,3.5}×P∈{2,3,4} (`T_ref=20` and the score formula are never changed). Writes
`e4_all_evaluations.csv`, `e4_summary.csv`, `e4_summary.json`, `e4_run_log.txt`,
`e4_sensitivity_heatmap.png` under `results_ndss/E4_detector_sensitivity/`. Includes a built-in
guard that refuses to write any output if the (τ=3.0, P=3) cell fails to reproduce the
already-published primary-validation numbers exactly.

## 12. Which Steps Require GPU Training vs. Are Offline Analysis

| Step | GPU training? |
|---|---|
| Primary validation, generalization, Multi-Krum, E1, E2, E3 | **Yes** |
| Component ablation (magnitude-only/temporal-residual-only/peer-only) | No — computed from already-existing primary traces |
| E4 detector-parameter sensitivity | No — offline sweep over already-existing traces |
| `scripts/ndss_e{1,2,3}_postprocess.py` | No — reads already-completed `client_round_trace.csv` files |
| `scripts/generate_paper_figures.py`, `scripts/generate_paper_tables.py` | No — reads already-existing results only |
| Lightweight test suite (§4 above) | No |

## 13. Expected Output Directories

- `results/federated/<name>/` — primary/generalization/Multi-Krum runs.
- `results_ndss/E1_10clients/`, `E2_attack_severity/`, `E3_intermittent/`,
  `E4_detector_sensitivity/` — NDSS robustness/sensitivity experiments and their compact analysis
  outputs.
- `checkpoints/`, `checkpoints_ndss/` — model checkpoints; **excluded from version control** (see
  `.gitignore`) due to size (several GB across all experiments). Every reported forensic result is
  reconstructible from the committed trace/summary files without the checkpoints themselves.
- `paper_assets/{tables,figures}/` — generated, paper-ready tables and figures with a per-value
  provenance CSV (`FIGURE_DATA_PROVENANCE.csv`, `TABLE_DATA_PROVENANCE.csv`).
