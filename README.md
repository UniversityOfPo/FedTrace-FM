# Detection Is Not Attribution

**A Matched Reliability Audit of Federated-Learning Forensics under Poisoning and Benign Drift**

Al Amin, Rayhana Taskin, Abena Primo, Azubike Okpalaeze, Rohan Thompson, Kamrul Hasan

> **Status:** manuscript prepared for the ICASSP 2027 C4AI4C satellite workshop (not yet accepted or
> published). This repository is the research artifact. The earlier NDSS-oriented README is kept at
> [docs/README_NDSS_ARCHIVE.md](docs/README_NDSS_ARCHIVE.md).

## What the paper does

Forensic monitors for federated learning are usually judged by whether they raise an alarm. An operator
must also name the right client, date the intervention and avoid accusing honest clients. The paper
introduces a **matched reliability audit**: clean, BadNets, relabel-only and honest-drift arms share
bit-identical histories before the intervention round, and five outcomes are reported separately
(attack effectiveness, detection, first attribution, onset error / delay, honest accusations).

The audit is applied, without changing code or thresholds, to the frozen monitor **FedTrace v3**
(τ = 3, P = 3, T_ref = 20) and three sequential norm detectors (z-score persistence h = 1.75,
CUSUM h = 11.75, Page–Hinkley h = 6.5) on five-client CIFAR-100 (ResNet-18, Dirichlet α = 0.1,
50 rounds, client_002 intervened from round 20).

Main findings (seed level, never pooled):

- Norm z-score persistence and CUSUM name the BadNets attacker on all four seeds but accuse honest
  clients in every drift run and in one clean run.
- FedTrace v3 raises no clean alarm but misses both preregistered attackers (one with 63.3 % ASR).
- A conditional peer-median displacement, seen on one development seed, lets the intervened client
  push honest clients over threshold.

## Where things are

| Path | Content |
|---|---|
| `exports/FedTrace_V9/`, `exports/FedTrace_V9.zip` | LaTeX source of the locked manuscript (V9) |
| `exports/FedTrace_V10/` | working copy for the protocol-37 extension results |
| `exports/ICASSP2027_C4AI4C_Research_Package/` | full research package (paper, assets, scripts, evidence) |
| `manuscript_c4ai4c27/` | figure/table generation scripts and earlier drafts |
| `paper_plan/` | preregistrations, protocols, claim–evidence ledger (`04_claim_evidence_ledger.md`), decision log, seed registry |
| `src/rev/` | audit code: runner, attacks, benign drift, online FedTrace v3, baselines, contamination analysis, outcome tables |
| `src/analysis/fedtrace_detector.py` | frozen FedTrace v3 detector |
| `configs/rev/` | run configurations of every arm and seed |
| `results_rev/` | compact per-run outputs (client-round traces, metrics) and analysis outputs (`results_rev/pilot/`) |
| `results/`, `results_ndss/` | archived model-update perturbation runs (Table 4 of the paper) |
| `tests/` | unit and regression tests |

Not in the repository: raw update logs (≈ 13–23 GiB per run, `logs_rev/`), model checkpoints and the
CIFAR-100 data. Every run and analysis can be regenerated from the configurations and the commands in
`paper_plan/26_gpu_confirmation_protocol.md` (Amendment 1 §C) and `paper_plan/37_extension_protocol.md`.

## Runs

| Block | Seeds | Arms | Status |
|---|---|---|---|
| Development | 123, 2026 | clean, BadNets, relabel-only, honest drift | complete, analysed |
| Confirmation (protocol 26) | W1 1178015705, W2 1463065807 | clean, BadNets, honest drift | complete, analysed |
| Extension (protocol 37) | W3 1966286922, W4 1465380568, W5 431235112; K = 10 at 123 / 2026 | clean, BadNets, honest drift | partially executed (8 of 15 runs), **not analysed**; W5_drift interrupted and excluded |

## Reproduce

```
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python -m src.rev.preflight --runs W1_clean W1_badnets W1_drift
python -m src.rev.run_rev_experiment --config configs/rev/W1_clean.rev.yaml
python -m src.rev.unified_outcomes --confirmation
python -m pytest tests
```

## Integrity notes

- Detectors and thresholds were frozen before the confirmation seeds were drawn; nothing was tuned on them.
- `X4_drift_d2026` records `execution_clean = False` (an uncommitted documentation edit was present when it
  started; executed code and configs match the recorded commit; ledger L-68).
- Archived Multi-Krum, K = 10, reduced-severity and intermittent runs executed code not contained in their
  recorded commit (ledger L-17).
