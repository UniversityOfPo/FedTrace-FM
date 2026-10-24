# 02 — Experiment Matrix

Status: living document, created 2026-10-24. **No row has results yet unless marked ARCHIVED.**
Ordering = scientific value per GPU-hour. Runtimes are *measured* only where stated (K=5 FedAvg 1.33 h/run,
n=21; K=10 FedAvg 2.31 h/run, n=3; Multi-Krum K=5 1.27 h/run); everything else is an estimate and labelled.
Logged-run disk: 14.7 GiB (5 clients × 50 rounds + attacker pre-transform + 51 globals, fp32), computed from the
measured state size (44,919,088 B).

Common configuration unless stated: CIFAR-100, ResNet-18 (CIFAR variant), Dirichlet α = 0.1, K = 5 full
participation, R = 50, local epochs 1, SGD lr 0.01, batch 32, AMP on, FedAvg, attacker `client_002`,
onset t₀ = 20, frozen FedTrace v3 (τ = 3, P = 3, T_ref = 20) evaluated **online** (`src/rev/online.py`).

Seeds: development D = {123, 2026}; validation V = 5 fresh seeds sealed in the pre-registration before any
validation run (proposed derivation: first five 31-bit integers of SHA-256("FedTrace-rev-validation-2026") —
PI to approve; 42/123/2026 excluded). Matched control = same seed, same commit, same environment, attack the only
difference; enforced by the bit-identical pre-onset prefix check.

## MUST

| ID | Hypothesis / question | Reviewer objection | Configuration | Baselines | Metrics | Seeds | Paired controls | GPU-h (est.) | Output artifacts | Target | If negative | Prio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | Entry point runs end-to-end on GPU with logging; log verifies; replay passes; comparison script works on real GPU outputs | Reproducibility | S0a = NDSS runner on `configs/rev/S0a_smoke_ndss.yaml` (EXP_016 with R = 3, attack from round 1); S0b = rev entry point on the same config with logging (`configs/rev/S0b_smoke_logged.rev.yaml`) | — | `compare_runs --preset S0b-vs-S0a`: no NUMERICAL/VALUE/STRUCTURAL/MISSING finding | 123 | S0a | ≈ 0.2 (est.) | `results_rev/S0{a,b}_*`, `logs_rev/S0b_*` (≈ 0.9 GiB) | none (engineering gate, not evidence) | Fix code before R0/R1 | MUST |
| B-R0 | EXP_016 reproduces with the unmodified NDSS runner at HEAD | Reproducibility | `configs/rev/R0_exp016_repro.yaml` | — | Level A comparison vs archive (trace, history w/o timing, partition, summary, final model tensors) | 123 | archive EXP_016 | 1.33 (meas.) | `results_rev/R0_exp016_repro/` | App. A1 | Level B only with diagnosed cause; else BLOCKED | MUST |
| B-R1 | Logging and the rev entry point do not change training (R1 ≡ R0) | Reproducibility | `configs/rev/R1_exp016_logged.rev.yaml` | — | bit-identity R1 vs R0; log verify; replay aggregation; replayed trace = online trace | 123 | B-R0 | 1.33 (+ logging overhead, unmeasured) | `results_rev/R1_*`, `logs_rev/R1_*` (14.7 GiB) | App. A1 | BLOCKED (logging interferes) | MUST |
| X1 | H2: clean FedAvg K=5 has no persistent honest alarm | #368A FP/strong assumptions; specificity never measured in FedAvg K=5 | attack none | all detectors (CPU) | runs with honest alarm k/n; honest alarms /1,000 client-rounds; z_m drift profile | D + V (7) | is the control for X2/X3/X4 and (if B passes Level A) for archived EXP_016–019 | 9.3 (meas. rate) | `results_rev/X1_*`, logs for D + 1 V seed | T3, F5 | Report FP rate; investigate drift (z_m); restrict claims | MUST |
| X2 | H1: unconstrained BadNets backdoor is effective and attributed | #368A unrealistic/non-adaptive attacks; #368B non-security setup | `rev_attack: backdoor`, target 0, poison 0.5, 3×3 white patch, onset 20 (PROVISIONAL until dev runs) | all detectors | attack success (final; per-round from logged globals), paired Δ clean top-1, detection/attribution k/n [CP], alarm, onset error, censored delay | D (tuning) + V | X1 same seed | 9.3 | `results_rev/X2_*` incl. `rev_attack_records.json`; logs | T3, T4, F2, F3 | If success < threshold on D: adjust poison rate/trigger on D only, re-seal; if still ineffective: report, do not claim poisoning detection | MUST |
| X3 | H3: detector-aware norm-constrained backdoor trades success for stealth | #368A adaptive attackers | `constrained_backdoor`, κ = 2.5 (PROVISIONAL), same backdoor params as X2 | all detectors | success, constraint satisfaction rate, detection, delay, which signal fired | D + V | X1, X2 same seed | ~9.5 (est.: one extra norm computation/round) | `rev_attack_records.json` (constraint records) | T5, F3 | Either outcome reported; "evasion" claimed only if success ≥ threshold and undetected | MUST |
| C1 | Baselines at matched clean false-alarm level | #368A weak baseline; RQ5 | traces of B/X1–X3 (+ archived) | z-score+P, CUSUM, Page–Hinkley (same features), naive peer (uses the order-dependent peer field, L-16: report its decision margins), Multi-Krum proxy (archived) | as X2 + delay at matched FPR; thresholds chosen on D clean runs only | as runs | same runs | 0 (CPU) | `analysis_rev/baselines/*.csv` | T4 | Report honestly; contribution shifts to semantics/evaluation | MUST |
| C2 | Ablation and sensitivity (m-only, t_res-only, τ, P, T_ref) | #368A novelty; RQ6 | all traces | — | as X2 | as runs | same | 0 (CPU) | `analysis_rev/ablation/*.csv` | T6, App. A2 | report | MUST |
| C3 | Clean Multi-Krum honest alarm (archived EXP_020/021) — mechanism | Specificity; robust aggregation vs attribution | archived traces | — | z_m trajectory, partition size, round of confirmation | 123, 2026 (ARCHIVED) | — | 0 (CPU) | `analysis_rev/mk_clean_fp.md` | F5 | Disclose as limitation | MUST |

MUST GPU total ≈ 31 h on one RTX 5060 (S0 + R0 + R1 + 21 runs), excluding reruns. Disk if all MUST runs keep
full logs: ≈ 24 × 14.7 ≈ 350 GiB of 635 GB free minus 100 GiB reserve → feasible but leaves little for SHOULD;
default policy: full logs for D seeds, B-R1 and **one** V seed per condition; other V runs keep traces,
checkpoints and per-round state hashes only (update-based baselines then run on the logged subset only).

## SHOULD

| ID | Hypothesis / question | Reviewer objection | Configuration | Baselines | Metrics | Seeds | Paired controls | GPU-h (est.) | Output | Target | If negative | Prio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X4 | H4: calibration contamination (onset < T_ref) degrades detection | #368A strong calibration assumption | backdoor with onset 5, `allow_calibration_contamination: true` | all | as X2 + post-onset score inflation | V (5) or 3 | X1 | 6.7 | as X2 | T7 | Report as a limitation with the masking bound | SHOULD |
| X5 | Partial participation (K = 10, 5 per round) | #368A partial participation | `num_clients: 10, clients_per_round: 5`; backdoor + clean | all | as X2; note v3 streak breaks on unselected rounds | 3 + 3 | clean same seed | ~8 (unmeasured; K=10 full = 2.31 h) | as X2 | T7 | Report; propose participation-aware persistence as future work, do not retune | SHOULD |
| C4 | Direct forensic baselines (FedREDefense, FLForensics; FLDetector; [10] once identified) | #368A, #368B baselines | logged runs only | only methods whose papers/code have been read and whose assumptions match | as C1 | logged subset | same | 0–few (CPU/GPU replay) | `analysis_rev/published_baselines/` | T4 | If not verifiable in time: state "not compared" with reason in T4 and §Related Work; never re-implement from memory | **MUST-attempt** (blocked on sources, P-07) |
| X6 | Two colluding backdoor attackers | #368A multiple attackers | needs engineering (multi-client rev attack + set-valued attribution metrics) | all | per-attacker attribution, FP | 3 | X1 | ~4 | as X2 | T7 | Report | SHOULD (only if engineering ≤ 0.5 day) |

## OPTIONAL

| ID | Question | Objection | Note | GPU-h | Prio |
|---|---|---|---|---|---|
| X7 | Targeted label flip | realistic poisoning (second objective) | interface exists; 2 D + 3 V | ~6.7 | OPTIONAL |
| X8 | Extend V to 10 seeds for X1/X2 | statistical strength (5/5 gives CP lower bound 0.48) | only if GPU idle | 13 | OPTIONAL |
| X9 | Second, security-relevant dataset | #368B narrow setup | requires download + licence approval; not feasible in 7 days | — | OPTIONAL / future |
| X10 | Archived geometry stress tests (Class A) re-run under the new code | consistency | not needed if Stage B passes Level A; archive used as Class A evidence | 0 | OPTIONAL |

## Stage C — first matched pair and candidate evaluations (CVPR framing, blueprint §0)

| ID | Hypothesis | Objection | Configuration | Baselines | Metrics | Seeds | Paired control | GPU-h | Output | Target | If negative | Prio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1-clean = X1-D123 | H-02 (dev only); control for H-03; benign-drift evidence for H-B | specificity; drift | `configs/rev/X1_clean_d123.rev.yaml` (base = clean seed-123 config, logging ON, 5 states/round) | all detectors (CPU) | honest alarms; per-client trajectories; per-round clean top-1 and ASR of each logged global (ASR on a clean model = trigger baseline) | 123 | is the control | 1.39 (est. from R1: 4,982 s loop + startup) | `results_rev/X1_clean_d123/`, `logs_rev/X1_clean_d123/` (12.6 GiB) | F5/F-B (dev panel, appendix) | if honest alarms appear: diagnose with H-B before any further run | MUST |
| P1-badnets = X2-D123 | H-03 (dev): attack effective and attributed | realistic poisoning | `configs/rev/X2_badnets_d123.rev.yaml`: client_002, onset 20, 3×3 white patch post-augmentation, target 0, poison 50 % of its (all non-target) samples (PROVISIONAL) | all detectors (CPU) | final and per-round ASR (denominator 9,900 non-target test images), paired Δ clean top-1, online alarm, actionable round, estimated onset, delay (censored if missed), attribution, honest alarms, trajectories | 123 | P1-clean | 1.39 | as above | F2/F3 dev panels (appendix); parameters for validation | if final ASR < 0.5 (provisional threshold): adjust poison fraction on dev seeds only, re-seal | MUST |
| Pair analysis | matched-prefix validity | reproducibility | `python -m src.rev.pilot --clean X1_clean_d123 --attacked X2_badnets_d123 --malicious client_002 --onset 20 --target-class 0` | — | prefix bit-identity (globals −1..19, 100 submissions, 100 trace + 20 history rows) | 123 | — | ≈ 0.1–0.2 (51 × 2 evaluations of 19,900 images per run; est.) | `results_rev/pilot/X2_badnets_d123__vs__X1_clean_d123/` | — | invalid pair ⇒ no scientific comparison; diagnose | MUST |
| C5 (A) | H-A | calibration assumption | CPU on traces; plus X4 (onset < T_ref) once a dev seed is approved | v3 | honest alarms, attribution under contamination | dev | X1 | 0 (+ X4 1.39 per seed) | `analysis_rev/candidate_A/` | F-A | drop A | SHOULD |
| C6 (B) | H-B | benign drift | CPU on archived clean MK + X1/X2 | v3 | honest alarms (clean), attribution/delay (BadNets) | dev | X1 | 0 | `analysis_rev/candidate_B/` | F-B | drop B | MUST (cheapest test of the core question) |
| C7 (C) | H-C | onset localisation | CPU on archived E2 ρ = 0.5 + X2/X3 | v3 streak start | onset abs. error, alarms unchanged | dev | — | 0 | `analysis_rev/candidate_C/` | F-C | drop C | SHOULD |

Every GPU run above maps to a hypothesis and a planned figure; development-seed results go to the appendix
(development table) and are never pooled with validation seeds.

## Execution status

| ID | Status (2026-10-24) | Ledger |
|---|---|---|
| S0 (S0a, S0b) | DONE — exit 0 / 0; S0b ≡ S0a (33 IDENTICAL) | L-19 |
| B-R0 | DONE — exit 0; Level A bit-identical to EXP_016 (23 IDENTICAL) | L-20 |
| B-R1 | DONE — exit 0; bit-identical to R0 (33 IDENTICAL), log verified and replayed | L-21 |
| Measured | K = 5, 50 rounds: R0 4,714 s, R1 4,982 s training loop (logging +5.7 %); disk used by Stage B: logs 16 GiB, checkpoints 2.1 GiB, results 3.7 MiB | — |

## Execution order (critical path)

S0 → B-R0 → B-R1 → (X1, X2, X3 on D) → seal pre-registration → (X1, X2, X3 on V) → X4 → X5 → X6/X7.
CPU work (C1–C3, analysis scripts, figures) runs in parallel with GPU batches.
