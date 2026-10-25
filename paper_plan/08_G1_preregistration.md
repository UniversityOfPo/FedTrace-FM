# 08 — G1 Preregistration: replication of the P1 forensic failure mechanism (development seed 2026)

Status: **FROZEN before any G1 run.** Written 2026-10-25 under PI decision P-21. Any change after G1 outputs exist
must be a dated amendment below the original text, never an edit of it. G1 is DEVELOPMENT evidence; seeds 123 and
2026 are never validation seeds.

## 1. Purpose

Determine whether three mechanisms observed on development seed 123 (pilot P1; ledger L-30–L-33) reproduce on a
second partition: (i) the BadNets attacker's update norm falls below its matched clean counterfactual; (ii) the
attacker's temporal self-consistency rises relative to the round median ("Observation D"); (iii) the first
calibration rounds inflate the calibration spread. No detector is selected, no threshold is tuned and no candidate
(A, B, C, D) is adopted on the basis of G1.

## 2. Design (fixed)

| Item | Value |
|---|---|
| Runs | `X1_clean_d2026` (`configs/rev/X1_clean_d2026.rev.yaml`) and `X2_badnets_d2026` (`configs/rev/X2_badnets_d2026.rev.yaml`) |
| Base config | `configs/cifar100_fedavg_dirichlet_a01_seed2026_50rounds.yaml` (identical to the seed-123 base except seed, identity and output locations; enforced by `tests/test_rev_config.py`) |
| Seed | 2026 (development) |
| Model / data / partition | ResNet-18 (CIFAR variant), CIFAR-100, Dirichlet α = 0.1 partition algorithm with seed 2026, K = 5, full participation |
| Schedule | 50 rounds, 1 local epoch, SGD lr 0.01, momentum 0.9, wd 5e-4, batch 32, AMP on — unchanged |
| Malicious client rule | `client_002`, the fixed identity used in every NDSS and pilot run; fixed before the seed-2026 partition was inspected for this purpose. Its seed-2026 size (9,171 samples, from the archived EXP_009 partition) is reported, not used for selection |
| Attack | BadNets data poisoning of `client_002` from round 20: 3×3 white bottom-right patch (normalised space, after augmentation), target class 0, poison fraction 0.5 of its non-target-class samples — unchanged from P1 (P-20: not retuned) |
| Clean arm | identical config with `rev_attack: none` |
| Detector | frozen FedTrace v3 (τ = 3, P = 3, T_ref = 20, components m and t_res), evaluated online (`src/rev/online.py`) — unchanged |
| Logging | fp32 update logs in both arms (5 states/round + 51 globals) |
| Matching requirement | bit-identical initial model, client selection, submissions and global states for rounds < 20 (`src.rev.pilot`); otherwise the pair is INVALID and no endpoint is interpreted |

## 3. Primary diagnostic endpoints and fixed decision rules

Implemented in `src/rev/diagnostics.py` (`diagnose` mode); constants are listed in `PREREGISTERED` and pinned by
`tests/test_rev_diagnostics.py::test_preregistered_constants_are_fixed`. Signed residual z_k^r = (d_k^r − μ_k^d)/σ_k^d
with d_k^r = c(r) − t(k, r) and v3's own residual calibration (rounds 1–19); v3 itself uses max(0, z).

| # | Endpoint | Fixed rule |
|---|---|---|
| E1 | Attacker norm ratio vs its matched clean counterfactual | mean ratio over rounds 25–35: **REPLICATED** if ≤ 0.90 and every honest client's mean ratio is in [0.97, 1.03]; **NOT_REPLICATED** if ≥ 0.97 (flagged INCREASE if > 1.03); otherwise PARTIAL |
| E2 | Attacker signed residual trajectory, rounds 15–35 | reported in full; used by the Observation D rule |
| E3 | Honest-client signed residual distribution (rounds ≥ 20, both runs) | reported: n, min, 5 %, median, 95 %, max, fraction ≤ −3, fraction ≥ +3 |
| E4 | Calibration spread with and without burn-in | per client sd(rounds 0–19)/sd(rounds 2–19) on the clean run: **REPLICATED** if ≥ 1.5 for ≥ 3 of 5 clients; **NOT_REPLICATED** if ≥ 1.5 for ≤ 1 client; else PARTIAL. b = 2 is preregistered (chosen from the seed-123 observation, disclosed); b ∈ {1, 3, 5} is EXPLORATORY. The detector used for E5 is not altered |
| E5 | Frozen v3: attribution, alarm, actionable round, estimated onset, onset error, delay (censored if missed), honest false alarms in both runs | reported as computed; no rule |
| E6 | Clean vs attacked ASR (9,900 non-target triggered test images, identical procedure) and clean top-1 | **EFFECTIVE** if final attacked ASR ≥ 0.50; **PARTIALLY_EFFECTIVE** if < 0.50 and attacked − clean ≥ 0.10; else INEFFECTIVE. The 50 % criterion is unchanged |

### Observation D (attacker becomes more temporally self-consistent) — support / falsification

- **SUPPORTED** only if all hold:
  S1 min attacker z over rounds 20–22 ≤ −3;
  S2 median attacker z over rounds 20–29 ≤ −3;
  S3 in the clean run no client has ≥ 3 consecutive scored rounds (≥ 20) with z ≤ −3;
  S4 in the attacked run no honest client has ≥ 3 consecutive scored rounds with z ≤ −3.
- **FALSIFIED** if either holds:
  F1 median attacker z over rounds 20–29 > −1 (no consistency shift);
  F2 any client in the clean run has ≥ 3 consecutive scored rounds with z ≤ −3 (the shift also arises benignly).
- Otherwise **INCONCLUSIVE**.

A SUPPORTED verdict on one more development seed would justify a preregistered method-development comparison; it
would not establish a detector, novelty or generalisation.

## 4. Causal and leakage audit (Part 3)

| Quantity | Available at decision time (round r ≥ 20)? | Note |
|---|---|---|
| Update norm m_k^r | yes — from k's submitted update and the pre-round global | |
| Norm calibration (μ_k, σ_k) | yes — rounds 0–19 complete at the end of round 19 | scores of rounds < 20 are retrospective |
| Self-cosine t(k, r) | yes — needs k's previous submitted update | |
| Round median c(r) | yes — median over round-r submissions, computable before aggregation | |
| Signed residual z (E2, E3, Observation D) | yes — same inputs as v3's z_t_res, sign not clipped | rounds < 20: retrospective |
| Persistence state, alarm, streak-start onset | yes — rounds ≤ r | |
| v3 offline attribution tie-break | **no** — uses post-confirmation scores; the online wrapper uses scores up to the alarm only | |
| Retrospective (offline) onset | **no** — reported separately, never used for decisions | |
| Matched clean counterfactual (E1 norm ratios) | **no** — experimental diagnosis only; never a detector input | |
| True onset (round 20), malicious identity | **no** — evaluation windows and metrics only; never read by v3, the online wrapper or the diagnostics' detector-side functions | |
| Test-set accuracy and ASR (E6) | **no** — evaluation only | |
| NDSS naive peer baseline | **leaks** — its threshold window uses the configured attack start round; inactive in G1 (NDSS attack type none) and not used | |
| Threshold calibration | none in G1. The P1 exploratory thresholds were set on the clean development run only; they are not reused | |

## 5. Execution and analysis commands (in order; stop at the first failure)

```
python -m src.rev.preflight --runs X1_clean_d2026 X2_badnets_d2026
python -m src.rev.run_rev_experiment --config configs/rev/X1_clean_d2026.rev.yaml
python -m src.rev.run_rev_experiment --config configs/rev/X2_badnets_d2026.rev.yaml
python -m src.rev.pilot --clean X1_clean_d2026 --attacked X2_badnets_d2026 --malicious client_002 --onset 20 --target-class 0 --trigger-size 3 --seed 2026
python -m src.rev.diagnostics diagnose --clean X1_clean_d2026 --attacked X2_badnets_d2026 --malicious client_002 --onset 20
```

Not part of G1: G2, G3, validation seeds, baseline GPU runs, any candidate implementation, any exploratory detector
comparison on the G1 traces (may be proposed afterwards, labelled POST-HOC EXPLORATORY).

## 6. Amendments

(none)
