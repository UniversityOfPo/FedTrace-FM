# 37 — Extension Protocol (C4AI4C'27): additional seeds and K = 10 — FROZEN BEFORE ANY RUN

Status: written and frozen 2026-11-03, before any of the runs below started. PI approval: 2026-11-03 (session
message: GPU runs on the laptop GPU approved; spare W3 plus two further rule-derived seeds; sealed C1 / C2 stay
sealed; the CIFAR-10 dataset only later). The submitted manuscript remains Version 9 (locked). These results enter a
separate Version 10 only if every integrity endpoint passes and the audit is finished by 2026-11-09; otherwise
Version 9 is submitted unchanged. Labels: PROPOSED (all), CLAUDE-REPORTED (feasibility facts in §3).

## 1. Seeds

- **Extension seeds (K = 5).** These continue the protocol-26 rule (doc 26 §1). Candidate positions 3–5 are
  **W3 = 1966286922** (the recorded spare), **W4 = 1465380568** and **W5 = 431235112**. Rule:
  `SHA-256("FedTrace-C4AI4C-workshop-confirmation|2026-10-26")`, eight big-endian words mod 2^31, recomputed on
  2026-11-03 and identical to doc 26. Neither W4 nor W5 appears in the registry (42, 123, 2026, C1, C2, spare) or
  among the five skipped validation candidates (doc 11).
- **K = 10 seeds.** The development seeds 123 and 2026. No threshold or rule was ever derived from K = 10 runs of
  these arms, so they test transfer of the frozen K = 5 operating points to K = 10.
- **Untouched:** sealed C1 = 978310317, C2 = 478485063 and spare 1082418221.

## 2. Runs (15) and execution order

| Block | Runs | Base config |
|---|---|---|
| A: seeds | `W3_{clean,badnets,drift}`, `W4_*`, `W5_*` | `configs/cifar100_fedavg_dirichlet_a01_seed<seed>_50rounds.yaml`, byte-identical to the W1 base except identity, paths and `seed` |
| B: K = 10 | `K10_{clean,badnets,drift}_d123`, `K10_*_d2026` | `configs/cifar100_fedavg_dirichlet_a01_k10_seed<seed>_50rounds.yaml`, identical to the seed-123 / 2026 K = 5 base except identity, paths and `num_clients` / `clients_per_round` / `minimum_updates` = 10 |

- Rev configs are copies of `W1_{clean,badnets,drift}.rev.yaml`, differing only in run id, stage, base config,
  output paths, comments and (K = 10) `update_log.max_gib` 26.
- The arms are unchanged:
  - intervened client client_002 from round 20, T_ref = 20;
  - BadNets: target class 0, 50 % of non-target samples, 3 × 3 trigger;
  - benign drift: retained_sample_fraction 0.5.
- Order: W3 clean → badnets → drift, then W4, then W5, then K10 d123 (clean → badnets → drift), then K10 d2026. Each
  run starts only after the previous one exits 0, from a single sequential runner.

## 3. Feasibility facts (CLAUDE-REPORTED, CPU, 2026-11-03; no training)

**Preflight** (`src.rev.preflight` on all 15 runs):
- configs, outputs, archive, dataset and environment PASS: same torch 2.11.0+cu128, CUDA 12.8 and GPU (RTX 5060
  Laptop) as the archive;
- disk PASS: 423 GiB free, 259 GiB planned, 100 GiB reserve;
- git PASS only after the configs and the `preflight.PLANNED` entries are committed.

**Partitions.** The check script reproduces the published W1 facts (sizes 10,223 / 7,437 / 8,827 / 9,460 / 9,053;
client_002 target-class count 347).

| Seed | Client sizes | client_002 classes | Target-class (0) samples at client_002 |
|---|---|---|---|
| W3 | 6,427 / 10,859 / 9,335 / 9,139 / 9,240 | 55 | 0 |
| W4 | 9,843 / 8,043 / 9,937 / 7,240 / 9,937 | 60 | 0 |
| W5 | 7,345 / 8,712 / 8,476 / 9,566 / 10,901 | 59 | 5 (disclosed; not a reason to replace) |
| K10, seed 123 | 2,740–6,344 | 51 | 0 |
| K10, seed 2026 | 3,375–5,656 | 53 | 0 |

**Measured cost:** K = 5 run ≈ 69–75 min; archived K = 10 run 171 min. Total ≈ 9 × 1.25 h + 6 × 3.0 h ≈ 29 GPU-h.

## 4. Frozen detectors and endpoints

- **Detectors, frozen** (unchanged code, doc 26 Amendment 1 §A commits):
  - v3 (τ = 3, P = 3, T_ref = 20);
  - v3 magnitude-only, t_res-only, LOO-, trimmed- and historical-reference variants;
  - ZP-norm h = 1.75, CUSUM-norm h = 11.75, PH-norm h = 6.5.
- **No recalibration** on any new run, including K = 10.
- **Endpoints, as in doc 26 §4:**
  1. integrity (exit 0, `execution_clean`, sealed logs, matched-prefix check per pair);
  2. ASR on the 9,900 triggered non-target images (criterion ≥ 50 %; a run below it is "partially effective" and is
     never rerun with other parameters);
  3. per-detector detection, first attribution, onset error, delay and honest confirmations;
  4. drift and clean: any confirmation is an honest false accusation;
  5. top-1 and Δ against the same seed's clean run.

## 5. Interpretation (frozen now)

Seed-level only: no pooled rates, and no confidence intervals from client-rounds. For every new seed or K = 10
block, the following facts are reported verbatim:

| Fact | Content |
|---|---|
| F1 | v3 confirms the BadNets attacker (and whether first) |
| F2 | ZP / CUSUM / PH confirm the BadNets attacker (and whether first) |
| F3 | honest confirmations by each detector in the drift arm |
| F4 | honest confirmations by each detector in the clean arm |
| F5 | ASR and accuracy |

**"Trade-off pattern present on a seed"** means all three of:
- ZP or CUSUM attributes the BadNets attacker first;
- ZP or CUSUM confirms an honest client in the drift or clean arm of the same seed;
- v3 confirms no honest client in the clean arm.

Any other combination is "pattern absent on that seed" and is reported in the paper with equal prominence. The
paper may state "present in n of m seeds" as a count, never as a rate or probability.

**K = 10** is reported as a separate block (two seeds). It is not pooled with K = 5.

**Contamination analysis (protocol 25)** may be applied unchanged as a secondary analysis. If it is, an honest v3
accusation that the frozen necessity test does not explain is reported as a limitation.

## 6. Analysis code and provenance

- No tracked file is modified while any run is executing.
- Post-run analysis uses the frozen detector functions through new, separately named outputs
  (`*_extension.json`). Development and confirmation outputs are never overwritten.

## 7. Failure and abort criteria

Doc 26 §7 applies unchanged:
- crash or non-zero exit: stop the sequence and report;
- wall time > 2× the estimate: stop;
- disk reserve reached: stop;
- invalid prefix: excluded from paired diagnostics only;
- tracked modification during a run: recorded as a provenance deviation.

There is no seed replacement and no rerun without a PI decision.

## 8. Amendments

### Amendment 1 — 2026-11-03 17:15, launcher fix before any training round

- **What happened.** The first launch (runner started 16:59:30, commit 3a666ef) ended during CIFAR-100 loading of
  W3_clean, before any training round: the run log stops after "Using GPU", the update log has 0 states, and no
  metrics were produced.
- **Cause.** Under `$ErrorActionPreference = "Stop"`, Windows PowerShell 5.1 turns a redirected native-stderr line
  (here a NumPy `VisibleDeprecationWarning` printed while CIFAR loads) into a terminating error. Preflight was not
  affected because it ran with `-W ignore`.
- **Fix.** The runner now launches the native commands through `cmd /c` with file redirection and logs any runner
  error. No training code, config, seed or threshold changed.
- **Disposition of the aborted outputs.** They are preserved, not deleted, under
  `logs_rev/protocol37/aborted_launch_20261103T1659/`. No outcome data existed, so nothing was observed before the
  relaunch. The relaunch uses the same 15 configs in the same order.
