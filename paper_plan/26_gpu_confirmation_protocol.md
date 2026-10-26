# 26 — GPU Confirmation Protocol (C4AI4C'27): FROZEN PROPOSAL — NOT AUTHORISED TO RUN

Status: written and frozen 2026-10-26 (Prompt 33, Task 7). **No run may start until the PI explicitly approves this
protocol (P-56).** Seed policy S1 (P-50): two new seeds derived by a fixed rule; the sealed seeds C1 = 978310317,
C2 = 478485063 and spare 1082418221 remain untouched. Methods and thresholds are frozen; nothing is tuned on these
runs. Labels: PROPOSED (all), CLAUDE-REPORTED (feasibility facts in §3).

## 1. Seeds (generation and registration)

- Rule: `d = SHA-256(UTF-8 "FedTrace-C4AI4C-workshop-confirmation|2026-10-26")` =
  `463717d9d7349ccff533284ad757eed899b4202886d6e4cb3cc2f19b2937d9dc`; split into eight big-endian 32-bit words; each
  modulo 2^31 → 1178015705, 1463065807, 1966286922, 1465380568, 431235112, 114746571, 1019408795, 691526108; walk in
  order, skipping every seed in the registry (42, 123, 2026, C1, C2, spare) and the five candidates of the unsealed
  validation rule (doc 11). **W1 = 1178015705, W2 = 1463065807, spare W3 = 1966286922.**
- Derived before any run, independent of any outcome; recorded in `11_seed_registry.md` (2026-10-26 entry) with role
  WORKSHOP-CONFIRMATION, status NOT RUN. W3 replaces W1 or W2 only for a dated, recorded reason (§7).
- The values were checked against every `seed:` in `configs/` (only 42, 123, 2026 occur).

## 2. Runs and configurations (prepared, not run)

| Run | Config | Base config | Intervention |
|---|---|---|---|
| W1_clean | `configs/rev/W1_clean.rev.yaml` | `configs/cifar100_fedavg_dirichlet_a01_seed1178015705_50rounds.yaml` | none |
| W1_badnets | `configs/rev/W1_badnets.rev.yaml` | same | BadNets, client_002, from round 20 |
| W1_drift | `configs/rev/W1_drift.rev.yaml` | same | benign class-composition drift, client_002, from round 20 |
| W2_clean / W2_badnets / W2_drift | `configs/rev/W2_*.rev.yaml` | `configs/cifar100_fedavg_dirichlet_a01_seed1463065807_50rounds.yaml` | as above |

- Base configs are byte-identical to the seed-2026 base (`configs/cifar100_fedavg_dirichlet_a01_seed2026_50rounds.yaml`)
  except experiment identity, output paths and `seed` (diff recorded at preparation). CIFAR-100, ResNet-18-CIFAR,
  K = 5, full participation, 50 rounds, 1 local epoch, SGD lr 0.01, batch 32, Dirichlet α = 0.1, validation 10 %.
- Rev configs are identical to X1 / X2 / X4 (d2026) except run id, base config and comments.
- **Intervened client:** client_002 (fixed rule, unchanged from all development arms). **Intervention round:** 20.
  **Calibration boundary:** T_ref = 20 (rounds < 20).
- **BadNets:** target class 0; 50 % of client_002's non-target samples get a 3 × 3 white bottom-right patch (applied
  after augmentation, normalised space) and label 0 (identical to X2).
- **Benign drift:** protocol 13 §2 construction with retained_sample_fraction 0.5 (identical to X4): classes in a
  private random order (keyed on seed, client, "benign_class_drift") until ≥ 50 % of samples are covered; every other
  sample replaced by a fixed uniformly drawn sample of a retained class from the same client; true labels; length
  preserved.
- Update logging on (fp32, sealed manifest), matching the development arms (needed for the prefix check and for any
  approved FLDetector diagnostic).
- Run order per seed: clean → BadNets → drift (W1 then W2), each launched only after the previous exits 0.

## 3. Feasibility facts (CLAUDE-REPORTED, CPU, 2026-10-26; no training)

All six rev configs parse; the Dirichlet partition succeeds for both seeds. Client sizes — W1: 10,223 / 7,437 /
**8,827** / 9,460 / 9,053; W2: 7,036 / 10,615 / **7,128** / 8,649 / 11,572 (client_000…004). client_002 holds 56
classes (W1) and 59 classes (W2). **Disclosed difference from the development seeds:** under W1 client_002 holds 347
samples of the target class 0 (development seeds: none, L-48); under W2: none. This is reported, not used to reselect
seeds.

## 4. Endpoints (frozen; computed after all runs complete)

1. **Integrity:** exit 0; `execution_clean = True` (no tracked edits during runs); sealed logs verify; matched prefix
   valid per pair (globals after rounds −1…19, submissions of rounds 0…19 bit-identical; honest round-20 submissions
   identical) — `src.rev.pilot` prefix check.
2. **Attack effectiveness:** final ASR on the 9,900 non-target triggered test images; criterion ≥ 50 % (unchanged); a
   run below the criterion is reported as "partially effective", never re-run with other parameters.
3. **Detectors (frozen):** v3 (online), v3 magnitude-only / t_res-only, v3 with LOO / trimmed / historical reference,
   ZP-norm h 1.75, CUSUM-norm h 11.75, PH-norm h 6.5 (thresholds from the development clean runs; **not**
   recalibrated on W1 / W2).
4. **Outcomes per run and detector:** detection, attribution, honest false accusations, onset error, delay (doc 28
   format); benign drift: any confirmation = honest false accusation (H-BEN per detector).
5. **Contamination (protocol 25, applied unchanged):** H-C2, H-C3, H-C4 on the new runs; the deployable displacement
   series is computed and saved before detector outcomes are inspected (timestamped outputs).
6. **Accuracy:** final top-1 per run; Δ vs the same seed's clean run.

Reporting: seed-level outcomes only; no rates; development (123 / 2026) and confirmation (W1 / W2) rows kept separate.

## 5. Cost

- GPU: 6 runs × 1.2–1.4 h (loop 69–84 min + evaluation, development measurements) ≈ **7.2–8.4 GPU-h**; pair analyses
  ≈ 10 min each on GPU or ≈ 30 min each on CPU.
- Disk: 6 × (≈ 13 GiB update log + 0.5 GiB checkpoints + 1 MiB results) ≈ **81 GiB**; free 505 GB; logger reserve
  100 GiB enforced.

## 6. Code changes required before the runs (each committed before launch, none during)

(a) add the six runs to `src/rev/preflight.py::PLANNED`; (b) parametrise `evidence_matrix`, `contamination` and
`unified_outcomes` run lists to include W1 / W2 (no change to any computation); (c) tests for (b). Preflight must PASS
on the committed tree before the first launch.

## 7. Failure and abort criteria

| Event | Action |
|---|---|
| Preflight FAIL | do not launch; report |
| Run crash / non-zero exit | stop the sequence; report; one rerun of the same config only with PI approval |
| Wall time > 2 × expected for a run | stop; report |
| Disk below reserve (logger refuses) | stop; report |
| Matched prefix INVALID | pair excluded from paired diagnostics; outcomes still reported; no substitute run |
| Partition or config failure for a seed | replace that seed by W3 with a dated record; no other replacement |
| Tracked-file modification during a run (`execution_clean = False`) | report as a provenance deviation (P-34 precedent); no rerun without PI decision |
| ASR < 50 % | report as partially effective; no parameter change |

## 8. What the runs can and cannot establish

They add two independent seeds for the same setting and intervened client; they can show whether the seed-2026
contamination pattern, the drift false accusations and the norm-baseline behaviour recur. They cannot establish rates,
generalisation to other datasets, K, clients or onsets, or malicious intent.
